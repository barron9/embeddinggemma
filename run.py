"""
Null-space audio injection for EmbeddingGemma-2.

Pipeline
--------
1. Load audio from two domains (Expresso speech, Speech Commands).
2. Extract Conformer features from layer 2 (X1 = [mean; std]) and layer 6 (X2 = mean).
3. Fit one domain-balanced, soft-whitened PCA per feature block.
4. Take the orthogonal complement of the model's projection matrix (U_perp)
   and save everything to `nullspace_exact_head.pt`.
5. Wrap the model so audio embeddings are
       v_final = normalize(v_base + gamma * z @ U_perp^T)
   and evaluate it with MTEB.
"""

import math
import os
from pathlib import Path
from typing import Any

import librosa
import mteb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Audio, load_dataset
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta
from mteb.types import Array, BatchedInput, PromptType
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoProcessor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_ID = "google/embeddinggemma-2"
SAMPLE_RATE = 16_000
NUM_SAMPLES = 300          # audio clips per domain
N_COMPONENTS = 128         # PCA components per feature block
EMBED_DIM = 768            # embedding size of the base model
PROJ_RANK = 512            # output rank of the projection layer

LAYER2_IDX = 3             # hidden_states index for Conformer layer 2
LAYER6_IDX = 7             # hidden_states index for Conformer layer 6

SCRIPT_DIR = Path(__file__).resolve().parent
HEAD_PATH = SCRIPT_DIR / "nullspace_exact_head.pt"

# Local test files (change to your own paths)
TEST_AUDIO_PATHS = [
    r"F:\chrome downloads\robot_sim_test\synth\8.wav",
    r"F:\chrome downloads\robot_sim_test\synth\5.wav",
]
MTEB_TASKS = ["CREMA_DClustering", "CREMA_D"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(DEVICE).eval()
    return processor, model


def compute_null_space_basis(model) -> np.ndarray:
    """Return an orthonormal basis (768 x 256) of the complement of the
    column space of the final projection matrix W (768 x 512)."""
    proj_layer = getattr(model, "embedding_projection", None) or model.language_model.embedding_projection
    W = proj_layer.weight.detach().cpu().numpy()                  # (768, 512)
    U, _, _ = np.linalg.svd(W, full_matrices=True)
    return U[:, PROJ_RANK:]                                       # (768, 256)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_audio_clips(dataset, num_samples: int) -> list[np.ndarray]:
    """Take the first `num_samples` clips from a (possibly streaming) dataset
    as float32 waveforms at 16 kHz."""
    dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    clips = []
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break
        clips.append(np.asarray(sample["audio"]["array"], dtype=np.float32))
    return clips


def load_domains() -> list[list[np.ndarray]]:
    """Domain 1: Expresso (expressive speech). Domain 2: Speech Commands."""
    expresso = load_dataset("ylacombe/expresso", "read", split="train")
    commands = load_dataset("charcot-ai/marimba-bench", "speech_commands", split="train", streaming=True)
    return [
        load_audio_clips(expresso, NUM_SAMPLES),
        load_audio_clips(commands, NUM_SAMPLES),
    ]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def masked_stats(seq: torch.Tensor, mask: torch.Tensor):
    """Masked mean and std over time.
    seq: (B, T, C), mask: (B, T)  ->  two tensors of shape (B, C)."""
    mask = mask.unsqueeze(-1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    mean = (seq * mask).sum(dim=1) / counts
    var = (((seq - mean.unsqueeze(1)) ** 2) * mask).sum(dim=1) / counts
    return mean, torch.sqrt(var.clamp(min=1e-12))


def conformer_features(audio_tower, inputs):
    """Compute the two feature blocks used by the PCA heads.

    X1 = L2-normalised [mean; std] of layer 2   -> (B, 2048)
    X2 = L2-normalised mean of layer 6          -> (B, 1024)
    """
    frame_mask = inputs.get("input_features_mask", None)
    out = audio_tower(
        input_features=inputs.input_features,
        attention_mask=frame_mask,
        output_hidden_states=True,
    )
    h2 = out.hidden_states[LAYER2_IDX]
    h6 = out.hidden_states[LAYER6_IDX]
    B, T_out, _ = h2.shape

    # Align the frame mask with the (subsampled) Conformer time axis
    if frame_mask is not None:
        mask = F.interpolate(frame_mask.unsqueeze(1).float(), size=T_out, mode="nearest").squeeze(1)
    else:
        mask = torch.ones((B, T_out), device=h2.device)

    mu2, std2 = masked_stats(h2, mask)
    mu6, _ = masked_stats(h6, mask)

    x1 = F.normalize(torch.cat([mu2, std2], dim=-1), p=2, dim=-1)
    x2 = F.normalize(mu6, p=2, dim=-1)
    return x1, x2


@torch.no_grad()
def extract_features(model, processor, audio_list):
    """Run `conformer_features` on each clip and stack the results."""
    x1_all, x2_all = [], []
    for audio in audio_list:
        inputs = processor(audio=audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(DEVICE)
        x1, x2 = conformer_features(model.audio_tower, inputs)
        x1_all.append(x1.cpu().numpy())
        x2_all.append(x2.cpu().numpy())
    return np.vstack(x1_all), np.vstack(x2_all)


# ---------------------------------------------------------------------------
# Balanced soft-whitened PCA
# ---------------------------------------------------------------------------
def fit_balanced_soft_whitened_pca(X_list, n_components: int):
    """Fit a PCA shared across domains, weighting each domain equally (1/K).

    Returns
    -------
    mean : (d,)            average of the per-domain means
    P    : (d, n_comp)     top eigenvectors of the balanced covariance
    s    : (n_comp,)       soft-whitening scale, eigenvalue^(1/4)
                           (i.e. sqrt of the singular value)
    """
    K = len(X_list)
    d = X_list[0].shape[1]

    mean = torch.stack([torch.tensor(X.mean(axis=0), dtype=torch.float32) for X in X_list]).mean(dim=0)

    cov = torch.zeros((d, d), dtype=torch.float32)
    for X in X_list:
        Xc = torch.tensor(X, dtype=torch.float32) - mean
        cov += (Xc.T @ Xc) / max(len(Xc) - 1, 1) / K

    eigvals, eigvecs = torch.linalg.eigh(cov)
    top = torch.argsort(eigvals, descending=True)[:n_components]
    eigvals, P = eigvals[top], eigvecs[:, top]

    sigma = torch.sqrt(torch.clamp(eigvals, min=1e-12))
    s = torch.sqrt(sigma)
    return mean, P, s


def build_and_save_head(model, processor, domains, path: Path):
    """Fit both PCA blocks, attach the null-space basis, and save to disk."""
    feats = [extract_features(model, processor, clips) for clips in domains]
    X1_list = [f[0] for f in feats]
    X2_list = [f[1] for f in feats]

    m1, P1, s1 = fit_balanced_soft_whitened_pca(X1_list, N_COMPONENTS)
    m2, P2, s2 = fit_balanced_soft_whitened_pca(X2_list, N_COMPONENTS)

    U_perp_full = compute_null_space_basis(model)
    U_perp = torch.tensor(U_perp_full[:, : 2 * N_COMPONENTS], dtype=torch.float32)  # (768, 2 * n_comp)

    torch.save(
        {
            "m1": m1, "P1": P1, "s1": s1,
            "m2": m2, "P2": P2, "s2": s2,
            "U_perp": U_perp,
            "gamma": 1.0,
            "eps": 1e-5,
        },
        path,
    )
    print(f"Saved: {path} (block size: {N_COMPONENTS} + {N_COMPONENTS} = {2 * N_COMPONENTS}d)")


# ---------------------------------------------------------------------------
# MTEB wrapper
# ---------------------------------------------------------------------------
class ExactNullSpaceAudioGemma(AbsEncoder, nn.Module):
    """EmbeddingGemma with audio embeddings injected into the projection's null space."""

    def __init__(self, base_model, processor, head_path):
        nn.Module.__init__(self)
        AbsEncoder.__init__(self)
        self.model = base_model.eval()
        self.processor = processor

        state = torch.load(head_path, map_location=self.model.device)
        for key in ["m1", "P1", "s1", "m2", "P2", "s2", "U_perp"]:
            self.register_buffer(key, state[key].to(self.model.device))
        self.gamma = state.get("gamma", 1.0)
        self.eps = state.get("eps", 1e-5)

        self.mteb_model_meta = self._build_model_meta()

    @staticmethod
    def _build_model_meta():
        """Build MTEB metadata, keeping only the fields this MTEB version supports
        and filling any remaining required fields with None."""
        meta = dict(
            name="modgemma",
            revision="custom",
            release_date=None,
            languages=None,
            modalities=["text", "audio"],
            embed_dim=EMBED_DIM,
            n_parameters=None,
            max_tokens=None,
            license=None,
            open_weights=False,
            public_training_code=None,
            public_training_data=None,
            framework=["PyTorch"],
            reference=None,
            similarity_fn_name="cosine",
            use_instructions=False,
            training_datasets=None,
        )
        fields = ModelMeta.model_fields
        meta = {k: v for k, v in meta.items() if k in fields}
        for name, info in fields.items():
            if name not in meta and info.is_required():
                meta[name] = None
        return ModelMeta.model_construct(**meta)

    # ----- encoders -----
    def _mean_pool(self, inputs):
        """Attention-masked mean pooling of the last hidden state, L2-normalised."""
        outputs = self.model(**inputs)
        mask = inputs.attention_mask.unsqueeze(-1).float()
        pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return F.normalize(pooled, p=2, dim=-1)

    @torch.no_grad()
    def encode_text(self, texts):
        inputs = self.processor(text=texts, padding=True, truncation=True, return_tensors="pt").to(self.model.device)
        return self._mean_pool(inputs)

    @torch.no_grad()
    def encode_audio(self, audio_arrays, sampling_rate=SAMPLE_RATE):
        # 1. Base 768d embedding
        inputs = self.processor(
            text=["<|audio|>"] * len(audio_arrays),
            audio=audio_arrays,
            sampling_rate=sampling_rate,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        v_base = self._mean_pool(inputs)                                   # (B, 768)

        # 2. Conformer layer-2 / layer-6 features
        X1, X2 = conformer_features(self.model.audio_tower, inputs)

        # 3. Soft-whitened PCA: z_k = normalize((X_k - m_k) @ P_k / (s_k + eps))
        z1 = F.normalize(((X1 - self.m1) @ self.P1) / (self.s1 + self.eps), p=2, dim=-1)
        z2 = F.normalize(((X2 - self.m2) @ self.P2) / (self.s2 + self.eps), p=2, dim=-1)

        # 4. Merge the blocks with equal weight: z = normalize([sqrt(.5) z1 ; sqrt(.5) z2])
        scale = math.sqrt(0.5)
        z = F.normalize(torch.cat([scale * z1, scale * z2], dim=-1), p=2, dim=-1)  # (B, 2 * n_comp)

        # 5. Null-space injection: v_final = normalize(v_base + gamma * z @ U_perp^T)
        return F.normalize(v_base + self.gamma * (z @ self.U_perp.T), p=2, dim=-1)  # (B, 768)

    # ----- audio helpers -----
    @staticmethod
    def _to_mono_16k(arr, sr, target_sr=SAMPLE_RATE):
        """Convert an array/tensor to a mono float32 waveform at `target_sr`."""
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().float().cpu().numpy()
        arr = np.asarray(arr, dtype=np.float32)

        if arr.ndim == 2:  # average over the channel axis (the smaller dimension)
            channel_axis = 0 if arr.shape[0] <= arr.shape[1] else 1
            arr = arr.mean(axis=channel_axis)
        elif arr.ndim > 2:
            arr = arr.squeeze().reshape(-1)

        if sr != target_sr:
            import torchaudio.functional as AF
            arr = AF.resample(torch.from_numpy(arr), sr, target_sr).numpy()
        return arr

    def _extract_audio(self, item):
        """Return a mono 16 kHz float32 array from any MTEB audio item format."""
        if isinstance(item, dict) and "array" in item:
            arr, sr = item["array"], item.get("sampling_rate", SAMPLE_RATE)
        elif hasattr(item, "get_all_samples"):  # torchcodec AudioDecoder
            samples = item.get_all_samples()
            arr, sr = samples.data, getattr(samples, "sample_rate", SAMPLE_RATE)
        else:
            arr, sr = item, SAMPLE_RATE
        return self._to_mono_16k(arr, sr)

    # ----- MTEB entry point -----
    @torch.no_grad()
    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: Any,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        self.eval()
        batch_size = kwargs.get("batch_size", 16)
        all_embeddings = []
        print(f"[encode] task={task_metadata.name} split={hf_split} subset={hf_subset}")

        for batch in inputs:
            if isinstance(batch, dict) and "audio" not in batch and "text" in batch:
                # Text-only batch (audio-text tasks)
                texts = batch["text"]
                texts = [texts] if isinstance(texts, str) else list(texts)
                emb = self.encode_text(texts)
            else:
                # Audio batch
                if isinstance(batch, dict):
                    items = batch.get("audio", [])
                elif isinstance(batch, (list, tuple)):
                    items = batch
                else:
                    items = [batch]
                if isinstance(items, dict) and "array" in items:  # single item
                    items = [items]

                audio_arrays = [self._extract_audio(it) for it in items]

                # Sub-batch in case the DataLoader batch exceeds `batch_size`
                chunks = [
                    self.encode_audio(audio_arrays[i : i + batch_size])
                    for i in range(0, len(audio_arrays), batch_size)
                ]
                emb = torch.cat(chunks, dim=0)

            emb = emb.detach().float().cpu().numpy()
            all_embeddings.append(emb[None, :] if emb.ndim == 1 else emb)

        return np.vstack(all_embeddings)  # (N_total, D)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    processor, base_model = load_model()

    # --- Fit and save the head ---
    domains = load_domains()
    print(f"Loaded clips per domain: {[len(d) for d in domains]}")
    build_and_save_head(base_model, processor, domains, HEAD_PATH)

    # --- Quick sanity check ---
    model_ns = ExactNullSpaceAudioGemma(base_model, processor, HEAD_PATH).to(DEVICE)

    test_audios = [librosa.load(p, sr=SAMPLE_RATE, mono=True)[0] for p in TEST_AUDIO_PATHS]
    emb_audios = model_ns.encode_audio(test_audios)
    print("Similarity matrix:")
    print(emb_audios @ emb_audios.T)

    # --- MTEB evaluation ---
    tasks = mteb.get_tasks(tasks=MTEB_TASKS)
    results = mteb.evaluate(model_ns, tasks=tasks, overwrite_strategy="always", cache=None)

    for task_res in results.task_results:
        print(f"{task_res.task_name}: {task_res.main_score:.4f}")


if __name__ == "__main__":
    main()
