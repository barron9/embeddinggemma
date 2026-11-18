import os
import math
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor
from mteb.models.model_meta import ModelMeta
from typing import Any
import mteb
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta
from mteb.types import Array, BatchedInput, PromptType
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoProcessor


class ExactNullSpaceAudioGemma(AbsEncoder, nn.Module):
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
        # --- MTEB metadata ---
        meta_kwargs = dict(
            name="modgemma",
            revision="custom",
            release_date=None,
            languages=None,
            modalities=["text", "audio"],
            embed_dim=768,                 # final embedding size (base_emb dim)
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

        # Keep only fields that exist in this MTEB version, then fill any
        # remaining *required* fields with None so attribute access never fails.
        fields = ModelMeta.model_fields
        meta_kwargs = {k: v for k, v in meta_kwargs.items() if k in fields}
        for fname, finfo in fields.items():
            if fname not in meta_kwargs and finfo.is_required():
                meta_kwargs[fname] = None

        self.mteb_model_meta = ModelMeta.model_construct(**meta_kwargs)

    @torch.no_grad()
    def encode_text(self, texts):
        inputs = self.processor(text=texts, padding=True, truncation=True, return_tensors="pt").to(self.model.device)
        outputs = self.model(**inputs)
        mask = inputs.attention_mask.unsqueeze(-1).float()
        pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return F.normalize(pooled, p=2, dim=-1)

    @torch.no_grad()
    def encode_audio(self, audio_arrays, sampling_rate=16000):
        # 1. Temel 768d embedding (v_base, model tarafından L2 normalize edilir)
        text_prompts = ["<|audio|>"] * len(audio_arrays)
        inputs = self.processor(
            text=text_prompts,
            audio=audio_arrays,
            sampling_rate=sampling_rate,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        outputs = self.model(**inputs)
        mask = inputs.attention_mask.unsqueeze(-1).float()
        v_base = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        v_base = F.normalize(v_base, p=2, dim=-1)  # [B, 768]

        # 2. Hooksuz Conformer Katman 2 ve Katman 6 Özellikleri
        audio_out = self.model.audio_tower(
            input_features=inputs.input_features,
            attention_mask=inputs.get("input_features_mask", None),
            output_hidden_states=True
        )
        
        h2_seq = audio_out.hidden_states[3]
        h6_seq = audio_out.hidden_states[7]
        
        B,T_out,C = h2_seq.shape
        raw_mask = inputs.get("attention_mask", None)
        if raw_mask is not None:
            t_mask = F.interpolate(raw_mask.unsqueeze(1).float(), size=T_out, mode="nearest").squeeze(1)
        else:
            t_mask = torch.ones((B, T_out), device=self.model.device)
            
        t_mask_exp = t_mask.unsqueeze(-1)
        valid_counts = t_mask_exp.sum(dim=1).clamp(min=1.0)

        # X₁ = [μ₂ ; σ₂] (2048d, L2-normalised)
        mu_2 = (h2_seq * t_mask_exp).sum(dim=1) / valid_counts
        var_2 = (((h2_seq - mu_2.unsqueeze(1)) ** 2) * t_mask_exp).sum(dim=1) / valid_counts
        std_2 = torch.sqrt(var_2.clamp(min=1e-12))
        X1 = F.normalize(torch.cat([mu_2, std_2], dim=-1), p=2, dim=-1)

        # X₂ = μ₆ (1024d, L2-normalised)
        mu_6 = (h6_seq * t_mask_exp).sum(dim=1) / valid_counts
        X2 = F.normalize(mu_6, p=2, dim=-1)

        # 3. Soft-Whitened PCA Projeksiyonları: z_k = normalize(((X_k - m_k) @ P_k) / (s_k + ε))
        z1 = F.normalize(((X1 - self.m1) @ self.P1) / (self.s1 + self.eps), p=2, dim=-1)
        z2 = F.normalize(((X2 - self.m2) @ self.P2) / (self.s2 + self.eps), p=2, dim=-1)

        # 4. İki bloğu √0.5 ile birleştir: z = normalize([√0.5 · z₁ ; √0.5 · z₂])
        scale = math.sqrt(0.5)
        z = F.normalize(torch.cat([scale * z1, scale * z2], dim=-1), p=2, dim=-1)  # [B, 2*n_comp]

        # 5. Null-space enjeksiyonu: v_final = normalize(v_base + γ · (z @ U_⊥^T))
        v_final = F.normalize(v_base + self.gamma * (z @ self.U_perp.T), p=2, dim=-1)  # [B, 768]
        return v_final
    # ---------- helpers ----------
    @staticmethod
    def _to_mono_16k(arr, sr, target_sr=16000):
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().float().cpu().numpy()
        arr = np.asarray(arr, dtype=np.float32)

        # to mono: average over the channel axis (the smaller of the two dims)
        if arr.ndim == 2:
            ch_axis = 0 if arr.shape[0] <= arr.shape[1] else 1
            arr = arr.mean(axis=ch_axis)
        elif arr.ndim > 2:
            arr = arr.squeeze()
            if arr.ndim > 1:
                arr = arr.reshape(-1)

        # resample if needed
        if sr != target_sr:
            import torchaudio.functional as AF
            arr = AF.resample(torch.from_numpy(arr), sr, target_sr).numpy()
        return arr

    def _extract_audio(self, item):
        """Return (mono 16k float32 array) from any MTEB audio item format."""
        if isinstance(item, dict) and "array" in item:
            arr, sr = item["array"], item.get("sampling_rate", 16000)
        elif hasattr(item, "get_all_samples"):          # torchcodec AudioDecoder
            s = item.get_all_samples()
            arr, sr = s.data, getattr(s, "sample_rate", 16000)
        else:
            arr, sr = item, 16000
        return self._to_mono_16k(arr, sr)

    # ---------- main entry ----------
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
        print(f"[encode] called | task={task_metadata.name} split={hf_split} subset={hf_subset}")

        for batch in inputs:
            # ---- text-only batch (audio-text tasks) ----
            if isinstance(batch, dict) and "audio" not in batch and "text" in batch:
                texts = batch["text"]
                texts = [texts] if isinstance(texts, str) else list(texts)
                emb = self.encode_text(texts)
            else:
                # ---- audio batch ----
                if isinstance(batch, dict):
                    items = batch.get("audio", [])
                elif isinstance(batch, (list, tuple)):
                    items = batch
                else:
                    items = [batch]
                if isinstance(items, dict) and "array" in items:   # single item
                    items = [items]

                audio_arrays = [self._extract_audio(it) for it in items]

                # sub-batch in case the DataLoader batch is larger than batch_size
                embs = []
                for i in range(0, len(audio_arrays), batch_size):
                    embs.append(
                        self.encode_audio(audio_arrays[i : i + batch_size], sampling_rate=16000)
                    )
                emb = torch.cat(embs, dim=0)

            emb = emb.detach().float().cpu().numpy()
            if emb.ndim == 1:
                emb = emb[None, :]
            all_embeddings.append(emb)          # <-- now inside the batch loop

        return np.vstack(all_embeddings)        # (N_total, D)    


# ==========================================
# TEST ÇALIŞTIRMASI
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "google/embeddinggemma-2"

processor = AutoProcessor.from_pretrained(MODEL_ID)
base_model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device)

script_dir = os.path.dirname(os.path.abspath(__file__))
head_path = os.path.join(script_dir, "nullspace_exact_head.pt")


model_ns = ExactNullSpaceAudioGemma(base_model, processor, head_path).to(device)

ses_0, _ = librosa.load(r"F:\chrome downloads\robot_sim_test\synth\8.wav", sr=16000, mono=True)
ses_1, _ = librosa.load(r"F:\chrome downloads\robot_sim_test\synth\5.wav", sr=16000, mono=True)

emb_audios = model_ns.encode_audio([ses_0, ses_1])
emb_text = model_ns.encode_text(["merhaba"])

# print("Ses 0 - Ses 1 Benzerliği:", (emb_audios @ emb_audios.T))
# print("Metin - Ses 0 Benzerliği:", (emb_text @ emb_audios.T))
#emb_audios = F.normalize(emb_audios, p=2, dim=-1)

similarity_matrix = emb_audios @ emb_audios.T

print("Benzerlik Matrisi:")
print(similarity_matrix)


import mteb

# 1. Load the MAEB benchmark (30 tasks) or filter specific audio tasks
benchmark = mteb.get_tasks(tasks=["CREMA_DClustering","CREMA_D"]) #mteb.get_benchmark("MAEB(audio)")
# Alternatively, get a single task:
# tasks = mteb.get_tasks(tasks=["CREMA_D"])

# 2. Load a pre-registered model (e.g., Whisper, CLAP, wav2vec, or VGGish)
#model = mteb.get_model("google/embeddinggemma-2")

# 3. Run the evaluation and save results locally
results = mteb.evaluate(
    model_ns,
    tasks=benchmark,
    overwrite_strategy="always", cache=None
)

crema_result = results.task_results[0]
print("Main Score:", crema_result.main_score)
print("Detailed Scores:", crema_result.scores)

# 2. Convert results to a dictionary or Pandas DataFrame
for task_res in results.task_results:
    print(f"{task_res.task_name}: {task_res.main_score:.4f}")


# import mteb, inspect, traceback, logging
# logging.basicConfig(level=logging.INFO)
# print("mteb", mteb.__version__)

# wrapper = model_ns

# # 1. Does the class own its methods?
# for name in ["encode", "encode_audio", "encode_text"]:
#     f = getattr(type(wrapper), name)
#     print(name, "->", f.__qualname__, "|", inspect.getsourcefile(f))

# # 2. Make EVERY encode path scream and print who called it
# def spy(obj, name="encode"):
#     orig = getattr(obj, name)
#     def wrapped(*a, **k):
#         print(f"### {type(obj).__name__}.{name} CALLED (id={id(obj)})", flush=True)
#         traceback.print_stack(limit=6)
#         return orig(*a, **k)
#     setattr(obj, name, wrapped)

# spy(wrapper)
# #spy(base_model)          # if THIS prints instead, MTEB is using base_model

# # 3. See what evaluate actually receives
# orig_eval = mteb.evaluate
# def eval_spy(model, *a, **k):
#     print("evaluate received:", type(model), "id", id(model), flush=True)
#     print("wrapper id        :", id(wrapper), flush=True)
#     return orig_eval(model, *a, **k)
# mteb.evaluate = eval_spy

# tasks = mteb.get_tasks(tasks=["CREMA_D"])
# mteb.evaluate(wrapper, tasks, overwrite_strategy="always", cache=None)
