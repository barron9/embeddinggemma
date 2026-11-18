import os
import glob
import math
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoProcessor

# 1. Model ve Processor Yükleme
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "google/embeddinggemma-2"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device).eval()

# 2. Terminal Projeksiyon Matrisinden (768 x 512) Ortogonal Tümleyeni (U_perp) Çıkarma
proj_layer = getattr(model, "embedding_projection", None) or model.language_model.embedding_projection
W = proj_layer.weight.detach().cpu().numpy()  # (768, 512)
P_svd, _, _ = np.linalg.svd(W, full_matrices=True)
U_perp_full = P_svd[:, 512:]  # (768, 256)
from pathlib import Path

# # 3. Klasördeki .wav Dosyalarını Yükleme
# folder_path = r"F:\chrome downloads\robot_sim_test\synth"
# wav_files = sorted(
#     glob.glob(os.path.join(folder_path, "*.wav")),
#     key=lambda x: int(Path(x).stem) if Path(x).stem.isdigit() else Path(x).stem
# )


from datasets import load_dataset, Audio
from sklearn.decomposition import PCA

# 1. Load Expresso dataset
dataset = load_dataset(
    "ylacombe/expresso",
    "read",
    split="train"
)

# 2. Automatically resample audio to 16 kHz
dataset = dataset.cast_column(
    "audio",
    Audio(sampling_rate=16000)
)

# 3. Select 300 real audio samples
num_samples = 300

audios = []

for i in range(num_samples):

    sample = dataset[i]

    # Extract audio waveform
    audio = sample["audio"]["array"]

    # Convert to float32
    audio = np.asarray(audio, dtype=np.float32)

    audios.append(audio)


# audios = []
# for f in wav_files:
#     y, _ = librosa.load(f, sr=16000, mono=True)
#     if len(y) > 1600:
#         audios.append(y.astype(np.float32))

print(f"Yüklenen ses sayısı: {len(audios)}")

# 4. Masked Pooling Yardımcı Fonksiyonu (X₁ = [μ₂; σ₂] ve X₂ = μ₆)
@torch.no_grad()
def extract_x1_x2(audio_list):
    x1_list, x2_list = [], []
    for audio in audio_list:
        inputs = processor(audio=audio, sampling_rate=16000, return_tensors="pt").to(device)
        
        # Hook kullanmadan audio_tower üzerinden tüm gizli katmanları al
        audio_out = model.audio_tower(
            input_features=inputs.input_features,
            attention_mask=inputs.get("input_features_mask", None),
            output_hidden_states=True
        )
        
        # Zaman boyutu (T) için geçerli maskeyi ayarla
        h2_seq = audio_out.hidden_states[3]  # Katman 2 -> (1, T_out, 1024)
        h6_seq = audio_out.hidden_states[7]  # Katman 6 -> (1, T_out, 1024)
        
        # Eğer ses küçültülmüş (subsampled) maske varsa kullan, yoksa tüm çerçeveleri geçerli say
        B,T_out,C = h2_seq.shape
        raw_mask = inputs.get("input_features_mask", None)
        if raw_mask is not None:
            # Conformer zaman küçültme oranına göre maskeyi hizala
            mask = F.interpolate(raw_mask.unsqueeze(1).float(), size=T_out, mode="nearest").squeeze(1)
        else:
            mask = torch.ones((B, T_out), device=device)
            
        mask_exp = mask.unsqueeze(-1)  # (1, T_out, 1)
        valid_counts = mask_exp.sum(dim=1).clamp(min=1.0)
        
        # X₁: Katman 2 -> Masked Mean (μ₂) ve Masked Std (σ₂), ardından L2 Normalize
        mu_2 = (h2_seq * mask_exp).sum(dim=1) / valid_counts
        var_2 = (((h2_seq - mu_2.unsqueeze(1)) ** 2) * mask_exp).sum(dim=1) / valid_counts
        std_2 = torch.sqrt(var_2.clamp(min=1e-12))
        x1 = F.normalize(torch.cat([mu_2, std_2], dim=-1), p=2, dim=-1)  # (1, 2048)
        
        # X₂: Katman 6 -> Masked Mean (μ₆), ardından L2 Normalize
        mu_6 = (h6_seq * mask_exp).sum(dim=1) / valid_counts
        x2 = F.normalize(mu_6, p=2, dim=-1)  # (1, 1024)
        
        x1_list.append(x1.cpu().numpy())
        x2_list.append(x2.cpu().numpy())
        
    return np.vstack(x1_list), np.vstack(x2_list)

X1_all, X2_all = extract_x1_x2(audios)

# 5. İki Ayrı Soft-Whitened PCA Bloğu Uydurma (k = 1, 2)
n_comp = min(128, len(audios))  # 76 dosya varsa 76, >=128 dosya varsa 128 olur

def fit_soft_whitened_pca(X, n_components):
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X)
    m_k = torch.tensor(pca.mean_, dtype=torch.float32)                 # [d_k]
    P_k = torch.tensor(pca.components_.T, dtype=torch.float32)         # [d_k, n_comp]
    # s_k = σ_k^(1/2) -> Tekil değerlerin (explained_variance_'in karekökü) tekrar karekökü
    sigma_k = np.sqrt(np.maximum(pca.explained_variance_, 1e-12))
    s_k = torch.tensor(np.sqrt(sigma_k), dtype=torch.float32)          # [n_comp]
    return m_k, P_k, s_k

m1, P1, s1 = fit_soft_whitened_pca(X1_all, n_comp)
m2, P2, s2 = fit_soft_whitened_pca(X2_all, n_comp)

# Toplam 2 * n_comp boyutuna uygun U_perp dilimi ([768, 2 * n_comp])
U_perp = torch.tensor(U_perp_full[:, :2 * n_comp], dtype=torch.float32)

# 6. Scriptin Olduğu Klasöre Kaydetme
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, "nullspace_exact_head.pt")

torch.save({
    "m1": m1, "P1": P1, "s1": s1,
    "m2": m2, "P2": P2, "s2": s2,
    "U_perp": U_perp,
    "gamma": 1.0,
    "eps": 1e-5
}, save_path)

print(f"Başarıyla kaydedildi: {save_path} (Blok boyutu: {n_comp} + {n_comp} = {2*n_comp}d)")
