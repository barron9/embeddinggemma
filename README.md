# embeddinggemma2

null space embedding mteb

#### before
CREMA_DClustering: 0.005
CREMA_D: 0.1962
#### after
CREMA_DClustering: 0.0491
CREMA_D: 0.3625

``

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

``
