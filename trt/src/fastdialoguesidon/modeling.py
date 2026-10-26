"""Slim inference-only model definitions + HF-bundle loader.

Inference-only module definitions adapted from the Sidon codebase,
depending only on torch / transformers / safetensors; the
t=999 -> s=0 constants of the one-step update are baked into the bundle
config.

``load_bundle(path)`` returns a :class:`FastDialogueSidon` holding
- ``encoder``  : DialogueEncoder (w2v-BERT + FlexRelSelfAttention)
- ``head``     : CTMTransformerHead
- ``ctm``      : CTMOneStep (single t=999 -> s=0 jump)
- ``vocoder``  : WaveFitNeXtLatentDecoder
- ``decoder``  : latent denorm + speaker repack + vocoder wrapper
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wavefit_next import WaveFitNeXtLatentDecoder


# ---- encoder ----------------------------------------------------------------


class FlexRelSelfAttention(nn.Module):
    """flex_attention drop-in for ``Wav2Vec2BertSelfAttention``.

    Keeps the ``relative_key`` position term exactly:
    ``scores += Q.R[clamp(j-i,-left,right)]^T / sqrt(d)`` with the 73-entry
    ``distance_embedding``. The per-pair bias is never materialized.
    """

    _flex_attention = None

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        left_max_position: int = 64,
        right_max_position: int = 8,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.left_max_position = left_max_position
        self.right_max_position = right_max_position
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, hidden_size)
        self.distance_embedding = nn.Embedding(
            left_max_position + right_max_position + 1, self.head_dim
        )

    @classmethod
    def _flex(cls):
        if cls._flex_attention is None:
            import os

            from torch.nn.attention.flex_attention import flex_attention

            if (
                os.environ.get("FDS_FLEX_NO_COMPILE") == "1"
                or not torch.cuda.is_available()
            ):
                cls._flex_attention = flex_attention
            else:
                cls._flex_attention = torch.compile(
                    flex_attention, dynamic=False
                )
        return cls._flex_attention

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        relative_position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **_: object,
    ) -> tuple[torch.Tensor, None]:
        batch, length, hidden = hidden_states.shape
        shape = (batch, length, self.num_heads, self.head_dim)
        query = self.linear_q(hidden_states).view(shape).transpose(1, 2)
        key = self.linear_k(hidden_states).view(shape).transpose(1, 2)
        value = self.linear_v(hidden_states).view(shape).transpose(1, 2)
        query_rel = torch.einsum(
            "bhld,rd->bhlr", query, self.distance_embedding.weight
        ) / math.sqrt(self.head_dim)
        left, right = self.left_max_position, self.right_max_position

        if attention_mask is None:

            def score_mod(score, b, h, q_idx, kv_idx):
                return score + query_rel[
                    b, h, q_idx,
                    torch.clamp(kv_idx - q_idx, -left, right) + left,
                ]

        else:
            additive_mask = attention_mask[:, 0]

            def score_mod(score, b, h, q_idx, kv_idx):
                return (
                    score
                    + query_rel[
                        b, h, q_idx,
                        torch.clamp(kv_idx - q_idx, -left, right) + left,
                    ]
                    + additive_mask[b, q_idx, kv_idx]
                )

        context = self._flex()(query, key, value, score_mod=score_mod)
        context = context.transpose(1, 2).reshape(batch, length, hidden)
        return self.linear_out(context), None


def replace_attention_with_flex(ssl_model: nn.Module) -> int:
    config = ssl_model.config
    layers = ssl_model.encoder.layers
    for layer in layers:
        layer.self_attn = FlexRelSelfAttention(
            int(config.hidden_size),
            int(config.num_attention_heads),
            int(config.left_max_position_embeddings),
            int(config.right_max_position_embeddings),
        )
    return len(layers)


class DialogueEncoder(nn.Module):
    """[B, L, 160] fbank -> [B, L, 2*latent + hidden] conditioning."""

    def __init__(self, ssl_model: nn.Module, latent_size: int) -> None:
        super().__init__()
        self.model = ssl_model
        hidden_size = int(ssl_model.config.hidden_size)
        self.speaker_1 = nn.Linear(hidden_size, latent_size)
        self.speaker_2 = nn.Linear(hidden_size, latent_size)

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.model(
            input_features=input_features, attention_mask=attention_mask
        )
        features = outputs.last_hidden_state
        return torch.cat(
            [self.speaker_1(features), self.speaker_2(features), features],
            dim=-1,
        )


# ---- DiT head ---------------------------------------------------------------


def _modulate(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )
        self.frequency_embedding_size = frequency_embedding_size
        half = frequency_embedding_size // 2
        self.register_buffer(
            "frequency_indices",
            torch.arange(start=0, end=half, dtype=torch.int64),
            persistent=False,
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float()
        frequencies = torch.exp(
            -math.log(10000)
            * self.frequency_indices.float()
            / len(self.frequency_indices)
        )
        args = t[:, None] * frequencies[None]
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            t_freq = torch.cat(
                [t_freq, torch.zeros_like(t_freq[:, :1])], dim=-1
            )
        return self.mlp(t_freq)


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        use_rotary: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_rotary = use_rotary
        self.cache_rotary = False
        self._rotary_cache: dict = {}
        self.norm1 = nn.LayerNorm(
            hidden_size, eps=1e-6, elementwise_affine=False
        )
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.norm2 = nn.LayerNorm(
            hidden_size, eps=1e-6, elementwise_affine=False
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * ffn_ratio), bias=True),
            nn.GELU(),
            nn.Linear(int(hidden_size * ffn_ratio), hidden_size, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor):
        seq_len = q.shape[1]
        cache_key = (seq_len, q.device, q.dtype)
        cached = (
            self._rotary_cache.get(cache_key) if self.cache_rotary else None
        )
        if cached is None:
            half_dim = self.head_dim // 2
            inv_freq = 1.0 / (
                10000
                ** (
                    torch.arange(
                        0, half_dim, device=q.device, dtype=torch.float32
                    )
                    / float(half_dim)
                )
            )
            positions = torch.arange(
                seq_len, device=q.device, dtype=torch.float32
            )
            freqs = torch.outer(positions, inv_freq)
            cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1).to(
                dtype=q.dtype
            )
            sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1).to(
                dtype=q.dtype
            )
            cos = cos.unsqueeze(0).unsqueeze(2)
            sin = sin.unsqueeze(0).unsqueeze(2)
            if self.cache_rotary:
                self._rotary_cache[cache_key] = (cos, sin)
        else:
            cos, sin = cached
        q = (q * cos) + (self._rotate_half(q) * sin)
        k = (k * cos) + (self._rotate_half(k) * sin)
        return q, k

    def enable_rotary_cache(self, enabled: bool = True) -> None:
        self.cache_rotary = enabled
        if not enabled:
            self._rotary_cache.clear()

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        h = _modulate(self.norm1(x), shift_attn, scale_attn)
        batch_size, seq_len, hidden_size = h.shape
        q = self.q_proj(h).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        k = self.k_proj(h).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        v = self.v_proj(h).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        if self.use_rotary:
            q, k = self._apply_rope(q, k)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        h = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=False
        )
        h = (
            h.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, seq_len, hidden_size)
        )
        x = x + gate_attn * self.out_proj(h)
        h = _modulate(self.norm2(x), shift_mlp, scale_mlp)
        return x + gate_mlp * self.mlp(h)


class DiTFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size, eps=1e-6, elementwise_affine=False
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        self.linear = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return self.linear(_modulate(self.norm_final(x), shift, scale))


class CTMTransformerHead(nn.Module):
    """Two-time (t -> s) DiT head (bundle weight layout)."""

    def __init__(
        self,
        latent_size: int,
        cond_size: int,
        hidden_size: int = 768,
        num_layers: int = 8,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        use_positional: bool = True,
        guidance: bool = False,
    ) -> None:
        super().__init__()
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.s_embedder = TimestepEmbedder(hidden_size)
        self.omega_embedder = TimestepEmbedder(hidden_size) if guidance else None
        self.null_conditioning = (
            nn.Parameter(torch.zeros(1, 1, hidden_size)) if guidance else None
        )
        self.latent_proj = nn.Linear(latent_size, hidden_size, bias=False)
        self.cond_proj = nn.Linear(cond_size, hidden_size, bias=False)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio,
                    use_rotary=use_positional,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer = DiTFinalLayer(hidden_size, latent_size)

    def project_conditioning(self, conditioning: torch.Tensor) -> torch.Tensor:
        return self.cond_proj(conditioning)

    def forward_ts_with_projected_conditioning(
        self,
        noisy_latents: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
        projected_conditioning: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        omega: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = noisy_latents.shape
        x = self.latent_proj(noisy_latents)
        embedding = self.t_embedder(t) + self.s_embedder(s)
        if omega is not None:
            if self.omega_embedder is None:
                raise RuntimeError("omega supplied to a non-guided CTM head")
            embedding = embedding + self.omega_embedder(1.0 - omega.reciprocal())
        time_embed = (
            embedding
            .unsqueeze(1)
            .expand(batch_size, seq_len, -1)
        )
        c = projected_conditioning + time_embed
        for block in self.blocks:
            x = block(x, c, attn_mask)
        return self.final_layer(x, c)

    def enable_rotary_cache(self, enabled: bool = True) -> None:
        for block in self.blocks:
            block.enable_rotary_cache(enabled)


class CTMOneStep(nn.Module):
    """The exact t=999 -> s=0 CTM update, constants baked from the bundle."""

    def __init__(self, head: CTMTransformerHead, ctm_config: dict) -> None:
        super().__init__()
        self.head = head
        self.register_buffer("t", torch.tensor([999.0]))
        self.register_buffer("s", torch.tensor([0.0]))
        self.register_buffer(
            "omega", torch.tensor([float(ctm_config.get("guidance_scale", 1.0))])
        )
        for name in ("alpha_t", "sigma_t", "c_skip", "c_out"):
            self.register_buffer(
                name, torch.tensor(float(ctm_config[name])).view(1, 1, 1)
            )

    def forward(
        self, conditioning: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        projected = self.head.project_conditioning(conditioning)
        velocity = self.head.forward_ts_with_projected_conditioning(
            noise, self.t, self.s, projected,
            omega=self.omega if self.head.omega_embedder is not None else None,
        )
        x0 = self.alpha_t * noise - self.sigma_t * velocity
        return (self.c_skip * noise + self.c_out * x0).to(conditioning.dtype)


# ---- decoder wrapper --------------------------------------------------------


class LatentWaveFitDecoder(nn.Module):
    """Latent denormalization + speaker repack + WaveFitNeXt."""

    def __init__(
        self,
        wavefit: nn.Module,
        latent_mean: torch.Tensor,
        latent_std: torch.Tensor,
        latent_dim: int,
        skip_initial_stft: bool = True,
    ) -> None:
        super().__init__()
        self.wavefit = wavefit
        self.latent_dim = int(latent_dim)
        self.skip_initial_stft = skip_initial_stft
        self.register_buffer("latent_mean", latent_mean.detach().float())
        self.register_buffer("latent_std", latent_std.detach().float())

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        raw = latents.float() * self.latent_std + self.latent_mean
        speakers = (
            torch.cat(
                [raw[..., : self.latent_dim], raw[..., self.latent_dim :]],
                dim=0,
            )
            .transpose(1, 2)
            .contiguous()
        )
        if self.skip_initial_stft:
            return self.wavefit(speakers, skip_initial_stft=True)
        return self.wavefit(speakers)


# ---- bundle loader ----------------------------------------------------------


class FastDialogueSidon:
    def __init__(self, bundle_dir: Path, device: str = "cuda") -> None:
        from safetensors.torch import load_file
        from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel

        bundle_dir = Path(bundle_dir)
        self.config = json.loads((bundle_dir / "config.json").read_text())
        device_t = torch.device(device)

        encoder_config = Wav2Vec2BertConfig.from_dict(
            json.loads((bundle_dir / "encoder/config.json").read_text())
        )
        ssl = Wav2Vec2BertModel(encoder_config)
        replace_attention_with_flex(ssl)
        self.encoder = DialogueEncoder(
            ssl, latent_size=int(self.config["latent_dim"])
        )
        state = load_file(str(bundle_dir / "encoder/model.safetensors"))
        self.encoder.load_state_dict(state, strict=True)
        self.encoder.to(device_t).eval()

        head_config = json.loads(
            (bundle_dir / "diffusion_head/config.json").read_text()
        )
        self.head = CTMTransformerHead(
            latent_size=int(head_config["latent_size"]),
            cond_size=int(head_config["cond_size"]),
            hidden_size=int(head_config["hidden_size"]),
            num_layers=int(head_config["head_layers"]),
            num_heads=int(head_config["num_heads"]),
            ffn_ratio=float(head_config["head_ffn_ratio"]),
            use_positional=bool(head_config["use_positional"]),
            guidance=bool(head_config.get("guidance", False)),
        )
        state = load_file(str(bundle_dir / "diffusion_head/model.safetensors"))
        self.head.load_state_dict(state, strict=True)
        self.head.to(device_t).eval()
        self.ctm = CTMOneStep(self.head, self.config["ctm"]).to(device_t).eval()

        vocoder_config = json.loads(
            (bundle_dir / "vocoder/config.json").read_text()
        )
        self.vocoder = WaveFitNeXtLatentDecoder(
            num_iteration=int(vocoder_config["num_iteration"]),
            latent_dim=int(vocoder_config["latent_dim"]),
            sample_rate=int(vocoder_config["sample_rate"]),
            latent_rate=int(vocoder_config["latent_rate"]),
            n_fft_stft=int(vocoder_config["n_fft_stft"]),
            generator_args=dict(vocoder_config["generator_args"]),
            latent_upsample=int(vocoder_config.get("latent_upsample", 1)),
        )
        state = load_file(str(bundle_dir / "vocoder/model.safetensors"))
        self.vocoder.load_state_dict(state, strict=True)
        self.vocoder.to(device_t).eval()

        stats = load_file(str(bundle_dir / "stats.safetensors"))
        self.latent_mean = stats["latent_norm_mean"].to(device_t)
        self.latent_std = stats["latent_norm_std"].to(device_t)
        self.decoder = (
            LatentWaveFitDecoder(
                self.vocoder,
                self.latent_mean,
                self.latent_std,
                latent_dim=int(self.config["latent_dim"]),
            )
            .to(device_t)
            .eval()
        )


def load_bundle(bundle_dir, device: str = "cuda") -> FastDialogueSidon:
    return FastDialogueSidon(Path(bundle_dir), device=device)
