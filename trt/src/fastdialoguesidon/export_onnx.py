"""ONNX export wrappers for the three TensorRT engines.

- Wave encoder: raw 16 kHz waveform -> conditioning [B, L, 1088], with the
  SeamlessM4T fbank front-end folded in as fp32 ONNX ops and the fp16
  encoder using the sidon::FlexRelAttnQRD plugin op (QR bias table
  computed by a plain MatMul node, [B, L, H, D] layout — no transposes).
- DiT head: exact t=999 -> s=0 CTM one-step with fp16-safe constants
  (time embedding precomputed, RoPE tables baked as fp32 initializers).
- Decoder: latent denorm + speaker repack + WaveFitNeXt; iteration-1 uses
  the zero-STFT sliced-conv path, iteration-2 a fixed conv1d STFT.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import flexrel_triton
from .fbank_export import SeamlessFbankFrontend
from .modeling import DialogueEncoder, CTMOneStep, FlexRelSelfAttention


# ---- encoder ----------------------------------------------------------------


class ResampleFront(nn.Module):
    """Sinc resampler as a single conv1d, bit-exact vs torchaudio.

    torchaudio.functional.resample IS conv1d(sinc_kernel, stride) under the
    hood; baking the same kernel here lets the engine take waveform at the
    deployment sample rate directly. Input length must be a multiple of
    (orig_rate // gcd) * 320 so the resampled length lands on 20 ms frames.
    """

    def __init__(self, orig_rate: int, new_rate: int = 16_000) -> None:
        super().__init__()
        from torchaudio.functional.functional import (
            _get_sinc_resample_kernel,
        )

        gcd = math.gcd(orig_rate, new_rate)
        self.orig_g = orig_rate // gcd
        self.new_g = new_rate // gcd
        kernel, self.width = _get_sinc_resample_kernel(
            orig_rate, new_rate, gcd, 6, 0.99, "sinc_interp_hann", None,
            torch.device("cpu"), torch.float64,
        )
        self.register_buffer("kernel", kernel.float())

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        batch, samples = waveform.shape
        padded = F.pad(
            waveform.unsqueeze(1), (self.width, self.width + self.orig_g)
        )
        resampled = F.conv1d(padded, self.kernel, stride=self.orig_g)
        resampled = resampled.transpose(1, 2).reshape(batch, -1)
        return resampled[:, : samples * self.new_g // self.orig_g]


class ExportFlexRelSelfAttention(nn.Module):
    """FlexRelSelfAttention twin whose core is the TRT plugin custom op."""

    def __init__(
        self, source: FlexRelSelfAttention, fold_qkv_bias: bool
    ) -> None:
        super().__init__()
        self.num_heads = source.num_heads
        self.head_dim = source.head_dim
        self.fold_qkv_bias = fold_qkv_bias
        self.linear_out = source.linear_out
        self.distance_embedding = source.distance_embedding
        if fold_qkv_bias:
            # Blackwell Myelin can segfault at create_execution_context()
            # when a GEMM+bias epilogue feeds the AOT plugin. Fold each qkv
            # bias into its GEMM (ones column appended to the activations).
            for name, linear in (
                ("wq_aug", source.linear_q),
                ("wk_aug", source.linear_k),
                ("wv_aug", source.linear_v),
            ):
                self.register_buffer(
                    name,
                    torch.cat(
                        [linear.weight.t(), linear.bias[None, :]], dim=0
                    ).detach().clone(),
                )
        else:
            # Pre-Blackwell TensorRT handles the bias epilogue correctly,
            # and this form is ~3% faster for the encoder on H100.
            self.linear_q = source.linear_q
            self.linear_k = source.linear_k
            self.linear_v = source.linear_v

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
        if self.fold_qkv_bias:
            augmented = torch.cat(
                [hidden_states, hidden_states.new_ones(batch, length, 1)],
                dim=-1,
            )
            query = torch.matmul(augmented, self.wq_aug).view(shape)
            key = torch.matmul(augmented, self.wk_aug).view(shape)
            value = torch.matmul(augmented, self.wv_aug).view(shape)
        else:
            query = self.linear_q(hidden_states).view(shape)
            key = self.linear_k(hidden_states).view(shape)
            value = self.linear_v(hidden_states).view(shape)
        query_rel = torch.matmul(query, self.distance_embedding.weight.t())
        context = flexrel_triton.FlexRelAttnQRDynOp.apply(
            query, key, value, query_rel
        )
        return self.linear_out(context.reshape(batch, length, hidden)), None


def fold_qkv_bias_for_capability(capability: tuple[int, int]) -> bool:
    """Use the Blackwell-safe QKV graph on sm_100 and newer GPUs."""
    return capability >= (10, 0)


def swap_flex_for_export(
    encoder: DialogueEncoder, fold_qkv_bias: bool
) -> int:
    count = 0
    for layer in encoder.model.encoder.layers:
        assert isinstance(layer.self_attn, FlexRelSelfAttention)
        layer.self_attn = ExportFlexRelSelfAttention(
            layer.self_attn, fold_qkv_bias=fold_qkv_bias
        )
        count += 1
    return count


class WaveEncoderWrapper(nn.Module):
    """fp32 (resample +) fbank front-end + reduced-precision encoder,
    strongly typed. The build default is fp16 (more accurate than bf16 at
    the same speed); bf16 is selectable."""

    def __init__(
        self,
        front: nn.Module,
        encoder: nn.Module,
        resampler: Optional[nn.Module] = None,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.resampler = resampler
        self.front = front
        self.encoder = encoder
        self._dtype = dtype

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.resampler is not None:
            waveform = self.resampler(waveform)
        features_fp32 = self.front(waveform)
        return self.encoder(features_fp32.to(self._dtype), None).float()


def export_wave_encoder(
    encoder: DialogueEncoder,
    waveform_example: torch.Tensor,
    onnx_path: Path,
    input_rate: int = 16_000,
    dtype: torch.dtype = torch.float16,
) -> None:
    """encoder must be a throwaway copy — attentions are swapped in place
    and the encoder is cast to `dtype` (callers pass fp16 by default). waveform_example
    must be at input_rate; a sinc-conv resample front is prepended when
    != 16 kHz."""
    capability = torch.cuda.get_device_capability(waveform_example.device)
    fold_qkv_bias = fold_qkv_bias_for_capability(capability)
    qkv_mode = "bias-folded" if fold_qkv_bias else "linear+bias"
    print(
        f"encoder qkv export: sm_{capability[0]}{capability[1]} "
        f"uses {qkv_mode}",
        flush=True,
    )
    swap_flex_for_export(encoder, fold_qkv_bias=fold_qkv_bias)
    resampler = (
        None
        if input_rate == 16_000
        else ResampleFront(input_rate).to(waveform_example.device)
    )
    wrapper = (
        WaveEncoderWrapper(
            SeamlessFbankFrontend().to(waveform_example.device),
            encoder.to(dtype),
            resampler=resampler,
            dtype=dtype,
        )
        .to(waveform_example.device)
        .eval()
    )
    torch.onnx.export(
        wrapper,
        (waveform_example,),
        str(onnx_path),
        input_names=["waveform"],
        output_names=["conditioning"],
        dynamic_axes={
            "waveform": {0: "batch", 1: "samples"},
            "conditioning": {0: "batch", 1: "seq"},
        },
        opset_version=17,
        dynamo=False,
    )


# ---- DiT head ---------------------------------------------------------------


class ExportableCTM(nn.Module):
    """ONNX-exportable CTM one-step with FP16-safe constants.

    RoPE cos/sin and the t=999/s=0 timestep embedding are constant for
    this pipeline and numerically destroyed by fp16 in-engine evaluation,
    so both are precomputed as fp32 initializers (at the MAX sequence
    length; runtime lengths slice a prefix).
    """

    def __init__(self, ctm: CTMOneStep, seq_len: int) -> None:
        super().__init__()
        self.head = ctm.head
        for name in ("alpha_t", "sigma_t", "c_skip", "c_out"):
            self.register_buffer(name, getattr(ctm, name).clone())
        with torch.no_grad():
            time_embed = self.head.t_embedder(ctm.t) + self.head.s_embedder(ctm.s)
            if self.head.omega_embedder is not None:
                time_embed = time_embed + self.head.omega_embedder(
                    1.0 - ctm.omega.reciprocal()
                )
        self.register_buffer("time_embed", time_embed.float())
        block = self.head.blocks[0]
        half_dim = block.head_dim // 2
        inv_freq = 1.0 / (
            10000
            ** (
                torch.arange(0, half_dim, dtype=torch.float32)
                / float(half_dim)
            )
        )
        freqs = torch.outer(
            torch.arange(seq_len, dtype=torch.float32), inv_freq
        )
        cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
        sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
        self.register_buffer("rope_cos", cos.unsqueeze(0).unsqueeze(2))
        self.register_buffer("rope_sin", sin.unsqueeze(0).unsqueeze(2))

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        first, second = x.chunk(2, dim=-1)
        return torch.cat([-second, first], dim=-1)

    @staticmethod
    def _modulate(x, shift, scale):
        return x * (1 + scale) + shift

    def _block_forward(self, block, x, c):
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            block.adaLN_modulation(c).chunk(6, dim=-1)
        )
        h = self._modulate(block.norm1(x), shift_attn, scale_attn)
        batch, length, hidden = h.shape
        q = block.q_proj(h).view(batch, length, block.num_heads, block.head_dim)
        k = block.k_proj(h).view(batch, length, block.num_heads, block.head_dim)
        v = block.v_proj(h).view(batch, length, block.num_heads, block.head_dim)
        rope_cos = self.rope_cos[:, :length]
        rope_sin = self.rope_sin[:, :length]
        q = q * rope_cos + self._rotate_half(q) * rope_sin
        k = k * rope_cos + self._rotate_half(k) * rope_sin
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.permute(0, 2, 1, 3).reshape(batch, length, hidden)
        x = x + gate_attn * block.out_proj(h)
        h = self._modulate(block.norm2(x), shift_mlp, scale_mlp)
        return x + gate_mlp * block.mlp(h)

    def forward(
        self, conditioning: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        head = self.head
        projected = head.project_conditioning(conditioning)
        x = head.latent_proj(noise)
        c = projected + self.time_embed.unsqueeze(1)
        for block in head.blocks:
            x = self._block_forward(block, x, c)
        velocity = head.final_layer(x, c)
        x0 = self.alpha_t * noise - self.sigma_t * velocity
        return self.c_skip * noise + self.c_out * x0


def _wrap_layernorms_fp32(
    module: nn.Module, dtype: torch.dtype = torch.bfloat16
) -> int:
    """Compute every LayerNorm in fp32 with casts (autocast-style)."""
    count = 0
    for m in module.modules():
        if isinstance(m, nn.LayerNorm):
            def fwd(x, _m=m):
                w = _m.weight.float() if _m.weight is not None else None
                b = _m.bias.float() if _m.bias is not None else None
                return F.layer_norm(
                    x.float(), _m.normalized_shape, w, b, _m.eps
                ).to(dtype)
            m.forward = fwd
            count += 1
    return count


class STExportableCTM(nn.Module):
    """Strongly-typed DiT export: fp16 (default) or bf16 weights, fp32
    LayerNorms, fp32 IO.

    Measured vs the weak-typed BF16-flag build: -0.3 ms and +0.1 dB —
    the ST build lets myelin pick better tactics.
    """

    def __init__(
        self, exportable_ctm: ExportableCTM, dtype: torch.dtype = torch.float16
    ) -> None:
        super().__init__()
        self._dtype = dtype
        # precomputed RoPE tables / time embedding are cast with the weights
        # (they are tables, not in-graph trig, so fp16/bf16 is safe here)
        self.m = exportable_ctm.to(dtype)
        _wrap_layernorms_fp32(self.m, dtype)

    def forward(
        self, conditioning: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        return self.m(
            conditioning.to(self._dtype), noise.to(self._dtype)
        ).float()


def export_dit(
    exportable_ctm: nn.Module,
    conditioning_example: torch.Tensor,
    noise_example: torch.Tensor,
    onnx_path: Path,
) -> None:
    torch.onnx.export(
        exportable_ctm,
        (conditioning_example, noise_example),
        str(onnx_path),
        input_names=["conditioning", "noise"],
        output_names=["latents"],
        dynamic_axes={
            "conditioning": {0: "batch", 1: "seq"},
            "noise": {0: "batch", 1: "seq"},
            "latents": {0: "batch", 1: "seq"},
        },
        opset_version=17,
        dynamo=False,
    )


# ---- decoder ----------------------------------------------------------------


class ConvSTFT(nn.Module):
    """torch.stft(center=True, reflect pad) as slice+concat+GEMM.

    Produces the WaveNeXtGenerator channel layout:
    ``cat([real (n_fft//2+1), imag[1:-1] (n_fft//2-1)])`` = n_fft channels.
    Frames (kernel n_fft @ hop) are materialized as ceil(n_fft/hop)+1
    hop-sized rows so the DFT is one [T, rows*hop] x [rows*hop, n_fft]
    GEMM — TRT runs a C_in=1 big-kernel conv1d ~15x slower (2.1 ms
    profiled) than the equivalent GEMM.
    """

    def __init__(self, n_fft: int, hop: int, window: torch.Tensor) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.rows_per_frame = -(-n_fft // hop)  # ceil(n_fft / hop)
        positions = torch.arange(n_fft, dtype=torch.float64)
        bins = torch.arange(n_fft // 2 + 1, dtype=torch.float64)
        angles = 2.0 * math.pi * bins[:, None] * positions[None, :] / n_fft
        window64 = window.detach().cpu().double()[None, :]
        real_weight = (torch.cos(angles) * window64).float()
        imag_weight = (-torch.sin(angles) * window64).float()[1:-1]
        weight = torch.cat([real_weight, imag_weight], dim=0)  # [n_fft, n_fft]
        weight = F.pad(
            weight, (0, self.rows_per_frame * hop - n_fft)
        )  # [n_fft, rows*hop]
        self.register_buffer("weight", weight.t().contiguous())

    def forward(self, waveform: torch.Tensor, num_frame: int) -> torch.Tensor:
        # waveform [B, 1, num_frame*hop]
        padded = F.pad(
            waveform, (self.n_fft // 2, self.n_fft // 2), mode="reflect"
        )[:, 0]
        batch, total = padded.shape
        # zero-pad so the row grid is hop-aligned and covers the last frame
        target = (
            (total + self.rows_per_frame * self.hop + self.hop - 1)
            // self.hop
        ) * self.hop
        padded = F.pad(padded, (0, target - total))
        rows = padded.reshape(batch, -1, self.hop)
        pieces = [
            rows[:, offset : offset + num_frame]
            for offset in range(self.rows_per_frame)
        ]
        frames = torch.cat(pieces, dim=-1)  # [B, T, rows*hop]
        spectrum = torch.matmul(frames, self.weight)  # [B, T, n_fft]
        return spectrum.transpose(1, 2)


class ExportableWaveFitDecoder(nn.Module):
    """latents [B, L, 64] -> audio [2B, 1, samples]."""

    def __init__(
        self,
        wavefit: nn.Module,
        latent_mean: torch.Tensor,
        latent_std: torch.Tensor,
        latent_dim: int,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.register_buffer("latent_mean", latent_mean.detach().float())
        self.register_buffer("latent_std", latent_std.detach().float())
        upsampler = wavefit.upsampler
        if upsampler is not None:
            # sm_120 (Blackwell) Myelin has no BF16 ConvTranspose tactic in
            # TRT 10.16 (engine build fails); carry the upsampler as a GEMM
            # weight instead — see _upsample_gemm.
            weight = upsampler.weight.detach()  # [C_in, C_out, 2*stride]
            self.up_stride = int(upsampler.stride[0])
            self.up_pad = int(upsampler.padding[0])
            self.register_buffer(
                "up_weight",
                weight.permute(0, 2, 1).reshape(weight.shape[0], -1).clone(),
            )  # [C_in, 2*stride*C_out]
            self.register_buffer("up_bias", upsampler.bias.detach().clone())
        else:
            self.up_weight = None
        self.generators = wavefit.generator
        generator = wavefit.generator[0]
        self.conv_stft = ConvSTFT(
            generator.n_fft, generator.hop_size, generator.fft_win
        )

    def _trunk(self, generator: nn.Module, x: torch.Tensor) -> torch.Tensor:
        x = generator.input_norm(x)
        for block in generator.blocks:
            x = block(x)
        x = generator.final_norm(x)
        x = x.transpose(1, 2)
        x = generator.linear2(generator.linear1(x))
        return x.reshape(x.shape[0], 1, x.shape[1] * generator.hop_size)

    @staticmethod
    def _proj_k3_gemm(
        features: torch.Tensor, conv: nn.Conv1d
    ) -> torch.Tensor:
        """k=3/pad=1 conv1d as slice+concat+GEMM ([B,C,T] -> [B,Cout,T]).

        TRT profiles the direct conv at 1.7 ms (fp32 fallback); the GEMM
        form runs on bf16 tensor cores.
        """
        batch, channels, num_frame = features.shape
        transposed = F.pad(features, (1, 1)).transpose(1, 2)  # [B,T+2,C]
        windows = torch.cat(
            [
                transposed[:, 0:num_frame],
                transposed[:, 1 : num_frame + 1],
                transposed[:, 2 : num_frame + 2],
            ],
            dim=-1,
        )  # [B, T, 3C]
        weight = (
            conv.weight.permute(2, 1, 0).reshape(3 * channels, -1)
        )  # [3C, Cout]
        out = torch.matmul(windows, weight) + conv.bias
        return out.transpose(1, 2)

    def _upsample_gemm(self, latents: torch.Tensor) -> torch.Tensor:
        """ConvTranspose1d(C, C, K=2s, stride=s, pad=s//2) as GEMM +
        phase interleave ([B,C,L] -> [B,C,L*s]).

        sm_120 Myelin (TRT 10.16) cannot build the BF16 ConvTranspose;
        the GEMM form runs on bf16 tensor cores everywhere.
        """
        batch, _, num_frame = latents.shape
        stride = self.up_stride
        mixed = torch.matmul(
            latents.transpose(1, 2), self.up_weight
        ).reshape(batch, num_frame, 2, stride, -1)
        low = mixed[:, :, 0].reshape(batch, num_frame * stride, -1)
        high = mixed[:, :, 1].reshape(batch, num_frame * stride, -1)
        overlapped = (
            F.pad(low, (0, 0, 0, stride)) + F.pad(high, (0, 0, stride, 0))
        )
        # crop end-relative only: start:start+L*s bakes the length into the
        # trace and breaks dynamic-shape export
        cropped = overlapped[:, self.up_pad : -self.up_pad]
        return (cropped + self.up_bias).transpose(1, 2)

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
        if self.up_weight is not None:
            speakers = self._upsample_gemm(speakers)
        num_frame = speakers.shape[-1]
        generator_1 = self.generators[0]
        x = F.conv1d(
            speakers,
            generator_1.spec_proj.weight[:, : speakers.shape[1]],
            generator_1.spec_proj.bias,
            padding=generator_1.spec_proj.padding[0],
        )
        waveform = self._trunk(generator_1, x)
        generator_2 = self.generators[1]
        stft_feat = self.conv_stft(waveform, num_frame)
        x = self._proj_k3_gemm(
            torch.cat([speakers, stft_feat], dim=1), generator_2.spec_proj
        )
        return self._trunk(generator_2, x)


def export_decoder(
    decoder: ExportableWaveFitDecoder,
    latents_example: torch.Tensor,
    onnx_path: Path,
) -> None:
    torch.onnx.export(
        decoder,
        (latents_example,),
        str(onnx_path),
        input_names=["latents"],
        output_names=["audio"],
        dynamic_axes={
            "latents": {0: "batch", 1: "seq"},
            "audio": {0: "speaker_batch", 2: "samples"},
        },
        opset_version=17,
        dynamo=False,
    )
