"""FastDialogueSidon forward in Triton kernels + cuBLAS GEMMs.

Weights are taken from the fp32 reference modules (``reference.modeling``) and
re-laid out once at load time. Exact algebraic folds (computed in fp64, then
cast):

- encoder FFN ``* 0.5`` folded into the output dense weight/bias;
- speaker heads + ``cond_proj`` + the constant t=999/s=0 time embedding folded
  into one [1024 -> 768] projection (the 1088-wide conditioning is never
  materialized);
- decoder ConvNeXt layer-scale ``gamma`` folded into pwconv2;
- decoder ``linear2(linear1(x))`` folded into one [512 -> hop] projection.

Precision: fp16 (default) or bf16 GEMM operands with fp32 accumulation, fp32 residual streams,
LayerNorm / softmax / RoPE / depthwise convs in fp32 registers. The fbank
front-end, latent denorm/upsampler and the CTM update run in fp32.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from . import kernels as K
from .reference.fbank_export import SeamlessFbankFrontend

BF16 = torch.bfloat16
FP16 = torch.float16
# DiT attention (no bias): cuDNN is ~1.45x faster than flash on H100 at
# L=6000/12 heads and CUDA-graph safe; flash is the fallback elsewhere.
SDPA_BACKENDS = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]


def _cast(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return t.detach().to(dtype).contiguous()


def _f32(t: torch.Tensor) -> torch.Tensor:
    return t.detach().float().contiguous()


def _ln(module) -> tuple:
    return (_f32(module.weight), _f32(module.bias), float(module.eps))


def linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None = None) -> torch.Tensor:
    return F.linear(x, w, b)


def accumulate(x: torch.Tensor, a: torch.Tensor, w: torch.Tensor) -> None:
    """x += a @ w^T in place: fp16/bf16 operands, fp32 accumulate straight into
    the fp32 residual (cuBLAS beta=1) — no reduced-precision GEMM output, no
    separate add."""
    torch.addmm(x, a, w.t(), out_dtype=torch.float32, out=x)


def linear_gelu(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, tanh: bool) -> torch.Tensor:
    """GELU(x W^T + b). tanh=True (default): cuBLASLt bias+GELU epilogue, tanh
    form, one memory pass fewer (-0.6 ms / 120 s; measured equal or better
    SNR vs the fp32 erf reference). tanh=False: exact erf GELU."""
    if tanh:
        return torch._addmm_activation(b, x, w.t(), use_gelu=True)
    return F.gelu(F.linear(x, w, b))


class TritonModel:
    """Holds the re-laid-out weights and the three stage forwards."""

    def __init__(self, bundle, max_seq: int = 6000, gelu_tanh: bool = True,
                 accumulate_rows: int | None = 0, attn_impl: str | None = None,
                 dtype: torch.dtype = FP16) -> None:
        """dtype: GEMM operand / activation storage type, fp16 (default) or
        bf16. Residual streams, norms, softmax and accumulation stay fp32,
        so fp16 has no overflow here and is ~15 dB closer to the fp32
        reference than bf16 at the same speed (measured 2026-10-26).
        accumulate_rows: encoder residual GEMMs accumulate straight into the
        fp32 stream when batch*frames >= this (None: never, 0: always).
        attn_impl: encoder attention, "split" (cuDNN causal halves + Triton band
        merge), "fused" (one Triton kernel) or "auto" (default): split for
        windows >= ``attn_split_min_len`` frames when cuDNN SDPA is available
        (H100 crossover between 80 and 90 s), fused otherwise."""
        attn_impl = attn_impl or "auto"
        assert attn_impl in ("auto", "split", "fused")
        self.attn_impl = attn_impl
        self.attn_split_min_len = 4250
        self._cudnn_attn = K.cudnn_attention_available()
        self.gelu_tanh = gelu_tanh
        self.accumulate_rows = accumulate_rows
        assert dtype in (BF16, FP16)
        self.dtype = dtype
        self.device = next(bundle.encoder.parameters()).device
        with torch.no_grad():
            self._prep_encoder(bundle)
            self._prep_dit(bundle, max_seq)
            self._prep_decoder(bundle)
        self.front = SeamlessFbankFrontend().to(self.device).eval()

    # ------------------------------------------------------------ weights

    def _w(self, t: torch.Tensor) -> torch.Tensor:
        return _cast(t, self.dtype)

    def _res_norm(self, *args, **kwargs):
        return K.res_norm(*args, xn_dtype=self.dtype, **kwargs)

    def _prep_encoder(self, bundle) -> None:
        ssl = bundle.encoder.model
        cfg = ssl.config
        self.enc_heads = int(cfg.num_attention_heads)
        fp = ssl.feature_projection
        self.fp_ln = _ln(fp.layer_norm)
        self.fp_w, self.fp_b = self._w(fp.projection.weight), self._w(fp.projection.bias)
        layers = []
        for layer in ssl.encoder.layers:
            att = layer.self_attn
            conv = layer.conv_module
            p = {
                "ffn1_ln": _ln(layer.ffn1_layer_norm),
                "ffn1_w1": self._w(layer.ffn1.intermediate_dense.weight),
                "ffn1_b1": self._w(layer.ffn1.intermediate_dense.bias),
                "ffn1_w2": self._w(layer.ffn1.output_dense.weight * 0.5),
                "ffn1_b2": self._w(layer.ffn1.output_dense.bias * 0.5),
                "attn_ln": _ln(layer.self_attn_layer_norm),
                "qkv_w": self._w(torch.cat([att.linear_q.weight, att.linear_k.weight, att.linear_v.weight])),
                "qkv_b": self._w(torch.cat([att.linear_q.bias, att.linear_k.bias, att.linear_v.bias])),
                "rel": self._w(att.distance_embedding.weight),
                "out_w": self._w(att.linear_out.weight),
                "out_b": self._w(att.linear_out.bias),
                "conv_ln": _ln(conv.layer_norm),
                "pw1_w": self._w(conv.pointwise_conv1.weight[:, :, 0]),
                "dwt": _f32(conv.depthwise_conv.weight[:, 0, :].t()),
                "dw_ln": _ln(conv.depthwise_layer_norm),
                "pw2_w": self._w(conv.pointwise_conv2.weight[:, :, 0]),
                "ffn2_ln": _ln(layer.ffn2_layer_norm),
                "ffn2_w1": self._w(layer.ffn2.intermediate_dense.weight),
                "ffn2_b1": self._w(layer.ffn2.intermediate_dense.bias),
                "ffn2_w2": self._w(layer.ffn2.output_dense.weight * 0.5),
                "ffn2_b2": self._w(layer.ffn2.output_dense.bias * 0.5),
                "final_ln": _ln(layer.final_layer_norm),
            }
            assert att.left_max_position == K.LEFT and att.right_max_position == K.RIGHT
            # Residual GEMMs accumulate bias-free into the fp32 stream (cuBLAS
            # beta=1); the biases they skip are carried as a running vector and
            # added inside the next norm kernel (exact: x_true = x + bias).
            b1 = layer.ffn1.output_dense.bias.double() * 0.5
            b2 = b1 + att.linear_out.bias.double()
            b4 = b2 + layer.ffn2.output_dense.bias.double() * 0.5
            p["cum_bias"] = (_f32(b1), _f32(b2), _f32(b4))
            layers.append(p)
        self.enc_layers = layers

    def _prep_dit(self, bundle, max_seq: int) -> None:
        ctm, head = bundle.ctm, bundle.head
        enc = bundle.encoder
        latent = enc.speaker_1.weight.shape[0]
        wc = head.cond_proj.weight.double()  # [768, 2*latent + hidden]
        w_fold = (
            wc[:, :latent] @ enc.speaker_1.weight.double()
            + wc[:, latent:2 * latent] @ enc.speaker_2.weight.double()
            + wc[:, 2 * latent:]
        )
        b_fold = (
            wc[:, :latent] @ enc.speaker_1.bias.double()
            + wc[:, latent:2 * latent] @ enc.speaker_2.bias.double()
        )
        time_embed = head.t_embedder(ctm.t) + head.s_embedder(ctm.s)
        if head.omega_embedder is not None:
            time_embed = time_embed + head.omega_embedder(1.0 - ctm.omega.reciprocal())
        self.cond_w = self._w(w_fold)
        self.cond_b = _f32(b_fold + time_embed[0].double())
        self.latent_w = self._w(head.latent_proj.weight)
        block0 = head.blocks[0]
        self.dit_heads = block0.num_heads
        self.dit_hidden = block0.q_proj.weight.shape[0]
        self.dit_blocks = [
            {
                "ada_w": self._w(blk.adaLN_modulation[1].weight),
                "ada_b": self._w(blk.adaLN_modulation[1].bias),
                "qkv_w": self._w(torch.cat([blk.q_proj.weight, blk.k_proj.weight, blk.v_proj.weight])),
                "qkv_b": self._w(torch.cat([blk.q_proj.bias, blk.k_proj.bias, blk.v_proj.bias])),
                "out_w": self._w(blk.out_proj.weight),
                "out_b": self._w(blk.out_proj.bias),
                "mlp_w1": self._w(blk.mlp[0].weight),
                "mlp_b1": self._w(blk.mlp[0].bias),
                "mlp_w2": self._w(blk.mlp[2].weight),
                "mlp_b2": self._w(blk.mlp[2].bias),
                "eps1": float(blk.norm1.eps),
                "eps2": float(blk.norm2.eps),
            }
            for blk in head.blocks
        ]
        final = head.final_layer
        self.final_ada_w = self._w(final.adaLN_modulation[1].weight)
        self.final_ada_b = self._w(final.adaLN_modulation[1].bias)
        self.final_eps = float(final.norm_final.eps)
        self.final_w = self._w(final.linear.weight)
        self.ctm_consts = {n: float(getattr(ctm, n)) for n in ("alpha_t", "sigma_t", "c_skip", "c_out")}
        half = block0.head_dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half, dtype=torch.float32) / float(half)))
        freqs = torch.outer(torch.arange(max_seq, dtype=torch.float32), inv_freq)
        self.rope_cos = torch.cat([freqs.cos(), freqs.cos()], -1).to(self.device).contiguous()
        self.rope_sin = torch.cat([freqs.sin(), freqs.sin()], -1).to(self.device).contiguous()

    def _prep_decoder(self, bundle) -> None:
        voc = bundle.vocoder
        self.latent_dim = int(voc.latent_dim)
        self.latent_mean = _f32(bundle.latent_mean)
        self.latent_std = _f32(bundle.latent_std)
        up = voc.upsampler
        self.up = None
        if up is not None:
            stride = int(up.stride[0])
            assert up.kernel_size[0] == 2 * stride and up.padding[0] == stride // 2 and stride == 2
            w = up.weight.detach()  # [C_in, C_out, 4]
            self.up = (_f32(w.permute(0, 2, 1).reshape(w.shape[0], -1)), _f32(up.bias))
        gens = []
        for index, gen in enumerate(voc.generator):
            cin = gen.spec_proj.weight.shape[1] if index > 0 else self.latent_dim
            sw = gen.spec_proj.weight.detach()[:, :cin]  # [512, cin, 3]
            p = {
                "spec_w": [self._w(sw[:, :, k]) for k in range(3)],  # taps prev / cur / next
                "spec_b": _f32(gen.spec_proj.bias),
                "in_ln": _ln(gen.input_norm.norm),
                "final_ln": _ln(gen.final_norm.norm),
                "blocks": [
                    {
                        "dwt": _f32(blk.dwconv.weight[:, 0, :].t()),
                        "dw_b": _f32(blk.dwconv.bias),
                        "ln": _ln(blk.norm.norm),
                        "w1": self._w(blk.pwconv1.weight),
                        "b1": self._w(blk.pwconv1.bias),
                        "w2": self._w(blk.gamma[:, None] * blk.pwconv2.weight),
                        "b2": self._w(blk.gamma * blk.pwconv2.bias),
                    }
                    for blk in gen.blocks
                ],
                "out_w": self._w(gen.linear2.weight.double() @ gen.linear1.weight.double()),
                "out_b": _f32(gen.linear2.weight.double() @ gen.linear1.bias.double()),
            }
            gens.append(p)
        self.gens = gens
        gen0 = voc.generator[0]
        self.hop = int(gen0.hop_size)
        self.n_fft = int(gen0.n_fft)
        n = self.n_fft
        pos = torch.arange(n, dtype=torch.float64)
        bins = torch.arange(n // 2 + 1, dtype=torch.float64)
        ang = 2.0 * math.pi * bins[:, None] * pos[None, :] / n
        win = gen0.fft_win.detach().cpu().double()[None, :]
        stft_w = torch.cat([torch.cos(ang) * win, (-torch.sin(ang) * win)[1:-1]], 0)  # [n_fft ch, n_fft taps]
        self.stft_w = self._w(stft_w).to(self.device)

    # ------------------------------------------------------------ stages

    def encoder(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform [B, samples] fp32 16 kHz -> encoder hidden [B*L, 1024] (model dtype)."""
        batch = waveform.shape[0]
        features = self.front(waveform)  # [B, L, 160] fp32
        rows = features.shape[0] * features.shape[1]
        feats = features.reshape(rows, -1)
        fn = self._res_norm(feats, norm2=self.fp_ln, write_x=False)
        x = linear(fn, self.fp_w, self.fp_b).float()  # residual stream fp32
        layers = self.enc_layers
        xn = self._res_norm(x, norm2=layers[0]["ffn1_ln"], write_x=False)
        accum = self.accumulate_rows is not None and rows >= self.accumulate_rows
        attn = self.attn_impl
        if attn == "auto":
            length = rows // batch
            attn = "split" if self._cudnn_attn and length >= self.attn_split_min_len else "fused"
        for i, p in enumerate(layers):
            if accum:
                # residual GEMMs accumulate bias-free into x in place; the skipped
                # biases ride along as a running vector added in the next norm
                b1, b2, b4 = p["cum_bias"]
                res = lambda a, w, b, bias, **kw: (  # noqa: E731
                    accumulate(x, a, w), self._res_norm(x, bias=bias, **kw))[1]
            else:
                res = lambda a, w, b, bias, **kw: self._res_norm(x, linear(a, w, b), **kw)  # noqa: E731
                b1 = b2 = b4 = None
            last = i + 1 == len(layers)
            # FFN1 (x0.5 folded)
            h = linear(xn, p["ffn1_w1"], p["ffn1_b1"])
            F.silu(h, inplace=True)
            xn = res(h, p["ffn1_w2"], p["ffn1_b2"], b1, norm2=p["attn_ln"], write_x=not accum)
            # self-attention with in-kernel relative_key bias
            qkv = linear(xn, p["qkv_w"], p["qkv_b"])
            ctx = K.encoder_attention(qkv, batch, self.enc_heads, p["rel"], attn)
            xn = res(ctx, p["out_w"], p["out_b"], b2, norm2=p["conv_ln"], write_x=not accum)
            # conformer conv module
            g = K.glu(linear(xn, p["pw1_w"]))
            s = K.causal_dw_norm(g, batch, p["dwt"], *p["dw_ln"])
            xn = res(s, p["pw2_w"], None, b2, norm2=p["ffn2_ln"], write_x=not accum)
            # FFN2 (x0.5 folded) + final LayerNorm (+ next layer's FFN1 norm)
            h = linear(xn, p["ffn2_w1"], p["ffn2_b1"])
            F.silu(h, inplace=True)
            xn = res(h, p["ffn2_w2"], p["ffn2_b2"], b4, norm1=p["final_ln"],
                     norm2=None if last else layers[i + 1]["ffn1_ln"])
        return x.to(self.dtype)

    def dit(self, hidden: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """hidden [B*L, 1024] (model dtype), noise [B, L, 64] fp32 -> latents [B, L, 64] fp32."""
        batch, length, latent = noise.shape
        rows = batch * length
        c = torch.mm(hidden, self.cond_w.t(), out_dtype=torch.float32)
        c += self.cond_b
        silu_c = F.silu(c).to(self.dtype)
        noise_rows = noise.reshape(rows, latent)
        x = torch.mm(noise_rows.to(self.dtype), self.latent_w.t(), out_dtype=torch.float32)
        hid = self.dit_hidden
        blocks = self.dit_blocks
        mod = linear(silu_c, blocks[0]["ada_w"], blocks[0]["ada_b"])
        h = self._res_norm(x, norm2=(None, None, blocks[0]["eps1"]),
                       modulate=(mod[:, 0:hid], mod[:, hid:2 * hid]), write_x=False)
        for i, p in enumerate(blocks):
            qkv = linear(h, p["qkv_w"], p["qkv_b"])
            q, k, v = K.rope_split(qkv, length, self.dit_heads, self.rope_cos, self.rope_sin)
            shape = (batch, length, self.dit_heads, hid // self.dit_heads)
            with sdpa_kernel(SDPA_BACKENDS, set_priority=True):
                att = F.scaled_dot_product_attention(
                    q.view(shape).transpose(1, 2), k.view(shape).transpose(1, 2), v.view(shape).transpose(1, 2)
                )
            ctx = att.transpose(1, 2).reshape(rows, hid)
            y = linear(ctx, p["out_w"], p["out_b"])
            h = self._res_norm(x, y, gate=mod[:, 2 * hid:3 * hid], norm2=(None, None, p["eps2"]),
                           modulate=(mod[:, 3 * hid:4 * hid], mod[:, 4 * hid:5 * hid]))
            u = linear_gelu(h, p["mlp_w1"], p["mlp_b1"], self.gelu_tanh)
            y = linear(u, p["mlp_w2"], p["mlp_b2"])
            gate = mod[:, 5 * hid:6 * hid]
            if i + 1 < len(blocks):
                mod_next = linear(silu_c, blocks[i + 1]["ada_w"], blocks[i + 1]["ada_b"])
                eps, shift, scale = blocks[i + 1]["eps1"], mod_next[:, 0:hid], mod_next[:, hid:2 * hid]
            else:
                mod_next = linear(silu_c, self.final_ada_w, self.final_ada_b)
                eps, shift, scale = self.final_eps, mod_next[:, 0:hid], mod_next[:, hid:2 * hid]
            h = self._res_norm(x, y, gate=gate, norm2=(None, None, eps), modulate=(shift, scale))
            mod = mod_next
        velocity = torch.mm(h, self.final_w.t(), out_dtype=torch.float32).view(batch, length, latent)
        cc = self.ctm_consts
        x0 = cc["alpha_t"] * noise - cc["sigma_t"] * velocity
        return cc["c_skip"] * noise + cc["c_out"] * x0

    def _spec_proj(self, inp: torch.Tensor, w: list) -> torch.Tensor:
        """k=3/pad=1 conv over frames (bias added by the caller's norm kernel):
        inp [S, T, Cin] (model dtype) -> [S*T, 512] fp32, as three row-shifted GEMMs over
        one zero-padded copy accumulated in place (no window materialization)."""
        speakers, frames, cin = inp.shape
        flat = F.pad(inp, (0, 0, 1, 1)).view(-1, cin)
        n = flat.shape[0] - 2
        acc = torch.mm(flat[0:n], w[0].t(), out_dtype=torch.float32)
        for k in (1, 2):
            accumulate(acc, flat[k:k + n], w[k])
        # valid output row of (speaker s, frame t) is s*(frames+2) + t < n
        cout = acc.shape[1]
        res = acc.as_strided((speakers, frames, cout), ((frames + 2) * cout, cout, 1))
        return res.reshape(speakers * frames, cout)

    def _trunk(self, p: dict, spec: torch.Tensor, speakers: int, out_fp32: bool) -> torch.Tensor:
        """spec_proj output (no bias) [S*T, 512] fp32 -> waveform [S, T*hop]."""
        self._res_norm(spec, norm1=p["in_ln"], bias=p["spec_b"])  # in place: input_norm(spec + b)
        x = spec
        x_alt = torch.empty_like(x)
        y = None
        for blk in p["blocks"]:
            out_x = x_alt if y is not None else x
            xn = K.convnext_dw_norm(x, y, speakers, blk["dwt"], blk["dw_b"], *blk["ln"], out_x=out_x,
                                    xn_dtype=self.dtype)
            if y is not None:
                x, x_alt = x_alt, x
            u = linear_gelu(xn, blk["w1"], blk["b1"], self.gelu_tanh)
            y = linear(u, blk["w2"], blk["b2"])
        xn = self._res_norm(x, y, norm2=p["final_ln"], write_x=False)
        if out_fp32:
            wave = torch.mm(xn, p["out_w"].t(), out_dtype=torch.float32)
            wave += p["out_b"]
        else:
            wave = linear(xn, p["out_w"], p["out_b"].to(self.dtype))
        return wave.view(speakers, -1)

    def decoder(self, latents: torch.Tensor) -> torch.Tensor:
        """latents [B, L, 64] fp32 -> audio [2B, 1, L*upsample*hop] fp32."""
        batch, length, _ = latents.shape
        raw = latents * self.latent_std + self.latent_mean
        d = self.latent_dim
        speakers = torch.cat([raw[..., :d], raw[..., d:]], 0)  # [2B, L, d] token-major
        if self.up is not None:
            up_w, up_b = self.up
            mixed = (speakers.reshape(-1, d) @ up_w).view(2 * batch, length, 4, d)
            even = mixed[:, :, 1].clone()
            even[:, 1:] += mixed[:, :-1, 3]
            odd = mixed[:, :, 2].clone()
            odd[:, :-1] += mixed[:, 1:, 0]
            speakers = torch.stack([even, odd], 2).view(2 * batch, 2 * length, d) + up_b
        S, T = speakers.shape[0], speakers.shape[1]
        g1, g2 = self.gens
        spk_bf = speakers.to(self.dtype)
        wave = self._trunk(g1, self._spec_proj(spk_bf, g1["spec_w"]), S, out_fp32=False)
        # iteration 2: STFT(center, reflect) of the iteration-1 waveform as one GEMM
        half = self.n_fft // 2
        padded = F.pad(wave.float().unsqueeze(1), (half, half), mode="reflect")[:, 0].to(self.dtype)
        frames = padded.unfold(-1, self.n_fft, self.hop)[:, :T].reshape(S * T, self.n_fft)
        stft = linear(frames, self.stft_w).view(S, T, self.n_fft)
        spec = self._spec_proj(torch.cat([spk_bf, stft], -1), g2["spec_w"])
        audio = self._trunk(g2, spec, S, out_fp32=True)
        return audio.view(S, 1, -1)

    def forward(self, waveform: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(waveform)
        latents = self.dit(hidden, noise)
        return self.decoder(latents)
