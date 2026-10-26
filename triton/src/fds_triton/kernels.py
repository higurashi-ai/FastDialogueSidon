"""Triton kernels for the FastDialogueSidon pipeline.

All activations are token-major ``[rows, C]`` (rows = batch * frames), so every
per-frame access is a contiguous channel row. The residual stream is fp32;
GEMM operands are fp16 or bf16 (cuBLAS, fp32 accumulate). Each kernel here fuses the
memory-bound glue between two GEMMs:

``res_norm``        x += gate * y; optional LayerNorm of x in place; LayerNorm
                    (+ adaLN modulate) of x -> fp16/bf16 GEMM operand.
``glu``             a * sigmoid(b) of the conformer pointwise-conv1 output.
``causal_dw_norm``  conformer causal depthwise conv (k=31) + LayerNorm + SiLU.
``convnext_dw_norm`` decoder ConvNeXt depthwise conv (k=7, same pad, bias) with
                    the previous block's pending residual folded into the taps
                    + LayerNorm.
``flexrel_attention`` flash attention with the w2v-BERT relative_key bias
                    generated in-kernel (Q.R^T per row tile, clamp gather).
``rope_split``      DiT q/k RoPE (fp32 tables) + q/k/v split into contiguous
                    [rows, H*D] buffers for SDPA.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

LOG2E = tl.constexpr(1.4426950408889634)


def _pow2(n: int) -> int:
    return 1 << (int(n) - 1).bit_length()


def _row_block(cp: int) -> int:
    return max(1, 4096 // cp)


def _warps(cp: int) -> int:
    return 4 if cp <= 512 else 8


@triton.jit
def _layernorm(x, w_ptr, b_ptr, c, cmask, eps, C: tl.constexpr, AFFINE: tl.constexpr):
    """Row LayerNorm of fp32 x [R, CP] (columns >= C are zero) -> fp32."""
    mean = tl.sum(x, axis=1) / C
    d = tl.where(cmask[None, :], x - mean[:, None], 0.0)
    var = tl.sum(d * d, axis=1) / C
    y = d * tl.math.rsqrt(var + eps)[:, None]
    if AFFINE:
        w = tl.load(w_ptr + c, mask=cmask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + c, mask=cmask, other=0.0).to(tl.float32)
        y = y * w[None, :] + b[None, :]
    return y


# ---------------------------------------------------------------- residual + norm


@triton.jit
def _res_norm_kernel(
    x_ptr, y_ptr, gate_ptr, gate_stride, out_x_ptr, xn_ptr, bias_ptr,
    n1_w, n1_b, n2_w, n2_b, mod_ptr, mod_stride, shift_off, scale_off,
    R, eps1, eps2,
    C: tl.constexpr, CP: tl.constexpr, BR: tl.constexpr,
    HAS_Y: tl.constexpr, HAS_GATE: tl.constexpr, NORM1: tl.constexpr,
    WRITE_X: tl.constexpr, NORM2: tl.constexpr, NORM2_AFFINE: tl.constexpr,
    MODULATE: tl.constexpr, HAS_BIAS: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64) * BR + tl.arange(0, BR)
    c = tl.arange(0, CP)
    cmask = c < C
    m = (r < R)[:, None] & cmask[None, :]
    offs = r[:, None] * C + c[None, :]
    x = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
    if HAS_BIAS:
        x += tl.load(bias_ptr + c, mask=cmask, other=0.0).to(tl.float32)[None, :]
    if HAS_Y:
        y = tl.load(y_ptr + offs, mask=m, other=0.0).to(tl.float32)
        if HAS_GATE:
            g = tl.load(gate_ptr + r[:, None] * gate_stride + c[None, :], mask=m, other=0.0)
            y = y * g.to(tl.float32)
        x = x + y
    if NORM1:
        x = _layernorm(x, n1_w, n1_b, c, cmask, eps1, C, True)
    if WRITE_X:
        tl.store(out_x_ptr + offs, x, mask=m)
    if NORM2:
        xn = _layernorm(x, n2_w, n2_b, c, cmask, eps2, C, NORM2_AFFINE)
        if MODULATE:
            mrow = mod_ptr + r[:, None] * mod_stride + c[None, :]
            shift = tl.load(mrow + shift_off, mask=m, other=0.0).to(tl.float32)
            scale = tl.load(mrow + scale_off, mask=m, other=0.0).to(tl.float32)
            xn = xn * (1.0 + scale) + shift
        tl.store(xn_ptr + offs, xn.to(xn_ptr.dtype.element_ty), mask=m)


def res_norm(
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    gate: torch.Tensor | None = None,
    norm1: tuple | None = None,
    write_x: bool = True,
    norm2: tuple | None = None,
    modulate: tuple | None = None,
    xn: torch.Tensor | None = None,
    out_x: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    xn_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor | None:
    """x [R, C] fp32 (updated in place unless out_x): x += bias + gate * y; x = LN1(x);
    returns LN2(x) [* (1 + scale) + shift] in ``xn_dtype`` (or into ``xn``).

    norm = (weight, bias, eps) or (None, None, eps) for a non-affine norm;
    gate / modulate=(shift, scale) are column slices of a 2-D tensor (row
    stride = its stride(0)), e.g. the DiT adaLN modulation output.
    """
    R, C = x.shape
    cp = _pow2(C)
    br = _row_block(cp)
    out_x = x if out_x is None else out_x
    if norm2 is not None and xn is None:
        xn = torch.empty(R, C, device=x.device, dtype=xn_dtype)
    dummy = x
    n1 = norm1 or (dummy, dummy, 0.0)
    n2 = norm2 or (dummy, dummy, 0.0)
    if modulate is not None:
        shift, scale = modulate
        base = shift if shift.data_ptr() < scale.data_ptr() else scale
        mod_ptr, mod_stride = base, base.stride(0)
        elem = base.element_size()
        shift_off = (shift.data_ptr() - base.data_ptr()) // elem
        scale_off = (scale.data_ptr() - base.data_ptr()) // elem
    else:
        mod_ptr, mod_stride, shift_off, scale_off = dummy, 0, 0, 0
    _res_norm_kernel[(triton.cdiv(R, br),)](
        x, y if y is not None else dummy,
        gate if gate is not None else dummy,
        gate.stride(0) if gate is not None else 0,
        out_x, xn if xn is not None else dummy, bias if bias is not None else dummy,
        n1[0] if n1[0] is not None else dummy, n1[1] if n1[1] is not None else dummy,
        n2[0] if n2[0] is not None else dummy, n2[1] if n2[1] is not None else dummy,
        mod_ptr, mod_stride, shift_off, scale_off,
        R, float(n1[2]), float(n2[2]),
        C=C, CP=cp, BR=br,
        HAS_Y=y is not None, HAS_GATE=gate is not None, NORM1=norm1 is not None,
        WRITE_X=write_x, NORM2=norm2 is not None,
        NORM2_AFFINE=norm2 is not None and norm2[0] is not None,
        MODULATE=modulate is not None, HAS_BIAS=bias is not None,
        num_warps=_warps(cp),
    )
    return xn


# ---------------------------------------------------------------- conformer conv module


@triton.jit
def _glu_kernel(h_ptr, g_ptr, N, C: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    row = i // C
    col = i % C
    a = tl.load(h_ptr + row * (2 * C) + col, mask=m, other=0.0).to(tl.float32)
    b = tl.load(h_ptr + row * (2 * C) + C + col, mask=m, other=0.0).to(tl.float32)
    tl.store(g_ptr + i, (a * tl.sigmoid(b)).to(g_ptr.dtype.element_ty), mask=m)


def glu(h: torch.Tensor) -> torch.Tensor:
    """h [R, 2C] fp16/bf16 -> h[:, :C] * sigmoid(h[:, C:]) (same dtype)."""
    R, C2 = h.shape
    C = C2 // 2
    g = torch.empty(R, C, device=h.device, dtype=h.dtype)
    block = 2048
    _glu_kernel[(triton.cdiv(R * C, block),)](h, g, R * C, C=C, BLOCK=block, num_warps=4)
    return g


def _causal_dw_configs():
    return [
        triton.Config({"BT": bt}, num_warps=w)
        for bt in (2, 4, 8, 16)
        for w in (4, 8)
    ]


# Fixed launch configs instead of autotuning: autotune keyed on (C, K) picked
# its config at whatever length ran first, and BT>=16 with 4 warps spills
# (3-16x slower at 120 s). BT=8/4 warps is best at long lengths; short ones
# drop to smaller BT so the grid still covers the SMs.
DW_WARPS = 4
_SMS: dict = {}


def _dw_block_t(frames: int, batch: int, device) -> int:
    key = device.index or 0
    if key not in _SMS:
        _SMS[key] = torch.cuda.get_device_properties(key).multi_processor_count
    for bt in (8, 4):
        if triton.cdiv(frames, bt) * batch >= 2 * _SMS[key]:
            return bt
    return 2


@triton.jit
def _causal_dw_norm_kernel(
    g_ptr, dwt_ptr, ln_w, ln_b, out_ptr, T, eps,
    C: tl.constexpr, K: tl.constexpr, BT: tl.constexpr,
):
    """out[t] = silu(LN(sum_k w[k] * g[t + k - (K-1)])) per batch (zero left pad)."""
    b = tl.program_id(1).to(tl.int64)
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    c = tl.arange(0, C)
    base = g_ptr + b * T * C
    acc = tl.zeros([BT, C], tl.float32)
    for k in tl.static_range(K):
        src = t + (k - (K - 1))
        valid = (src >= 0) & (src < T)
        row = tl.load(base + src[:, None] * C + c[None, :], mask=valid[:, None], other=0.0)
        acc += row.to(tl.float32) * tl.load(dwt_ptr + k * C + c)[None, :]
    y = _layernorm(acc, ln_w, ln_b, c, c < C, eps, C, True)
    y = y * tl.sigmoid(y)
    tl.store(out_ptr + b * T * C + t[:, None] * C + c[None, :], y.to(out_ptr.dtype.element_ty),
             mask=(t < T)[:, None])


def causal_dw_norm(g: torch.Tensor, batch: int, dwt: torch.Tensor, ln_w, ln_b, eps: float) -> torch.Tensor:
    """g [B*T, C] fp16/bf16, dwt [K, C] fp32 -> silu(LN(causal depthwise conv)) (same dtype).
    (A two-pass [BT, BC]-tiled variant measured 1.3x slower on H100.)"""
    R, C = g.shape
    T = R // batch
    out = torch.empty_like(g)
    bt = _dw_block_t(T, batch, g.device)
    _causal_dw_norm_kernel[(triton.cdiv(T, bt), batch)](g, dwt, ln_w, ln_b, out, T, eps, C=C,
                                                        K=dwt.shape[0], BT=bt, num_warps=DW_WARPS)
    return out


# ---------------------------------------------------------------- ConvNeXt (decoder)


def _convnext_dw_configs():
    return [
        triton.Config({"BT": bt}, num_warps=w)
        for bt in (4, 8, 16, 32)
        for w in (4, 8)
    ]


@triton.jit
def _convnext_dw_norm_kernel(
    x_ptr, y_ptr, out_x_ptr, xn_ptr, dwt_ptr, dw_b, ln_w, ln_b, T, eps,
    C: tl.constexpr, K: tl.constexpr, PENDING: tl.constexpr, BT: tl.constexpr,
):
    """Residual stream x (+ pending y) -> store x; xn = LN(dwconv_same(x) + bias)."""
    b = tl.program_id(1).to(tl.int64)
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    c = tl.arange(0, C)
    xb = x_ptr + b * T * C
    yb = y_ptr + b * T * C
    half: tl.constexpr = K // 2
    acc = tl.zeros([BT, C], tl.float32)
    for k in tl.static_range(K):
        src = t + (k - half)
        valid = ((src >= 0) & (src < T))[:, None]
        offs = src[:, None] * C + c[None, :]
        row = tl.load(xb + offs, mask=valid, other=0.0)
        if PENDING:
            row += tl.load(yb + offs, mask=valid, other=0.0).to(tl.float32)
            if k == half:
                tl.store(out_x_ptr + b * T * C + offs, row, mask=valid)
        acc += row * tl.load(dwt_ptr + k * C + c)[None, :]
    acc += tl.load(dw_b + c)[None, :]
    xn = _layernorm(acc, ln_w, ln_b, c, c < C, eps, C, True)
    tl.store(xn_ptr + b * T * C + t[:, None] * C + c[None, :], xn.to(xn_ptr.dtype.element_ty),
             mask=(t < T)[:, None])


def convnext_dw_norm(x, y, batch, dwt, dw_b, ln_w, ln_b, eps, out_x, xn_dtype=torch.bfloat16):
    """x [B*T, C] fp32 residual, y pending fp16/bf16 (or None); writes x + y to out_x
    (must not alias x: neighbouring programs still read x) and returns
    LN(dwconv(x + y)) in ``xn_dtype``."""
    R, C = x.shape
    assert y is None or out_x.data_ptr() != x.data_ptr()
    T = R // batch
    xn = torch.empty(R, C, device=x.device, dtype=xn_dtype)
    bt = _dw_block_t(T, batch, x.device)
    _convnext_dw_norm_kernel[(triton.cdiv(T, bt), batch)](
        x, y if y is not None else x, out_x, xn, dwt, dw_b, ln_w, ln_b, T, eps,
        C=C, K=dwt.shape[0], PENDING=y is not None, BT=bt, num_warps=DW_WARPS,
    )
    return xn


# ---------------------------------------------------------------- FlexRel attention

LEFT, RIGHT = 64, 8
NUM_POS = LEFT + RIGHT + 1  # 73
NUM_POS_PAD = 128


@triton.jit
def _softmax_update(s, m_i, l_i, acc, v_tile):
    """s is already in log2 units."""
    m_new = tl.maximum(m_i, tl.max(s, 1))
    alpha = tl.math.exp2(m_i - m_new)
    p = tl.math.exp2(s - m_new[:, None])
    l_i = l_i * alpha + tl.sum(p, 1)
    acc = acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile)
    return m_new, l_i, acc


def _attn_configs():
    return [
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=w, num_stages=s)
        for bm, bn, w, s in (
            (64, 64, 4, 2), (64, 64, 4, 3), (64, 64, 4, 4),
            (128, 64, 4, 3), (128, 64, 8, 3), (128, 64, 8, 4),
            (64, 128, 4, 3), (128, 128, 8, 3), (128, 32, 4, 3),
        )
    ]


@triton.autotune(configs=_attn_configs(), key=["H", "L_BUCKET"])
@triton.jit
def _flexrel_kernel(
    Q, K, V, R, O,
    stride_b, stride_m, stride_ob, stride_om,
    H, L, L_BUCKET, qk_scale,
    HEAD_DIM: tl.constexpr, NPOS: tl.constexpr, NPOS_PAD: tl.constexpr,
    CLAMP_LEFT: tl.constexpr, CLAMP_RIGHT: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """scores = (Q.K^T + Q.R[clamp(j-i)]^T) * scale; qk_scale = scale * log2(e).

    Q/K/V rows are [b, t, h*D + d] with row stride ``stride_m`` (they may be
    column slices of one fused qkv GEMM output). O is [b, t, h*D + d].
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = (pid_bh // H).to(tl.int64)
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    m_valid = offs_m < L
    head = b * stride_b + h * HEAD_DIM

    q = tl.load(Q + head + offs_m[:, None] * stride_m + offs_d[None, :],
                mask=m_valid[:, None], other=0.0)
    offs_p = tl.arange(0, NPOS_PAD)
    r = tl.load(R + offs_p[:, None] * HEAD_DIM + offs_d[None, :],
                mask=offs_p[:, None] < NPOS, other=0.0)
    qr = tl.dot(q, tl.trans(r)) * qk_scale
    qr_left = tl.sum(tl.where(offs_p[None, :] == 0, qr, 0.0), 1)
    qr_right = tl.sum(tl.where(offs_p[None, :] == NPOS - 1, qr, 0.0), 1)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    k_base = K + head
    v_base = V + head
    m_lo = pid_m * BLOCK_M
    m_hi = m_lo + BLOCK_M - 1
    # [0, n_left) pure-left clamp | [n_left, band_end) diagonal band (gather)
    # [n_right, n_full) pure-right clamp | [n_tail, L) right tail (masked)
    n_left = tl.maximum((m_lo - CLAMP_LEFT + 1) // BLOCK_N, 0) * BLOCK_N
    n_right = ((m_hi + CLAMP_RIGHT + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
    n_full = (L // BLOCK_N) * BLOCK_N
    band_end = tl.minimum(n_right, L)
    n_tail = tl.maximum(n_right, n_full)

    # The clamp-saturated regions add a per-row constant (qr_left / qr_right):
    # run them bias-free and shift the running max afterwards (exact).
    for n0 in tl.range(0, n_left, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        k_tile = tl.load(k_base + offs_n[:, None] * stride_m + offs_d[None, :])
        s = tl.dot(q, tl.trans(k_tile)) * qk_scale
        v_tile = tl.load(v_base + offs_n[:, None] * stride_m + offs_d[None, :])
        m_i, l_i, acc = _softmax_update(s, m_i, l_i, acc, v_tile)
    m_i += qr_left

    for n0 in tl.range(n_left, band_end, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        n_valid = offs_n < L
        k_tile = tl.load(k_base + offs_n[:, None] * stride_m + offs_d[None, :],
                         mask=n_valid[:, None], other=0.0)
        rel = tl.minimum(tl.maximum(offs_n[None, :] - offs_m[:, None], -CLAMP_LEFT),
                         CLAMP_RIGHT) + CLAMP_LEFT
        s = tl.dot(q, tl.trans(k_tile)) * qk_scale + tl.gather(qr, rel, axis=1)
        s = tl.where(n_valid[None, :], s, float("-inf"))
        v_tile = tl.load(v_base + offs_n[:, None] * stride_m + offs_d[None, :],
                         mask=n_valid[:, None], other=0.0)
        m_i, l_i, acc = _softmax_update(s, m_i, l_i, acc, v_tile)

    m_i -= qr_right
    for n0 in tl.range(n_right, n_full, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        k_tile = tl.load(k_base + offs_n[:, None] * stride_m + offs_d[None, :])
        s = tl.dot(q, tl.trans(k_tile)) * qk_scale
        v_tile = tl.load(v_base + offs_n[:, None] * stride_m + offs_d[None, :])
        m_i, l_i, acc = _softmax_update(s, m_i, l_i, acc, v_tile)

    for n0 in tl.range(n_tail, L, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        n_valid = offs_n < L
        k_tile = tl.load(k_base + offs_n[:, None] * stride_m + offs_d[None, :],
                         mask=n_valid[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k_tile)) * qk_scale
        s = tl.where(n_valid[None, :], s, float("-inf"))
        v_tile = tl.load(v_base + offs_n[:, None] * stride_m + offs_d[None, :],
                         mask=n_valid[:, None], other=0.0)
        m_i, l_i, acc = _softmax_update(s, m_i, l_i, acc, v_tile)

    acc = acc / l_i[:, None]
    tl.store(O + b * stride_ob + h * HEAD_DIM + offs_m[:, None] * stride_om + offs_d[None, :],
             acc.to(O.dtype.element_ty), mask=m_valid[:, None])


def flexrel_attention(qkv: torch.Tensor, batch: int, heads: int, rel: torch.Tensor,
                      out: torch.Tensor | None = None) -> torch.Tensor:
    """qkv [B*L, 3*H*D] fp16/bf16 (q | k | v column blocks), rel [73, D] (same dtype)
    -> context [B*L, H*D] (qkv dtype)."""
    rows, width = qkv.shape
    hidden = width // 3
    dim = hidden // heads
    length = rows // batch
    assert dim == 64 and rel.shape == (NUM_POS, dim) and rel.is_contiguous()
    if out is None:
        out = torch.empty(rows, hidden, device=qkv.device, dtype=qkv.dtype)
    q, k, v = qkv[:, :hidden], qkv[:, hidden:2 * hidden], qkv[:, 2 * hidden:]
    grid = lambda meta: (triton.cdiv(length, meta["BLOCK_M"]), batch * heads)  # noqa: E731
    _flexrel_kernel[grid](
        q, k, v, rel, out,
        length * width, width, length * hidden, hidden,
        heads, length, _pow2(length), LOG2E.value / math.sqrt(dim),
        HEAD_DIM=dim, NPOS=NUM_POS, NPOS_PAD=NUM_POS_PAD,
        CLAMP_LEFT=LEFT, CLAMP_RIGHT=RIGHT,
    )
    return out


# ---------------------------------------------------------------- DiT RoPE


@triton.jit
def _rope_split_kernel(qkv_ptr, cos_ptr, sin_ptr, out_ptr, NH, L, HID: tl.constexpr,
                       D: tl.constexpr, BR: tl.constexpr):
    """Rows i = (token, head): q/k rotated with fp32 tables, v copied.
    out [3, tokens, HID] (qkv dtype)."""
    i = tl.program_id(0).to(tl.int64) * BR + tl.arange(0, BR)
    tok = i // (HID // D)
    hd = (i % (HID // D)) * D
    d = tl.arange(0, D)
    half: tl.constexpr = D // 2
    partner = (d + half) % D
    sign = tl.where(d < half, -1.0, 1.0)
    m = (i < NH)[:, None]
    pos = tok % L
    cos = tl.load(cos_ptr + pos[:, None] * D + d[None, :], mask=m, other=0.0)
    sin = tl.load(sin_ptr + pos[:, None] * D + d[None, :], mask=m, other=0.0)
    src = qkv_ptr + tok[:, None] * (3 * HID) + hd[:, None]
    dst = tok[:, None] * HID + hd[:, None]
    tokens = NH // (HID // D)
    for part in tl.static_range(2):
        x = tl.load(src + part * HID + d[None, :], mask=m, other=0.0).to(tl.float32)
        xp = tl.load(src + part * HID + partner[None, :], mask=m, other=0.0).to(tl.float32)
        y = x * cos + sign[None, :] * xp * sin
        tl.store(out_ptr + part * tokens * HID + dst + d[None, :], y.to(out_ptr.dtype.element_ty), mask=m)
    v = tl.load(src + 2 * HID + d[None, :], mask=m, other=0.0)
    tl.store(out_ptr + 2 * tokens * HID + dst + d[None, :], v, mask=m)


def rope_split(qkv: torch.Tensor, length: int, heads: int, cos: torch.Tensor,
               sin: torch.Tensor) -> torch.Tensor:
    """qkv [B*L, 3*HID] fp16/bf16 -> [3, B*L, HID] (q, k rotated; v), same dtype."""
    rows, width = qkv.shape
    hid = width // 3
    dim = hid // heads
    out = torch.empty(3, rows, hid, device=qkv.device, dtype=qkv.dtype)
    nh = rows * heads
    br = 64
    _rope_split_kernel[(triton.cdiv(nh, br),)](qkv, cos, sin, out, nh, length, HID=hid, D=dim, BR=br,
                                               num_warps=4)
    return out


# ---------------------------------------------------------------- FlexRel attention, cuDNN split
#
# Outside the 71-column diagonal band the relative_key bias is a per-row
# constant (clamp saturated): qr_left[i] for j - i <= -64, qr_right[i] for
# j - i >= 8. So
#   left  region = causal attention of (Q[64:], K[:L-64], V[:L-64])
#   right region = causal attention of the time-reversed (Q[:L-8], K[8:], V[8:])
# both run by cuDNN (which also returns the log-sum-exp), and one Triton kernel
# computes the band with its gathered bias and merges all three exactly in
# log space with the row constants.


def _band_configs():
    return [
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=w, num_stages=st)
        for bm, bn, w, st in (
            (64, 64, 4, 2), (64, 32, 4, 2), (32, 32, 2, 2), (32, 64, 4, 2),
            (128, 64, 8, 2), (128, 32, 8, 2), (64, 128, 4, 2), (16, 32, 1, 2),
        )
    ]


@triton.autotune(configs=_band_configs(), key=["H", "L_BUCKET"])
@triton.jit
def _band_merge_kernel(
    Q, K, V, R, O, OL, LSEL, OR, LSER, SCR,
    stride_b, stride_m, stride_ob, stride_om,
    sol_b, sol_h, sol_m, slse_l_b, slse_l_h,
    sor_b, sor_h, sor_m, slse_r_b, slse_r_h,
    H, L, L_BUCKET, qk_scale,
    HEAD_DIM: tl.constexpr, NPOS: tl.constexpr, NPOS_PAD: tl.constexpr,
    CLAMP_LEFT: tl.constexpr, CLAMP_RIGHT: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """The Q.R^T tile is parked in per-program scratch SCR and the band bias is
    gathered with pointer loads (L1 hits): register tl.gather made this kernel
    ~1.6x slower on H100 (0.13 vs 0.08 ms at L=6000, 16 heads)."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = (pid_bh // H).to(tl.int64)
    h = pid_bh % H
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    m_valid = offs_m < L
    head = b * stride_b + h * HEAD_DIM

    q = tl.load(Q + head + offs_m[:, None] * stride_m + offs_d[None, :], mask=m_valid[:, None], other=0.0)
    offs_p = tl.arange(0, NPOS_PAD)
    r = tl.load(R + offs_p[:, None] * HEAD_DIM + offs_d[None, :], mask=offs_p[:, None] < NPOS, other=0.0)
    qr = tl.dot(q, tl.trans(r)) * qk_scale
    qr_left = tl.sum(tl.where(offs_p[None, :] == 0, qr, 0.0), 1)
    qr_right = tl.sum(tl.where(offs_p[None, :] == NPOS - 1, qr, 0.0), 1)
    local = tl.arange(0, BLOCK_M)
    scr = SCR + (pid_bh.to(tl.int64) * tl.num_programs(0) + pid_m) * (BLOCK_M * NPOS_PAD)
    tl.store(scr + local[:, None] * NPOS_PAD + offs_p[None, :], qr)
    tl.debug_barrier()

    # left part: rows i >= CLAMP_LEFT, stored at i - CLAMP_LEFT
    has_l = m_valid & (offs_m >= CLAMP_LEFT)
    il = tl.where(has_l, offs_m - CLAMP_LEFT, 0)
    lse_l = tl.load(LSEL + b * slse_l_b + h * slse_l_h + il, mask=has_l, other=0.0)
    t_l = tl.where(has_l, lse_l * LOG2E + qr_left, float("-inf"))
    # right part (time-reversed): rows i <= L-1-CLAMP_RIGHT, stored at L-1-CLAMP_RIGHT-i
    has_r = m_valid & (offs_m <= L - 1 - CLAMP_RIGHT)
    ir = tl.where(has_r, L - 1 - CLAMP_RIGHT - offs_m, 0)
    lse_r = tl.load(LSER + b * slse_r_b + h * slse_r_h + ir, mask=has_r, other=0.0)
    t_r = tl.where(has_r, lse_r * LOG2E + qr_right, float("-inf"))

    m_i = tl.maximum(t_l, t_r)
    m_safe = tl.where(m_i == float("-inf"), 0.0, m_i)
    w_l = tl.math.exp2(t_l - m_safe)
    w_r = tl.math.exp2(t_r - m_safe)
    l_i = w_l + w_r
    o_l = tl.load(OL + b * sol_b + h * sol_h + il[:, None] * sol_m + offs_d[None, :],
                  mask=has_l[:, None], other=0.0).to(tl.float32)
    o_r = tl.load(OR + b * sor_b + h * sor_h + ir[:, None] * sor_m + offs_d[None, :],
                  mask=has_r[:, None], other=0.0).to(tl.float32)
    acc = w_l[:, None] * o_l + w_r[:, None] * o_r

    # diagonal band: j - i in (-CLAMP_LEFT, CLAMP_RIGHT)
    m_lo = pid_m * BLOCK_M
    n_lo = tl.maximum(m_lo - CLAMP_LEFT + 1, 0)
    n_lo = (n_lo // BLOCK_N) * BLOCK_N
    n_hi = tl.minimum(m_lo + BLOCK_M + CLAMP_RIGHT - 1, L)
    k_base = K + head
    v_base = V + head
    for n0 in tl.range(n_lo, n_hi, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        n_valid = offs_n < L
        k_tile = tl.load(k_base + offs_n[:, None] * stride_m + offs_d[None, :], mask=n_valid[:, None], other=0.0)
        diff = offs_n[None, :] - offs_m[:, None]
        in_band = (diff > -CLAMP_LEFT) & (diff < CLAMP_RIGHT) & n_valid[None, :]
        rel = tl.minimum(tl.maximum(diff, -CLAMP_LEFT), CLAMP_RIGHT) + CLAMP_LEFT
        bias = tl.load(scr + local[:, None] * NPOS_PAD + rel, mask=in_band, other=0.0)
        s = tl.dot(q, tl.trans(k_tile)) * qk_scale + bias
        s = tl.where(in_band, s, float("-inf"))
        v_tile = tl.load(v_base + offs_n[:, None] * stride_m + offs_d[None, :], mask=n_valid[:, None], other=0.0)
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(O + b * stride_ob + h * HEAD_DIM + offs_m[:, None] * stride_om + offs_d[None, :],
             acc.to(O.dtype.element_ty), mask=m_valid[:, None])


@triton.jit
def _reverse_rows_kernel(src_ptr, dst_ptr, T, src_off, seq, row_stride, C: tl.constexpr, BR: tl.constexpr,
                         BC: tl.constexpr):
    """dst[part, b, t, :] = src[b, src_off + T - 1 - t, part*C : (part+1)*C] (q|k|v blocks)."""
    t = tl.program_id(0) * BR + tl.arange(0, BR)
    part = tl.program_id(1)
    b = tl.program_id(2).to(tl.int64)
    nb = tl.num_programs(2)
    m = (t < T)[:, None]
    src_t = src_off + T - 1 - t
    for c0 in tl.static_range(0, C, BC):
        c = c0 + tl.arange(0, BC)
        x = tl.load(src_ptr + (b * seq + src_t)[:, None] * row_stride + part * C + c[None, :], mask=m)
        tl.store(dst_ptr + ((part * nb + b) * T + t)[:, None] * C + c[None, :], x, mask=m)


def _reverse_qkv(qkv: torch.Tensor, batch: int, length: int, hidden: int):
    """Time-reversed q[:L-8], k[8:], v[8:] -> [3, B, L-8, hidden]."""
    T = length - RIGHT
    out = torch.empty(3, batch, T, hidden, device=qkv.device, dtype=qkv.dtype)
    br = 8
    # q reads rows [0, T); k and v read rows [RIGHT, L)
    for part, off in ((0, 0), (1, RIGHT)):
        parts = 1 if part == 0 else 2
        _reverse_rows_kernel[(triton.cdiv(T, br), parts, batch)](
            qkv[:, part * hidden:], out[part:part + parts], T, off, length, qkv.shape[1],
            C=hidden, BR=br, BC=min(hidden, 1024), num_warps=4)
    return out


def _cudnn_causal(q, k, v, scale):
    out = torch.ops.aten._scaled_dot_product_cudnn_attention(q, k, v, None, True, 0.0, True, False, scale=scale)
    return out[0], out[1]


def flexrel_attention_split(qkv: torch.Tensor, batch: int, heads: int, rel: torch.Tensor,
                            out: torch.Tensor | None = None) -> torch.Tensor:
    """Same contract as :func:`flexrel_attention`; bulk via cuDNN (see above)."""
    rows, width = qkv.shape
    hidden = width // 3
    dim = hidden // heads
    length = rows // batch
    assert dim == 64 and rel.shape == (NUM_POS, dim) and length > LEFT + RIGHT
    if out is None:
        out = torch.empty(rows, hidden, device=qkv.device, dtype=qkv.dtype)
    view = qkv.view(batch, length, 3, heads, dim)
    q, k, v = (view[:, :, i].transpose(1, 2) for i in range(3))  # [B, H, L, D] strided
    scale = 1.0 / math.sqrt(dim)
    o_l, lse_l = _cudnn_causal(q[:, :, LEFT:], k[:, :, : length - LEFT], v[:, :, : length - LEFT], scale)
    rev = _reverse_qkv(qkv, batch, length, hidden).view(3, batch, length - RIGHT, heads, dim).transpose(2, 3)
    o_r, lse_r = _cudnn_causal(rev[0], rev[1], rev[2], scale)
    lse_l = lse_l.float().reshape(batch, heads, -1).contiguous()
    lse_r = lse_r.float().reshape(batch, heads, -1).contiguous()
    # one [BLOCK_M, NPOS_PAD] fp32 scratch tile per program (any BLOCK_M <= 128)
    scratch = torch.empty((length + 128) * batch * heads * NUM_POS_PAD, device=qkv.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(length, meta["BLOCK_M"]), batch * heads)  # noqa: E731
    _band_merge_kernel[grid](
        qkv[:, :hidden], qkv[:, hidden:2 * hidden], qkv[:, 2 * hidden:], rel, out,
        o_l, lse_l, o_r, lse_r, scratch,
        length * width, width, length * hidden, hidden,
        o_l.stride(0), o_l.stride(1), o_l.stride(2), lse_l.stride(0), lse_l.stride(1),
        o_r.stride(0), o_r.stride(1), o_r.stride(2), lse_r.stride(0), lse_r.stride(1),
        heads, length, _pow2(length), LOG2E.value / math.sqrt(dim),
        HEAD_DIM=dim, NPOS=NUM_POS, NPOS_PAD=NUM_POS_PAD, CLAMP_LEFT=LEFT, CLAMP_RIGHT=RIGHT,
    )
    return out


def cudnn_attention_available(device=None) -> bool:
    """cuDNN SDPA (sm80+ with a cuDNN build) is what the split path needs."""
    if not torch.backends.cudnn.is_available():
        return False
    return torch.cuda.get_device_capability(device) >= (8, 0)


def encoder_attention(qkv, batch, heads, rel, impl: str = "split"):
    if impl == "split":
        return flexrel_attention_split(qkv, batch, heads, rel)
    return flexrel_attention(qkv, batch, heads, rel)
