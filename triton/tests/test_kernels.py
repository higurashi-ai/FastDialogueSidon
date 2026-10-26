"""GPU unit tests: every Triton kernel vs a plain torch fp32 implementation."""

import math

import torch
import torch.nn.functional as F

from fds_triton import kernels as K

try:
    import pytest
except ImportError:  # plain-python runner (see bottom): minimal parametrize shim
    class _Mark:
        @staticmethod
        def parametrize(names, values):
            def wrap(fn):
                fn._params = (names, values)
                return fn
            return wrap

    class pytest:  # noqa: N801
        mark = _Mark()

DEV = "cuda"


def snr(ref, act):
    ref, act = ref.float(), act.float()
    return (20 * torch.log10(ref.norm() / (ref - act).norm().clamp_min(1e-30))).item()


@pytest.mark.parametrize("C", [160, 512, 768, 1024])
def test_res_norm_full(C):
    torch.manual_seed(0)
    R = 333
    x = torch.randn(R, C, device=DEV)
    y = torch.randn(R, C, device=DEV).bfloat16()
    mod = torch.randn(R, 6 * C, device=DEV).bfloat16()
    w1, b1 = torch.randn(C, device=DEV), torch.randn(C, device=DEV)
    w2, b2 = torch.randn(C, device=DEV), torch.randn(C, device=DEV)
    ref_x = F.layer_norm(x + y.float() * mod[:, 2 * C:3 * C].float(), (C,), w1, b1, 1e-5)
    ref = F.layer_norm(ref_x, (C,), w2, b2, 1e-6)
    ref = ref * (1 + mod[:, C:2 * C].float()) + mod[:, :C].float()
    xx = x.clone()
    out = K.res_norm(xx, y, gate=mod[:, 2 * C:3 * C], norm1=(w1, b1, 1e-5), norm2=(w2, b2, 1e-6),
                     modulate=(mod[:, :C], mod[:, C:2 * C]))
    assert snr(ref_x, xx) > 100
    assert snr(ref, out) > 45  # bf16 output


def test_res_norm_plain():
    torch.manual_seed(0)
    R, C = 100, 768
    x = torch.randn(R, C, device=DEV)
    out = K.res_norm(x.clone(), norm2=(None, None, 1e-6), write_x=False)
    assert snr(F.layer_norm(x, (C,), eps=1e-6), out) > 45


def test_glu():
    h = torch.randn(257, 2048, device=DEV).bfloat16()
    ref = F.glu(h.float(), dim=-1)
    assert snr(ref, K.glu(h)) > 45


@pytest.mark.parametrize("B,T", [(1, 300), (2, 77)])
def test_causal_dw_norm(B, T):
    torch.manual_seed(0)
    C, KS = 1024, 31
    g = torch.randn(B * T, C, device=DEV).bfloat16()
    w = torch.randn(C, 1, KS, device=DEV) * 0.2
    lw, lb = torch.randn(C, device=DEV), torch.randn(C, device=DEV)
    inp = F.pad(g.float().view(B, T, C).transpose(1, 2), (KS - 1, 0))
    ref = F.conv1d(inp, w, groups=C).transpose(1, 2)
    ref = F.silu(F.layer_norm(ref, (C,), lw, lb, 1e-5)).reshape(B * T, C)
    out = K.causal_dw_norm(g, B, w[:, 0].t().contiguous(), lw, lb, 1e-5)
    assert snr(ref, out) > 45


@pytest.mark.parametrize("pending", [False, True])
def test_convnext_dw_norm(pending):
    torch.manual_seed(0)
    B, T, C, KS = 2, 203, 512, 7
    x = torch.randn(B * T, C, device=DEV)
    y = torch.randn(B * T, C, device=DEV).bfloat16() if pending else None
    w = torch.randn(C, 1, KS, device=DEV) * 0.3
    wb = torch.randn(C, device=DEV)
    lw, lb = torch.randn(C, device=DEV), torch.randn(C, device=DEV)
    xs = x + y.float() if pending else x
    ref = F.conv1d(xs.view(B, T, C).transpose(1, 2), w, wb, padding=KS // 2, groups=C).transpose(1, 2)
    ref = F.layer_norm(ref, (C,), lw, lb, 1e-6).reshape(B * T, C)
    out_x = torch.empty_like(x) if pending else x
    xn = K.convnext_dw_norm(x, y, B, w[:, 0].t().contiguous(), wb, lw, lb, 1e-6, out_x=out_x)
    assert snr(ref, xn) > 45
    if pending:
        assert snr(xs, out_x) > 100


def _attn_ref(qkv, B, H, rel):
    rows, width = qkv.shape
    hid = width // 3
    D = hid // H
    L = rows // B
    q, k, v = (qkv[:, i * hid:(i + 1) * hid].float().view(B, L, H, D).transpose(1, 2) for i in range(3))
    pos = torch.arange(L, device=qkv.device)
    idx = torch.clamp(pos[None, :] - pos[:, None], -K.LEFT, K.RIGHT) + K.LEFT
    qr = torch.einsum("bhld,rd->bhlr", q, rel.float())
    bias = torch.gather(qr, 3, idx.view(1, 1, L, L).expand(B, H, -1, -1))
    s = (q @ k.transpose(-1, -2) + bias) / math.sqrt(D)
    return (s.softmax(-1) @ v).transpose(1, 2).reshape(rows, hid)


@pytest.mark.parametrize("B,L", [(1, 500), (2, 333), (1, 160)])
def test_flexrel_attention(B, L):
    torch.manual_seed(0)
    H, D = 16, 64
    qkv = torch.randn(B * L, 3 * H * D, device=DEV).bfloat16()
    rel = torch.randn(K.NUM_POS, D, device=DEV).bfloat16()
    out = K.flexrel_attention(qkv, B, H, rel)
    assert snr(_attn_ref(qkv, B, H, rel), out) > 40


@pytest.mark.parametrize("B,L", [(1, 500), (2, 333), (1, 160)])
def test_flexrel_attention_split(B, L):
    torch.manual_seed(0)
    H, D = 16, 64
    qkv = torch.randn(B * L, 3 * H * D, device=DEV).bfloat16()
    rel = torch.randn(K.NUM_POS, D, device=DEV).bfloat16()
    out = K.flexrel_attention_split(qkv, B, H, rel)
    assert snr(_attn_ref(qkv, B, H, rel), out) > 40


def test_rope_split():
    torch.manual_seed(0)
    B, L, H, D = 2, 97, 12, 64
    hid = H * D
    qkv = torch.randn(B * L, 3 * hid, device=DEV).bfloat16()
    half = D // 2
    inv = 1.0 / (10000 ** (torch.arange(0, half, dtype=torch.float32) / half))
    fr = torch.outer(torch.arange(200, dtype=torch.float32), inv)
    cos = torch.cat([fr.cos(), fr.cos()], -1).to(DEV)
    sin = torch.cat([fr.sin(), fr.sin()], -1).to(DEV)
    out = K.rope_split(qkv, L, H, cos, sin)
    c = cos[:L].view(1, L, 1, D)
    s = sin[:L].view(1, L, 1, D)
    for part in range(2):
        x = qkv[:, part * hid:(part + 1) * hid].float().view(B, L, H, D)
        rot = torch.cat([-x[..., half:], x[..., :half]], -1)
        ref = (x * c + rot * s).reshape(B * L, hid)
        assert snr(ref, out[part]) > 45
    assert torch.equal(out[2], qkv[:, 2 * hid:])


if __name__ == "__main__":
    import sys
    failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        names, values = getattr(fn, "_params", ("", [()]))
        keys = [k.strip() for k in names.split(",")] if names else []
        for value in values:
            value = value if isinstance(value, tuple) else (value,)
            label = f"{name}{list(value) if keys else ''}"
            try:
                fn(**dict(zip(keys, value)))
                print(f"PASS {label}", flush=True)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"FAIL {label}: {exc!r}", flush=True)
    print("TESTS_OK" if not failed else f"TESTS_FAILED {failed}")
    sys.exit(1 if failed else 0)
