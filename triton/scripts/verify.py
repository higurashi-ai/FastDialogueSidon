#!/usr/bin/env python3
"""Accuracy check vs the fp32 torch reference (and optionally the TRT engines).

For each window length: stage-isolated errors (each Triton stage fed the fp32
reference input) and chained errors (the full Triton chain), as SNR in dB
against the fp32 reference with identical noise. With --trt-engine-dir the
FastDialogueSidon TensorRT chain is run on the same waveform + noise and
scored the same way (needs --trt-repo on the path).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fds_triton.model import TritonModel  # noqa: E402
from fds_triton.reference.fbank_export import SeamlessFbankFrontend  # noqa: E402
from fds_triton.reference.modeling import load_bundle  # noqa: E402
from fds_triton.runtime import OUT_PER_FRAME, SAMPLES_PER_FRAME, TritonPipeline, load_waveform  # noqa: E402


def errors(ref: torch.Tensor, act: torch.Tensor) -> dict:
    ref, act = ref.float().reshape(-1), act.float().reshape(-1)
    noise = (ref - act).norm()
    return {
        "snr_db": round((20 * torch.log10(ref.norm() / noise.clamp_min(1e-30))).item(), 2),
        "max_abs": round((ref - act).abs().max().item(), 4),
    }


@torch.no_grad()
def reference_chain(bundle, front, waveform, noise):
    torch.set_float32_matmul_precision("highest")
    features = front(waveform)
    conditioning = bundle.encoder(features, None).float()
    latents = bundle.ctm(conditioning, noise).float()
    audio = bundle.decoder(latents).float()
    return conditioning, latents, audio


def conditioning_from_hidden(bundle, hidden):
    h = hidden.float().view(1, -1, hidden.shape[-1])
    enc = bundle.encoder
    return torch.cat([enc.speaker_1(h), enc.speaker_2(h), h], -1)


def runtime_checks(bundle_dir, model, wav, noise_full) -> dict:
    """CUDA-graph replay vs eager, batched vs single, bucketed separate()."""
    pipe = TritonPipeline(bundle_dir, model=model, max_batch=2)
    seq = 750  # 15 s
    samples = seq * SAMPLES_PER_FRAME
    out_n = seq * OUT_PER_FRAME
    windows = [wav[:samples], wav[samples:2 * samples]]
    noises = [noise_full[:, :seq], noise_full[:, seq:2 * seq]]
    with torch.inference_mode():
        eager = [model.forward(w.view(1, -1), n.contiguous()) for w, n in zip(windows, noises)]
    result = {}
    for batch in (1, 2):
        with torch.cuda.stream(pipe.compute):
            pipe.wave_buffer.view(-1)[: batch * samples].copy_(torch.cat(windows[:batch]))
            pipe.noise_buffer.view(-1)[: batch * seq * 64].copy_(torch.cat(noises[:batch], 0).reshape(-1))
            pipe.run_chain(seq, batch, fresh_noise=False)  # eager warm-up + capture + replay
            pipe.run_chain(seq, batch, fresh_noise=False)  # replay
        torch.cuda.synchronize()
        audio = pipe.audio_buffer.view(-1)[: 2 * batch * out_n].view(2, batch, out_n)
        for index in range(batch):
            result[f"graph_b{batch}_win{index}"] = errors(eager[index].view(2, out_n), audio[:, index])
    # stage-level batch consistency (B=2 vs B=1 with identical stage inputs)
    with torch.inference_mode():
        wave2 = torch.stack(windows)
        hid2 = model.encoder(wave2).view(2, seq, -1)
        hid1 = [model.encoder(w.view(1, -1)) for w in windows]
        noise2 = torch.cat(noises, 0).contiguous()
        lat2 = model.dit(torch.cat(hid1, 0), noise2)
        lat1 = [model.dit(h, n.contiguous()) for h, n in zip(hid1, noises)]
        aud2 = model.decoder(torch.cat(lat1, 0)).view(2, 2, out_n)
        aud1 = [model.decoder(lt).view(2, out_n) for lt in lat1]
        for index in range(2):
            result[f"batch2_stage_win{index}"] = {
                "encoder": errors(hid1[index], hid2[index]),
                "dit": errors(lat1[index], lat2[index]),
                "decoder": errors(aud1[index], aud2[:, index]),
            }
    long_wave = wav[: int(37.3 * 16000)]
    separated = pipe.separate(long_wave)
    result["separate_37.3s"] = {
        "shape": list(separated.shape),
        "expected_samples": long_wave.numel() * 3 // 2,
        "finite": bool(torch.isfinite(separated).all()),
        "rms": round(separated.square().mean().sqrt().item(), 4),
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--wav", type=Path, default=None)
    ap.add_argument("--seconds", default="15,60,120")
    ap.add_argument("--trt-engine-dir", type=Path, default=None)
    ap.add_argument("--trt-repo", type=Path, default=ROOT.parent / "trt")
    ap.add_argument("--gelu-erf", action="store_true", help="exact erf GELU instead of the tanh epilogue")
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument("--attn-impl", choices=["auto", "split", "fused"], default=None)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = torch.device("cuda")
    bundle = load_bundle(args.bundle, device="cuda")
    model = TritonModel(bundle, gelu_tanh=not args.gelu_erf, attn_impl=args.attn_impl,
                        dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype])
    front = SeamlessFbankFrontend().to(dev).eval()
    lengths = [int(s) for s in args.seconds.split(",")]
    total = max(lengths) * 16000
    if args.wav is not None:
        wav = load_waveform(args.wav)
        reps = -(-total // wav.numel())
        wav = wav.repeat(reps)[:total]
    else:
        wav = torch.randn(total, device=dev) * 0.1
    noise_full = torch.randn(1, max(lengths) * 50, 64, device=dev)

    trt = None
    if args.trt_engine_dir is not None:
        sys.path.insert(0, str(args.trt_repo / "src"))
        from fastdialoguesidon.runtime import TRTPipeline

        trt = TRTPipeline(args.trt_engine_dir, use_graph=False, bucket_seconds=None)

    report = {"gpu": torch.cuda.get_device_name(0), "wav": str(args.wav), "gelu_tanh": not args.gelu_erf,
              "attn_impl": model.attn_impl, "dtype": args.dtype,
              "lengths": {}}
    for seconds in lengths:
        seq = seconds * 50
        waveform = wav[: seq * SAMPLES_PER_FRAME].view(1, -1).contiguous()
        noise = noise_full[:, :seq].contiguous()
        ref_c, ref_l, ref_a = reference_chain(bundle, front, waveform, noise)
        entry = {}
        with torch.inference_mode():
            hidden = model.encoder(waveform)
            entry["isolated"] = {
                "conditioning": errors(ref_c, conditioning_from_hidden(bundle, hidden)),
                "latents": errors(ref_l, model.dit(ref_c[0, :, 64:].to(model.dtype), noise)),
                "audio": errors(ref_a, model.decoder(ref_l)),
            }
            lat = model.dit(hidden, noise)
            audio = model.decoder(lat)
            entry["chained"] = {
                "conditioning": entry["isolated"]["conditioning"],
                "latents": errors(ref_l, lat),
                "audio": errors(ref_a, audio),
            }
        if trt is not None:
            with torch.cuda.stream(trt.compute):
                trt._set_length(seq)
                trt.wave_buffer.view(-1)[: waveform.numel()].copy_(waveform.view(-1))
                trt.noise_buffer.view(-1)[: noise.numel()].copy_(noise.view(-1))
                trt._enqueue_chain()
            torch.cuda.synchronize()
            n = seq
            entry["trt_chained"] = {
                "conditioning": errors(ref_c, trt.conditioning_buffer.view(-1)[: n * 1088]),
                "latents": errors(ref_l, trt.latent_buffer.view(-1)[: n * 64]),
                "audio": errors(ref_a, trt.audio_buffer.view(-1)[: 2 * n * 480]),
            }
        report["lengths"][f"{seconds}s"] = entry
        print(f"{seconds}s: {json.dumps(entry)}", flush=True)
    report["runtime"] = runtime_checks(args.bundle, model, wav, noise_full)
    print(f"runtime: {json.dumps(report['runtime'])}", flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=1))
    print("VERIFY_OK")


if __name__ == "__main__":
    main()
