#!/usr/bin/env python3
"""Separation failure rate over CTM noise seeds: Triton fp16/bf16, TRT builds, fp32.

For every sample (input/mixture_120s.wav + reference_speaker{1,2}_120s.wav) and
seed, all systems get the SAME noise draw, so differences are implementation /
precision only. Score per 10 s segment: assignment margin
  m = (corr(p1,r1) + corr(p2,r2) - corr(p1,r2) - corr(p2,r1)) / 2
on 20 ms log-energy envelopes, after the global speaker order is aligned
(+2 = perfectly separated, 0 = both outputs equally like either speaker).
A run FAILS when any segment has m < --fail-margin. Failed runs are saved as
stereo wavs (L = speaker 1, R = speaker 2) for listening.
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fds_triton.model import TritonModel  # noqa: E402
from fds_triton.reference.fbank_export import SeamlessFbankFrontend  # noqa: E402
from fds_triton.reference.modeling import load_bundle  # noqa: E402
from fds_triton.runtime import SEQ, load_waveform  # noqa: E402

SEG = 500  # 10 s of 20 ms frames


def read_mono(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as h:
        return np.frombuffer(h.readframes(h.getnframes()), np.int16).astype(np.float32) / 32768.0


def envelope(x: np.ndarray, rate: int) -> np.ndarray:
    hop = rate // 50
    n = len(x) // hop
    return np.log(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-8)


def margins(pred: np.ndarray, env_ref) -> tuple[list[float], bool]:
    p = [envelope(c, 24_000) for c in pred]
    n = min(len(p[0]), len(env_ref[0]))
    c = lambda a, b, s=slice(0, n): float(np.corrcoef(a[s], b[s])[0, 1])  # noqa: E731
    swapped = c(p[0], env_ref[1]) + c(p[1], env_ref[0]) > c(p[0], env_ref[0]) + c(p[1], env_ref[1])
    if swapped:
        p = p[::-1]
    out = []
    for a in range(0, n - SEG + 1, SEG):
        s = slice(a, a + SEG)
        out.append((c(p[0], env_ref[0], s) + c(p[1], env_ref[1], s)
                    - c(p[0], env_ref[1], s) - c(p[1], env_ref[0], s)) / 2)
    return out, swapped


def write_stereo(path: Path, pred: np.ndarray, swapped: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pred = pred[::-1] if swapped else pred
    pcm = (np.clip(pred, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as h:
        h.setnchannels(2)
        h.setsampwidth(2)
        h.setframerate(24_000)
        h.writeframes(pcm.T.copy().tobytes())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-set", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--trt-engine-dir", type=Path, default=None, help="TRT bf16 engines")
    ap.add_argument("--trt-fp16-engine-dir", type=Path, default=None)
    ap.add_argument("--trt-mixed-engine-dir", type=Path, default=None, help="TRT default: fp16 enc+DiT, bf16 decoder")
    ap.add_argument("--trt-repo", type=Path, default=ROOT.parent / "trt")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--fail-margin", type=float, default=0.5)
    ap.add_argument("--no-fp32", action="store_true")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    dev = torch.device("cuda")
    bundle = load_bundle(args.bundle, device="cuda")
    models = {
        "triton_bf16": TritonModel(bundle, max_seq=SEQ, dtype=torch.bfloat16),
        "triton_fp16": TritonModel(bundle, max_seq=SEQ, dtype=torch.float16),
    }
    front = SeamlessFbankFrontend().to(dev).eval()
    trts = {}
    if any(d is not None for d in (args.trt_engine_dir, args.trt_fp16_engine_dir, args.trt_mixed_engine_dir)):
        sys.path.insert(0, str(args.trt_repo / "src"))
        from fastdialoguesidon.runtime import TRTPipeline

        for name, d in (("trt_bf16", args.trt_engine_dir), ("trt_fp16", args.trt_fp16_engine_dir),
                        ("trt_default", args.trt_mixed_engine_dir)):
            if d is not None:
                trts[name] = TRTPipeline(d, use_graph=False, bucket_seconds=None, check_output=name != "trt_fp16")

    samples = sorted(p for p in args.source_set.iterdir() if p.is_dir() and (p / "input").is_dir())
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for sample in samples:
        env_ref = [envelope(read_mono(sample / "input" / f"reference_speaker{i}_120s.wav"), 16_000) for i in (1, 2)]
        waveform = load_waveform(sample / "input" / "mixture_120s.wav")
        seq = waveform.numel() // 320
        wave_in = waveform[: seq * 320].view(1, -1).contiguous()
        with torch.inference_mode():
            hidden = {k: m.encoder(wave_in) for k, m in models.items()}
        cond32 = None
        if not args.no_fp32:
            torch.set_float32_matmul_precision("highest")
            with torch.no_grad():
                cond32 = bundle.encoder(front(wave_in), None).float()
        for seed in range(args.seeds):
            gen = torch.Generator(device="cuda").manual_seed(seed)
            noise = torch.randn(1, seq, 64, device="cuda", generator=gen)
            preds = {}
            with torch.inference_mode():
                for k, m in models.items():
                    preds[k] = m.decoder(m.dit(hidden[k], noise)).view(2, -1).float().cpu().numpy()
            for name, trt in trts.items():
                with torch.cuda.stream(trt.compute):
                    trt._set_length(seq)
                    trt.wave_buffer.view(-1)[: wave_in.numel()].copy_(wave_in.view(-1))
                    trt.noise_buffer.view(-1)[: noise.numel()].copy_(noise.view(-1))
                    trt._enqueue_chain()
                torch.cuda.synchronize()
                preds[name] = trt.audio_buffer.view(-1)[: 2 * seq * 480].view(2, -1).float().cpu().numpy()
            if cond32 is not None:
                with torch.no_grad():
                    preds["fp32_torch"] = bundle.decoder(bundle.ctm(cond32, noise).float()).view(2, -1).cpu().numpy()
            for system, pred in preds.items():
                nonfinite = float((~np.isfinite(pred)).mean())
                pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
                m, swapped = margins(pred, env_ref)
                m = [x if np.isfinite(x) else -2.0 for x in m]
                failed = min(m) < args.fail_margin or nonfinite > 0
                run = {"sample": sample.name, "seed": seed, "system": system, "min_margin": round(min(m), 3),
                       "mean_margin": round(float(np.mean(m)), 3), "failed": failed,
                       "bad_segments": sum(x < args.fail_margin for x in m), "nonfinite": nonfinite,
                       "margins": [round(x, 3) for x in m]}
                if failed:
                    path = out / "failed" / sample.name / f"{system}_seed{seed:02d}.wav"
                    write_stereo(path, pred, swapped)
                    run["wav"] = str(path)
                runs.append(run)
            line = " ".join(f"{r['system']}={r['min_margin']:+.2f}{'*' if r['failed'] else ''}"
                            for r in runs[-len(preds):])
            print(f"{sample.name} seed {seed:02d}: {line}", flush=True)

    systems = sorted({r["system"] for r in runs})
    summary = {"fail_margin": args.fail_margin, "seeds": args.seeds, "per_system": {}, "per_sample": {}}
    for system in systems:
        rs = [r for r in runs if r["system"] == system]
        summary["per_system"][system] = {
            "runs": len(rs), "failed": sum(r["failed"] for r in rs),
            "fail_rate": round(sum(r["failed"] for r in rs) / len(rs), 3),
            "mean_margin": round(float(np.mean([r["mean_margin"] for r in rs])), 3),
            "worst_min_margin": min(r["min_margin"] for r in rs),
        }
        for sample in samples:
            ss = [r for r in rs if r["sample"] == sample.name]
            summary["per_sample"].setdefault(sample.name, {})[system] = {
                "failed": sum(r["failed"] for r in ss), "runs": len(ss),
                "min_margin": min(r["min_margin"] for r in ss),
                "mean_margin": round(float(np.mean([r["mean_margin"] for r in ss])), 3)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "runs.json").write_text(json.dumps(runs, indent=1))
    print(json.dumps(summary["per_system"], indent=1))
    for name, d in summary["per_sample"].items():
        print(name, " ".join(f"{k}:{v['failed']}/{v['runs']}(min{v['min_margin']:+.2f})" for k, v in d.items()))
    print("SEED_SWEEP_OK")


if __name__ == "__main__":
    main()
