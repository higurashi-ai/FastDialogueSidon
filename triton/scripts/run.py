#!/usr/bin/env python3
"""Separate dialogue wav files into two 24 kHz speaker tracks (Triton pipeline).

Each input (any rate / channels, <= 120 s) -> <stem>_spk0.wav, <stem>_spk1.wav.
Inputs longer than 120 s must be chunked by the caller.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fds_triton.runtime import TritonPipeline, load_waveform, write_wav_int16  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, required=True, help="HF bundle dir (higurashi-ai/FastDialogueSidon)")
    ap.add_argument("--wav", type=Path, nargs="*", default=[])
    ap.add_argument("--input-dir", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--bucket-seconds", type=int, default=10, help="0 = exact-length graphs")
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    args = ap.parse_args()

    paths = list(args.wav)
    if args.input_dir is not None:
        paths += sorted(args.input_dir.glob("*.wav"))
    if not paths:
        ap.error("no inputs (--wav / --input-dir)")
    pipe = TritonPipeline(args.bundle, bucket_seconds=args.bucket_seconds or None,
                          dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    total_seconds = 0.0
    for path in paths:
        waveform = load_waveform(path)
        total_seconds += waveform.numel() / 16000
        audio = pipe.separate(waveform)
        pcm = (audio.clamp(-1, 1) * 32767).to(torch.int16).cpu().numpy()
        for spk in (0, 1):
            write_wav_int16(args.output_dir / f"{path.stem}_spk{spk}.wav", pcm[spk])
        print(f"{path.name}: {waveform.numel() / 16000:.1f} s", flush=True)
    wall = time.perf_counter() - started
    print(f"done: {len(paths)} files, {total_seconds:.1f} s audio in {wall:.2f} s wall "
          f"(incl. first-call autotune/graph capture)")


if __name__ == "__main__":
    main()
