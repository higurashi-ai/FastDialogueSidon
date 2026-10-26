#!/usr/bin/env python3
"""Separate dialogue audio into two speaker wavs with TensorRT.

Single file:   run.py --engine-dir E --wav in.wav --output-dir out/
               (any sample rate / channels, any length <= 120 s)
Streaming:     run.py --engine-dir E --wav-list files.txt --output-dir out/
               (each listed file must be >= 120 s of 16 kHz mono; the first
               120 s window is processed — chunking is up to the caller)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastdialoguesidon.runtime import (  # noqa: E402
    OUT_SAMPLES,
    SAMPLES,
    TRTPipeline,
    load_waveform,
    read_wav_int16,
    stream_files,
    write_wav_int16,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--wav", type=Path)
    parser.add_argument(
        "--wav-list", type=Path,
        help="text file with one wav path per line (streaming mode)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--bucket-seconds", type=int, default=10,
        help="windows are padded up to this bucket granularity for "
        "efficiency (default 10 s; 0 disables bucketing and CUDA graphs "
        "for exact-length execution)",
    )
    args = parser.parse_args()
    if not args.wav and not args.wav_list:
        parser.error("one of --wav / --wav-list is required")

    bucket = args.bucket_seconds if args.bucket_seconds > 0 else None
    pipeline = TRTPipeline(
        args.engine_dir,
        use_graph=bucket is not None,
        bucket_seconds=bucket,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.wav_list:
        paths = [
            Path(line.strip())
            for line in args.wav_list.read_text().splitlines()
            if line.strip()
        ]
        per_window = stream_files(pipeline, paths, args.output_dir)
        print(
            f"streamed {len(paths)} windows: {per_window:.1f} ms/window "
            f"(rtf {per_window / 1000 / 120:.2e})"
        )
        return

    # any sample rate / channel count -> the ENGINE's input rate (in-engine
    # resample when the engine was built with --input-rate != 16000)
    waveform = load_waveform(args.wav, target_rate=pipeline.input_rate)
    audio_seconds = waveform.numel() / pipeline.input_rate
    started = time.perf_counter()
    audio = pipeline.separate(waveform)  # variable length up to 120 s
    elapsed = (time.perf_counter() - started) * 1e3
    clipped = (audio.clamp(-1, 1) * 32767).to(torch.int16).cpu().numpy()
    for spk in (0, 1):
        out = args.output_dir / f"{args.wav.stem}_spk{spk}.wav"
        write_wav_int16(out, clipped[spk])
        print(f"wrote {out}")
    print(f"separated {audio_seconds:.1f} s in {elapsed:.1f} ms")


if __name__ == "__main__":
    main()
