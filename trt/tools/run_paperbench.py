#!/usr/bin/env python3
"""Run the FastDialogueSidon TensorRT runtime on a paper-benchmark manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastdialoguesidon.runtime import (  # noqa: E402
    TRTPipeline,
    load_waveform,
    write_wav_int16,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_stereo(path: Path, channels: np.ndarray, rate: int = 24_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(channels.T.copy().tobytes())


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def load_input(path: Path, target_rate: int) -> torch.Tensor:
    """Use the fast PCM reader, with torchaudio for non-RIFF containers."""
    try:
        return load_waveform(path, target_rate=target_rate)
    except (wave.Error, EOFError):
        import torchaudio

        waveform, rate = torchaudio.load(path)
        waveform = waveform.float().mean(dim=0, keepdim=True)
        if rate != target_rate:
            waveform = torchaudio.functional.resample(
                waveform, rate, target_rate
            )
        return waveform[0, : 120 * target_rate].to("cuda")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=260409344)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    records = read_jsonl(args.manifest)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise RuntimeError(f"empty manifest: {args.manifest}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    pipeline = TRTPipeline(args.engine_dir, use_graph=True, bucket_seconds=10)
    bundle_config = json.loads((args.bundle / "config.json").read_text())
    guidance_scale = float(bundle_config.get("ctm", {}).get("guidance_scale", 1.0))
    revision = git_revision()
    report_path = args.engine_dir / "build_report.json"
    precision = (
        json.loads(report_path.read_text()).get("precision", "unknown")
        if report_path.exists() else "unknown"
    )
    predictions = []
    rtfs = []
    for index, record in enumerate(records):
        waveform = load_input(Path(record["mixture"]), pipeline.input_rate)
        duration = waveform.numel() / pipeline.input_rate
        inference_seed = args.seed + index
        torch.manual_seed(inference_seed)
        started = time.perf_counter()
        separated = pipeline.separate(waveform)
        elapsed = time.perf_counter() - started
        seq = int(round(duration * 50))
        noise_cpu = (
            pipeline.noise_buffer[:1, :seq].detach().float().cpu().contiguous().numpy()
        )
        initial_noise_sha256 = hashlib.sha256(noise_cpu.tobytes()).hexdigest()
        pcm = (
            (separated.clamp(-1, 1) * 32767)
            .to(torch.int16)
            .cpu()
            .numpy()
        )
        stem = record["id"]
        speaker_1 = audio_dir / f"{stem}.spk1.wav"
        speaker_2 = audio_dir / f"{stem}.spk2.wav"
        stereo = audio_dir / f"{stem}.wav"
        write_wav_int16(speaker_1, pcm[0])
        write_wav_int16(speaker_2, pcm[1])
        write_stereo(stereo, pcm)
        rtf = elapsed / duration
        rtfs.append(rtf)
        predictions.append(
            {
                **record,
                "input_duration_seconds": duration,
                "model": "FastDialogueSidon",
                "runtime": "TensorRT",
                "precision": precision,
                "num_steps": 1,
                "guidance_scale": guidance_scale,
                "decoder": "WaveFitNeXt",
                "inference_seed": inference_seed,
                "initial_noise_sha256": initial_noise_sha256,
                "bundle": str(args.bundle),
                "engine_dir": str(args.engine_dir),
                "git_revision": revision,
                "prediction": str(stereo),
                "prediction_1": str(speaker_1),
                "prediction_2": str(speaker_2),
                "prediction_sample_rate": 24_000,
                "prediction_frames": int(pcm.shape[1]),
                "inference_seconds": elapsed,
                "rtf": rtf,
            }
        )
        print(
            f"FDS_PAPER_INFER_OK {index + 1}/{len(records)} "
            f"id={stem} rtf={rtf:.8f}",
            flush=True,
        )

    prediction_path = args.output_dir / "predictions.jsonl"
    prediction_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions)
    )
    summary = {
        "status": "ok",
        "model": "FastDialogueSidon",
        "runtime": "TensorRT",
        "precision": precision,
        "num_steps": 1,
        "decoder": "WaveFitNeXt",
        "guidance_scale": guidance_scale,
        "manifest": str(args.manifest),
        "bundle": str(args.bundle),
        "engine_dir": str(args.engine_dir),
        "git_revision": revision,
        "num_samples": len(predictions),
        "seed": args.seed,
        "rtf_mean": float(np.mean(rtfs)),
        "rtf_median": float(np.median(rtfs)),
        "gpu": torch.cuda.get_device_name(0),
    }
    (args.output_dir / "inference_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"FDS_PAPERBENCH_OK output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
