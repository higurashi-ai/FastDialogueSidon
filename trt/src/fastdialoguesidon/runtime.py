"""TensorRT pipeline runtime: zero-copy chain, CUDA graph, streaming.

The chain is waveform [B, 1920000] fp32 -> encoder (fbank in-engine) ->
DiT (fresh noise per window) -> WaveFitNeXt -> audio [2B, 1, 2880000].
Enqueue always happens on a non-default stream (plugin race rule).
"""

from __future__ import annotations

import os
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from . import flexrel_triton
from .engines import TRTStage, deserialize

WINDOW_SECONDS = 120  # maximum window; shorter inputs run variable-length
IN_RATE, OUT_RATE = 16_000, 24_000
SAMPLES = WINDOW_SECONDS * IN_RATE  # 1_920_000 (max)
SEQ = WINDOW_SECONDS * 50  # 6000 latent frames (max)
OUT_SAMPLES = WINDOW_SECONDS * OUT_RATE  # 2_880_000 per speaker (max)
SAMPLES_PER_FRAME = 320  # 1 latent frame per 20 ms of input
OUT_PER_FRAME = 480  # output samples per latent frame (24 kHz)
MIN_SEQ = 160  # engine optimization-profile minimum (3.2 s)

ENGINE_FILES = {
    "encoder": "encoder_wave.plan",
    "dit": "dit_ctm1step.plan",
    "decoder": "decoder_wavefitnext.plan",
}


def read_wav_int16(path: Path) -> np.ndarray:
    """Fast path for 16 kHz mono int16 wavs (no resample)."""
    with wave.open(str(path), "rb") as handle:
        assert handle.getnchannels() == 1, "mono input expected"
        assert handle.getframerate() == IN_RATE, (
            f"expected {IN_RATE} Hz input; use load_waveform() for "
            f"other rates (got {handle.getframerate()})"
        )
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype=np.int16)


def load_waveform(
    path: Path, device: str = "cuda", target_rate: int = IN_RATE
) -> torch.Tensor:
    """Any-rate/any-channel wav -> target_rate mono fp32 tensor on GPU.
    Files already at target_rate skip the resampler entirely.

    Resampling runs on the GPU (torchaudio sinc kernel, ~0.3 ms for a
    120 s 48 kHz stereo input on H100) — it sits in front of the TensorRT
    engines, not inside them, but is a negligible fraction of the chain.
    """
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        assert handle.getsampwidth() == 2, "16-bit PCM expected"
        raw = handle.readframes(handle.getnframes())
    data = np.frombuffer(raw, dtype=np.int16).reshape(-1, channels)
    waveform = (
        torch.from_numpy(data.astype(np.float32) / 32768.0)
        .to(device)
        .t()
        .contiguous()
    )
    if channels > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if rate != target_rate:
        import torchaudio.functional as AF

        waveform = AF.resample(waveform, rate, target_rate)
    max_samples = SEQ * SAMPLES_PER_FRAME * target_rate // IN_RATE
    return waveform[0][:max_samples]


def write_wav_int16(path: Path, data: np.ndarray, rate: int = OUT_RATE):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(data.tobytes())


class TRTPipeline:
    """Three chained engines with fixed max-size buffers.

    Variable length: any input up to 120 s runs at its own length (padded
    to a 20 ms multiple). Buffers are allocated at the maximum and
    per-length shapes are set on the execution contexts.

    Default: windows are padded up to 10 s buckets (bucket_seconds=10)
    and one CUDA graph is cached per bucket (12 graphs max) — the fastest
    configuration for arbitrary mixed lengths. Zero padding does not change
    the result for the true-length part (validated at 15-120 s); output is
    trimmed to the true length.

    bucket_seconds=None disables bucketing (exact-length execution); set
    use_graph=False with it if the workload has unbounded distinct
    lengths (a graph is cached per length otherwise).
    """

    def __init__(
        self,
        engine_dir: Path,
        use_graph: bool = True,
        bucket_seconds: int | None = 10,
        max_batch: int = 1,
        check_output: bool = True,
    ) -> None:
        """max_batch > 1 enables separate_batch() for same-length windows
        (engines must have been built with --max-batch >= max_batch).
        B=1 sequential is the recommended default: the GPU is already
        saturated at batch 1 and batching adds only ~8% throughput."""
        assert os.environ.get("NVIDIA_TF32_OVERRIDE") == "0", (
            "engines are built with NVIDIA_TF32_OVERRIDE=0 and must run "
            "under the same setting (TRT enforces build/run consistency)"
        )
        engine_dir = Path(engine_dir)
        flexrel_triton.register_trt_plugin_qr_dyn()
        engines = {
            tag: deserialize(engine_dir / name)
            for tag, name in ENGINE_FILES.items()
        }
        device = torch.device("cuda")
        self.use_graph = use_graph
        self.bucket_frames = (
            None if bucket_seconds is None else int(bucket_seconds) * 50
        )
        # engine input rate (in-engine sinc resample when != 16 kHz)
        report_path = engine_dir / "build_report.json"
        self.input_rate = 16_000
        if report_path.exists():
            import json

            self.input_rate = int(
                json.loads(report_path.read_text()).get(
                    "input_rate", 16_000
                )
            )
        self.in_per_frame = (
            SAMPLES_PER_FRAME * self.input_rate // IN_RATE
        )
        self.max_batch = int(max_batch)
        self.wave_buffer = torch.zeros(
            self.max_batch, SEQ * self.in_per_frame, device=device
        )
        self.noise_buffer = torch.empty(
            self.max_batch, SEQ, 64, device=device
        )
        self.encoder_stage = TRTStage(
            engines["encoder"], {"waveform": self.wave_buffer}
        )
        self.conditioning_buffer = next(
            iter(self.encoder_stage.outputs.values())
        )
        self.dit_stage = TRTStage(
            engines["dit"],
            {
                "conditioning": self.conditioning_buffer,
                "noise": self.noise_buffer,
            },
        )
        self.latent_buffer = next(iter(self.dit_stage.outputs.values()))
        self.decoder_stage = TRTStage(
            engines["decoder"], {"latents": self.latent_buffer}
        )
        self.audio_buffer = next(iter(self.decoder_stage.outputs.values()))
        self.compute = torch.cuda.Stream()
        self._seq = SEQ
        self._batch = self.max_batch
        self._graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}

        with torch.cuda.stream(self.compute):
            self.wave_buffer.normal_()
            self.noise_buffer.normal_()
            self._enqueue_chain()
        torch.cuda.synchronize()
        if check_output:  # False only to inspect broken builds (e.g. an fp16 decoder)
            audio = self.audio_buffer.float()
            assert torch.isfinite(audio).all(), "engine chain produced NaN/Inf"
            assert audio.std().item() > 0
        self.wave_buffer.zero_()

    def _set_length(self, seq: int, batch: int = 1) -> None:
        if seq == self._seq and batch == self._batch:
            return
        assert MIN_SEQ <= seq <= SEQ, seq
        assert 1 <= batch <= self.max_batch, batch
        samples = seq * self.in_per_frame
        cond_dim = self.conditioning_buffer.shape[-1]
        enc = self.encoder_stage.context
        enc.set_input_shape("waveform", (batch, samples))
        dit = self.dit_stage.context
        dit.set_input_shape("conditioning", (batch, seq, cond_dim))
        dit.set_input_shape("noise", (batch, seq, 64))
        dec = self.decoder_stage.context
        dec.set_input_shape("latents", (batch, seq, 64))
        self._seq = seq
        self._batch = batch

    def _enqueue_chain(self) -> None:
        stream = torch.cuda.current_stream()
        self.encoder_stage.enqueue(stream)
        self.dit_stage.enqueue(stream)
        self.decoder_stage.enqueue(stream)

    def run_chain(self, seq: int = SEQ, batch: int = 1) -> None:
        """Launch one window batch on the CURRENT stream."""
        self._set_length(seq, batch)
        self.noise_buffer.normal_()
        if not self.use_graph:
            self._enqueue_chain()
            return
        graph = self._graphs.get((seq, batch))
        if graph is None:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                self._enqueue_chain()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._enqueue_chain()
            self._graphs[(seq, batch)] = graph
        graph.replay()

    def separate(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform at self.input_rate (<= 120 s) fp32 on GPU ->
        audio [2, true_output_samples] fp32 (trimmed to input length)."""
        num_samples = int(waveform.numel())
        assert num_samples <= SEQ * self.in_per_frame, (
            "input longer than the 120 s window"
        )
        seq = max(
            MIN_SEQ,
            (num_samples + self.in_per_frame - 1) // self.in_per_frame,
        )
        if self.bucket_frames is not None:
            seq = min(
                SEQ,
                (
                    (seq + self.bucket_frames - 1)
                    // self.bucket_frames
                )
                * self.bucket_frames,
            )
        padded_samples = seq * self.in_per_frame
        out_samples = num_samples * OUT_RATE // self.input_rate
        with torch.cuda.stream(self.compute):
            self.wave_buffer.view(-1)[:num_samples].copy_(
                waveform.view(-1)
            )
            self.wave_buffer.view(-1)[num_samples:padded_samples].zero_()
            self.run_chain(seq)
            audio = (
                self.audio_buffer.view(-1)[: 2 * seq * OUT_PER_FRAME]
                .view(2, seq * OUT_PER_FRAME)[:, :out_samples]
                .float()
                .clone()
            )
        torch.cuda.synchronize()
        return audio

    def separate_batch(self, waveforms: list[torch.Tensor]) -> torch.Tensor:
        """Batch same-length windows -> audio [B, 2, out_samples].

        All waveforms must round to the SAME bucketed length; requires
        max_batch >= len(waveforms). B=1 sequential is usually the better
        interface — batching gains only ~8% throughput at 8x latency.
        """
        batch = len(waveforms)
        assert 1 <= batch <= self.max_batch, (
            f"pipeline built with max_batch={self.max_batch}"
        )
        lengths = {int(w.numel()) for w in waveforms}
        num_samples = max(lengths)
        seq = max(
            MIN_SEQ,
            (num_samples + self.in_per_frame - 1) // self.in_per_frame,
        )
        if self.bucket_frames is not None:
            seq = min(
                SEQ,
                ((seq + self.bucket_frames - 1) // self.bucket_frames)
                * self.bucket_frames,
            )
        padded_samples = seq * self.in_per_frame
        out_samples = num_samples * OUT_RATE // self.input_rate
        with torch.cuda.stream(self.compute):
            self.wave_buffer[:batch].zero_()
            for index, wave in enumerate(waveforms):
                self.wave_buffer[index, : wave.numel()].copy_(wave.view(-1))
            self.run_chain(seq, batch)
            # decoder output is [2B, 1, T]: first B rows are speaker 1
            audio = (
                self.audio_buffer.view(-1)[: 2 * batch * seq * OUT_PER_FRAME]
                .view(2, batch, seq * OUT_PER_FRAME)
                .permute(1, 0, 2)[:, :, :out_samples]
                .float()
                .clone()
            )
        torch.cuda.synchronize()
        return audio


def stream_files(
    pipeline: TRTPipeline,
    wav_paths: list[Path],
    output_dir: Path,
) -> float:
    """Overlapped streaming: reader thread + pinned double buffers + async
    H2D/D2H + writer thread. Returns wall milliseconds per window."""
    import time

    assert pipeline.input_rate == IN_RATE, (
        "streaming assumes pre-prepared 16 kHz windows; build the engine "
        "with --input-rate 16000 or feed via separate()"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    in_pinned = [
        torch.empty(SAMPLES, dtype=torch.int16).pin_memory() for _ in range(2)
    ]
    out_pinned = [
        torch.empty(2, OUT_SAMPLES, dtype=torch.int16).pin_memory()
        for _ in range(2)
    ]
    audio_i16 = [
        torch.empty(2, OUT_SAMPLES, device=device, dtype=torch.int16)
        for _ in range(2)
    ]
    h2d_done = [torch.cuda.Event() for _ in range(2)]
    d2h_done = [torch.cuda.Event() for _ in range(2)]
    copy_stream = torch.cuda.Stream()
    reader = ThreadPoolExecutor(max_workers=1)
    writer = ThreadPoolExecutor(max_workers=1)

    def read_into(slot: int, path: Path) -> int:
        data = read_wav_int16(path)
        assert data.shape[0] >= SAMPLES, f"{path}: shorter than 120 s"
        in_pinned[slot].copy_(torch.from_numpy(data[:SAMPLES].copy()))
        return slot

    def write_out(slot: int, stem: str) -> None:
        d2h_done[slot].synchronize()
        data = out_pinned[slot].numpy()
        for spk in (0, 1):
            write_wav_int16(output_dir / f"{stem}_spk{spk}.wav", data[spk])

    started = time.perf_counter()
    pending_read = reader.submit(read_into, 0, wav_paths[0])
    pending_writes = []
    for index, path in enumerate(wav_paths):
        slot = index % 2
        pending_read.result()
        if index + 1 < len(wav_paths):
            next_slot = (index + 1) % 2
            h2d_done[next_slot].synchronize()
            pending_read = reader.submit(
                read_into, next_slot, wav_paths[index + 1]
            )
        with torch.cuda.stream(pipeline.compute):
            wav_dev = in_pinned[slot].to(device, non_blocking=True)
            h2d_done[slot].record()
            pipeline.wave_buffer.copy_(
                (wav_dev.float() / 32768.0).unsqueeze(0)
            )
            pipeline.compute.wait_event(d2h_done[slot])
            pipeline.run_chain()
            audio_i16[slot].copy_(
                (
                    pipeline.audio_buffer.view(2, OUT_SAMPLES).clamp(-1, 1)
                    * 32767
                ).to(torch.int16)
            )
        copy_stream.wait_stream(pipeline.compute)
        with torch.cuda.stream(copy_stream):
            out_pinned[slot].copy_(audio_i16[slot], non_blocking=True)
            d2h_done[slot].record()
        pending_writes.append(writer.submit(write_out, slot, path.stem))
        if len(pending_writes) > 1:
            pending_writes.pop(0).result()
    for future in pending_writes:
        future.result()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1e3 / len(wav_paths)
