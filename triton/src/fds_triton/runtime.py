"""Triton pipeline runtime: static buffers, per-length CUDA graphs, streaming.

Same interface as the TensorRT ``fastdialoguesidon.runtime.TRTPipeline``:
waveform [B, <=1_920_000] fp32 16 kHz -> encoder (fbank on GPU) -> DiT
(fresh noise per window) -> WaveFitNeXt -> audio [2B, 1, <=2_880_000] fp32
(first B rows speaker 1). No engine build step: any CUDA GPU Triton supports.
"""

from __future__ import annotations

import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from .model import TritonModel
from .reference.modeling import load_bundle

WINDOW_SECONDS = 120  # maximum window; shorter inputs run variable-length
IN_RATE, OUT_RATE = 16_000, 24_000
SAMPLES = WINDOW_SECONDS * IN_RATE  # 1_920_000 (max)
SEQ = WINDOW_SECONDS * 50  # 6000 latent frames (max)
OUT_SAMPLES = WINDOW_SECONDS * OUT_RATE  # 2_880_000 per speaker (max)
SAMPLES_PER_FRAME = 320  # 1 latent frame per 20 ms of input
OUT_PER_FRAME = 480  # output samples per latent frame (24 kHz)
MIN_SEQ = 160  # 3.2 s (same floor as the TRT engines)


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


def load_waveform(path: Path, device: str = "cuda", target_rate: int = IN_RATE) -> torch.Tensor:
    """Any-rate/any-channel wav -> target_rate mono fp32 tensor on GPU
    (GPU sinc resample when the file is not already at target_rate).
    16-bit PCM is read with ``wave``; other encodings via torchaudio."""
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            channels = handle.getnchannels()
            assert handle.getsampwidth() == 2, "16-bit PCM expected"
            raw = handle.readframes(handle.getnframes())
        data = np.frombuffer(raw, dtype=np.int16).reshape(-1, channels)
        waveform = torch.from_numpy(data.astype(np.float32) / 32768.0).to(device).t().contiguous()
    except (wave.Error, AssertionError):
        import torchaudio

        waveform, rate = torchaudio.load(str(path))
        waveform = waveform.float().to(device)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if rate != target_rate:
        import torchaudio.functional as AF

        waveform = AF.resample(waveform, rate, target_rate)
    return waveform[0][: SEQ * SAMPLES_PER_FRAME * target_rate // IN_RATE]


def write_wav_int16(path: Path, data: np.ndarray, rate: int = OUT_RATE) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(data.tobytes())


class TritonPipeline:
    """Fixed max-size I/O buffers + one CUDA graph per (length, batch).

    Default: windows are padded up to 10 s buckets (bucket_seconds=10) and one
    graph is cached per bucket; zero padding does not change the result for
    the true-length part and the output is trimmed to the true length.
    bucket_seconds=None runs exact (20 ms-padded) lengths.
    """

    def __init__(
        self,
        bundle_dir: Path,
        use_graph: bool = True,
        bucket_seconds: int | None = 10,
        max_batch: int = 1,
        model: TritonModel | None = None,
        gelu_tanh: bool = True,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        torch.set_float32_matmul_precision("highest")  # fp32 fbank: no TF32
        device = torch.device("cuda")
        if model is None:
            bundle = load_bundle(bundle_dir, device="cuda")
            model = TritonModel(bundle, max_seq=SEQ, gelu_tanh=gelu_tanh, dtype=dtype)
            del bundle
            torch.cuda.empty_cache()
        self.model = model
        self.use_graph = use_graph
        self.bucket_frames = None if bucket_seconds is None else int(bucket_seconds) * 50
        self.input_rate = IN_RATE
        self.in_per_frame = SAMPLES_PER_FRAME
        self.max_batch = int(max_batch)
        self.wave_buffer = torch.zeros(self.max_batch, SEQ * SAMPLES_PER_FRAME, device=device)
        self.noise_buffer = torch.empty(self.max_batch, SEQ, 64, device=device)
        self.audio_buffer = torch.zeros(2 * self.max_batch, 1, SEQ * OUT_PER_FRAME, device=device)
        self.compute = torch.cuda.Stream()
        self._graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        self._pool = None

    def _inputs(self, seq: int, batch: int):
        """Flat (contiguous) views, the layout separate*/TRT callers write."""
        samples = seq * SAMPLES_PER_FRAME
        wave_in = self.wave_buffer.view(-1)[: batch * samples].view(batch, samples)
        noise = self.noise_buffer.view(-1)[: batch * seq * 64].view(batch, seq, 64)
        return wave_in, noise

    @torch.inference_mode()
    def _forward(self, seq: int, batch: int) -> None:
        wave_in, noise = self._inputs(seq, batch)
        audio = self.model.forward(wave_in, noise)  # [2B, 1, seq*480]
        samples = seq * OUT_PER_FRAME
        out = self.audio_buffer.view(-1)[: 2 * batch * samples].view(2 * batch, 1, samples)
        out.copy_(audio)

    def stage_fns(self, seq: int, batch: int = 1):
        """Per-stage callables (for benchmarking) on the static buffers."""
        m = self.model
        wave_in, noise = self._inputs(seq, batch)
        with torch.inference_mode():
            hidden = m.encoder(wave_in)
            latents = m.dit(hidden, noise)

        def enc():
            with torch.inference_mode():
                m.encoder(wave_in)

        def dit():
            with torch.inference_mode():
                m.dit(hidden, noise)

        def dec():
            with torch.inference_mode():
                m.decoder(latents)

        return {"encoder": enc, "dit": dit, "decoder": dec}

    def run_chain(self, seq: int = SEQ, batch: int = 1, fresh_noise: bool = True) -> None:
        """Launch one window batch on the CURRENT stream (fresh noise unless
        fresh_noise=False, e.g. for reproducibility checks)."""
        assert MIN_SEQ <= seq <= SEQ and 1 <= batch <= self.max_batch, (seq, batch)
        if fresh_noise:
            self.noise_buffer.normal_()
        if not self.use_graph:
            self._forward(seq, batch)
            return
        graph = self._graphs.get((seq, batch))
        if graph is None:
            # eager warm-up on a side stream (Triton autotune + cuBLAS heuristics)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                self._forward(seq, batch)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            if self._pool is None:
                self._pool = torch.cuda.graph_pool_handle()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._pool):
                self._forward(seq, batch)
            self._graphs[(seq, batch)] = graph
        graph.replay()

    def _bucket(self, num_samples: int) -> int:
        seq = max(MIN_SEQ, (num_samples + SAMPLES_PER_FRAME - 1) // SAMPLES_PER_FRAME)
        if self.bucket_frames is not None:
            seq = min(SEQ, ((seq + self.bucket_frames - 1) // self.bucket_frames) * self.bucket_frames)
        return seq

    def separate(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform 16 kHz (<= 120 s) fp32 on GPU -> audio [2, out_samples] fp32."""
        num_samples = int(waveform.numel())
        assert num_samples <= SAMPLES, "input longer than the 120 s window"
        seq = self._bucket(num_samples)
        out_samples = num_samples * OUT_RATE // IN_RATE
        with torch.cuda.stream(self.compute):
            self.wave_buffer.view(-1)[:num_samples].copy_(waveform.view(-1))
            self.wave_buffer.view(-1)[num_samples : seq * SAMPLES_PER_FRAME].zero_()
            self.run_chain(seq)
            audio = (
                self.audio_buffer.view(-1)[: 2 * seq * OUT_PER_FRAME]
                .view(2, seq * OUT_PER_FRAME)[:, :out_samples]
                .clone()
            )
        torch.cuda.synchronize()
        return audio

    def separate_batch(self, waveforms: list[torch.Tensor]) -> torch.Tensor:
        """Same-bucket windows -> audio [B, 2, out_samples] (B <= max_batch)."""
        batch = len(waveforms)
        assert 1 <= batch <= self.max_batch, f"pipeline built with max_batch={self.max_batch}"
        num_samples = max(int(w.numel()) for w in waveforms)
        seq = self._bucket(num_samples)
        out_samples = num_samples * OUT_RATE // IN_RATE
        with torch.cuda.stream(self.compute):
            wave_view = self.wave_buffer.view(-1)[: batch * seq * SAMPLES_PER_FRAME].view(batch, -1)
            wave_view.zero_()
            for index, wave_in in enumerate(waveforms):
                wave_view[index, : wave_in.numel()].copy_(wave_in.view(-1))
            self.run_chain(seq, batch)
            audio = (
                self.audio_buffer.view(-1)[: 2 * batch * seq * OUT_PER_FRAME]
                .view(2, batch, seq * OUT_PER_FRAME)
                .permute(1, 0, 2)[:, :, :out_samples]
                .clone()
            )
        torch.cuda.synchronize()
        return audio


def stream_files(pipeline: TritonPipeline, wav_paths: list[Path], output_dir: Path) -> float:
    """Overlapped streaming of 120 s 16 kHz windows: reader thread + pinned
    double buffers + async H2D/D2H + writer thread. Returns wall ms/window."""
    import time

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    in_pinned = [torch.empty(SAMPLES, dtype=torch.int16).pin_memory() for _ in range(2)]
    out_pinned = [torch.empty(2, OUT_SAMPLES, dtype=torch.int16).pin_memory() for _ in range(2)]
    audio_i16 = [torch.empty(2, OUT_SAMPLES, device=device, dtype=torch.int16) for _ in range(2)]
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
            pending_read = reader.submit(read_into, next_slot, wav_paths[index + 1])
        with torch.cuda.stream(pipeline.compute):
            wav_dev = in_pinned[slot].to(device, non_blocking=True)
            h2d_done[slot].record()
            pipeline.wave_buffer[:1].copy_((wav_dev.float() / 32768.0).unsqueeze(0))
            pipeline.compute.wait_event(d2h_done[slot])
            pipeline.run_chain()
            audio_i16[slot].copy_(
                (pipeline.audio_buffer[:2].view(2, OUT_SAMPLES).clamp(-1, 1) * 32767).to(torch.int16)
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
