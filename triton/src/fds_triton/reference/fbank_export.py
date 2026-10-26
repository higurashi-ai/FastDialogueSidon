#!/usr/bin/env python3
"""ONNX/TensorRT-exportable SeamlessM4T fbank front-end.

Replicates ``extract_seamless_m4t_features`` (torchaudio kaldi fbank with
dither=0, remove_dc_offset, preemphasis 0.97, povey window, 80 mel bins,
25/10 ms, per-mel-bin utterance normalization, stride-2 stacking) with
plain ONNX ops so it can live inside the encoder TRT engine:

- The whole per-frame chain DC-removal -> pre-emphasis (replicate pad) ->
  povey window -> zero-pad to 512 -> rFFT is LINEAR in the 400-sample
  frame, so it is baked into ONE conv1d (kernel 400, stride 160, 514
  output channels = real 257 + imag 257).
- Conv weights are scaled by 1/S so fp16 power spectra cannot overflow;
  the log-domain constant this introduces cancels exactly in the per-bin
  mean subtraction (clamp epsilon is scaled to keep parity with fp32).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SAMPLE_RATE = 16_000
FRAME_SIZE = 400
FRAME_SHIFT = 160
N_FFT = 512
N_BINS = 257
NUM_MEL = 80
PREEMPH = 0.97
EPS = 1.1920928955078125e-07
SCALE = 8.0  # conv weights / SCALE -> power / SCALE^2, fp16-safe


def _povey_window(size: int) -> torch.Tensor:
    n = torch.arange(size, dtype=torch.float64)
    return (0.5 - 0.5 * torch.cos(2 * math.pi * n / (size - 1))) ** 0.85


def _frame_transform() -> torch.Tensor:
    """[514, 400] combined linear map frame -> (real, imag) rFFT bins."""
    eye = torch.eye(FRAME_SIZE, dtype=torch.float64)
    dc = eye - torch.full(
        (FRAME_SIZE, FRAME_SIZE), 1.0 / FRAME_SIZE, dtype=torch.float64
    )
    pre = eye.clone()
    for row in range(FRAME_SIZE):
        prev = max(row - 1, 0)
        pre[row, prev] -= PREEMPH if row > 0 else 0.0
        if row == 0:
            pre[0, 0] -= PREEMPH
    window = torch.diag(_povey_window(FRAME_SIZE))
    combined = window @ pre @ dc  # [400, 400]

    k = torch.arange(N_BINS, dtype=torch.float64)[:, None]
    n = torch.arange(FRAME_SIZE, dtype=torch.float64)[None, :]
    angle = 2 * math.pi * k * n / N_FFT
    real = torch.cos(angle) @ combined
    imag = -torch.sin(angle) @ combined
    return torch.cat([real, imag], dim=0) / SCALE


def _mel_banks() -> torch.Tensor:
    from torchaudio.compliance import kaldi

    bins, _ = kaldi.get_mel_banks(
        NUM_MEL, N_FFT, float(SAMPLE_RATE), 20.0, 0.0, 100.0, -500.0, 1.0
    )
    return F.pad(bins, (0, 1)).double()  # [80, 257]


class SeamlessFbankFrontend(nn.Module):
    """waveform [B, num_samples] (16 kHz) -> features [B, frames/2, 160].

    The overlapping frames (400 @ hop 160) are materialized as three
    slice+concat of 160-sample rows (frame = rows i, i+1, i+2 zero-padded
    to 480), so the DFT becomes one plain [T,480]@[480,514] GEMM — TRT
    handles this far better than a C_in=1 conv1d (3.15 ms).
    """

    def __init__(self, with_norm: bool = True) -> None:
        super().__init__()
        self.with_norm = with_norm
        weight = _frame_transform().float()  # [514, 400]
        weight = F.pad(weight, (0, 3 * FRAME_SHIFT - FRAME_SIZE))  # [514,480]
        self.register_buffer("frame_weight", weight.t().contiguous())
        self.register_buffer("mel_weight", _mel_banks().t().float())  # [257,80]

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        peak = waveform.abs().max(dim=-1, keepdim=True).values.clamp_min(1e-6)
        normalized = 0.9 * waveform / peak
        padded = F.pad(normalized, (160, 160))
        batch, total = padded.shape
        rows = padded.view(batch, total // FRAME_SHIFT, FRAME_SHIFT)
        frames = rows.shape[1] - 2
        windows = torch.cat(
            [
                rows[:, 0:frames],
                rows[:, 1 : frames + 1],
                rows[:, 2 : frames + 2],
            ],
            dim=-1,
        )  # [B, T, 480]
        spectrum = torch.matmul(windows, self.frame_weight)  # [B, T, 514]
        real, imag = spectrum[..., :N_BINS], spectrum[..., N_BINS:]
        power = real * real + imag * imag  # [B, T, 257]
        mel = torch.matmul(power, self.mel_weight)  # [B, T, 80]
        log_mel = torch.clamp_min(mel, EPS / (SCALE * SCALE)).log()
        if not self.with_norm:
            return log_mel
        mean = log_mel.mean(dim=1, keepdim=True)
        centered = log_mel - mean
        var = centered.square().sum(dim=1, keepdim=True) / (frames - 1)
        normed = centered / torch.sqrt(var + 1e-5)  # [B, T, 80]
        return normed.reshape(batch, frames // 2, 2 * NUM_MEL)
