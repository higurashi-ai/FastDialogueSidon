"""TensorRT engine building and execution helpers.

Hard-won rules baked in (see docs/SETUP.md, "Known pitfalls"):
- register the FlexRel plugin BEFORE any TRT builder/runtime work
  (importing tensorrt.plugin after engines exist segfaults);
- the wave encoder engine is STRONGLY TYPED (fp32 fbank front-end + fp16/bf16
  encoder dtypes carried in the ONNX) and requires NVIDIA_TF32_OVERRIDE=0
  — weakly-typed per-layer precision constraints are ignored by myelin,
  and TF32 GEMM noise wrecks quiet-bin log-mel;
- engines must be BUILT and RUN under the same NVIDIA_TF32_OVERRIDE
  setting (TRT refuses to create contexts otherwise);
- enqueue on a NON-DEFAULT stream: with the legacy default stream TRT runs
  layers on an internal stream, racing the plugin kernel.
"""

from __future__ import annotations

import os
import statistics
import time
from pathlib import Path

import torch


def timed_cuda(fn, warmup: int, runs: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return {
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def errors(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    delta = (reference.float() - actual.float()).abs()
    signal = reference.float().square().mean().sqrt()
    noise = delta.square().mean().sqrt()
    return {
        "max_abs": delta.max().item(),
        "relative_rmse": (noise / signal.clamp_min(1e-12)).item(),
        "snr_db": (20 * torch.log10(signal / noise.clamp_min(1e-12))).item(),
    }


def _parse(onnx_path: Path, network, logger):
    import tensorrt as trt

    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        raise RuntimeError(
            "ONNX parse failed: "
            + "; ".join(
                str(parser.get_error(i)) for i in range(parser.num_errors)
            )
        )


def build_engine(
    onnx_path: Path,
    profile_shapes: dict,
    engine_path: Path,
    workspace_gib: int = 24,
    precision: str = "bf16",
):
    """Weakly-typed bf16 engine (WaveFitNeXt decoder).

    bf16 only: an fp16 decoder overflows in the vocoder (NaN in 7-38 % of
    the output samples on CallFriend 120 s, measured 2026-10-26), so fp16 is not supported for this stage.
    """
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    _parse(onnx_path, network, logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_gib << 30
    )
    if precision != "bf16":
        raise ValueError(
            f"decoder precision {precision!r} unsupported: only bf16 "
            "(fp16 produces NaN in the vocoder)"
        )
    config.set_flag(trt.BuilderFlag.BF16)
    profile = builder.create_optimization_profile()
    for name, (mn, opt, mx) in profile_shapes.items():
        profile.set_shape(name, mn, opt, mx)
    config.add_optimization_profile(profile)
    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_seconds = time.perf_counter() - started
    if serialized is None:
        raise RuntimeError(f"TensorRT build failed for {onnx_path}")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)
    runtime = trt.Runtime(logger)
    return runtime.deserialize_cuda_engine(bytes(serialized)), build_seconds


def build_wave_engine(
    onnx_path: Path,
    profile_shapes: dict,
    engine_path: Path,
    workspace_gib: int = 24,
):
    """Strongly-typed build for the waveform-input encoder engine."""
    import tensorrt as trt

    assert os.environ.get("NVIDIA_TF32_OVERRIDE") == "0", (
        "set NVIDIA_TF32_OVERRIDE=0: TF32 GEMM noise breaks the fbank "
        "front-end (quiet-bin log-mel turns chaotic)"
    )
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    _parse(onnx_path, network, logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_gib << 30
    )
    profile = builder.create_optimization_profile()
    for name, (mn, opt, mx) in profile_shapes.items():
        profile.set_shape(name, mn, opt, mx)
    config.add_optimization_profile(profile)
    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_seconds = time.perf_counter() - started
    if serialized is None:
        raise RuntimeError(f"TensorRT build failed for {onnx_path}")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)
    runtime = trt.Runtime(logger)
    return runtime.deserialize_cuda_engine(bytes(serialized)), build_seconds


def deserialize(engine_path: Path):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"failed to deserialize {engine_path}")
    return engine


class TRTStage:
    """One engine execution with fixed tensor addresses (zero-copy chain)."""

    def __init__(self, engine, inputs: dict[str, torch.Tensor]):
        import tensorrt as trt

        self.context = engine.create_execution_context()
        self.outputs: dict[str, torch.Tensor] = {}
        dtype_map = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.BF16: torch.bfloat16,
        }
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                tensor = inputs[name]
                self.context.set_input_shape(name, tuple(tensor.shape))
                self.context.set_tensor_address(name, int(tensor.data_ptr()))
            else:
                shape = tuple(self.context.get_tensor_shape(name))
                buffer = torch.empty(
                    shape,
                    device="cuda",
                    dtype=dtype_map[engine.get_tensor_dtype(name)],
                )
                self.outputs[name] = buffer
                self.context.set_tensor_address(name, int(buffer.data_ptr()))

    def enqueue(self, stream) -> None:
        self.context.execute_async_v3(stream.cuda_stream)
