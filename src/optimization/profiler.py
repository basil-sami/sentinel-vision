"""Per-frame pipeline profiler: measures every stage with GPU/CPU/IO tracking."""

import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class StageStats:
    calls: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0

    def record(self, ms: float):
        self.calls += 1
        self.total_ms += ms
        if ms < self.min_ms:
            self.min_ms = ms
        if ms > self.max_ms:
            self.max_ms = ms

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.calls if self.calls else 0.0

    def __str__(self) -> str:
        return f"{self.avg_ms:>7.1f}  {self.min_ms:>7.1f}  {self.max_ms:>7.1f}  {self.calls:>8d}"


@dataclass
class Counter:
    value: int = 0

    def inc(self, n: int = 1):
        self.value += n


class StageTimer:
    """Context manager for timing a pipeline stage."""

    def __init__(self, stats: StageStats):
        self._stats = stats

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        ms = (time.perf_counter() - self._start) * 1000
        self._stats.record(ms)


class PipelineProfiler:
    """Collects per-stage timing, counters, and system metrics."""

    def __init__(self):
        self.stages: dict[str, StageStats] = defaultdict(StageStats)
        self.counters: dict[str, Counter] = defaultdict(Counter)
        self._frame_times: list[float] = []
        self._detections_per_frame: list[int] = []
        self._gpu_samples: list[dict] = []
        self._cpu_samples: list[float] = []
        self._sampling = False
        self._sampler_thread: threading.Thread | None = None

    def timer(self, stage: str) -> StageTimer:
        return StageTimer(self.stages[stage])

    def count(self, name: str, n: int = 1):
        self.counters[name].inc(n)

    def record_frame_time(self, ms: float):
        self._frame_times.append(ms)

    def record_detections(self, count: int):
        self._detections_per_frame.append(count)

    def start_system_sampling(self, interval: float = 0.5):
        """Start background GPU/CPU sampling thread."""
        self._sampling = True

        def _sample():
            import subprocess
            import os
            while self._sampling:
                try:
                    gpu = subprocess.run(
                        [
                            "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True, text=True, timeout=5,
                    )
                    if gpu.returncode == 0:
                        parts = gpu.stdout.strip().split(", ")
                        if len(parts) >= 3:
                            self._gpu_samples.append({
                                "util": float(parts[0]),
                                "mem_used": float(parts[1]),
                                "mem_total": float(parts[2]),
                            })
                except Exception:
                    pass
                try:
                    import psutil
                    self._cpu_samples.append(psutil.cpu_percent(interval=0))
                except Exception:
                    pass
                time.sleep(interval)

        self._sampler_thread = threading.Thread(target=_sample, daemon=True)
        self._sampler_thread.start()

    def stop_system_sampling(self):
        self._sampling = False
        if self._sampler_thread:
            self._sampler_thread.join(timeout=3)

    @property
    def fps(self) -> float:
        if not self._frame_times:
            return 0.0
        total = sum(self._frame_times) / 1000.0
        return len(self._frame_times) / total if total > 0 else 0.0

    @property
    def avg_frame_ms(self) -> float:
        return sum(self._frame_times) / len(self._frame_times) if self._frame_times else 0.0

    def report(self) -> str:
        lines = []
        lines.append("=" * 72)
        lines.append("  PERFORMANCE INVESTIGATION REPORT")
        lines.append("=" * 72)
        lines.append("")

        # Per-frame timing
        lines.append("--- Per-Frame Timing ---")
        lines.append(f"{'Stage':<25s} {'Avg(ms)':>8s} {'Min(ms)':>8s} {'Max(ms)':>8s} {'Calls':>8s}")
        lines.append("-" * 60)
        sorted_stages = sorted(self.stages.items(), key=lambda x: x[1].total_ms, reverse=True)
        for name, stats in sorted_stages:
            lines.append(f"  {name:<23s} {stats}")
        lines.append(f"\n  {'Total frame time':<23s} {self.avg_frame_ms:>7.1f} ms")
        lines.append(f"  {'Pipeline FPS':<23s} {self.fps:>7.1f}")
        lines.append("")

        # Detections per frame
        if self._detections_per_frame:
            avg_det = sum(self._detections_per_frame) / len(self._detections_per_frame)
            lines.append("--- Detections Per Frame ---")
            lines.append(f"  Average:           {avg_det:>8.1f}")
            lines.append(f"  Peak:              {max(self._detections_per_frame):>8d}")
            lines.append(f"  Total (all frames): {sum(self._detections_per_frame):>8d}")
            lines.append("")

        # Counters
        if self.counters:
            lines.append("--- Call Counters ---")
            for name, c in sorted(self.counters.items(), key=lambda x: x[1].value, reverse=True):
                lines.append(f"  {name:<30s} {c.value:>8d}")
            lines.append("")

        # GPU
        if self._gpu_samples:
            utils = [s["util"] for s in self._gpu_samples]
            mems = [s["mem_used"] for s in self._gpu_samples]
            mem_total = self._gpu_samples[0]["mem_total"] if self._gpu_samples else 0
            lines.append("--- GPU Utilization ---")
            lines.append(f"  Average util:     {sum(utils)/len(utils):>7.1f}%")
            lines.append(f"  Peak util:        {max(utils):>7.1f}%")
            lines.append(f"  Average memory:   {sum(mems)/len(mems):>7.0f} MiB / {mem_total:.0f} MiB")
            lines.append(f"  Peak memory:      {max(mems):>7.0f} MiB")
            lines.append("")

        # CPU
        if self._cpu_samples:
            lines.append("--- CPU Utilization ---")
            lines.append(f"  Average:  {sum(self._cpu_samples)/len(self._cpu_samples):>7.1f}%")
            lines.append(f"  Peak:     {max(self._cpu_samples):>7.1f}%")
            lines.append(f"  Samples:  {len(self._cpu_samples):>8d}")
            lines.append("")

        # Summary
        lines.append("--- Top Bottlenecks (by total time) ---")
        cutoff = sum(s.total_ms for _, s in sorted_stages[:5])
        total = sum(s.total_ms for _, s in sorted_stages) or 1
        for i, (name, stats) in enumerate(sorted_stages[:10], 1):
            pct = stats.total_ms / total * 100
            lines.append(f"  {i:>2d}. {name:<25s} {stats.total_ms:>8.0f} ms total ({pct:>5.1f}%)")
        lines.append("")

        lines.append("=" * 72)
        return "\n".join(lines)
