"""Small, dependency-free NVIDIA telemetry sampler for the web dashboard."""

import csv
import subprocess
import threading
import time


class NvidiaTelemetry:
    """Read a compact GPU snapshot from ``nvidia-smi`` with a short cache."""

    QUERY_FIELDS = (
        "index",
        "name",
        "utilization.gpu",
        "utilization.memory",
        "memory.used",
        "memory.total",
        "temperature.gpu",
        "power.draw",
        "power.limit",
        "clocks.current.graphics",
        "fan.speed",
    )

    def __init__(self, cache_seconds=1.0, runner=None, clock=None):
        self.cache_seconds = cache_seconds
        self._runner = runner or subprocess.run
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._cached_at = 0.0
        self._cached = None

    @staticmethod
    def _number(value, cast=float):
        value = value.strip()
        if not value or value.upper() == "N/A":
            return None
        return cast(float(value))

    @classmethod
    def _parse(cls, output):
        rows = list(csv.reader(output.splitlines()))
        if not rows:
            return {"available": False}

        values = [value.strip() for value in rows[0]]
        if len(values) != len(cls.QUERY_FIELDS):
            return {"available": False}

        return {
            "available": True,
            "index": cls._number(values[0], int),
            "name": values[1],
            "utilization_percent": cls._number(values[2], int),
            "memory_utilization_percent": cls._number(values[3], int),
            "memory_used_mib": cls._number(values[4], int),
            "memory_total_mib": cls._number(values[5], int),
            "temperature_c": cls._number(values[6], int),
            "power_w": cls._number(values[7]),
            "power_limit_w": cls._number(values[8]),
            "graphics_clock_mhz": cls._number(values[9], int),
            "fan_percent": cls._number(values[10], int),
        }

    def snapshot(self):
        """Return current GPU statistics, or ``available: false`` on failure."""
        with self._lock:
            now = self._clock()
            if self._cached is not None and now - self._cached_at < self.cache_seconds:
                return dict(self._cached)

            try:
                result = self._runner(
                    [
                        "nvidia-smi",
                        f"--query-gpu={','.join(self.QUERY_FIELDS)}",
                        "--format=csv,noheader,nounits",
                        "--id=0",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=2,
                )
                snapshot = self._parse(result.stdout)
            except (OSError, subprocess.SubprocessError, ValueError):
                snapshot = {"available": False}

            self._cached = snapshot
            self._cached_at = now
            return dict(snapshot)
