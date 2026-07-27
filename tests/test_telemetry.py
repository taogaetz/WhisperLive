import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from whisper_live.telemetry import NvidiaTelemetry


class TestNvidiaTelemetry(unittest.TestCase):
    SAMPLE = (
        "0, NVIDIA GeForce GTX 1080 Ti, 73, 19, 2048, 11264, "
        "62, 146.50, 250.00, 1670, 44\n"
    )

    def test_parses_relevant_gpu_fields(self):
        runner = Mock(return_value=SimpleNamespace(stdout=self.SAMPLE))
        telemetry = NvidiaTelemetry(runner=runner)

        self.assertEqual(
            telemetry.snapshot(),
            {
                "available": True,
                "index": 0,
                "name": "NVIDIA GeForce GTX 1080 Ti",
                "utilization_percent": 73,
                "memory_utilization_percent": 19,
                "memory_used_mib": 2048,
                "memory_total_mib": 11264,
                "temperature_c": 62,
                "power_w": 146.5,
                "power_limit_w": 250.0,
                "graphics_clock_mhz": 1670,
                "fan_percent": 44,
            },
        )

    def test_reuses_snapshot_during_cache_window(self):
        runner = Mock(return_value=SimpleNamespace(stdout=self.SAMPLE))
        clock = Mock(side_effect=[10.0, 10.5])
        telemetry = NvidiaTelemetry(
            cache_seconds=1.0,
            runner=runner,
            clock=clock,
        )

        telemetry.snapshot()
        telemetry.snapshot()

        runner.assert_called_once()

    def test_reports_unavailable_without_leaking_command_error(self):
        runner = Mock(side_effect=subprocess.CalledProcessError(1, "nvidia-smi"))
        telemetry = NvidiaTelemetry(runner=runner)

        self.assertEqual(telemetry.snapshot(), {"available": False})

    def test_accepts_unavailable_optional_fields(self):
        sample = self.SAMPLE.replace(", 44\n", ", N/A\n")
        telemetry = NvidiaTelemetry(
            runner=Mock(return_value=SimpleNamespace(stdout=sample)),
        )

        self.assertIsNone(telemetry.snapshot()["fan_percent"])
