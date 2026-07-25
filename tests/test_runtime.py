import unittest
from unittest.mock import patch

from whisper_live.runtime import (
    resolve_compute_type,
    resolve_device,
    resolve_runtime,
)


class TestRuntimeSelection(unittest.TestCase):
    @patch("whisper_live.runtime.ctranslate2.get_cuda_device_count", return_value=1)
    def test_cuda_is_detected_without_torch(self, _get_count):
        self.assertEqual(resolve_device(), "cuda")

    @patch(
        "whisper_live.runtime.ctranslate2.get_supported_compute_types",
        return_value={"float32", "int8_float32"},
    )
    def test_pascal_default_prefers_int8_float32(self, _get_types):
        self.assertEqual(resolve_compute_type("cuda"), "int8_float32")

    @patch(
        "whisper_live.runtime.ctranslate2.get_supported_compute_types",
        return_value={"float32"},
    )
    def test_pascal_falls_back_to_float32(self, _get_types):
        self.assertEqual(resolve_compute_type("cuda"), "float32")

    @patch(
        "whisper_live.runtime.ctranslate2.get_supported_compute_types",
        return_value={"float32"},
    )
    def test_invalid_override_is_rejected(self, _get_types):
        with self.assertRaisesRegex(ValueError, "unavailable"):
            resolve_compute_type("cuda", "float16")

    @patch(
        "whisper_live.runtime.ctranslate2.get_supported_compute_types",
        return_value={"int8", "float32"},
    )
    @patch("whisper_live.runtime.ctranslate2.get_cuda_device_count", return_value=0)
    def test_cpu_default_is_int8(self, _get_count, _get_types):
        self.assertEqual(resolve_runtime(), ("cpu", "int8"))


if __name__ == "__main__":
    unittest.main()
