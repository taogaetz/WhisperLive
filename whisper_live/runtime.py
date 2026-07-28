"""Runtime device and precision selection without importing PyTorch."""

import os

import ctranslate2


def resolve_device(requested=None):
    """Return the requested device or auto-detect a CTranslate2 CUDA device."""
    if requested:
        return requested
    return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"


def resolve_compute_type(device, requested=None):
    """Select a compute type supported by the chosen CTranslate2 device."""
    configured = requested or os.getenv("PASCALSCRIBE_COMPUTE_TYPE")
    supported = ctranslate2.get_supported_compute_types(device)

    if configured:
        if configured not in supported:
            supported_list = ", ".join(sorted(supported))
            raise ValueError(
                f"Compute type '{configured}' is unavailable on {device}; "
                f"supported values: {supported_list}"
            )
        return configured

    if device == "cuda":
        # Pascal has fast INT8 arithmetic but no native FP16 tensor cores.
        # int8_float32 is both faster and smaller than float32 on a GTX 1080 Ti.
        for candidate in ("int8_float32", "float32"):
            if candidate in supported:
                return candidate
    elif "int8" in supported:
        return "int8"

    return "float32"


def resolve_runtime(requested_device=None, requested_compute_type=None):
    """Return a validated ``(device, compute_type)`` pair."""
    device = resolve_device(requested_device)
    return device, resolve_compute_type(device, requested_compute_type)
