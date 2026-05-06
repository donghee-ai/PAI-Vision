from __future__ import annotations

from functools import lru_cache


@lru_cache
def resolve_yolo_device(device: str) -> str:
    """Resolve a user-facing device hint into a concrete Ultralytics device string."""
    normalized = device.strip().lower()
    if normalized not in {"auto", ""}:
        return device

    try:
        import torch
    except Exception:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda:0"

    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"

    return "cpu"
