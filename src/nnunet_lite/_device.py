from __future__ import annotations

import warnings

import torch


def resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def warn_cpu_fallback(what: str, device: torch.device, error: BaseException) -> None:
    # Called from the function that falls back, so the warning points at its caller.
    reason = (str(error).strip() or type(error).__name__).splitlines()[0]
    warnings.warn(
        f"{what} on {device} failed ({reason}); falling back to CPU", RuntimeWarning, stacklevel=3
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
