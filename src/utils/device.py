"""Resolución automática del dispositivo de cómputo (CUDA / MPS / CPU)."""
from __future__ import annotations

from .logging_utils import get_logger

logger = get_logger(__name__)


def resolve_device(requested: str | None = "auto") -> str:
    """Devuelve un dispositivo válido para la máquina actual.

    - "auto"/None -> cuda si hay GPU NVIDIA; si no, mps (Apple Silicon); si no, cpu.
    - "cuda" pedido pero no disponible -> baja a mps/cpu con aviso.
    - "mps" pedido pero no disponible -> baja a cpu con aviso.
    - "cpu" -> cpu.
    """
    try:
        import torch
    except ImportError:
        return "cpu"

    has_cuda = torch.cuda.is_available()
    has_mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()

    req = (requested or "auto").lower()

    if req == "auto":
        dev = "cuda" if has_cuda else ("mps" if has_mps else "cpu")
        logger.info("Dispositivo auto -> %s (cuda=%s, mps=%s)", dev, has_cuda, has_mps)
        return dev

    if req.startswith("cuda"):
        if has_cuda:
            return req
        fallback = "mps" if has_mps else "cpu"
        logger.warning("CUDA no disponible; usando '%s' en su lugar.", fallback)
        return fallback

    if req == "mps":
        if has_mps:
            return "mps"
        logger.warning("MPS no disponible; usando 'cpu' en su lugar.")
        return "cpu"

    return "cpu"


_cuda_ready = False


def enable_cuda_optimizations() -> bool:
    """Activa los ajustes de rendimiento de NVIDIA (una sola vez):
      - cudnn.benchmark: autotune de convoluciones (tamaños de entrada fijos).
      - TF32 en matmul/cudnn: mucho más rápido en Ampere+ con calidad equivalente.
    Devuelve True si hay CUDA. fp16 lo gestiona cada detector con half=True.
    """
    global _cuda_ready
    if _cuda_ready:
        return True
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    torch.backends.cudnn.benchmark = True
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
    name = torch.cuda.get_device_name(0)
    logger.info("CUDA activado: %s | cudnn.benchmark=on, TF32=on, fp16 disponible", name)
    _cuda_ready = True
    return True
