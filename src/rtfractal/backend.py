"""Compute-backend selection (CUDA GPU or numba CPU).

The choice is made once per call site through resolve(); RTFRACTAL_BACKEND
(cpu | cuda | auto) overrides the default 'auto'.
"""
import os

_CUDA = None


def cuda_available():
    global _CUDA
    if _CUDA is None:
        try:
            from numba import cuda
            _CUDA = bool(cuda.is_available())
        except Exception:
            _CUDA = False
    return _CUDA


def resolve(backend=None):
    """Return 'cuda' or 'cpu'."""
    backend = backend or os.environ.get('RTFRACTAL_BACKEND', 'auto')
    if backend == 'auto':
        return 'cuda' if cuda_available() else 'cpu'
    if backend == 'cuda' and not cuda_available():
        raise RuntimeError("backend 'cuda' requested but no CUDA device is available")
    if backend not in ('cpu', 'cuda'):
        raise ValueError(f"unknown backend {backend!r}")
    return backend
