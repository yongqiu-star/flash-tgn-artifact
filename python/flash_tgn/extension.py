from __future__ import annotations

import importlib
from types import ModuleType


_CUDA_EXTENSION: ModuleType | None = None


def cuda_extension() -> ModuleType:
    """Return the Flash-TGN CUDA extension.

    The clean Flash-TGN package intentionally imports `flash_tgn._C` directly.
    The import is lazy so metadata modules remain importable before the
    extension is built.
    """
    global _CUDA_EXTENSION
    if _CUDA_EXTENSION is None:
        try:
            _CUDA_EXTENSION = importlib.import_module("flash_tgn._C")
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "flash_tgn._C is not built. Run `python setup.py build_ext "
                "--inplace` from the Flash-TGN project root."
            ) from exc
    return _CUDA_EXTENSION
