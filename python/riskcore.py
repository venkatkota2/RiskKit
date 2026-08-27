"""Zero-dependency ctypes adapter for the riskcore shared library."""

from __future__ import annotations

import ctypes
import math
import os
import platform
from collections.abc import Iterable
from pathlib import Path


def _library_name() -> str:
    system = platform.system()
    if system == "Windows":
        return "riskcore.dll"
    if system == "Darwin":
        return "libriskcore.dylib"
    return "libriskcore.so"


def _default_library_path() -> Path:
    configured = os.environ.get("RISKCORE_LIBRARY")
    if configured:
        return Path(configured)
    root = Path(__file__).resolve().parents[1]
    release = root / "target" / "release" / _library_name()
    if release.exists():
        return release
    return root / "target" / "debug" / _library_name()


class RiskCore:
    def __init__(self, library_path: str | Path | None = None) -> None:
        path = Path(library_path) if library_path else _default_library_path()
        if not path.is_file():
            raise FileNotFoundError(
                f"riskcore shared library not found at {path}; run 'cargo build' "
                "or set RISKCORE_LIBRARY"
            )
        try:
            self._library = ctypes.CDLL(str(path))
        except OSError as error:
            raise OSError(f"could not load riskcore shared library at {path}: {error}") from error
        double_pointer = ctypes.POINTER(ctypes.c_double)
        for name in ("riskcore_historical_var", "riskcore_expected_shortfall"):
            function = getattr(self._library, name)
            function.argtypes = [double_pointer, ctypes.c_size_t, ctypes.c_double]
            function.restype = ctypes.c_double
        self._library.riskcore_maximum_drawdown.argtypes = [
            double_pointer,
            ctypes.c_size_t,
        ]
        self._library.riskcore_maximum_drawdown.restype = ctypes.c_double

    @staticmethod
    def _array(values: Iterable[float]) -> tuple[ctypes.Array, int]:
        observations = [float(value) for value in values]
        if not observations or any(not math.isfinite(value) for value in observations):
            raise ValueError("returns must contain finite observations")
        array_type = ctypes.c_double * len(observations)
        return array_type(*observations), len(observations)

    @staticmethod
    def _checked(value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("riskcore rejected the supplied arguments")
        return value

    def historical_var(self, returns: Iterable[float], confidence: float = 0.99) -> float:
        values, length = self._array(returns)
        return self._checked(self._library.riskcore_historical_var(values, length, confidence))

    def expected_shortfall(self, returns: Iterable[float], confidence: float = 0.99) -> float:
        values, length = self._array(returns)
        return self._checked(self._library.riskcore_expected_shortfall(values, length, confidence))

    def maximum_drawdown(self, returns: Iterable[float]) -> float:
        values, length = self._array(returns)
        return self._checked(self._library.riskcore_maximum_drawdown(values, length))
