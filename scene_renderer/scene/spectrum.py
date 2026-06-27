from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np


class Spectrum(ABC):
    """信号成分の周波数特性を返す抽象インターフェース。"""

    @abstractmethod
    def evaluate(self, freq_axis: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True)
class ToneSpectrum(Spectrum):
    """単一周波数トーンを表す最小スペクトラム実装。"""

    frequency: float

    def evaluate(self, freq_axis: np.ndarray) -> np.ndarray:
        freq_axis = np.asarray(freq_axis, dtype=float)
        out = np.zeros_like(freq_axis, dtype=complex)
        if freq_axis.size == 0:
            return out
        idx = int(np.argmin(np.abs(freq_axis - self.frequency)))
        out[idx] = 1.0 + 0.0j
        return out
