from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class Envelope(ABC):
    """時間軸に対する振幅包絡を返す抽象インターフェース。"""

    @abstractmethod
    def evaluate(self, axis: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ConstantEnvelope(Envelope):
    """全時間で一定値 1 を返す包絡。"""

    def evaluate(self, axis: np.ndarray) -> np.ndarray:
        return np.ones_like(np.asarray(axis, dtype=float), dtype=float)
