from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np


class ArrayGeometry(ABC):
    """ArrayFrame 上の素子配置を返すアレイ形状インターフェース。"""

    @abstractmethod
    def positions(self) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True)
class LinearArray(ArrayGeometry):
    """単一直線上に素子を並べる最小アレイ実装。"""

    n_ch: int
    spacing: float
    axis: int = 0
    centered: bool = True

    def __post_init__(self) -> None:
        if self.n_ch <= 0:
            raise ValueError("n_ch must be positive")
        if self.spacing <= 0:
            raise ValueError("spacing must be positive")
        if self.axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1, or 2")

    def positions(self) -> np.ndarray:
        pos = np.zeros((self.n_ch, 3), dtype=float)
        coord = np.arange(self.n_ch, dtype=float) * self.spacing
        # centered=True ではアレイ原点を幾何中心に合わせる。
        if self.centered:
            coord -= np.mean(coord)
        pos[:, self.axis] = coord
        return pos
