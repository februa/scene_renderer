from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray


Array: TypeAlias = NDArray[Any]


class ArrayGeometry(ABC):
    """ArrayFrame 上の素子配置を返すアレイ形状インターフェース。

    このクラスは、受波器固定座標での素子位置 positions を shape [n_ch, 3]、単位 m で返す契約を定義する。

    受波器の姿勢、音源方向、ビームフォーミング重み、信号投影は責務に含めない。
    信号処理上は、CH 間到達遅延 tau[ch] = r[ch] dot u / c の r[ch] を与える幾何入力である。
    """

    @abstractmethod
    def positions(self) -> Array:
        """アレイ素子位置を返す。

        Args:
            なし。形状パラメータは実装クラスが保持する。

        Returns:
            ArrayFrame 上の素子位置。shape は [n_ch, 3]、axis=0 は CH、axis=1 は [Bow, Starboard, Up]、単位は m。

        Raises:
            実装クラスに依存する。抽象メソッド自体は NotImplementedError を送出する。
        """

        raise NotImplementedError


@dataclass(frozen=True)
class LinearArray(ArrayGeometry):
    """単一直線上に等間隔で素子を並べるアレイ形状。

    このクラスは、n_ch 個の素子を ArrayFrame の指定 axis 上に spacing [m] 間隔で配置し、
    positions shape [n_ch, 3] を返す。

    曲線アレイ、任意配置、素子感度、ビーム重みは責務に含めない。
    信号処理上は、狭帯域平面波の CH 間位相差を決める位置ベクトル r[ch] を与える。
    """

    n_ch: int
    spacing: float
    axis: int = 0
    centered: bool = True

    def __post_init__(self) -> None:
        """線形アレイの幾何パラメータを検証する。

        Args:
            なし。dataclass の保持値を検証する。

        Returns:
            なし。

        Raises:
            ValueError: n_ch が 0 以下、spacing が 0 以下、axis が 0/1/2 以外の場合。
        """

        if self.n_ch <= 0:
            raise ValueError("n_ch must be positive")
        if self.spacing <= 0:
            raise ValueError("spacing must be positive")
        if self.axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1, or 2")

    def positions(self) -> Array:
        """ArrayFrame 上の線形アレイ素子位置を返す。

        Args:
            なし。

        Returns:
            素子位置。shape は [n_ch, 3]、axis=0 は CH、axis=1 は [Bow, Starboard, Up]、単位は m。

        Raises:
            なし。n_ch、spacing、axis は __post_init__ で検証済み。
        """

        pos = np.zeros((self.n_ch, 3), dtype=float)
        coord = np.arange(self.n_ch, dtype=float) * self.spacing
        if self.centered:
            # centered=True では幾何中心を原点に置き、共通遅延ではなく CH 間相対遅延だけを残す。
            coord -= np.mean(coord)
        # pos shape: [n_ch, xyz=3]。指定 axis の列だけに座標を入れ、他軸は ArrayFrame 原点上に置く。
        pos[:, self.axis] = coord
        return pos
