from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray


Array: TypeAlias = NDArray[Any]


class Envelope(ABC):
    """時間軸に対する振幅包絡を返す抽象インターフェース。

    このクラスは、axis shape [n_sample]、単位 s を入力として、同じ shape の線形振幅包絡を
    返す契約を定義する。

    搬送波生成、周波数特性、アレイ投影は責務に含めない。
    信号処理上は、音源基準信号 s(t) の時間変動成分 env(t) を与える。
    """

    @abstractmethod
    def evaluate(self, axis: Array) -> Array:
        """時間軸上で包絡を評価する。

        Args:
            axis: 時間軸。shape は [n_sample]、axis=0 はサンプル、単位は s。

        Returns:
            線形振幅包絡。shape は [n_sample]、axis=0 は入力時間サンプルに対応する。

        Raises:
            実装クラスに依存する。抽象メソッド自体は NotImplementedError を送出する。
        """

        raise NotImplementedError


class ConstantEnvelope(Envelope):
    """全時間で一定値 1 を返す包絡。

    このクラスは、入力時間軸 axis と同じ shape の実数配列を返し、全サンプルの線形振幅を 1 とする。

    振幅スケーリング、発音区間の制御、フェード処理は責務に含めない。
    信号処理上は、s(t) = A exp(j 2π f t) の定常 tone 条件を表す。
    """

    def evaluate(self, axis: Array) -> Array:
        """入力時間軸と同じ shape の 1 配列を返す。

        Args:
            axis: 時間軸。shape は [n_sample]、axis=0 はサンプル、単位は s。

        Returns:
            線形振幅包絡。shape は [n_sample]、全要素 1.0。

        Raises:
            ValueError: axis が 1 次元でない場合。
        """

        axis_array = np.asarray(axis, dtype=float)
        if axis_array.ndim != 1:
            raise ValueError(f"axis must be 1-D, got shape {axis_array.shape}")
        # 定常 tone の包絡なので、時間サンプル数だけを引き継ぎ、物理振幅は SourceComponent 側で掛ける。
        return np.ones_like(axis_array, dtype=float)
