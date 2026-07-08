from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray


Array: TypeAlias = NDArray[Any]


class Spectrum(ABC):
    """周波数軸に対する信号成分の複素スペクトルを返す抽象インターフェース。

    このクラスは、freq_axis shape [n_bin]、単位 Hz を入力として、同じ shape の複素振幅を
    返す契約を定義する。

    時間波形の生成、包絡の適用、アレイ投影は責務に含めない。
    信号処理上は、周波数領域レンダリングへ拡張する際の音源スペクトル定義である。
    """

    @abstractmethod
    def evaluate(self, freq_axis: Array) -> Array:
        """周波数軸上でスペクトルを評価する。

        Args:
            freq_axis: 周波数軸。shape は [n_bin]、axis=0 は周波数ビン、単位は Hz。

        Returns:
            複素スペクトル。shape は [n_bin]、axis=0 は freq_axis と同じ周波数ビン。

        Raises:
            実装クラスに依存する。抽象メソッド自体は NotImplementedError を送出する。
        """

        raise NotImplementedError


@dataclass(frozen=True)
class ToneSpectrum(Spectrum):
    """単一周波数 tone を表すスペクトル実装。

    このクラスは、frequency [Hz] に最も近い周波数ビンへ複素振幅 1 を置き、それ以外を 0 とする
    離散スペクトルを返す。

    時間包絡、振幅スケーリング、複数 tone の合成は責務に含めない。
    信号処理上は、狭帯域平面波投影で使う単一周波数 f を保持する音源スペクトルである。
    """

    frequency: float

    def evaluate(self, freq_axis: Array) -> Array:
        """frequency に最も近いビンだけが 1 となる複素スペクトルを返す。

        Args:
            freq_axis: 周波数軸。shape は [n_bin]、axis=0 は周波数ビン、単位は Hz。

        Returns:
            複素スペクトル。shape は [n_bin]。空の freq_axis では空配列を返す。

        Raises:
            ValueError: freq_axis が 1 次元でない場合。
        """

        freq_axis_array = np.asarray(freq_axis, dtype=float)
        if freq_axis_array.ndim != 1:
            raise ValueError(f"freq_axis must be 1-D, got shape {freq_axis_array.shape}")
        out = np.zeros_like(freq_axis_array, dtype=complex)
        if freq_axis_array.size == 0:
            # 空の周波数軸は、上位の探索処理で候補がないケースを安全に扱えるよう空スペクトルで返す。
            return out
        # argmin は周波数ビン軸 axis=0 上で最近傍ビンを選ぶ。bin 幅内の補間はこのクラスの責務外とする。
        idx = int(np.argmin(np.abs(freq_axis_array - self.frequency)))
        out[idx] = 1.0 + 0.0j
        return out
