from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from scene_renderer.receiver import Receiver


Array: TypeAlias = NDArray[Any]


class SensorNoiseGenerator:
    """受波器ごとの独立雑音を生成するための生成器。

    このクラスは、センサ自己雑音の振幅指定を保持し、Receiver と axis_t から
    x_sensor[ch, t] を返す契約を提供する。現在の実装は安全な無寄与としてゼロ信号を返す。

    局所音源、背景雑音場、Scene 全体の合成は責務に含めない。
    信号処理上は、将来 CH 独立雑音を [ch, t] 格子に生成するための境界である。
    """

    def __init__(
        self,
        amplitude: float | ArrayLike = 0.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        """センサ雑音生成器を作成する。

        Args:
            amplitude: 雑音標準偏差の線形振幅。scalar または shape [n_ch] の配列。0 以上。
            rng: 将来の確率雑音生成に使う乱数生成器。None の場合は default_rng を作成する。

        Returns:
            なし。

        Raises:
            ValueError: amplitude が負、または 1 次元を超える配列の場合。
        """

        amplitude_array = np.asarray(amplitude, dtype=float)
        if amplitude_array.ndim > 1:
            raise ValueError("amplitude must be a scalar or 1-D array")
        # 標準偏差として扱う量なので負値は物理的に意味を持たず、無音化ではなく入力エラーにする。
        if bool(np.any(amplitude_array < 0.0)):
            raise ValueError("amplitude must be non-negative")
        self.amplitude: float | Array = float(amplitude_array) if amplitude_array.ndim == 0 else amplitude_array
        self.rng = rng or np.random.default_rng()

    def generate(
        self,
        receiver: Receiver,
        axis_t: Array,
        fs: float,
    ) -> Array:
        """センサ自己雑音の多CH寄与を生成する。

        Args:
            receiver: 受波器定義。array.positions() は shape [n_ch, 3]、単位 m。
            axis_t: 時間軸。shape は [n_sample]、単位は s。
            fs: サンプリング周波数。単位は Hz。現在のゼロ出力では使わない。

        Returns:
            センサ雑音寄与。shape は [n_ch, n_sample]、dtype は float32。現在は全要素 0。

        Raises:
            なし。
        """

        del fs
        n_ch = receiver.array.positions().shape[0]
        n_sample = np.asarray(axis_t, dtype=float).size
        # 雑音振幅指定だけで乱数を返すと再現性や PSD 基準が曖昧になるため、生成モデル接続までは無寄与に固定する。
        return np.zeros((n_ch, n_sample), dtype=np.float32)
