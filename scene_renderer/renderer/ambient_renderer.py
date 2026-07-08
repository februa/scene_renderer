from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from scene_renderer.receiver import Receiver
from scene_renderer.scene import AmbientField


Array: TypeAlias = NDArray[Any]


class AmbientFieldRenderer:
    """背景雑音場定義から多CH雑音信号を生成する。

    このクラスは、AmbientField のリストと Receiver から背景雑音寄与 x_ambient[ch, t] を返す。
    現在の実装は設計上の安全な無寄与として shape [n_ch, n_sample] の float32 ゼロを返す。

    局所音源、センサ自己雑音、Scene 全体の合成は責務に含めない。
    信号処理上は、将来 covariance = L L^T に基づく相関雑音を入れるための [ch, t] 出力境界である。
    """

    def __init__(
        self,
        rng: np.random.Generator | None = None,
    ) -> None:
        """背景雑音レンダラを作成する。

        Args:
            rng: 将来の確率雑音生成に使う乱数生成器。None の場合は default_rng を作成する。

        Returns:
            なし。

        Raises:
            なし。
        """

        self.rng = rng or np.random.default_rng()

    def render(
        self,
        fields: list[AmbientField],
        receiver: Receiver,
        axis_t: Array,
        fs: float,
    ) -> Array:
        """背景雑音場の多CH寄与を生成する。

        Args:
            fields: 背景雑音場リスト。現段階では shape 契約維持のため保持のみ。
            receiver: 受波器定義。array.positions() は shape [n_ch, 3]、単位 m。
            axis_t: 時間軸。shape は [n_sample]、単位は s。
            fs: サンプリング周波数。単位は Hz。将来のスペクトル雑音生成で使う。

        Returns:
            背景雑音寄与。shape は [n_ch, n_sample]、dtype は float32。現在は全要素 0。

        Raises:
            なし。
        """

        del fields, fs
        n_ch = receiver.array.positions().shape[0]
        n_sample = np.asarray(axis_t, dtype=float).size
        # 背景雑音の統計モデルが未接続の段階で非ゼロ乱数を返すと評価基準が曖昧になるため、無寄与を明示する。
        return np.zeros((n_ch, n_sample), dtype=np.float32)
