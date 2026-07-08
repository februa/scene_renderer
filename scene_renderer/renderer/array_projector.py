from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from scene_renderer.receiver import Receiver
from scene_renderer.scene import Environment

from .propagation import PropagatedSource
from .source_projector import SourceProjector


Array: TypeAlias = NDArray[Any]


class ArrayProjector:
    """音源単位の projector を呼び出して多CH寄与を加算する。

    このクラスは、PropagatedSource と SourceProjector を 1 対 1 に対応させ、各 projector が返す
    x_i[ch, t] を shape [n_ch, n_sample] 上で加算する。

    個別経路の位相式、音源波形生成、contributor 間の合成は責務に含めない。
    信号処理上は、音源ごとの投影結果を受波器アレイの総局所音源寄与へ集約する段である。
    """

    def project(
        self,
        propagated_sources: list[PropagatedSource],
        projectors: list[SourceProjector],
        receiver: Receiver,
        environment: Environment,
        axis_t: Array,
        fs: float,
    ) -> Array:
        """伝搬済み音源を対応 projector で投影して加算する。

        Args:
            propagated_sources: 伝搬済み音源リスト。
            projectors: 各 propagated_source に対応する projector リスト。
            receiver: 受波器定義。array.positions() は shape [n_ch, 3]、単位 m。
            environment: 伝搬環境。音速 c の単位は m/s。
            axis_t: 時間軸。shape は [n_sample]、単位は s。
            fs: サンプリング周波数。単位は Hz。

        Returns:
            多CH複素信号。shape は [n_ch, n_sample]。

        Raises:
            ValueError: propagated_sources と projectors の要素数が一致しない場合。
        """

        if len(propagated_sources) != len(projectors):
            raise ValueError("propagated_sources and projectors must have the same length")

        n_ch = receiver.array.positions().shape[0]
        n_sample = np.asarray(axis_t, dtype=float).size
        out = np.zeros((n_ch, n_sample), dtype=np.complex64)

        for propagated_source, projector in zip(propagated_sources, projectors, strict=True):
            contribution = projector.project(
                paths=propagated_source.paths,
                receiver=receiver,
                environment=environment,
                axis_t=axis_t,
                fs=fs,
            )
            if contribution.shape != out.shape:
                raise ValueError(f"projector output must have shape {out.shape}, got {contribution.shape}")
            # 各 projector の戻り値は同じ [ch, t] 格子に揃っているため、CH 軸・時間軸を保ったまま線形加算する。
            out += contribution.astype(np.complex64, copy=False)
        return out
