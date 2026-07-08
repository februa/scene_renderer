from __future__ import annotations

import numpy as np

from scene_renderer.receiver import Receiver
from scene_renderer.renderer.source_renderer import RenderedSource
from scene_renderer.scene import Environment

from .base import Array, PropagationModel
from .path import PropagatedSource, PropagationPath


class FreeFieldPropagation(PropagationModel):
    """自由音場の直達波 1 経路を生成する伝搬モデル。

    このクラスは、音源位置と受波器位置の差分から distance [m]、direction_world shape [3]、
    delay = distance / c [s] を計算し、path_type="direct" の 1 経路として返す。

    反射、吸収、球面拡散 gain、ドップラー、絶対遅延の信号反映は責務に含めない。
    信号処理上は、アレイ投影に必要な到来方向と伝搬遅延を決める最小の幾何モデルである。
    """

    def propagate(
        self,
        rendered_sources: list[RenderedSource],
        environment: Environment,
        receiver: Receiver,
        axis_t: Array,
    ) -> list[PropagatedSource]:
        """描画済み音源から自由音場の直達波経路を作る。

        Args:
            rendered_sources: 描画済み音源リスト。各 signal の shape は [n_sample]。
            environment: 伝搬環境。音速 c の単位は m/s。
            receiver: 受波器定義。trajectory.position(t0) は shape [3]、単位 m。
            axis_t: 時間軸。shape は [n_sample]、単位は s。先頭時刻 t0 で幾何を固定する。

        Returns:
            伝搬済み音源リスト。各音源に direct path を 1 つ持つ。

        Raises:
            ValueError: 音源位置と受波器位置が同一で到来方向を定義できない場合。
        """

        axis_t_array = np.asarray(axis_t, dtype=float)
        t0 = float(axis_t_array[0]) if axis_t_array.size else 0.0
        receiver_pos = receiver.trajectory.position(t0)
        propagated_sources: list[PropagatedSource] = []

        for rendered in rendered_sources:
            source_pos = rendered.source.trajectory.position(t0)
            # diff_world は受波器から音源へ向かう位置差 [m]。到来方向の符号は projector の tau 式と対応させる。
            diff_world = source_pos - receiver_pos
            distance = float(np.linalg.norm(diff_world))
            if distance <= 0:
                # 同一点では direction_world = diff / distance が定義できず、CH 間位相差も物理的に曖昧になる。
                raise ValueError("source and receiver positions must not be identical")
            direction_world = diff_world / distance
            propagated_sources.append(
                PropagatedSource(
                    rendered_source=rendered,
                    paths=[
                        PropagationPath(
                            signal=rendered.signal,
                            direction_world=direction_world,
                            # delay = distance / c。最小構成では保持のみで、時間波形への絶対遅延はまだ反映しない。
                            delay=distance / environment.c,
                            # 球面拡散を未接続のまま 1/distance だけ入れるとレベル基準が変わるため、現段階では 1.0 に固定する。
                            gain=1.0,
                            path_type="direct",
                            virtual_source_pos_world=np.asarray(source_pos, dtype=float),
                            frequency=rendered.frequency,
                        )
                    ],
                )
            )
        return propagated_sources
