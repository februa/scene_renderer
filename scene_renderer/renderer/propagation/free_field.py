from __future__ import annotations

import numpy as np

from scene_renderer.scene import Environment
from scene_renderer.receiver import Receiver
from scene_renderer.renderer.source_renderer import RenderedSource
from .base import PropagationModel
from .path import PropagatedSource, PropagationPath


class FreeFieldPropagation(PropagationModel):
    """自由音場の直達波 1 経路だけを生成する最小伝搬モデル。"""

    def propagate(
        self,
        rendered_sources: list[RenderedSource],
        environment: Environment,
        receiver: Receiver,
        axis_t: np.ndarray,
    ) -> list[PropagatedSource]:
        axis_t = np.asarray(axis_t, dtype=float)
        t0 = float(axis_t[0]) if axis_t.size else 0.0
        receiver_pos = receiver.trajectory.position(t0)
        propagated_sources: list[PropagatedSource] = []

        for rendered in rendered_sources:
            source_pos = rendered.source.trajectory.position(t0)
            diff_world = source_pos - receiver_pos
            distance = float(np.linalg.norm(diff_world))
            if distance <= 0:
                raise ValueError("source and receiver positions must not be identical")
            direction_world = diff_world / distance
            propagated_sources.append(
                PropagatedSource(
                    rendered_source=rendered,
                    paths=[
                        PropagationPath(
                            signal=rendered.signal,
                            direction_world=direction_world,
                            # 絶対遅延は将来拡張用に保持し、最小構成では信号へ反映しない。
                            delay=distance / environment.c,
                            gain=1.0,
                            path_type="direct",
                            virtual_source_pos_world=np.asarray(source_pos, dtype=float),
                            frequency=rendered.frequency,
                        )
                    ],
                )
            )
        return propagated_sources
