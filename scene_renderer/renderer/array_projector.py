from __future__ import annotations

import numpy as np

from scene_renderer.scene import Environment
from scene_renderer.receiver import Receiver

from .propagation import PropagatedSource
from .source_projector import SourceProjector


class ArrayProjector:
    """音源単位の projector を呼び出して多CH寄与を加算する。"""

    def project(
        self,
        propagated_sources: list[PropagatedSource],
        projectors: list[SourceProjector],
        receiver: Receiver,
        environment: Environment,
        axis_t: np.ndarray,
        fs: float,
    ) -> np.ndarray:
        if len(propagated_sources) != len(projectors):
            raise ValueError("propagated_sources and projectors must have the same length")

        n_ch = receiver.array.positions().shape[0]
        n_sample = np.asarray(axis_t, dtype=float).size
        out = np.zeros((n_ch, n_sample), dtype=np.complex64)

        for propagated_source, projector in zip(propagated_sources, projectors, strict=True):
            out += projector.project(
                paths=propagated_source.paths,
                receiver=receiver,
                environment=environment,
                axis_t=axis_t,
                fs=fs,
            )
        return out
