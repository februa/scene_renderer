from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from scene_renderer.renderer.source_renderer import RenderedSource


@dataclass(frozen=True)
class PropagationPath:
    """単一路の伝搬結果を保持する経路表現。"""

    signal: np.ndarray
    direction_world: np.ndarray
    delay: float
    gain: float
    path_type: str
    virtual_source_pos_world: np.ndarray | None = None
    frequency: float | None = None


@dataclass(frozen=True)
class PropagatedSource:
    """一つの描画済み音源に対する伝搬経路群を束ねる。"""

    rendered_source: RenderedSource
    paths: list[PropagationPath]
