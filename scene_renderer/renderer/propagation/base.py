from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from scene_renderer.scene import Environment
from scene_renderer.receiver import Receiver
from scene_renderer.renderer.source_renderer import RenderedSource
from .path import PropagatedSource


class PropagationModel(ABC):
    """描画済み音源から伝搬経路群を生成する抽象インターフェース。"""

    @abstractmethod
    def propagate(
        self,
        rendered_sources: list[RenderedSource],
        environment: Environment,
        receiver: Receiver,
        axis_t: np.ndarray,
    ) -> list[PropagatedSource]:
        raise NotImplementedError
