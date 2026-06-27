from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from scene_renderer.receiver import Receiver
from scene_renderer.scene import Scene


class MultiChannelContributor(ABC):
    """Scene 全体から多チャンネル寄与を生成する共通抽象。"""

    @abstractmethod
    def render(
        self,
        scene: Scene,
        receiver: Receiver,
        axis_t: np.ndarray,
        fs: float,
    ) -> np.ndarray:
        raise NotImplementedError
