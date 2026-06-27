from __future__ import annotations

from scene_renderer.receiver import Receiver
from scene_renderer.scene import Scene

from .ambient_renderer import AmbientFieldRenderer
from .contributor import MultiChannelContributor


class AmbientFieldContributor(MultiChannelContributor):
    """背景雑音場の寄与を多CH信号へ変換する contributor。"""

    def __init__(
        self,
        ambient_renderer: AmbientFieldRenderer | None = None,
    ) -> None:
        self.ambient_renderer = ambient_renderer or AmbientFieldRenderer()

    def render(
        self,
        scene: Scene,
        receiver: Receiver,
        axis_t,
        fs: float,
    ):
        return self.ambient_renderer.render(
            fields=scene.ambient_fields,
            receiver=receiver,
            axis_t=axis_t,
            fs=fs,
        )
