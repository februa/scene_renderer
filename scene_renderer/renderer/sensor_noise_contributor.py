from __future__ import annotations

from scene_renderer.receiver import Receiver
from scene_renderer.scene import Scene

from .contributor import MultiChannelContributor
from .sensor_noise import SensorNoiseGenerator


class SensorNoiseContributor(MultiChannelContributor):
    """受波器固有の雑音寄与を生成する contributor。"""

    def __init__(
        self,
        sensor_noise_generator: SensorNoiseGenerator | None = None,
    ) -> None:
        self.sensor_noise_generator = sensor_noise_generator or SensorNoiseGenerator()

    def render(
        self,
        scene: Scene,
        receiver: Receiver,
        axis_t,
        fs: float,
    ):
        del scene
        return self.sensor_noise_generator.generate(
            receiver=receiver,
            axis_t=axis_t,
            fs=fs,
        )
