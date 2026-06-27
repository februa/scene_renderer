from __future__ import annotations

from scene_renderer.receiver import Receiver
from scene_renderer.scene import AcousticSource, Environment

from .source_projector import NarrowbandPlaneWaveProjector, SourceProjector


class ProjectorFactory:
    """音源と環境に応じて適切な projector 実装を選択する。"""

    def resolve(
        self,
        source: AcousticSource,
        receiver: Receiver,
        environment: Environment,
    ) -> SourceProjector:
        del source, receiver, environment
        return NarrowbandPlaneWaveProjector()
