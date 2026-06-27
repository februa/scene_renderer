from __future__ import annotations

from scene_renderer.receiver import Receiver
from scene_renderer.scene import Scene

from .array_projector import ArrayProjector
from .contributor import MultiChannelContributor
from .projector_factory import ProjectorFactory
from .propagation import FreeFieldPropagation, PropagationModel
from .source_renderer import SourceRenderer


class SourceFieldContributor(MultiChannelContributor):
    """局所音源の寄与を描画・伝搬・投影して多CH化する。"""

    def __init__(
        self,
        source_renderer: SourceRenderer | None = None,
        propagation_model: PropagationModel | None = None,
        projector_factory: ProjectorFactory | None = None,
        array_projector: ArrayProjector | None = None,
    ) -> None:
        self.source_renderer = source_renderer or SourceRenderer()
        self.propagation_model = propagation_model or FreeFieldPropagation()
        self.projector_factory = projector_factory or ProjectorFactory()
        self.array_projector = array_projector or ArrayProjector()

    def render(
        self,
        scene: Scene,
        receiver: Receiver,
        axis_t,
        fs: float,
    ):
        # contributor の責務分割を明示し、将来の差し替え点を残したまま処理を流す。
        rendered_sources = self.source_renderer.render(scene.sources, axis_t)
        propagated_sources = self.propagation_model.propagate(
            rendered_sources=rendered_sources,
            environment=scene.environment,
            receiver=receiver,
            axis_t=axis_t,
        )
        projectors = [
            self.projector_factory.resolve(
                source=propagated_source.rendered_source.source,
                receiver=receiver,
                environment=scene.environment,
            )
            for propagated_source in propagated_sources
        ]
        return self.array_projector.project(
            propagated_sources=propagated_sources,
            projectors=projectors,
            receiver=receiver,
            environment=scene.environment,
            axis_t=axis_t,
            fs=fs,
        )
