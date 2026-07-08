from __future__ import annotations

from typing import Any, TypeAlias

from numpy.typing import NDArray

from scene_renderer.receiver import Receiver
from scene_renderer.scene import Scene

from .array_projector import ArrayProjector
from .contributor import MultiChannelContributor
from .projector_factory import ProjectorFactory
from .propagation import FreeFieldPropagation, PropagationModel
from .source_renderer import SourceRenderer


Array: TypeAlias = NDArray[Any]


class SourceFieldContributor(MultiChannelContributor):
    """局所音源の寄与を描画・伝搬・投影して多CH化する。

    このクラスは、Scene.sources を SourceRenderer、PropagationModel、ProjectorFactory、ArrayProjector の
    順に渡し、局所音源だけの x_source[ch, t] を生成する。

    背景雑音、センサ雑音、最終 dtype 変換は責務に含めない。
    信号処理上は、音源基準信号 s(t) をアレイ受信信号へ写す局所音源寄与の処理段である。
    """

    def __init__(
        self,
        source_renderer: SourceRenderer | None = None,
        propagation_model: PropagationModel | None = None,
        projector_factory: ProjectorFactory | None = None,
        array_projector: ArrayProjector | None = None,
    ) -> None:
        """局所音源 contributor を作成する。

        Args:
            source_renderer: 音源波形生成器。None の場合は SourceRenderer。
            propagation_model: 伝搬モデル。None の場合は FreeFieldPropagation。
            projector_factory: projector 選択器。None の場合は ProjectorFactory。
            array_projector: 音源別 projector の集約器。None の場合は ArrayProjector。

        Returns:
            なし。

        Raises:
            なし。None は既定実装へ解決する。
        """

        self.source_renderer = source_renderer or SourceRenderer()
        self.propagation_model = propagation_model or FreeFieldPropagation()
        self.projector_factory = projector_factory or ProjectorFactory()
        self.array_projector = array_projector or ArrayProjector()

    def render(
        self,
        scene: Scene,
        receiver: Receiver,
        axis_t: Array,
        fs: float,
    ) -> Array:
        """局所音源による多CH寄与を生成する。

        Args:
            scene: 音場定義。sources と environment を使う。
            receiver: 受波器定義。array.positions() は shape [n_ch, 3]、単位 m。
            axis_t: 時間軸。shape は [n_sample]、単位は s。
            fs: サンプリング周波数。単位は Hz。

        Returns:
            局所音源寄与。shape は [n_ch, n_sample]。

        Raises:
            SourceRenderer、PropagationModel、ArrayProjector が送出する例外を伝搬する。
        """

        # projector が広帯域 fractional delay FIR に必要とする halo を先に確保し、chunk 境界でも同じ信号値を生成する。
        sample_margin = self.projector_factory.required_sample_margin(receiver, scene.environment, fs)
        # 音源波形、伝搬幾何、アレイ投影を分けることで、各段の shape と物理量を局所化する。
        rendered_sources = self.source_renderer.render(scene.sources, axis_t, fs=fs, sample_margin=sample_margin)
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
