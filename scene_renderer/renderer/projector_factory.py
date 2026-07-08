from __future__ import annotations

from scene_renderer.receiver import Receiver
from scene_renderer.scene import AcousticSource, Environment

from .source_projector import NarrowbandPlaneWaveProjector, SourceProjector


class ProjectorFactory:
    """音源と環境に応じて適切な projector 実装を選択する。

    このクラスは、AcousticSource、Receiver、Environment の組から SourceProjector を返す。
    現在は最小構成として常に NarrowbandPlaneWaveProjector を返す。

    projector 自体の位相計算、伝搬経路生成、音源波形生成は責務に含めない。
    信号処理上は、将来 broadband/球面波/反射経路用 projector を選ぶための分岐点である。
    """

    def resolve(
        self,
        source: AcousticSource,
        receiver: Receiver,
        environment: Environment,
    ) -> SourceProjector:
        """音源・受波器・環境に対応する projector を返す。

        Args:
            source: 局所音源定義。
            receiver: 受波器定義。
            environment: 伝搬環境。

        Returns:
            SourceProjector。現在は NarrowbandPlaneWaveProjector。

        Raises:
            なし。現在の最小構成では条件分岐しない。
        """

        del source, receiver, environment
        # 現在の SourceRenderer は ToneSpectrum のみを描画するため、狭帯域平面波 projector を選ぶ。
        return NarrowbandPlaneWaveProjector()
