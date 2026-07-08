from __future__ import annotations

from scene_renderer.receiver import Receiver
from scene_renderer.scene import AcousticSource, Environment

from .source_projector import PlaneWaveProjector, SourceProjector


class ProjectorFactory:
    """音源と環境に応じて適切な projector 実装を選択する。

    このクラスは、AcousticSource、Receiver、Environment の組から SourceProjector を返す。
    現在は自由音場の平面波近似として常に PlaneWaveProjector を返す。

    projector 自体の位相計算、伝搬経路生成、音源波形生成は責務に含めない。
    信号処理上は、将来の球面波や反射経路用 projector を選ぶための波面モデル分岐点である。
    """

    def __init__(self, plane_wave_projector: PlaneWaveProjector | None = None) -> None:
        """ProjectorFactory を作成する。

        Args:
            plane_wave_projector: 既定の平面波 projector。None の場合は PlaneWaveProjector を生成する。

        Returns:
            なし。

        Raises:
            PlaneWaveProjector の生成時例外を伝搬する。
        """

        self.plane_wave_projector = plane_wave_projector or PlaneWaveProjector()

    def required_sample_margin(self, receiver: Receiver, environment: Environment, fs: float) -> int:
        """既定 projector が要求する source signal の halo サンプル数を返す。

        Args:
            receiver: 受波器定義。array.positions() は shape [n_ch, 3]、単位 m。
            environment: 伝搬環境。音速 c の単位は m/s。
            fs: サンプリング周波数。単位は Hz。

        Returns:
            必要 halo。単位は sample。

        Raises:
            projector が送出する例外を伝搬する。
        """

        return self.plane_wave_projector.required_sample_margin(receiver, environment, fs)

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
            SourceProjector。現在は PlaneWaveProjector。

        Raises:
            なし。現在の構成では帯域で分岐せず、波面モデルとして平面波を選ぶ。
        """

        del source, receiver, environment
        # 狭帯域/広帯域の違いは PlaneWaveProjector 内の遅延適用方式で扱い、Factory は波面モデルだけを選ぶ。
        return self.plane_wave_projector
