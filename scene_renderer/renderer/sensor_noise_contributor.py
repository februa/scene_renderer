from __future__ import annotations

from typing import Any, TypeAlias

from numpy.typing import NDArray

from scene_renderer.receiver import Receiver
from scene_renderer.scene import Scene

from .contributor import MultiChannelContributor
from .sensor_noise import SensorNoiseGenerator
from .render_result import RenderedContribution


Array: TypeAlias = NDArray[Any]


class SensorNoiseContributor(MultiChannelContributor):
    """受波器固有の雑音寄与を生成する contributor。

    このクラスは、Receiver 側のセンサ自己雑音を SensorNoiseGenerator へ委譲し、x_sensor[ch, t] を生成する。

    局所音源、背景雑音場、最終合成は責務に含めない。
    信号処理上は、Scene に属さない受波器固有雑音を [ch, t] 加算格子へ接続する段である。
    """

    def __init__(
        self,
        sensor_noise_generator: SensorNoiseGenerator | None = None,
    ) -> None:
        """センサ雑音 contributor を作成する。

        Args:
            sensor_noise_generator: センサ雑音生成器。None の場合は SensorNoiseGenerator。

        Returns:
            なし。

        Raises:
            なし。
        """

        self.sensor_noise_generator = sensor_noise_generator or SensorNoiseGenerator()

    def render(
        self,
        scene: Scene,
        receiver: Receiver,
        axis_t: Array,
        fs: float,
    ) -> Array:
        """センサ自己雑音による多CH寄与を生成する。

        Args:
            scene: 音場定義。センサ雑音は受波器側の現象なので現在は使わない。
            receiver: 受波器定義。array.positions() は shape [n_ch, 3]、単位 m。
            axis_t: 時間軸。shape は [n_sample]、単位は s。
            fs: サンプリング周波数。単位は Hz。

        Returns:
            センサ雑音寄与。shape は [n_ch, n_sample]。

        Raises:
            SensorNoiseGenerator が送出する例外を伝搬する。
        """

        del scene
        return self.sensor_noise_generator.generate(
            receiver=receiver,
            axis_t=axis_t,
            fs=fs,
        )

    def render_contributions(
        self, scene: Scene, receiver: Receiver, axis_t: Array, fs: float
    ) -> tuple[RenderedContribution, ...]:
        """センサ自己雑音を成分情報付きで返す。

        Args:
            scene: 音場定義。センサ雑音生成には使用しない。
            receiver: 受波器定義。
            axis_t: 時間軸。shapeは[n_sample]、単位はs。
            fs: サンプリング周波数。単位はHz。

        Returns:
            sensor_noise寄与1件。signal shapeは[n_ch, n_sample]。

        Raises:
            SensorNoiseGeneratorが送出する例外を伝搬する。
        """

        return (
            RenderedContribution(
                identifier="sensor_noise",
                role="noise",
                kind="sensor",
                signal=self.render(scene=scene, receiver=receiver, axis_t=axis_t, fs=fs),
            ),
        )
