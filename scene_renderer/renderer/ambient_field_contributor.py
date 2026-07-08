from __future__ import annotations

from typing import Any, TypeAlias

from numpy.typing import NDArray

from scene_renderer.receiver import Receiver
from scene_renderer.scene import Scene

from .ambient_renderer import AmbientFieldRenderer
from .contributor import MultiChannelContributor


Array: TypeAlias = NDArray[Any]


class AmbientFieldContributor(MultiChannelContributor):
    """背景雑音場の寄与を多CH信号へ変換する contributor。

    このクラスは、Scene.ambient_fields を AmbientFieldRenderer へ渡し、x_ambient[ch, t] を生成する。

    局所音源、センサ自己雑音、最終合成は責務に含めない。
    信号処理上は、非局所的な空間雑音場を SceneRenderer の [ch, t] 加算格子へ接続する段である。
    """

    def __init__(
        self,
        ambient_renderer: AmbientFieldRenderer | None = None,
    ) -> None:
        """背景雑音 contributor を作成する。

        Args:
            ambient_renderer: 背景雑音レンダラ。None の場合は AmbientFieldRenderer。

        Returns:
            なし。

        Raises:
            なし。
        """

        self.ambient_renderer = ambient_renderer or AmbientFieldRenderer()

    def render(
        self,
        scene: Scene,
        receiver: Receiver,
        axis_t: Array,
        fs: float,
    ) -> Array:
        """背景雑音場による多CH寄与を生成する。

        Args:
            scene: 音場定義。ambient_fields を使う。
            receiver: 受波器定義。array.positions() は shape [n_ch, 3]、単位 m。
            axis_t: 時間軸。shape は [n_sample]、単位は s。
            fs: サンプリング周波数。単位は Hz。

        Returns:
            背景雑音寄与。shape は [n_ch, n_sample]。

        Raises:
            AmbientFieldRenderer が送出する例外を伝搬する。
        """

        return self.ambient_renderer.render(
            fields=scene.ambient_fields,
            receiver=receiver,
            axis_t=axis_t,
            fs=fs,
        )
