from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeAlias

from numpy.typing import NDArray

from scene_renderer.receiver import Receiver
from scene_renderer.scene import Scene
from .render_result import RenderedContribution


Array: TypeAlias = NDArray[Any]


class MultiChannelContributor(ABC):
    """Scene 全体から多チャンネル寄与を生成する共通抽象。

    このクラスは、Scene と Receiver と時間軸 axis_t から、受信信号への 1 種類の寄与を
    shape [n_ch, n_sample] で返す契約を定義する。

    複数 contributor の合成、最終 dtype 変換、公開 API の時間軸検証は責務に含めない。
    信号処理上は、局所音源、背景雑音、センサ雑音などの物理的寄与を同じ [ch, t] 格子に揃える境界である。
    """

    @abstractmethod
    def render(
        self,
        scene: Scene,
        receiver: Receiver,
        axis_t: Array,
        fs: float,
    ) -> Array:
        """多CH寄与を生成する。

        Args:
            scene: 音場定義。局所音源、背景雑音場、環境を含む。
            receiver: 受波器定義。array.positions() は shape [n_ch, 3]、単位 m。
            axis_t: 時間軸。shape は [n_sample]、axis=0 は時間サンプル、単位は s。
            fs: サンプリング周波数。単位は Hz。

        Returns:
            多CH寄与。shape は [n_ch, n_sample]、axis=0 は CH、axis=1 は時間サンプル。

        Raises:
            実装クラスに依存する。抽象メソッド自体は NotImplementedError を送出する。
        """

        raise NotImplementedError

    def render_contributions(
        self,
        scene: Scene,
        receiver: Receiver,
        axis_t: Array,
        fs: float,
    ) -> tuple[RenderedContribution, ...]:
        """このcontributorが生成する寄与を成分情報付きで返す。

        Args:
            scene: 音場定義。
            receiver: 受波器定義。
            axis_t: 時間軸。shapeは[n_sample]、単位はs。
            fs: サンプリング周波数。単位はHz。

        Returns:
            寄与tuple。各signalのshapeは[n_ch, n_sample]。

        Raises:
            `render`実装が送出する例外を伝搬する。

        個別成分を提供しない独自contributorにも互換性を保つため、既定実装ではaggregate寄与を1件返す。
        """

        return (
            RenderedContribution(
                identifier=type(self).__name__,
                role="unspecified",
                kind="custom",
                signal=self.render(scene=scene, receiver=receiver, axis_t=axis_t, fs=fs),
            ),
        )
