from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeAlias

from numpy.typing import NDArray

from scene_renderer.receiver import Receiver
from scene_renderer.renderer.source_renderer import RenderedSource
from scene_renderer.scene import Environment

from .path import PropagatedSource


Array: TypeAlias = NDArray[Any]


class PropagationModel(ABC):
    """描画済み音源から伝搬経路群を生成する抽象インターフェース。

    このクラスは、RenderedSource と Environment と Receiver から、WorldFrame 上の伝搬方向・遅延・gain を持つ
    PropagatedSource を返す契約を定義する。

    音源波形の生成、アレイ CH への位相投影、背景雑音生成は責務に含めない。
    信号処理上は、音源位置と受波器位置から到来方向と伝搬時間を求める幾何段である。
    """

    @abstractmethod
    def propagate(
        self,
        rendered_sources: list[RenderedSource],
        environment: Environment,
        receiver: Receiver,
        axis_t: Array,
    ) -> list[PropagatedSource]:
        """描画済み音源を伝搬経路へ変換する。

        Args:
            rendered_sources: 描画済み音源リスト。各 signal の shape は [n_sample]。
            environment: 伝搬環境。音速 c の単位は m/s。
            receiver: 受波器定義。trajectory から評価時刻の位置を得る。
            axis_t: 時間軸。shape は [n_sample]、単位は s。

        Returns:
            伝搬済み音源リスト。各 PropagationPath は direction_world shape [3] と delay [s] を持つ。

        Raises:
            実装クラスに依存する。抽象メソッド自体は NotImplementedError を送出する。
        """

        raise NotImplementedError
