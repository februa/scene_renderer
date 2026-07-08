from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

from numpy.typing import NDArray

from scene_renderer.renderer.source_renderer import RenderedSource


Array: TypeAlias = NDArray[Any]


@dataclass(frozen=True)
class PropagationPath:
    """単一路の伝搬結果を保持する経路表現。

    このクラスは、伝搬後も音源基準の signal、元 chunk を表す center_slice、到来方向 direction_world、
    絶対遅延 delay、経路 gain を保持する。signal は halo を含みうる。

    アレイ CH への投影、複数経路の合成、時間シフトの実適用は責務に含めない。
    信号処理上は、PropagationModel と SourceProjector の間で幾何・経路情報を渡す中間表現である。
    """

    signal: Array
    center_slice: slice
    direction_world: Array
    delay: float
    gain: float
    path_type: str
    virtual_source_pos_world: Array | None = None
    frequency: float | None = None


@dataclass(frozen=True)
class PropagatedSource:
    """一つの描画済み音源に対する伝搬経路群を束ねる。

    このクラスは、RenderedSource と、その音源から受波器へ届く PropagationPath リストを対応づける。
    paths の各 signal は同じ時間軸と center_slice を前提とする。

    projector の選択、経路ごとのアレイ投影、Scene 全体の合成は責務に含めない。
    信号処理上は、音源単位で伝搬幾何をまとめ、ArrayProjector が扱いやすい単位にする表現である。
    """

    rendered_source: RenderedSource
    paths: list[PropagationPath]
