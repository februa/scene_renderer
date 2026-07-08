from __future__ import annotations

from dataclasses import dataclass

from scene_renderer.scene.trajectory import Trajectory

from .array import ArrayGeometry


@dataclass(frozen=True)
class Receiver:
    """受波器の軌跡とアレイ形状を保持する定義オブジェクト。

    このクラスは、Trajectory と ArrayGeometry を束ね、任意時刻の受波器姿勢と CH 配置を
    SceneRenderer へ渡す。array.positions() の shape は [n_ch, 3]、単位は m である。

    信号生成、伝搬、ビームフォーミング、受信信号の保持は責務に含めない。
    信号処理上は、Scene と対になる受波器側の幾何条件を表す入力である。
    """

    trajectory: Trajectory
    array: ArrayGeometry
