from __future__ import annotations

from dataclasses import dataclass
from scene_renderer.scene.trajectory import Trajectory
from .array import ArrayGeometry


@dataclass(frozen=True)
class Receiver:
    """受波器の軌跡とアレイ形状を保持する定義オブジェクト。"""

    trajectory: Trajectory
    array: ArrayGeometry
