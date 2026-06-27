from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import ArrayLike


def _as_vec3(value: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}")
    return arr


def rotation_array_to_world_from_heading(heading_deg: float) -> np.ndarray:
    """ArrayFrame から WorldFrame への回転行列を返す。

    WorldFrame: X=East, Y=North, Z=Up
    ArrayFrame: X=Bow, Y=Starboard, Z=Up
    heading_deg: 0=North, 90=East
    """
    h = np.deg2rad(heading_deg)
    x_axis_world = np.array([np.sin(h), np.cos(h), 0.0], dtype=float)
    y_axis_world = np.array([np.cos(h), -np.sin(h), 0.0], dtype=float)
    z_axis_world = np.array([0.0, 0.0, 1.0], dtype=float)
    return np.column_stack([x_axis_world, y_axis_world, z_axis_world])


@dataclass(frozen=True)
class Pose:
    """WorldFrame 上の位置と姿勢を表す値オブジェクト。"""

    position_world: ArrayLike
    heading_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_world", _as_vec3(self.position_world, "position_world"))

    def rotation_array_to_world(self) -> np.ndarray:
        return rotation_array_to_world_from_heading(self.heading_deg)

    def rotation_world_to_array(self) -> np.ndarray:
        return self.rotation_array_to_world().T

    def array_vector_to_world(self, v_array: ArrayLike) -> np.ndarray:
        return self.rotation_array_to_world() @ _as_vec3(v_array, "v_array")

    def world_vector_to_array(self, v_world: ArrayLike) -> np.ndarray:
        return self.rotation_world_to_array() @ _as_vec3(v_world, "v_world")


class Trajectory(ABC):
    """時刻から Pose を返す軌跡インターフェース。"""

    @abstractmethod
    def pose(self, t: float) -> Pose:
        raise NotImplementedError

    def position(self, t: float) -> np.ndarray:
        return self.pose(t).position_world

    def velocity(self, t: float) -> np.ndarray:
        return np.zeros(3, dtype=float)


@dataclass(frozen=True)
class StaticPose(Trajectory):
    """時刻によらず同一の Pose を返す固定軌跡。"""

    position_world: ArrayLike
    heading_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_world", _as_vec3(self.position_world, "position_world"))

    def pose(self, t: float) -> Pose:
        del t
        return Pose(
            position_world=self.position_world,
            heading_deg=self.heading_deg,
            pitch_deg=self.pitch_deg,
            roll_deg=self.roll_deg,
        )


@dataclass(frozen=True)
class StaticPosition(StaticPose):
    """姿勢が不要な対象を固定位置として表す。"""

    def __init__(self, position_world: ArrayLike):
        # 音源は最小構成では姿勢を使わないため、向きは明示的に捨てる。
        super().__init__(position_world=position_world, heading_deg=0.0, pitch_deg=0.0, roll_deg=0.0)
