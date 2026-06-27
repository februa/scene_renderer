from __future__ import annotations

from dataclasses import dataclass
from numpy.typing import ArrayLike
import numpy as np

from .trajectory import Pose, StaticPosition, Trajectory
from .spectrum import Spectrum
from .envelope import Envelope


# 絶対方位は航法系の定義をそのまま WorldFrame に写像する。
def unit_vector_from_absolute_bearing(bearing_deg: float, elevation_deg: float = 0.0) -> np.ndarray:
    az = np.deg2rad(bearing_deg)
    el = np.deg2rad(elevation_deg)
    return np.array([
        np.cos(el) * np.sin(az),
        np.cos(el) * np.cos(az),
        np.sin(el),
    ], dtype=float)


# 相対方位はまず ArrayFrame で定義し、後段で WorldFrame へ回転する。
def unit_vector_from_relative_bearing(bearing_deg: float, elevation_deg: float = 0.0) -> np.ndarray:
    az = np.deg2rad(bearing_deg)
    el = np.deg2rad(elevation_deg)
    return np.array([
        np.cos(el) * np.cos(az),
        np.cos(el) * np.sin(az),
        np.sin(el),
    ], dtype=float)


@dataclass(frozen=True)
class SourceComponent:
    """一つの音源成分のスペクトラム・包絡・振幅指定を持つ。"""

    spectrum: Spectrum
    envelope: Envelope
    amplitude: float | None = 1.0
    level_db: float | None = None

    def __post_init__(self) -> None:
        if self.amplitude is not None and self.level_db is not None:
            raise ValueError("Specify either amplitude or level_db, not both")
        if self.amplitude is not None and self.amplitude < 0:
            raise ValueError("amplitude must be non-negative")

    @property
    def amplitude_value(self) -> float:
        # 描画側は指定方法に依らず線形振幅だけを見ればよいようにする。
        if self.level_db is not None:
            return float(10 ** (self.level_db / 20.0))
        if self.amplitude is None:
            return 1.0
        return float(self.amplitude)

    @classmethod
    def from_amplitude(
        cls,
        spectrum: Spectrum,
        envelope: Envelope,
        amplitude: float = 1.0,
    ) -> "SourceComponent":
        return cls(spectrum=spectrum, envelope=envelope, amplitude=amplitude)

    @classmethod
    def from_level_db20(
        cls,
        spectrum: Spectrum,
        envelope: Envelope,
        level_db: float,
    ) -> "SourceComponent":
        return cls(spectrum=spectrum, envelope=envelope, amplitude=None, level_db=level_db)


@dataclass(frozen=True)
class AcousticSource:
    """空間上の位置軌跡と複数成分を持つ局所音源。"""

    trajectory: Trajectory
    components: list[SourceComponent]

    @classmethod
    def from_position(
        cls,
        position_world: ArrayLike,
        components: list[SourceComponent],
    ) -> "AcousticSource":
        return cls(trajectory=StaticPosition(position_world), components=components)

    @classmethod
    def from_absolute_bearing(
        cls,
        bearing_deg: float,
        distance: float,
        receiver_pose: Pose,
        components: list[SourceComponent],
        elevation_deg: float = 0.0,
    ) -> "AcousticSource":
        if distance <= 0:
            raise ValueError("distance must be positive")
        direction_world = unit_vector_from_absolute_bearing(bearing_deg, elevation_deg)
        position_world = receiver_pose.position_world + distance * direction_world
        return cls.from_position(position_world, components)

    @classmethod
    def from_relative_bearing(
        cls,
        bearing_deg: float,
        distance: float,
        receiver_pose: Pose,
        components: list[SourceComponent],
        elevation_deg: float = 0.0,
    ) -> "AcousticSource":
        if distance <= 0:
            raise ValueError("distance must be positive")
        direction_array = unit_vector_from_relative_bearing(bearing_deg, elevation_deg)
        # 利用側の指定方法に関わらず、内部保持は WorldFrame に統一する。
        direction_world = receiver_pose.array_vector_to_world(direction_array)
        position_world = receiver_pose.position_world + distance * direction_world
        return cls.from_position(position_world, components)
