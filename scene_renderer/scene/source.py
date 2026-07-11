from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .envelope import Envelope
from .spectrum import NoiseSpectrum, Spectrum, ToneSpectrum
from .trajectory import Pose, StaticPosition, Trajectory
from scene_renderer.level import tone_rms_level_db_to_peak_amplitude


VectorArray: TypeAlias = NDArray[Any]


def unit_vector_from_absolute_bearing(bearing_deg: float, elevation_deg: float = 0.0) -> VectorArray:
    """絶対方位と仰角から WorldFrame 上の単位方向ベクトルを作る。

    Args:
        bearing_deg: 絶対方位。単位は deg。0 deg は北、90 deg は東を表す。
        elevation_deg: 仰角。単位は deg。水平面を 0、上向きを正とする。

    Returns:
        WorldFrame 方向ベクトル。shape は [3]、成分は [East, North, Up]、無次元。

    Raises:
        なし。角度の範囲は周期性を持つため正規化せず、NumPy の三角関数へ渡す。
    """

    az = np.deg2rad(bearing_deg)
    el = np.deg2rad(elevation_deg)
    # 航法方位は北を 0 deg とするため、WorldFrame の East 成分が sin、North 成分が cos になる。
    return np.array(
        [
            np.cos(el) * np.sin(az),
            np.cos(el) * np.cos(az),
            np.sin(el),
        ],
        dtype=float,
    )


def unit_vector_from_relative_bearing(bearing_deg: float, elevation_deg: float = 0.0) -> VectorArray:
    """相対方位と仰角から ArrayFrame 上の単位方向ベクトルを作る。

    Args:
        bearing_deg: 相対方位。単位は deg。0 deg は艦首、90 deg は右舷を表す。
        elevation_deg: 仰角。単位は deg。ArrayFrame の水平面を 0、上向きを正とする。

    Returns:
        ArrayFrame 方向ベクトル。shape は [3]、成分は [Bow, Starboard, Up]、無次元。

    Raises:
        なし。角度は周期量として扱うため範囲外も許容する。
    """

    az = np.deg2rad(bearing_deg)
    el = np.deg2rad(elevation_deg)
    # 相対方位は艦首を 0 deg とするため、ArrayFrame の Bow 成分が cos、Starboard 成分が sin になる。
    return np.array(
        [
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el),
        ],
        dtype=float,
    )


@dataclass(frozen=True)
class SourceComponent:
    """一つの音源成分のスペクトラム・包絡・振幅指定を保持する。

    このクラスは、周波数特性 Spectrum、時間包絡 Envelope、線形振幅または 20log10 の
    振幅 dB 指定を受け取り、SourceRenderer が使う実効振幅へ解決する。

    音源位置、受波器、伝搬、アレイ投影は責務に含めない。
    信号処理上は、基準音源信号 s(t) = A env(t) exp(j 2π f t) または
    広帯域ノイズ信号の A、seed、FIR 長を定義する。
    """

    spectrum: Spectrum
    envelope: Envelope
    amplitude: float | None = 1.0
    level_db: float | None = None
    noise_seed: int | None = None
    noise_filter_length: int = 257

    def __post_init__(self) -> None:
        # 線形振幅と dB 振幅を同時に受けると、どちらを正本にするか曖昧になるため API エラーにする。
        if self.amplitude is not None and self.level_db is not None:
            raise ValueError("Specify either amplitude or level_db, not both")
        if self.amplitude is not None and self.amplitude < 0:
            raise ValueError("amplitude must be non-negative")
        if self.noise_filter_length < 3 or self.noise_filter_length % 2 == 0:
            raise ValueError("noise_filter_length must be an odd integer greater than or equal to 3")
        if isinstance(self.spectrum, NoiseSpectrum) and self.noise_seed is None:
            # ノイズ波形は chunk 分割非依存にするため、呼び出し順に依存する RNG 状態ではなく seed を必須にする。
            raise ValueError("noise_seed is required for NoiseSpectrum")

    @property
    def amplitude_value(self) -> float:
        """音源信号に掛ける線形振幅を返す。

        Args:
            なし。amplitude または level_db はインスタンス生成時に保持済み。

        Returns:
            線形振幅 A。無次元。level_db 指定時は A = 10^(level_db / 20)。

        Raises:
            なし。排他条件と負値条件は __post_init__ で検証済み。
        """

        # dB は振幅比の 20log10 指定として扱う。絶対音圧基準ではないため単位は持たない。
        if self.level_db is not None:
            return float(10 ** (self.level_db / 20.0))
        # 両方 None は上位設定から省略指定された場合の既定振幅として 1.0 を採用する。
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
        """線形振幅指定で音源成分を作成する。

        Args:
            spectrum: 周波数特性。ToneSpectrum なら単位 Hz の tone 周波数を持つ。
            envelope: 時間包絡。入力 shape [n_sample] に対して同じ shape を返す契約。
            amplitude: 線形振幅。無次元。0 以上。

        Returns:
            SourceComponent。

        Raises:
            ValueError: amplitude が負の場合。
        """

        return cls(spectrum=spectrum, envelope=envelope, amplitude=amplitude)

    @classmethod
    def from_level_db20(
        cls,
        spectrum: Spectrum,
        envelope: Envelope,
        level_db: float,
    ) -> "SourceComponent":
        """20log10 の振幅 dB 指定で音源成分を作成する。

        Args:
            spectrum: 周波数特性。ToneSpectrum なら単位 Hz の tone 周波数を持つ。
            envelope: 時間包絡。入力 shape [n_sample] に対して同じ shape を返す契約。
            level_db: 振幅比 dB。単位は dB re amplitude 1.0。

        Returns:
            SourceComponent。

        Raises:
            なし。dB 指定は正負どちらも振幅比として有効である。
        """

        return cls(spectrum=spectrum, envelope=envelope, amplitude=None, level_db=level_db)


@dataclass(frozen=True)
class AcousticSource:
    """空間上の位置軌跡と複数の信号成分を持つ局所音源。

    このクラスは、Trajectory と SourceComponent のリストを保持し、直交座標・絶対方位・
    相対方位から WorldFrame 上の音源位置を生成する補助 API を提供する。

    受波器アレイ、伝搬損失、CH 間位相差、信号合成は責務に含めない。
    信号処理上は、SceneRenderer へ渡す局所音源の幾何条件と基準信号条件を束ねる定義である。
    """

    trajectory: Trajectory
    components: list[SourceComponent]
    identifier: str = "source"
    role: str = "source"

    def __post_init__(self) -> None:
        """音源を成分分離結果で識別するための文字列を検証する。"""

        if not self.identifier.strip():
            raise ValueError("identifier must not be empty")
        if not self.role.strip():
            raise ValueError("role must not be empty")

    @classmethod
    def from_position(
        cls,
        position_world: ArrayLike,
        components: list[SourceComponent],
        identifier: str = "source",
        role: str = "source",
    ) -> "AcousticSource":
        """WorldFrame 直交座標で固定音源を作成する。

        Args:
            position_world: 音源位置。shape は [3]、成分は [East, North, Up]、単位は m。
            components: 音源成分リスト。各成分は同じ位置から放射される。
            identifier: 成分分離結果で音源個体を識別する非空文字列。
            role: target、interferenceなど後段分類用の非空文字列。

        Returns:
            AcousticSource。内部 trajectory は StaticPosition。

        Raises:
            ValueError: position_world が shape [3] でない場合。
        """

        return cls(
            trajectory=StaticPosition(position_world),
            components=components,
            identifier=identifier,
            role=role,
        )

    @classmethod
    def from_absolute_bearing(
        cls,
        bearing_deg: float,
        distance: float,
        receiver_pose: Pose,
        components: list[SourceComponent],
        elevation_deg: float = 0.0,
        identifier: str = "source",
        role: str = "source",
    ) -> "AcousticSource":
        """受波器位置から絶対方位・距離で固定音源を作成する。

        Args:
            bearing_deg: WorldFrame の絶対方位。単位は deg。0 deg は北、90 deg は東。
            distance: 受波器から音源までの距離。単位は m。0 より大きい値。
            receiver_pose: 距離原点となる受波器 Pose。position_world は shape [3]、単位 m。
            components: 音源成分リスト。
            elevation_deg: 仰角。単位は deg。水平面を 0、上向きを正とする。
            identifier: 成分分離結果で音源個体を識別する非空文字列。
            role: target、interferenceなど後段分類用の非空文字列。

        Returns:
            AcousticSource。音源位置は receiver_pose.position_world + distance * direction_world。

        Raises:
            ValueError: distance が 0 以下、または生成位置が shape [3] として不正な場合。
        """

        if distance <= 0:
            raise ValueError("distance must be positive")
        direction_world = unit_vector_from_absolute_bearing(bearing_deg, elevation_deg)
        receiver_position_world = np.asarray(receiver_pose.position_world, dtype=float)
        # direction_world は無次元単位ベクトルなので、distance [m] を掛けて WorldFrame 位置差 [m] にする。
        position_world = receiver_position_world + distance * direction_world
        return cls.from_position(position_world, components, identifier=identifier, role=role)

    @classmethod
    def from_relative_bearing(
        cls,
        bearing_deg: float,
        distance: float,
        receiver_pose: Pose,
        components: list[SourceComponent],
        elevation_deg: float = 0.0,
        identifier: str = "source",
        role: str = "source",
    ) -> "AcousticSource":
        """受波器 ArrayFrame の相対方位・距離で固定音源を作成する。

        Args:
            bearing_deg: ArrayFrame の相対方位。単位は deg。0 deg は艦首、90 deg は右舷。
            distance: 受波器から音源までの距離。単位は m。0 より大きい値。
            receiver_pose: ArrayFrame から WorldFrame への変換を与える Pose。
            components: 音源成分リスト。
            elevation_deg: 仰角。単位は deg。ArrayFrame 水平面を 0、上向きを正とする。
            identifier: 成分分離結果で音源個体を識別する非空文字列。
            role: target、interferenceなど後段分類用の非空文字列。

        Returns:
            AcousticSource。内部位置は WorldFrame の shape [3]、単位 m。

        Raises:
            ValueError: distance が 0 以下、または生成位置が shape [3] として不正な場合。
        """

        if distance <= 0:
            raise ValueError("distance must be positive")
        direction_array = unit_vector_from_relative_bearing(bearing_deg, elevation_deg)
        # 利用側が相対方位で指定しても、伝搬モデルは WorldFrame だけを扱えるよう内部表現を統一する。
        direction_world = receiver_pose.array_vector_to_world(direction_array)
        receiver_position_world = np.asarray(receiver_pose.position_world, dtype=float)
        position_world = receiver_position_world + distance * direction_world
        return cls.from_position(position_world, components, identifier=identifier, role=role)


def tone_component_from_rms_level_db(
    frequency_hz: float,
    level_db_re_rms: float,
    envelope: Envelope,
) -> SourceComponent:
    """RMS levelを明示して実正弦波に対応するtone成分を作る。

    Args:
        frequency_hz: tone周波数。単位はHz。
        level_db_re_rms: tone RMS level。単位はdB re amplitude 1 RMS。
        envelope: 時間包絡。shape [n_sample]の時間軸を同じshapeへ写す。

    Returns:
        `SourceRenderer`の複素toneを実部化したとき、指定RMSとなる`SourceComponent`。

    Raises:
        ValueError: levelが有限でない場合。

    位置、伝搬、受波器はこの関数の責務に含めない。
    """

    return SourceComponent(
        spectrum=ToneSpectrum(frequency_hz),
        envelope=envelope,
        amplitude=tone_rms_level_db_to_peak_amplitude(level_db_re_rms),
    )
