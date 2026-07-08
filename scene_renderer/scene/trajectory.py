from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray


VectorArray: TypeAlias = NDArray[Any]


def _as_vec3(value: ArrayLike, name: str) -> VectorArray:
    """3 次元ベクトル入力を float64 の shape [3] に正規化する。

    Args:
        value: WorldFrame または ArrayFrame 上の 3 成分ベクトル。shape は [3]、単位は呼び出し側の物理量に従う。
        name: 例外メッセージに使う引数名。

    Returns:
        float64 配列。shape は [3]。

    Raises:
        ValueError: value が 3 成分ベクトルとして解釈できない場合。
    """

    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}")
    return arr


def rotation_array_to_world_from_heading(heading_deg: float) -> VectorArray:
    """ArrayFrame から WorldFrame への水平回転行列を返す。

    Args:
        heading_deg: 艦首方位。単位は deg。0 deg は北、90 deg は東を表す。

    Returns:
        回転行列。shape は [3, 3]。列 0/1/2 は ArrayFrame の X/Y/Z 軸を
        WorldFrame の East/North/Up 成分で表した単位ベクトル。

    Raises:
        なし。NaN や inf は NumPy の三角関数結果として伝搬させ、上位の入力検証に委ねる。
    """

    h = np.deg2rad(heading_deg)
    # ArrayFrame X 軸は艦首方向であり、航法方位 heading を WorldFrame の East/North 成分へ写す。
    x_axis_world = np.array([np.sin(h), np.cos(h), 0.0], dtype=float)
    # ArrayFrame Y 軸は右舷方向であり、艦首方向から時計回り 90 deg の水平単位ベクトルになる。
    y_axis_world = np.array([np.cos(h), -np.sin(h), 0.0], dtype=float)
    # 最小構成では pitch/roll を回転に含めないため、鉛直軸は WorldFrame と共有する。
    z_axis_world = np.array([0.0, 0.0, 1.0], dtype=float)
    # column_stack の出力 shape は [world_axis=3, array_axis=3]。
    # v_world = R @ v_array として、ArrayFrame 成分を WorldFrame 成分へ変換する。
    return np.column_stack([x_axis_world, y_axis_world, z_axis_world])


@dataclass(frozen=True)
class Pose:
    """WorldFrame 上の受波器または物体の位置と姿勢を表す。

    このクラスは、位置 position_world と heading/pitch/roll を保持し、WorldFrame と
    ArrayFrame の 3 次元ベクトル変換を提供する。入力は ArrayLike でも受け取り、
    内部では position_world を shape [3]、単位 m の float64 配列として扱う。

    信号波形の生成、伝搬経路の生成、アレイ素子配置の保持は責務に含めない。
    信号処理上は、音源方向を受波器固定座標へ変換するための幾何情報に位置づける。
    """

    position_world: ArrayLike
    heading_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0

    def __post_init__(self) -> None:
        # 位置ベクトルは以降の内積・差分計算で shape [3] を前提にするため、生成時点で固定する。
        object.__setattr__(self, "position_world", _as_vec3(self.position_world, "position_world"))

    def rotation_array_to_world(self) -> VectorArray:
        """ArrayFrame から WorldFrame への回転行列を返す。

        Args:
            なし。姿勢はこの Pose が保持する heading_deg を使う。

        Returns:
            回転行列。shape は [3, 3]。axis=0 は WorldFrame 成分、axis=1 は ArrayFrame 軸。

        Raises:
            なし。最小構成では pitch_deg/roll_deg は保持のみで、回転には使わない。
        """

        return rotation_array_to_world_from_heading(self.heading_deg)

    def rotation_world_to_array(self) -> VectorArray:
        """WorldFrame から ArrayFrame への回転行列を返す。

        Args:
            なし。

        Returns:
            回転行列。shape は [3, 3]。axis=0 は ArrayFrame 成分、axis=1 は WorldFrame 軸。

        Raises:
            なし。回転行列は正規直交であるため、逆行列ではなく転置で逆変換する。
        """

        # R_array_to_world は正規直交行列なので、数値的にも意味的にも転置が逆変換になる。
        return self.rotation_array_to_world().T

    def array_vector_to_world(self, v_array: ArrayLike) -> VectorArray:
        """ArrayFrame 上の 3 次元ベクトルを WorldFrame へ変換する。

        Args:
            v_array: ArrayFrame ベクトル。shape は [3]。位置差なら単位 m、方向なら無次元。

        Returns:
            WorldFrame ベクトル。shape は [3]。入力と同じ物理単位を保つ。

        Raises:
            ValueError: v_array が shape [3] でない場合。
        """

        # 行列積 [3, 3] @ [3] により、ArrayFrame の軸成分を WorldFrame の成分へ写す。
        return self.rotation_array_to_world() @ _as_vec3(v_array, "v_array")

    def world_vector_to_array(self, v_world: ArrayLike) -> VectorArray:
        """WorldFrame 上の 3 次元ベクトルを ArrayFrame へ変換する。

        Args:
            v_world: WorldFrame ベクトル。shape は [3]。位置差なら単位 m、方向なら無次元。

        Returns:
            ArrayFrame ベクトル。shape は [3]。入力と同じ物理単位を保つ。

        Raises:
            ValueError: v_world が shape [3] でない場合。
        """

        # アレイ投影では direction_world をこの変換で受波器基準の direction_array に揃える。
        return self.rotation_world_to_array() @ _as_vec3(v_world, "v_world")


class Trajectory(ABC):
    """時刻から Pose を返す軌跡インターフェース。

    このクラスは、任意時刻 t [s] における位置・姿勢を Pose として返す契約を定義する。
    出力 Pose の position_world は shape [3]、単位 m である。

    信号の生成、伝搬モデル、速度からのドップラー計算は責務に含めない。
    信号処理上は、レンダリング時刻における幾何条件を供給する入力境界である。
    """

    @abstractmethod
    def pose(self, t: float) -> Pose:
        """時刻 t における位置・姿勢を返す。

        Args:
            t: 評価時刻。単位は s。axis_t と同じ時間原点を前提とする。

        Returns:
            Pose。position_world の shape は [3]、単位は m。

        Raises:
            実装クラスに依存する。抽象メソッド自体は NotImplementedError を送出する。
        """

        raise NotImplementedError

    def position(self, t: float) -> VectorArray:
        """時刻 t における WorldFrame 位置を返す。

        Args:
            t: 評価時刻。単位は s。

        Returns:
            WorldFrame 位置。shape は [3]、単位は m。

        Raises:
            pose(t) の実装が送出する例外をそのまま伝搬する。
        """

        return _as_vec3(self.pose(t).position_world, "position_world")

    def velocity(self, t: float) -> VectorArray:
        """時刻 t における WorldFrame 速度を返す。

        Args:
            t: 評価時刻。単位は s。

        Returns:
            WorldFrame 速度。shape は [3]、単位は m/s。既定実装では静止を表すゼロベクトル。

        Raises:
            なし。既定実装はドップラーを扱わない静止近似を表す。
        """

        del t
        # 速度未定義の軌跡でも後段が shape [3] を仮定できるよう、静止を安全側の既定値にする。
        return np.zeros(3, dtype=float)


@dataclass(frozen=True)
class StaticPose(Trajectory):
    """時刻によらず同一の Pose を返す固定軌跡。

    このクラスは、固定された WorldFrame 位置と姿勢を入力として保持し、任意時刻 t [s]
    に対して同一 Pose を返す。position_world は shape [3]、単位 m に正規化される。

    移動、加速度、ドップラーを表すことは責務に含めない。
    信号処理上は、固定受波器や固定音源の幾何条件を与える軌跡である。
    """

    position_world: ArrayLike
    heading_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0

    def __post_init__(self) -> None:
        # StaticPose でも Pose と同じ shape 契約を保ち、呼び出しごとの再検証を避ける。
        object.__setattr__(self, "position_world", _as_vec3(self.position_world, "position_world"))

    def pose(self, t: float) -> Pose:
        """固定 Pose を返す。

        Args:
            t: 評価時刻。単位は s。固定軌跡なので値には依存しない。

        Returns:
            Pose。position_world の shape は [3]、単位は m。

        Raises:
            ValueError: 保持している position_world が shape [3] として不正な場合。
        """

        del t
        # 毎回 Pose として返すことで、Trajectory 利用側は固定・移動の違いを意識せず扱える。
        return Pose(
            position_world=self.position_world,
            heading_deg=self.heading_deg,
            pitch_deg=self.pitch_deg,
            roll_deg=self.roll_deg,
        )


@dataclass(frozen=True)
class StaticPosition(StaticPose):
    """姿勢を使わない対象を固定 WorldFrame 位置として表す。

    このクラスは、音源のように位置だけが必要な対象を shape [3]、単位 m の固定位置として保持する。
    出力 Pose の heading/pitch/roll は常に 0 deg である。

    受波器姿勢やアレイ座標変換を表すことは責務に含めない。
    信号処理上は、局所音源の幾何位置を Trajectory と同じインターフェースで供給するための薄い表現である。
    """

    def __init__(self, position_world: ArrayLike):
        """固定位置を作成する。

        Args:
            position_world: WorldFrame 位置。shape は [3]、単位は m。

        Returns:
            なし。dataclass の初期化によりインスタンスを構築する。

        Raises:
            ValueError: position_world が shape [3] でない場合。
        """

        # 音源は最小構成では姿勢を使わないため、向きは明示的に 0 deg へ固定する。
        super().__init__(position_world=position_world, heading_deg=0.0, pitch_deg=0.0, roll_deg=0.0)
