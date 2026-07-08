from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from scene_renderer.receiver import Receiver
from scene_renderer.scene import Environment

from .propagation import PropagationPath


Array: TypeAlias = NDArray[Any]


class SourceProjector(ABC):
    """伝搬経路をアレイ観測信号へ投影する抽象インターフェース。

    このクラスは、PropagationPath の信号と方向を受け取り、多CH信号 x[ch, t] を返す契約を定義する。
    戻り値 shape は [n_ch, n_sample]、axis=0 は CH、axis=1 は時間サンプルである。

    音源波形の生成、伝搬経路の探索、contributor 間の加算は責務に含めない。
    信号処理上は、幾何遅延を CH 間位相差として受信信号に反映する段である。
    """

    @abstractmethod
    def project(
        self,
        paths: list[PropagationPath],
        receiver: Receiver,
        environment: Environment,
        axis_t: Array,
        fs: float,
    ) -> Array:
        """伝搬経路を多CH信号へ投影する。

        Args:
            paths: 伝搬経路リスト。各 path.signal の shape は [n_sample]。
            receiver: 受波器定義。array.positions() は shape [n_ch, 3]、単位 m。
            environment: 伝搬環境。音速 c の単位は m/s。
            axis_t: 時間軸。shape は [n_sample]、単位は s。
            fs: サンプリング周波数。単位は Hz。

        Returns:
            多CH信号。shape は [n_ch, n_sample]。

        Raises:
            実装クラスに依存する。抽象メソッド自体は NotImplementedError を送出する。
        """

        raise NotImplementedError


class NarrowbandPlaneWaveProjector(SourceProjector):
    """平面波近似で tone 信号を各 CH に投影する。

    このクラスは、各経路の到来方向 direction_world を受波器 ArrayFrame に変換し、
    tau[ch] = r[ch] dot direction_array / c から位相 exp(-j 2π f tau[ch]) を掛ける。

    broadband 信号の小数遅延、球面波の距離差、絶対伝搬遅延の時間シフトは責務に含めない。
    信号処理上は、狭帯域ビームフォーミング評価に必要な CH 間位相差を生成する投影器である。
    """

    def project(
        self,
        paths: list[PropagationPath],
        receiver: Receiver,
        environment: Environment,
        axis_t: Array,
        fs: float,
    ) -> Array:
        """狭帯域平面波の CH 間位相差を反映した多CH信号を返す。

        Args:
            paths: 伝搬経路リスト。frequency が None でない tone 経路のみ対応する。
            receiver: 受波器定義。array.positions() の shape は [n_ch, 3]、単位 m。
            environment: 伝搬環境。音速 c の単位は m/s。
            axis_t: 時間軸。shape は [n_sample]、単位は s。
            fs: サンプリング周波数。単位は Hz。狭帯域投影では式に直接使わない。

        Returns:
            多CH複素信号。shape は [n_ch, n_sample]、dtype は complex64。

        Raises:
            NotImplementedError: path.frequency が None の場合。
            ValueError: path.signal が axis_t と同じサンプル数でない場合。
        """

        del fs
        axis_t_array = np.asarray(axis_t, dtype=float)
        n_sample = axis_t_array.size
        element_pos_array = np.asarray(receiver.array.positions(), dtype=float)
        n_ch = element_pos_array.shape[0]
        out = np.zeros((n_ch, n_sample), dtype=np.complex64)

        t0 = float(axis_t_array[0]) if n_sample else 0.0
        receiver_pose = receiver.trajectory.pose(t0)

        for path in paths:
            if path.frequency is None:
                raise NotImplementedError("NarrowbandPlaneWaveProjector supports tone paths only")
            path_signal = np.asarray(path.signal, dtype=np.complex64)
            if path_signal.shape != (n_sample,):
                raise ValueError(f"path.signal must have shape ({n_sample},), got {path_signal.shape}")
            # 伝搬側は絶対座標までを担当し、投影側で受波器姿勢を用いて ArrayFrame の到来方向へ変換する。
            direction_array = receiver_pose.world_vector_to_array(path.direction_world)
            # tau[ch] = r[ch] dot u / c。r は [m]、c は [m/s] なので tau は CH ごとの相対遅延 [s]。
            tau = (element_pos_array @ direction_array) / environment.c
            # 狭帯域 tone では時間領域の小数遅延を、周波数 f における位相回転 exp(-j 2π f tau) で表す。
            phase = np.exp(-1j * 2.0 * np.pi * float(path.frequency) * tau).astype(np.complex64)
            # phase[:, np.newaxis] は shape [n_ch, 1]、path_signal[np.newaxis, :] は [1, n_sample]。
            # broadcasting により [n_ch, n_sample] の CH 別 tone 信号へ展開して加算する。
            out += path.gain * phase[:, np.newaxis] * path_signal[np.newaxis, :]
        return out
