from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from scene_renderer.receiver import Receiver
from scene_renderer.scene import Environment

from .propagation import PropagationPath


Array: TypeAlias = NDArray[Any]
_DEFAULT_FRACTIONAL_DELAY_FILTER_LENGTH = 65


class SourceProjector(ABC):
    """伝搬経路をアレイ観測信号へ投影する抽象インターフェース。

    このクラスは、PropagationPath の信号と方向を受け取り、多CH信号 x[ch, t] を返す契約を定義する。
    戻り値 shape は [n_ch, n_sample]、axis=0 は CH、axis=1 は時間サンプルである。

    音源波形の生成、伝搬経路の探索、contributor 間の加算は責務に含めない。
    信号処理上は、幾何遅延を CH 間位相差または小数遅延として受信信号に反映する段である。
    """

    def required_sample_margin(self, receiver: Receiver, environment: Environment, fs: float) -> int:
        """この projector が入力 signal の前後に要求する halo サンプル数を返す。

        Args:
            receiver: 受波器定義。array.positions() は shape [n_ch, 3]、単位 m。
            environment: 伝搬環境。音速 c の単位は m/s。
            fs: サンプリング周波数。単位は Hz。

        Returns:
            必要 halo。単位は sample。既定は 0。

        Raises:
            なし。
        """

        del receiver, environment, fs
        return 0

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
            paths: 伝搬経路リスト。各 path.signal は center_slice の前後 halo を含みうる。
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


class PlaneWaveProjector(SourceProjector):
    """平面波近似で音源信号を各 CH に投影する。

    このクラスは、各経路の到来方向 direction_world を受波器 ArrayFrame に変換し、
    tau[ch] = r[ch] dot direction_array / c を計算する。Tone 経路では exp(-j 2π f tau[ch]) を掛け、
    広帯域経路では windowed-sinc fractional delay FIR を適用する。

    球面波の距離差、絶対伝搬遅延の時間シフト、streaming 状態管理は責務に含めない。
    信号処理上は、平面波の CH 間相対遅延を受信信号 x[ch, t] へ反映する投影器である。
    """

    def __init__(self, fractional_delay_filter_length: int = _DEFAULT_FRACTIONAL_DELAY_FILTER_LENGTH) -> None:
        """平面波 projector を作成する。

        Args:
            fractional_delay_filter_length: 広帯域小数遅延に使う FIR 長。奇数、3 以上。

        Returns:
            なし。

        Raises:
            ValueError: fractional_delay_filter_length が奇数 3 以上でない場合。
        """

        if fractional_delay_filter_length < 3 or fractional_delay_filter_length % 2 == 0:
            raise ValueError("fractional_delay_filter_length must be an odd integer greater than or equal to 3")
        self.fractional_delay_filter_length = fractional_delay_filter_length

    def required_sample_margin(self, receiver: Receiver, environment: Environment, fs: float) -> int:
        """広帯域小数遅延 FIR に必要な halo サンプル数を返す。

        Args:
            receiver: 受波器定義。array.positions() は shape [n_ch, 3]、単位 m。
            environment: 伝搬環境。音速 c の単位は m/s。
            fs: サンプリング周波数。単位は Hz。

        Returns:
            前後に必要な halo。単位は sample。

        Raises:
            ValueError: fs が 0 以下の場合。
        """

        if fs <= 0.0:
            raise ValueError("fs must be positive")
        positions = np.asarray(receiver.array.positions(), dtype=float)
        if positions.shape[0] <= 1:
            max_aperture_m = 0.0
        else:
            # 任意到来方向での最大 CH 間遅延は、素子間距離の最大値 / c で上から抑えられる。
            pairwise = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            max_aperture_m = float(np.max(np.linalg.norm(pairwise, axis=2)))
        max_delay_samples = int(np.ceil(max_aperture_m / environment.c * fs))
        return self.fractional_delay_filter_length // 2 + max_delay_samples

    def project(
        self,
        paths: list[PropagationPath],
        receiver: Receiver,
        environment: Environment,
        axis_t: Array,
        fs: float,
    ) -> Array:
        """平面波の CH 間相対遅延を反映した多CH信号を返す。

        Args:
            paths: 伝搬経路リスト。frequency が None でない場合は tone 高速経路、None の場合は広帯域経路。
            receiver: 受波器定義。array.positions() の shape は [n_ch, 3]、単位 m。
            environment: 伝搬環境。音速 c の単位は m/s。
            axis_t: 時間軸。shape は [n_sample]、単位は s。
            fs: サンプリング周波数。単位は Hz。

        Returns:
            多CH複素信号。shape は [n_ch, n_sample]、dtype は complex64。

        Raises:
            ValueError: path.signal や center_slice が axis_t と整合しない場合。
        """

        if fs <= 0.0:
            raise ValueError("fs must be positive")
        axis_t_array = np.asarray(axis_t, dtype=float)
        n_sample = axis_t_array.size
        element_pos_array = np.asarray(receiver.array.positions(), dtype=float)
        n_ch = element_pos_array.shape[0]
        out = np.zeros((n_ch, n_sample), dtype=np.complex64)

        t0 = float(axis_t_array[0]) if n_sample else 0.0
        receiver_pose = receiver.trajectory.pose(t0)

        for path in paths:
            path_signal = np.asarray(path.signal, dtype=np.complex64)
            _validate_center_slice(path.center_slice, path_signal.size, n_sample)
            # 伝搬側は絶対座標までを担当し、投影側で受波器姿勢を用いて ArrayFrame の到来方向へ変換する。
            direction_array = receiver_pose.world_vector_to_array(path.direction_world)
            # tau[ch] = r[ch] dot u / c。r は [m]、c は [m/s] なので tau は CH ごとの相対遅延 [s]。
            tau = (element_pos_array @ direction_array) / environment.c
            if path.frequency is not None:
                out += path.gain * self._project_narrowband(path_signal, path.center_slice, tau, float(path.frequency))
                continue
            out += path.gain * self._project_broadband(path_signal, path.center_slice, tau, fs)
        return out

    def _project_narrowband(self, signal: Array, center_slice: slice, tau: Array, frequency: float) -> Array:
        """tone 信号へ CH ごとの位相回転を掛ける。

        Args:
            signal: halo 付きまたは中心のみの tone 信号。shape は [n_signal]。
            center_slice: 元 chunk に対応する slice。
            tau: CH ごとの相対遅延。shape は [n_ch]、単位は s。
            frequency: tone 周波数。単位は Hz。

        Returns:
            多CH tone 信号。shape は [n_ch, n_sample]。

        Raises:
            なし。shape は呼び出し元で検証済み。
        """

        center_signal = np.asarray(signal[center_slice], dtype=np.complex64)
        # 狭帯域 tone では時間領域の小数遅延を、周波数 f における位相回転 exp(-j 2π f tau) で表す。
        phase = np.exp(-1j * 2.0 * np.pi * frequency * tau).astype(np.complex64)
        # phase[:, np.newaxis] は [n_ch, 1]、center_signal[np.newaxis, :] は [1, n_sample]。
        return phase[:, np.newaxis] * center_signal[np.newaxis, :]

    def _project_broadband(self, signal: Array, center_slice: slice, tau: Array, fs: float) -> Array:
        """広帯域信号へ CH ごとの fractional delay FIR を適用する。

        Args:
            signal: halo 付き広帯域信号。shape は [n_signal]。
            center_slice: 元 chunk に対応する slice。
            tau: CH ごとの相対遅延。shape は [n_ch]、単位は s。
            fs: サンプリング周波数。単位は Hz。

        Returns:
            多CH広帯域信号。shape は [n_ch, n_sample]。

        Raises:
            ValueError: signal の halo が不足している場合。
        """

        # 共通遅延は scene_renderer の責務外なので、min(tau) を基準にして CH 間相対遅延だけを非負遅延として表す。
        delay_samples = (tau - float(np.min(tau))) * fs
        center_start = _slice_start(center_slice)
        center_stop = _slice_stop(center_slice)
        center_indices = np.arange(center_start, center_stop, dtype=np.int64)
        out = np.zeros((delay_samples.size, center_indices.size), dtype=np.complex64)
        for ch, delay_sample in enumerate(delay_samples):
            out[ch, :] = _apply_fractional_delay_fir(
                signal=signal,
                center_indices=center_indices,
                delay_samples=float(delay_sample),
                filter_length=self.fractional_delay_filter_length,
            )
        return out


class NarrowbandPlaneWaveProjector(PlaneWaveProjector):
    """後方互換のために残す旧名称の平面波 projector。

    このクラスは PlaneWaveProjector と同じ実装を使う。新規コードでは PlaneWaveProjector を使う。

    旧名称の維持以外の責務は持たない。
    信号処理上の位置づけは PlaneWaveProjector と同じである。
    """


def _validate_center_slice(center_slice: slice, signal_size: int, n_sample: int) -> None:
    """center_slice が signal 内の n_sample 区間を表すことを検証する。

    Args:
        center_slice: 元 chunk を表す slice。
        signal_size: signal のサンプル数。
        n_sample: 元 chunk のサンプル数。

    Returns:
        なし。

    Raises:
        ValueError: slice が閉区間として解釈できない、または n_sample と一致しない場合。
    """

    start = _slice_start(center_slice)
    stop = _slice_stop(center_slice)
    if start < 0 or stop > signal_size or stop < start:
        raise ValueError("center_slice must be inside path.signal")
    if stop - start != n_sample:
        raise ValueError(f"center_slice length must be {n_sample}, got {stop - start}")


def _slice_start(value: slice) -> int:
    if value.start is None:
        return 0
    return int(value.start)


def _slice_stop(value: slice) -> int:
    if value.stop is None:
        raise ValueError("center_slice.stop must not be None")
    return int(value.stop)


def _apply_fractional_delay_fir(signal: Array, center_indices: Array, delay_samples: float, filter_length: int) -> Array:
    """windowed-sinc FIR で非負の小数遅延を適用する。

    Args:
        signal: halo 付き入力信号。shape は [n_signal]。
        center_indices: 出力対象の signal index。shape は [n_sample]。
        delay_samples: 遅延量。単位は sample。0 以上。
        filter_length: FIR 長。奇数。

    Returns:
        遅延後信号。shape は [n_sample]。

    Raises:
        ValueError: 遅延量が負、または halo が不足している場合。
    """

    if delay_samples < 0.0:
        raise ValueError("delay_samples must be non-negative")
    integer_delay = int(np.floor(delay_samples))
    fractional_delay = float(delay_samples - integer_delay)
    half = filter_length // 2
    p = np.arange(-half, half + 1, dtype=np.int64)
    # y[n] = sum_p x[n - integer_delay - p] sinc(p - fractional_delay)。p 軸は FIR tap を表す。
    kernel = np.sinc(p.astype(float) - fractional_delay) * np.hanning(filter_length)
    kernel_sum = float(np.sum(kernel))
    if abs(kernel_sum) <= 0.0:
        raise ValueError("fractional delay kernel has zero sum")
    kernel = kernel / kernel_sum
    sample_indices = center_indices[:, np.newaxis] - integer_delay - p[np.newaxis, :]
    if bool(np.any(sample_indices < 0)) or bool(np.any(sample_indices >= np.asarray(signal).size)):
        raise ValueError("path.signal does not contain enough halo for fractional delay")
    delayed = np.asarray(signal, dtype=np.complex64)[sample_indices] @ kernel.astype(np.float32)
    return np.asarray(delayed, dtype=np.complex64)
