from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray


Array: TypeAlias = NDArray[Any]


class Spectrum(ABC):
    """周波数軸に対する信号成分の複素スペクトルを返す抽象インターフェース。

    このクラスは、freq_axis shape [n_bin]、単位 Hz を入力として、同じ shape の複素振幅を
    返す契約を定義する。

    時間波形の生成、包絡の適用、アレイ投影は責務に含めない。
    信号処理上は、周波数領域レンダリングへ拡張する際の音源スペクトル定義である。
    """

    @abstractmethod
    def evaluate(self, freq_axis: Array) -> Array:
        """周波数軸上でスペクトルを評価する。

        Args:
            freq_axis: 周波数軸。shape は [n_bin]、axis=0 は周波数ビン、単位は Hz。

        Returns:
            複素スペクトル。shape は [n_bin]、axis=0 は freq_axis と同じ周波数ビン。

        Raises:
            実装クラスに依存する。抽象メソッド自体は NotImplementedError を送出する。
        """

        raise NotImplementedError


class NoiseSpectrum(Spectrum, ABC):
    """決定論的白色系列を整形するためのノイズ用スペクトル。

    このクラスは、SourceRenderer が生成した分散 1 の白色系列に対して、FIR 設計用の
    目標振幅応答 abs(evaluate(freq_axis)) を与える契約を定義する。

    乱数生成、FIR 係数生成、時間波形への畳み込みは責務に含めない。
    信号処理上は、広帯域ノイズの帯域・色付けを周波数軸上の振幅特性として表す。
    """


@dataclass(frozen=True)
class ToneSpectrum(Spectrum):
    """単一周波数 tone を表すスペクトル実装。

    このクラスは、frequency [Hz] に最も近い周波数ビンへ複素振幅 1 を置き、それ以外を 0 とする
    離散スペクトルを返す。

    時間包絡、振幅スケーリング、複数 tone の合成は責務に含めない。
    信号処理上は、狭帯域平面波投影で使う単一周波数 f を保持する音源スペクトルである。
    """

    frequency: float

    def evaluate(self, freq_axis: Array) -> Array:
        """frequency に最も近いビンだけが 1 となる複素スペクトルを返す。

        Args:
            freq_axis: 周波数軸。shape は [n_bin]、axis=0 は周波数ビン、単位は Hz。

        Returns:
            複素スペクトル。shape は [n_bin]。空の freq_axis では空配列を返す。

        Raises:
            ValueError: freq_axis が 1 次元でない場合。
        """

        freq_axis_array = np.asarray(freq_axis, dtype=float)
        if freq_axis_array.ndim != 1:
            raise ValueError(f"freq_axis must be 1-D, got shape {freq_axis_array.shape}")
        out = np.zeros_like(freq_axis_array, dtype=complex)
        if freq_axis_array.size == 0:
            # 空の周波数軸は、上位の探索処理で候補がないケースを安全に扱えるよう空スペクトルで返す。
            return out
        # argmin は周波数ビン軸 axis=0 上で最近傍ビンを選ぶ。bin 幅内の補間はこのクラスの責務外とする。
        idx = int(np.argmin(np.abs(freq_axis_array - self.frequency)))
        out[idx] = 1.0 + 0.0j
        return out


@dataclass(frozen=True)
class BandLimitedNoiseSpectrum(NoiseSpectrum):
    """指定帯域内で平坦な広帯域ノイズ用スペクトル。

    このクラスは、f_low_hz から f_high_hz までの周波数ビンに振幅 1 を与え、それ以外を 0 とする
    目標振幅応答を返す。

    白色系列生成、FIR 設計、時間波形の RMS 正規化は責務に含めない。
    信号処理上は、SourceRenderer が分散 1 の白色系列を帯域制限するための通過帯域を定義する。
    """

    f_low_hz: float
    f_high_hz: float

    def __post_init__(self) -> None:
        """帯域端の物理的な順序を検証する。

        Args:
            なし。dataclass の保持値を検証する。

        Returns:
            なし。

        Raises:
            ValueError: f_low_hz が負、または f_high_hz が f_low_hz 以下の場合。
        """

        if self.f_low_hz < 0.0:
            raise ValueError("f_low_hz must be non-negative")
        if self.f_high_hz <= self.f_low_hz:
            raise ValueError("f_high_hz must be greater than f_low_hz")

    def evaluate(self, freq_axis: Array) -> Array:
        """平坦な帯域制限ノイズの目標振幅応答を返す。

        Args:
            freq_axis: 周波数軸。shape は [n_bin]、axis=0 は周波数ビン、単位は Hz。

        Returns:
            複素振幅応答。shape は [n_bin]。通過帯域は 1、それ以外は 0。

        Raises:
            ValueError: freq_axis が 1 次元でない場合。
        """

        freq_axis_array = np.asarray(freq_axis, dtype=float)
        if freq_axis_array.ndim != 1:
            raise ValueError(f"freq_axis must be 1-D, got shape {freq_axis_array.shape}")
        # passband mask は FIR 設計で abs(H[f]) として使うため、ここでは位相を持たない実振幅だけを返す。
        mask = (self.f_low_hz <= freq_axis_array) & (freq_axis_array <= self.f_high_hz)
        return np.asarray(mask, dtype=complex)


@dataclass(frozen=True)
class PinkNoiseSpectrum(NoiseSpectrum):
    """PSD が 1/f に比例するピンクノイズ用スペクトル。

    このクラスは、f_low_hz から f_high_hz までの帯域で、パワースペクトル密度が 1/f になるよう
    振幅応答 1/sqrt(f) を返す。

    白色系列生成、FIR 設計、時間波形の RMS 正規化は責務に含めない。
    信号処理上は、SourceRenderer が白色系列をピンクノイズへ色付けするための振幅特性を定義する。
    """

    f_low_hz: float
    f_high_hz: float
    reference_frequency_hz: float = 1.0

    def __post_init__(self) -> None:
        """ピンクノイズ帯域と参照周波数を検証する。

        Args:
            なし。dataclass の保持値を検証する。

        Returns:
            なし。

        Raises:
            ValueError: 周波数範囲または参照周波数が不正な場合。
        """

        if self.f_low_hz <= 0.0:
            raise ValueError("f_low_hz must be positive for pink noise")
        if self.f_high_hz <= self.f_low_hz:
            raise ValueError("f_high_hz must be greater than f_low_hz")
        if self.reference_frequency_hz <= 0.0:
            raise ValueError("reference_frequency_hz must be positive")

    def evaluate(self, freq_axis: Array) -> Array:
        """PSD 1/f に対応する振幅応答 1/sqrt(f) を返す。

        Args:
            freq_axis: 周波数軸。shape は [n_bin]、axis=0 は周波数ビン、単位は Hz。

        Returns:
            複素振幅応答。shape は [n_bin]。帯域外は 0。

        Raises:
            ValueError: freq_axis が 1 次元でない場合。
        """

        freq_axis_array = np.asarray(freq_axis, dtype=float)
        if freq_axis_array.ndim != 1:
            raise ValueError(f"freq_axis must be 1-D, got shape {freq_axis_array.shape}")
        magnitude = np.zeros_like(freq_axis_array, dtype=float)
        mask = (self.f_low_hz <= freq_axis_array) & (freq_axis_array <= self.f_high_hz)
        # ピンクノイズは PSD が 1/f なので、FIR 設計に渡す振幅応答は sqrt(1/f) になる。
        magnitude[mask] = np.sqrt(self.reference_frequency_hz / freq_axis_array[mask])
        return magnitude.astype(complex)


@dataclass(frozen=True)
class CustomNoiseSpectrum(NoiseSpectrum):
    """テーブル指定の任意振幅応答を持つノイズ用スペクトル。

    このクラスは、frequencies_hz と amplitudes で与えた非負の振幅応答を線形補間して返す。

    白色系列生成、FIR 設計、時間波形の RMS 正規化は責務に含めない。
    信号処理上は、帯域制限やピンクノイズ以外の任意の広帯域ノイズ色付けを表す。
    """

    frequencies_hz: ArrayLike
    amplitudes: ArrayLike

    def __post_init__(self) -> None:
        """補間テーブルの shape と単調性を検証する。

        Args:
            なし。dataclass の保持値を検証する。

        Returns:
            なし。

        Raises:
            ValueError: 周波数または振幅の shape、単調性、非負性が不正な場合。
        """

        frequencies = np.asarray(self.frequencies_hz, dtype=float)
        amplitudes = np.asarray(self.amplitudes, dtype=float)
        if frequencies.ndim != 1 or amplitudes.ndim != 1:
            raise ValueError("frequencies_hz and amplitudes must be 1-D")
        if frequencies.shape != amplitudes.shape:
            raise ValueError("frequencies_hz and amplitudes must have the same shape")
        if frequencies.size < 2:
            raise ValueError("at least two frequency points are required")
        if bool(np.any(frequencies < 0.0)):
            raise ValueError("frequencies_hz must be non-negative")
        if bool(np.any(np.diff(frequencies) <= 0.0)):
            raise ValueError("frequencies_hz must be strictly increasing")
        if bool(np.any(amplitudes < 0.0)):
            raise ValueError("amplitudes must be non-negative")
        object.__setattr__(self, "frequencies_hz", frequencies)
        object.__setattr__(self, "amplitudes", amplitudes)

    def evaluate(self, freq_axis: Array) -> Array:
        """テーブル指定の振幅応答を周波数軸へ補間する。

        Args:
            freq_axis: 周波数軸。shape は [n_bin]、axis=0 は周波数ビン、単位は Hz。

        Returns:
            複素振幅応答。shape は [n_bin]。テーブル範囲外は 0。

        Raises:
            ValueError: freq_axis が 1 次元でない場合。
        """

        freq_axis_array = np.asarray(freq_axis, dtype=float)
        if freq_axis_array.ndim != 1:
            raise ValueError(f"freq_axis must be 1-D, got shape {freq_axis_array.shape}")
        frequencies = np.asarray(self.frequencies_hz, dtype=float)
        amplitudes = np.asarray(self.amplitudes, dtype=float)
        # np.interp は範囲外を端点値で埋める既定なので、帯域外を明示的に 0 にする。
        magnitude = np.interp(freq_axis_array, frequencies, amplitudes, left=0.0, right=0.0)
        return magnitude.astype(complex)
