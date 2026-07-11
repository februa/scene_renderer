"""sample indexで再現可能なスペクトル整形雑音波形を生成する。"""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from scene_renderer.scene import (
    BandLimitedNoiseSpectrum,
    CustomNoiseSpectrum,
    NoiseSpectrum,
    PinkNoiseSpectrum,
)


Array: TypeAlias = NDArray[Any]
_SAMPLE_INDEX_ATOL = 1.0e-6
_SPLITMIX64_INCREMENT = np.uint64(0x9E3779B97F4A7C15)
_SPLITMIX64_MUL_1 = np.uint64(0xBF58476D1CE4E5B9)
_SPLITMIX64_MUL_2 = np.uint64(0x94D049BB133111EB)
_UINT53_SCALE = float(2**53)


def sample_index_from_time(t0: float, fs: float) -> int:
    """絶対時刻から整数sample indexを復元する。

    Args:
        t0: chunk先頭時刻。単位はs。
        fs: サンプリング周波数。単位はHz。

    Returns:
        global sample index。

    Raises:
        ValueError: fsが0以下、またはt0がsample格子に乗らない場合。

    音源、背景場、アレイ投影は責務に含めない。雑音生成をchunk分割非依存にするための
    共通sample座標だけを定義する。
    """

    if fs <= 0.0:
        raise ValueError("fs must be positive")
    sample_index_float = t0 * fs
    sample_index = int(round(sample_index_float))
    # deterministic noiseはsample indexで値を決めるため、格子外時刻を曖昧に丸めない。
    if abs(sample_index_float - sample_index) > _SAMPLE_INDEX_ATOL:
        raise ValueError("axis_t[0] must align with an integer sample index for NoiseSpectrum")
    return sample_index


def render_indexed_noise(
    spectrum: NoiseSpectrum,
    start_sample_index: int,
    n_sample: int,
    fs: float,
    seed: int,
    filter_length: int,
) -> Array:
    """決定論的白色系列をFIRで整形して雑音波形を作る。

    Args:
        spectrum: FIRの目標振幅応答を与える雑音スペクトル。
        start_sample_index: 出力先頭のglobal sample index。
        n_sample: 出力sample数。
        fs: サンプリング周波数。単位はHz。
        seed: 白色系列を決める整数seed。
        filter_length: スペクトル整形FIR長。奇数かつ3以上。

    Returns:
        雑音波形。shapeは[n_sample]、期待RMSは1。

    Raises:
        ValueError: sample数、fs、FIR長、スペクトル応答が不正な場合。

    振幅、包絡、空間共分散は呼び出し側の責務とし、この関数は時間波形の期待RMSを1へ正規化する。
    """

    if n_sample < 0:
        raise ValueError("n_sample must be non-negative")
    if fs <= 0.0:
        raise ValueError("fs must be positive")
    if filter_length < 3 or filter_length % 2 == 0:
        raise ValueError("filter_length must be an odd integer greater than or equal to 3")
    validate_noise_spectrum_for_sampling(spectrum, fs)
    half = filter_length // 2
    white_start = start_sample_index - half
    white_indices = np.arange(white_start, white_start + n_sample + filter_length - 1, dtype=np.int64)
    white = _deterministic_standard_normal(white_indices, seed)
    fir = _design_noise_shaping_fir(spectrum, fs, filter_length)
    # valid畳み込みで各出力sampleが同じglobal index近傍を参照し、chunk境界でも値を一致させる。
    shaped = np.convolve(white, fir, mode="valid")
    if shaped.shape != (n_sample,):
        raise ValueError(f"internal noise convolution returned unexpected shape {shaped.shape}")
    return np.asarray(shaped, dtype=np.float64)


def validate_noise_spectrum_for_sampling(spectrum: NoiseSpectrum, fs: float) -> None:
    """NoiseSpectrumの全定義周波数がNyquist内にあることを検証する。

    Args:
        spectrum: 帯域制限、pink、またはtable指定のNoiseSpectrum。
        fs: サンプリング周波数。単位はHz。

    Returns:
        なし。

    Raises:
        ValueError: fsが不正、または定義周波数がNyquistを超える場合。
        NotImplementedError: 周波数上端を解釈できないNoiseSpectrum実装の場合。

    Nyquist外の帯域を暗黙に切り捨てると、ASDから換算した帯域RMSと実際の生成帯域が不一致になる。
    そのため、FIR設計前にscene条件の誤りとして拒否する。
    """

    if not np.isfinite(fs) or fs <= 0.0:
        raise ValueError("fs must be finite and positive")
    if isinstance(spectrum, (BandLimitedNoiseSpectrum, PinkNoiseSpectrum)):
        maximum_frequency_hz = float(spectrum.f_high_hz)
    elif isinstance(spectrum, CustomNoiseSpectrum):
        maximum_frequency_hz = float(np.max(np.asarray(spectrum.frequencies_hz, dtype=float)))
    else:
        raise NotImplementedError("NoiseSpectrum sampling validation is not defined for this spectrum type")
    if maximum_frequency_hz > fs / 2.0:
        raise ValueError(
            f"noise spectrum maximum frequency {maximum_frequency_hz} Hz exceeds Nyquist {fs / 2.0} Hz"
        )


def _deterministic_standard_normal(sample_indices: Array, seed: int) -> Array:
    """sample indexとseedだけで決まる標準正規白色系列を返す。"""

    indices_uint = np.asarray(sample_indices, dtype=np.int64).astype(np.uint64, copy=False)
    seed_uint = np.uint64(seed & 0xFFFFFFFFFFFFFFFF)
    # 隣接seedをindexへ直接XORすると、seed+chで生成するCH間に相関が残る。
    # seed自体を先にSplitMix64でhashし、独立性の高いkeyへ写してからsample indexと混合する。
    seed_key_1 = _splitmix64(np.asarray([seed_uint], dtype=np.uint64))[0]
    seed_key_2 = _splitmix64(
        np.asarray([seed_uint ^ _SPLITMIX64_INCREMENT], dtype=np.uint64)
    )[0]
    # sample indexを直接hashする性質は維持するため、呼出順やchunkサイズには依存しない。
    uniform_1 = _uint64_to_unit_float(_splitmix64(indices_uint ^ seed_key_1))
    uniform_2 = _uint64_to_unit_float(_splitmix64(indices_uint ^ seed_key_2))
    # Box-Muller変換。uniform_1は開区間(0,1)なのでlog(0)を生じない。
    return np.sqrt(-2.0 * np.log(uniform_1)) * np.cos(2.0 * np.pi * uniform_2)


def _splitmix64(values: Array) -> Array:
    """uint64配列をSplitMix64でhashする。"""

    z = np.asarray(values, dtype=np.uint64) + _SPLITMIX64_INCREMENT
    z = (z ^ (z >> np.uint64(30))) * _SPLITMIX64_MUL_1
    z = (z ^ (z >> np.uint64(27))) * _SPLITMIX64_MUL_2
    return z ^ (z >> np.uint64(31))


def _uint64_to_unit_float(values: Array) -> Array:
    """uint64 hash値を開区間(0,1)のfloat64一様乱数へ写す。"""

    mantissa = np.asarray(values, dtype=np.uint64) >> np.uint64(11)
    # 0.5を足して端点を避け、Box-Mullerのlog(0)を防ぐ。
    return (mantissa.astype(np.float64) + 0.5) / _UINT53_SCALE


def _design_noise_shaping_fir(spectrum: NoiseSpectrum, fs: float, filter_length: int) -> Array:
    """NoiseSpectrumの振幅応答から期待出力RMS 1の線形位相FIRを設計する。"""

    n_fft = _next_power_of_two(max(1024, filter_length * 8))
    freq_axis = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    response = np.asarray(spectrum.evaluate(freq_axis), dtype=complex)
    if response.shape != freq_axis.shape:
        raise ValueError(f"spectrum response must have shape {freq_axis.shape}, got {response.shape}")
    magnitude = np.abs(response)
    if not bool(np.any(magnitude > 0.0)):
        raise ValueError("NoiseSpectrum response must contain at least one non-zero bin")
    # 周波数サンプリング法のゼロ位相応答を中央filter_length点へ有限長化する。
    impulse = np.fft.fftshift(np.fft.irfft(magnitude, n=n_fft))
    center = n_fft // 2
    half = filter_length // 2
    fir = np.asarray(impulse[center - half : center + half + 1], dtype=float)
    # Hann窓で帯域外リップルと時間応答の振動を抑える。
    fir *= np.hanning(filter_length)
    energy = float(np.sum(fir * fir))
    if energy <= 0.0:
        raise ValueError("designed noise FIR has zero energy")
    # 分散1の白色入力に対する出力分散sum(h^2)を1へ正規化する。
    return np.asarray(fir / np.sqrt(energy), dtype=float)


def _next_power_of_two(value: int) -> int:
    """正の整数value以上の最小2冪を返す。"""

    if value <= 0:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()
