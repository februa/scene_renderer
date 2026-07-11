"""生成した実信号のRMS・スペクトル・ASD整合を定量評価する。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray


FloatArray: TypeAlias = NDArray[np.floating[Any]]


@dataclass(frozen=True)
class OneSidedRmsSpectrum:
    """実信号のone-sided RMS power spectrumとParseval指標を保持する。

    `frequencies_hz`と`rms_power_per_bin`のshapeは[n_bin]で、axis=0はDCからNyquistまでの
    周波数binである。`rms_power_per_bin`を全binで加算すると`time_rms**2`へ一致する。

    窓関数、スペクトル平滑化、信号生成は責務に含めない。scene_rendererが生成した有限長実信号の
    レベル保存を、時間領域と周波数領域の両側から検証する診断結果である。
    """

    frequencies_hz: FloatArray
    rms_power_per_bin: FloatArray
    time_rms: float
    spectrum_rms: float
    parseval_relative_error: float
    frequency_resolution_hz: float


@dataclass(frozen=True)
class BandRmsMetrics:
    """one-sided物理帯域内のRMS levelと平均ASDを保持する。

    `band_rms`は選択binのRMS power和の平方根、`band_level_db_re_reference_rms`は指定基準への
    20log10比、`mean_asd_level_db_re_reference_rms_per_sqrt_hz`は選択binの平均PSDを
    10log10で表した値である。

    帯域端の補間、窓漏れ補正、合否判定は責務に含めない。狭帯域と広帯域を同じ帯域積分規約で
    比較するための定量指標である。
    """

    f_low_hz: float
    f_high_hz: float
    selected_bin_count: int
    band_rms: float
    band_level_db_re_reference_rms: float
    mean_asd_level_db_re_reference_rms_per_sqrt_hz: float


def calculate_one_sided_rms_spectrum(
    signal: NDArray[Any], sampling_frequency_hz: float
) -> OneSidedRmsSpectrum:
    """1次元実信号をone-sided RMS power spectrumへ変換する。

    Args:
        signal: 実時間信号。shapeは[n_sample]、axis=0は時間sample。
        sampling_frequency_hz: サンプリング周波数。単位はHz。

    Returns:
        one-sided bin RMS powerとParseval整合指標。

    Raises:
        ValueError: signalが1次元実数でない、2 sample未満、非有限、またはfsが不正な場合。

    窓なしFFTを用いる。DCと偶数長のNyquistを除く正周波数binだけを2倍し、
    `sum(P[k]) == mean(x**2)`となるRMS power規約を採用する。
    """

    signal_array = np.asarray(signal)
    if signal_array.ndim != 1 or signal_array.size < 2:
        raise ValueError("signal must have shape [n_sample>=2]")
    if np.iscomplexobj(signal_array):
        raise ValueError("signal must be real for one-sided RMS evaluation")
    signal_float = np.asarray(signal_array, dtype=np.float64)
    if not bool(np.all(np.isfinite(signal_float))):
        raise ValueError("signal must contain only finite values")
    if not np.isfinite(sampling_frequency_hz) or sampling_frequency_hz <= 0.0:
        raise ValueError("sampling_frequency_hz must be finite and positive")

    n_sample = int(signal_float.size)
    spectrum = np.fft.rfft(signal_float)
    # Parsevalのbin powerは|X[k]/N|^2。実信号の負周波数分をinterior positive binへ2倍して集約する。
    rms_power_per_bin = np.abs(spectrum / float(n_sample)) ** 2
    if n_sample % 2 == 0:
        rms_power_per_bin[1:-1] *= 2.0
    else:
        rms_power_per_bin[1:] *= 2.0
    time_power = float(np.mean(signal_float * signal_float))
    spectrum_power = float(np.sum(rms_power_per_bin))
    # 無音では相対誤差の分母が0になるため、両者の絶対差をそのまま0基準で扱う。
    parseval_relative_error = (
        abs(spectrum_power - time_power) / time_power
        if time_power > 0.0
        else abs(spectrum_power - time_power)
    )
    frequency_resolution_hz = float(sampling_frequency_hz / n_sample)
    return OneSidedRmsSpectrum(
        frequencies_hz=np.asarray(
            np.fft.rfftfreq(n_sample, d=1.0 / sampling_frequency_hz), dtype=np.float64
        ),
        rms_power_per_bin=np.asarray(rms_power_per_bin, dtype=np.float64),
        time_rms=float(np.sqrt(time_power)),
        spectrum_rms=float(np.sqrt(spectrum_power)),
        parseval_relative_error=float(parseval_relative_error),
        frequency_resolution_hz=frequency_resolution_hz,
    )


def evaluate_one_sided_band(
    spectrum: OneSidedRmsSpectrum,
    f_low_hz: float,
    f_high_hz: float,
    reference_rms: float = 1.0,
) -> BandRmsMetrics:
    """one-sided spectrumを指定物理帯域で積分する。

    Args:
        spectrum: `calculate_one_sided_rms_spectrum`の結果。
        f_low_hz: 帯域下端。単位はHz。0以上。
        f_high_hz: 帯域上端。単位はHz。f_low_hz以上かつNyquist以下。
        reference_rms: dB基準の線形RMS amplitude。0より大きい値。

    Returns:
        帯域RMS、基準相対level、平均one-sided ASD level。

    Raises:
        ValueError: 帯域、基準、bin選択が不正な場合。

    帯域端を含むFFT binを選択する。non-integer-bin toneでは漏れを含む十分な帯域を呼び出し側が
    指定し、狭帯域・広帯域とも同じpower和で評価する。
    """

    nyquist_hz = float(spectrum.frequencies_hz[-1])
    if not np.isfinite(f_low_hz) or not np.isfinite(f_high_hz):
        raise ValueError("band edges must be finite")
    if f_low_hz < 0.0 or f_high_hz < f_low_hz or f_high_hz > nyquist_hz:
        raise ValueError("band must satisfy 0 <= f_low_hz <= f_high_hz <= Nyquist")
    if not np.isfinite(reference_rms) or reference_rms <= 0.0:
        raise ValueError("reference_rms must be finite and positive")
    mask = (spectrum.frequencies_hz >= f_low_hz) & (spectrum.frequencies_hz <= f_high_hz)
    selected_bin_count = int(np.count_nonzero(mask))
    if selected_bin_count == 0:
        raise ValueError("band must contain at least one FFT bin")
    band_power = float(np.sum(spectrum.rms_power_per_bin[mask]))
    band_rms = float(np.sqrt(band_power))
    band_level_db = _power_level_db(band_power, reference_rms * reference_rms)
    # 各bin powerはPSD×delta_fなので、平均ASD levelはmean(P[k]/delta_f)の10log10で求める。
    mean_psd = float(np.mean(spectrum.rms_power_per_bin[mask] / spectrum.frequency_resolution_hz))
    mean_asd_level_db = _power_level_db(mean_psd, reference_rms * reference_rms)
    return BandRmsMetrics(
        f_low_hz=float(f_low_hz),
        f_high_hz=float(f_high_hz),
        selected_bin_count=selected_bin_count,
        band_rms=band_rms,
        band_level_db_re_reference_rms=band_level_db,
        mean_asd_level_db_re_reference_rms_per_sqrt_hz=mean_asd_level_db,
    )


def _power_level_db(power: float, reference_power: float) -> float:
    """非負powerを基準powerに対する10log10 levelへ変換する。"""

    if power == 0.0:
        return float("-inf")
    return float(10.0 * np.log10(power / reference_power))
