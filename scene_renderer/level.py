"""音響信号のレベル指定を線形振幅へ変換する。"""

from __future__ import annotations

import numpy as np


def tone_rms_level_db_to_peak_amplitude(level_db_re_rms: float) -> float:
    """実正弦波の RMS level を peak amplitude へ変換する。

    Args:
        level_db_re_rms: RMS 振幅基準の level。単位は dB re amplitude 1 RMS。

    Returns:
        実正弦波の線形 peak amplitude。無次元。

    Raises:
        ValueError: level_db_re_rms が有限値でない場合。

    境界条件として任意の有限な正負 dB を許容する。音源波形の生成、伝搬、アレイ投影は
    責務に含めず、音響シーン定義に渡す振幅の意味だけを一意にする。
    """

    if not np.isfinite(level_db_re_rms):
        raise ValueError("level_db_re_rms must be finite")
    # 10^(SL/20) は tone の RMS amplitude であり、実正弦波の peak は crest factor √2 を掛ける。
    return float(np.sqrt(2.0) * 10.0 ** (level_db_re_rms / 20.0))


def noise_asd_level_db_to_band_rms(
    level_db_re_rms_per_sqrt_hz: float,
    bandwidth_hz: float,
) -> float:
    """one-sided noise ASD level を指定帯域の RMS amplitude へ変換する。

    Args:
        level_db_re_rms_per_sqrt_hz: one-sided ASD level。単位は
            dB re amplitude 1 RMS/sqrt(Hz)。
        bandwidth_hz: 積分する one-sided 帯域幅。単位は Hz。0 より大きい値。

    Returns:
        帯域積分後の線形 RMS amplitude。無次元。

    Raises:
        ValueError: levelまたは帯域幅が有限でない、あるいは帯域幅が0以下の場合。

    FFT長や分解能は入力に取らない。同じ物理帯域を積分したRMSをFFT条件から独立させるためである。
    """

    if not np.isfinite(level_db_re_rms_per_sqrt_hz):
        raise ValueError("level_db_re_rms_per_sqrt_hz must be finite")
    if not np.isfinite(bandwidth_hz) or bandwidth_hz <= 0.0:
        raise ValueError("bandwidth_hz must be finite and positive")
    # ASD を二乗して帯域積分し平方根を取るため、振幅表現では 10^(NL/20) * sqrt(B) となる。
    return float(10.0 ** (level_db_re_rms_per_sqrt_hz / 20.0) * np.sqrt(bandwidth_hz))


def white_noise_asd_level_db_to_sample_rms(
    level_db_re_rms_per_sqrt_hz: float,
    sampling_frequency_hz: float,
) -> float:
    """DCからNyquistまでの白色雑音ASDをsample RMSへ変換する。

    Args:
        level_db_re_rms_per_sqrt_hz: one-sided ASD level。単位は
            dB re amplitude 1 RMS/sqrt(Hz)。
        sampling_frequency_hz: サンプリング周波数。単位は Hz。0 より大きい値。

    Returns:
        実白色雑音のsample RMS amplitude。無次元。

    Raises:
        ValueError: sampling_frequency_hz が有限でない、または0以下の場合。

    Nyquist帯域幅は fs/2 とする。帯域制限雑音にはこの関数ではなく、実際の帯域幅を指定する関数を使う。
    """

    if not np.isfinite(sampling_frequency_hz) or sampling_frequency_hz <= 0.0:
        raise ValueError("sampling_frequency_hz must be finite and positive")
    return noise_asd_level_db_to_band_rms(
        level_db_re_rms_per_sqrt_hz,
        sampling_frequency_hz / 2.0,
    )
