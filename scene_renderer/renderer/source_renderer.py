from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from scene_renderer.scene import AcousticSource, NoiseSpectrum, ToneSpectrum


Array: TypeAlias = NDArray[Any]
_SAMPLE_INDEX_ATOL = 1.0e-6
_SPLITMIX64_INCREMENT = np.uint64(0x9E3779B97F4A7C15)
_SPLITMIX64_MUL_1 = np.uint64(0xBF58476D1CE4E5B9)
_SPLITMIX64_MUL_2 = np.uint64(0x94D049BB133111EB)
_UINT53_SCALE = float(2**53)


@dataclass(frozen=True)
class RenderedSource:
    """描画済みの単一音源成分波形を保持する中間表現。

    このクラスは、元の AcousticSource、時間領域の基準信号 signal、元 chunk に対応する
    center_slice、tone 周波数 frequency を保持する。signal は必要に応じて前後に halo を含み、
    center_slice が SceneRenderer に渡された axis_t 区間を表す。

    伝搬方向、経路遅延、アレイ CH への投影は責務に含めない。
    信号処理上は、s(t) を伝搬モデルへ渡す境界表現であり、広帯域 FIR 遅延に必要な halo を明示する。
    """

    source: AcousticSource
    signal: Array
    center_slice: slice
    frequency: float | None = None


class SourceRenderer:
    """音源成分を時間領域の基準信号へ展開する。

    このクラスは、AcousticSource の各 SourceComponent を axis_t [s] 上で評価し、
    RenderedSource のリストを返す。ToneSpectrum は解析式で生成し、NoiseSpectrum は
    sample index と seed から決定論的白色系列を作って FIR で整形する。

    伝搬遅延、アレイ投影、背景雑音、センサ雑音の生成は責務に含めない。
    信号処理上は、音源固有の波形 s(t) を幾何・アレイ処理から分離する段である。
    """

    def render(
        self,
        sources: list[AcousticSource],
        axis_t: Array,
        fs: float | None = None,
        sample_margin: int = 0,
    ) -> list[RenderedSource]:
        """局所音源成分を基準複素信号へ描画する。

        Args:
            sources: 局所音源リスト。各 source は複数 SourceComponent を持てる。
            axis_t: 時間軸。shape は [n_sample]、axis=0 は時間サンプル、単位は s。
            fs: サンプリング周波数。単位は Hz。None の場合は axis_t から導出する。
            sample_margin: 前後に追加生成する halo サンプル数。単位は sample。

        Returns:
            RenderedSource リスト。各 signal の shape は [n_sample + 2 * sample_margin]、dtype は complex64。

        Raises:
            ValueError: axis_t、fs、sample_margin、envelope shape、noise_seed が不正な場合。
            NotImplementedError: ToneSpectrum / NoiseSpectrum 以外の Spectrum が渡された場合。
        """

        axis_t_array = np.asarray(axis_t, dtype=float)
        if axis_t_array.ndim != 1:
            raise ValueError(f"axis_t must be 1-D, got shape {axis_t_array.shape}")
        if sample_margin < 0:
            raise ValueError("sample_margin must be non-negative")
        if axis_t_array.size == 0:
            # SceneRenderer は空 axis を拒否するが、単体利用では「描画すべきサンプルなし」として空リストに倒す。
            return []
        fs_value = _resolve_fs(axis_t_array, fs)
        center_slice = slice(sample_margin, sample_margin + axis_t_array.size)
        extended_axis_t = _make_extended_axis_t(axis_t_array, fs_value, sample_margin)

        rendered: list[RenderedSource] = []
        for source in sources:
            for component in source.components:
                envelope = np.asarray(component.envelope.evaluate(extended_axis_t), dtype=float)
                if envelope.shape != extended_axis_t.shape:
                    raise ValueError(f"envelope must have shape {extended_axis_t.shape}, got {envelope.shape}")
                if isinstance(component.spectrum, ToneSpectrum):
                    frequency = float(component.spectrum.frequency)
                    # 狭帯域 tone の基準信号は s(t)=A env(t) exp(j 2π f t)。絶対時刻 axis_t により chunk 間位相を保つ。
                    signal = component.amplitude_value * envelope * np.exp(1j * 2.0 * np.pi * frequency * extended_axis_t)
                    rendered.append(
                        RenderedSource(
                            source=source,
                            signal=np.asarray(signal, dtype=np.complex64),
                            center_slice=center_slice,
                            frequency=frequency,
                        )
                    )
                    continue
                if isinstance(component.spectrum, NoiseSpectrum):
                    noise_seed = component.noise_seed
                    if noise_seed is None:
                        raise ValueError("noise_seed is required for NoiseSpectrum")
                    start_sample_index = _sample_index_from_time(axis_t_array[0], fs_value) - sample_margin
                    shaped_noise = _render_indexed_noise(
                        spectrum=component.spectrum,
                        start_sample_index=start_sample_index,
                        n_sample=extended_axis_t.size,
                        fs=fs_value,
                        seed=noise_seed,
                        filter_length=component.noise_filter_length,
                    )
                    # NoiseSpectrum の FIR は期待 RMS=1 に正規化済みなので、ここで SourceComponent の振幅と包絡を掛ける。
                    signal = component.amplitude_value * envelope * shaped_noise
                    rendered.append(
                        RenderedSource(
                            source=source,
                            signal=np.asarray(signal, dtype=np.complex64),
                            center_slice=center_slice,
                            frequency=None,
                        )
                    )
                    continue
                raise NotImplementedError("SourceRenderer supports ToneSpectrum and NoiseSpectrum only")
        return rendered


def _resolve_fs(axis_t: Array, fs: float | None) -> float:
    """時間軸または明示値からサンプリング周波数を決める。

    Args:
        axis_t: 時間軸。shape は [n_sample]、単位は s。
        fs: 明示サンプリング周波数。単位は Hz。

    Returns:
        サンプリング周波数。単位は Hz。

    Raises:
        ValueError: fs が 0 以下、または axis_t から導出できない場合。
    """

    if fs is not None:
        if fs <= 0.0:
            raise ValueError("fs must be positive")
        return float(fs)
    if axis_t.size < 2:
        raise ValueError("fs is required when axis_t has fewer than two samples")
    # axis_t は SceneRenderer 側で等間隔検証済みの想定だが、単体利用でも fs の符号をここで守る。
    dt = float(axis_t[1] - axis_t[0])
    if dt <= 0.0:
        raise ValueError("axis_t must be strictly increasing")
    return float(1.0 / dt)


def _make_extended_axis_t(axis_t: Array, fs: float, sample_margin: int) -> Array:
    """元 chunk の前後に sample_margin 分を足した時間軸を作る。

    Args:
        axis_t: 元の時間軸。shape は [n_sample]、単位は s。
        fs: サンプリング周波数。単位は Hz。
        sample_margin: 前後 halo。単位は sample。

    Returns:
        拡張時間軸。shape は [n_sample + 2 * sample_margin]、単位は s。

    Raises:
        なし。fs と sample_margin は呼び出し元で検証済み。
    """

    offsets = np.arange(-sample_margin, axis_t.size + sample_margin, dtype=float) / fs
    return np.asarray(axis_t[0] + offsets, dtype=float)


def _sample_index_from_time(t0: float, fs: float) -> int:
    """絶対時刻から整数サンプル index を復元する。

    Args:
        t0: chunk 先頭時刻。単位は s。
        fs: サンプリング周波数。単位は Hz。

    Returns:
        global sample index。

    Raises:
        ValueError: t0 * fs が整数サンプル位置から外れている場合。
    """

    sample_index_float = t0 * fs
    sample_index = int(round(sample_index_float))
    # deterministic noise は sample index で値を決めるため、時刻がサンプル格子に乗らない場合は曖昧に丸めない。
    if abs(sample_index_float - sample_index) > _SAMPLE_INDEX_ATOL:
        raise ValueError("axis_t[0] must align with an integer sample index for NoiseSpectrum")
    return sample_index


def _render_indexed_noise(
    spectrum: NoiseSpectrum,
    start_sample_index: int,
    n_sample: int,
    fs: float,
    seed: int,
    filter_length: int,
) -> Array:
    """決定論的白色系列を FIR で整形してノイズ波形を作る。

    Args:
        spectrum: ノイズ用スペクトル。FIR の目標振幅応答を返す。
        start_sample_index: 出力先頭の global sample index。
        n_sample: 出力サンプル数。
        fs: サンプリング周波数。単位は Hz。
        seed: 白色系列を決める整数 seed。
        filter_length: FIR 長。奇数。

    Returns:
        広帯域ノイズ信号。shape は [n_sample]、期待 RMS は 1。

    Raises:
        ValueError: FIR 設計で有効な通過成分が得られない場合。
    """

    half = filter_length // 2
    white_start = start_sample_index - half
    white_indices = np.arange(white_start, white_start + n_sample + filter_length - 1, dtype=np.int64)
    white = _deterministic_standard_normal(white_indices, seed)
    fir = _design_noise_shaping_fir(spectrum, fs, filter_length)
    # valid 畳み込みにより、white の [i-half, i+half] を使って出力 i を作る。chunk 境界でも同じ index なら同じ値になる。
    shaped = np.convolve(white, fir, mode="valid")
    if shaped.shape != (n_sample,):
        raise ValueError(f"internal noise convolution returned unexpected shape {shaped.shape}")
    return np.asarray(shaped, dtype=np.float64)


def _deterministic_standard_normal(sample_indices: Array, seed: int) -> Array:
    """sample index と seed だけで決まる標準正規白色系列を返す。

    Args:
        sample_indices: global sample index。shape は [n_sample]。
        seed: 白色系列の seed。

    Returns:
        標準正規系列。shape は [n_sample]、期待平均 0、期待分散 1。

    Raises:
        なし。
    """

    indices_uint = np.asarray(sample_indices, dtype=np.int64).astype(np.uint64, copy=False)
    seed_uint = np.uint64(seed & 0xFFFFFFFFFFFFFFFF)
    # SplitMix64 は index を直接 hash するため、呼び出し順や chunk サイズに依存しない乱数値を作れる。
    uniform_1 = _uint64_to_unit_float(_splitmix64(indices_uint ^ seed_uint))
    uniform_2 = _uint64_to_unit_float(_splitmix64(indices_uint ^ (seed_uint + _SPLITMIX64_INCREMENT)))
    # Box-Muller 変換。uniform_1 は (0, 1) にして log(0) を避ける。
    return np.sqrt(-2.0 * np.log(uniform_1)) * np.cos(2.0 * np.pi * uniform_2)


def _splitmix64(values: Array) -> Array:
    """uint64 配列を SplitMix64 で hash する。

    Args:
        values: uint64 配列。shape は任意。

    Returns:
        hash 後の uint64 配列。shape は入力と同じ。

    Raises:
        なし。uint64 の overflow は SplitMix64 の設計上の剰余演算である。
    """

    z = np.asarray(values, dtype=np.uint64) + _SPLITMIX64_INCREMENT
    z = (z ^ (z >> np.uint64(30))) * _SPLITMIX64_MUL_1
    z = (z ^ (z >> np.uint64(27))) * _SPLITMIX64_MUL_2
    return z ^ (z >> np.uint64(31))


def _uint64_to_unit_float(values: Array) -> Array:
    """uint64 hash 値を開区間 (0, 1) の float64 一様乱数へ写す。

    Args:
        values: uint64 配列。shape は任意。

    Returns:
        float64 一様乱数。shape は入力と同じ。

    Raises:
        なし。
    """

    mantissa = np.asarray(values, dtype=np.uint64) >> np.uint64(11)
    # 0.5 を足すことで 0 と 1 の端点を避け、Box-Muller の log(0) を防ぐ。
    return (mantissa.astype(np.float64) + 0.5) / _UINT53_SCALE


def _design_noise_shaping_fir(spectrum: NoiseSpectrum, fs: float, filter_length: int) -> Array:
    """NoiseSpectrum の振幅応答から線形位相 FIR を設計する。

    Args:
        spectrum: ノイズ用スペクトル。abs(evaluate(freq_axis)) を目標振幅応答として使う。
        fs: サンプリング周波数。単位は Hz。
        filter_length: FIR 長。奇数。

    Returns:
        FIR 係数。shape は [filter_length]、sum(h^2)=1 に正規化済み。

    Raises:
        ValueError: 振幅応答が全て 0、または正規化不能な場合。
    """

    n_fft = _next_power_of_two(max(1024, filter_length * 8))
    freq_axis = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    response = np.asarray(spectrum.evaluate(freq_axis), dtype=complex)
    if response.shape != freq_axis.shape:
        raise ValueError(f"spectrum response must have shape {freq_axis.shape}, got {response.shape}")
    magnitude = np.abs(response)
    if not bool(np.any(magnitude > 0.0)):
        raise ValueError("NoiseSpectrum response must contain at least one non-zero bin")
    # 周波数サンプリング法でゼロ位相のインパルス応答を作り、fftshift 後に中央 filter_length 点を切り出す。
    impulse = np.fft.fftshift(np.fft.irfft(magnitude, n=n_fft))
    center = n_fft // 2
    half = filter_length // 2
    fir = np.asarray(impulse[center - half : center + half + 1], dtype=float)
    # 有限長化で急峻な切り出しを避け、帯域外リップルと時間応答の振動を抑える。
    fir *= np.hanning(filter_length)
    energy = float(np.sum(fir * fir))
    if energy <= 0.0:
        raise ValueError("designed noise FIR has zero energy")
    # 入力白色系列の分散を 1 とすると、FIR 出力分散は sum(h^2)。期待 RMS が 1 になるよう係数側で正規化する。
    return np.asarray(fir / np.sqrt(energy), dtype=float)


def _next_power_of_two(value: int) -> int:
    """value 以上の最小の 2 冪を返す。

    Args:
        value: 正の整数。

    Returns:
        value 以上の最小 2 冪。

    Raises:
        ValueError: value が 0 以下の場合。
    """

    if value <= 0:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()
