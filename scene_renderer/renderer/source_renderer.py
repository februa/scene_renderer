from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from scene_renderer.scene import AcousticSource, NoiseSpectrum, ToneSpectrum

from .noise_waveform import render_indexed_noise, sample_index_from_time


Array: TypeAlias = NDArray[Any]


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
    共通の雑音波形生成部品へ委譲する。

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
                    # RMS toneのcrest factor √2はDC/Nyquistでは成立しないため、表現可能な開区間だけを許す。
                    if not np.isfinite(frequency) or frequency <= 0.0 or frequency >= fs_value / 2.0:
                        raise ValueError("tone frequency must satisfy 0 < frequency < Nyquist")
                    # 狭帯域toneはs(t)=A env(t) exp(j 2πft)。絶対時刻によりchunk間位相を保つ。
                    signal = component.amplitude_value * envelope * np.exp(
                        1j * 2.0 * np.pi * frequency * extended_axis_t
                    )
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
                    start_sample_index = sample_index_from_time(
                        float(axis_t_array[0]), fs_value
                    ) - sample_margin
                    shaped_noise = render_indexed_noise(
                        spectrum=component.spectrum,
                        start_sample_index=start_sample_index,
                        n_sample=extended_axis_t.size,
                        fs=fs_value,
                        seed=noise_seed,
                        filter_length=component.noise_filter_length,
                    )
                    # 共通雑音部品の出力は期待RMS=1なので、音源固有の振幅と包絡はここで適用する。
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
    """時間軸または明示値からサンプリング周波数を決める。"""

    if fs is not None:
        if fs <= 0.0:
            raise ValueError("fs must be positive")
        return float(fs)
    if axis_t.size < 2:
        raise ValueError("fs is required when axis_t has fewer than two samples")
    # axis_tはSceneRenderer側で等間隔検証済みだが、単体利用でも時間順序を守る。
    dt = float(axis_t[1] - axis_t[0])
    if dt <= 0.0:
        raise ValueError("axis_t must be strictly increasing")
    return float(1.0 / dt)


def _make_extended_axis_t(axis_t: Array, fs: float, sample_margin: int) -> Array:
    """元chunkの前後にsample_margin分を足した時間軸を作る。"""

    offsets = np.arange(-sample_margin, axis_t.size + sample_margin, dtype=float) / fs
    return np.asarray(axis_t[0] + offsets, dtype=float)
