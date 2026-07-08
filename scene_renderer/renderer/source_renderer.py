from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from scene_renderer.scene import AcousticSource, ToneSpectrum


Array: TypeAlias = NDArray[Any]


@dataclass(frozen=True)
class RenderedSource:
    """描画済みの単一音源成分波形を保持する中間表現。

    このクラスは、元の AcousticSource、時間領域の基準信号 signal、tone 周波数 frequency を保持する。
    signal の shape は [n_sample]、axis=0 は時間サンプル、dtype は complex64 を基本とする。

    伝搬方向、経路遅延、アレイ CH への投影は責務に含めない。
    信号処理上は、s(t) = A env(t) exp(j 2π f t) を伝搬モデルへ渡す境界表現である。
    """

    source: AcousticSource
    signal: Array
    frequency: float | None = None


class SourceRenderer:
    """音源成分を時間領域の基準信号へ展開する。

    このクラスは、AcousticSource の各 SourceComponent を axis_t [s] 上で評価し、
    RenderedSource のリストを返す。出力 signal は shape [n_sample] の複素時系列である。

    伝搬遅延、アレイ投影、背景雑音、センサ雑音の生成は責務に含めない。
    信号処理上は、音源固有の波形 s(t) を幾何・アレイ処理から分離する段である。
    """

    def render(self, sources: list[AcousticSource], axis_t: Array) -> list[RenderedSource]:
        """局所音源成分を基準複素信号へ描画する。

        Args:
            sources: 局所音源リスト。各 source は複数 SourceComponent を持てる。
            axis_t: 時間軸。shape は [n_sample]、axis=0 は時間サンプル、単位は s。

        Returns:
            RenderedSource リスト。各 signal の shape は [n_sample]、dtype は complex64。

        Raises:
            ValueError: axis_t が 1 次元でない、または envelope が axis_t と同じ shape を返さない場合。
            NotImplementedError: ToneSpectrum 以外の Spectrum が渡された場合。
        """

        axis_t_array = np.asarray(axis_t, dtype=float)
        if axis_t_array.ndim != 1:
            raise ValueError(f"axis_t must be 1-D, got shape {axis_t_array.shape}")
        if axis_t_array.size == 0:
            # SceneRenderer は空 axis を拒否するが、単体利用では「描画すべきサンプルなし」として空リストに倒す。
            return []

        rendered: list[RenderedSource] = []
        for source in sources:
            for component in source.components:
                if not isinstance(component.spectrum, ToneSpectrum):
                    raise NotImplementedError("SourceRenderer supports ToneSpectrum only")
                frequency = float(component.spectrum.frequency)
                envelope = np.asarray(component.envelope.evaluate(axis_t_array), dtype=float)
                if envelope.shape != axis_t_array.shape:
                    raise ValueError(f"envelope must have shape {axis_t_array.shape}, got {envelope.shape}")
                # 狭帯域 tone の基準信号は s(t)=A env(t) exp(j 2π f t)。ここではまだ伝搬遅延を入れない。
                signal = component.amplitude_value * envelope * np.exp(1j * 2.0 * np.pi * frequency * axis_t_array)
                rendered.append(
                    RenderedSource(
                        source=source,
                        signal=np.asarray(signal, dtype=np.complex64),
                        frequency=frequency,
                    )
                )
        return rendered
