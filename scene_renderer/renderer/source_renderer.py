from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from scene_renderer.scene import AcousticSource, ToneSpectrum


@dataclass(frozen=True)
class RenderedSource:
    """描画済みの単一音源成分波形を保持する中間表現。"""

    source: AcousticSource
    signal: np.ndarray
    frequency: float | None = None


class SourceRenderer:
    """音源成分を時間領域の基準信号へ展開する。"""

    def render(self, sources: list[AcousticSource], axis_t: np.ndarray) -> list[RenderedSource]:
        axis_t = np.asarray(axis_t, dtype=float)
        if axis_t.ndim != 1:
            raise ValueError(f"axis_t must be 1-D, got shape {axis_t.shape}")
        if axis_t.size == 0:
            return []

        rendered: list[RenderedSource] = []
        for source in sources:
            for component in source.components:
                if not isinstance(component.spectrum, ToneSpectrum):
                    raise NotImplementedError("minimal SourceRenderer supports ToneSpectrum only")
                frequency = float(component.spectrum.frequency)
                envelope = component.envelope.evaluate(axis_t)
                signal = component.amplitude_value * envelope * np.exp(1j * 2.0 * np.pi * frequency * axis_t)
                rendered.append(
                    RenderedSource(
                        source=source,
                        signal=np.asarray(signal, dtype=np.complex64),
                        frequency=frequency,
                    )
                )
        return rendered
