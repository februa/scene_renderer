from __future__ import annotations

import warnings

import numpy as np

from scene_renderer.receiver import Receiver
from scene_renderer.scene import Scene

from .ambient_field_contributor import AmbientFieldContributor
from .contributor import MultiChannelContributor
from .sensor_noise_contributor import SensorNoiseContributor
from .source_field_contributor import SourceFieldContributor


class SceneRenderer:
    """contributor 群を束ねて最終的な受信信号を合成する公開入口。"""

    def __init__(
        self,
        contributors: list[MultiChannelContributor] | None = None,
        dtype: np.dtype | type = np.complex64,
    ) -> None:
        self.contributors = contributors or [
            SourceFieldContributor(),
            AmbientFieldContributor(),
            SensorNoiseContributor(),
        ]
        self.dtype = np.dtype(dtype)

    def render(
        self,
        scene: Scene,
        receiver: Receiver,
        axis_t: np.ndarray,
    ) -> np.ndarray:
        axis_t = _validate_axis_t(axis_t)
        fs = _derive_fs(axis_t)
        n_ch = receiver.array.positions().shape[0]
        n_sample = axis_t.size

        out = np.zeros((n_ch, n_sample), dtype=np.complex64)
        has_complex_contribution = False
        for contributor in self.contributors:
            contribution = np.asarray(
                contributor.render(
                    scene=scene,
                    receiver=receiver,
                    axis_t=axis_t,
                    fs=fs,
                )
            )
            if contribution.shape != out.shape:
                raise ValueError(
                    f"contributor output must have shape {out.shape}, got {contribution.shape}"
                )
            # contributor ごとの返り値をここで複素配列へ正規化して合算する。
            if np.iscomplexobj(contribution) and np.any(np.abs(np.imag(contribution)) > 0.0):
                has_complex_contribution = True
            out += contribution.astype(np.complex64, copy=False)

        if self.dtype == np.dtype(np.float32):
            if has_complex_contribution:
                warnings.warn(
                    "Complex contributions detected; imaginary parts are discarded in float32 output.",
                    stacklevel=2,
                )
            return np.asarray(np.real(out), dtype=np.float32)
        if self.dtype != np.dtype(np.complex64):
            return out.astype(self.dtype)
        return out


def _validate_axis_t(axis_t: np.ndarray) -> np.ndarray:
    axis_t = np.asarray(axis_t, dtype=float)
    if axis_t.ndim != 1:
        raise ValueError(f"axis_t must be 1-D, got shape {axis_t.shape}")
    if axis_t.size == 0:
        raise ValueError("axis_t must not be empty")
    if axis_t.size == 1:
        raise ValueError("axis_t must contain at least two samples")
    diffs = np.diff(axis_t)
    if np.any(diffs <= 0.0):
        raise ValueError("axis_t must be strictly increasing")
    # 公開 API で fs を受けないため、時間軸は等間隔でなければならない。
    if not np.allclose(diffs, diffs[0], rtol=0.0, atol=1e-12):
        raise ValueError("axis_t must be uniformly sampled")
    return axis_t


def _derive_fs(axis_t: np.ndarray) -> float:
    return float(1.0 / (axis_t[1] - axis_t[0]))
