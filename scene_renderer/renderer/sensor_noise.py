from __future__ import annotations

import numpy as np

from scene_renderer.receiver import Receiver


class SensorNoiseGenerator:
    """受波器ごとの独立雑音を生成するための生成器。"""

    def __init__(
        self,
        amplitude: float | np.ndarray = 0.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.amplitude = amplitude
        self.rng = rng or np.random.default_rng()

    def generate(
        self,
        receiver: Receiver,
        axis_t: np.ndarray,
        fs: float,
    ) -> np.ndarray:
        del fs
        n_ch = receiver.array.positions().shape[0]
        n_sample = np.asarray(axis_t, dtype=float).size
        # CH ごとの雑音合成仕様が固まるまでは公開契約だけ維持する。
        return np.zeros((n_ch, n_sample), dtype=np.float32)
