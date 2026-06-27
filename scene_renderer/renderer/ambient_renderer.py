from __future__ import annotations

import numpy as np

from scene_renderer.scene import AmbientField
from scene_renderer.receiver import Receiver


class AmbientFieldRenderer:
    """背景雑音場定義から多CH雑音信号を生成する。"""

    def __init__(
        self,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.rng = rng or np.random.default_rng()

    def render(
        self,
        fields: list[AmbientField],
        receiver: Receiver,
        axis_t: np.ndarray,
        fs: float,
    ) -> np.ndarray:
        del fields, fs
        n_ch = receiver.array.positions().shape[0]
        n_sample = np.asarray(axis_t, dtype=float).size
        # 最小マイルストンでは配線だけ先に確定し、ここでは零出力を返す。
        return np.zeros((n_ch, n_sample), dtype=np.float32)
