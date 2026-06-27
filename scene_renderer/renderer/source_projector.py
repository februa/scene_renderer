from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from scene_renderer.scene import Environment
from scene_renderer.receiver import Receiver

from .propagation import PropagationPath


class SourceProjector(ABC):
    """伝搬経路をアレイ観測信号へ投影する抽象インターフェース。"""

    @abstractmethod
    def project(
        self,
        paths: list[PropagationPath],
        receiver: Receiver,
        environment: Environment,
        axis_t: np.ndarray,
        fs: float,
    ) -> np.ndarray:
        raise NotImplementedError


class NarrowbandPlaneWaveProjector(SourceProjector):
    """平面波近似でトーン信号を各 CH に投影する。"""

    def project(
        self,
        paths: list[PropagationPath],
        receiver: Receiver,
        environment: Environment,
        axis_t: np.ndarray,
        fs: float,
    ) -> np.ndarray:
        del fs
        axis_t = np.asarray(axis_t, dtype=float)
        n_sample = axis_t.size
        element_pos_array = receiver.array.positions()
        n_ch = element_pos_array.shape[0]
        out = np.zeros((n_ch, n_sample), dtype=np.complex64)

        t0 = float(axis_t[0]) if n_sample else 0.0
        receiver_pose = receiver.trajectory.pose(t0)

        for path in paths:
            if path.frequency is None:
                raise NotImplementedError("minimal projector supports tone paths only")
            # 伝搬側は絶対座標までを担当し、投影側でアレイ相対の位相差へ変換する。
            direction_array = receiver_pose.world_vector_to_array(path.direction_world)
            tau = (element_pos_array @ direction_array) / environment.c
            phase = np.exp(-1j * 2.0 * np.pi * float(path.frequency) * tau).astype(np.complex64)
            out += path.gain * phase[:, np.newaxis] * np.asarray(path.signal, dtype=np.complex64)[np.newaxis, :]
        return out
