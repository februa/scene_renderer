from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .spectrum import Spectrum


@dataclass(frozen=True)
class AmbientField:
    """単一位置を持たない背景雑音場の定義を保持する。"""

    spectrum: Spectrum
    amplitude: float = 0.0
    covariance: np.ndarray | None = None
    spatial_model: Any | None = None

    def __post_init__(self) -> None:
        if self.amplitude < 0.0:
            raise ValueError("amplitude must be non-negative")
        if self.covariance is None:
            return
        covariance = np.asarray(self.covariance, dtype=float)
        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
            raise ValueError("covariance must have shape (n_ch, n_ch)")
        if not np.allclose(covariance, covariance.T, atol=1e-12):
            raise ValueError("covariance must be symmetric")
        eigenvalues = np.linalg.eigvalsh(covariance)
        if np.any(eigenvalues < -1e-12):
            raise ValueError("covariance must be positive semidefinite")
        object.__setattr__(self, "covariance", covariance)
