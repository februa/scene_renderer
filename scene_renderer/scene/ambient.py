from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .spectrum import Spectrum


Array: TypeAlias = NDArray[Any]
_COVARIANCE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class AmbientField:
    """単一位置を持たない背景雑音場の定義を保持する。

    このクラスは、背景雑音のスペクトル、線形振幅、CH 間共分散 covariance を保持する。
    covariance は shape [n_ch, n_ch]、axis=0/1 は受波器 CH、値は線形パワー比の共分散である。

    具体的な雑音サンプル生成、アレイ形状からの空間相関計算、信号合成は責務に含めない。
    信号処理上は、AmbientFieldRenderer が多CH雑音を生成するための統計条件を表す。
    """

    spectrum: Spectrum
    amplitude: float = 0.0
    covariance: Array | None = None
    spatial_model: Any | None = None

    def __post_init__(self) -> None:
        """背景雑音場定義の物理的に必要な制約を検証する。

        Args:
            なし。dataclass の保持値を検証する。

        Returns:
            なし。

        Raises:
            ValueError: amplitude が負、covariance が正方でない、対称でない、または半正定値でない場合。
        """

        if self.amplitude < 0.0:
            raise ValueError("amplitude must be non-negative")
        if self.covariance is None:
            # covariance 未指定は、後段 renderer が既定の空間モデルまたは無寄与として扱うため許容する。
            return
        covariance = np.asarray(self.covariance, dtype=float)
        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
            raise ValueError("covariance must have shape (n_ch, n_ch)")
        # 共分散行列は R[ch_i, ch_j] = E[x_i x_j] なので、実数モデルでは対称でなければならない。
        # 1e-12 は float64 でユーザが数式から生成した行列の丸め誤差だけを許すための許容値である。
        if not np.allclose(covariance, covariance.T, atol=_COVARIANCE_TOLERANCE):
            raise ValueError("covariance must be symmetric")
        eigenvalues = np.linalg.eigvalsh(covariance)
        # 半正定値でない共分散からは物理的なガウス雑音を生成できないため、負固有値を API エラーにする。
        if bool(np.any(eigenvalues < -_COVARIANCE_TOLERANCE)):
            raise ValueError("covariance must be positive semidefinite")
        object.__setattr__(self, "covariance", covariance)
