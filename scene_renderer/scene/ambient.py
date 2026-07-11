from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .spectrum import BandLimitedNoiseSpectrum, Spectrum
from scene_renderer.level import noise_asd_level_db_to_band_rms


Array: TypeAlias = NDArray[Any]
_COVARIANCE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class AmbientField:
    """単一位置を持たない背景雑音場の定義を保持する。

    このクラスは、背景雑音のスペクトル、帯域積分RMS振幅、CH 間共分散 covariance、
    決定論的生成seed、成分識別子を保持する。
    covariance は shape [n_ch, n_ch]、axis=0/1 は受波器 CH、値は線形パワー比の共分散である。

    具体的な雑音サンプル生成、アレイ形状からの空間相関計算、信号合成は責務に含めない。
    信号処理上は、AmbientFieldRenderer が多CH雑音を生成するための統計条件を表す。
    """

    spectrum: Spectrum
    amplitude: float = 0.0
    covariance: Array | None = None
    spatial_model: Any | None = None
    noise_seed: int | None = None
    noise_filter_length: int = 257
    identifier: str = "ambient"
    role: str = "noise"

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
        if self.amplitude > 0.0 and self.noise_seed is None:
            # 背景雑音はchunk分割と呼出順に依存させないため、非零振幅ではseedを必須にする。
            raise ValueError("noise_seed is required when ambient amplitude is positive")
        if self.noise_filter_length < 3 or self.noise_filter_length % 2 == 0:
            raise ValueError("noise_filter_length must be an odd integer greater than or equal to 3")
        if not self.identifier.strip():
            raise ValueError("identifier must not be empty")
        if not self.role.strip():
            raise ValueError("role must not be empty")
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

    @classmethod
    def from_asd_level_db(
        cls,
        spectrum: BandLimitedNoiseSpectrum,
        level_db_re_rms_per_sqrt_hz: float,
        *,
        covariance: Array | None = None,
        noise_seed: int,
        noise_filter_length: int = 257,
        identifier: str = "ambient",
        role: str = "noise",
    ) -> "AmbientField":
        """one-sided ASD levelから背景雑音場を作成する。

        Args:
            spectrum: 平坦なASDを持つ帯域制限雑音スペクトル。帯域幅は
                `f_high_hz - f_low_hz`から導出する。
            level_db_re_rms_per_sqrt_hz: one-sided ASD level。単位は
                dB re amplitude 1 RMS/sqrt(Hz)。
            covariance: CH間の無次元共分散。shapeは[n_ch, n_ch]。Noneは単位行列。
            noise_seed: sample index依存の決定論的雑音seed。
            noise_filter_length: スペクトル整形FIR長。奇数かつ3以上。
            identifier: scene内で背景場を識別する名前。
            role: 後段が成分を分類する名前。既定はnoise。

        Returns:
            帯域積分RMS振幅を保持するAmbientField。

        Raises:
            ValueError: level、帯域幅、共分散、FIR長、識別子が不正な場合。

        FFT長やbin幅は責務に含めない。帯域の正本をspectrumへ一本化し、
        B=f_high_hz-f_low_hzに対してA=10^(NL/20)sqrt(B)を適用する。
        """

        bandwidth_hz = float(spectrum.f_high_hz - spectrum.f_low_hz)
        return cls(
            spectrum=spectrum,
            amplitude=noise_asd_level_db_to_band_rms(level_db_re_rms_per_sqrt_hz, bandwidth_hz),
            covariance=covariance,
            noise_seed=noise_seed,
            noise_filter_length=noise_filter_length,
            identifier=identifier,
            role=role,
        )
