from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from scene_renderer.receiver import Receiver
from scene_renderer.scene import AmbientField, NoiseSpectrum
from .source_renderer import _render_indexed_noise, _sample_index_from_time


Array: TypeAlias = NDArray[Any]


class AmbientFieldRenderer:
    """背景雑音場定義から多CH雑音信号を生成する。

    このクラスは、AmbientField のリストと Receiver から背景雑音寄与 x_ambient[ch, t] を返す。
    各AmbientFieldのスペクトル、RMS振幅、CH間共分散からshape [n_ch, n_sample]の雑音を生成する。

    局所音源、センサ自己雑音、Scene 全体の合成は責務に含めない。
    信号処理上は、将来 covariance = L L^T に基づく相関雑音を入れるための [ch, t] 出力境界である。
    """

    def __init__(
        self,
        rng: np.random.Generator | None = None,
    ) -> None:
        """背景雑音レンダラを作成する。

        Args:
            rng: 将来の確率雑音生成に使う乱数生成器。None の場合は default_rng を作成する。

        Returns:
            なし。

        Raises:
            なし。
        """

        self.rng = rng

    def render(
        self,
        fields: list[AmbientField],
        receiver: Receiver,
        axis_t: Array,
        fs: float,
    ) -> Array:
        """背景雑音場の多CH寄与を生成する。

        Args:
            fields: 背景雑音場リスト。現段階では shape 契約維持のため保持のみ。
            receiver: 受波器定義。array.positions() は shape [n_ch, 3]、単位 m。
            axis_t: 時間軸。shape は [n_sample]、単位は s。
            fs: サンプリング周波数。単位は Hz。将来のスペクトル雑音生成で使う。

        Returns:
            背景雑音寄与。shape は [n_ch, n_sample]、dtype は float32。現在は全要素 0。

        Raises:
            なし。
        """

        n_ch = receiver.array.positions().shape[0]
        n_sample = np.asarray(axis_t, dtype=float).size
        out = np.zeros((n_ch, n_sample), dtype=np.float64)
        for field in fields:
            out += self.render_field(field, receiver, axis_t, fs)
        return np.asarray(out, dtype=np.float32)

    def render_field(self, field: AmbientField, receiver: Receiver, axis_t: Array, fs: float) -> Array:
        """単一背景雑音場を共分散付き多CH信号へ展開する。

        Args:
            field: 背景雑音場。amplitudeは帯域積分後のCH当たりRMS振幅。
            receiver: 受波器定義。array.positions() shapeは[n_ch, 3]。
            axis_t: 時間軸。shapeは[n_sample]、単位はs。
            fs: サンプリング周波数。単位はHz。

        Returns:
            背景雑音寄与。shapeは[n_ch, n_sample]、dtypeはfloat32。

        Raises:
            ValueError: covariance shape、seed、時間sample格子が不正な場合。
            NotImplementedError: spectrumがNoiseSpectrumでない場合。
        """

        axis_t_array = np.asarray(axis_t, dtype=float)
        n_ch = receiver.array.positions().shape[0]
        if field.amplitude == 0.0:
            return np.zeros((n_ch, axis_t_array.size), dtype=np.float32)
        if not isinstance(field.spectrum, NoiseSpectrum):
            raise NotImplementedError("AmbientFieldRenderer supports NoiseSpectrum only")
        if field.noise_seed is None:
            raise ValueError("noise_seed is required when ambient amplitude is positive")
        covariance = np.eye(n_ch, dtype=float) if field.covariance is None else np.asarray(field.covariance, dtype=float)
        if covariance.shape != (n_ch, n_ch):
            raise ValueError(f"ambient covariance must have shape {(n_ch, n_ch)}, got {covariance.shape}")
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        # __post_init__で半正定値を確認済みだが、丸め由来の微小負値は平方根のNaNを避けるため0へ丸める。
        covariance_factor = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
        start_sample_index = _sample_index_from_time(float(axis_t_array[0]), fs)
        independent = np.stack(
            [
                _render_indexed_noise(
                    spectrum=field.spectrum,
                    start_sample_index=start_sample_index,
                    n_sample=axis_t_array.size,
                    fs=fs,
                    seed=int(field.noise_seed + ch),
                    filter_length=field.noise_filter_length,
                )
                for ch in range(n_ch)
            ],
            axis=0,
        )
        # x[:,t] = A L w[:,t]、LL^T=R。axis=0のCH共分散をRへ写し、axis=1の時間は保持する。
        return np.asarray(field.amplitude * (covariance_factor @ independent), dtype=np.float32)
