from __future__ import annotations

from typing import Any, TypeAlias
import warnings

import numpy as np
from numpy.typing import NDArray

from scene_renderer.receiver import Receiver
from scene_renderer.scene import Scene

from .ambient_field_contributor import AmbientFieldContributor
from .contributor import MultiChannelContributor
from .sensor_noise_contributor import SensorNoiseContributor
from .source_field_contributor import SourceFieldContributor
from .render_result import RenderedContribution, RenderedScene


Array: TypeAlias = NDArray[Any]
_AXIS_UNIFORM_ATOL_SECONDS = 1e-12


class SceneRenderer:
    """contributor 群を束ねて最終的な受信信号を合成する公開入口。

    このクラスは、Scene と Receiver と時間軸 axis_t から、多CH受信信号 x[ch, t] を生成する。
    出力 shape は [n_ch, n_sample]、axis=0 は CH、axis=1 は時間サンプルである。

    個別の音源波形生成、伝搬経路計算、雑音サンプル生成の詳細は責務に含めず、contributor へ委譲する。
    信号処理上は、物理寄与を同一の [ch, t] 格子に揃えて線形合成するオーケストレーション層である。
    """

    def __init__(
        self,
        contributors: list[MultiChannelContributor] | None = None,
        dtype: np.dtype[Any] | type[Any] = np.complex64,
    ) -> None:
        """SceneRenderer を作成する。

        Args:
            contributors: 多CH寄与生成器のリスト。None の場合は局所音源・背景雑音・センサ雑音の既定構成。
            dtype: 最終出力 dtype。既定は complex64。float32 では虚部を破棄する可能性がある。

        Returns:
            なし。

        Raises:
            TypeError: dtype を NumPy dtype として解釈できない場合。
        """

        # None は公開 API の簡便指定であり、内部属性には解決済みの contributor リストだけを保持する。
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
        axis_t: Array,
    ) -> Array:
        """Scene と Receiver から多CH受信信号を生成する。

        Args:
            scene: 音場定義。
            receiver: 受波器定義。array.positions() は shape [n_ch, 3]、単位 m。
            axis_t: 時間軸。shape は [n_sample]、単位は s。厳密単調増加かつ等間隔である必要がある。

        Returns:
            多CH受信信号。shape は [n_ch, n_sample]。既定 dtype は complex64。

        Raises:
            ValueError: axis_t が 1 次元でない、空、1 サンプル、非単調、非等間隔、または contributor 出力 shape が不一致の場合。
        """

        return self.render_components(scene, receiver, axis_t).mixed

    def render_components(self, scene: Scene, receiver: Receiver, axis_t: Array) -> RenderedScene:
        """Sceneを1回の公開呼出しで描画し、物理寄与ごとの受信信号も返す。

        Args:
            scene: 音場定義。音源、背景雑音場、伝搬環境を含む。
            receiver: 受波器定義。array.positions() shapeは[n_ch, 3]、単位はm。
            axis_t: 時間軸。shapeは[n_sample]、単位はs。厳密単調増加かつ等間隔。

        Returns:
            `RenderedScene`。mixed shapeは[n_ch, n_sample]、time_s shapeは[n_sample]、
            receiver_positions_m shapeは[n_ch, 3]。

        Raises:
            ValueError: 時間軸、identifier一意性、寄与shapeが不正な場合。

        個別波形生成や伝搬はcontributorへ委譲し、このメソッドは同じ[ch,t]格子への加算だけを担う。
        """

        axis_t_array = _validate_axis_t(axis_t)
        fs = _derive_fs(axis_t_array)
        receiver_positions_m = np.asarray(receiver.array.positions(), dtype=float)
        n_ch = receiver_positions_m.shape[0]
        n_sample = axis_t_array.size

        out = np.zeros((n_ch, n_sample), dtype=np.complex64)
        has_complex_contribution = False
        rendered_contributions: list[RenderedContribution] = []
        for contributor in self.contributors:
            components = contributor.render_contributions(scene, receiver, axis_t_array, fs)
            for component in components:
                contribution = np.asarray(component.signal)
                if contribution.shape != out.shape:
                    raise ValueError(f"contributor output must have shape {out.shape}, got {contribution.shape}")
                # contributorごとの返り値を[ch,t]複素配列へ正規化してから線形加算する。
                if np.iscomplexobj(contribution) and bool(np.any(np.abs(np.imag(contribution)) > 0.0)):
                    has_complex_contribution = True
                out += contribution.astype(np.complex64, copy=False)
                rendered_contributions.append(component)

        mixed: Array
        if self.dtype == np.dtype(np.float32):
            if has_complex_contribution:
                warnings.warn(
                    "Complex contributions detected; imaginary parts are discarded in float32 output.",
                    stacklevel=2,
                )
            mixed = np.asarray(np.real(out), dtype=np.float32)
        elif self.dtype != np.dtype(np.complex64):
            mixed = out.astype(self.dtype)
        else:
            mixed = out
        # mixedとcomponentsは同じdtypeで加法整合させる。float出力では各成分も同じ規約で実部化する。
        normalized_components = tuple(
            RenderedContribution(
                identifier=component.identifier,
                role=component.role,
                kind=component.kind,
                signal=np.asarray(
                    np.real(component.signal) if self.dtype == np.dtype(np.float32) else component.signal,
                    dtype=self.dtype,
                ),
            )
            for component in rendered_contributions
        )
        return RenderedScene(
            mixed=mixed,
            components=normalized_components,
            time_s=axis_t_array,
            receiver_positions_m=receiver_positions_m,
            sampling_frequency_hz=fs,
        )


def _validate_axis_t(axis_t: Array) -> Array:
    """公開 API の時間軸を検証し、float64 の 1 次元配列へ正規化する。

    Args:
        axis_t: 時間軸。shape は [n_sample]、単位は s。

    Returns:
        正規化済み時間軸。shape は [n_sample]、dtype は float64。

    Raises:
        ValueError: 1 次元でない、2 サンプル未満、非有限、非単調、または非等間隔の場合。
    """

    axis_t_array = np.asarray(axis_t, dtype=float)
    if axis_t_array.ndim != 1:
        raise ValueError(f"axis_t must be 1-D, got shape {axis_t_array.shape}")
    if axis_t_array.size == 0:
        raise ValueError("axis_t must not be empty")
    if axis_t_array.size == 1:
        raise ValueError("axis_t must contain at least two samples")
    # NaN/inf を含む時間軸では fs と位相が定義できないため、原因が分かる段階で拒否する。
    if not bool(np.all(np.isfinite(axis_t_array))):
        raise ValueError("axis_t must contain only finite values")
    diffs = np.diff(axis_t_array)
    if bool(np.any(diffs <= 0.0)):
        raise ValueError("axis_t must be strictly increasing")
    # 公開 API で fs を受けないため、時間軸は等間隔でなければならない。
    # 1e-12 s は float64 の丸め誤差を許しつつ、サンプル周期の意図しない揺れを検出するための絶対許容値である。
    if not np.allclose(diffs, diffs[0], rtol=0.0, atol=_AXIS_UNIFORM_ATOL_SECONDS):
        raise ValueError("axis_t must be uniformly sampled")
    return axis_t_array


def _derive_fs(axis_t: Array) -> float:
    """等間隔時間軸からサンプリング周波数を導出する。

    Args:
        axis_t: 検証済み時間軸。shape は [n_sample]、単位は s、n_sample >= 2。

    Returns:
        サンプリング周波数。単位は Hz。

    Raises:
        IndexError: 未検証の 2 サンプル未満の配列を渡した場合。
    """

    # fs = 1 / dt。dt は axis_t[1] - axis_t[0] [s] なので、結果の単位は Hz になる。
    return float(1.0 / (axis_t[1] - axis_t[0]))
