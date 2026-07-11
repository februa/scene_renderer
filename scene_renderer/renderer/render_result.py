"""SceneRendererが生成した寄与別受信信号の結果型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray


Array: TypeAlias = NDArray[Any]


@dataclass(frozen=True)
class RenderedContribution:
    """単一の物理寄与を多CH受信信号として保持する。

    `signal`のshapeは[n_ch, n_sample]で、axis=0が受波器CH、axis=1が時間sampleである。
    `identifier`は同一scene内の個体、`role`はtarget/interference/noiseなど利用目的、`kind`は
    source/ambient/sensorという物理分類を表す。

    波形生成、伝搬、成分加算は責務に含めない。SceneRendererと後段処理の間で、寄与の由来を
    private内部へ依存せず保持する境界型である。
    """

    identifier: str
    role: str
    kind: str
    signal: Array

    def __post_init__(self) -> None:
        """識別子とsignal shapeを検証し、配列を読み取り専用として保持する。"""

        if not self.identifier.strip() or not self.role.strip() or not self.kind.strip():
            raise ValueError("identifier, role, and kind must not be empty")
        signal_array = np.asarray(self.signal)
        if signal_array.ndim != 2:
            raise ValueError(f"signal must have shape [n_ch, n_sample], got {signal_array.shape}")
        signal_array.setflags(write=False)
        object.__setattr__(self, "signal", signal_array)


@dataclass(frozen=True)
class RenderedScene:
    """scene全体と成分別の多CH受信信号を保持する。

    `mixed`のshapeは[n_ch, n_sample]、`time_s`のshapeは[n_sample]、
    `receiver_positions_m`のshapeは[n_ch, 3]である。`components`は同じ格子上の物理寄与であり、
    総和が`mixed`と一致する。

    ビームフォーミング、共分散推定、BL評価は責務に含めない。音響シーン生成の結果を、後段が
    target-only/noise-onlyなどへ明示的に分解できる形で渡す。
    """

    mixed: Array
    components: tuple[RenderedContribution, ...]
    time_s: Array
    receiver_positions_m: Array
    sampling_frequency_hz: float

    def __post_init__(self) -> None:
        """mixed、時間軸、受波器位置、成分のshapeと加法整合性を検証する。

        Raises:
            ValueError: shape、sampling frequency、または成分総和がmixedと一致しない場合。

        レンダラ内部の不整合を結果オブジェクトから後段へ流さないため、生成境界で検証する。
        """

        mixed_array = np.asarray(self.mixed)
        time_array = np.asarray(self.time_s, dtype=float)
        positions_array = np.asarray(self.receiver_positions_m, dtype=float)
        if mixed_array.ndim != 2:
            raise ValueError(f"mixed must have shape [n_ch, n_sample], got {mixed_array.shape}")
        if time_array.shape != (mixed_array.shape[1],):
            raise ValueError("time_s length must match mixed sample axis")
        if positions_array.shape != (mixed_array.shape[0], 3):
            raise ValueError("receiver_positions_m must have shape [n_ch, 3]")
        if not np.isfinite(self.sampling_frequency_hz) or self.sampling_frequency_hz <= 0.0:
            raise ValueError("sampling_frequency_hz must be finite and positive")
        for component in self.components:
            if component.signal.shape != mixed_array.shape:
                raise ValueError("every component signal must have the same shape as mixed")
        if self.components:
            component_sum = np.sum(
                np.stack([component.signal for component in self.components], axis=0),
                axis=0,
                dtype=mixed_array.dtype,
            )
        else:
            component_sum = np.zeros_like(mixed_array)
        # complex64/float32の加算順序による丸めだけを許し、成分の欠落や二重加算を結果境界で検出する。
        if not np.allclose(component_sum, mixed_array, rtol=1.0e-5, atol=1.0e-6):
            raise ValueError("sum of component signals must equal mixed")
        mixed_array.setflags(write=False)
        time_array.setflags(write=False)
        positions_array.setflags(write=False)
        object.__setattr__(self, "mixed", mixed_array)
        object.__setattr__(self, "time_s", time_array)
        object.__setattr__(self, "receiver_positions_m", positions_array)

    def sum_by_role(self, role: str) -> Array:
        """指定roleの全成分を加算する。

        Args:
            role: target、interference、noiseなどの完全一致文字列。

        Returns:
            多CH信号。shapeは[n_ch, n_sample]、axis=0はCH、axis=1は時間sample。
            該当成分がない場合は同shapeのゼロ信号。

        Raises:
            ValueError: roleが空の場合。
        """

        if not role.strip():
            raise ValueError("role must not be empty")
        selected = [component.signal for component in self.components if component.role == role]
        if not selected:
            # 成分なしを例外にするとoptional roleを持つsceneの利用が煩雑になるため、加法単位元を返す。
            return np.zeros_like(self.mixed)
        return np.sum(np.stack(selected, axis=0), axis=0, dtype=self.mixed.dtype)
