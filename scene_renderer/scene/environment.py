from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Environment:
    """伝搬環境の共通パラメータを保持する基底クラス。"""

    c: float

    def __post_init__(self) -> None:
        if self.c <= 0:
            raise ValueError("c must be positive")


@dataclass(frozen=True)
class FreeField(Environment):
    """反射や吸収を持たない自由音場を表す。"""

    pass
