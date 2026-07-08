from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Environment:
    """伝搬環境の共通パラメータを保持する基底クラス。

    このクラスは、音速 c [m/s] を保持し、伝搬遅延 delay = distance / c と
    アレイ CH 間遅延 tau = r dot u / c の分母を提供する。

    反射、吸収、音速プロファイル、海底・海面境界条件は責務に含めない。
    信号処理上は、幾何から時間遅延と位相回転を計算するための媒質パラメータである。
    """

    c: float

    def __post_init__(self) -> None:
        """音速が物理的に有効な正値であることを確認する。

        Args:
            なし。dataclass の c を検証する。

        Returns:
            なし。

        Raises:
            ValueError: c が 0 以下の場合。
        """

        if self.c <= 0:
            raise ValueError("c must be positive")


@dataclass(frozen=True)
class FreeField(Environment):
    """反射や吸収を持たない自由音場を表す。

    このクラスは Environment の音速 c [m/s] だけを使い、直達波の幾何遅延を定義する。

    球面拡散、吸収、マルチパス、境界反射は責務に含めない。
    信号処理上は、最小構成の直接波シミュレーションで使う媒質モデルである。
    """

    pass
