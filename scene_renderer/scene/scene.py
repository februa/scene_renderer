from __future__ import annotations

from dataclasses import dataclass, field

from .ambient import AmbientField
from .environment import Environment, FreeField
from .source import AcousticSource


@dataclass(frozen=True)
class Scene:
    """音場の構成要素を束ねる読み取り専用コンテナ。

    このクラスは、局所音源 sources、背景雑音場 ambient_fields、伝搬環境 environment を保持する。
    SceneRenderer への入力として、Scene + Receiver から x[ch, t] を生成するための音場側条件を表す。

    信号サンプル生成、時間軸検証、アレイ投影、寄与加算は責務に含めない。
    信号処理上は、後段レンダラが多CH受信信号を合成するための宣言的なシナリオ定義である。
    """

    sources: list[AcousticSource] = field(default_factory=list)
    ambient_fields: list[AmbientField] = field(default_factory=list)
    environment: Environment = field(default_factory=lambda: FreeField(c=1500.0))
