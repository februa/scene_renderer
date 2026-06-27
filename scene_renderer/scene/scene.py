from __future__ import annotations

from dataclasses import dataclass, field
from .source import AcousticSource
from .ambient import AmbientField
from .environment import Environment, FreeField


@dataclass(frozen=True)
class Scene:
    """音場の構成要素を束ねる読み取り専用コンテナ。"""

    sources: list[AcousticSource] = field(default_factory=list)
    ambient_fields: list[AmbientField] = field(default_factory=list)
    environment: Environment = field(default_factory=lambda: FreeField(c=1500.0))
