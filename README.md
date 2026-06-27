# scene_renderer

`scene_renderer` は、受信信号生成を「音場定義」「伝搬」「アレイ投影」「寄与合成」に分けて扱うためのライブラリです。
信号処理そのものではなく、後段のビームフォーミングや検出処理に渡す多チャンネル受信信号 `x[ch, t]` を、拡張しやすい責務分離で生成することを設計思想にしています。

`scene_renderer` は、`Scene + Receiver -> x[ch, t]` を最小構成で実装した受信信号生成ライブラリです。
別プロジェクトからインストールして利用しつつ、このリポジトリ側で機能拡張した内容を更新追従できる形を前提にしています。

## 依存関係

実行時依存:

- `numpy>=1.23`

開発時依存:

- `pytest>=8.0`

依存関係の正本は [pyproject.toml](./pyproject.toml) です。

## インストール

開発中の本リポジトリを別プロジェクトから追従利用する場合は editable install を使います。

```bash
pip install -e /path/to/scene_renderer
```

依存込みで開発用に入れる場合は以下です。

```bash
pip install -e "/path/to/scene_renderer[dev]"
```

Git リポジトリとして利用する場合は、将来的に以下の形で導入できます。

```bash
pip install git+<repository-url>
```

## 利用例

```python
import numpy as np
from scene_renderer import (
    AcousticSource,
    ConstantEnvelope,
    FreeField,
    LinearArray,
    Receiver,
    Scene,
    SceneRenderer,
    SourceComponent,
    StaticPose,
    ToneSpectrum,
)

receiver = Receiver(
    trajectory=StaticPose([0.0, 0.0, 0.0], heading_deg=0.0),
    array=LinearArray(n_ch=32, spacing=0.075),
)

scene = Scene(
    sources=[
        AcousticSource.from_relative_bearing(
            bearing_deg=90.0,
            distance=1000.0,
            receiver_pose=receiver.trajectory.pose(0.0),
            components=[
                SourceComponent(
                    spectrum=ToneSpectrum(1000.0),
                    envelope=ConstantEnvelope(),
                    amplitude=1.0,
                )
            ],
        )
    ],
    ambient_fields=[],
    environment=FreeField(c=1500.0),
)

axis_t = np.arange(32768) / 32768
x = SceneRenderer().render(scene, receiver, axis_t)
```

## 開発コマンド

```bash
python -m pytest -q
python examples/example_minimal.py
```

## 実装範囲

- `scene`
  - `Scene`
  - `AcousticSource`
  - `SourceComponent`
  - `AmbientField`
  - `ToneSpectrum`
  - `ConstantEnvelope`
  - `FreeField`
  - `Pose / Trajectory / StaticPose / StaticPosition`
- `receiver`
  - `Receiver`
  - `LinearArray`
- `renderer`
  - `SceneRenderer`
  - `SourceFieldContributor`
  - `SourceRenderer`
  - `FreeFieldPropagation`
  - `NarrowbandPlaneWaveProjector`
  - `ArrayProjector`
  - `AmbientFieldRenderer`
  - `SensorNoiseGenerator`

最小構成では、自由音場の直達波、単一トーン、CH 間位相差のみを信号へ反映します。
絶対伝搬遅延、球面拡散、ドップラー、背景雑音場本体、センサ雑音本体は未実装または零出力です。

## 上位 API 向けの信号設定例

`SourceComponent` は `amplitude` を主指定とします。`level_db` は互換用の補助指定であり、`amplitude` と `level_db` の同時指定はエラーです。

```python
def make_source(cfg_signal, receiver) -> AcousticSource:
    return AcousticSource.from_relative_bearing(
        bearing_deg=cfg_signal["dir_azimuth_deg"],
        elevation_deg=cfg_signal.get("dir_elevation_deg", 0.0),
        distance=cfg_signal.get("distance", 1000.0),
        receiver_pose=receiver.trajectory.pose(0.0),
        components=[
            SourceComponent(
                spectrum=cfg_signal["spectrum"],
                envelope=cfg_signal["envelope"],
                amplitude=cfg_signal.get("amplitude", 1.0),
            )
        ],
    )
```

デシベル指定を使いたい場合は、20log10 の振幅レベルとして明示します。

```python
component = SourceComponent.from_level_db20(
    spectrum=ToneSpectrum(1000.0),
    envelope=ConstantEnvelope(),
    level_db=0.0,
)
```
