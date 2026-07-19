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

## GitHubリポジトリから外部projectへインストールする

`scene_renderer`を使用するprojectのdirectoryで、GitHubリポジトリを依存元として
登録する。Python 3.10以上とGitが必要である。

package metadata上の配布名は`scene-renderer`、Pythonのimport名は`scene_renderer`である。
以下では、意図したリポジトリから取得することを明確にするため、GitHub URLを含む
direct referenceを使用する。

### uvを使用する

新しいprojectを作成し、GitHub上の`scene-renderer`を依存関係に追加する。

```bash
mkdir my_project
cd my_project
uv init --python ">=3.10"
uv add "scene-renderer @ git+https://github.com/februa/scene_renderer.git"
```

`uv add`は`pyproject.toml`にGitHub依存を追加し、解決したcommitを`uv.lock`に記録する。
lock fileをversion controlに含め、実行時は`uv run`を使用する。

```bash
uv run python -c "import scene_renderer; print(scene_renderer.__version__)"
```

### venvとpipを使用する

project専用のvirtual environmentを作成し、GitHub URLからインストールする。

```bash
mkdir my_project
cd my_project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install "scene-renderer @ git+https://github.com/februa/scene_renderer.git"
python -c "import scene_renderer; print(scene_renderer.__version__)"
```

requirements fileで管理する場合は、次のdirect referenceを記載する。

```text
scene-renderer @ git+https://github.com/februa/scene_renderer.git
```

### 参照するversionを固定する

長期間再現する必要がある場合は、default branchの最新状態ではなく、
release tagまたはfull commit hashをGitHub URLの末尾に指定する。

```bash
uv add "scene-renderer @ git+https://github.com/februa/scene_renderer.git@FULL_COMMIT_HASH"
```

pipやrequirements fileでも同じdirect referenceを使用できる。

```text
scene-renderer @ git+https://github.com/februa/scene_renderer.git@FULL_COMMIT_HASH
```

### local checkoutをeditable installする

別のprojectを開発しながら`scene_renderer`本体も変更する場合は、先にrepositoryをcloneし、
外部project側からlocal checkoutをeditable dependencyとして登録する。

```bash
git clone https://github.com/februa/scene_renderer.git /path/to/scene_renderer
cd /path/to/my_project
uv add --editable /path/to/scene_renderer
```

pipの場合は、有効化したproject専用virtual environmentにeditable installする。

```bash
python -m pip install --editable /path/to/scene_renderer
```

本リポジトリ自体を開発する場合は、開発用依存も含めてeditable installする。

```bash
git clone https://github.com/februa/scene_renderer.git
cd scene_renderer
python -m pip install --editable ".[dev]"
```

## examples

サンプルコードは [examples](./examples) にあります。

- [examples/example_minimal.py](./examples/example_minimal.py)
  - 最小構成の `Scene + Receiver -> x[ch, t]` を生成する例
- [examples/example_config_api.py](./examples/example_config_api.py)
  - 上位設定 API から `AcousticSource` を組み立てる例
- [examples/example_component_rendering.py](./examples/example_component_rendering.py)
  - RMS SL・ASD NLを指定し、target/interference/noiseを成分別に得る例

最小 example の実行:

```bash
python examples/example_minimal.py
```

設定 API example の実行:

```bash
python examples/example_config_api.py
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

## RMS SL・ASD NLと成分別レンダリング

`SourceComponent.amplitude`は従来どおり線形振幅を直接指定する低水準APIである。実正弦波の
RMS source levelを指定する場合は、RMSとpeakを混同しないため次の生成関数を使う。

```python
component = tone_component_from_rms_level_db(
    frequency_hz=1000.0,
    level_db_re_rms=0.0,
    envelope=ConstantEnvelope(),
)
```

このとき複素toneの振幅は `sqrt(2)` であり、実部の時間RMSが1になる。

背景雑音の`NL`をone-sided amplitude spectral densityとして指定する場合、物理帯域は
`BandLimitedNoiseSpectrum`だけで指定する。帯域幅は`f_high_hz - f_low_hz`から導出され、
別引数との不一致は起こらない。

```python
ambient = AmbientField.from_asd_level_db(
    spectrum=BandLimitedNoiseSpectrum(100.0, 356.0),
    level_db_re_rms_per_sqrt_hz=-32.0,
    noise_seed=1234,
    identifier="ambient",
    role="noise",
)
```

変換は次の定義に従う。

```text
tone RMS amplitude                 = 10^(SL/20)
real-tone peak amplitude           = sqrt(2) * 10^(SL/20)
noise RMS in one-sided band B [Hz] = 10^(NL/20) * sqrt(B)
```

target、interference、noiseなどを同じsceneから分離する場合は`render_components`を使う。

```python
rendered = SceneRenderer(dtype=np.float32).render_components(scene, receiver, axis_t)
x_mixed = rendered.mixed
x_target = rendered.sum_by_role("target")
x_interference = rendered.sum_by_role("interference")
x_noise = rendered.sum_by_role("noise")
```

`SceneRenderer`は寄与の合成だけを担当し、音源波形、伝搬、アレイ投影、背景雑音生成は従来どおり
各renderer/contributorへ委譲する。共分散推定、ビームフォーミング、BL評価は本ライブラリの
責務に含めない。

## 生成信号の定量検証

狭帯域・広帯域・雑音を同じRMS power規約で検証するため、one-sided FFT診断APIを提供する。

```python
metrics = calculate_one_sided_rms_spectrum(real_signal, sampling_frequency_hz=8192.0)
band = evaluate_one_sided_band(
    metrics,
    f_low_hz=900.0,
    f_high_hz=1100.0,
    reference_rms=1.0,
)

print(metrics.time_rms)
print(metrics.spectrum_rms)
print(metrics.parseval_relative_error)
print(band.band_rms)
print(band.band_level_db_re_reference_rms)
print(band.mean_asd_level_db_re_reference_rms_per_sqrt_hz)
```

`rms_power_per_bin`は、DCとNyquistを除くinterior positive binだけを2倍する。全binのpower和は
時間領域mean-squareへ一致する。狭帯域toneも広帯域雑音も、入力が占有する物理帯域のbin powerを
加算してRMSを評価する。

Tone周波数は`0 < f < fs/2`、NoiseSpectrumの上端は`f <= fs/2`を要求する。Nyquist外を暗黙に
aliasまたは切り捨てるとSL/NLと生成levelが不一致になるため、レンダリング時に`ValueError`とする。

## 開発コマンド

```bash
python -m pytest -q
python examples/example_minimal.py
python examples/example_config_api.py
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
