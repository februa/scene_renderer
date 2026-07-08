# AGENTS.md

このリポジトリは、海洋音響・アレイ信号処理・ビームフォーミング・STFT・FIR・SLC などの信号処理コードを扱う。
実装では、動作することだけでなく、設計意図・数式・配列 shape・単位・境界条件が後から読んで分かることを重視する。

## 基本方針

- 識別子を長くしても、コメントを省略してはいけない。
- コメントは原則として日本語で書く。
- docstring も原則として日本語で書く。
- コメントは「コードを読めば分かる処理内容」ではなく、「なぜその処理が必要か」「何を前提としているか」「数式や設計とどう対応するか」を説明するために書く。
- コメント不足のコードは未完成とみなす。

## Codex 作業規約

- 初期実装や仮の実装という言葉を認めません。方式検討であっても、正しく方式を実装したうえで扱うこと。
- AGENTS.md の規約を必ず守ること。Pylance / Pyright の型エラーが残るコードは未完成とみなし、実装後は型エラーが出そうな箇所を自己レビューしてから提示すること。
- 評価時は、スキルの Beamforming Evaluation を使用し、評価が不足したまま方式検討を進めないこと。
- 方式が上手くいかない場合は、まず実装が設計通りかを確認すること。
- 検討結果は設計書を 1 つ作成し、章節を分けて追記していくこと。
- .venv 環境であれば足りないパッケージを pip install することを認める。ただし、その場合は pyproject.toml にも依存関係を追記すること。
- Beamforming Evaluation に対して、ユーザの指摘の方が理屈に合っている場合は、ユーザの指摘に合わせてスキルを更新すること。ユーザの指摘が間違っている場合は、そのように返答すること。
- Agent はユーザの指示に対して、更新結果を stage し、git commit するところまで作業すること。なお、exclude で除外されているファイルはコミット不要とする。

## Python 実装規約

Python コードを書く、または変更する場合は、以下を必ず守る。

### 1. クラス docstring

公開クラスには必ず docstring を書く。

docstring には最低限、次を書く。

- クラスの責務
- 入力と出力の概要
- このクラスが責務として持たないこと
- 信号処理上の位置づけ

例:

```python
class SteeringVectorCalculator:
    """アレイ形状と到来方向からステアリングベクトルを計算する。

    このクラスは、センサ位置、周波数軸、到来方向ベクトルを入力として、
    各チャンネル・各ビーム・各周波数ビンに対応する複素ステアリングベクトルを生成する。

    信号そのものの生成、ビーム出力の計算、SLC 重みの更新は責務に含めない。
    """
```

### 2. 公開メソッド docstring

公開メソッドには必ず docstring を書く。

docstring には最低限、次を書く。

- 引数の意味
- 戻り値の意味
- 配列 shape
- axis の意味
- 単位
- 例外条件
- 境界条件

例:

```python
def calculate(self, positions: np.ndarray, dirs: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """ステアリングベクトルを計算する。

    Args:
        positions: センサ位置。shape は [n_ch, 3]、単位は m。
        dirs: 到来方向ベクトル。shape は [n_beam, 3]。
        freqs: 周波数軸。shape は [n_bin]、単位は Hz。

    Returns:
        複素ステアリングベクトル。shape は [n_ch, n_beam, n_bin]。

    Raises:
        ValueError: positions, dirs, freqs の次元が想定と異なる場合。
    """
```

### 3. 信号処理式のコメント

以下の処理には、必ず数式または物理的意味との対応をコメントで書く。

- ステアリングベクトル
- 到達遅延
- 位相回転
- FFT / IFFT
- STFT
- FIR フィルタ
- overlap-save
- ポリフェーズフィルタバンク
- 忘却係数
- 共分散行列
- 正規化
- ビームフォーミング重み
- SLC / 適応サイドローブキャンセラ
- 対角ローディング
- ガード領域
- 異常時の fallback

例:

```python
# tau[ch, beam] は、センサ位置ベクトルと到来方向ベクトルの内積から求める相対遅延。
# 基準点に対する各センサの到達時刻差を表し、単位は秒。
tau = positions @ dirs.T / sound_speed

# exp(j 2π f tau) により、各周波数ビンで必要な位相回転を与える。
# ここでは時間領域の小数遅延フィルタではなく、周波数領域の位相差として遅延を表現する。
steering = np.exp(1j * 2.0 * np.pi * tau[:, :, None] * freqs[None, None, :])
```

### 4. shape / axis コメント

配列 shape が重要な処理では、必ず shape と axis の意味を書く。

特に以下の操作ではコメントを省略してはいけない。

- reshape
- transpose
- swapaxes
- moveaxis
- squeeze
- expand_dims
- broadcasting
- einsum
- matmul
- stack / concatenate
- FFT の axis 指定

例:

```python
# X shape: [n_ch, n_bin, n_frame]
# axis=0 はセンサチャンネル、axis=1 は周波数ビン、axis=2 は STFT フレームを表す。
X = np.asarray(X)

# ビーム重み W shape: [n_beam, n_ch, n_bin]
# X と周波数ビンを揃えるため、einsum で ch 軸を内積として畳み込む。
Y = np.einsum("bck,ckf->bkf", np.conj(W), X)
```

### 5. マジックナンバー・閾値・安定化項

以下の値を使う場合は、理由をコメントで説明する。

- `1e-12` などの小さい値
- `0.5`
- dB 閾値
- guard 幅
- 忘却係数
- 正則化係数
- 対角ローディング量
- クリップ範囲
- 無効化条件

例:

```python
# 数値安定化のため、対角ローディングを加える。
# スナップショット数が少ない場合や参照ビーム間の相関が高い場合に、
# 共分散行列が特異または悪条件になり、適応重みが発散することを防ぐ。
R_loaded = R + diagonal_loading * np.eye(R.shape[0])
```

### 6. 境界条件・異常時処理

境界条件や異常時処理には、必ず「なぜその挙動にするか」をコメントで書く。

特に以下はコメント必須とする。

- 入力不足時のゼロ詰め
- 初回フレーム処理
- NaN / inf 検出時の処理
- センサ異常時の処理
- 方位センサ異常時の処理
- SLC 無効化条件
- target 数が多すぎる場合の処理
- 参照ビーム不足時の処理
- ガード領域が確保できない場合の処理

例:

```python
# 参照ビームが不足している状態で SLC を更新すると、
# 目標方向の信号までキャンセルする危険がある。
# そのため、この条件では安全側として SLC 更新を停止し、固定整相の出力を採用する。
if n_reference_beams < min_reference_beams:
    return fixed_beam_output
```

### 7. 禁止するコメント

コードを読めば分かるだけのコメントは禁止する。

悪い例:

```python
# i を 1 増やす
i += 1

# 配列を作る
x = np.zeros(n)
```

良い例:

```python
# 初回フレームでは過去サンプルが存在しないため、
# overlap-save の履歴領域をゼロで初期化する。
history = np.zeros(history_length)
```

### 8. コメント密度

- 主要な処理ブロックごとに最低 1 つはコメントを書く。
- 数式に対応する処理には必ずコメントを書く。
- shape が変わる処理には、変換前後の shape をコメントまたは docstring に書く。
- 複雑な関数では、処理段階ごとにコメント見出しを書く。
- コメントが多すぎて処理が読みにくい場合は、関数分割を検討する。

## テスト・検証に関する規約

信号処理コードを変更した場合は、可能な範囲で以下を確認する。

- 入力 shape が想定通りであること
- 出力 shape が想定通りであること
- 単位が混在していないこと
- dB と振幅の変換が正しいこと
- 周波数ビン、サンプル数、FFT 長の対応が正しいこと
- 初回フレーム、最終フレーム、入力不足時の挙動が安全であること
- SLC や適応処理が異常条件で安全側に倒れること

テストコードを書く場合も、なぜその入力条件を選んだかをコメントで説明する。

## 実装後の自己レビュー

コードを出力する前に、必ず以下を確認する。

- クラス docstring があるか
- 公開メソッド docstring があるか
- 非自明な処理にコメントがあるか
- 数式と実装の対応がコメントされているか
- 配列 shape と axis の意味が説明されているか
- 単位が説明されているか
- 境界条件の理由が説明されているか
- 異常時に安全側へ倒す理由が説明されているか
- コメントなしでは設計意図が分からない処理が残っていないか

上記を満たさない場合、実装は完了していないものとして、コメントを追加してから提示する。


## Pylance / Pyright 型チェック規約

このリポジトリでは、Python コードは実行できるだけでは不十分である。
Pylance / Pyright の型チェックでエラーや警告を出さないことを品質条件とする。

Codex は Python コードを作成・修正した後、以下の型チェック観点で自己レビューし、Pylance の型エラーが残るコードを未完成として扱うこと。

### 1. NumPy scalar と Python 組み込み型を混同しない

`np.bool_`、`np.integer`、`np.floating` などの NumPy scalar は、Python の `bool`、`int`、`float` と型上は別物として扱う。

関数が `bool` を要求する場合、NumPy の比較結果をそのまま渡してはいけない。
明示的に `bool(...)` で Python の `bool` に変換する。

悪い例:

```python
require(np.all(mask), "mask must be valid")
```

良い例:

```python
# np.all の戻り値は np.bool_ になるため、require が要求する Python bool へ明示変換する。
require(bool(np.all(mask)), "mask must be valid")
```

### 2. Union 型を曖昧なまま次の関数へ渡さない

戻り値が `ndarray | tuple[ndarray, ndarray]` のような Union 型になる関数では、
呼び出し側で必ず型を分岐・確定してから、次の関数へ渡す。

悪い例:

```python
beam_output = run_beamformer(x)
metrics = evaluate_metrics(beam_output)
```

良い例:

```python
beam_result = run_beamformer(x)

# run_beamformer はデバッグ情報を返す設定では tuple を返すため、
# 評価処理に渡す主出力 ndarray をここで明示的に取り出す。
if isinstance(beam_result, tuple):
    beam_output, debug_info = beam_result
else:
    beam_output = beam_result
    debug_info = None

metrics = evaluate_metrics(beam_output)
```

より望ましい設計は、戻り値の型を設定で変えないことである。
主出力と補助情報が必要な場合は、`dataclass` や専用の結果クラスを使って戻り値を固定する。

例:

```python
@dataclass(frozen=True)
class BeamformerResult:
    """ビームフォーマの出力を表す。

    Attributes:
        beam_output: 主出力。shape は [n_beam, n_bin, n_frame]。
        reference_output: 参照出力。存在しない場合は None。
    """

    beam_output: NDArray[np.complexfloating[Any, Any]]
    reference_output: NDArray[np.complexfloating[Any, Any]] | None = None
```

### 3. 戻り値の型をフラグで変えない

`return_debug=True` のときだけ tuple を返す、という設計は Pylance の型推論を悪化させるため避ける。

避けるべき設計:

```python
def process(x: NDArray[Any], return_debug: bool = False) -> NDArray[Any] | tuple[NDArray[Any], NDArray[Any]]:
    ...
```

推奨する設計:

```python
def process(x: NDArray[Any]) -> ProcessResult:
    ...
```

または、主出力だけを返す関数と、デバッグ情報を含む関数を分ける。

```python
def process(x: NDArray[Any]) -> NDArray[Any]:
    ...

def process_with_debug(x: NDArray[Any]) -> ProcessDebugResult:
    ...
```

### 4. NumPy 配列の型注釈を省略しない

NumPy 配列を扱う関数では、可能な範囲で `numpy.typing.NDArray` を使って型注釈を書く。

例:

```python
from typing import Any

import numpy as np
from numpy.typing import NDArray


def calculate_power(x: NDArray[np.complexfloating[Any, Any]]) -> NDArray[np.floating[Any]]:
    """複素信号のパワーを計算する。

    Args:
        x: 複素信号。shape は [n_ch, n_sample]。

    Returns:
        パワー。shape は [n_ch, n_sample]。
    """

    return np.abs(x) ** 2
```

型が複雑すぎる場合でも、少なくとも `NDArray[Any]` を使い、`ndarray` の裸型や無注釈を避ける。

### 5. Optional を未確認のまま使わない

`None` の可能性がある値は、使用前に必ず `is None` / `is not None` で分岐する。
`assert value is not None` を使う場合は、実行時にもその前提が正しいことをコメントで説明する。

例:

```python
if config.reference_beams is None:
    # 参照ビームがない場合、SLC を安全側として無効化する。
    return fixed_output

reference_beams = config.reference_beams
```

### 6. 型を握りつぶす cast を乱用しない

`typing.cast` は最後の手段とする。
Pylance エラーを消すためだけの `cast` を使ってはいけない。

`cast` を使う場合は、直前に実行時検証を行い、なぜ安全に cast できるかを日本語コメントで説明する。

例:

```python
if not isinstance(value, np.ndarray):
    raise TypeError("value must be np.ndarray")

# 上の isinstance で ndarray であることを実行時に確認したため、
# ここでは Pylance に対して型を明示する目的で cast する。
array_value = cast(NDArray[Any], value)
```

### 7. Any を広げすぎない

`Any` は外部入力、設定ファイル読み込み、NumPy の複雑な dtype 表現など、型を厳密に書くことが難しい箇所に限定する。
一度検証した値は、できるだけ具体的な型の変数へ代入し直す。

例:

```python
raw_nfft: Any = params.get("nfft")
if not isinstance(raw_nfft, int):
    raise TypeError("nfft must be int")

# ここから先は int として扱う。
nfft: int = raw_nfft
```

### 8. Pylance 対応の自己レビュー

コードを出力する前に、以下を確認する。

- `np.bool_` を `bool` 引数へ直接渡していないか
- `np.integer` / `np.floating` を `int` / `float` 引数へ直接渡していないか
- `ndarray | tuple[...]` のような Union 型を未分岐のまま使っていないか
- `Optional` を `None` チェックなしで使っていないか
- 関数の戻り値型がフラグによって変化していないか
- `NDArray` の型注釈を省略していないか
- `cast` で型エラーを隠していないか
- Pylance の型エラーが残る可能性がある箇所に、実行時の型分岐または設計修正を入れたか

Pylance / Pyright の型エラーが残るコードは未完成とみなし、提示前に修正する。

## Codex への作業指示

このリポジトリで Python コードを作成・修正する場合は、常にこの `AGENTS.md` の規約に従うこと。

特に、信号処理、ビームフォーミング、STFT、FIR、SLC、フィルタバンク、配列 shape 変換に関するコードでは、日本語コメントを必須とする。

また、Pylance / Pyright の型チェック規約を満たすこと。実行上は問題がないコードであっても、Pylance の型エラーが残るコードは未完成とみなし、提示前に修正すること。



