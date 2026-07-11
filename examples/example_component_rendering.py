"""RMS SL・ASD NLを指定し、音響sceneの成分別受信信号を得る例。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene_renderer import (  # noqa: E402
    AcousticSource,
    AmbientField,
    BandLimitedNoiseSpectrum,
    ConstantEnvelope,
    FreeField,
    LinearArray,
    Receiver,
    Scene,
    SceneRenderer,
    StaticPose,
    tone_component_from_rms_level_db,
)


def main() -> None:
    """target、interference、noise、mixedの多CH信号を生成する。"""

    fs = 8192.0
    axis_t = np.arange(8192, dtype=float) / fs
    receiver = Receiver(StaticPose([0.0, 0.0, 0.0]), LinearArray(n_ch=8, spacing=0.075))
    receiver_pose = receiver.trajectory.pose(0.0)
    target = AcousticSource.from_relative_bearing(
        20.0,
        1000.0,
        receiver_pose,
        [tone_component_from_rms_level_db(1000.0, 0.0, ConstantEnvelope())],
        identifier="target",
        role="target",
    )
    interferer = AcousticSource.from_relative_bearing(
        -35.0,
        1000.0,
        receiver_pose,
        [tone_component_from_rms_level_db(1400.0, -6.0, ConstantEnvelope())],
        identifier="interferer",
        role="interference",
    )
    ambient = AmbientField.from_asd_level_db(
        spectrum=BandLimitedNoiseSpectrum(100.0, 3000.0),
        level_db_re_rms_per_sqrt_hz=-32.0,
        noise_seed=1234,
        identifier="ambient",
        role="noise",
    )
    scene = Scene([target, interferer], [ambient], FreeField(1500.0))
    rendered = SceneRenderer(dtype=np.float32).render_components(scene, receiver, axis_t)

    # 後段の共分散推定やビームフォーミングへ、同じ[ch,t]格子の成分を明示的に渡せる。
    print("mixed", rendered.mixed.shape)
    print("target", rendered.sum_by_role("target").shape)
    print("interference", rendered.sum_by_role("interference").shape)
    print("noise", rendered.sum_by_role("noise").shape)


if __name__ == "__main__":
    main()
