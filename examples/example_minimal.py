from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene_renderer import (
    Scene,
    AcousticSource,
    SourceComponent,
    ToneSpectrum,
    ConstantEnvelope,
    FreeField,
    StaticPose,
    Receiver,
    LinearArray,
    SceneRenderer,
)


def main() -> None:
    receiver = Receiver(
        trajectory=StaticPose(position_world=[0.0, 0.0, 0.0], heading_deg=0.0),
        array=LinearArray(n_ch=32, spacing=0.075),
    )

    component = SourceComponent(
        spectrum=ToneSpectrum(1000.0),
        envelope=ConstantEnvelope(),
        amplitude=1.0,
    )

    scene = Scene(
        sources=[
            AcousticSource.from_relative_bearing(
                bearing_deg=90.0,
                distance=1000.0,
                receiver_pose=receiver.trajectory.pose(0.0),
                components=[component],
                elevation_deg=0.0,
            )
        ],
        ambient_fields=[],
        environment=FreeField(c=1500.0),
    )

    axis = np.arange(32768) / 32768
    x = SceneRenderer().render(scene, receiver, axis)

    print("x.shape =", x.shape)
    print("x.dtype =", x.dtype)
    assert x.shape == (32, 32768)


if __name__ == "__main__":
    main()
