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


def make_source(cfg_signal: dict, receiver: Receiver) -> AcousticSource:
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


def main() -> None:
    fs = 32768.0
    sound_speed = 1500.0

    n_ch = 32
    spacing = sound_speed / (2 * fs)

    receiver = Receiver(
        trajectory=StaticPose(position_world=[0.0, 0.0, 0.0], heading_deg=0.0),
        array=LinearArray(n_ch=n_ch, spacing=spacing),
    )

    cfg_signal = [
        {
            "spectrum": ToneSpectrum(1000.0),
            "envelope": ConstantEnvelope(),
            "amplitude": 1.0,
            "dir_azimuth_deg": 90.0,
            "dir_elevation_deg": 0.0,
            "distance": 1000.0,
        }
    ]

    scene = Scene(
        sources=[make_source(cfg, receiver) for cfg in cfg_signal],
        ambient_fields=[],
        environment=FreeField(c=sound_speed),
    )

    axis = np.arange(32768) / fs
    x = SceneRenderer().render(scene, receiver, axis)

    print("spacing =", spacing)
    print("x.shape =", x.shape)


if __name__ == "__main__":
    main()
