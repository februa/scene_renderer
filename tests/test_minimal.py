from __future__ import annotations

import numpy as np
import pytest

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
from scene_renderer.scene import AmbientField


def make_component(f: float = 1000.0) -> SourceComponent:
    return SourceComponent(ToneSpectrum(f), ConstantEnvelope(), amplitude=1.0)


def test_render_shape() -> None:
    receiver = Receiver(StaticPose([0.0, 0.0, 0.0], heading_deg=0.0), LinearArray(32, 0.075))
    scene = Scene(
        sources=[AcousticSource.from_relative_bearing(90.0, 1000.0, receiver.trajectory.pose(0.0), [make_component()])],
        ambient_fields=[],
        environment=FreeField(1500.0),
    )
    axis = np.arange(1024) / 32768
    x = SceneRenderer().render(scene, receiver, axis)
    assert x.shape == (32, 1024)
    assert x.dtype == np.complex64
    assert np.iscomplexobj(x)


def test_absolute_bearing_position_east() -> None:
    receiver_pose = StaticPose([0.0, 0.0, 0.0], heading_deg=0.0).pose(0.0)
    source = AcousticSource.from_absolute_bearing(90.0, 1000.0, receiver_pose, [make_component()])
    np.testing.assert_allclose(source.trajectory.position(0.0), [1000.0, 0.0, 0.0], atol=1e-12)


def test_relative_bearing_heading0_starboard_is_east() -> None:
    receiver_pose = StaticPose([0.0, 0.0, 0.0], heading_deg=0.0).pose(0.0)
    source = AcousticSource.from_relative_bearing(90.0, 1000.0, receiver_pose, [make_component()])
    np.testing.assert_allclose(source.trajectory.position(0.0), [1000.0, 0.0, 0.0], atol=1e-12)


def test_relative_bearing_heading90_starboard_is_south() -> None:
    receiver_pose = StaticPose([0.0, 0.0, 0.0], heading_deg=90.0).pose(0.0)
    source = AcousticSource.from_relative_bearing(90.0, 1000.0, receiver_pose, [make_component()])
    np.testing.assert_allclose(source.trajectory.position(0.0), [0.0, -1000.0, 0.0], atol=1e-12)


def test_adjacent_phase_difference() -> None:
    f = 1000.0
    c = 1500.0
    d = 0.075
    receiver = Receiver(
        StaticPose([0.0, 0.0, 0.0], heading_deg=0.0),
        LinearArray(n_ch=2, spacing=d, axis=1, centered=False),
    )
    scene = Scene(
        sources=[AcousticSource.from_relative_bearing(90.0, 1000.0, receiver.trajectory.pose(0.0), [make_component(f)])],
        ambient_fields=[],
        environment=FreeField(c),
    )
    axis = np.arange(8) / 32768
    x = SceneRenderer().render(scene, receiver, axis)

    expected_phase = -2.0 * np.pi * f * d / c
    measured_phase = np.angle(x[1, 0] / x[0, 0])
    np.testing.assert_allclose(measured_phase, expected_phase, atol=1e-6)


def test_relative_bearing_elevation_sets_z() -> None:
    receiver_pose = StaticPose([0.0, 0.0, 0.0], heading_deg=0.0).pose(0.0)
    source = AcousticSource.from_relative_bearing(
        bearing_deg=90.0,
        distance=1000.0,
        receiver_pose=receiver_pose,
        components=[make_component()],
        elevation_deg=30.0,
    )
    expected = [1000.0 * np.cos(np.deg2rad(30.0)), 0.0, 1000.0 * np.sin(np.deg2rad(30.0))]
    np.testing.assert_allclose(source.trajectory.position(0.0), expected, atol=1e-12)


def test_source_component_rejects_amplitude_and_level_db() -> None:
    with pytest.raises(ValueError):
        SourceComponent(ToneSpectrum(1000.0), ConstantEnvelope(), amplitude=1.0, level_db=0.0)


def test_source_component_level_db20_compatibility() -> None:
    component = SourceComponent.from_level_db20(ToneSpectrum(1000.0), ConstantEnvelope(), level_db=6.0)
    np.testing.assert_allclose(component.amplitude_value, 10 ** (6.0 / 20.0), atol=1e-12)


def test_scene_renderer_rejects_empty_axis() -> None:
    receiver = Receiver(StaticPose([0.0, 0.0, 0.0]), LinearArray(2, 0.1))
    scene = Scene([], [], FreeField(1500.0))
    with pytest.raises(ValueError):
        SceneRenderer().render(scene, receiver, np.array([]))


def test_scene_renderer_rejects_non_monotonic_axis() -> None:
    receiver = Receiver(StaticPose([0.0, 0.0, 0.0]), LinearArray(2, 0.1))
    scene = Scene([], [], FreeField(1500.0))
    with pytest.raises(ValueError):
        SceneRenderer().render(scene, receiver, np.array([0.0, 0.1, 0.05]))


def test_scene_renderer_rejects_non_uniform_axis() -> None:
    receiver = Receiver(StaticPose([0.0, 0.0, 0.0]), LinearArray(2, 0.1))
    scene = Scene([], [], FreeField(1500.0))
    with pytest.raises(ValueError):
        SceneRenderer().render(scene, receiver, np.array([0.0, 0.1, 0.25]))


def test_ambient_field_rejects_non_symmetric_covariance() -> None:
    with pytest.raises(ValueError):
        AmbientField(
            spectrum=ToneSpectrum(1000.0),
            amplitude=1.0,
            covariance=np.array([[1.0, 1.0], [0.0, 1.0]]),
        )


def test_ambient_field_rejects_non_positive_semidefinite_covariance() -> None:
    with pytest.raises(ValueError):
        AmbientField(
            spectrum=ToneSpectrum(1000.0),
            amplitude=1.0,
            covariance=np.array([[1.0, 2.0], [2.0, 1.0]]),
        )
