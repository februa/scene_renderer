from __future__ import annotations

import numpy as np
import pytest

from scene_renderer import (
    AcousticSource,
    BandLimitedNoiseSpectrum,
    ConstantEnvelope,
    FreeField,
    LinearArray,
    Receiver,
    Scene,
    PinkNoiseSpectrum,
    SceneRenderer,
    SourceComponent,
    StaticPose,
    ToneSpectrum,
)
from scene_renderer.scene import AmbientField
from scene_renderer.renderer import SensorNoiseGenerator


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


def test_scene_renderer_rejects_nan_axis() -> None:
    receiver = Receiver(StaticPose([0.0, 0.0, 0.0]), LinearArray(2, 0.1))
    scene = Scene([], [], FreeField(1500.0))
    # NaN を含む時間軸では fs と位相回転 exp(j 2π f t) が定義できないため、入口で拒否する。
    with pytest.raises(ValueError):
        SceneRenderer().render(scene, receiver, np.array([0.0, np.nan, 0.2]))


def test_sensor_noise_generator_rejects_negative_amplitude() -> None:
    # センサ雑音振幅は将来の標準偏差として扱う量なので、負値は無音化ではなく入力エラーにする。
    with pytest.raises(ValueError):
        SensorNoiseGenerator(amplitude=np.array([0.0, -1.0]))



def test_band_limited_noise_spectrum_response_mask() -> None:
    spectrum = BandLimitedNoiseSpectrum(f_low_hz=100.0, f_high_hz=300.0)
    freq_axis = np.array([0.0, 100.0, 200.0, 300.0, 400.0])
    # NoiseSpectrum.evaluate は FIR 設計用の振幅応答なので、通過帯域内だけ 1 になることを確認する。
    np.testing.assert_allclose(np.abs(spectrum.evaluate(freq_axis)), [0.0, 1.0, 1.0, 1.0, 0.0])


def test_pink_noise_spectrum_uses_amplitude_inverse_sqrt_frequency() -> None:
    spectrum = PinkNoiseSpectrum(f_low_hz=100.0, f_high_hz=1000.0, reference_frequency_hz=100.0)
    freq_axis = np.array([100.0, 400.0])
    response = np.abs(spectrum.evaluate(freq_axis))
    # PSD 1/f のピンクノイズでは、振幅応答は 1/sqrt(f) になるため 100 Hz は 400 Hz の 2 倍になる。
    np.testing.assert_allclose(response[0] / response[1], 2.0, atol=1e-12)


def test_noise_spectrum_requires_seed() -> None:
    # 広帯域ノイズは sample index と seed で決定論的に生成するため、seed 未指定は API エラーにする。
    with pytest.raises(ValueError):
        SourceComponent(BandLimitedNoiseSpectrum(100.0, 300.0), ConstantEnvelope(), amplitude=1.0)


def test_broadband_render_is_chunk_partition_independent() -> None:
    fs = 4096.0
    n_sample = 1024
    chunk_size = 256
    receiver = Receiver(
        StaticPose([0.0, 0.0, 0.0], heading_deg=0.0),
        LinearArray(n_ch=3, spacing=0.15, axis=1, centered=True),
    )
    component = SourceComponent(
        BandLimitedNoiseSpectrum(f_low_hz=200.0, f_high_hz=900.0),
        ConstantEnvelope(),
        amplitude=0.5,
        noise_seed=12345,
        noise_filter_length=129,
    )
    scene = Scene(
        sources=[AcousticSource.from_relative_bearing(90.0, 1000.0, receiver.trajectory.pose(0.0), [component])],
        ambient_fields=[],
        environment=FreeField(1500.0),
    )
    full_axis = np.arange(n_sample) / fs
    full = SceneRenderer().render(scene, receiver, full_axis)

    chunks = []
    for start in range(0, n_sample, chunk_size):
        # 呼び出し側は絶対 sample index に対応する axis_t を渡す。ノイズ値と FIR halo は renderer 側で決定論的に補う。
        chunk_axis = (start + np.arange(chunk_size)) / fs
        chunks.append(SceneRenderer().render(scene, receiver, chunk_axis))
    chunked = np.concatenate(chunks, axis=1)
    np.testing.assert_allclose(chunked, full, atol=2e-5)
    assert full.shape == (3, n_sample)
    assert full.dtype == np.complex64
    assert float(np.max(np.abs(full))) > 0.0
