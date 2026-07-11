"""音響sceneのレベル定義と成分分離を検証する。"""

from __future__ import annotations

import inspect

import numpy as np

from scene_renderer import (
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
    noise_asd_level_db_to_band_rms,
    tone_component_from_rms_level_db,
    tone_rms_level_db_to_peak_amplitude,
    white_noise_asd_level_db_to_sample_rms,
)


def _receiver(n_ch: int = 4) -> Receiver:
    """既知の共分散と位相を確認しやすい固定ULAを返す。"""

    return Receiver(StaticPose([0.0, 0.0, 0.0]), LinearArray(n_ch=n_ch, spacing=0.1))


def test_level_conversions_follow_rms_and_one_sided_asd_definitions() -> None:
    """SL=0のpeakと、NLから帯域積分するRMSが解析式に一致する。"""

    np.testing.assert_allclose(tone_rms_level_db_to_peak_amplitude(0.0), np.sqrt(2.0))
    np.testing.assert_allclose(noise_asd_level_db_to_band_rms(-32.0, 256.0), 10.0 ** (-32.0 / 20.0) * 16.0)
    np.testing.assert_allclose(
        white_noise_asd_level_db_to_sample_rms(-32.0, 8192.0),
        10.0 ** (-32.0 / 20.0) * np.sqrt(8192.0 / 2.0),
    )


def test_tone_rms_factory_produces_unit_rms_real_waveform() -> None:
    """整数bin toneを実部化した時間RMSがSL=0の1になる。"""

    fs = 8192.0
    axis_t = np.arange(8192, dtype=float) / fs
    receiver = _receiver(n_ch=1)
    source = AcousticSource.from_relative_bearing(
        0.0,
        1000.0,
        receiver.trajectory.pose(0.0),
        [tone_component_from_rms_level_db(1024.0, 0.0, ConstantEnvelope())],
        identifier="target",
        role="target",
    )
    rendered = SceneRenderer(dtype=np.float32).render_components(
        Scene([source], [], FreeField(1500.0)), receiver, axis_t
    )
    np.testing.assert_allclose(np.sqrt(np.mean(rendered.mixed[0] ** 2)), 1.0, atol=1.0e-6)


def test_rendered_scene_components_sum_to_mixed_and_select_by_role() -> None:
    """target/interferenceを一度の公開描画で分離し、総和がmixedと一致する。"""

    fs = 4096.0
    axis_t = np.arange(4096, dtype=float) / fs
    receiver = _receiver(n_ch=3)
    target = AcousticSource.from_relative_bearing(
        20.0,
        1000.0,
        receiver.trajectory.pose(0.0),
        [tone_component_from_rms_level_db(512.0, 0.0, ConstantEnvelope())],
        identifier="target",
        role="target",
    )
    interferer = AcousticSource.from_relative_bearing(
        -30.0,
        1000.0,
        receiver.trajectory.pose(0.0),
        [tone_component_from_rms_level_db(768.0, -6.0, ConstantEnvelope())],
        identifier="interferer",
        role="interference",
    )
    rendered = SceneRenderer().render_components(
        Scene([target, interferer], [], FreeField(1500.0)), receiver, axis_t
    )
    component_sum = np.sum(np.stack([item.signal for item in rendered.components], axis=0), axis=0)
    np.testing.assert_allclose(component_sum, rendered.mixed, atol=2.0e-6)
    assert np.max(np.abs(rendered.sum_by_role("target"))) > 0.0
    assert np.max(np.abs(rendered.sum_by_role("noise"))) == 0.0


def test_ambient_noise_is_deterministic_and_follows_covariance() -> None:
    """背景雑音のchunk再現性とCH間共分散が指定値へ統計的に近づく。"""

    fs = 4096.0
    n_sample = 65536
    axis_t = np.arange(n_sample, dtype=float) / fs
    receiver = _receiver(n_ch=2)
    covariance = np.array([[1.0, 0.6], [0.6, 1.0]])
    field = AmbientField(
        spectrum=BandLimitedNoiseSpectrum(50.0, 1800.0),
        amplitude=2.0,
        covariance=covariance,
        noise_seed=1234,
        identifier="ambient_white",
        role="noise",
    )
    renderer = SceneRenderer(dtype=np.float32)
    rendered = renderer.render_components(Scene([], [field], FreeField(1500.0)), receiver, axis_t)
    repeated = renderer.render_components(Scene([], [field], FreeField(1500.0)), receiver, axis_t)
    np.testing.assert_array_equal(rendered.mixed, repeated.mixed)
    observed_covariance = np.cov(rendered.mixed)
    # 有限sampleの確率誤差を許しつつ、amplitude^2 Rという物理的な共分散規約を確認する。
    np.testing.assert_allclose(observed_covariance, 4.0 * covariance, atol=0.12)


def test_ambient_noise_is_chunk_partition_independent() -> None:
    """絶対sample index生成により、背景雑音がchunk境界で変化しない。"""

    fs = 2048.0
    receiver = _receiver(n_ch=2)
    field = AmbientField(
        spectrum=BandLimitedNoiseSpectrum(100.0, 800.0),
        amplitude=1.0,
        noise_seed=77,
    )
    scene = Scene([], [field], FreeField(1500.0))
    renderer = SceneRenderer(dtype=np.float32)
    full = renderer.render(scene, receiver, np.arange(2048, dtype=float) / fs)
    chunks = [
        renderer.render(scene, receiver, (start + np.arange(256, dtype=float)) / fs)
        for start in range(0, 2048, 256)
    ]
    np.testing.assert_allclose(np.concatenate(chunks, axis=1), full, atol=1.0e-6)


def test_ambient_field_asd_factory_keeps_bandwidth_definition_explicit() -> None:
    """ASD指定APIがSpectrumだけを帯域の正本としてRMSを決める。"""

    field = AmbientField.from_asd_level_db(
        spectrum=BandLimitedNoiseSpectrum(100.0, 356.0),
        level_db_re_rms_per_sqrt_hz=-32.0,
        noise_seed=10,
    )
    np.testing.assert_allclose(field.amplitude, 10.0 ** (-32.0 / 20.0) * np.sqrt(256.0))
    assert "bandwidth_hz" not in inspect.signature(AmbientField.from_asd_level_db).parameters


def test_ambient_noise_observed_asd_matches_input_level() -> None:
    """one-sided FFTの帯域内平均ASDが入力NLへ一致する。"""

    fs = 4096.0
    n_sample = 65536
    receiver = _receiver(n_ch=1)
    field = AmbientField.from_asd_level_db(
        spectrum=BandLimitedNoiseSpectrum(256.0, 1280.0),
        level_db_re_rms_per_sqrt_hz=-32.0,
        noise_seed=2468,
        noise_filter_length=513,
    )
    signal = SceneRenderer(dtype=np.float32).render(
        Scene([], [field], FreeField(1500.0)), receiver, np.arange(n_sample, dtype=float) / fs
    )[0]
    spectrum = np.fft.rfft(signal)
    frequencies_hz = np.fft.rfftfreq(n_sample, d=1.0 / fs)
    # interior one-sided binのPSDは2|X[k]|^2/(N fs)。帯域内で平均してASD levelへ戻す。
    one_sided_psd = 2.0 * np.abs(spectrum) ** 2 / (n_sample * fs)
    band_mask = (frequencies_hz >= 256.0) & (frequencies_hz <= 1280.0)
    observed_level_db = 10.0 * np.log10(np.mean(one_sided_psd[band_mask]))
    # 有限sampleの統計変動と有限長FIRの遷移帯域を含むため、0.3 dBの許容幅で物理基準を確認する。
    np.testing.assert_allclose(observed_level_db, -32.0, atol=0.3)
