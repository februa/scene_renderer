"""狭帯域・広帯域・雑音の生成信号を物理量とスペクトルで検証する。"""

from __future__ import annotations

import numpy as np
import pytest

from scene_renderer import (
    AcousticSource,
    AmbientField,
    BandLimitedNoiseSpectrum,
    ConstantEnvelope,
    FreeField,
    LinearArray,
    PinkNoiseSpectrum,
    Receiver,
    Scene,
    SceneRenderer,
    SourceComponent,
    StaticPose,
    ToneSpectrum,
    calculate_one_sided_rms_spectrum,
    evaluate_one_sided_band,
    tone_component_from_rms_level_db,
)


def _single_channel_receiver() -> Receiver:
    """アレイ投影利得を含めず生成波形levelを評価する1CH受波器を返す。"""

    return Receiver(StaticPose([0.0, 0.0, 0.0]), LinearArray(n_ch=1, spacing=0.1))


def _source(receiver: Receiver, component: SourceComponent) -> AcousticSource:
    """基準波形とCH0の位相が一致するbroadside固定音源を返す。"""

    return AcousticSource.from_relative_bearing(
        0.0,
        1000.0,
        receiver.trajectory.pose(0.0),
        [component],
        identifier="target",
        role="target",
    )


def test_one_sided_rms_spectrum_satisfies_parseval_for_arbitrary_real_signal() -> None:
    """one-sided bin power総和が時間領域mean-squareへ丸め誤差内で一致する。"""

    signal = np.random.default_rng(123).standard_normal(4097)
    metrics = calculate_one_sided_rms_spectrum(signal, 8192.0)
    np.testing.assert_allclose(metrics.spectrum_rms, metrics.time_rms, rtol=1.0e-12)
    assert metrics.parseval_relative_error < 1.0e-12


def test_integer_bin_tone_matches_time_rms_single_bin_power_and_source_level() -> None:
    """SL=0 dB re amplitude 1 RMSの整数bin toneが時間・周波数の両方でRMS 1になる。"""

    fs = 8192.0
    n_sample = 8192
    frequency_hz = 1024.0
    receiver = _single_channel_receiver()
    component = tone_component_from_rms_level_db(frequency_hz, 0.0, ConstantEnvelope())
    analytic = SceneRenderer().render(
        Scene([_source(receiver, component)], [], FreeField(1500.0)),
        receiver,
        np.arange(n_sample, dtype=float) / fs,
    )[0]
    metrics = calculate_one_sided_rms_spectrum(np.real(analytic), fs)
    band = evaluate_one_sided_band(metrics, frequency_hz, frequency_hz)
    np.testing.assert_allclose(metrics.time_rms, 1.0, atol=1.0e-6)
    np.testing.assert_allclose(band.band_rms, 1.0, atol=1.0e-6)
    np.testing.assert_allclose(band.band_level_db_re_reference_rms, 0.0, atol=1.0e-6)
    assert int(np.argmax(metrics.rms_power_per_bin)) == int(round(frequency_hz / metrics.frequency_resolution_hz))


@pytest.mark.parametrize("n_sample", [8192, 16384])
def test_non_integer_bin_tone_preserves_band_integrated_rms(n_sample: int) -> None:
    """非整数bin toneでも漏れを含む物理帯域積分RMSがFFT長に依存せずSLへ一致する。"""

    fs = 8192.0
    frequency_hz = 1000.37
    receiver = _single_channel_receiver()
    component = tone_component_from_rms_level_db(frequency_hz, -6.0, ConstantEnvelope())
    analytic = SceneRenderer().render(
        Scene([_source(receiver, component)], [], FreeField(1500.0)),
        receiver,
        np.arange(n_sample, dtype=float) / fs,
    )[0]
    metrics = calculate_one_sided_rms_spectrum(np.real(analytic), fs)
    band = evaluate_one_sided_band(metrics, frequency_hz - 128.0, frequency_hz + 128.0)
    expected_rms = 10.0 ** (-6.0 / 20.0)
    np.testing.assert_allclose(metrics.time_rms, expected_rms, rtol=2.0e-4)
    np.testing.assert_allclose(band.band_rms, expected_rms, rtol=1.5e-3)


def test_broadband_source_preserves_requested_rms_and_occupied_band_power() -> None:
    """帯域雑音音源の時間RMSとFFT帯域積分RMSが指定amplitudeへ一致する。"""

    fs = 4096.0
    n_sample = 65536
    receiver = _single_channel_receiver()
    component = SourceComponent(
        spectrum=BandLimitedNoiseSpectrum(300.0, 1500.0),
        envelope=ConstantEnvelope(),
        amplitude=0.75,
        noise_seed=321,
        noise_filter_length=513,
    )
    rendered = SceneRenderer().render(
        Scene([_source(receiver, component)], [], FreeField(1500.0)),
        receiver,
        np.arange(n_sample, dtype=float) / fs,
    )[0]
    metrics = calculate_one_sided_rms_spectrum(np.real(rendered), fs)
    band = evaluate_one_sided_band(metrics, 300.0, 1500.0)
    np.testing.assert_allclose(metrics.time_rms, 0.75, rtol=0.02)
    # 有限長FIRの遷移帯域に漏れるpowerだけを許し、occupied bandがほぼ全powerを説明することを確認する。
    assert band.band_rms / metrics.time_rms > 0.995


@pytest.mark.parametrize("noise_seed", [101, 202, 303])
def test_ambient_noise_matches_asd_band_rms_and_parseval_for_multiple_seeds(noise_seed: int) -> None:
    """複数seedでNL、帯域RMS、時間RMSが同じ物理基準へ一致する。"""

    fs = 4096.0
    n_sample = 65536
    f_low_hz = 256.0
    f_high_hz = 1280.0
    nl_db = -32.0
    receiver = _single_channel_receiver()
    field = AmbientField.from_asd_level_db(
        BandLimitedNoiseSpectrum(f_low_hz, f_high_hz),
        nl_db,
        noise_seed=noise_seed,
        noise_filter_length=513,
    )
    signal = SceneRenderer(dtype=np.float32).render(
        Scene([], [field], FreeField(1500.0)), receiver, np.arange(n_sample, dtype=float) / fs
    )[0]
    metrics = calculate_one_sided_rms_spectrum(signal, fs)
    band = evaluate_one_sided_band(metrics, f_low_hz, f_high_hz)
    expected_rms = 10.0 ** (nl_db / 20.0) * np.sqrt(f_high_hz - f_low_hz)
    np.testing.assert_allclose(metrics.time_rms, expected_rms, rtol=0.02)
    np.testing.assert_allclose(band.band_rms, expected_rms, rtol=0.02)
    np.testing.assert_allclose(
        band.mean_asd_level_db_re_reference_rms_per_sqrt_hz,
        nl_db,
        atol=0.25,
    )
    assert metrics.parseval_relative_error < 1.0e-12


def test_source_ambient_and_mixed_components_are_additively_consistent() -> None:
    """targetとambientを含むsceneで成分和がmixedと一致し、role別powerが説明可能である。"""

    fs = 4096.0
    n_sample = 8192
    receiver = _single_channel_receiver()
    source = _source(
        receiver,
        tone_component_from_rms_level_db(512.0, 0.0, ConstantEnvelope()),
    )
    ambient = AmbientField.from_asd_level_db(
        BandLimitedNoiseSpectrum(700.0, 1500.0),
        -40.0,
        noise_seed=88,
        identifier="ambient",
        role="noise",
    )
    # target解析信号を実部化するため、SceneRendererの警告契約を明示的に受ける。
    with pytest.warns(UserWarning, match="imaginary parts are discarded"):
        rendered = SceneRenderer(dtype=np.float32).render_components(
            Scene([source], [ambient], FreeField(1500.0)),
            receiver,
            np.arange(n_sample, dtype=float) / fs,
        )
    target = rendered.sum_by_role("target")
    noise = rendered.sum_by_role("noise")
    np.testing.assert_allclose(target + noise, rendered.mixed, atol=2.0e-6)
    assert float(np.sqrt(np.mean(target * target))) > float(np.sqrt(np.mean(noise * noise)))


def test_renderer_rejects_tone_and_noise_above_nyquist() -> None:
    """表現不能帯域を暗黙にaliasまたは切捨てせずscene条件エラーとして拒否する。"""

    fs = 4096.0
    receiver = _single_channel_receiver()
    axis_t = np.arange(1024, dtype=float) / fs
    tone = _source(
        receiver,
        SourceComponent(ToneSpectrum(3000.0), ConstantEnvelope(), amplitude=1.0),
    )
    with pytest.raises(ValueError, match="Nyquist"):
        SceneRenderer().render(Scene([tone], [], FreeField(1500.0)), receiver, axis_t)

    ambient = AmbientField.from_asd_level_db(
        BandLimitedNoiseSpectrum(100.0, 3000.0),
        -32.0,
        noise_seed=1,
    )
    with pytest.raises(ValueError, match="Nyquist"):
        SceneRenderer().render(Scene([], [ambient], FreeField(1500.0)), receiver, axis_t)


@pytest.mark.parametrize(
    ("bearing_deg", "frequency_hz"),
    [(0.0, 500.0), (90.0, 1000.0), (-45.0, 1500.0)],
)
def test_narrowband_channel_phase_matches_geometry_for_multiple_conditions(
    bearing_deg: float, frequency_hz: float
) -> None:
    """複数方位・周波数でCH間位相がexp(-j2πfτ)の解析値へ一致する。"""

    fs = 8192.0
    sound_speed = 1500.0
    spacing_m = 0.1
    receiver = Receiver(
        StaticPose([0.0, 0.0, 0.0]),
        LinearArray(n_ch=2, spacing=spacing_m, axis=1, centered=False),
    )
    component = SourceComponent(ToneSpectrum(frequency_hz), ConstantEnvelope(), amplitude=1.0)
    source = AcousticSource.from_relative_bearing(
        bearing_deg,
        1000.0,
        receiver.trajectory.pose(0.0),
        [component],
    )
    signal = SceneRenderer().render(
        Scene([source], [], FreeField(sound_speed)), receiver, np.arange(32, dtype=float) / fs
    )
    expected_delay_s = spacing_m * np.sin(np.deg2rad(bearing_deg)) / sound_speed
    expected_phase_rad = -2.0 * np.pi * frequency_hz * expected_delay_s
    measured_phase_rad = float(np.angle(signal[1, 0] / signal[0, 0]))
    # 位相差を[-π,π]へwrapして、角度表現の2π不定性を除いて比較する。
    wrapped_error_rad = float(np.angle(np.exp(1j * (measured_phase_rad - expected_phase_rad))))
    np.testing.assert_allclose(wrapped_error_rad, 0.0, atol=1.0e-6)


def test_broadband_channel_delay_matches_geometry_in_samples() -> None:
    """広帯域波面のCH間時間遅延がr・u/cの解析値へsample単位で一致する。"""

    fs = 4096.0
    n_sample = 8192
    sound_speed = 1024.0
    spacing_m = 0.5
    expected_delay_samples = 2
    receiver = Receiver(
        StaticPose([0.0, 0.0, 0.0]),
        LinearArray(n_ch=2, spacing=spacing_m, axis=1, centered=False),
    )
    component = SourceComponent(
        BandLimitedNoiseSpectrum(300.0, 1500.0),
        ConstantEnvelope(),
        amplitude=1.0,
        noise_seed=12,
        noise_filter_length=257,
    )
    source = AcousticSource.from_relative_bearing(
        90.0,
        1000.0,
        receiver.trajectory.pose(0.0),
        [component],
    )
    signal = np.real(
        SceneRenderer().render(
            Scene([source], [], FreeField(sound_speed)),
            receiver,
            np.arange(n_sample, dtype=float) / fs,
        )
    )
    cross_correlation = np.correlate(signal[1], signal[0], mode="full")
    measured_delay_samples = int(np.argmax(cross_correlation) - (n_sample - 1))
    assert measured_delay_samples == expected_delay_samples
    # 整数sample遅延条件ではfractional-delay FIRを通しても対応sampleが数値的に一致する。
    np.testing.assert_allclose(
        signal[1, expected_delay_samples:],
        signal[0, :-expected_delay_samples],
        atol=1.0e-6,
    )


def test_rank_deficient_three_channel_ambient_covariance_is_reproduced() -> None:
    """半正定値rank-deficient共分散でも3CH雑音の観測共分散が解析値へ一致する。"""

    fs = 4096.0
    n_sample = 65536
    receiver = Receiver(StaticPose([0.0, 0.0, 0.0]), LinearArray(3, 0.1))
    covariance_vector = np.array([1.0, 0.5, -0.25])
    covariance = np.outer(covariance_vector, covariance_vector)
    field = AmbientField(
        BandLimitedNoiseSpectrum(200.0, 1600.0),
        amplitude=1.5,
        covariance=covariance,
        noise_seed=900,
    )
    signal = SceneRenderer(dtype=np.float32).render(
        Scene([], [field], FreeField(1500.0)), receiver, np.arange(n_sample, dtype=float) / fs
    )
    observed_covariance = np.cov(signal)
    np.testing.assert_allclose(observed_covariance, (1.5**2) * covariance, atol=0.04)


def test_rendered_pink_noise_psd_follows_inverse_frequency_slope() -> None:
    """pink noiseの実生成PSDが周波数比に対する1/f傾斜を持つ。"""

    fs = 4096.0
    n_sample = 65536
    receiver = _single_channel_receiver()
    component = SourceComponent(
        PinkNoiseSpectrum(100.0, 1500.0, reference_frequency_hz=100.0),
        ConstantEnvelope(),
        amplitude=1.0,
        noise_seed=700,
        noise_filter_length=513,
    )
    signal = np.real(
        SceneRenderer().render(
            Scene([_source(receiver, component)], [], FreeField(1500.0)),
            receiver,
            np.arange(n_sample, dtype=float) / fs,
        )[0]
    )
    metrics = calculate_one_sided_rms_spectrum(signal, fs)
    psd = metrics.rms_power_per_bin / metrics.frequency_resolution_hz
    low_mask = (metrics.frequencies_hz >= 180.0) & (metrics.frequencies_hz <= 220.0)
    high_mask = (metrics.frequencies_hz >= 780.0) & (metrics.frequencies_hz <= 820.0)
    observed_ratio_db = 10.0 * np.log10(float(np.mean(psd[low_mask]) / np.mean(psd[high_mask])))
    expected_ratio_db = 10.0 * np.log10(800.0 / 200.0)
    # 有限sample periodogramと有限長FIRを平均帯域で評価し、理論1/f傾斜へ1 dB以内で一致させる。
    np.testing.assert_allclose(observed_ratio_db, expected_ratio_db, atol=1.0)
