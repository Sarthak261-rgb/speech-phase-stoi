"""Part 6: noise mixing, metrics and band-pass filters."""
import numpy as np
import pytest
from scipy import signal

import phaselab as pl


@pytest.mark.parametrize("snr", [-2, 0, 2, 5, 10])
def test_mix_at_snr_hits_target(speech, rng, snr):
    """The noise is scaled so that the global SNR equals the target exactly."""
    noisy, n = pl.mix_at_snr(speech, rng.standard_normal(len(speech) + 500), snr)
    measured = 10 * np.log10(np.sum(speech ** 2) / np.sum(n ** 2))
    assert abs(measured - snr) < 1e-9 and np.allclose(noisy, speech + n)


def test_noise_segment_has_requested_length(rng):
    """A random noise segment of exactly the speech length is returned."""
    seg = pl.pick_noise_segment(rng.standard_normal(100000), 12345, rng)
    assert len(seg) == 12345


def test_stoi_of_identical_signals_is_one(speech):
    """STOI(x, x) = 1."""
    assert pl.stoi(speech, speech) == pytest.approx(1.0, abs=1e-6)


def test_stoi_is_gain_invariant(speech):
    """STOI ignores an overall gain, so level differences do not bias comparisons."""
    assert pl.stoi(speech, 0.05 * speech) == pytest.approx(1.0, abs=1e-6)


def test_stoi_decreases_with_noise(speech, rng):
    """STOI falls monotonically as the SNR drops."""
    v = rng.standard_normal(len(speech))
    s = [pl.stoi(speech, pl.mix_at_snr(speech, v, snr)[0]) for snr in (10, 0, -10)]
    assert s[0] > s[1] > s[2]


def test_per_band_stoi_matches_pystoi(speech, rng):
    """The 15 per-band STOI values average exactly to the pystoi score."""
    y = pl.mix_at_snr(speech, rng.standard_normal(len(speech)), 0)[0]
    _, d = pl.stoi_per_band(speech, y)
    assert len(d) == 15 and d.mean() == pytest.approx(pl.stoi(speech, y), abs=1e-9)


def test_spectral_convergence_zero_for_identical():
    """Spectral convergence is 0 for identical magnitudes."""
    M = np.abs(np.random.default_rng(0).standard_normal((10, 10)))
    assert pl.spectral_convergence(M, M) == 0.0


def _tone(f, n=pl.FS):
    return np.sin(2 * np.pi * f * np.arange(n) / pl.FS)


def test_bpf_passes_midband_tone():
    """A 1 kHz tone passes the 300-3400 Hz zero-phase Butterworth with less than 0.1 dB change."""
    t = _tone(1000)
    y = pl.apply_bandpass(t, 300, 3400)
    sl = slice(2000, -2000)
    assert abs(20 * np.log10(np.std(y[sl]) / np.std(t[sl]))) < 0.1


@pytest.mark.parametrize("f", [60, 6000])
def test_bpf_rejects_out_of_band_tones(f):
    """Tones well outside the pass band are attenuated by more than 40 dB."""
    t = _tone(f)
    y = pl.apply_bandpass(t, 300, 3400)
    sl = slice(4000, -4000)
    assert 20 * np.log10(np.std(y[sl]) / np.std(t[sl])) < -40


def test_zero_phase_filter_has_no_delay_but_causal_does(speech):
    """Forward-backward filtering introduces no delay; the causal filter shifts the signal."""
    def lag(a, b):
        c = signal.correlate(b, a, mode="full", method="fft")
        return int(np.argmax(c)) - (len(a) - 1)
    assert lag(speech, pl.apply_bandpass(speech, 300, 3400)) == 0
    assert lag(speech, pl.apply_bandpass(speech, 300, 3400, kind="butter_causal")) > 0


def test_fir_delay_is_compensated(speech):
    """The linear-phase FIR output is time-aligned with its input."""
    y = pl.apply_bandpass(speech, 300, 3400, kind="fir_linear_phase")
    c = signal.correlate(y, speech, mode="full", method="fft")
    assert int(np.argmax(c)) - (len(speech) - 1) == 0


def test_lti_filter_keeps_passband_band_snr(speech, rng):
    """An LTI band-pass filter leaves the SNR of every pass-band third-octave band unchanged (< 0.1 dB)."""
    v = pl.mix_at_snr(speech, rng.standard_normal(len(speech)), 0)[1]
    cf, b0 = pl.third_octave_band_snr(speech, v)
    _, b1 = pl.third_octave_band_snr(pl.apply_bandpass(speech, 300, 3400), pl.apply_bandpass(v, 300, 3400))
    band = (cf > 400) & (cf < 3000)
    assert np.max(np.abs(b1[band] - b0[band])) < 0.1


def test_brickwall_removes_out_of_band_energy(rng):
    """The ideal brick-wall filter leaves no energy outside its pass band."""
    y = pl.apply_bandpass(rng.standard_normal(pl.FS), 300, 3400, kind="brickwall")
    Y = np.abs(np.fft.rfft(y))
    f = np.fft.rfftfreq(pl.FS, 1 / pl.FS)
    assert np.max(Y[(f < 300) | (f > 3400)]) < 1e-9 * np.max(Y)
