"""Parts 3 and 4: which phase arrays give a real sig2, and what zero / random phase do."""
import numpy as np
import pytest

import phaselab as pl


def test_odd_random_phase_has_required_symmetry(rng):
    """Generated PH1 satisfies PH1[N-k] = -PH1[k] and has DC and Nyquist phases in {0, π}."""
    N = pl.N_FFT
    ph = pl.odd_random_phase(N, 200, rng)
    assert np.allclose(ph[N // 2 + 1:], -ph[1:N // 2][::-1])
    assert set(np.unique(ph[0])) <= {0.0, np.pi}
    assert set(np.unique(ph[N // 2])) <= {0.0, np.pi}


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_odd_phase_gives_real_sig2(speech, seed):
    """MG with any odd random PH1 inverts to a real signal (imaginary part at rounding level)."""
    X = pl.stft(speech)
    ph = pl.odd_random_phase(pl.N_FFT, X.shape[1], np.random.default_rng(seed))
    y = pl.istft(np.abs(X) * np.exp(1j * ph), len(speech))
    assert np.max(np.abs(y.imag)) < 1e-12


def test_unconstrained_phase_gives_complex_sig2(speech, rng):
    """MG with an unconstrained random PH1 of the same size as PH gives a clearly complex signal."""
    X = pl.stft(speech)
    y = pl.istft(np.abs(X) * np.exp(1j * pl.unconstrained_random_phase(pl.N_FFT, X.shape[1], rng)), len(speech))
    assert 0.4 < pl.imag_energy_ratio(y) < 0.6


def test_dc_or_nyquist_phase_outside_0_pi_breaks_realness(speech, rng):
    """An odd PH1 whose DC/Nyquist phases are not 0 or π leaves a non-zero imaginary part."""
    X = pl.stft(speech)
    ph = pl.odd_random_phase(pl.N_FFT, X.shape[1], rng)
    ph[0] = np.pi / 3
    y = pl.istft(np.abs(X) * np.exp(1j * ph), len(speech))
    assert np.max(np.abs(y.imag)) > 1e-6


def test_constant_offset_breaks_oddness(speech, rng):
    """Adding a constant π/4 to an odd PH1 destroys the symmetry and makes sig2 complex."""
    X = pl.stft(speech)
    ph = pl.odd_random_phase(pl.N_FFT, X.shape[1], rng) + np.pi / 4
    assert pl.imag_energy_ratio(pl.istft(np.abs(X) * np.exp(1j * ph), len(speech))) > 0.3


def test_zero_phase_frame_is_real_and_even(speech):
    """IFFT of a frame's magnitude (zero phase) is real and even: r[n] = r[N-n]."""
    X = pl.stft(speech)
    r = np.fft.ifft(np.abs(X[:, 200]))
    assert np.max(np.abs(r.imag)) < 1e-12
    assert np.allclose(r.real[1:], r.real[1:][::-1])


def test_zero_phase_creates_buzz_at_hop_rate(rng):
    """sig1 of white noise becomes periodic with the hop: clear autocorrelation peak at lag H, none in the input."""
    x = rng.standard_normal(4 * pl.FS)
    s1 = pl.istft(np.abs(pl.stft(x)).astype(complex), len(x)).real

    def ac(y, lag):
        y = y - y.mean()
        return np.dot(y[:-lag], y[lag:]) / np.dot(y, y)
    assert ac(s1, pl.HOP) > 0.2 and abs(ac(x, pl.HOP)) < 0.05
    assert ac(s1, pl.HOP) > 5 * abs(ac(s1, pl.HOP // 2))


def test_random_phase_keeps_one_over_R_of_energy(rng):
    """Odd random phase on a redundant STFT keeps about H/N = 1/4 of the energy (−6 dB ± 1 dB)."""
    x = rng.standard_normal(5 * pl.FS)
    X = pl.stft(x)
    y = pl.istft(np.abs(X) * np.exp(1j * pl.odd_random_phase(pl.N_FFT, X.shape[1], rng)), len(x)).real
    ratio_db = 10 * np.log10(np.sum(y ** 2) / np.sum(x ** 2))
    assert abs(ratio_db - 10 * np.log10(pl.HOP / pl.N_FFT)) < 1.0
