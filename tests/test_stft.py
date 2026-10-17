"""STFT / inverse STFT behaviour that every later part of the assignment relies on."""
import numpy as np
import pytest

import phaselab as pl


@pytest.mark.parametrize("n_fft,hop", [(512, 128), (256, 64), (1024, 256), (16384, 4096)])
def test_perfect_reconstruction(rng, n_fft, hop):
    """ISTFT(STFT(x)) returns x to machine precision for every window/hop used in the study."""
    x = rng.standard_normal(3 * n_fft + 77)
    y = pl.istft(pl.stft(x, n_fft, hop), len(x), hop)
    assert np.max(np.abs(y - x)) < 1e-10


def test_stft_shape_is_two_sided(rng):
    """The STFT keeps all N bins (two-sided spectrum), not only 0..N/2."""
    X = pl.stft(rng.standard_normal(16000))
    assert X.shape[0] == pl.N_FFT
    assert X.shape[1] == int(np.ceil((16000 + pl.N_FFT // 2) / pl.HOP)) + 1


def test_real_signal_has_hermitian_stft(speech):
    """Every frame of the STFT of a real signal is conjugate-symmetric: X[N-k] = conj(X[k])."""
    assert pl.is_hermitian(pl.stft(speech))


def test_hann_squared_overlap_is_constant():
    """With hop N/4 the squared Hann windows overlap-add to a constant (1.5), so the inverse is well conditioned."""
    w2 = pl.hann(pl.N_FFT) ** 2
    acc = sum(np.roll(w2, k * pl.HOP) for k in range(pl.N_FFT // pl.HOP))
    assert np.allclose(acc, 1.5)


def test_istft_returns_complex_for_non_hermitian_input(rng):
    """A spectrogram that is not conjugate-symmetric gives a complex-valued inverse."""
    X = rng.standard_normal((pl.N_FFT, 50)) + 1j * rng.standard_normal((pl.N_FFT, 50))
    y = pl.istft(X, 40 * pl.HOP)
    assert pl.imag_energy_ratio(y) > 0.3


def test_stft_of_istft_is_a_projection(rng):
    """STFT(ISTFT(Y)) is idempotent: applying it twice equals applying it once (consistency projection)."""
    L = 30 * pl.HOP
    M = np.abs(pl.stft(rng.standard_normal(L)))
    Y = M * np.exp(1j * pl.odd_random_phase(pl.N_FFT, M.shape[1], rng))
    P1 = pl.stft(pl.istft(Y, L).real)
    P2 = pl.stft(pl.istft(P1, L).real)
    assert np.max(np.abs(P1 - P2)) < 1e-9 * np.max(np.abs(P1))
