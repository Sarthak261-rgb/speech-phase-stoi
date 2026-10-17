"""Part 5: scrambling and the three recovery routes."""
import numpy as np

import phaselab as pl


def test_keyed_scrambler_is_real_and_exactly_invertible(speech):
    """The phase one-time pad gives a real signal and the right key restores it exactly."""
    s = pl.otp_scramble(speech, 16384, key=7)
    assert np.isrealobj(s)
    r = pl.otp_descramble(s, 16384, key=7, length=len(speech))
    assert pl.snr_db(speech, r, align_scale=False) > 100


def test_keyed_scrambler_wrong_key_fails(speech):
    """Descrambling with a wrong key does not give the speech back."""
    s = pl.otp_scramble(speech, 16384, key=7)
    r = pl.otp_descramble(s, 16384, key=8, length=len(speech))
    assert pl.stoi(speech, r) < 0.6


def test_keyed_scrambler_preserves_frame_magnitudes(speech):
    """The keyed scrambler changes only the phase: every frame keeps its magnitude spectrum."""
    n = 16384
    s = pl.otp_scramble(speech, n, key=3)
    xp = np.zeros(len(s))
    xp[:len(speech)] = speech
    A = np.abs(np.fft.fft(xp.reshape(-1, n), axis=1))
    B = np.abs(np.fft.fft(s.reshape(-1, n), axis=1))
    assert np.allclose(A, B, atol=1e-9 * A.max())


def test_scrambling_makes_speech_unintelligible(speech):
    """Odd random phase on 1.024 s frames drives ESTOI close to zero."""
    X = pl.stft(speech, 16384, 4096)
    ph = pl.odd_random_phase(16384, X.shape[1], np.random.default_rng(0))
    s = pl.istft(np.abs(X) * np.exp(1j * ph), len(speech), 4096).real
    assert pl.stoi(speech, s, extended=True) < 0.2


def test_scrambled_signal_is_a_griffin_lim_fixed_point(speech):
    """Griffin-Lim started from the scrambled signal's own phase does not move: blind recovery cannot prefer x."""
    X = pl.stft(speech, 16384, 4096)
    s = pl.istft(np.abs(X) * np.exp(1j * pl.odd_random_phase(16384, X.shape[1], np.random.default_rng(1))),
                 len(speech), 4096).real
    S = pl.stft(s, 16384, 4096)
    y = pl.griffin_lim(np.abs(S), len(s), 4096, n_iter=5, alpha=0.0, init_phase=np.angle(S))
    assert pl.snr_db(s, y) > 60


def test_griffin_lim_recovers_from_true_magnitude(speech):
    """Given the true 32 ms magnitude, Fast Griffin-Lim restores an intelligible signal (STOI > 0.95)."""
    y = pl.griffin_lim(np.abs(pl.stft(speech)), len(speech), n_iter=100)
    assert pl.stoi(speech, y) > 0.95
