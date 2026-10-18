"""File handling, helper metrics, input checks and the recover.py command-line tool."""
import numpy as np
import pytest
import soundfile as sf

import phaselab as pl
import recover


@pytest.fixture
def speech_wav(tmp_path, speech):
    path = tmp_path / "speech.wav"
    pl.save_wav(str(path), speech, peak=None)
    return path


def test_wav_round_trip_keeps_samples(tmp_path, speech):
    """Saving and loading a 16-bit wav changes the samples by no more than one quantisation step."""
    pl.save_wav(str(tmp_path / "a.wav"), speech, peak=None)
    y = pl.load_wav(str(tmp_path / "a.wav"))
    assert len(y) == len(speech) and np.max(np.abs(y - speech)) <= 1 / 32768 + 1e-12


def test_save_wav_normalises_peak(tmp_path, speech):
    """With the default setting the saved file peaks at 0.89 of full scale."""
    pl.save_wav(str(tmp_path / "b.wav"), 3 * speech)
    y, _ = sf.read(str(tmp_path / "b.wav"))
    assert np.max(np.abs(y)) == pytest.approx(0.89, abs=1e-3)


def test_load_wav_resamples_to_16k(tmp_path):
    """An 8 kHz file is resampled to 16 kHz without moving the frequency of a tone."""
    t = np.arange(8000) / 8000
    sf.write(str(tmp_path / "tone8k.wav"), 0.5 * np.sin(2 * np.pi * 500 * t), 8000)
    y = pl.load_wav(str(tmp_path / "tone8k.wav"))
    peak_hz = np.argmax(np.abs(np.fft.rfft(y))) * pl.FS / len(y)
    assert len(y) == 16000 and abs(peak_hz - 500) < 2


def test_describe_of_identical_signals(speech):
    """The comparison numbers are ideal when a signal is compared with itself."""
    d = pl.describe(speech, speech)
    assert d["corr"] == pytest.approx(1) and d["SC"] == pytest.approx(0, abs=1e-12)
    assert d["LSD_dB"] == pytest.approx(0, abs=1e-9) and d["STOI"] == pytest.approx(1, abs=1e-6)


def test_snr_with_scaling_ignores_sign(speech):
    """The scaled waveform SNR treats -x as identical to x, as phase retrieval can only recover x up to sign."""
    assert pl.snr_db(speech, -speech) > 100 and pl.snr_db(speech, -speech, align_scale=False) < 0


def test_mix_at_snr_rejects_short_or_silent_noise(speech):
    """Mixing refuses a noise clip that is shorter than the speech or completely silent."""
    with pytest.raises(ValueError):
        pl.mix_at_snr(speech, np.ones(10), 0)
    with pytest.raises(ValueError):
        pl.mix_at_snr(speech, np.zeros(len(speech)), 0)


def test_noise_segment_edge_cases(rng):
    """A clip of exactly the requested length is returned as it is; a shorter clip is refused."""
    v = rng.standard_normal(1000)
    assert np.array_equal(pl.pick_noise_segment(v, 1000, rng), v)
    with pytest.raises(ValueError):
        pl.pick_noise_segment(v, 1001, rng)


def test_noise_segment_respects_activity(rng):
    """For bursty noise the chosen segment contains the required share of active frames."""
    v = np.zeros(200000)
    v[100000:140000] = rng.standard_normal(40000)
    seg = pl.pick_noise_segment(v, 20000, rng, min_activity=0.5)
    assert np.mean(seg != 0) >= 0.5


def test_unknown_filter_kind_is_rejected(speech):
    """Asking for a filter type that does not exist raises an error instead of returning the input."""
    with pytest.raises(ValueError):
        pl.apply_bandpass(speech, 300, 3400, kind="chebyshev")


def test_descramble_rejects_wrong_length(speech):
    """Descrambling a signal that is not a whole number of frames raises a clear error."""
    with pytest.raises(ValueError):
        pl.otp_descramble(speech[:10000], 16384, key=1, length=10000)


def test_recover_key_mode_end_to_end(tmp_path, speech):
    """recover.py --mode key restores a keyed file written to disk."""
    pl.save_wav(str(tmp_path / "keyed.wav"), pl.otp_scramble(speech, 16384, key=7))
    y = recover.main([str(tmp_path / "keyed.wav"), "--mode", "key", "--key", "7", "--length", str(len(speech)),
                      "-o", str(tmp_path / "out.wav")])
    assert pl.stoi(speech, y) > 0.99 and (tmp_path / "out.wav").exists()


def test_recover_magnitude_mode_end_to_end(tmp_path, speech, speech_wav):
    """recover.py --mode magnitude rebuilds intelligible speech from the true magnitude."""
    y = recover.main([str(speech_wav), "--mode", "magnitude", "--mag-from", str(speech_wav), "--iters", "60",
                      "-o", str(tmp_path / "out.wav")])
    assert pl.stoi(speech, y) > 0.9


def test_recover_blind_mode_does_not_recover(tmp_path, speech):
    """recover.py --mode blind runs but cannot bring back scrambled speech."""
    X = pl.stft(speech, 16384, 4096)
    s = pl.istft(np.abs(X) * np.exp(1j * pl.odd_random_phase(16384, X.shape[1], np.random.default_rng(2))),
                 len(speech), 4096).real
    pl.save_wav(str(tmp_path / "scr.wav"), s)
    y = recover.main([str(tmp_path / "scr.wav"), "--mode", "blind", "--iters", "20", "-o", str(tmp_path / "o.wav")])
    assert len(y) == len(s) and pl.stoi(speech, y) < 0.6


@pytest.mark.parametrize("args", [["--mode", "key"], ["--mode", "magnitude"]])
def test_recover_requires_its_inputs(tmp_path, speech_wav, args):
    """recover.py stops with a usage error when the key or the magnitude file is missing."""
    with pytest.raises(SystemExit):
        recover.main([str(speech_wav), *args, "-o", str(tmp_path / "x.wav")])
