"""Part 1 and the data used in Part 6."""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import phaselab as pl

ROOT = Path(__file__).resolve().parents[1]


def test_recording_is_about_ten_seconds_at_16k():
    """The voice recording is a mono 16 kHz file of roughly 10 s (9 to 11 s)."""
    info = sf.info(str(ROOT / "data" / "speech" / "original.wav"))
    assert info.samplerate == 16000 and info.channels == 1 and 9.0 <= info.duration <= 11.0


def test_recording_has_minimal_background_noise():
    """Loud speech frames are more than 40 dB above the quietest frames (minimal background noise)."""
    x = pl.load_wav(str(ROOT / "data" / "speech" / "original.wav"))
    fr = np.lib.stride_tricks.sliding_window_view(x, 320)[::160]
    e = 10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-20)
    assert np.percentile(e, 95) - np.percentile(e, 5) > 40


@pytest.mark.parametrize("name", ["white", "babble", "machinegun"])
def test_noisex_files_resampled_to_16k(name):
    """Each NoiseX-92 noise is available at 16 kHz and is long enough to cut a 10 s segment."""
    info = sf.info(str(ROOT / "data" / "noise" / f"{name}_16k.wav"))
    assert info.samplerate == 16000 and info.duration > 60
