# Magnitude, phase and intelligibility of speech

Speech signal processing assignment by Sarthak Tripathi (USN 1MV23EC096), Department of Electronics and
Communication Engineering, Sir M. Visvesvaraya Institute of Technology.

The report is in [`report/Speech_Phase_STOI_Report.pdf`](report/Speech_Phase_STOI_Report.pdf) (also as a Word
file), and every step can be followed in the notebook [`Speech_Phase_STOI.ipynb`](Speech_Phase_STOI.ipynb).

## What was done

1. A 9.9 s clean recording (LibriSpeech utterance 1272-128104-0003) was analysed with a 32 ms Hann window and an
   8 ms hop.
2. The STFT magnitude MG and phase PH were plotted.
3. The signal was rebuilt from MG with zero phase (sig1) and with a random phase PH1 (sig2).
4. The condition on PH1 for a real sig2 was derived and checked: the phase must be odd-symmetric, with the DC and
   Nyquist phases equal to 0 or π.
5. The speech was made unintelligible with an odd random phase (1.024 s frames) and recovered in three ways.
6. White, babble and machine-gun noise from NOISEX-92 were added at −2, 0, 2, 5 and 10 dB SNR, steps 2 to 5 were
   repeated, STOI was computed, and three band-pass filters were compared.

## Main results

| | STOI |
|---|---|
| sig1, zero phase | 0.839 (buzzy, robotic, 125 Hz frame-rate pitch) |
| sig2, odd random phase | 0.871 (rough, noise-like) |
| Scrambled, 1.024 s frames | 0.396 (ESTOI 0.072) |
| Recovered with the true magnitude / with the key | 0.999 / 1.000 |
| Blind recovery | 0.395 (not possible) |
| Band-pass filter 300–3400 Hz | 0.050 lower than without the filter, on average |

## Running it

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv/Scripts/python.exe run_all.py          # whole study, about half an hour
.venv/Scripts/python.exe -m pytest           # 50 tests
.venv/Scripts/ruff.exe check .               # lint
.venv/Scripts/python.exe make_report.py      # PDF and Word report
```

On Linux or macOS use `.venv/bin/` instead of `.venv/Scripts/`.

To try the recovery exercise on the scrambled files in `outputs/challenge/`:

```bash
.venv/Scripts/python.exe recover.py outputs/challenge/clean__scrambled.wav --mode blind --ref data/speech/original.wav
.venv/Scripts/python.exe recover.py outputs/challenge/clean__keyed.wav --mode key --key 2026 --ref data/speech/original.wav
```

## Files

| Path | Contents |
|---|---|
| `phaselab.py` | STFT, least-squares inverse STFT, phase generators, fast Griffin-Lim, keyed scrambler, noise mixing, STOI, filters |
| `experiments.py` | one function per part of the assignment |
| `run_all.py` | runs everything and writes `outputs/` |
| `recover.py` | recovery of scrambled files (blind, with magnitude, with key) |
| `make_report.py`, `report_text.py` | build the report from the results |
| `build_notebook.py` | builds the notebook |
| `tests/` | pytest tests |
| `data/raw_speech`, `data/raw_noise` | the LibriSpeech recording and the three NOISEX-92 noises as downloaded |
| `outputs/tables`, `outputs/results.json` | all numbers used in the report |
| `outputs/challenge` | scrambled versions of all 16 signals for the recovery exercise |

Figures and audio files are not stored in the repository; `run_all.py` recreates them in `outputs/figures` and
`outputs/audio`.

## Data

- Speech: LibriSpeech (Panayotov et al., 2015), CC BY 4.0.
- Noise: NOISEX-92 (Varga and Steeneken, 1993), from https://github.com/speechdnn/Noises.
