"""Builds Speech_Phase_STOI.ipynb. Execute it with:

    jupyter nbconvert --to notebook --execute --inplace Speech_Phase_STOI.ipynb --ExecutePreprocessor.timeout=-1
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md(r"""
# Magnitude, Phase and Intelligibility of Speech

This notebook carries out the complete assignment and shows every figure, table and audio signal. The signal
processing is implemented in `phaselab.py` and each question has its own function in `experiments.py`. The full
discussion of the results is given in the report, `report/Speech_Phase_STOI_Report.pdf`.

| Question | Content | Function |
|---|---|---|
| 1 | Voice recording of about 10 s | `prepare_data` |
| 2 | STFT; magnitude MG and phase PH | `part2_stft` |
| 3 | sig1 (zero phase); PH1 and sig2 | `part3_magnitude_phase` |
| 4 | Phase pattern for a real sig2; number of patterns | `part4_realness` |
| 5 | Unintelligible version with odd random phase; recovery | `part5_window_sweep`, `part5_scramble_recover` |
| 6 | NOISEX-92 noise, STOI, Questions 2 to 5 repeated, band-pass filter | `make_noisy`, `part6_*` |

Analysis settings: sampling rate 16 kHz, Hann window of 512 samples (32 ms), hop of 128 samples (8 ms, 75 %
overlap), two-sided DFT, least-squares inverse STFT.
""")
code(r"""
import io
import json

import pandas as pd
from IPython.display import Audio, Image, display
from PIL import Image as PILImage

import experiments as E

pd.set_option("display.precision", 3)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def show(path, width=900):
    # figures are embedded as compressed JPEG so that the executed notebook stays small
    im = PILImage.open(path).convert("RGB")
    if im.width > 1300:
        im = im.resize((1300, round(im.height * 1300 / im.width)), PILImage.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=72)
    display(Image(data=buf.getvalue(), format="jpeg", width=width))


EMBED_AUDIO = False   # True shows audio players, but makes the saved notebook much larger


def play(folder, name):
    path = E.AUD / folder / f"{name}.wav"
    print(name, "->", path.relative_to(E.ROOT).as_posix())
    if EMBED_AUDIO:
        display(Audio(str(path)))
""")

md("## Question 1: Voice recording")
code(r"""
info = E.prepare_data()
x = E.load_clean()
print(json.dumps(info, indent=1))
play("clean", "original")
""")
md(r"""
The recording is LibriSpeech dev-clean utterance 1272-128104-0003 (male speaker, 9.9 s, 16 kHz, lossless). The
loudest speech frames are about 47 dB above the pauses, so the background noise is negligible.
""")

md("## Question 2: STFT magnitude and phase")
code(r"""
p2 = E.part2_stft(x, "clean")
print("X shape (bins x frames):", p2["shape"])
show(p2["figs"]["MG"])
show(p2["figs"]["PH"])
""")
md(r"""
The magnitude shows harmonics, formants, fricatives and pauses. The wrapped phase looks like noise because the phase
of each bin advances by 2πkH/N per hop and wraps at its own rate; its time and frequency derivatives, shown next, are
clearly structured. Every frame has an even magnitude and an odd phase, which is the basis of Questions 3 and 4.
""")
code(r"""
show(p2["figs"]["phase_structure"])
show(p2["figs"]["symmetry"])
""")

md("## Question 3: sig1 (zero phase) and sig2 (phase array PH1)")
code(r"""
p3 = E.part3_magnitude_phase(x, p2["X"], "clean")
display(p3["metrics"])
display(pd.Series(p3["realness"], name="realness checks").to_frame())
display(p3["wstats"])
""")
code(r"""
for key in ["frame_zero_phase", "waveforms", "specs", "ltas_env", "complex_sig2"]:
    show(p3["figs"][key])
for name in ["sig1_zero_phase", "sig1c_zero_phase_centred", "sig2_odd_random_phase",
             "sig2_unconstrained_real_part", "control_phase_only"]:
    play("clean", name)
""")
md(r"""
* sig1: zero phase turns every frame into a symmetric pulse at the frame origin. The pulses repeat every hop, so sig1
  is a buzzy monotone at fs/H = 125 Hz with the correct formants, and its waveform is uncorrelated with the original.
* With an unconstrained PH1, sig2 is complex (about half of its energy is imaginary). It is real only if PH1 is
  odd-symmetric (Question 4). The real sig2 is rough and noise-like and keeps about H/N = 1/4 of the energy.
* sig1 and sig2 have the same magnitude; sig1 is coherent and periodic, sig2 incoherent and aperiodic.
""")

md("## Question 4: Phase pattern that guarantees a real sig2")
code(r"""
p4 = E.part4_realness(p2["MG"])
display(p4["table"])
show(p4["figs"]["realness"])
print(json.dumps(p4["count"], indent=1, default=float))
""")
md(r"""
A real sig2 requires PH1(m, N−k) = −PH1(m, k) for k = 1, ..., N/2−1 and PH1(m, 0), PH1(m, N/2) equal to 0 or π in
every frame. N/2−1 = 255 phases per frame remain free, so there are infinitely many valid patterns.
""")

md("## Question 5: Unintelligible version with odd random phase, and recovery")
code(r"""
sw = E.part5_window_sweep(x)
display(sw["table"])
show(sw["figs"]["sweep"])
""")
code(r"""
p5 = E.part5_scramble_recover(x, x, "clean", detailed=True)
display(pd.DataFrame({k: v for k, v in p5["metrics"].items() if not k.startswith("_")}).T)
print({k: v for k, v in p5["metrics"].items() if k.startswith("_")})
show(p5["figs"]["recovery_specs"])
show(p5["figs"]["gla_convergence"], 700)
for name in ["scrambled_odd_random_phase_1s", "recovered_R1_blind_griffin_lim", "recovered_R2_griffin_lim_true_MG",
             "scrambled_keyed_otp_1s", "recovered_R3_keyed"]:
    play("clean", name)
""")
md(r"""
With 32 ms frames the random-phase version remains intelligible, so the scrambler uses 1.024 s frames (ESTOI close to
zero). Blind recovery is impossible because the scrambled signal is itself a consistent solution for its own
magnitude; recovery succeeds with the true magnitude as side information, and exactly with a key.
""")

md("## Question 6: NOISEX-92 noise, STOI, repetition of Questions 2 to 5, band-pass filtering")
code(r"""
noisy = E.make_noisy(x)
f6 = E.part6_noise_figs(x, noisy)
show(f6["noises"])
for n in E.NOISES:
    show(f6[f"MG_PH_{n}"], 950)
""")
code(r"""
# Questions 2 to 5 for all 15 noisy signals and the band-pass filters take about 20 minutes (run_all.py does the
# same). With RECOMPUTE_PART6 = False the saved results of that run are loaded instead.
RECOMPUTE_PART6 = False
if RECOMPUTE_PART6:
    p6 = E.part6_run_all(x, noisy)
else:
    p6 = {"STOI_wide": pd.read_csv(E.TAB / "p6_STOI_wide.csv", index_col=[0, 1]),
          "ESTOI_wide": pd.read_csv(E.TAB / "p6_ESTOI_wide.csv", index_col=[0, 1]),
          "realness": pd.read_csv(E.TAB / "p6_realness_noisy.csv"),
          "bpf": pd.read_csv(E.TAB / "p6_bpf.csv"),
          "figs": {"stoi_noisy": str(E.FIG / "p6_stoi_noisy.png"),
                   "phase_ops_noise": str(E.FIG / "p6_phase_ops_under_noise.png"),
                   "bpf_stoi": str(E.FIG / "p6_bpf_stoi.png"),
                   **{f"variants_{n}": str(E.FIG / f"p6_variants_{n}_0dB.png") for n in E.NOISES}}}
display(p6["STOI_wide"])
display(p6["ESTOI_wide"])
display(p6["realness"])
for key in ["stoi_noisy", "phase_ops_noise"] + [f"variants_{n}" for n in E.NOISES]:
    show(p6["figs"][key])
""")
code(r"""
fd = E.part6_filter_design(x, noisy)
show(fd["figs"]["bpf_design"])
display(fd["diagnostics"])
b = p6["bpf"]
for value in ["STOI", "dSTOI", "SNR gain (dB)"]:
    display(b.pivot_table(index=["noise", "SNR_dB"], columns="filter", values=value, sort=False))
show(p6["figs"]["bpf_stoi"])
show(fd["figs"]["band_snr"])
bc = E.part6_band_contrib(x, noisy)
show(bc["figs"]["band_stoi"])
for name in ["noisy", "bpf_telephone_300_3400", "bpf_stoi_band_120_4500", "oracle_wiener"]:
    play("white_p0dB", name)
""")
md(r"""
A fixed band-pass filter multiplies speech and noise by the same gain at every frequency, so the SNR inside each
pass-band third-octave band, which is what STOI measures after its per-band normalisation, does not change. The filter
raises the broadband SNR but cannot raise STOI; it can only lower it by removing STOI bands. Only a time-varying gain
(the oracle Wiener reference) improves STOI.
""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "Speech_Phase_STOI.ipynb")
print("wrote Speech_Phase_STOI.ipynb with", len(cells), "cells")
