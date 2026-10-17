"""
One function per part of the assignment. Each writes its figures, tables and audio
under outputs/ and returns the numbers. All random draws are seeded.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd
from scipy import signal

import phaselab as pl

ROOT = Path(__file__).resolve().parent
RAW_SPEECH = ROOT / "data" / "raw_speech"
RAW_NOISE = ROOT / "data" / "raw_noise"
DATA = ROOT / "data"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
AUD = OUT / "audio"
TAB = OUT / "tables"
CHAL = OUT / "challenge"

SPEECH_ID = "1272-128104-0003"            # LibriSpeech dev-clean, speaker 1272, 9.9 s
NOISES = ["white", "babble", "machinegun"]
SNRS = [-2, 0, 2, 5, 10]
SEED = 2026
SCR_N, SCR_H = 16384, 4096                # 1.024 s scrambling frames, 75 % overlap
GLA_ITERS = 200
FS, N, H = pl.FS, pl.N_FFT, pl.HOP

# ------------------------------------------------------------------ plot style
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
NOISE_COLOR = dict(zip(NOISES, SERIES[:3]))
MARKERS = ["o", "s", "^", "D", "v", "P"]

SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f2a9a8", "#e34948", "#9e2423"])
CYCLIC = "twilight"   # phase is a circular quantity: -pi and +pi are the same colour

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#bdbcb6", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 10.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
    "legend.frameon": False, "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
})


def _dirs():
    for d in (FIG, AUD, TAB, CHAL, DATA / "speech", DATA / "noise"):
        d.mkdir(parents=True, exist_ok=True)


def _save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    return str(FIG / name)


def _spec_db(M, ref=None, floor=-80):
    ref = ref if ref is not None else M.max()
    return np.maximum(20 * np.log10(np.maximum(M, 1e-12) / ref), floor)


def _imshow_spec(ax, M, hop=H, fs=FS, ref=None, fmax=8000, cmap=SEQ, vmin=-80, vmax=0, data=None):
    half = M.shape[0] // 2 + 1
    img = data if data is not None else _spec_db(M[:half], ref)
    t_end = M.shape[1] * hop / fs
    im = ax.imshow(img[:half], origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[0, t_end, 0, fs / 2], interpolation="nearest")
    ax.set_ylim(0, fmax)
    ax.set_ylabel("Frequency (Hz)")
    return im


# ======================================================================= DATA
def prepare_data() -> dict:
    """Copy the chosen LibriSpeech utterance to data/speech/original.wav and
    resample the three NoiseX-92 noises to 16 kHz."""
    _dirs()
    x = pl.load_wav(str(RAW_SPEECH / f"{SPEECH_ID}.flac"))
    pl.save_wav(str(DATA / "speech" / "original.wav"), x, peak=None)
    shutil.copy(RAW_SPEECH / f"{SPEECH_ID}.txt", DATA / "speech" / "transcript.txt")
    info = {"speech_id": SPEECH_ID, "duration_s": len(x) / FS, "fs": FS,
            "transcript": (RAW_SPEECH / f"{SPEECH_ID}.txt").read_text().strip()}
    import soundfile as sf
    for n in NOISES:
        meta = sf.info(str(RAW_NOISE / f"{n}.wav"))
        v = pl.load_wav(str(RAW_NOISE / f"{n}.wav"))
        v = v - v.mean()                                   # machinegun is unsigned 8-bit: remove DC
        pl.save_wav(str(DATA / "noise" / f"{n}_16k.wav"), v, peak=0.95)
        info[f"noise_{n}"] = {"orig_fs": meta.samplerate, "orig_subtype": meta.subtype,
                              "duration_s": round(meta.duration, 2)}
    # frame-energy based estimate of the recording's own noise floor
    fr = np.lib.stride_tricks.sliding_window_view(x, 320)[::160]
    e = 10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-20)
    info["speech_dynamic_range_dB"] = float(np.percentile(e, 95) - np.percentile(e, 5))
    return info


def load_clean() -> np.ndarray:
    return pl.load_wav(str(DATA / "speech" / "original.wav"))


def make_noisy(x: np.ndarray) -> dict:
    """{(noise, snr): (noisy, scaled_noise)} with one fixed random segment per noise type."""
    out = {}
    for i, n in enumerate(NOISES):
        v = pl.load_wav(str(DATA / "noise" / f"{n}_16k.wav"))
        seg = pl.pick_noise_segment(v, len(x), np.random.default_rng(SEED + i),
                                    min_activity=0.4 if n == "machinegun" else 0.0)
        for snr in SNRS:
            out[(n, snr)] = pl.mix_at_snr(x, seg, snr)
    return out


# ===================================================================== PART 2
def part2_stft(x: np.ndarray, tag: str = "clean", detailed: bool = True) -> dict:
    """STFT; separate MG and PH plots (+ phase-structure and symmetry figures)."""
    X = pl.stft(x)
    MG, PH = pl.mag_phase(X)
    t = np.arange(len(x)) / FS
    figs = {}

    fig, ax = plt.subplots(2, 1, figsize=(10, 5.2), gridspec_kw={"height_ratios": [1, 2.4]}, sharex=True)
    ax[0].plot(t, x, color=SERIES[0], lw=0.6)
    ax[0].set_ylabel("Amplitude")
    ax[0].set_title(f"Waveform and STFT magnitude MG = |X(m,k)|  ({tag})", loc="left")
    im = _imshow_spec(ax[1], MG)
    ax[1].set_xlabel("Time (s)")
    cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.01)
    cb.set_label("dB re. max")
    figs["MG"] = _save(fig, f"p2_MG_{tag}.png")

    fig, ax = plt.subplots(figsize=(10, 3.9))
    im = _imshow_spec(ax, MG, data=PH, cmap=CYCLIC, vmin=-np.pi, vmax=np.pi)
    ax.set_xlabel("Time (s)")
    ax.set_title(f"STFT phase PH = angle X(m,k), wrapped to (−π, π]  ({tag})", loc="left")
    cb = fig.colorbar(im, ax=ax, pad=0.01, ticks=[-np.pi, 0, np.pi])
    cb.ax.set_yticklabels(["−π", "0", "π"])
    cb.set_label("rad")
    figs["PH"] = _save(fig, f"p2_PH_{tag}.png")

    if detailed:
        figs.update(_phase_structure_figs(x, X, MG, PH))
    return {"X": X, "MG": MG, "PH": PH, "figs": figs, "shape": X.shape}


def _phase_structure_figs(x, X, MG, PH) -> dict:
    """The raw phase looks like noise; its derivatives do not."""
    figs = {}
    t0, t1, fmax = 1.0, 2.0, 4000
    m0, m1 = int(t0 * FS / H), int(t1 * FS / H)
    kmax = int(fmax / FS * N)
    k = np.arange(N)[:, None]
    # instantaneous-frequency deviation from the bin centre (Hz)
    dphi = np.angle(np.exp(1j * (PH[:, 1:] - PH[:, :-1] - 2 * np.pi * k * H / N)))
    ifdev = np.concatenate([np.zeros((N, 1)), dphi], axis=1) * FS / (2 * np.pi * H)
    # group delay w.r.t. the window centre (ms)
    phc = PH + np.pi * k
    dk = np.angle(np.exp(1j * (phc[1:] - phc[:-1])))
    gd = np.concatenate([-dk * N / (2 * np.pi), np.zeros((1, PH.shape[1]))], axis=0) / FS * 1000
    sl = (slice(0, kmax), slice(m0, m1))
    ext = [m0 * H / FS, m1 * H / FS, 0, kmax * FS / N]
    mask = _spec_db(MG)[sl] > -55      # show derivatives only where there is energy

    fig, ax = plt.subplots(2, 2, figsize=(11, 6.4), sharex=True, sharey=True)
    a = ax.ravel()
    im0 = a[0].imshow(_spec_db(MG)[sl], origin="lower", aspect="auto", cmap=SEQ, vmin=-80, vmax=0, extent=ext)
    a[0].set_title("(a) MG (dB): harmonics and formants", loc="left")
    im1 = a[1].imshow(PH[sl], origin="lower", aspect="auto", cmap=CYCLIC, vmin=-np.pi, vmax=np.pi, extent=ext)
    a[1].set_title("(b) PH wrapped: looks like noise", loc="left")
    im2 = a[2].imshow(np.where(mask, ifdev[sl], np.nan), origin="lower", aspect="auto", cmap=DIV,
                      vmin=-60, vmax=60, extent=ext)
    a[2].set_title("(c) dPH/dt: inst. frequency deviation (Hz)", loc="left")
    im3 = a[3].imshow(np.where(mask, gd[sl], np.nan), origin="lower", aspect="auto", cmap=DIV,
                      vmin=-8, vmax=8, extent=ext)
    a[3].set_title("(d) -dPH/dw: group delay (ms)", loc="left")
    for i, im in enumerate([im0, im1, im2, im3]):
        fig.colorbar(im, ax=a[i], pad=0.01, shrink=0.9)
        a[i].set_facecolor("#f0efec")
    for i in (0, 2):
        a[i].set_ylabel("Frequency (Hz)")
    for i in (2, 3):
        a[i].set_xlabel("Time (s)")
    fig.suptitle("Phase is structured once it is differentiated (1.0 to 2.0 s, 0 to 4 kHz)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    figs["phase_structure"] = _save(fig, "p2_phase_structure.png")

    # Hermitian symmetry of one voiced frame across the FULL two-sided spectrum
    m = int(np.argmax(np.sum(MG[:kmax] ** 2, axis=0)))
    kk = np.arange(N)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.3))
    ax[0].plot(kk, 20 * np.log10(MG[:, m] / MG.max() + 1e-12), color=SERIES[0], lw=1.0)
    ax[0].axvline(N / 2, color=INK2, lw=0.8, ls="--")
    ax[0].set_title(f"MG of frame m={m}: even, MG[N-k] = MG[k]", loc="left")
    ax[0].set_xlabel("FFT bin k (0..N-1)")
    ax[0].set_ylabel("dB")
    ax[1].plot(kk, PH[:, m], ".", color=SERIES[1], ms=2.5)
    ax[1].axvline(N / 2, color=INK2, lw=0.8, ls="--")
    ax[1].set_title("PH of the same frame: odd, PH[N-k] = -PH[k]", loc="left")
    ax[1].set_xlabel("FFT bin k (0..N-1)")
    ax[1].set_ylabel("rad")
    ax[1].set_yticks([-np.pi, 0, np.pi], ["−π", "0", "π"])
    fig.tight_layout()
    figs["symmetry"] = _save(fig, "p2_frame_symmetry.png")
    return figs


# ===================================================================== PART 3
def part3_magnitude_phase(x: np.ndarray, X: np.ndarray, tag: str = "clean", detailed: bool = True,
                          ref: np.ndarray | None = None, save_audio: bool = True) -> dict:
    """sig1 (zero phase), sig2 (random PH1: unconstrained and odd-symmetric)."""
    ref = x if ref is None else ref
    L = len(x)
    MG, PH = np.abs(X), np.angle(X)
    k = np.arange(N)[:, None]
    rng = np.random.default_rng(SEED + 3)

    sig1 = pl.istft(MG.astype(np.complex128), L)                       # literally phase = 0
    sig1c = pl.istft(MG * np.exp(1j * np.pi * k), L)                   # zero phase about window centre
    PH1_u = pl.unconstrained_random_phase(N, X.shape[1], rng)          # same size as PH, no symmetry
    sig2_c = pl.istft(MG * np.exp(1j * PH1_u), L)                      # complex!
    PH1_o = pl.odd_random_phase(N, X.shape[1], rng)                    # odd-symmetric random
    sig2 = pl.istft(MG * np.exp(1j * PH1_o), L)
    phase_only = pl.istft(np.exp(1j * PH), L)                          # control: |X| = 1

    variants = {
        "sig1 (zero phase)": sig1,
        "sig1c (zero phase about window centre)": sig1c,
        "sig2 unconstrained PH1: real part": sig2_c.real,
        "sig2 odd-symmetric PH1": sig2,
        "control: phase only (MG = 1)": phase_only,
    }
    if not detailed:      # noisy signals: only the two signals the assignment asks for
        variants = {k: variants[k] for k in ("sig1 (zero phase)", "sig2 odd-symmetric PH1")}
    rows = []
    for name, y in variants.items():
        d = pl.describe(ref, y)
        d["variant"] = name
        rows.append(d)
    df = pd.DataFrame(rows).set_index("variant")
    realness = {
        "sig1_max_abs_imag": float(np.max(np.abs(sig1.imag))),
        "sig2_unconstrained_imag_energy_ratio": pl.imag_energy_ratio(sig2_c),
        "sig2_unconstrained_max_abs_imag": float(np.max(np.abs(sig2_c.imag))),
        "sig2_odd_max_abs_imag": float(np.max(np.abs(sig2.imag))),
        "PH1_unconstrained_is_hermitian": pl.is_hermitian(np.exp(1j * PH1_u)),
        "PH1_odd_is_hermitian": pl.is_hermitian(np.exp(1j * PH1_o)),
    }
    # waveform statistics that separate a coherent buzz from an incoherent noise
    def stats(y):
        y = np.real(y)
        ac = signal.correlate(y[:32000], y[:32000], mode="full", method="fft")[32000 - 1:]
        ac = ac / ac[0]
        return {"kurtosis": float(np.mean((y - y.mean()) ** 4) / np.var(y) ** 2),
                "crest_dB": float(20 * np.log10(np.max(np.abs(y)) / np.sqrt(np.mean(y ** 2)))),
                "autocorr_at_lag_H": float(ac[H])}
    wstats = pd.DataFrame({"original": stats(x), "sig1": stats(sig1), "sig1c": stats(sig1c),
                           "sig2": stats(sig2)}).T

    if save_audio:
        d = AUD / tag
        d.mkdir(parents=True, exist_ok=True)
        pl.save_wav(str(d / "sig1_zero_phase.wav"), sig1)
        pl.save_wav(str(d / "sig2_odd_random_phase.wav"), sig2)
        if detailed:
            pl.save_wav(str(d / "original.wav"), x, peak=None)
            pl.save_wav(str(d / "sig1c_zero_phase_centred.wav"), sig1c)
            pl.save_wav(str(d / "sig2_unconstrained_real_part.wav"), sig2_c.real)
            pl.save_wav(str(d / "sig2_unconstrained_imag_part.wav"), sig2_c.imag)
            pl.save_wav(str(d / "control_phase_only.wav"), phase_only)

    figs = {}
    if detailed:
        df.to_csv(TAB / "p3_metrics_clean.csv")
        wstats.to_csv(TAB / "p3_waveform_stats_clean.csv")
        figs = _part3_figs(x, X, MG, sig1, sig1c, sig2_c, sig2, PH1_u, PH1_o)
    return {"sig1": sig1.real, "sig1c": sig1c.real, "sig2": sig2.real, "sig2_complex": sig2_c,
            "metrics": df, "realness": realness, "wstats": wstats, "figs": figs}


def _part3_figs(x, X, MG, sig1, sig1c, sig2_c, sig2, PH1_u, PH1_o) -> dict:
    figs = {}
    kmax = int(4000 / FS * N)
    m = int(np.argmax(np.sum(MG[:kmax] ** 2, axis=0)))
    w = pl.hann(N)
    n = np.arange(N)
    frame = np.real(np.fft.ifft(X[:, m]))
    r0 = np.real(np.fft.ifft(MG[:, m]))
    rc = np.real(np.fft.ifft(MG[:, m] * np.exp(1j * np.pi * n)))

    # ---- what zero phase does to ONE frame
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.3), sharey=True)
    sc = np.max(np.abs(r0))
    ax[0].plot(n, frame / sc, color=SERIES[0], lw=0.9)
    ax[0].set_title("(a) analysed frame w[n]x[n+mH]", loc="left")
    ax[1].plot(n, r0 / sc, color=SERIES[1], lw=0.9, label="IFFT(MG): zero phase")
    ax[1].plot(n, w, color=INK2, lw=0.9, ls="--", label="synthesis window")
    ax[1].plot(n, w * r0 / sc, color=SERIES[2], lw=1.4, label="what survives OLA")
    ax[1].set_title("(b) zero phase: pulse at n = 0", loc="left")
    ax[1].legend(loc="lower center", fontsize=8)
    ax[2].plot(n, rc / sc, color=SERIES[1], lw=0.9)
    ax[2].plot(n, w, color=INK2, lw=0.9, ls="--")
    ax[2].plot(n, w * rc / sc, color=SERIES[2], lw=1.4)
    ax[2].set_title("(c) zero phase about centre: n = N/2", loc="left")
    for a in ax:
        a.set_xlabel("n (samples in frame)")
    fig.tight_layout()
    figs["frame_zero_phase"] = _save(fig, "p3_frame_zero_phase.png")

    # ---- waveforms: full and a 60 ms zoom
    t = np.arange(len(x)) / FS
    nz = lambda y: np.real(y) / np.max(np.abs(np.real(y)))
    fig = plt.figure(figsize=(11, 8.2))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1.5])
    ax = [fig.add_subplot(gs[i, :]) for i in range(3)]
    for i, (y, lab, c) in enumerate([(x, "original", SERIES[0]), (sig1, "sig1 (zero phase)", SERIES[1]),
                                      (sig2, "sig2 (odd random phase)", SERIES[2])]):
        ax[i].plot(t, nz(y), color=c, lw=0.5)
        ax[i].set_ylabel(lab, fontsize=8.5)
        ax[i].set_ylim(-1.05, 1.05)
        ax[i].set_xlim(0, t[-1])
    ax[0].set_title("Waveforms (each peak-normalised) and two 60 ms zooms", loc="left")
    ax[2].set_xlabel("Time (s)")
    for j, (za, title) in enumerate([(1.300, "onset at 1.33 s: energy appears BEFORE the onset (pre-echo)"),
                                      (1.440, "steady vowel: sig1 repeats every hop H = 8 ms")]):
        a = fig.add_subplot(gs[3, j])
        z0, z1 = int(za * FS), int((za + 0.06) * FS)
        tz = t[z0:z1] * 1000
        for i, (y, lab) in enumerate([(x, "original"), (sig1, "sig1"), (sig2, "sig2")]):
            yy = np.real(y)[z0:z1]
            a.plot(tz, yy / np.max(np.abs(yy)) + 2.2 * (2 - i), color=SERIES[i], lw=1.0)
            a.text(tz[-1] + 0.8, 2.2 * (2 - i), lab, va="center", fontsize=8.5, color=INK)
        for gx in np.arange(np.ceil(tz[0] / 8) * 8, tz[-1], 8.0):
            a.axvline(gx, color=GRID, lw=0.8, zorder=0)
        a.set_yticks([])
        a.set_xlim(tz[0], tz[-1] + 9)
        a.set_xlabel("Time (ms); grey lines every 8 ms")
        a.set_title(title, loc="left", fontsize=9.5)
    fig.tight_layout()
    figs["waveforms"] = _save(fig, "p3_waveforms.png")

    # ---- spectrograms of re-analysed signals + LTAS + energy envelope
    M1, M2 = np.abs(pl.stft(sig1.real)), np.abs(pl.stft(sig2.real))
    g1 = np.linalg.norm(MG) / np.linalg.norm(M1)
    g2 = np.linalg.norm(MG) / np.linalg.norm(M2)
    fig, ax = plt.subplots(3, 1, figsize=(10, 7.6), sharex=True)
    for a, M, lab in [(ax[0], MG, "original |STFT|"), (ax[1], g1 * M1, "|STFT(sig1)|, zero phase: flat 125 Hz comb"),
                      (ax[2], g2 * M2, "|STFT(sig2)|, odd random phase: harmonics smeared")]:
        im = _imshow_spec(a, M, ref=MG.max(), fmax=4000)
        a.set_xlim(1.0, 3.0)
        a.set_title(lab + " (re-analysed, gain-matched, 1 to 3 s)", loc="left")
    ax[2].set_xlabel("Time (s)")
    fig.colorbar(im, ax=ax, shrink=0.6, pad=0.01).set_label("dB re. max of original")
    figs["specs"] = _save(fig, "p3_spectrograms_sig1_sig2.png")

    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    f = np.arange(N // 2 + 1) * FS / N
    for i, (M, lab) in enumerate([(MG, "original"), (g1 * M1, "sig1"), (g2 * M2, "sig2")]):
        ltas = 10 * np.log10(np.mean(M[: N // 2 + 1] ** 2, axis=1))
        ax[0].plot(f[:65], ltas[:65] - ltas.max(), color=SERIES[i], lw=1.6, marker=MARKERS[i], ms=3.5,
                   markevery=4, label=lab)
    ax[0].set_xlabel("Frequency (Hz)")
    ax[0].set_ylabel("dB")
    ax[0].set_title("Long-term spectrum 0 to 2 kHz", loc="left")
    ax[0].legend()
    for xv in range(125, 2001, 125):
        ax[0].axvline(xv, color=GRID, lw=0.6, zorder=0)
    tt = np.arange(MG.shape[1]) * H / FS
    for i, (M, lab) in enumerate([(MG, "original"), (g1 * M1, "sig1"), (g2 * M2, "sig2")]):
        env = 10 * np.log10(np.sum(M ** 2, axis=0) + 1e-12)
        ax[1].plot(tt, env - 10 * np.log10(np.sum(MG ** 2, axis=0).max()), color=SERIES[i], lw=1.0, label=lab)
    ax[1].set_xlim(1.0, 3.0)
    ax[1].set_ylim(-60, 5)
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Frame energy (dB)")
    ax[1].set_title("Temporal envelope (1 to 3 s)", loc="left")
    ax[1].legend(loc="lower right")
    fig.tight_layout()
    figs["ltas_env"] = _save(fig, "p3_ltas_envelope.png")

    # ---- sig2 with unconstrained phase is complex
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.2))
    t = np.arange(len(x)) / FS
    ax[0].plot(t, sig2_c.real, color=SERIES[0], lw=0.4, label="Re{sig2}")
    ax[0].plot(t, sig2_c.imag, color=SERIES[1], lw=0.4, alpha=0.8, label="Im{sig2}")
    ax[0].set_title(f"Unconstrained PH1: Im carries {100 * pl.imag_energy_ratio(sig2_c):.1f}% of energy",
                    loc="left")
    ax[0].set_xlabel("Time (s)")
    ax[0].legend(loc="upper right")
    kk = np.arange(N)
    ax[1].plot(kk, PH1_u[:, m], ".", color=SERIES[1], ms=2.2, label="unconstrained PH1")
    ax[1].plot(kk, PH1_o[:, m] + 0.0, ".", color=SERIES[0], ms=2.2, label="odd PH1")
    ax[1].axvline(N / 2, color=INK2, lw=0.8, ls="--")
    ax[1].set_xlabel("FFT bin k")
    ax[1].set_yticks([-np.pi, 0, np.pi], ["−π", "0", "π"])
    ax[1].set_title("One frame of PH1: only the odd one is point-symmetric about N/2", loc="left")
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, markerscale=3)
    fig.tight_layout()
    figs["complex_sig2"] = _save(fig, "p3_sig2_complex.png")
    return figs


# ===================================================================== PART 4
def part4_realness(MG: np.ndarray, n_trials: int = 200) -> dict:
    """Numerically verify which phase patterns give a real inverse, and count them."""
    rng = np.random.default_rng(SEED + 4)
    nf = MG.shape[1]
    L = (nf - 1) * H          # any length works for this check

    def trial(ph):
        y = pl.istft(MG * np.exp(1j * ph), L)
        return pl.imag_energy_ratio(y), float(np.max(np.abs(y.imag)))

    def odd_dcnyq_arbitrary():
        ph = pl.odd_random_phase(N, nf, rng)
        ph[0] = rng.uniform(-np.pi, np.pi, nf)
        ph[N // 2] = rng.uniform(-np.pi, np.pi, nf)
        return ph

    def odd_plus_const():
        return pl.odd_random_phase(N, nf, rng) + np.pi / 4

    def even_random():
        ph = pl.odd_random_phase(N, nf, rng)
        ph[N // 2 + 1:] = ph[1:N // 2][::-1]          # even instead of odd
        return ph

    gens = {
        "zero phase (PH1 = 0)": lambda: np.zeros((N, nf)),
        "linear phase (pure delay, −2πkd/N)": lambda: (-2 * np.pi * np.arange(N)[:, None] * 37 / N) * np.ones((1, nf)),
        "odd random, DC and Nyquist in {0, π}": lambda: pl.odd_random_phase(N, nf, rng),
        "odd random, DC and Nyquist arbitrary": odd_dcnyq_arbitrary,
        "odd random + constant π/4": odd_plus_const,
        "even random (wrong symmetry)": even_random,
        "unconstrained i.i.d. uniform": lambda: pl.unconstrained_random_phase(N, nf, rng),
    }
    rows = []
    for name, g in gens.items():
        reps = n_trials if "random" in name or "unconstrained" in name else 1
        reps = min(reps, 20)
        r = np.array([trial(g()) for _ in range(reps)])
        rows.append({"PH1 pattern": name, "trials": reps, "imag/total energy (mean)": r[:, 0].mean(),
                     "max |Im{sig2}|": r[:, 1].max(), "real?": bool(r[:, 1].max() < 1e-12)})
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "p4_realness.csv", index=False)

    # counting
    free = N // 2 - 1
    count = {"N": N, "frames_M": nf, "free_phases_per_frame": free, "binary_choices_per_frame": 4,
             "real_degrees_of_freedom_total": free * nf}
    for bits in (1, 3, 8, 16):
        per_frame = free * bits * np.log10(2) + np.log10(4)
        count[f"log10_patterns_{bits}bit"] = {"per_frame": per_frame, "whole_signal": per_frame * nf}

    # figure: log imag-energy ratio per case
    fig, ax = plt.subplots(figsize=(10, 3.4))
    vals = np.maximum(df["imag/total energy (mean)"].to_numpy(), 1e-34)
    ypos = np.arange(len(df))[::-1]
    ax.barh(ypos, np.log10(vals) + 35, left=-35, color=SERIES[0], height=0.55)
    for yv, v in zip(ypos, vals):
        lab = "real (machine precision)" if v < 1e-25 else f"{v:.3g}"
        ax.text(np.log10(v) + 0.5 if v >= 1e-25 else np.log10(v) + 0.5, yv, lab, va="center", fontsize=8.5, color=INK)
    ax.set_yticks(ypos, df["PH1 pattern"])
    ax.set_xlim(-35, 6)
    ax.set_xlabel("log10( energy of Im{sig2} / total energy )")
    ax.set_title("Only odd-symmetric PH1 with DC and Nyquist phase in {0, π} gives a real sig2", loc="left")
    ax.axvline(-15, color=GRID, lw=0.8)
    fig.tight_layout()
    fig_p = _save(fig, "p4_realness.png")
    return {"table": df, "count": count, "figs": {"realness": fig_p}}


# ===================================================================== PART 5
def part5_window_sweep(x: np.ndarray, windows=(256, 512, 1024, 2048, 4096, 8192, 16384, 32768),
                       iters: int = 100) -> dict:
    """How long must the frame be before an odd-random-phase version is unintelligible?"""
    L = len(x)
    rows = []
    for n in windows:
        h = n // 4
        rng = np.random.default_rng(SEED + 5)
        Xn = pl.stft(x, n, h)
        Mn = np.abs(Xn)
        s = pl.istft(Mn * np.exp(1j * pl.odd_random_phase(n, Xn.shape[1], rng)), L, h).real
        b = pl.griffin_lim(np.abs(pl.stft(s, n, h)), L, h, n_iter=iters, rng=np.random.default_rng(1))
        o = pl.griffin_lim(Mn, L, h, n_iter=iters, rng=np.random.default_rng(1))
        otp = pl.otp_scramble(x, n, SEED)[:L]
        rows.append({"window_ms": n * 1000 / FS, "N": n,
                     "STOI scrambled": pl.stoi(x, s), "ESTOI scrambled": pl.stoi(x, s, extended=True),
                     "STOI blind GLA": pl.stoi(x, b), "ESTOI blind GLA": pl.stoi(x, b, extended=True),
                     "STOI GLA true MG": pl.stoi(x, o), "ESTOI GLA true MG": pl.stoi(x, o, extended=True),
                     "STOI keyed OTP scrambled": pl.stoi(x, otp),
                     "ESTOI keyed OTP scrambled": pl.stoi(x, otp, extended=True),
                     "energy scrambled/original dB": 10 * np.log10(np.sum(s ** 2) / np.sum(x ** 2))})
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "p5_window_sweep.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(12, 3.9), sharey=True)
    for j, metric in enumerate(["STOI", "ESTOI"]):
        a = ax[j]
        for i, (col, lab) in enumerate([("scrambled", "scrambled (odd random phase)"),
                                        ("blind GLA", "blind Griffin-Lim on scrambled audio"),
                                        ("GLA true MG", "Griffin-Lim given the true MG")]):
            a.plot(df["window_ms"], df[f"{metric} {col}"], color=SERIES[i], marker=MARKERS[i],
                   ms=6 if i != 1 else 9, mfc="none" if i == 1 else SERIES[i], lw=2, label=lab)
        a.set_xscale("log", base=2)
        a.xaxis.set_minor_formatter(NullFormatter())
        a.set_xticks(df["window_ms"], [f"{v:g}" for v in df["window_ms"]])
        a.set_xlabel("Frame length (ms), hop = 1/4 frame")
        a.set_title(metric, loc="left")
        a.axvline(1024, color=INK2, ls="--", lw=0.8)
        a.grid(True, axis="y")
    ax[0].set_ylabel("score vs original")
    ax[0].text(1024 * 1.05, 0.95, "chosen\nscrambler", fontsize=8, color=INK2, va="top")
    ax[1].legend(loc="lower left")
    fig.tight_layout()
    return {"table": df, "figs": {"sweep": _save(fig, "p5_window_sweep.png")}}


def part5_scramble_recover(x_in: np.ndarray, ref: np.ndarray, tag: str, detailed: bool = False,
                           save_audio: bool = True, key: int = SEED) -> dict:
    """Scramble with odd random phase (1.024 s frames) and try three ways to get it back."""
    L = len(x_in)
    Xs = pl.stft(x_in, SCR_N, SCR_H)
    Ms = np.abs(Xs)
    PH1 = pl.odd_random_phase(SCR_N, Xs.shape[1], np.random.default_rng(key))
    scr = pl.istft(Ms * np.exp(1j * PH1), L, SCR_H)
    imag_max = float(np.max(np.abs(scr.imag)))
    scr = scr.real
    # R1 blind: receiver only has the scrambled audio (and knows the frame settings)
    blind, hist_b = pl.griffin_lim(np.abs(pl.stft(scr, SCR_N, SCR_H)), L, SCR_H, n_iter=GLA_ITERS,
                                   rng=np.random.default_rng(1), return_history=True)
    # R2 side information: receiver is also sent the true magnitude MG (32 ms STFT)
    MG = np.abs(pl.stft(x_in))
    oracle, hist_o = pl.griffin_lim(MG, L, H, n_iter=GLA_ITERS, rng=np.random.default_rng(1),
                                    return_history=True)
    # R3 keyed: phase one-time pad on non-overlapping 1.024 s frames
    otp = pl.otp_scramble(x_in, SCR_N, key)
    keyed = pl.otp_descramble(otp, SCR_N, key, L)

    res = {}
    for name, y in [("scrambled (replace phase)", scr), ("R1 blind Griffin-Lim", blind),
                    ("R2 Griffin-Lim with true MG", oracle), ("scrambled (keyed one-time pad)", otp[:L]),
                    ("R3 keyed descramble", keyed)]:
        res[name] = {"STOI": pl.stoi(ref, y), "ESTOI": pl.stoi(ref, y, extended=True),
                     "SNR_vs_input_dB": pl.snr_db(x_in, y)}
    res["_imag_max_scrambled"] = imag_max
    res["_SC_blind_vs_true_MG"] = pl.spectral_convergence(
        np.abs(pl.stft(blind, SCR_N, SCR_H)), Ms)
    Mscr = np.abs(pl.stft(scr, SCR_N, SCR_H))
    res["_SC_scrambled_vs_true_MG"] = pl.spectral_convergence(Mscr * np.linalg.norm(Ms) / np.linalg.norm(Mscr), Ms)

    if save_audio:
        d = AUD / tag
        d.mkdir(parents=True, exist_ok=True)
        pl.save_wav(str(d / "scrambled_odd_random_phase_1s.wav"), scr)
        pl.save_wav(str(d / "recovered_R1_blind_griffin_lim.wav"), blind)
        pl.save_wav(str(d / "recovered_R2_griffin_lim_true_MG.wav"), oracle)
        pl.save_wav(str(d / "scrambled_keyed_otp_1s.wav"), otp)
        pl.save_wav(str(d / "recovered_R3_keyed.wav"), keyed)
        pl.save_wav(str(CHAL / f"{tag}__scrambled.wav"), scr)
        pl.save_wav(str(CHAL / f"{tag}__keyed.wav"), otp)

    figs = {}
    if detailed:
        fig, ax = plt.subplots(5, 1, figsize=(10, 10.5), sharex=True)
        ref_max = np.abs(pl.stft(x_in)).max()
        for a, (y, lab) in zip(ax, [(x_in, "input"), (scr, "scrambled: odd random phase, 1.024 s frames"),
                                    (blind, "R1: blind Griffin-Lim from scrambled audio"),
                                    (oracle, "R2: Griffin-Lim given the true 32 ms MG"),
                                    (keyed, "R3: keyed descrambling (phase one-time pad)")]):
            My = np.abs(pl.stft(y))
            g = np.linalg.norm(MG) / max(np.linalg.norm(My), 1e-12)
            _imshow_spec(a, g * My, ref=ref_max)
            st = pl.stoi(ref, y)
            a.set_title(f"{lab}    STOI {st:.3f}", loc="left")
        ax[-1].set_xlabel("Time (s)")
        fig.tight_layout()
        figs["recovery_specs"] = _save(fig, f"p5_recovery_{tag}.png")

        fig, ax = plt.subplots(figsize=(7.5, 3.3))
        ax.plot(np.arange(1, GLA_ITERS + 1), hist_b, color=SERIES[0], lw=2,
                label="R1 blind (target = scrambled magnitude)")
        ax.plot(np.arange(1, GLA_ITERS + 1), hist_o, color=SERIES[1], lw=2, label="R2 with true MG")
        ax.set_yscale("log")
        ax.set_xlabel("Fast Griffin-Lim iteration")
        ax.set_ylabel("spectral convergence")
        ax.set_title("Both runs converge to their own target magnitude", loc="left")
        ax.legend()
        ax.grid(True, which="both", axis="y")
        fig.tight_layout()
        figs["gla_convergence"] = _save(fig, f"p5_gla_convergence_{tag}.png")
    return {"metrics": res, "figs": figs, "scr": scr, "blind": blind, "oracle": oracle, "keyed": keyed}


# ===================================================================== PART 6
FILTERS = {
    "BPF-T 300-3400 Hz (telephone), Butterworth, zero-phase": dict(lo=300, hi=3400, kind="butter_zero_phase"),
    "BPF-S 120-4500 Hz (STOI band), Butterworth, zero-phase": dict(lo=120, hi=4500, kind="butter_zero_phase"),
    "BPF-T ideal brick-wall": dict(lo=300, hi=3400, kind="brickwall"),
}
SHORT = {k: s for k, s in zip(FILTERS, ["BPF-T", "BPF-S", "BPF-T brick-wall"])}


def oracle_wiener(noisy: np.ndarray, clean: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """Time-frequency Wiener gain with oracle PSDs: the reference point a fixed BPF is compared with."""
    S, Nn, Y = pl.stft(clean), pl.stft(noise), pl.stft(noisy)
    G = np.abs(S) ** 2 / (np.abs(S) ** 2 + np.abs(Nn) ** 2 + 1e-12)
    return pl.istft(G * Y, len(noisy)).real


def part6_noise_figs(x: np.ndarray, noisy: dict) -> dict:
    figs = {}
    # the noises themselves
    fig, ax = plt.subplots(2, 3, figsize=(13, 5.0), gridspec_kw={"height_ratios": [1, 2]}, layout="constrained")
    for j, n in enumerate(NOISES):
        v = noisy[(n, 0)][1]
        t = np.arange(len(v)) / FS
        ax[0, j].plot(t, v, color=NOISE_COLOR[n], lw=0.4)
        ax[0, j].set_title(f"{n}", loc="left")
        ax[0, j].set_xlim(0, t[-1])
        im = _imshow_spec(ax[1, j], np.abs(pl.stft(v)))
        ax[1, j].set_xlim(0, t[-1])
        ax[1, j].set_xlabel("Time (s)")
        if j:
            ax[1, j].set_ylabel("")
    fig.colorbar(im, ax=ax[:, 2], shrink=0.6, pad=0.02, location="right").set_label("dB")
    fig.suptitle("NoiseX-92 segments used (resampled to 16 kHz; machinegun was 8 kHz so it stops at 4 kHz)",
                 x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    figs["noises"] = _save(fig, "p6_noises.png")

    # Part 2 repeated: MG and PH for every SNR, one figure per noise
    for n in NOISES:
        fig, ax = plt.subplots(len(SNRS), 2, figsize=(13, 2.0 * len(SNRS) + 0.6), sharex=True, sharey=True,
                               layout="constrained")
        for i, snr in enumerate(SNRS):
            Xn = pl.stft(noisy[(n, snr)][0])
            im0 = _imshow_spec(ax[i, 0], np.abs(Xn))
            im1 = _imshow_spec(ax[i, 1], np.abs(Xn), data=np.angle(Xn), cmap=CYCLIC, vmin=-np.pi, vmax=np.pi)
            ax[i, 0].set_title(f"MG, {n} at {snr:+d} dB", loc="left", fontsize=10)
            ax[i, 1].set_title(f"PH, {n} at {snr:+d} dB", loc="left", fontsize=10)
            ax[i, 1].set_ylabel("")
        ax[-1, 0].set_xlabel("Time (s)")
        ax[-1, 1].set_xlabel("Time (s)")
        fig.colorbar(im0, ax=ax[:, 0], shrink=0.4, pad=0.01, location="right").set_label("dB")
        cb = fig.colorbar(im1, ax=ax[:, 1], shrink=0.4, pad=0.01, location="right", ticks=[-np.pi, 0, np.pi])
        cb.ax.set_yticklabels(["−π", "0", "π"])
        cb.set_label("rad")
        figs[f"MG_PH_{n}"] = _save(fig, f"p6_MG_PH_{n}.png")
    return figs


def part6_run_all(x: np.ndarray, noisy: dict) -> dict:
    """Repeat Parts 2 to 5 for every noisy signal and apply the band-pass filters."""
    rows, bpf_rows, realness_rows = [], [], []
    per_signal_figs = {}
    for (n, snr), (y, v) in noisy.items():
        tag = f"{n}_{snr:+d}dB".replace("+", "p").replace("-", "m")
        d = AUD / tag
        d.mkdir(parents=True, exist_ok=True)
        pl.save_wav(str(d / "noisy.wav"), y)
        Xn = pl.stft(y)
        p3 = part3_magnitude_phase(y, Xn, tag=tag, detailed=False, ref=x)
        p5 = part5_scramble_recover(y, x, tag=tag)
        base = {"noise": n, "SNR_dB": snr}
        rows.append({**base, "variant": "noisy", "STOI": pl.stoi(x, y), "ESTOI": pl.stoi(x, y, extended=True)})
        for name, key in [("sig1 (zero phase)", "sig1 (zero phase)"),
                          ("sig2 (odd random phase)", "sig2 odd-symmetric PH1")]:
            rows.append({**base, "variant": name, "STOI": p3["metrics"].loc[key, "STOI"],
                         "ESTOI": p3["metrics"].loc[key, "ESTOI"]})
        for name, mval in p5["metrics"].items():
            if not name.startswith("_"):
                rows.append({**base, "variant": name, "STOI": mval["STOI"], "ESTOI": mval["ESTOI"]})
        realness_rows.append({**base, **{k: v_ for k, v_ in p3["realness"].items() if "max_abs" in k or "ratio" in k},
                              "scrambled_max_abs_imag": p5["metrics"]["_imag_max_scrambled"]})
        # band-pass filters
        st0, est0 = pl.stoi(x, y), pl.stoi(x, y, extended=True)
        for fname, cfg in FILTERS.items():
            yf = pl.apply_bandpass(y, **cfg)
            xf = pl.apply_bandpass(x, **cfg)
            vf = pl.apply_bandpass(v, **cfg)
            st, est = pl.stoi(x, yf), pl.stoi(x, yf, extended=True)
            bpf_rows.append({**base, "filter": SHORT[fname], "STOI": st, "ESTOI": est,
                             "dSTOI": st - st0, "dESTOI": est - est0,
                             "broadband SNR after filter (dB)": 10 * np.log10(np.sum(xf ** 2) / np.sum(vf ** 2)),
                             "SNR gain (dB)": 10 * np.log10(np.sum(xf ** 2) / np.sum(vf ** 2)) - snr})
            if SHORT[fname] == "BPF-T":
                pl.save_wav(str(d / "bpf_telephone_300_3400.wav"), yf)
            if SHORT[fname] == "BPF-S":
                pl.save_wav(str(d / "bpf_stoi_band_120_4500.wav"), yf)
        w = oracle_wiener(y, x, v)
        pl.save_wav(str(d / "oracle_wiener.wav"), w)
        st, est = pl.stoi(x, w), pl.stoi(x, w, extended=True)
        bpf_rows.append({**base, "filter": "oracle Wiener (reference)", "STOI": st, "ESTOI": est,
                         "dSTOI": st - st0, "dESTOI": est - est0,
                         "broadband SNR after filter (dB)": pl.snr_db(x, w, align_scale=False),
                         "SNR gain (dB)": pl.snr_db(x, w, align_scale=False) - snr})
        if snr == 0:
            per_signal_figs[n] = _noisy_variant_fig(x, y, p3, p5, n, snr)
        print(f"  done {n:10s} {snr:+3d} dB  STOI noisy {st0:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(TAB / "p6_all_variants_long.csv", index=False)
    piv = df.pivot_table(index=["noise", "SNR_dB"], columns="variant", values="STOI", sort=False)
    piv.to_csv(TAB / "p6_STOI_wide.csv")
    piv_e = df.pivot_table(index=["noise", "SNR_dB"], columns="variant", values="ESTOI", sort=False)
    piv_e.to_csv(TAB / "p6_ESTOI_wide.csv")
    bdf = pd.DataFrame(bpf_rows)
    bdf.to_csv(TAB / "p6_bpf.csv", index=False)
    rdf = pd.DataFrame(realness_rows)
    rdf.to_csv(TAB / "p6_realness_noisy.csv", index=False)
    figs = _part6_summary_figs(df, bdf)
    figs.update({f"variants_{k}": v for k, v in per_signal_figs.items()})
    return {"long": df, "STOI_wide": piv, "ESTOI_wide": piv_e, "bpf": bdf, "realness": rdf, "figs": figs}


def _noisy_variant_fig(x, y, p3, p5, n, snr):
    fig, ax = plt.subplots(5, 1, figsize=(10, 9.6), sharex=True)
    MG = np.abs(pl.stft(y))
    for a, (s, lab) in zip(ax, [(y, f"noisy ({n}, {snr:+d} dB)"), (p3["sig1"], "sig1: zero phase"),
                                (p3["sig2"], "sig2: odd random phase"),
                                (p5["scr"], "scrambled (1.024 s odd random phase)"),
                                (p5["oracle"], "R2: Griffin-Lim with true MG")]):
        Ms = np.abs(pl.stft(s))
        g = np.linalg.norm(MG) / max(np.linalg.norm(Ms), 1e-12)
        _imshow_spec(a, g * Ms, ref=MG.max())
        a.set_title(f"{lab}    STOI vs clean {pl.stoi(x, s):.3f}", loc="left")
    ax[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    return _save(fig, f"p6_variants_{n}_0dB.png")


def _part6_summary_figs(df, bdf) -> dict:
    figs = {}
    # STOI and ESTOI of the noisy mixtures
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.8), sharey=True)
    for j, met in enumerate(["STOI", "ESTOI"]):
        for i, n in enumerate(NOISES):
            s = df[(df.noise == n) & (df.variant == "noisy")]
            ax[j].plot(s.SNR_dB, s[met], color=NOISE_COLOR[n], marker=MARKERS[i], ms=6, lw=2, label=n)
        ax[j].set_title(f"{met} of noisy speech vs clean reference", loc="left")
        ax[j].set_xlabel("Input SNR (dB)")
        ax[j].set_xticks(SNRS)
        ax[j].set_xlim(-3, 11)
        ax[j].grid(True, axis="y")
    ax[0].set_ylabel("score")
    ax[0].legend(loc="lower right")
    fig.tight_layout()
    figs["stoi_noisy"] = _save(fig, "p6_stoi_noisy.png")

    # phase manipulations under noise
    variants = [("noisy", "noisy input"), ("sig1 (zero phase)", "sig1 zero phase"),
                ("sig2 (odd random phase)", "sig2 odd random phase"),
                ("scrambled (replace phase)", "scrambled 1 s (= R1 blind)"),
                ("R2 Griffin-Lim with true MG", "R2 GLA with true MG")]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.0), sharey=True)
    for j, n in enumerate(NOISES):
        for i, (v, lab) in enumerate(variants):
            s = df[(df.noise == n) & (df.variant == v)]
            ax[j].plot(s.SNR_dB, s.STOI, color=SERIES[i], marker=MARKERS[i], ms=6 if i != 4 else 9,
                       mfc="none" if i == 4 else SERIES[i], lw=2 if i != 4 else 1.2, ls="--" if i == 4 else "-",
                       label=lab)
        ax[j].set_title(n, loc="left")
        ax[j].set_xlabel("Input SNR (dB)")
        ax[j].set_xticks(SNRS)
        ax[j].grid(True, axis="y")
    ax[0].set_ylabel("STOI vs clean original")
    ax[2].legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    figs["phase_ops_noise"] = _save(fig, "p6_phase_ops_under_noise.png")

    # band-pass filtering
    fl = [("noisy", None), ("BPF-T", "BPF-T 300-3400 Hz"), ("BPF-S", "BPF-S 120-4500 Hz"),
          ("BPF-T brick-wall", "BPF-T ideal brick-wall"), ("oracle Wiener (reference)", "oracle Wiener (not a BPF)")]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.0), sharey=True)
    for j, n in enumerate(NOISES):
        base = df[(df.noise == n) & (df.variant == "noisy")]
        for i, (f, lab) in enumerate(fl):
            if f == "noisy":
                ax[j].plot(base.SNR_dB, base.STOI, color=SERIES[0], marker=MARKERS[0], ms=6, lw=2,
                           label="noisy (no filter)")
                continue
            s = bdf[(bdf.noise == n) & (bdf["filter"] == f)]
            ax[j].plot(s.SNR_dB, s.STOI, color=SERIES[i], marker=MARKERS[i], ms=8 if i == 1 else 6,
                       mfc="none" if i == 1 else SERIES[i], lw=2 if i != 1 else 1.4,
                       ls="--" if i == 1 else "-", label=lab)
        ax[j].set_title(n, loc="left")
        ax[j].set_xlabel("Input SNR (dB)")
        ax[j].set_xticks(SNRS)
        ax[j].grid(True, axis="y")
    ax[0].set_ylabel("STOI vs clean original")
    ax[2].legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    figs["bpf_stoi"] = _save(fig, "p6_bpf_stoi.png")
    return figs


def part6_filter_design(x: np.ndarray, noisy: dict) -> dict:
    """Frequency/phase responses, design diagnostics on clean speech, band-SNR invariance."""
    figs = {}
    sos = pl.butter_bandpass(300, 3400, 4)
    sos_w = pl.butter_bandpass(120, 4500, 4)
    f, hT = signal.sosfreqz(sos, worN=4096, fs=FS)
    _, hS = signal.sosfreqz(sos_w, worN=4096, fs=FS)
    hfir = signal.firwin(511, [300, 3400], pass_zero=False, fs=FS)
    _, hF = signal.freqz(hfir, worN=4096, fs=FS)
    w_gd, gd_c = signal.group_delay((*signal.sos2tf(sos),), w=np.linspace(50, FS / 2, 4096), fs=FS)

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 3.9))
    db = lambda h: 20 * np.log10(np.maximum(np.abs(h), 1e-8))
    ax[0].axvspan(150 * 2 ** (-1 / 6), 150 * 2 ** (14 / 3 + 1 / 6), color="#f0efec", zorder=0)
    ax[0].text(760, -45, "band analysed by STOI", fontsize=8, color=INK2, ha="center")
    ax[0].plot(f, db(hT) * 2, color=SERIES[0], lw=2, label="BPF-T Butterworth 4, zero-phase (|H|^2)")
    ax[0].plot(f, db(hT), color=SERIES[0], lw=1.2, ls="--", label="same filter applied once (causal)")
    ax[0].plot(f, db(hS) * 2, color=SERIES[1], lw=2, label="BPF-S 120-4500 Hz, zero-phase")
    ax[0].plot(f, db(hF), color=SERIES[2], lw=0.8, label="BPF-T FIR 511 taps (linear phase)")
    ax[0].set_xscale("log")
    ax[0].xaxis.set_minor_formatter(NullFormatter())
    ax[0].set_xlim(40, 8000)
    ax[0].set_ylim(-90, 5)
    ax[0].set_xlabel("Frequency (Hz)")
    ax[0].set_ylabel("dB")
    ax[0].set_title("Magnitude responses", loc="left")
    ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=7.8)
    ax[0].grid(True, which="major")
    ax[1].plot(w_gd, gd_c / FS * 1000, color=SERIES[0], lw=2, ls="--", label="Butterworth causal: non-linear phase")
    ax[1].axhline(0, color=SERIES[0], lw=2, label="Butterworth forward-backward: 0 ms")
    ax[1].axhline(255 / FS * 1000, color=SERIES[2], lw=1.4, label="FIR 511 taps: constant 15.9 ms (compensated)")
    ax[1].set_xscale("log")
    ax[1].xaxis.set_minor_formatter(NullFormatter())
    ax[1].set_xlim(100, 8000)
    ax[1].set_ylim(-1, 20)
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("Group delay (ms)")
    ax[1].set_title("Phase behaviour of the designs", loc="left")
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1, fontsize=7.8)
    ax[1].grid(True, which="major")
    fig.tight_layout()
    figs["bpf_design"] = _save(fig, "p6_bpf_design.png")

    # design diagnostics on CLEAN speech and on white noise at 0 dB
    designs = {
        "Butterworth-4 300-3400, zero-phase (filtfilt)": dict(lo=300, hi=3400, kind="butter_zero_phase"),
        "Butterworth-4 300-3400, causal (sosfilt)": dict(lo=300, hi=3400, kind="butter_causal"),
        "same |H| as causal, zero phase (FFT)": dict(lo=300, hi=3400, kind="butter_mag_only"),
        "FIR 511 taps 300-3400, linear phase": dict(lo=300, hi=3400, kind="fir_linear_phase", numtaps=511),
        "ideal brick-wall 300-3400": dict(lo=300, hi=3400, kind="brickwall"),
        "Butterworth-4 120-4500, zero-phase": dict(lo=120, hi=4500, kind="butter_zero_phase"),
        "Butterworth-4 500-2500, zero-phase (narrow)": dict(lo=500, hi=2500, kind="butter_zero_phase"),
        "ideal brick-wall 500-2500 (narrow)": dict(lo=500, hi=2500, kind="brickwall"),
    }
    rows = []
    yw = noisy[("white", 0)][0]
    for name, cfg in designs.items():
        xf = pl.apply_bandpass(x, **cfg)
        yf = pl.apply_bandpass(yw, **cfg)
        rows.append({"design": name, "STOI(clean, BPF(clean))": pl.stoi(x, xf),
                     "ESTOI(clean, BPF(clean))": pl.stoi(x, xf, extended=True),
                     "STOI white 0 dB after BPF": pl.stoi(x, yf)})
    ddf = pd.DataFrame(rows)
    ddf.to_csv(TAB / "p6_bpf_design_diagnostics.csv", index=False)

    # per-band SNR before/after BPF-T (the reason STOI does not move)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6), sharey=True)
    bs_rows = []
    for j, n in enumerate(NOISES):
        y, v = noisy[(n, 0)]
        cf, b0 = pl.third_octave_band_snr(x, v)
        _, b1 = pl.third_octave_band_snr(pl.apply_bandpass(x, 300, 3400), pl.apply_bandpass(v, 300, 3400))
        bs_rows += [{"noise": n, "band_Hz": c, "SNR_before_dB": a, "SNR_after_BPF_T_dB": b}
                    for c, a, b in zip(cf, b0, b1)]
        ax[j].plot(cf, b0, color=SERIES[0], marker="o", ms=6, lw=2, label="before BPF-T")
        ax[j].plot(cf, b1, color=SERIES[1], marker="x", ms=8, lw=1.4, ls="--", label="after BPF-T")
        ax[j].set_xscale("log")
        ax[j].xaxis.set_minor_formatter(NullFormatter())
        ax[j].set_xticks([150, 300, 600, 1200, 2400], ["150", "300", "600", "1.2k", "2.4k"])
        ax[j].set_title(f"{n}, 0 dB: SNR in each STOI band", loc="left")
        ax[j].set_xlabel("Third-octave band centre (Hz)")
        ax[j].grid(True, axis="y")
    ax[0].set_ylabel("band SNR (dB)")
    ax[0].legend()
    fig.tight_layout()
    figs["band_snr"] = _save(fig, "p6_band_snr.png")
    pd.DataFrame(bs_rows).to_csv(TAB / "p6_band_snr.csv", index=False)
    return {"diagnostics": ddf, "figs": figs}


def part6_mgph_snapshot(noisy: dict, snr: int = 0) -> str:
    """MG and PH of the three noisy signals at one SNR, in one compact figure."""
    fig, ax = plt.subplots(3, 2, figsize=(12, 7.2), sharex=True, sharey=True, layout="constrained")
    for i, n in enumerate(NOISES):
        Xn = pl.stft(noisy[(n, snr)][0])
        im0 = _imshow_spec(ax[i, 0], np.abs(Xn))
        im1 = _imshow_spec(ax[i, 1], np.abs(Xn), data=np.angle(Xn), cmap=CYCLIC, vmin=-np.pi, vmax=np.pi)
        label = "machine-gun" if n == "machinegun" else n
        ax[i, 0].set_title(f"MG, {label} noise at {snr} dB", loc="left")
        ax[i, 1].set_title(f"PH, {label} noise at {snr} dB", loc="left")
        ax[i, 1].set_ylabel("")
    ax[2, 0].set_xlabel("Time (s)")
    ax[2, 1].set_xlabel("Time (s)")
    fig.colorbar(im0, ax=ax[:, 0], shrink=0.5, pad=0.01).set_label("dB")
    cb = fig.colorbar(im1, ax=ax[:, 1], shrink=0.5, pad=0.01, ticks=[-np.pi, 0, np.pi])
    cb.ax.set_yticklabels(["−π", "0", "π"])
    return _save(fig, f"p6_MG_PH_{snr}dB_all.png")


def part6_band_contrib(x: np.ndarray, noisy: dict) -> dict:
    """Where does a band-pass filter lose STOI? Per-band STOI for noisy vs filtered (0 dB) and clean vs filtered."""
    rows = []
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.7), sharey=True)
    cases = [("clean", x)] + [(n, noisy[(n, 0)][0]) for n in NOISES]
    for j, (n, y) in enumerate(cases):
        for i, (lab, cfg) in enumerate([("no filter", None), ("BPF-T 300-3400", dict(lo=300, hi=3400)),
                                        ("BPF-S 120-4500", dict(lo=120, hi=4500))]):
            yy = y if cfg is None else pl.apply_bandpass(y, **cfg)
            cf, d = pl.stoi_per_band(x, yy)
            for c, v in zip(cf, d):
                rows.append({"signal": n if n == "clean" else f"{n} 0 dB", "filter": lab, "band_Hz": c, "score": v})
            ax[j].plot(cf, d, color=SERIES[i], marker=MARKERS[i], ms=6 if i != 1 else 8,
                       mfc="none" if i == 1 else SERIES[i],
                       lw=2 if i != 1 else 1.5, ls="--" if i == 1 else "-", label=lab)
        ax[j].set_xscale("log")
        ax[j].xaxis.set_minor_formatter(NullFormatter())
        ax[j].set_xticks([150, 300, 600, 1200, 2400], ["150", "300", "600", "1.2k", "2.4k"])
        ax[j].axvspan(300, 3400, color="#f0efec", zorder=0)
        ax[j].set_title("clean speech" if n == "clean" else f"{n}, 0 dB", loc="left")
        ax[j].set_xlabel("Band centre (Hz)")
        ax[j].grid(True, axis="y")
    ax[0].set_ylabel("per-band STOI (mean = STOI)")
    ax[0].legend(loc="lower center")
    fig.tight_layout()
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "p6_stoi_per_band.csv", index=False)
    return {"table": df, "figs": {"band_stoi": _save(fig, "p6_stoi_per_band.png")}}


# ======================================================================= MAIN
def run_all() -> dict:
    _dirs()
    R = {}
    print("[data] preparing speech and noises", flush=True)
    R["data"] = prepare_data()
    x = load_clean()
    print("[part 2] STFT, MG, PH", flush=True)
    p2 = part2_stft(x, "clean")
    R["part2"] = {"shape": list(p2["shape"]), "figs": p2["figs"]}
    print("[part 3] sig1 and sig2", flush=True)
    p3 = part3_magnitude_phase(x, p2["X"], "clean")
    R["part3"] = {"metrics": p3["metrics"].reset_index().to_dict("records"), "realness": p3["realness"],
                  "wstats": p3["wstats"].reset_index().to_dict("records"), "figs": p3["figs"]}
    print("[part 4] realness patterns", flush=True)
    p4 = part4_realness(p2["MG"])
    R["part4"] = {"table": p4["table"].to_dict("records"), "count": p4["count"], "figs": p4["figs"]}
    print("[part 5] window sweep + scramble/recover (clean)", flush=True)
    sw = part5_window_sweep(x)
    p5 = part5_scramble_recover(x, x, "clean", detailed=True)
    R["part5"] = {"sweep": sw["table"].to_dict("records"), "clean": p5["metrics"],
                  "figs": {**sw["figs"], **p5["figs"]}}
    print("[part 6] noisy mixtures", flush=True)
    noisy = make_noisy(x)
    R["part6"] = {"figs": part6_noise_figs(x, noisy)}
    R["part6"]["figs"]["MG_PH_0dB_all"] = part6_mgph_snapshot(noisy, 0)
    p6 = part6_run_all(x, noisy)
    fd = part6_filter_design(x, noisy)
    bc = part6_band_contrib(x, noisy)
    R["part6"]["figs"].update(p6["figs"])
    R["part6"]["figs"].update(fd["figs"])
    R["part6"]["figs"].update(bc["figs"])
    R["part6"]["bpf_design"] = fd["diagnostics"].to_dict("records")
    (OUT / "results.json").write_text(json.dumps(R, indent=2, default=float))
    return R


if __name__ == "__main__":
    run_all()
