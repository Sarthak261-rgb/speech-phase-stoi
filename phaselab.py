"""
Signal processing used in the assignment: two-sided STFT, least-squares inverse STFT,
phase generators, fast Griffin-Lim, the keyed scrambler, noise mixing, STOI and filters.

The full N-point FFT is used (not rfft) so that the conjugate symmetry needed for a
real output stays visible.
"""
from __future__ import annotations

import numpy as np
import soundfile as sf
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal
from pystoi import stoi as _pystoi

FS = 16_000          # working sample rate for the whole study
N_FFT = 512          # 32 ms analysis window
HOP = 128            # 8 ms hop  -> 75 % overlap, redundancy N/H = 4


# --------------------------------------------------------------------------- I/O
def load_wav(path: str, fs: int = FS) -> np.ndarray:
    """Read audio, convert to mono float64 and resample to `fs` if needed."""
    x, fs_in = sf.read(path, dtype="float64", always_2d=True)
    x = x.mean(axis=1)
    if fs_in != fs:
        g = np.gcd(int(fs), int(fs_in))
        x = signal.resample_poly(x, fs // g, fs_in // g)
    return x


def save_wav(path: str, x: np.ndarray, fs: int = FS, peak: float | None = 0.89) -> None:
    """Write 16-bit PCM. `peak` normalises for listening (None keeps the scale)."""
    y = np.real(np.asarray(x, dtype=np.complex128)).astype(np.float64)
    if peak is not None:
        m = np.max(np.abs(y))
        if m > 0:
            y = y * (peak / m)
    y = np.clip(y, -1.0, 1.0)
    sf.write(path, y, fs, subtype="PCM_16")


# -------------------------------------------------------------------------- STFT
def hann(n: int) -> np.ndarray:
    """Periodic Hann window."""
    return signal.windows.hann(n, sym=False)


def stft(x: np.ndarray, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    """Two-sided STFT. Returns X with shape (n_fft, n_frames): rows = bins k=0..N-1.

    Frames are centred: the signal is zero-padded by n_fft/2 on the left so the
    first frame is centred on sample 0. The FFT time origin of every frame is the
    *first* sample of the frame (standard numpy/librosa convention).
    """
    w = hann(n_fft)
    pad_l = n_fft // 2
    n_frames = int(np.ceil((len(x) + n_fft // 2) / hop)) + 1
    total = (n_frames - 1) * hop + n_fft
    xp = np.zeros(total, dtype=np.result_type(x, np.float64))
    xp[pad_l:pad_l + len(x)] = x
    frames = sliding_window_view(xp, n_fft)[::hop][:n_frames] * w
    return np.fft.fft(frames, axis=1).T


def istft(X: np.ndarray, length: int, hop: int = HOP) -> np.ndarray:
    """Least-squares inverse STFT (Griffin & Lim 1984, eq. 6).

    y[n] = sum_m w[n-mH] * ifft(X_m)[n-mH] / sum_m w^2[n-mH]

    This is the signal whose STFT is closest (in the Frobenius sense) to X, which
    is exactly what we need when X is a *modified* STFT (e.g. MG with a new phase)
    that does not correspond to any real signal. The output is complex in general;
    it is real only if every column of X is conjugate-symmetric.
    """
    n_fft, n_frames = X.shape
    w = hann(n_fft)
    frames = np.fft.ifft(X, axis=0).T * w            # (n_frames, n_fft), complex
    y = _overlap_add(frames, hop)
    wss = _overlap_add(np.broadcast_to(w ** 2, frames.shape), hop).real
    y /= np.maximum(wss, 1e-12)
    return y[n_fft // 2: n_fft // 2 + length]


def _overlap_add(frames: np.ndarray, hop: int) -> np.ndarray:
    """Vectorised overlap-add of (n_frames, n_fft) frames with hop | n_fft."""
    n_frames, n_fft = frames.shape
    r = n_fft // hop
    assert r * hop == n_fft, "hop must divide n_fft"
    out = np.zeros((n_frames + r - 1, hop), dtype=np.complex128)
    for i in range(r):                                   # r = 4 shifted block adds
        out[i:i + n_frames] += frames[:, i * hop:(i + 1) * hop]
    return out.reshape(-1)


def mag_phase(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.abs(X), np.angle(X)


# ------------------------------------------------------------- phase generators
def odd_random_phase(n_fft: int, n_frames: int, rng: np.random.Generator,
                     edge_values: tuple[float, ...] = (0.0, np.pi)) -> np.ndarray:
    """Random phase with the symmetry that guarantees a real inverse FFT.

    For every frame:  PH1[N-k] = -PH1[k]  (k = 1..N/2-1)  -> odd symmetry
                      PH1[0], PH1[N/2] in {0, pi}         -> real DC / Nyquist bins
    The N/2-1 free phases are i.i.d. uniform on [-pi, pi).
    """
    assert n_fft % 2 == 0
    half = n_fft // 2
    ph = np.zeros((n_fft, n_frames))
    ph[1:half] = rng.uniform(-np.pi, np.pi, size=(half - 1, n_frames))
    ph[half + 1:] = -ph[1:half][::-1]
    ph[0] = rng.choice(edge_values, size=n_frames)
    ph[half] = rng.choice(edge_values, size=n_frames)
    return ph


def unconstrained_random_phase(n_fft: int, n_frames: int, rng: np.random.Generator) -> np.ndarray:
    """Independent uniform phase in every bin, with no symmetry."""
    return rng.uniform(-np.pi, np.pi, size=(n_fft, n_frames))


def is_hermitian(X: np.ndarray, tol: float = 1e-9) -> bool:
    """True if every column satisfies X[k] = conj(X[(N-k) mod N])."""
    idx = (-np.arange(X.shape[0])) % X.shape[0]
    return bool(np.max(np.abs(X - np.conj(X[idx]))) <= tol * max(1.0, np.max(np.abs(X))))


def imag_energy_ratio(y: np.ndarray) -> float:
    """Share of the energy in the imaginary part."""
    y = np.asarray(y, dtype=np.complex128)
    e = np.sum(np.abs(y) ** 2)
    return float(np.sum(y.imag ** 2) / e) if e > 0 else 0.0


# ------------------------------------------------------------ Griffin-Lim (FGLA)
def griffin_lim(mag: np.ndarray, length: int, hop: int = HOP, n_iter: int = 300,
                alpha: float = 0.99, init_phase: np.ndarray | None = None,
                rng: np.random.Generator | None = None, return_history: bool = False):
    """Fast Griffin-Lim (Perraudin, Balazs & Sondergaard, 2013).

    Alternating projections between (a) the set of STFTs with the target
    magnitude and (b) the set of consistent STFTs (STFTs of real signals),
    with Nesterov-like momentum `alpha` (alpha = 0 gives classic GLA).
    """
    n_fft = mag.shape[0]
    if init_phase is None:
        rng = rng or np.random.default_rng(0)
        init_phase = odd_random_phase(n_fft, mag.shape[1], rng)
    c = mag * np.exp(1j * init_phase)
    t_prev = np.zeros_like(c)
    hist = []
    for _ in range(n_iter):
        y = istft(c, length, hop).real                # projection onto consistent set
        t = stft(y, n_fft, hop)
        if return_history:
            hist.append(spectral_convergence(np.abs(t), mag))
        c_new = t + alpha * (t - t_prev)              # momentum
        t_prev = t
        c = mag * c_new / np.maximum(np.abs(c_new), 1e-12)   # projection onto magnitude set
    y = istft(c, length, hop).real
    return (y, np.array(hist)) if return_history else y


# ------------------------------------------------- keyed "phase one-time pad"
def otp_scramble(x: np.ndarray, n: int, key: int) -> np.ndarray:
    """Keyed scrambler: add an odd random phase K (the key) to every non-overlapping
    rectangular frame. |X| is untouched, the output is exactly real, and the
    operation is exactly invertible by anyone who holds the key."""
    n_fr = int(np.ceil(len(x) / n))
    xp = np.zeros(n_fr * n)
    xp[:len(x)] = x
    K = odd_random_phase(n, n_fr, np.random.default_rng(key)).T
    s = np.fft.ifft(np.fft.fft(xp.reshape(n_fr, n), axis=1) * np.exp(1j * K), axis=1)
    return s.real.reshape(-1)


def otp_descramble(s: np.ndarray, n: int, key: int, length: int) -> np.ndarray:
    if len(s) % n:
        raise ValueError(f"keyed signal must have a whole number of {n}-sample frames, got {len(s)} samples")
    n_fr = len(s) // n
    K = odd_random_phase(n, n_fr, np.random.default_rng(key)).T
    r = np.fft.ifft(np.fft.fft(s.reshape(n_fr, n), axis=1) * np.exp(-1j * K), axis=1)
    return r.real.reshape(-1)[:length]


# ---------------------------------------------------------------------- metrics
def spectral_convergence(m_est: np.ndarray, m_ref: np.ndarray) -> float:
    """|| |Y| - |X| ||_F / || |X| ||_F   (0 = identical magnitudes)."""
    return float(np.linalg.norm(m_est - m_ref) / np.linalg.norm(m_ref))


def log_spectral_distance(m_est: np.ndarray, m_ref: np.ndarray, floor_db: float = -80.0) -> float:
    """Mean over frames of RMS difference of log power spectra, in dB."""
    ref_max = np.max(m_ref)
    a = 20 * np.log10(np.maximum(m_ref, ref_max * 10 ** (floor_db / 20)))
    b = 20 * np.log10(np.maximum(m_est, ref_max * 10 ** (floor_db / 20)))
    return float(np.mean(np.sqrt(np.mean((a - b) ** 2, axis=0))))


def snr_db(ref: np.ndarray, est: np.ndarray, align_scale: bool = True) -> float:
    """Waveform SNR. With align_scale, `est` is first optimally scaled (allows sign flip)."""
    ref = np.real(ref)
    est = np.real(est)
    if align_scale:
        g = np.dot(ref, est) / max(np.dot(est, est), 1e-20)
        est = g * est
    return float(10 * np.log10(np.sum(ref ** 2) / max(np.sum((ref - est) ** 2), 1e-20)))


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.real(a) - np.mean(np.real(a))
    b = np.real(b) - np.mean(np.real(b))
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-20))


def stoi(ref: np.ndarray, est: np.ndarray, fs: int = FS, extended: bool = False) -> float:
    return float(_pystoi(np.real(ref), np.real(est), fs, extended=extended))


def stoi_per_band(ref: np.ndarray, est: np.ndarray, fs: int = FS) -> tuple[np.ndarray, np.ndarray]:
    """STOI split into its 15 third-octave bands (same steps as pystoi; the mean over
    bands equals the STOI score). Returns (band centre frequencies, per-band score)."""
    from pystoi import utils as U
    import importlib
    S = importlib.import_module("pystoi.stoi")
    x, y = np.real(ref).astype(np.float64), np.real(est).astype(np.float64)
    if fs != S.FS:
        x, y = U.resample_oct(x, S.FS, fs), U.resample_oct(y, S.FS, fs)
    x, y = U.remove_silent_frames(x, y, S.DYN_RANGE, S.N_FRAME, S.N_FRAME // 2)
    xs = np.sqrt(S.OBM @ np.abs(U.stft(x, S.N_FRAME, S.NFFT, overlap=2).T) ** 2)
    ys = np.sqrt(S.OBM @ np.abs(U.stft(y, S.N_FRAME, S.NFFT, overlap=2).T) ** 2)
    xseg = np.array([xs[:, m - S.N:m] for m in range(S.N, xs.shape[1] + 1)])
    yseg = np.array([ys[:, m - S.N:m] for m in range(S.N, xs.shape[1] + 1)])
    yseg = yseg * np.linalg.norm(xseg, axis=2, keepdims=True) / (np.linalg.norm(yseg, axis=2, keepdims=True) + U.EPS)
    yseg = np.minimum(yseg, xseg * (1 + 10 ** (-S.BETA / 20)))
    yseg = yseg - yseg.mean(axis=2, keepdims=True)
    xseg = xseg - xseg.mean(axis=2, keepdims=True)
    yseg /= np.linalg.norm(yseg, axis=2, keepdims=True) + U.EPS
    xseg /= np.linalg.norm(xseg, axis=2, keepdims=True) + U.EPS
    return np.asarray(S.CF), np.sum(xseg * yseg, axis=2).mean(axis=0)


def describe(ref: np.ndarray, est: np.ndarray, n_fft: int = N_FFT, hop: int = HOP) -> dict:
    """Comparison numbers used in the report tables."""
    est_r = np.real(est)
    M_ref = np.abs(stft(ref, n_fft, hop))
    M_est = np.abs(stft(est_r, n_fft, hop))
    g = np.linalg.norm(M_ref) / max(np.linalg.norm(M_est), 1e-20)   # gain-normalised SC
    return {
        "energy_ratio_dB": 10 * np.log10(np.sum(est_r ** 2) / np.sum(np.real(ref) ** 2)),
        "corr": corr(ref, est_r),
        "SNR_dB": snr_db(ref, est_r),
        "SC": spectral_convergence(g * M_est, M_ref),
        "LSD_dB": log_spectral_distance(g * M_est, M_ref),
        "STOI": stoi(ref, est_r),
        "ESTOI": stoi(ref, est_r, extended=True),
    }


# ----------------------------------------------------------------------- noise
def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr: float) -> tuple[np.ndarray, np.ndarray]:
    """Scale `noise` so that 10 log10(P_clean / P_noise) = snr. Returns (noisy, scaled_noise)."""
    if len(noise) < len(clean):
        raise ValueError(f"noise is shorter than the speech ({len(noise)} < {len(clean)} samples)")
    noise = noise[: len(clean)]
    p_s = np.mean(clean ** 2)
    p_n = np.mean(noise ** 2)
    if p_n == 0:
        raise ValueError("noise segment is silent")
    k = np.sqrt(p_s / (p_n * 10 ** (snr / 10)))
    return clean + k * noise, k * noise


def pick_noise_segment(noise: np.ndarray, n: int, rng: np.random.Generator,
                       min_activity: float = 0.0) -> np.ndarray:
    """Random segment of length n. For bursty noise (machine-gun) segments are redrawn until the share of
    active frames (within 30 dB of the loudest frame of the whole recording) reaches min_activity."""
    if len(noise) < n:
        raise ValueError(f"noise is shorter than the requested segment ({len(noise)} < {n} samples)")
    if len(noise) == n:
        return noise.copy()
    frame_db = lambda v: 10 * np.log10(np.mean(sliding_window_view(v, 320)[::160] ** 2, axis=1) + 1e-20)
    threshold = frame_db(noise).max() - 30
    for _ in range(200):
        start = int(rng.integers(0, len(noise) - n))
        seg = noise[start:start + n]
        if np.mean(frame_db(seg) > threshold) >= min_activity:
            return seg.copy()
    return seg.copy()


# --------------------------------------------------------------------- filters
def butter_bandpass(lo: float, hi: float, order: int = 4, fs: int = FS):
    return signal.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")


def apply_bandpass(x: np.ndarray, lo: float, hi: float, kind: str = "butter_zero_phase",
                   order: int = 4, numtaps: int = 511, fs: int = FS) -> np.ndarray:
    """Band-pass x.

    butter_zero_phase : Butterworth, forward-backward (sosfiltfilt) -> zero phase,
                        effective order 2*order, magnitude response squared.
    butter_causal     : same Butterworth applied once (non-linear phase, group delay).
    fir_linear_phase  : windowed-sinc FIR (Hamming), delay (numtaps-1)/2 compensated.
    brickwall         : ideal FFT-domain mask (gain exactly 0 outside the band).
    """
    if kind == "butter_zero_phase":
        return signal.sosfiltfilt(butter_bandpass(lo, hi, order, fs), x)
    if kind == "butter_causal":
        return signal.sosfilt(butter_bandpass(lo, hi, order, fs), x)
    if kind == "butter_mag_only":
        # same |H| as the causal single pass, but zero phase (applied in the FFT domain)
        f = np.fft.rfftfreq(len(x), 1 / fs)
        _, h = signal.sosfreqz(butter_bandpass(lo, hi, order, fs), worN=f, fs=fs)
        return np.fft.irfft(np.fft.rfft(x) * np.abs(h), n=len(x))
    if kind == "fir_linear_phase":
        h = signal.firwin(numtaps, [lo, hi], pass_zero=False, fs=fs)
        y = signal.fftconvolve(x, h)[(numtaps - 1) // 2:]
        return y[: len(x)]
    if kind == "brickwall":
        Xf = np.fft.rfft(x)
        f = np.fft.rfftfreq(len(x), 1 / fs)
        Xf[(f < lo) | (f > hi)] = 0
        return np.fft.irfft(Xf, n=len(x))
    raise ValueError(kind)


def third_octave_band_snr(clean: np.ndarray, noise: np.ndarray, fs: int = FS) -> tuple[np.ndarray, np.ndarray]:
    """Long-term SNR in the 15 one-third-octave bands STOI uses (centres 150 Hz .. 3.8 kHz)."""
    cf = 150 * 2 ** (np.arange(15) / 3)
    # Welch PSD with a Hann window: a single full-length FFT (rectangular window) leaks the edge
    # discontinuity of the noise into strongly attenuated stop-band bands and fakes an SNR change there.
    f, S = signal.welch(clean, fs, window="hann", nperseg=4096)
    _, Nn = signal.welch(noise, fs, window="hann", nperseg=4096)
    out = []
    for c in cf:
        b = (f >= c * 2 ** (-1 / 6)) & (f < c * 2 ** (1 / 6))
        out.append(10 * np.log10(S[b].sum() / max(Nn[b].sum(), 1e-20)))
    return cf, np.array(out)
