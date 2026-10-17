"""
recover.py - try to recover speech that was scrambled with an odd random phase.

Three ways, matching R1/R2/R3 in the report:

  # R1 blind: you only have the scrambled audio (and know the frame settings)
  python recover.py outputs/challenge/clean__scrambled.wav --mode blind -o rec_blind.wav

  # R2 side information: you were also sent the true STFT magnitude (here: computed
  #    from a reference wav, 32 ms Hann, 75 % overlap) and run Griffin-Lim on it
  python recover.py outputs/challenge/clean__scrambled.wav --mode magnitude \
         --mag-from data/speech/original.wav -o rec_mag.wav

  # R3 keyed: the file was made with the phase one-time pad and you hold the key
  python recover.py outputs/challenge/clean__keyed.wav --mode key --key 2026 -o rec_key.wav

Add --ref data/speech/original.wav to print STOI / ESTOI of the result.
"""
import argparse

import numpy as np

import phaselab as pl


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav")
    ap.add_argument("--mode", choices=["blind", "magnitude", "key"], required=True)
    ap.add_argument("-o", "--out", default="recovered.wav")
    ap.add_argument("--n_fft", type=int, default=16384, help="scrambler frame length (blind/key)")
    ap.add_argument("--hop", type=int, default=4096, help="scrambler hop (blind)")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--key", type=int, help="integer key for --mode key")
    ap.add_argument("--length", type=int, help="original length in samples (key mode; default: from --ref)")
    ap.add_argument("--mag-from", help="wav whose 32 ms STFT magnitude is used (magnitude mode)")
    ap.add_argument("--ref", help="clean reference wav for STOI/ESTOI")
    a = ap.parse_args()

    s = pl.load_wav(a.wav)
    ref = pl.load_wav(a.ref) if a.ref else None
    if a.mode == "blind":
        L = len(s)
        y = pl.griffin_lim(np.abs(pl.stft(s, a.n_fft, a.hop)), L, a.hop, n_iter=a.iters)
    elif a.mode == "magnitude":
        m = pl.load_wav(a.mag_from)
        y = pl.griffin_lim(np.abs(pl.stft(m)), len(m), pl.HOP, n_iter=a.iters)
    else:
        if a.key is None:
            ap.error("--mode key needs --key")
        L = a.length or (len(ref) if ref is not None else len(s))
        y = pl.otp_descramble(s, a.n_fft, a.key, L)
    pl.save_wav(a.out, y)
    print(f"wrote {a.out}")
    if ref is not None:
        n = min(len(ref), len(y))
        print(f"STOI  {pl.stoi(ref[:n], y[:n]):.3f}")
        print(f"ESTOI {pl.stoi(ref[:n], y[:n], extended=True):.3f}")


if __name__ == "__main__":
    main()
