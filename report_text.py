"""Text of the report. All numbers are read from outputs/ so the report always matches the last run."""
from __future__ import annotations

import json

import pandas as pd
from reportlab.lib.units import cm

from make_report import OUT, TAB, W

TRANSCRIPT = ("He has grave doubts whether Sir Frederick Leighton's work is really Greek after all and can "
              "discover in it but little of rocky Ithaca.")


def num(v, d=3, sign=False):
    s = f"{v:+.{d}f}" if sign else f"{v:.{d}f}"
    return s.replace("-", "−")


def sci(v):
    m, e = f"{v:.1e}".split("e")
    return f"{m}×10<super>{int(e)}</super>".replace("-", "−")


def nm(noise):
    return "machine-gun" if noise == "machinegun" else noise


def load():
    R = json.loads((OUT / "results.json").read_text())
    rd = lambda f, **k: pd.read_csv(TAB / f, **k)
    T = {"p3": rd("p3_metrics_clean.csv", index_col=0), "ws": rd("p3_waveform_stats_clean.csv", index_col=0),
         "p4": rd("p4_realness.csv"), "sw": rd("p5_window_sweep.csv"), "st": rd("p6_STOI_wide.csv"),
         "es": rd("p6_ESTOI_wide.csv"), "long": rd("p6_all_variants_long.csv"), "bpf": rd("p6_bpf.csv"),
         "diag": rd("p6_bpf_design_diagnostics.csv"), "rn": rd("p6_realness_noisy.csv"),
         "pb": rd("p6_stoi_per_band.csv"), "bs": rd("p6_band_snr.csv"), "tests": rd("test_results.csv")}
    cov = TAB / "coverage.json"
    T["coverage"] = json.loads(cov.read_text())["files"] if cov.exists() else {}
    return R, T


# ------------------------------------------------------------------------------------------------ front
def title_page(D, m):
    D.space(1.5)
    D.center(f"<b>{m['institution'].upper()}</b>")
    D.center(m["department"])
    D.space(3.0)
    D.center("<i>Assignment Report</i>")
    D.space(0.6)
    D.center("Magnitude, Phase and Intelligibility of Speech", "title")
    D.center("Zero-phase and random-phase reconstruction, phase scrambling, and STOI of speech in "
             "NOISEX-92 noise", "subtitle")
    D.space(3.2)
    D.center("Submitted by")
    D.center(f"<b>{m['name']}</b>")
    D.center(f"USN: {m['usn']}")
    D.space(2.8)
    D.center(m["date"])
    D.brk()


def abstract_intro(D, R, T, m):
    p3, p5 = T["p3"], R["part5"]["clean"]
    s1, s2 = p3.loc["sig1 (zero phase)"], p3.loc["sig2 odd-symmetric PH1"]
    D.h1("Abstract")
    D.p(f"This report looks at what the magnitude and the phase of the short-time Fourier transform (STFT) each "
        f"contribute to a speech signal. A 9.9 s clean utterance from LibriSpeech was analysed with a 32 ms Hann "
        f"window and an 8 ms hop. Rebuilding the signal from the magnitude with zero phase gave a buzzy, robotic "
        f"voice, and rebuilding it with a random phase gave a rough, noisy one, yet both stayed largely intelligible "
        f"(STOI {num(s1['STOI'])} and {num(s2['STOI'])}). A real signal is guaranteed when the phase is "
        f"odd-symmetric in every frame, and infinitely many such phases exist. Random phase made the speech "
        f"unintelligible only when "
        f"the frame was made about one second long (ESTOI {num(p5['scrambled (replace phase)']['ESTOI'])}), and such "
        f"speech could be recovered only when the magnitude or the random phase itself was kept. The experiments were "
        f"repeated with NOISEX-92 white, babble and machine-gun noise at −2 to 10 dB SNR. A band-pass filter "
        f"did not improve STOI in any of the fifteen noisy conditions.")
    D.h1("1. Introduction")
    D.p("Most speech processing is done on the STFT, where every time-frequency point has a magnitude and a phase. "
        "Classical enhancement methods change only the magnitude and keep the noisy phase, on the assumption that "
        "the ear is not very sensitive to phase. Oppenheim and Lim [2] showed that this is not true in general, and "
        "listening tests [3], [4] found that the answer depends on the length of the analysis window.")
    D.p("The assignment tests these ideas on a real recording. The signal is rebuilt from its magnitude alone and from "
        "a random phase, the condition for a modified spectrum to give a real signal is worked out, the speech is made "
        "unintelligible by scrambling its phase and then recovered, and intelligibility is measured with STOI in clean "
        "and noisy conditions, with and without a band-pass filter. Section 2 describes the data and the choices made, "
        "Section 3 answers the questions in order, Section 4 describes the code and its tests, and Section 5 gives "
        "the conclusions. The code, the executed notebook and all results are in the GitHub repository "
        f"{m['repo'].removeprefix('https://')}.")


# ------------------------------------------------------------------------------------------------ method
def method(D, R, T):
    d = R["data"]
    D.h1("2. Data and Method")
    D.h2("2.1 Speech and noise")
    D.p(f"The recording is utterance {d['speech_id']} from the LibriSpeech dev-clean set [10], read by a male "
        f"speaker: “{TRANSCRIPT}” It lasts {d['duration_s']:.1f} s and is stored losslessly at 16 kHz "
        f"with 16-bit samples. The pauses are about {d['speech_dynamic_range_dB']:.0f} dB below the speech, so there "
        f"is very little background noise. A public studio recording was used instead of a phone recording for two "
        f"reasons: STOI needs a clean reference, and anyone can download the same file and repeat the work.")
    D.p("White, babble and machine-gun noise were taken from the NOISEX-92 folder of the speechdnn/Noises repository "
        "[11] and resampled to 16 kHz. The machine-gun file was recorded at 8 kHz, so it has no energy above 4 kHz. "
        "For each noise one 9.9 s segment was picked with a fixed random seed and used at all five SNRs, so that the "
        "SNR conditions differ only in noise level. The SNR is measured over the whole utterance.")
    D.h2("2.2 Analysis settings")
    D.p(f"Table {D.ref('t_par')} lists the settings and the reason for each. Two of them need a short explanation. "
        "First, the FFT keeps all N bins, a two-sided spectrum. Library STFTs based on the real FFT keep only the "
        "positive frequencies, and their inverse always returns a real signal; that would hide the question asked in "
        "Part 3. Second, the inverse STFT is the least-squares method of Griffin and Lim [1]:")
    D.eq("y[n] = Σ<sub>m</sub> w[n − mH] y<sub>m</sub>[n − mH]  /  "
         "Σ<sub>m</sub> w<super>2</super>[n − mH],   y<sub>m</sub> = IDFT of frame m", 1)
    D.p("For an unchanged STFT it returns the original signal exactly (the error is 2×10<super>−16</super>). "
        "For a modified STFT it returns the signal whose STFT is closest to the modified one, and its output is "
        "complex if the modified spectrum is not conjugate-symmetric.")
    rows = [
        ["Window", "Hann, N = 512 (32 ms)", "speech is nearly stationary over 20 to 40 ms; bins 31.25 Hz apart "
         "separate harmonics 116 Hz apart; low side lobes"],
        ["Hop", "H = 128 (8 ms), 75% overlap", "the squared windows add to a constant, so the inverse is stable; the "
         "fourfold redundancy is also what makes phase recovery possible in Part 5"],
        ["FFT", "two-sided, N = 512", "keeps the symmetry studied in Parts 3 and 4 visible"],
        ["Inverse STFT", "least-squares [1]", "correct inverse for a modified STFT"],
        ["Random phase", "uniform on [−π, π), fixed seed", "no preferred phase; repeatable"],
        ["Scrambler frame", "1.024 s, 75% overlap", "chosen from the sweep in Section 3.5"],
        ["Phase recovery", "fast Griffin–Lim [7], 200 iterations", "standard method; settled within 200 "
         "iterations in every run"],
        ["Band-pass filter", "Butterworth (order 4 per edge), forward-backward", "flat pass band; no phase "
         "distortion or delay"],
        ["Intelligibility", "STOI [8] and ESTOI [9]", "STOI is required; ESTOI copes better with fluctuating noise "
         "and falls to about 0 for unintelligible speech"],
    ]
    D.table("t_par", pd.DataFrame(rows, columns=["Setting", "Value", "Reason"]),
            "Analysis settings and the reason for each choice.",
            col_widths=[3.0 * cm, 5.0 * cm, W - 8.0 * cm], index=False)
    D.h2("2.3 Measures")
    D.p("STOI compares the band envelopes of a processed signal with those of the clean signal in 15 third-octave "
        "bands between 150 Hz and about 4 kHz, over 384 ms segments [8]. It is close to 1 for intact speech, but "
        "because of its normalisation an unrelated speech-like sound still scores around 0.35 to 0.4; ESTOI [9] does "
        "not have this floor. The clean recording is always the reference. To describe changes that STOI does not "
        "see, the waveform correlation r with the original and the spectral convergence SC (the relative error "
        "between the STFT magnitude of the result and MG) are also given.")


# ------------------------------------------------------------------------------------------------ results
def part2(D, R, T):
    D.h1("3. Results and Discussion")
    D.h2("3.1 Recording, STFT magnitude and phase (Parts 1 and 2)")
    D.q("Use a voice recording of about 10 s with minimal background noise. Take its STFT and plot the magnitude "
        "(MG) and the phase (PH) separately.")
    D.fig("mg", "p2_MG_clean.png", "Waveform and STFT magnitude MG of the recording (dB relative to the maximum).",
          width=0.92)
    D.fig("ph", "p2_PH_clean.png", "STFT phase PH, wrapped to (−π, π]. The colour scale is cyclic, "
          "so −π and π have the same colour.", width=0.92)
    D.p(f"MG (Figure {D.ref('mg')}) shows the speech clearly: harmonics as thin horizontal lines, formants as darker "
        f"bands, fricatives above 4 kHz, and the pauses between words. PH (Figure {D.ref('ph')}) looks like random "
        "noise. The phase is not actually random: for a steady tone it advances by 2πkH/N = πk/2 radians "
        "per hop, so every bin wraps around ±π at a different rate and the plot shows mostly the wrapping. "
        "Its rate of change over time, the instantaneous frequency, does follow the harmonics (see the notebook). "
        "In every frame MG is symmetric and PH antisymmetric about bin N/2, as expected for a real signal. This is "
        "used in Part 4.")


def part3(D, R, T):
    p3, ws, re = T["p3"], T["ws"], R["part3"]["realness"]
    s1 = p3.loc["sig1 (zero phase)"]
    s1c = p3.loc["sig1c (zero phase about window centre)"]
    s2u = p3.loc["sig2 unconstrained PH1: real part"]
    s2 = p3.loc["sig2 odd-symmetric PH1"]
    w = lambda r, c: ws.loc[r, c]

    D.h2("3.2 Magnitude only, zero phase: sig1 (Part 3)")
    D.q("Using only the magnitude (zero phase), do the inverse transform and call the result sig1. How does it "
        "differ from the original, and why?")
    D.p(f"sig1 was computed with Eq. (1) from MG alone, with every phase set to zero. It is real, it follows the "
        f"spectral envelope of the original, and it is still mostly intelligible (STOI {num(s1['STOI'])}, ESTOI "
        f"{num(s1['ESTOI'])}). It sounds like a buzzy, monotone, robotic voice, and its waveform has almost nothing "
        f"in common with the original (r = {num(s1['corr'])}, Table {D.ref('t_p3')}).")
    D.p(f"The reason is that with zero phase every frame turns into a short symmetric pulse at the start of the frame. "
        f"All the cosine components of the frame are in phase at n = 0 and add up there (Figure {D.ref('f0')}). "
        f"Three things follow. The timing inside each 32 ms frame, where the glottal pulses and onsets were, is "
        f"lost. The pulses repeat every hop of 8 ms, so sig1 is excited at f<sub>s</sub>/H = 125 Hz instead of at "
        f"the speaker's own varying pitch (median 116 Hz); its spectrum becomes a comb with lines at multiples of "
        f"125 Hz (Figure {D.ref('ltas')}), and the autocorrelation at a lag of 8 ms rises from "
        f"{num(w('original', 'autocorr_at_lag_H'), 2)} to {num(w('sig1', 'autocorr_at_lag_H'), 2)}. Finally, "
        f"because the pulse falls at the edge of the frame, where the Hann window is zero, most of it is removed "
        f"in the overlap-add, and sig1 is {num(-s1['energy_ratio_dB'], 1)} dB quieter than the original.")
    D.p(f"Zero phase therefore depends on where the time origin of a frame is taken. Taking it at the centre of the "
        f"window (sig1c) puts the pulse at the peak of the window: the loss drops to {num(-s1c['energy_ratio_dB'], 1)} "
        f"dB and STOI rises to {num(s1c['STOI'])}, but the 125 Hz buzz remains.")
    D.fig("f0", "p3_frame_zero_phase.png", "One voiced frame. (a) Original frame. (b) Inverse DFT of its magnitude "
          "with zero phase, a pulse at n = 0, and the small part that survives the synthesis window (green). "
          "(c) Zero phase about the window centre puts the pulse at n = N/2.")
    D.fig("ltas", "p3_ltas_envelope.png", "Left: long-term spectrum up to 2 kHz, grey lines every 125 Hz. sig1 is a "
          "comb at multiples of 125 Hz, sig2 follows the original. Right: frame energy; both keep the syllable "
          "rhythm of the original.")

    D.h2("3.3 Random phase PH1: sig2 (Part 3)")
    D.q("Create an array PH1 of the same size as PH and use MG and PH1 to obtain sig2. Is a real-valued sig2 "
        "possible? How does sig2 differ from the original, and from sig1?")
    D.h3("Is a real sig2 possible?")
    D.p(f"When PH1 is filled with independent random values for all 512 × 1241 entries, sig2 comes out complex, "
        f"with {100 * re['sig2_unconstrained_imag_energy_ratio']:.1f}% of its energy in the imaginary part. A signal "
        f"is real only if its spectrum satisfies X[N − k] = X*[k]. MG already meets the magnitude part of this "
        f"condition, so PH1 must satisfy PH1[N − k] = −PH1[k], and an unrestricted random array almost never "
        f"does. A real sig2 is therefore possible: it is guaranteed when PH1 has this symmetry in every frame "
        f"(Section 3.4). With such a "
        f"PH1 the largest imaginary value is {sci(re['sig2_odd_max_abs_imag'])}, which is rounding error. Taking "
        f"the real part of the complex sig2 also gives a real signal, but it scales the magnitude by a random factor "
        f"and loses another {num(s2['energy_ratio_dB'] - s2u['energy_ratio_dB'], 1)} dB "
        f"({num(s2u['energy_ratio_dB'], 1)} dB against {num(s2['energy_ratio_dB'], 1)} dB). "
        f"Below, sig2 means the real signal obtained with a symmetric random PH1.")
    D.h3("sig2 compared with the original")
    D.p(f"Inside each frame the harmonics now add with random phases, so each frame becomes a burst of noise with the "
        f"right spectral shape. sig2 sounds rough and hoarse, and its waveform is "
        f"uncorrelated with the original (r = {num(s2['corr'])}). It is {num(-s2['energy_ratio_dB'], 1)} dB weaker. "
        f"This agrees with a simple argument: a random-phase spectrogram is not the STFT of any signal, and the "
        f"inverse STFT keeps only the part that is consistent, which for a random phase is about H/N = 1/4 of the "
        f"energy (−6.0 dB). The rest is lost, so the harmonics are blurred and onsets are spread over up to one frame "
        f"(SC = {num(s2['SC'])}). STOI is still {num(s2['STOI'])}, because at 32 ms the magnitude carries the formant "
        f"movements and the syllable rhythm on which intelligibility depends [3], [4].")
    D.h3("sig1 compared with sig2")
    D.p(f"The two signals have exactly the same magnitude and differ only in phase. Zero phase is the same in every "
        f"frame, so the frames line up and sig1 is a periodic pulse train (kurtosis {num(w('sig1', 'kurtosis'), 1)}, "
        f"autocorrelation at 8 ms {num(w('sig1', 'autocorr_at_lag_H'), 2)}). Random phase changes from frame to "
        f"frame, so sig2 is noise-like (kurtosis {num(w('sig2', 'kurtosis'), 1)}, autocorrelation "
        f"{num(w('sig2', 'autocorr_at_lag_H'), 2)}). sig1 has an artificial 125 Hz comb in its spectrum, sig2 a smooth "
        f"but blurred one. Their STOI values are close ({num(s1['STOI'])} and {num(s2['STOI'])}), yet they sound "
        f"completely different: one is buzzy and robotic, the other rough and noisy.")
    tab = pd.DataFrame({
        "Energy (dB)": [0.0, s1["energy_ratio_dB"], s1c["energy_ratio_dB"], s2u["energy_ratio_dB"],
                        s2["energy_ratio_dB"]],
        "r": [1.0, s1["corr"], s1c["corr"], s2u["corr"], s2["corr"]],
        "SC": [0.0, s1["SC"], s1c["SC"], s2u["SC"], s2["SC"]],
        "Kurtosis": [w("original", "kurtosis"), w("sig1", "kurtosis"), w("sig1c", "kurtosis"), "–",
                     w("sig2", "kurtosis")],
        "Autocorr. at 8 ms": [w("original", "autocorr_at_lag_H"), w("sig1", "autocorr_at_lag_H"),
                              w("sig1c", "autocorr_at_lag_H"), "–", w("sig2", "autocorr_at_lag_H")],
        "STOI": [1.0, s1["STOI"], s1c["STOI"], s2u["STOI"], s2["STOI"]],
        "ESTOI": [1.0, s1["ESTOI"], s1c["ESTOI"], s2u["ESTOI"], s2["ESTOI"]],
    }, index=["Original", "sig1 (zero phase)", "sig1c (zero phase, centred)", "Re{sig2}, unrestricted PH1",
              "sig2, symmetric PH1"])
    tab.index.name = "Signal"
    D.table("t_p3", tab, "sig1 and sig2 compared with the original recording.",
            col_widths=[4.6 * cm] + [(W - 4.6 * cm) / 7] * 7, fmt="{:.3f}")


def part4(D, R, T):
    p4, c = T["p4"], R["part4"]["count"]
    D.h2("3.4 Phase pattern that gives a real sig2 (Part 4)")
    D.q("Is there a pattern in PH1 that ensures a real-valued sig2? How many such patterns are possible?")
    D.p("Yes. sig2 is guaranteed to be real if, in every frame m, the phase is odd about N/2 and the DC and "
        "Nyquist bins are real:")
    D.eq("PH1(m, N − k) = −PH1(m, k)  (mod 2π),   k = 1, ..., N/2 − 1", 2)
    D.eq("PH1(m, 0) and PH1(m, N/2) ∈ {0, π}", 3)
    D.p(f"Wherever MG is non-zero, these conditions are necessary and sufficient for the inverse DFT of each frame "
        f"to be real, and the overlap-add of real frames is then real. Table {D.ref('t_p4')} checks them "
        f"numerically. Zero "
        f"phase, a linear phase (a pure delay) and symmetric random phases all give imaginary parts at the "
        f"10<super>−16</super> level. A symmetric phase with arbitrary DC and Nyquist values, a symmetric phase "
        f"plus a constant π/4, an even phase and an unrestricted phase all give complex signals.")
    tab = p4.rename(columns={"imag/total energy (mean)": "Imaginary energy / total", "max |Im{sig2}|": "Max |Im|",
                             "real?": "Real?", "trials": "Trials"}).set_index("PH1 pattern")
    D.table("t_p4", tab, "Is sig2 real? Random patterns were tried 20 times each.",
            col_widths=[6.4 * cm, 1.4 * cm, 3.4 * cm, 2.6 * cm, W - 13.8 * cm], fmt="{:.3g}")
    D.p(f"As for the number of patterns: in each frame the {c['free_phases_per_frame']} phases of bins 1 to "
        f"N/2 − 1 can take any value, the other half is then fixed by Eq. (2), and the DC and Nyquist bins have "
        f"two choices each. Since the phases are continuous, the number of valid patterns is infinite. Even if each "
        f"phase could only be 0 or π there would be 4·2<super>255</super> ≈ "
        f"10<super>{c['log10_patterns_1bit']['per_frame']:.0f}</super> patterns per frame, and about "
        f"10<super>{c['log10_patterns_1bit']['whole_signal']:,.0f}</super> for the {c['frames_M']} frames of this "
        f"recording. Familiar examples are zero phase, a linear phase, the original phase PH, and the phase of the DFT "
        f"of any real sequence. Different patterns can still give the same sig2, because the inverse STFT discards "
        f"part of the spectrogram.")


def part5(D, R, T):
    sw, p5 = T["sw"].set_index("N"), R["part5"]["clean"]
    s2 = T["p3"].loc["sig2 odd-symmetric PH1"]
    D.h2("3.5 Unintelligible speech and its recovery (Part 5)")
    D.q("Create an unintelligible version of every speech signal using an odd but random phase, and ask others to "
        "recover the original.")
    D.p(f"At 32 ms, random phase does not make speech unintelligible (STOI {num(s2['STOI'])} above). Figure "
        f"{D.ref('sweep')} shows what happens as the frame gets longer, with the hop kept at a quarter of the frame. "
        f"STOI falls from {num(sw.loc[256, 'STOI scrambled'])} at 16 ms to {num(sw.loc[32768, 'STOI scrambled'])} "
        f"at 2 s, and ESTOI from {num(sw.loc[256, 'ESTOI scrambled'])} to {num(sw.loc[32768, 'ESTOI scrambled'])}. "
        f"Most of the drop happens between about 128 and 512 ms, roughly the length of a syllable: once a frame "
        f"covers several syllables, random phase smears them together. For long frames the magnitude alone no longer "
        f"carries the intelligibility; earlier studies found that the phase then does [2], [4]. A frame of 1.024 s "
        f"was chosen for the "
        f"scrambler. It gives STOI {num(p5['scrambled (replace phase)']['STOI'])} and ESTOI "
        f"{num(p5['scrambled (replace phase)']['ESTOI'])}, which is unintelligible. Scrambled versions were made for "
        f"the clean recording and for all fifteen noisy signals.")
    D.fig("sweep", "p5_window_sweep.png", "Effect of frame length. Blue: random-phase version. Orange: blind "
          "Griffin–Lim recovery from it, which lies on the blue curve. Green: Griffin–Lim given the true "
          "magnitude.")
    D.p(f"Recovering the original was tried in three ways (Table {D.ref('t_rec')}, Figure {D.ref('rec')}).")
    D.p(f"<i>Blind recovery.</i> Only the scrambled file is available. Fast Griffin–Lim was run for 200 "
        f"iterations on its magnitude. It converged, but STOI stayed at {num(p5['R1 blind Griffin-Lim']['STOI'])}. "
        f"This cannot work: the scrambled signal is itself a perfect solution for its own magnitude, and the original "
        f"phase was thrown away, not hidden. Even the original magnitude can no longer be read from the scrambled "
        f"file (SC = {num(p5['_SC_scrambled_vs_true_MG'])}).")
    D.p(f"<i>Recovery with the magnitude.</i> If the true 32 ms magnitude is sent along with the file, Griffin–Lim "
        f"rebuilds the speech almost perfectly (STOI {num(p5['R2 Griffin-Lim with true MG']['STOI'])}), since a "
        f"redundant STFT magnitude in general fixes the signal up to its sign [5], [6]. The waveform is not identical "
        f"(SNR about 0 dB) because the sign of separate speech segments cannot be decided, which does not affect "
        f"STOI.")
    D.p(f"<i>Recovery with a key.</i> Instead of replacing the phase, an odd random phase K, generated from a secret "
        f"number, is added to the phase of non-overlapping 1.024 s frames. The result is real, because the sum of two "
        f"odd phases is odd, and just as unintelligible (STOI {num(p5['scrambled (keyed one-time pad)']['STOI'])}, "
        f"ESTOI {num(p5['scrambled (keyed one-time pad)']['ESTOI'])}). Whoever has the key subtracts K and gets the "
        f"original back exactly (STOI {num(p5['R3 keyed descramble']['STOI'])}). The frames must not overlap, "
        f"otherwise neighbouring keys mix and the recovery is only approximate.")
    D.p("Phase scrambling is therefore reversible only if extra information is kept: the magnitude or the random "
        "phase itself. The scrambled files of all sixteen signals and a script for the three recovery methods "
        "(recover.py) are in the repository, so others can try to recover the speech.")
    rows = {k: v for k, v in p5.items() if not k.startswith("_")}
    tab = pd.DataFrame(rows).T[["STOI", "ESTOI"]]
    tab.index = ["Scrambled (phase replaced)", "Blind Griffin–Lim", "Griffin–Lim with true magnitude",
                 "Scrambled with key", "Descrambled with key"]
    tab.index.name = "Signal"
    D.table("t_rec", tab, "Scrambling and recovery of the clean recording.",
            col_widths=[7.0 * cm, 3.0 * cm, 3.0 * cm])
    D.fig("rec", "p5_recovery_clean.png", "Spectrograms of the original, the scrambled signal and the three "
          "recovery attempts.", width=0.78)


def part6(D, R, T):
    st, lg, bpf, diag, rn = T["st"], T["long"], T["bpf"], T["diag"], T["rn"]
    noisy = lg[lg.variant == "noisy"]
    D.h2("3.6 Noise, STOI and band-pass filtering (Part 6)")
    D.q("Add white, machine-gun and babble noise from NOISEX-92 at −2, 0, 2, 5 and 10 dB SNR and compute STOI. "
        "Repeat Parts 2 to 5 for each noise. Apply a band-pass filter, compare STOI for each noise and comment on the "
        "filter design and the scores.")
    D.h3("STOI of the noisy speech")
    tab = pd.DataFrame({(n, m): noisy[noisy.noise == n].set_index("SNR_dB")[m]
                        for n in ["white", "babble", "machinegun"] for m in ["STOI", "ESTOI"]})
    tab.columns = [f"{nm(n).capitalize()} {m}" for n, m in tab.columns]
    tab.index = tab.index.astype(int)
    tab.index.name = "SNR (dB)"
    D.table("t_noisy", tab, "STOI and ESTOI of the noisy speech.", col_widths=[2.2 * cm] + [(W - 2.2 * cm) / 6] * 6)
    ms = noisy.groupby("noise")["STOI"].mean()
    D.p(f"STOI rises with SNR for all three noises (Table {D.ref('t_noisy')}). At the same SNR babble is the most "
        f"harmful (mean STOI {num(ms['babble'])}), then white noise ({num(ms['white'])}), then machine-gun noise "
        f"({num(ms['machinegun'])}). Babble is itself speech, so it overlaps the target in both frequency and rhythm. "
        f"White noise covers the weak high-frequency consonants. Machine-gun noise comes in bursts with near-silent "
        f"gaps in between, where the speech is heard clearly, and it has no energy above 4 kHz.")
    D.h3("Parts 2 to 5 for the noisy signals")
    d1 = st["sig1 (zero phase)"] - st["noisy"]
    d2 = st["sig2 (odd random phase)"] - st["noisy"]
    D.fig("mgph", "p6_MG_PH_0dB_all.png", "MG and PH of the three noisy signals at 0 dB SNR (all SNRs are in the "
          "notebook).", width=0.9)
    D.p(f"The STFT plots of the noisy signals (Figure {D.ref('mgph')}) show the noise filling the pauses and the high "
        f"frequencies of MG; in those places PH is simply the phase of the noise. The answers to Parts 3 and 4 do "
        f"not change: an unrestricted PH1 gives a complex sig2 "
        f"({100 * rn['sig2_unconstrained_imag_energy_ratio'].min():.1f} to "
        f"{100 * rn['sig2_unconstrained_imag_energy_ratio'].max():.1f}% imaginary energy) and a symmetric PH1 a real "
        f"one. sig1 and sig2 lower STOI further, by {num(-d2.max())} to {num(-d1.min())} compared with the noisy "
        f"input, since changing the phase cannot separate speech from noise. Scrambling makes every noisy signal "
        f"unintelligible (STOI {num(st['scrambled (replace phase)'].min())} to "
        f"{num(st['scrambled (replace phase)'].max())}), blind recovery fails in every case, and recovery with the "
        f"magnitude or with the key gives back the noisy signal, not the clean one (Table {D.ref('t_ops')}).")
    cols = ["noisy", "sig1 (zero phase)", "sig2 (odd random phase)", "scrambled (replace phase)",
            "R1 blind Griffin-Lim", "R2 Griffin-Lim with true MG", "R3 keyed descramble"]
    t = st.set_index(["noise", "SNR_dB"])[cols].reset_index()
    t.columns = ["Noise", "SNR", "Noisy", "sig1", "sig2", "Scrambled", "Blind", "Magnitude", "Key"]
    t["Noise"] = t["Noise"].map(nm)
    t["SNR"] = t["SNR"].astype(int)
    D.table("t_ops", t.set_index("Noise"), "STOI after each operation of Parts 3 and 5 for every noisy signal. "
            "Blind, Magnitude and Key are the three recovery methods.",
            col_widths=[2.3 * cm, 1.1 * cm] + [(W - 3.4 * cm) / 7] * 7)

    D.h3("Band-pass filter: design")
    D.p("Three filters were compared. BPF-T passes 300 to 3400 Hz, the usual telephone band [12]. BPF-S passes "
        "120 to 4500 Hz; it was chosen so that all 15 STOI bands (133 Hz to 4.3 kHz) lie inside its pass band while "
        "the noise above them is still removed. Both are 4th-order Butterworth filters, chosen because their pass "
        "band is flat, and both are applied forward and backward so that there is no phase distortion or delay. An "
        "ideal brick-wall version of BPF-T and an oracle Wiener filter, which uses the true speech and noise spectra "
        "and is not a band-pass filter, were added for comparison.")
    pv = bpf.pivot_table(index=["noise", "SNR_dB"], columns="filter", values="STOI", sort=False)
    dv = bpf.pivot_table(index=["noise", "SNR_dB"], columns="filter", values="dSTOI", sort=False)
    gv = bpf.pivot_table(index=["noise", "SNR_dB"], columns="filter", values="SNR gain (dB)", sort=False)
    t = pd.DataFrame(index=pv.index)
    t["Noisy"] = st.set_index(["noise", "SNR_dB"])["noisy"]
    for f, s in [("BPF-T", "BPF-T"), ("BPF-S", "BPF-S"), ("BPF-T brick-wall", "Brick-wall"),
                 ("oracle Wiener (reference)", "Wiener")]:
        t[s] = pv[f]
    t["SNR gain, BPF-T (dB)"] = gv["BPF-T"]
    t = t.reset_index()
    t.columns = ["Noise", "SNR"] + list(t.columns[2:])
    t["Noise"] = t["Noise"].map(nm)
    t["SNR"] = t["SNR"].astype(int)
    D.table("t_bpf", t.set_index("Noise"), "STOI with and without band-pass filtering, and the change in overall SNR "
            "caused by BPF-T.", col_widths=[2.3 * cm, 1.1 * cm] + [(W - 3.4 * cm) / 6] * 6)
    D.h3("Band-pass filter: results")
    D.p(f"No band-pass filter improved STOI for any noise or SNR (Table {D.ref('t_bpf')}). On average BPF-T lowered "
        f"STOI by {num(-dv['BPF-T'].mean())}, BPF-S by {num(-dv['BPF-S'].mean())} and the brick-wall filter by "
        f"{num(-dv['BPF-T brick-wall'].mean())}, even though BPF-T raised the overall SNR by up to "
        f"{num(gv['BPF-T'].max(), 1)} dB for machine-gun noise and {num(gv.xs('white')['BPF-T'].mean(), 1)} dB for "
        f"white noise. For babble the overall SNR even fell ({num(gv.xs('babble')['BPF-T'].mean(), 1)} dB), because "
        f"babble has the same spectrum as speech. The Wiener filter, in contrast, raised STOI by "
        f"{num(dv['oracle Wiener (reference)'].min())} to {num(dv['oracle Wiener (reference)'].max())}.")
    bs = T["bs"].copy()
    bs["d"] = (bs["SNR_after_BPF_T_dB"] - bs["SNR_before_dB"]).abs()
    pas = bs[(bs.band_Hz > 400) & (bs.band_Hz < 3000)]
    pb = T["pb"]
    c = pb[(pb.signal == "clean") & (pb["filter"] == "BPF-T 300-3400")].set_index("band_Hz")["score"]
    g = diag.set_index("design")["STOI(clean, BPF(clean))"]
    D.p(f"The reason is that a fixed filter multiplies speech and noise at each frequency by the same gain. Inside "
        f"its pass band the SNR of each STOI band does not change (largest change {num(pas['d'].max(), 2)} dB), and "
        f"STOI's normalisation cancels the gain itself, so the per-band scores in the pass band are identical with "
        f"and without the filter (Figure {D.ref('bstoi')}). The only effect on STOI comes from the bands the filter "
        f"removes. Once a band is attenuated by tens of decibels, STOI measures leakage from neighbouring bands there "
        f"instead of speech, and the score of that band falls; even for clean speech the 150 Hz band drops to "
        f"{num(c.iloc[0], 2)} after BPF-T. With 15 equally weighted bands, losing three or four costs about 0.04 to "
        f"0.06, which matches the results. Further checks on clean speech show that the phase of the filter does not "
        f"matter to STOI (causal Butterworth {num(g['Butterworth-4 300-3400, causal (sosfilt)'])}, zero-phase filter "
        f"with the same magnitude {num(g['same |H| as causal, zero phase (FFT)'])}); what matters is how sharply the "
        f"bands outside the pass band are cut (forward-backward "
        f"{num(g['Butterworth-4 300-3400, zero-phase (filtfilt)'])}, 511-tap FIR "
        f"{num(g['FIR 511 taps 300-3400, linear phase'])}, brick-wall {num(g['ideal brick-wall 300-3400'])}).")
    D.fig("bstoi", "p6_stoi_per_band.png", "STOI split into its 15 bands, for clean speech and the three noises at "
          "0 dB. Inside the 300 to 3400 Hz pass band (grey) the curves with and without the filter coincide.")
    D.h3("Comment on the filter design")
    D.p("For STOI, a band-pass filter should be at least as wide as the band STOI uses, about 120 Hz to 4.5 kHz. "
        "Inside that band the type of filter and its phase hardly matter, and a zero-phase Butterworth is a safe "
        "choice. A narrower filter such as the telephone band raises the overall SNR but costs intelligibility, and "
        "the sharper its edges, the larger the loss. A fixed filter cannot remove noise that lies inside the speech "
        "band; that needs a gain that changes with time and frequency, as the Wiener result shows.")


# ------------------------------------------------------------------------------------------------ code, end
def code_and_tests(D, R, T, m):
    tests = T["tests"]
    counts = tests.groupby("file").size()
    passed = int((tests["outcome"] == "passed").sum())
    D.h1("4. Code and Testing")
    D.p("All processing was written in Python with NumPy, SciPy and pystoi. The STFT, its inverse, the phase "
        "generators, Griffin–Lim and the filters were written directly rather than taken from a library, so that "
        "each step could be checked. The repository contains phaselab.py (signal processing), experiments.py (one "
        "function per part), run_all.py (runs everything), recover.py (the recovery exercise), the executed notebook "
        f"and the tests ({m['repo'].removeprefix('https://')}).")
    rows = [
        ["STFT and inverse", counts.get("test_stft.py", 0), "exact reconstruction for every window size used; "
         "two-sided output; conjugate symmetry of a real signal; constant window overlap; complex output for a "
         "non-symmetric input"],
        ["Parts 3 and 4: phase", counts.get("test_phase.py", 0), "symmetric phase gives a real signal (5 seeds); "
         "unrestricted, wrong-DC, offset and even phases give complex signals; zero-phase frame is even; 125 Hz "
         "buzz; the 1/4 energy rule"],
        ["Part 5: recovery", counts.get("test_recovery.py", 0), "key restores the signal exactly and a wrong key "
         "does not; magnitudes are kept; scrambled speech is unintelligible; the scrambled signal is a Griffin–"
         "Lim fixed point; recovery from the true magnitude works"],
        ["Part 6: noise, STOI, filters", counts.get("test_noise_metrics_filters.py", 0), "noise added at exactly "
         "the target SNR; STOI equals 1 for identical signals, ignores gain and falls with noise; per-band STOI "
         "matches pystoi; filter gains, delays and band SNRs"],
        ["Part 1: data", counts.get("test_data.py", 0), "recording length, sampling rate and noise floor; noise "
         "files present at 16 kHz"],
        ["Files and tools", counts.get("test_io_and_tools.py", 0), "wav saving, loading and resampling; metric "
         "sanity checks; input validation; recover.py run end to end in all three modes"],
    ]
    tab = pd.DataFrame(rows, columns=["Area", "Tests", "What is checked"])
    tab["Result"] = "all passed" if passed == len(tests) else f"{passed}/{len(tests)} passed"
    verdict = "all of them pass" if passed == len(tests) else f"{passed} pass"
    cov = T["coverage"]
    pct = lambda f: cov[f]["summary"]["percent_covered"] if f in cov else float("nan")
    D.p(f"{len(tests)} automated tests (pytest) check the properties that the results depend on. Table "
        f"{D.ref('t_tests')} summarises them; {verdict}. They run {pct('phaselab.py'):.0f}% of the lines of "
        f"phaselab.py and {pct('recover.py'):.0f}% of recover.py. The code also passes the Ruff linter without "
        "warnings. Every random value uses a fixed seed: a fresh copy of the repository was run from scratch and "
        "reproduced every result table to within rounding error (largest difference 2×10<super>−14</super>).")
    D.table("t_tests", tab, "Automated tests.", col_widths=[3.6 * cm, 1.3 * cm, W - 7.0 * cm, 2.1 * cm],
            index=False)


def conclusion(D, R, T):
    p3, p5 = T["p3"], R["part5"]["clean"]
    s1, s2 = p3.loc["sig1 (zero phase)"], p3.loc["sig2 odd-symmetric PH1"]
    dv = T["bpf"].pivot_table(index=["noise", "SNR_dB"], columns="filter", values="dSTOI")
    D.h1("5. Conclusion")
    D.p("The magnitude and the phase of the STFT carry different information, and the frame length decides which one "
        "matters more. With the usual 32 ms frames the magnitude carries most of the intelligibility, while the phase "
        "carries the pitch and the fine timing that make speech sound natural. With frames of about one second the "
        "magnitude alone is no longer enough, and randomising the phase destroys the speech. A phase used for "
        "reconstruction should be odd-symmetric to guarantee a real result. Intelligibility in noise cannot be "
        "improved with a fixed "
        "band-pass filter, because the filter does not change the signal-to-noise ratio inside the bands that matter.")
    D.p("The answers to the individual parts are:")
    D.items([
        f"sig1 (zero phase) is a buzzy, monotone voice pitched at the 125 Hz frame rate; its waveform is unrelated to "
        f"the original, but STOI is {num(s1['STOI'])}.",
        f"An unrestricted random PH1 gives a complex sig2. With a symmetric PH1, sig2 is real, rough and noise-like, "
        f"about 6 dB weaker, with STOI {num(s2['STOI'])}. sig1 and sig2 differ only in phase coherence.",
        "A real sig2 is guaranteed when PH1(N − k) = −PH1(k), with the DC and Nyquist phases equal to 0 or "
        "π; "
        "infinitely many such patterns exist.",
        f"Random phase makes speech unintelligible only with frames of about 1 s (ESTOI "
        f"{num(p5['scrambled (replace phase)']['ESTOI'])}). Blind recovery is impossible; recovery works with the "
        f"true magnitude (STOI {num(p5['R2 Griffin-Lim with true MG']['STOI'])}) or exactly with a key.",
        "Babble is the most damaging noise at equal SNR and machine-gun noise the least. The phase experiments behave "
        "the same way with noise, and recovery can at best return the noisy signal.",
        f"No band-pass filter improved STOI. The telephone band lowered it by {num(-dv['BPF-T'].mean())} on average; "
        f"a 120 to 4500 Hz filter left it almost unchanged.",
    ], numbered=True)
    D.p("The study used one utterance from one speaker and objective measures only; a few more speakers and a short "
        "listening test would make the conclusions stronger.")


def references(D):
    D.h1("References")
    D.refs([
        "D. W. Griffin and J. S. Lim, “Signal estimation from modified short-time Fourier transform,” "
        "<i>IEEE Trans. ASSP</i>, vol. 32, no. 2, pp. 236–243, 1984.",
        "A. V. Oppenheim and J. S. Lim, “The importance of phase in signals,” <i>Proc. IEEE</i>, vol. 69, "
        "no. 5, pp. 529–541, 1981.",
        "K. K. Paliwal and L. D. Alsteris, “On the usefulness of STFT phase spectrum in human listening "
        "tests,” <i>Speech Communication</i>, vol. 45, no. 2, pp. 153–170, 2005.",
        "M. Kazama, S. Gotoh, M. Tohyama and T. Houtgast, “On the significance of phase in the short term Fourier "
        "spectrum for speech intelligibility,” <i>J. Acoust. Soc. Am.</i>, vol. 127, no. 3, pp. 1432–1439, "
        "2010.",
        "S. H. Nawab, T. F. Quatieri and J. S. Lim, “Signal reconstruction from short-time Fourier transform "
        "magnitude,” <i>IEEE Trans. ASSP</i>, vol. 31, no. 4, pp. 986–998, 1983.",
        "R. Balan, P. Casazza and D. Edidin, “On signal reconstruction without phase,” <i>Appl. Comput. "
        "Harmon. Anal.</i>, vol. 20, no. 3, pp. 345–356, 2006.",
        "N. Perraudin, P. Balazs and P. L. Søndergaard, “A fast Griffin–Lim algorithm,” "
        "<i>Proc. IEEE WASPAA</i>, 2013.",
        "C. H. Taal, R. C. Hendriks, R. Heusdens and J. Jensen, “An algorithm for intelligibility prediction of "
        "time–frequency weighted noisy speech,” <i>IEEE Trans. ASLP</i>, vol. 19, no. 7, "
        "pp. 2125–2136, 2011.",
        "J. Jensen and C. H. Taal, “An algorithm for predicting the intelligibility of speech masked by modulated "
        "noise maskers,” <i>IEEE/ACM Trans. ASLP</i>, vol. 24, no. 11, pp. 2009–2022, 2016.",
        "V. Panayotov, G. Chen, D. Povey and S. Khudanpur, “LibriSpeech: an ASR corpus based on public domain "
        "audio books,” <i>Proc. IEEE ICASSP</i>, 2015.",
        "A. Varga and H. J. M. Steeneken, “Assessment for automatic speech recognition II: NOISEX-92,” "
        "<i>Speech Communication</i>, vol. 12, no. 3, pp. 247–251, 1993.",
        "ITU-T Recommendation G.712, <i>Transmission performance characteristics of pulse code modulation "
        "channels</i>, 2001.",
    ])


def build(D, meta):
    R, T = load()
    title_page(D, meta)
    abstract_intro(D, R, T, meta)
    method(D, R, T)
    part2(D, R, T)
    part3(D, R, T)
    part4(D, R, T)
    part5(D, R, T)
    part6(D, R, T)
    code_and_tests(D, R, T, meta)
    conclusion(D, R, T)
    references(D)
