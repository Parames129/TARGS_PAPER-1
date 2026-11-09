"""revision_sim_v3.py - an INDEPENDENT signal generator for the dyslexia EEG pipeline.

It deliberately shares none of the signal assumptions of simulators v1/v2:
  * sources  : 68 cortical regions (Desikan-Killiany 'aparc', MNE 'sample' subject), each a
               Jansen-Rit neural-mass model (Jansen & Rit 1995) plus its own aperiodic background;
               rhythms EMERGE from the dynamics - no band, peak frequency or amplitude is imposed;
  * head     : label-level lead fields from a 3-layer boundary-element model of a real head
               (MNE 'sample' subject), not a Gaussian distance kernel;
  * effect   : 'nmm'  - increased excitatory synaptic gain and noisier input in left temporo-parietal-
               occipital populations (cortical hyperexcitability, 'neural noise' hypothesis,
               Hancock et al. 2017); its spectral consequences are whatever the model produces;
               'aperiodic' - a steeper 1/f exponent in the same regions, no oscillatory change;
  * variability: subject-level neural-mass parameters (gain, inhibitory rate, input level), regional
               jitter and background-to-rhythm ratio; no age model.
Recording nuisances (DC, mains, transient, bad channel, units, blinks, EMG, label styles, montages,
sampling rates) are shared with v2 on purpose: they test the pipeline, not the generator.
"""
import numpy as np
from scipy import signal

import revision_sim as rs

LEADFIELD = None
EFFECT_LABELS = ("supramarginal-lh", "inferiorparietal-lh", "superiortemporal-lh", "middletemporal-lh",
                 "bankssts-lh", "lateraloccipital-lh", "inferiortemporal-lh", "fusiform-lh")

SIM3 = dict(
    fs_int=500.0,                  # integration rate of the neural-mass models (Hz)
    burn_s=2.0,
    A=(3.25, 0.10), B=(22.0, 1.0), a=(100.0, 4.0), b=(51.0, 0.6), C=(135.0, 5.0),   # subject-level N(mean, sd)
    p_mean=(235.0, 12.0), p_sd=22.0,                                           # input firing rate (Hz)
    node_jitter=0.015,             # regional multiplicative jitter of A and b
    bg_chi=(0.8, 2.0),             # aperiodic exponent range per subject (power ~ f^-chi)
    bg_ratio=(0.6, 1.6),           # background SD / neural-mass SD per subject
    effect_gain=1.0,               # A *= 1 + effect * effect_gain   (excitatory gain)
    effect_noise=3.0,              # p_sd *= 1 + effect * effect_noise (input noise)
    effect_jitter=(0.5, 1.5),      # per-subject multiplier of the effect, uniform
    aperiodic_shift=1.0,           # 'aperiodic' effect: chi += effect * aperiodic_shift in effect regions
    sensor_noise_uv=2.0, scale_to_uv=15.0, gain_sd=0.25,
)


def load_leadfield(path):
    global LEADFIELD
    if LEADFIELD is None:
        d = np.load(path, allow_pickle=False)
        LEADFIELD = {k: d[k] for k in d.files}
    return LEADFIELD


def jansen_rit(A, B, a, b, C, p_mean, p_sd, fs, n_out, rng, burn_s):
    """Euler-Maruyama integration of n independent Jansen-Rit columns; returns y1 - y2 (n, n_out)."""
    n = len(A)
    dt = 1.0 / fs
    nb = int(burn_s * fs)
    N = n_out + nb
    e0, v0, r = 2.5, 6.0, 0.56
    C1, C2, C3, C4 = C, 0.8 * C, 0.25 * C, 0.25 * C
    y0 = np.zeros(n); y1 = np.zeros(n); y2 = np.zeros(n)
    y3 = np.zeros(n); y4 = np.zeros(n); y5 = np.zeros(n)
    out = np.empty((n, n_out))
    p = p_mean[None, :] + rng.standard_normal((N, n)) * p_sd[None, :]
    Aa, Bb, aa, bb = A * a, B * b, a * a, b * b
    for k in range(N):
        s_pyr = 2 * e0 / (1 + np.exp(r * (v0 - (y1 - y2))))
        s_exc = 2 * e0 / (1 + np.exp(r * (v0 - C1 * y0)))
        s_inh = 2 * e0 / (1 + np.exp(r * (v0 - C3 * y0)))
        d3 = Aa * s_pyr - 2 * a * y3 - aa * y0
        d4 = Aa * (p[k] + C2 * s_exc) - 2 * a * y4 - aa * y1
        d5 = Bb * C4 * s_inh - 2 * b * y5 - bb * y2
        y0 = y0 + dt * y3; y1 = y1 + dt * y4; y2 = y2 + dt * y5
        y3 = y3 + dt * d3; y4 = y4 + dt * d4; y5 = y5 + dt * d5
        if k >= nb:
            out[:, k - nb] = y1 - y2
    return out


def simulate_v3(label, sensors, fs, duration_s, rng, effect, effect_type="nmm", net="1020",
                unit=None, mains="random", bad="random", p=SIM3, lf_path="leadfield_sample.npz", effect_labels=None):
    """sensors: canonical 10-20 names (subset of rs.BASE_CANON) or EGI E-names."""
    LF = load_leadfield(lf_path)
    labels = list(LF["labels"])
    nl = len(labels)
    fs_int = p["fs_int"]
    n_int = int(duration_s * fs_int)
    eff_mask = np.array([lb in (effect_labels or EFFECT_LABELS) for lb in labels])
    # -- subject-level neural-mass parameters
    par = {k: rng.normal(*p[k]) for k in ("A", "B", "a", "b", "C", "p_mean")}
    jit = lambda: 1 + p["node_jitter"] * rng.standard_normal(nl)
    A = par["A"] * jit(); b = par["b"] * jit()
    B = np.full(nl, par["B"]); a = np.full(nl, par["a"]); C = np.full(nl, par["C"])
    pm = par["p_mean"] + 5.0 * rng.standard_normal(nl)
    psd = np.full(nl, p["p_sd"])
    chi = np.full(nl, rng.uniform(*p["bg_chi"]))
    eps_i = effect * rng.uniform(*p["effect_jitter"]) if (label and effect > 0) else 0.0
    if eps_i:
        if effect_type == "nmm":
            A[eff_mask] *= 1 + eps_i * p["effect_gain"]
            psd[eff_mask] *= 1 + eps_i * p["effect_noise"]
        elif effect_type == "aperiodic":
            chi[eff_mask] += eps_i * p["aperiodic_shift"]
        else:
            raise ValueError(effect_type)
    S = jansen_rit(A, B, a, b, C, pm, psd, fs_int, n_int, rng, p["burn_s"])
    S -= S.mean(axis=1, keepdims=True)
    # regime diagnostics of the effect-region columns (no random numbers used)
    from scipy import stats as _st
    fq, Pq = signal.welch(S[eff_mask], fs_int, nperseg=int(4 * fs_int))
    mq = (fq >= 1) & (fq <= 40)
    pk = fq[mq][Pq[:, mq].argmax(axis=1)]
    sk = _st.skew(S[eff_mask], axis=1)
    diag = dict(eff_peak_hz=float(np.median(pk)), eff_frac_nonalpha=float(np.mean((pk < 7.5) | (pk > 13))),
                eff_skew=float(np.median(sk)), eff_frac_skew1=float(np.mean(np.abs(sk) > 1)),
                eff_sd_ratio=float(np.median(S[eff_mask].std(axis=1)) / np.median(S[~eff_mask].std(axis=1))))
    # -- aperiodic background per region (own exponent), scaled relative to the neural-mass output
    f = np.fft.rfftfreq(n_int, 1 / fs_int)
    shape = np.where(f[None, :] > 0, np.maximum(f[None, :], 0.5) ** (-chi[:, None] / 2), 0.0)
    bg = rs._colored(n_int, fs_int, shape, rng, nl)
    S = S + rng.uniform(*p["bg_ratio"]) * np.median(S.std(axis=1)) * bg
    # -- physical forward model (3-layer BEM of a real head)
    if net == "egi":
        G = LF["egi128_G"]
        rows = [list(LF["egi128_ch"]).index(s) for s in sensors]
    else:
        G = LF["base19_G"]
        rows = [rs.BASE_CANON.index(s) for s in sensors]
    X = G[rows] @ S
    # -- resample from the integration rate to the recording rate
    from fractions import Fraction
    fr = Fraction(int(fs), int(fs_int)).limit_denominator(1000)
    X = signal.resample_poly(X, fr.numerator, fr.denominator, axis=-1)
    n = int(fs * duration_s)
    X = X[:, :n]
    nch = X.shape[0]
    X = X / np.median(X.std(axis=1)) * p["scale_to_uv"]
    bgamp = np.where(np.fft.rfftfreq(n, 1 / fs) > 0, np.maximum(np.fft.rfftfreq(n, 1 / fs), 0.5) ** -0.5, 0.0)
    X += p["sensor_noise_uv"] * rs._colored(n, fs, bgamp, rng, nch)
    gain = rng.lognormal(0, p["gain_sd"])
    X *= gain
    # -- physiological artefacts and recording nuisances shared with v2
    sens_xyz = rs.positions(sensors, net)
    t = np.arange(n) / fs
    bl = rs.SIM2["blink"]
    blink = rng.random() < bl["p"]
    if blink:
        w = np.exp(-((sens_xyz - rs._sph(100, 0)) ** 2).sum(1) / (2 * bl["width"] ** 2))[:, None]
        rate = rng.uniform(*bl["rate_hz"])
        for tc in np.cumsum(rng.exponential(1 / rate, int(rate * duration_s * 2) + 2)):
            d = rng.uniform(*bl["dur_s"]); i0, i1 = int(tc * fs), int((tc + d) * fs)
            if i1 >= n:
                break
            X[:, i0:i1] += w * rng.uniform(*bl["amp_uv"]) * gain * np.sin(np.linspace(0, np.pi, i1 - i0))
    em = rs.SIM2["emg"]
    emg = rng.random() < em["p"]
    if emg:
        c = rs._sph(*rs._ANALYTIC[str(rng.choice(["T7", "T8"]))])
        w = np.exp(-((sens_xyz - c) ** 2).sum(1) / (2 * em["width"] ** 2))[:, None]
        sos = signal.butter(4, em["hp_hz"], btype="highpass", fs=fs, output="sos")
        nz = signal.sosfilt(sos, rng.standard_normal((nch, n)), axis=-1)
        nz /= nz.std(axis=-1, keepdims=True)
        env = np.zeros(n)
        for _ in range(int(rng.integers(em["bursts"][0], em["bursts"][1] + 1))):
            L = int(rng.uniform(*em["burst_s"]) * fs); s0 = int(rng.integers(0, max(1, n - L)))
            env[s0:s0 + L] = np.maximum(env[s0:s0 + L], signal.windows.tukey(len(env[s0:s0 + L]), 0.3))
        X += w * rng.uniform(*em["rms_uv"]) * gain * nz * env
    X, info = rs._add_nuisance(X, t, fs, rng, mains, bad, unit, nch, 0.7, 0.15, 0.3)
    info.update(**diag)
    info.update(gain=gain, eps_i=eps_i, A=par["A"], b=par["b"], p_mean=par["p_mean"], chi=float(chi[0]),
                blink=bool(blink), emg=bool(emg), age=np.nan, alpha_pf=np.nan)
    return X, info


def subject_worker_v3(task):
    """Same contract as revision_sim._subject_worker, for the independent generator."""
    rng = np.random.default_rng(task["ss"])
    load_leadfield(task["lf_path"])
    sig, info = simulate_v3(task["label"], task["sensors"], task["fs"], task["dur"], rng, task["effect"],
                            effect_type=task.get("effect_type", "nmm"), net=task["net"], lf_path=task["lf_path"],
                            p=task.get("p3", SIM3), effect_labels=task.get("effect_labels"))
    if task["net"] == "egi":
        names, style = list(task["sensors"]), "egi"
    else:
        names, style = rs.style_names(task["canon"], rng, task.get("style"))
    out = {"meta": dict(sid=task["sid"], label=task["label"], kind=task["kind"], fs=task["fs"], dur=task["dur"],
                        nch=len(names), n_regions=rs.n_regions(task["canon"]) if task["net"] != "egi" else 11,
                        style=style, **{k: v for k, v in info.items() if k != "bad_idx"}), "feats": {}}
    bad_name = None if info["bad_idx"] is None else names[info["bad_idx"]]
    for pname, kinds in task["plan"].items():
        try:
            epochs, canon, rep = rs.prep(sig, task["fs"], names, rs.PREP_CFGS[pname])
            bp = rs.band_powers(epochs)
            for k in kinds:
                out["feats"][f"{pname}|{k}"] = rs.FEATURE_FNS[k](epochs, canon, bp)
            if pname == "full":
                out["meta"].update(det_unit=rep["units"].split()[0], det_mains=rep["mains_hz"],
                                   det_bad=[names[i] for i in rep["bad_idx"]], inj_bad=bad_name,
                                   n_epochs=rep["n_epochs"], regions_used=rep["regions"], n_mapped=rep["n_mapped"])
        except ValueError as err:
            for k in kinds:
                out["feats"][f"{pname}|{k}"] = None
            if pname == "full":
                out["meta"]["error"] = str(err)
    return out
