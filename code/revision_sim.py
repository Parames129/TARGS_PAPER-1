"""revision_sim.py - simulators (v1 original, v2 realistic) and a configurable copy of the
eeg_adaptive pipeline, used by the revision experiments. Released with the paper for reproducibility.

Every function is deterministic given a numpy Generator; cohort workers derive one seed per subject.
"""
import re
from functools import lru_cache

import numpy as np
from scipy import signal

import eeg_adaptive as eeg

try:
    import mne
    mne.set_log_level("ERROR")
    HAVE_MNE = True
except ImportError:
    HAVE_MNE = False

BASE_CH = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
           "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]
BASE_CANON = [eeg.normalize_channel_name(c) for c in BASE_CH]          # 'FP1', 'FZ', ...
OPENBCI8 = ["Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2"]
OPENBCI8_CANON = [eeg.normalize_channel_name(c) for c in OPENBCI8]
EGI_NAMES = [f"E{i}" for i in range(1, 129)]

# analytic fallback (polar angle from Cz, azimuth from the nose, degrees) if MNE is not installed
_ANALYTIC = {"FP1": (72, -18), "FP2": (72, 18), "F7": (72, -54), "F3": (49, -40), "FZ": (36, 0),
             "F4": (49, 40), "F8": (72, 54), "T7": (72, -90), "C3": (36, -90), "CZ": (0, 0),
             "C4": (36, 90), "T8": (72, 90), "P7": (72, -126), "P3": (49, -140), "PZ": (36, 180),
             "P4": (49, 140), "P8": (72, 126), "O1": (72, -162), "O2": (72, 162), "FPZ": (72, 0)}


def _sph(theta, phi):
    t, p = np.radians(theta), np.radians(phi)
    return np.array([np.sin(t) * np.sin(p), np.sin(t) * np.cos(p), np.cos(t)])


@lru_cache(maxsize=None)
def _montage_xyz(montage_name):
    pos = mne.channels.make_standard_montage(montage_name).get_positions()["ch_pos"]
    names = list(pos)
    xyz = eeg._unit_sphere([pos[n] for n in names])
    return {n.upper(): xyz[i] for i, n in enumerate(names)}


def positions(names, net="1020"):
    """Unit-sphere xyz (x right, y front, z up) for canonical 10-20 names or EGI E-names."""
    if net == "egi":
        d = _montage_xyz("GSN-HydroCel-128")
        return np.array([d[n.upper()] for n in names])
    if HAVE_MNE:
        d = _montage_xyz("standard_1005")
        return np.array([d[n] for n in names])
    return np.array([_sph(*_ANALYTIC[n]) for n in names])


def mixing_matrix(sensor_xyz, source_xyz, lam):
    d2 = ((sensor_xyz[:, None, :] - source_xyz[None, :, :]) ** 2).sum(-1)
    m = np.exp(-d2 / (2 * lam ** 2))
    return m / m.sum(axis=1, keepdims=True)


def n_regions(canon_names):
    return len({eeg.channel_region(c) for c in canon_names})


SRC_XYZ = positions(BASE_CANON)


SIM2 = dict(
    bg_rms_uv=10.0,                 # 1/f background per source
    chi_range=(1.0, 2.0),           # 1/f power exponent per subject (amplitude ~ f^-chi/2)
    rhythms={                       # base amplitude (sinusoid-equivalent peak, uV), bandwidth (Hz), effect sign
        "theta": dict(base_uv=2.5, bw=1.0, sign=+1.0, f_mean=6.0, f_sd=0.5, f_clip=(4.5, 7.5)),
        "alpha": dict(base_uv=4.0, bw=1.0, sign=-0.5),          # peak frequency depends on age (below)
        "beta": dict(base_uv=1.5, bw=2.5, sign=-0.5, f_mean=20.0, f_sd=2.5, f_clip=(15.0, 26.0)),
    },
    topography={                    # amplitude multiplier per lobe (alpha posterior, theta fronto-midline)
        "theta": {"F": 1.2, "C": 1.0, "T": 1.0, "P": 1.0, "O": 1.0, "M": 1.2},
        "alpha": {"F": 0.6, "C": 1.0, "T": 1.0, "P": 1.4, "O": 1.6, "M": 1.0},
        "beta": {"F": 1.0, "C": 1.2, "T": 1.0, "P": 1.0, "O": 1.0, "M": 1.1},
    },
    subj_amp_sd=0.30,               # lognormal SD of each rhythm's amplitude between subjects
    source_amp_sd=0.15,             # lognormal SD between sources within a subject
    source_freq_sd=0.2,             # Hz jitter of the peak between sources
    effect_sd_frac=0.5,             # eps_i = eps * (1 + 0.5 N(0,1)), drawn per dyslexic subject
    effect_regions=("T_L", "P_L", "O_L"), effect_region_gain=1.6,
    age_range=(7.0, 12.0),          # years, uniform, independent of class
    alpha_pf=dict(intercept=8.0, slope=0.30, sd=0.6, clip=(7.0, 11.5)),   # Hz at age 7, Hz/year
    theta_age_slope=-0.05,          # theta amplitude falls 5 %/year (developmental, class-independent)
    vc_lambda=0.45,                 # volume-conduction kernel width (unit-sphere chord)
    sensor_noise_uv=3.0,            # independent 1/f sensor noise
    scale_to_uv=15.0,               # median sensor SD after mixing, before gain
    gain_sd=0.25,                   # amplifier/skin gain, lognormal, multiplies everything
    blink=dict(p=0.5, rate_hz=(0.1, 0.4), amp_uv=(80.0, 200.0), dur_s=(0.25, 0.40), width=0.40),
    emg=dict(p=0.3, rms_uv=(5.0, 15.0), bursts=(2, 6), burst_s=(1.0, 4.0), width=0.35, hp_hz=20.0),
    transient_p=0.3, mains_p=0.7, bad_p=0.15,
)

_SRC_LOBE = [("M" if eeg.channel_region(c) == "MID" else eeg.channel_region(c)[0]) for c in BASE_CANON]
_SRC_REGION = [eeg.channel_region(c) for c in BASE_CANON]


def _colored(n, fs, amp, rng, k):
    """k unit-variance Gaussian processes with amplitude spectrum amp (len(f)) or (k, len(f))."""
    f = np.fft.rfftfreq(n, 1 / fs)
    a = amp(f) if callable(amp) else amp
    spec = (rng.standard_normal((k, len(f))) + 1j * rng.standard_normal((k, len(f)))) * a
    spec[..., 0] = 0
    x = np.fft.irfft(spec, n=n, axis=-1)
    return x / (x.std(axis=-1, keepdims=True) + 1e-12)


def _add_nuisance(sig, t, fs, rng, mains, bad, unit, ch_count, mains_p, bad_p, transient_p):
    """DC + drift, mains, movement transient, bad channel, units (as in the original v1 simulator)."""
    n = sig.shape[1]
    sig += rng.normal(0, 50, (ch_count, 1)) + 30 * np.sin(2 * np.pi * 0.1 * t + rng.uniform(0, 6.28))
    if mains == "random":
        mains = float(rng.choice([50.0, 60.0])) if rng.random() < mains_p else None
    if mains is not None and mains < fs / 2 - 1:
        sig += rng.uniform(2, 8) * np.sin(2 * np.pi * mains * t + rng.uniform(0, 6.28))
    else:
        mains = None
    transient = rng.random() < transient_p
    if transient:
        c = int(rng.integers(int(0.5 * fs), n - int(0.5 * fs)))
        half = int(0.15 * fs)
        sig[:, c - half:c + half] += 500 * np.hanning(2 * half)
    if bad == "random":
        bad = str(rng.choice(["flat", "noisy"])) if rng.random() < bad_p else None
    bad_idx = None
    if bad is not None:
        bad_idx = int(rng.integers(ch_count))
        sig[bad_idx] = rng.normal(0, 300, n) if bad == "noisy" else 0.0
    unit = unit or str(rng.choice(["V", "uV"]))
    if unit == "V":
        sig = sig * 1e-6
    return sig, dict(unit=unit, mains=mains, bad_type=bad, bad_idx=bad_idx, transient=bool(transient))


def simulate_v1(label, canon, fs, duration_s, rng, effect, unit=None, mains="random", bad="random"):
    """The paper's original simulator (unchanged apart from the effect parameter)."""
    n, nch = int(fs * duration_s), len(canon)
    freqs = np.fft.rfftfreq(n, 1 / fs)
    shaping = np.ones_like(freqs)
    shaping[1:] = 1 / np.sqrt(freqs[1:])
    sig = np.fft.irfft(np.fft.rfft(rng.standard_normal((nch, n)), axis=-1) * shaping, n=n, axis=-1)
    sig = 10 * sig / sig.std(axis=-1, keepdims=True)
    t = np.arange(n) / fs
    w = np.array([1.6 if eeg.channel_region(c) in ("T_L", "P_L", "O_L") else 1.0 for c in canon])[:, None]
    gain = rng.lognormal(0, 0.25)
    for centre, base, eff in [(6.0, 2.5, +effect), (10.0, 4.0, -0.5 * effect), (20.0, 1.5, -0.5 * effect)]:
        phase = rng.uniform(0, 2 * np.pi, (nch, 1))
        sig += base * gain * (1 + eff * w * label) * np.sin(2 * np.pi * (centre + rng.normal(0, 0.5)) * t + phase)
    sig, info = _add_nuisance(sig, t, fs, rng, mains, bad, unit, nch, 0.7, 0.15, 0.3)
    info.update(gain=gain, eps_i=effect if label else 0.0, age=np.nan, alpha_pf=10.0, chi=1.0,
                blink=False, emg=False)
    return sig, info


def simulate_v2(label, sensor_names, fs, duration_s, rng, effect, net="1020",
                unit=None, mains="random", bad="random", p=SIM2):
    """Realistic simulator: subject biology + volume conduction + physiological artefacts."""
    n, nch = int(fs * duration_s), len(sensor_names)
    t = np.arange(n) / fs
    f = np.fft.rfftfreq(n, 1 / fs)
    # -- subject-level biology (independent of class)
    age = rng.uniform(*p["age_range"])
    chi = rng.uniform(*p["chi_range"])
    ap = p["alpha_pf"]
    pf = {"alpha": float(np.clip(ap["intercept"] + ap["slope"] * (age - 7) + rng.normal(0, ap["sd"]), *ap["clip"]))}
    for r in ("theta", "beta"):
        q = p["rhythms"][r]
        pf[r] = float(np.clip(rng.normal(q["f_mean"], q["f_sd"]), *q["f_clip"]))
    amp = {r: rng.lognormal(0, p["subj_amp_sd"]) for r in p["rhythms"]}
    amp["theta"] *= max(0.2, 1 + p["theta_age_slope"] * (age - 7))
    eps_i = effect * (1 + p["effect_sd_frac"] * rng.standard_normal()) if (label and effect > 0) else 0.0
    # -- 19 cortical sources
    bg_amp = np.where(f > 0, np.maximum(f, 0.5) ** (-chi / 2), 0.0)
    S = p["bg_rms_uv"] * _colored(n, fs, bg_amp, rng, 19)
    w_eff = np.array([p["effect_region_gain"] if r in p["effect_regions"] else 1.0 for r in _SRC_REGION])
    for r, q in p["rhythms"].items():
        fc = pf[r] + rng.normal(0, p["source_freq_sd"], 19)
        shape = np.exp(-(f[None, :] - fc[:, None]) ** 2 / (2 * q["bw"] ** 2))
        x = _colored(n, fs, shape, rng, 19)
        topo = np.array([p["topography"][r][lb] for lb in _SRC_LOBE])
        jitter = rng.lognormal(0, p["source_amp_sd"], 19)
        cls = np.clip(1 + q["sign"] * eps_i * w_eff * label, 0.1, None)
        S += (q["base_uv"] / np.sqrt(2) * amp[r] * topo * jitter * cls)[:, None] * x
    # -- volume conduction to the sensors
    sens_xyz = positions(sensor_names, net)
    X = mixing_matrix(sens_xyz, SRC_XYZ, p["vc_lambda"]) @ S
    X += p["sensor_noise_uv"] * _colored(n, fs, bg_amp, rng, nch)
    X *= p["scale_to_uv"] / np.median(X.std(axis=1))
    gain = rng.lognormal(0, p["gain_sd"])
    X *= gain
    # -- physiological artefacts (class-independent)
    b = p["blink"]
    blink = rng.random() < b["p"]
    if blink:
        w = np.exp(-((sens_xyz - _sph(100, 0)) ** 2).sum(1) / (2 * b["width"] ** 2))[:, None]
        rate = rng.uniform(*b["rate_hz"])
        for tc in np.cumsum(rng.exponential(1 / rate, int(rate * duration_s * 2) + 2)):
            d = rng.uniform(*b["dur_s"])
            i0, i1 = int(tc * fs), int((tc + d) * fs)
            if i1 >= n:
                break
            X[:, i0:i1] += w * rng.uniform(*b["amp_uv"]) * gain * np.sin(np.linspace(0, np.pi, i1 - i0))
    e = p["emg"]
    emg = rng.random() < e["p"]
    if emg:
        sides = {"L": [_sph(*_ANALYTIC["T7"])], "R": [_sph(*_ANALYTIC["T8"])]}
        centres = sides[str(rng.choice(["L", "R"]))] if rng.random() < 0.5 else sides["L"] + sides["R"]
        w = np.max([np.exp(-((sens_xyz - c) ** 2).sum(1) / (2 * e["width"] ** 2)) for c in centres], axis=0)[:, None]
        sos = signal.butter(4, e["hp_hz"], btype="highpass", fs=fs, output="sos")
        noise = signal.sosfilt(sos, rng.standard_normal((nch, n)), axis=-1)
        noise /= noise.std(axis=-1, keepdims=True)
        env = np.zeros(n)
        for _ in range(int(rng.integers(e["bursts"][0], e["bursts"][1] + 1))):
            L = int(rng.uniform(*e["burst_s"]) * fs)
            s0 = int(rng.integers(0, max(1, n - L)))
            env[s0:s0 + L] = np.maximum(env[s0:s0 + L], signal.windows.tukey(len(env[s0:s0 + L]), 0.3))
        X += w * rng.uniform(*e["rms_uv"]) * gain * noise * env
    X, info = _add_nuisance(X, t, fs, rng, mains, bad, unit, nch, p["mains_p"], p["bad_p"], p["transient_p"])
    info.update(gain=gain, eps_i=eps_i, age=age, alpha_pf=pf["alpha"], chi=chi, blink=bool(blink), emg=bool(emg))
    return X, info


def style_names(canon_plain, rng, style=None):
    """Channel-name styles from the paper ('Fp1', 'EEG Fp1-REF', 'FP1', legacy T3/T5)."""
    style = style or str(rng.choice(["plain", "ref", "upper", "legacy"], p=[0.4, 0.2, 0.2, 0.2]))
    pretty = {c: n for c, n in zip(BASE_CANON, BASE_CH)}
    legacy = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
    fmt = {"plain": lambda n: n, "ref": lambda n: f"EEG {n}-REF", "upper": lambda n: n.upper(),
           "legacy": lambda n: legacy.get(n, n)}[style]
    return [fmt(pretty[c]) for c in canon_plain], style


MONTAGE_P = {"full19": 0.35, "openbci8": 0.25, "random": 0.40}     # as in the paper


def random_montage(rng, kind=None, probs=None):
    probs = probs or MONTAGE_P
    kind = kind or str(rng.choice(list(probs), p=list(probs.values())))
    if kind == "full19":
        return kind, list(BASE_CANON)
    if kind == "openbci8":
        return kind, list(OPENBCI8_CANON)
    while True:
        idx = sorted(rng.choice(19, int(rng.integers(8, 19)), replace=False))
        ch = [BASE_CANON[i] for i in idx]
        if n_regions(ch) >= 4:
            return kind, ch


PREP_CFGS = {
    "full":      dict(names=True, unit=True, notch=True, bad=True, car=True, reject=True),
    "no_names":  dict(names=False, unit=True, notch=True, bad=True, car=True, reject=True),
    "no_unit":   dict(names=True, unit=False, notch=True, bad=True, car=True, reject=True),
    "no_notch":  dict(names=True, unit=True, notch=False, bad=True, car=True, reject=True),
    "no_bad":    dict(names=True, unit=True, notch=True, bad=False, car=True, reject=True),
    "no_car":    dict(names=True, unit=True, notch=True, bad=True, car=False, reject=True),
    "no_reject": dict(names=True, unit=True, notch=True, bad=True, car=True, reject=False),
    "naive":     dict(names=False, unit=False, notch=False, bad=False, car=False, reject=False),
}
_EXACT = set(BASE_CH)          # a conventional pipeline expects exactly these labels
DEFAULT_SITE_DISTANCE = eeg._MAX_SITE_DISTANCE


def prep(data, fs, names, cfg, site_dist=None):
    """Raw recording -> (epochs, canonical names, report). Raises ValueError if unusable."""
    rep = {}
    data = np.asarray(data, dtype=float)
    if cfg["names"]:
        eeg._MAX_SITE_DISTANCE = DEFAULT_SITE_DISTANCE if site_dist is None else site_dist
        keep, canon, dropped, rep["mapping"] = eeg.resolve_channels(list(names), None, 4)
    else:
        keep = [i for i, nm in enumerate(names) if nm in _EXACT]
        canon = [eeg.normalize_channel_name(names[i]) for i in keep]
        if n_regions(canon) < 4:
            raise ValueError("exact-name matching found < 4 regions")
    rep["n_mapped"] = len(keep)
    data = data[keep]
    if cfg["unit"]:
        data, rep["units"] = eeg.to_microvolts(data, "auto")
    data = data - data.mean(axis=1, keepdims=True)
    rep["mains_hz"] = eeg.detect_mains(data, fs) if cfg["notch"] else None
    if rep["mains_hz"]:
        b, a = signal.iirnotch(rep["mains_hz"], Q=30.0, fs=fs)
        data = signal.filtfilt(b, a, data, axis=-1)
    data = eeg.resample_to(data, fs)
    sos = signal.butter(4, eeg.BANDPASS, btype="bandpass", fs=eeg.TARGET_FS, output="sos")
    data = signal.sosfiltfilt(sos, data, axis=-1)
    rep["bad_idx"] = []
    if cfg["bad"]:
        bad = eeg.find_bad_channels(data)
        rep["bad_idx"] = [keep[i] for i in np.flatnonzero(bad)]
        data, canon = data[~bad], [c for c, b_ in zip(canon, bad) if not b_]
    if n_regions(canon) < 4:
        raise ValueError("< 4 regions after bad-channel removal")
    if cfg["car"] and len(canon) >= 4:
        data = data - data.mean(axis=0, keepdims=True)
    epochs = eeg.make_epochs(data)
    if cfg["reject"]:
        ptp = np.ptp(epochs, axis=-1).max(axis=1)
        if len(ptp) >= 3:
            epochs = epochs[ptp <= 3.0 * np.median(ptp)]
    rep["n_epochs"] = len(epochs)
    rep["regions"] = n_regions(canon)
    return epochs, canon, rep


# ----------------------------------------------------------------------------- feature variants
def band_powers(epochs):
    freqs, psd = signal.welch(epochs, fs=eeg.TARGET_FS, nperseg=int(2 * eeg.TARGET_FS), axis=-1)
    df = freqs[1] - freqs[0]

    def band(lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        return psd[..., m].sum(axis=-1) * df
    bp = np.stack([band(lo, hi) for lo, hi in eeg.BANDS.values()], axis=-1)     # (ep, ch, 5)
    return bp, band(*eeg.BANDPASS) + 1e-12


def region_of(canon, scheme=""):
    r = eeg.channel_region(canon)
    if "p78T" in scheme and canon in ("P7", "P8"):
        r = "T_L" if canon == "P7" else "T_R"
    if "splitmid" in scheme and r == "MID":
        pre = re.match(r"[A-Z]+", canon).group(0)
        r = "MID_F" if pre in ("FP", "AF", "F") else "MID_C" if pre in ("FC", "C") else "MID_P"
    return r


def region_list(scheme=""):
    regs = [r for r in eeg.REGIONS if r != "MID"]
    return regs + (["MID_F", "MID_C", "MID_P"] if "splitmid" in scheme else ["MID"])


def region_features(epochs, canon, bp=None, power="rel", scheme=""):
    """Same layout as eeg.epoch_features (77 values) for the default scheme with relative power."""
    bp, total = bp if bp is not None else band_powers(epochs)
    eps = 1e-6
    vals = bp / total[..., None] if power == "rel" else np.log10(bp + 1e-12)
    regs = np.array([region_of(c, scheme) for c in canon])

    def block(v, raw):
        tbr = (np.log10((v[:, 1] + eps) / (v[:, 3] + eps)) if power == "rel"
               else np.log10((raw[:, 1] + 1e-12) / (raw[:, 3] + 1e-12)))
        return np.column_stack([v, tbr])
    feats, alpha = [], {}
    order = region_list(scheme)
    if not scheme:
        order = eeg.REGIONS
    for r in order:
        idx = np.flatnonzero(regs == r)
        if idx.size:
            v, raw = vals[:, idx, :].mean(axis=1), bp[:, idx, :].mean(axis=1)
        else:
            v = raw = np.full((bp.shape[0], 5), np.nan)
        feats.append(block(v, raw))
        alpha[r] = v[:, 2] if power == "rel" else raw[:, 2]
    feats.append(np.column_stack([np.log10((alpha[f"{lb}_R"] + eps) / (alpha[f"{lb}_L"] + eps)) for lb in eeg.LOBES]))
    feats.append(block(vals.mean(axis=1), bp.mean(axis=1)))
    return np.concatenate(feats, axis=1)


def channel_features(epochs, canon, bp=None, power="rel"):
    """Per-channel features on the 19 10-20 sites (6 each = 114); missing sites are NaN."""
    bp, total = bp if bp is not None else band_powers(epochs)
    vals = bp / total[..., None] if power == "rel" else np.log10(bp + 1e-12)
    out = []
    canon = np.array(canon)
    for c in BASE_CANON:
        idx = np.flatnonzero(canon == c)
        if idx.size:
            v, raw = vals[:, idx, :].mean(axis=1), bp[:, idx, :].mean(axis=1)
            out.append(np.column_stack([v, np.log10((raw[:, 1] + 1e-12) / (raw[:, 3] + 1e-12))]))
        else:
            out.append(np.full((bp.shape[0], 6), np.nan))
    return np.concatenate(out, axis=1)


def spline_region_features(epochs, canon, bp=None):
    """Spherical-spline interpolation of missing 10-20 channels (MNE), then the standard 77 features."""
    if not HAVE_MNE or any(c not in BASE_CANON for c in canon):
        return None
    try:
        return _spline(epochs, canon)
    except Exception:                       # interpolation failure -> treat as unprocessable
        return None


def _spline(epochs, canon):
    full = np.zeros((epochs.shape[0], 19, epochs.shape[2]))
    present = []
    for j, c in enumerate(BASE_CANON):
        idx = [i for i, cc in enumerate(canon) if cc == c]
        if idx:
            full[:, j] = epochs[:, idx].mean(axis=1)
            present.append(j)
    info = mne.create_info(BASE_CH, eeg.TARGET_FS, "eeg")
    ep = mne.EpochsArray(full * 1e-6, info, verbose=False)
    ep.set_montage("standard_1020")
    ep.info["bads"] = [BASE_CH[j] for j in range(19) if j not in present]
    if ep.info["bads"]:
        ep.interpolate_bads(reset_bads=True, verbose=False)
    return eeg.epoch_features(ep.get_data() * 1e6, BASE_CANON)


FEATURE_FNS = {
    "region_rel": lambda e, c, bp: region_features(e, c, bp, "rel"),
    "region_abs": lambda e, c, bp: region_features(e, c, bp, "abs"),
    "region_rel_splitmid": lambda e, c, bp: region_features(e, c, bp, "rel", "splitmid"),
    "region_rel_p78T": lambda e, c, bp: region_features(e, c, bp, "rel", "p78T"),
    "channel_rel": lambda e, c, bp: channel_features(e, c, bp, "rel"),
    "channel_abs": lambda e, c, bp: channel_features(e, c, bp, "abs"),
    "region_rel_spline": lambda e, c, bp: spline_region_features(e, c, bp),
}
CHANNEL_COLS = {c: list(range(6 * j, 6 * j + 6)) for j, c in enumerate(BASE_CANON)}


# ----------------------------------------------------------------------------- cohort workers
def _subject_worker(task):
    rng = np.random.default_rng(task["ss"])
    if task["sim"] == "v1":
        sig, info = simulate_v1(task["label"], task["canon"], task["fs"], task["dur"], rng, task["effect"])
    else:
        sig, info = simulate_v2(task["label"], task["sensors"], task["fs"], task["dur"], rng,
                                task["effect"], net=task["net"], p=task.get("p", SIM2))
    if task["net"] == "egi":
        names, style = list(EGI_NAMES), "egi"
    else:
        names, style = style_names(task["canon"], rng, task.get("style"))
    out = {"meta": dict(sid=task["sid"], label=task["label"], kind=task["kind"], fs=task["fs"], dur=task["dur"],
                        nch=len(names), n_regions=n_regions(task["canon"]) if task["net"] != "egi" else 11,
                        style=style, **{k: v for k, v in info.items() if k != "bad_idx"}),
           "feats": {}}
    bad_name = None if info["bad_idx"] is None else names[info["bad_idx"]]
    for pname, kinds in task["plan"].items():
        try:
            epochs, canon, rep = prep(sig, task["fs"], names, PREP_CFGS[pname], task.get("site_dist"))
            bp = band_powers(epochs)
            for k in kinds:
                out["feats"][f"{pname}|{k}"] = FEATURE_FNS[k](epochs, canon, bp)
            if pname == "full":
                out["meta"].update(
                    det_unit=rep["units"].split()[0], det_mains=rep["mains_hz"],
                    det_bad=[names[i] for i in rep["bad_idx"]], inj_bad=bad_name,
                    n_epochs=rep["n_epochs"], regions_used=rep["regions"], n_mapped=rep["n_mapped"])
        except ValueError as err:
            for k in kinds:
                out["feats"][f"{pname}|{k}"] = None
            if pname == "full":
                out["meta"]["error"] = str(err)
    return out


def _multi_montage_worker(task):
    """E7/E13: one subject recorded with all 19 channels, then re-analysed under many channel subsets."""
    rng = np.random.default_rng(task["ss"])
    sig, info = simulate_v2(task["label"], BASE_CANON, task["fs"], task["dur"], rng, task["effect"],
                            p=task.get("p", SIM2))
    names = list(BASE_CH)
    out = []
    for mname, chans in task["montages"]:
        idx = [BASE_CANON.index(c) for c in chans]
        if n_regions(chans) < 4:                     # the pipeline refuses these montages
            out.append(dict(sid=task["sid"], label=task["label"], montage=mname, n_ch=len(chans),
                            n_regions=n_regions(chans), x=None))
            continue
        try:
            epochs, canon, rep = prep(sig[idx], task["fs"], [names[i] for i in idx], PREP_CFGS["full"])
            x = region_features(epochs, canon)
            regs = rep["regions"]
        except ValueError:
            x, regs = None, n_regions(chans)
        out.append(dict(sid=task["sid"], label=task["label"], montage=mname, n_ch=len(chans),
                        n_regions=regs, x=x))
    return out


def _psd_worker(task):
    """E11: mean preprocessed spectrum (global and left-parietal) for one simulated subject."""
    rng = np.random.default_rng(task["ss"])
    if task["sim"] == "v1":
        sig, _ = simulate_v1(task["label"], task["canon"], task["fs"], task["dur"], rng, task["effect"])
    else:
        sig, _ = simulate_v2(task["label"], task["canon"], task["fs"], task["dur"], rng, task["effect"],
                             p=task.get("p", SIM2))
    names, _ = style_names(task["canon"], rng, "plain")
    epochs, canon, _ = prep(sig, task["fs"], names, PREP_CFGS["full"])
    freqs, psd = signal.welch(epochs, fs=eeg.TARGET_FS, nperseg=int(2 * eeg.TARGET_FS), axis=-1)
    psd = psd.mean(axis=0)                                           # (ch, f)
    psd = psd / psd[:, (freqs >= 1) & (freqs <= 45)].sum(axis=1, keepdims=True)   # relative spectrum
    lp = [i for i, c in enumerate(canon) if eeg.channel_region(c) == "P_L"]
    return dict(label=task["label"], kind=task["kind"], freqs=freqs, glob=psd.mean(axis=0),
                pl=psd[lp].mean(axis=0) if lp else None)
