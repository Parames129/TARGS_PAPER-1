"""Third-round revision: shared simulation and feature code.

- simulate_v2_mod: the v2 simulator with optional mismatch effects (alpha-peak slowing, source coupling,
  effect centred between region boundaries) and an optional recorded reference electrode.
- prep_ref: the released preprocessing with a choice of re-reference (none / CAR / median / REST).
- IAF-adjusted band powers.
"""
import numpy as np
from scipy import signal
import eeg_adaptive as eeg
import revision_sim as rs

LEFT_POST = ("T_L", "P_L", "O_L")


# ------------------------------------------------------------------ simulator (v2 + mismatch modes)
def simulate_v2_mod(label, sensor_names, fs, duration_s, rng, effect, p=rs.SIM2, net="1020",
                    mode="regional", strength=0.0):
    """Identical to rs.simulate_v2 for mode='regional'. Other modes (dyslexic subjects only):
    'alpha_slow'  alpha peak lowered by `strength` Hz, no amplitude effect
    'coupling'    left temporo-parieto-occipital sources share a common component (fraction `strength`
                  of variance), each source's power unchanged
    'boundary'    the usual amplitude effect (size `effect`), but weighted by a Gaussian centred at CP5,
                  between the central, temporal and parietal regions, instead of by region
    """
    n, nch = int(fs * duration_s), len(sensor_names)
    t = np.arange(n) / fs
    f = np.fft.rfftfreq(n, 1 / fs)
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
    if mode == "alpha_slow" and label:
        pf["alpha"] -= strength
    bg_amp = np.where(f > 0, np.maximum(f, 0.5) ** (-chi / 2), 0.0)
    S = p["bg_rms_uv"] * rs._colored(n, fs, bg_amp, rng, 19)
    if mode == "boundary":
        d = np.linalg.norm(rs.SRC_XYZ - rs._sph(*_cp5()), axis=1)
        w_eff = 1 + (p["effect_region_gain"] - 1) * np.exp(-d ** 2 / (2 * 0.25 ** 2)) / 1.0
        w_eff = 1 + (w_eff - 1) / (w_eff.max() - 1) * (p["effect_region_gain"] - 1)   # peak gain 1.6
    else:
        w_eff = np.array([p["effect_region_gain"] if r in p["effect_regions"] else 1.0 for r in rs._SRC_REGION])
    amp_eff = eps_i if mode in ("regional", "boundary") else 0.0
    for r, q in p["rhythms"].items():
        fc = pf[r] + rng.normal(0, p["source_freq_sd"], 19)
        shape = np.exp(-(f[None, :] - fc[:, None]) ** 2 / (2 * q["bw"] ** 2))
        x = rs._colored(n, fs, shape, rng, 19)
        topo = np.array([p["topography"][r][lb] for lb in rs._SRC_LOBE])
        jitter = rng.lognormal(0, p["source_amp_sd"], 19)
        cls = np.clip(1 + q["sign"] * amp_eff * w_eff * label, 0.1, None)
        S += (q["base_uv"] / np.sqrt(2) * amp[r] * topo * jitter * cls)[:, None] * x
    if mode == "coupling" and label and strength > 0:
        idx = [i for i, r in enumerate(rs._SRC_REGION) if r in LEFT_POST]
        z = S[idx] / S[idx].std(axis=1, keepdims=True)
        common = z.mean(axis=0); common /= common.std()
        sd = S[idx].std(axis=1, keepdims=True)
        S[idx] = sd * (np.sqrt(1 - strength) * z + np.sqrt(strength) * common)
    sens_xyz = rs.positions(sensor_names, net)
    X = rs.mixing_matrix(sens_xyz, rs.SRC_XYZ, p["vc_lambda"]) @ S
    X += p["sensor_noise_uv"] * rs._colored(n, fs, bg_amp, rng, nch)
    X *= p["scale_to_uv"] / np.median(X.std(axis=1))
    gain = rng.lognormal(0, p["gain_sd"])
    X *= gain
    b = p["blink"]
    blink = rng.random() < b["p"]
    if blink:
        w = np.exp(-((sens_xyz - rs._sph(100, 0)) ** 2).sum(1) / (2 * b["width"] ** 2))[:, None]
        rate = rng.uniform(*b["rate_hz"])
        for tc in np.cumsum(rng.exponential(1 / rate, int(rate * duration_s * 2) + 2)):
            dd = rng.uniform(*b["dur_s"])
            i0, i1 = int(tc * fs), int((tc + dd) * fs)
            if i1 >= n:
                break
            X[:, i0:i1] += w * rng.uniform(*b["amp_uv"]) * gain * np.sin(np.linspace(0, np.pi, i1 - i0))
    e = p["emg"]
    emg = rng.random() < e["p"]
    if emg:
        sides = {"L": [rs._sph(*rs._ANALYTIC["T7"])], "R": [rs._sph(*rs._ANALYTIC["T8"])]}
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
    X, info = rs._add_nuisance(X, t, fs, rng, "random", "random", None, nch, p["mains_p"], p["bad_p"], p["transient_p"])
    info.update(gain=gain, eps_i=eps_i, age=age, alpha_pf=pf["alpha"], chi=chi, blink=bool(blink), emg=bool(emg))
    return X, info


def _cp5():
    # CP5: midway (on the sphere) between C3/T7 and P3/P7
    xyz = rs.positions(["C3", "T7", "P3", "P7"]).mean(axis=0)
    xyz /= np.linalg.norm(xyz)
    th = np.degrees(np.arccos(xyz[2])); ph = np.degrees(np.arctan2(xyz[1], xyz[0]))
    return th, ph


# ------------------------------------------------------------------ REST
_REST = {}


def rest_leadfield():
    """Lead field of the 19 10-20 sites for a 3-shell spherical head and a 10-mm volume grid (MNE)."""
    if "G" not in _REST:
        import mne
        mne.set_log_level("ERROR")
        info = mne.create_info(rs.BASE_CH, 250.0, "eeg")
        info.set_montage("standard_1020")
        sphere = mne.make_sphere_model("auto", "auto", info)
        src = mne.setup_volume_source_space(sphere=sphere, pos=10.0, exclude=20.0)
        fwd = mne.make_forward_solution(info, trans=None, src=src, bem=sphere, eeg=True, meg=False)
        _REST["G"] = fwd["sol"]["data"]            # (19, 3 * n_sources), average reference not applied
    return _REST["G"]


def rest_matrix(canon):
    G = rest_leadfield()[[rs.BASE_CANON.index(c) for c in canon]]
    H = np.eye(len(canon)) - 1.0 / len(canon)
    Ga = H @ G
    # V_rest = G Ga^+ V_avg (Yao 2001), Tikhonov-regularised pseudo-inverse
    U, s, Vt = np.linalg.svd(Ga, full_matrices=False)
    lam = 1e-3 * s.max()
    Gpinv = Vt.T @ np.diag(s / (s ** 2 + lam ** 2)) @ U.T
    return G @ Gpinv @ H


# ------------------------------------------------------------------ preprocessing with reference choice
def prep_ref(data, fs, names, ref="car"):
    """Released preprocessing (rs.prep 'full'), with the CAR step replaced by `ref`."""
    data = np.asarray(data, dtype=float)
    keep, canon, _, _ = eeg.resolve_channels(list(names), None, 4)
    data = data[keep]
    data, _ = eeg.to_microvolts(data, "auto")
    data = data - data.mean(axis=1, keepdims=True)
    mains = eeg.detect_mains(data, fs)
    if mains:
        b, a = signal.iirnotch(mains, Q=30.0, fs=fs)
        data = signal.filtfilt(b, a, data, axis=-1)
    data = eeg.resample_to(data, fs)
    sos = signal.butter(4, eeg.BANDPASS, btype="bandpass", fs=eeg.TARGET_FS, output="sos")
    data = signal.sosfiltfilt(sos, data, axis=-1)
    bad = eeg.find_bad_channels(data)
    data, canon = data[~bad], [c for c, b_ in zip(canon, bad) if not b_]
    if rs.n_regions(canon) < 4:
        raise ValueError("< 4 regions")
    if ref == "car":
        data = data - data.mean(axis=0, keepdims=True)
    elif ref == "median":
        data = data - np.median(data, axis=0, keepdims=True)
    elif ref == "rest":
        data = rest_matrix(canon) @ data
    epochs = eeg.make_epochs(data)
    ptp = np.ptp(epochs, axis=-1).max(axis=1)
    if len(ptp) >= 3:
        epochs = epochs[ptp <= 3.0 * np.median(ptp)]
    return epochs, canon


# ------------------------------------------------------------------ IAF-adjusted bands
def psd(epochs):
    return signal.welch(epochs, fs=eeg.TARGET_FS, nperseg=int(2 * eeg.TARGET_FS), axis=-1)


def estimate_iaf(freqs, P, canon):
    """Alpha peak (7-13 Hz) of the mean posterior spectrum after removing a log-log 1/f fit."""
    post = [i for i, c in enumerate(canon) if eeg.channel_region(c)[0] in "PO"]
    spec = P[:, post if post else slice(None)].mean(axis=(0, 1))
    m = (freqs >= 2) & (freqs <= 35) & ~((freqs >= 6.5) & (freqs <= 14))
    coef = np.polyfit(np.log10(freqs[m]), np.log10(spec[m]), 1)
    resid = np.log10(spec) - np.polyval(coef, np.log10(np.maximum(freqs, 1e-3)))
    a = (freqs >= 7) & (freqs <= 13)
    w = np.clip(resid[a], 0, None)
    return float((freqs[a] * w).sum() / w.sum()) if w.sum() > 0 else 10.0     # centre of gravity of the excess


def band_powers_edges(freqs, P, edges):
    df = freqs[1] - freqs[0]

    def band(lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        return P[..., m].sum(axis=-1) * df
    bp = np.stack([band(lo, hi) for lo, hi in edges], axis=-1)
    return bp, band(*eeg.BANDPASS) + 1e-12


def iaf_edges(iaf):
    s = iaf - 10.0          # shift theta/alpha/beta boundaries with the individual alpha peak
    return [(1, 4 + s), (4 + s, 8 + s), (8 + s, 13 + s), (13 + s, 30), (30, 45)]


def features_fixed_and_iaf(epochs, canon):
    freqs, P = psd(epochs)
    fixed = rs.region_features(epochs, canon, band_powers_edges(freqs, P, list(eeg.BANDS.values())))
    iaf = estimate_iaf(freqs, P, canon)
    adj = rs.region_features(epochs, canon, band_powers_edges(freqs, P, iaf_edges(iaf)))
    return fixed, adj, iaf
