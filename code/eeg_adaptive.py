"""eeg_adaptive.py - montage-, sampling-rate- and unit-agnostic EEG preprocessing,
feature extraction and inference for the dyslexia demo pipeline.

Every automatic decision (units, mains frequency, bad channels, resampling...) is returned in a report.
"""
import pickle
import re
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import signal

TARGET_FS = 250.0                  # every recording is resampled to this rate
EPOCH_S = 4.0                      # epoch length in seconds
BANDPASS = (1.0, 45.0)
MIN_FS = 100.0                     # need Nyquist comfortably above 45 Hz
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
LABEL_NAMES = {0: "Control", 1: "Dyslexic"}

# Scalp regions: lobe x hemisphere (odd = left, even = right) plus midline ("z") electrodes
REGIONS = ["F_L", "F_R", "C_L", "C_R", "T_L", "T_R", "P_L", "P_R", "O_L", "O_R", "MID"]
LOBES = ["F", "C", "T", "P", "O"]
_LOBE_OF_PREFIX = {"FP": "F", "AF": "F", "F": "F", "FC": "C", "C": "C", "FT": "T", "T": "T",
                   "TP": "T", "CP": "P", "P": "P", "PO": "O", "O": "O"}
_LEGACY = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}     # old 10-20 names -> 10-10 names
_NAME_RE = re.compile(r"([A-Z]{1,2})(\d{1,2}|Z)")


# ----------------------------------------------------------------------------- channel handling
def normalize_channel_name(name):
    """'EEG Fp1-REF' / 'FP1' / 'Fp1.' -> 'FP1'; old T3/T4/T5/T6 -> T7/T8/P7/P8; None if not a scalp EEG label."""
    s = str(name).strip().upper()
    s = re.sub(r"^EEG[\s_:\-]*", "", s)
    s = re.split(r"[\s_\-/.]", s)[0]
    s = _LEGACY.get(s, s)
    m = _NAME_RE.fullmatch(s)
    if m is None or m.group(1) not in _LOBE_OF_PREFIX:
        return None
    return s


def channel_region(canon):
    """Map a normalised electrode label to one of REGIONS."""
    prefix, idx = _NAME_RE.fullmatch(canon).groups()
    if idx == "Z":
        return "MID"
    return f"{_LOBE_OF_PREFIX[prefix]}_{'L' if int(idx) % 2 else 'R'}"


def _name_route(ch_names):
    """Recognise scalp electrodes by label (drops EOG/ECG/ACC/triggers/A1/A2/duplicates)."""
    keep, canon, dropped, seen = [], [], [], set()
    for i, name in enumerate(ch_names):
        c = normalize_channel_name(name)
        if c is None or c in seen:
            dropped.append(str(name))
            continue
        keep.append(i)
        canon.append(c)
        seen.add(c)
    return keep, canon, dropped


# Numbered / unnamed nets (EGI E1..E128, BioSemi A1.., custom caps) are placed by 3-D electrode position:
# each electrode is matched to the nearest 10-10 scalp site, which gives it a region. Electrodes far from any
# site (face, neck, ear sensors) are dropped.
_MAX_SITE_DISTANCE = 0.38      # unit-sphere distance to the nearest 10-10 site
_SITE_CACHE = {}
_SKIP_MONTAGES = ("standard", "fsaverage", "colin27", "spherical")     # 10-20-style names: handled by the name route


def _unit_sphere(xyz):
    """Least-squares sphere fit, then centre and scale the points to a unit sphere."""
    xyz = np.asarray(xyz, dtype=float)
    sol = np.linalg.lstsq(np.c_[2 * xyz, np.ones(len(xyz))], (xyz ** 2).sum(axis=1), rcond=None)[0]
    centre = sol[:3]
    return (xyz - centre) / np.sqrt(sol[3] + centre @ centre)


def _reference_sites():
    """10-10 scalp sites (without the lower 9/10 ring and inion sites) as (labels, unit-sphere coordinates)."""
    if "sites" not in _SITE_CACHE:
        import mne
        pos = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
        names = list(pos)
        xyz = _unit_sphere([pos[n] for n in names])
        keep = [i for i, n in enumerate(names) if normalize_channel_name(n) and not re.search(r"(9|10)$", n)]
        _SITE_CACHE["sites"] = ([normalize_channel_name(names[i]) for i in keep], xyz[keep])
    return _SITE_CACHE["sites"]


def _bare_name(name):
    """Strip recording-system decorations before matching a standard net: 'EEG A1-REF' -> 'A1', 'E12 ' -> 'E12'."""
    s = re.sub(r"^(EEG|EXG)[\s_:\-]*", "", str(name).strip(), flags=re.IGNORECASE)
    return re.split(r"[\s_/\-]", s)[0]


def _builtin_montage_positions(ch_names):
    """Find the built-in MNE montage (EGI GSN, BioSemi, EasyCap...) whose channel names best match ch_names.
    Names are compared after removing prefixes and reference suffixes, so 'EEG A1-REF' matches BioSemi 'A1'."""
    import mne
    wanted = set(_bare_name(n) for n in ch_names)
    best, best_score = None, 0.0
    for name in mne.channels.get_builtin_montages():
        if name.startswith(_SKIP_MONTAGES):
            continue
        montage = mne.channels.make_standard_montage(name)
        have = set(montage.ch_names)
        if len(wanted & have) < 0.8 * len(wanted):
            continue
        score = len(wanted & have) / len(wanted | have)
        if score > best_score:
            best, best_score = (name, montage.get_positions()["ch_pos"]), score
    return best


def _position_route(ch_names, ch_pos, min_regions):
    try:
        found = _builtin_montage_positions(ch_names)
        pos, method = None, None
        if found:
            bare = {n: _bare_name(n) for n in ch_names}
            pos = {n: found[1][bare[n]] for n in ch_names if bare[n] in found[1]}
            method = f"MNE montage '{found[0]}' (3-D electrode positions, matched by channel names)"
        elif ch_pos:
            usable = [n for n in ch_names if n in ch_pos and np.all(np.isfinite(ch_pos[n]))]
            if len(usable) >= 0.8 * len(ch_names):
                pos, method = ch_pos, "3-D electrode positions stored in the file (assumed x right, y front, z up)"
        if pos is None:
            return None
        idx = [i for i, n in enumerate(ch_names) if n in pos and np.all(np.isfinite(pos[n]))]
        if len(idx) < 16:                              # too few points for a stable sphere fit
            return None
        xyz = _unit_sphere([pos[ch_names[i]] for i in idx])
        site_names, site_xyz = _reference_sites()
    except ImportError:                                # MNE not installed
        return None
    dist_all = np.linalg.norm(xyz[:, None, :] - site_xyz[None, :, :], axis=-1)
    nearest, dist = dist_all.argmin(axis=1), dist_all.min(axis=1)
    ok = dist <= _MAX_SITE_DISTANCE
    keep = [i for i, o in zip(idx, ok) if o]
    canon = [site_names[k] for k, o in zip(nearest, ok) if o]
    if len({channel_region(c) for c in canon}) < min_regions:
        return None
    kept = set(keep)
    return keep, canon, [str(n) for i, n in enumerate(ch_names) if i not in kept], method


def resolve_channels(ch_names, ch_pos=None, min_regions=4):
    """Decide which rows are scalp EEG and which 10-10 site each belongs to.

    Returns (row indices to keep, site label per kept row, names dropped, how the decision was made).
    Order: channel labels first; if they do not identify the scalp (e.g. E1..E128), 3-D positions.
    """
    ch_names = [str(n) for n in ch_names]
    keep, canon, dropped = _name_route(ch_names)
    name_regions = {channel_region(c) for c in canon}
    if len(name_regions) >= min_regions and len(canon) >= 0.5 * len(ch_names):
        return keep, canon, dropped, "channel names"
    placed = _position_route(ch_names, ch_pos, min_regions)
    if placed is not None:
        return placed
    if len(name_regions) >= min_regions:
        return keep, canon, dropped, "channel names"
    if canon:
        raise ValueError(f"only {len(name_regions)} scalp regions could be identified from the channel names "
                         f"({sorted(name_regions)}); need at least {min_regions}. "
                         "Pass ch_names=[...] with standard 10-20/10-10 labels.")
    raise ValueError(f"None of the {len(ch_names)} channel names look like 10-20/10-10 electrode labels "
                     "(e.g. 'Fp1', 'C3', 'Oz'), and electrode positions could not be determined (standard nets "
                     "such as EGI GSN HydroCel or BioSemi are recognised by channel names when MNE is installed; "
                     "otherwise the file must store electrode coordinates). "
                     "Pass ch_names=[...] with a standard label for each row.")


# ----------------------------------------------------------------------------- auto-detection helpers
def to_microvolts(data, units="auto"):
    """Return (data in microvolts, description). 'auto' guesses volts vs microvolts from signal size."""
    if units == "uV":
        return data, "uV (given)"
    if units == "V":
        return data * 1e6, "V (given)"
    if units != "auto":
        raise ValueError("units must be 'auto', 'V' or 'uV'")
    med_sd = float(np.median(data.std(axis=1)))
    if med_sd < 1e-2:
        return data * 1e6, "V (auto-detected)"
    if med_sd < 1e4:
        return data, "uV (auto-detected)"
    return data, "unknown scale, e.g. raw ADC counts (features are scale-free, so this is harmless)"


def detect_mains(data, fs, ratio_threshold=3.0):
    """Return 50.0, 60.0 or None, based on a narrow spectral peak vs its neighbouring frequencies."""
    if fs < 110:
        return None
    freqs, psd = signal.welch(data, fs=fs, nperseg=int(min(data.shape[1], 4 * fs)), axis=-1)
    psd = np.median(psd, axis=0)
    best, best_ratio = None, ratio_threshold
    for f0 in (50.0, 60.0):
        if f0 + 1.0 >= fs / 2:
            continue
        line = (freqs >= f0 - 0.75) & (freqs <= f0 + 0.75)
        side = ((freqs >= f0 - 6) & (freqs <= f0 - 3)) | ((freqs >= f0 + 3) & (freqs <= f0 + 6))
        if not line.any() or not side.any():
            continue
        ratio = psd[line].max() / (np.median(psd[side]) + 1e-20)
        if ratio > best_ratio:
            best, best_ratio = f0, ratio
    return best


def find_bad_channels(data):
    """Scale-free: flat (<0.1% of median SD) or noisy (>5x median SD) channels."""
    sd = data.std(axis=1)
    med = np.median(sd)
    if med <= 0:
        return np.ones(len(sd), dtype=bool)
    return (sd < 1e-3 * med) | (sd > 5.0 * med)


def resample_to(data, fs, target=TARGET_FS):
    if abs(fs - target) < 1e-9:
        return data
    ratio = Fraction(target / fs).limit_denominator(1000)
    return signal.resample_poly(data, ratio.numerator, ratio.denominator, axis=-1)


def make_epochs(data, fs=TARGET_FS, epoch_s=EPOCH_S):
    n = int(round(epoch_s * fs))
    k = data.shape[1] // n
    if k == 0:
        raise ValueError(f"recording is shorter than one {epoch_s:g} s epoch after resampling")
    return data[:, : k * n].reshape(data.shape[0], k, n).transpose(1, 0, 2)


# ----------------------------------------------------------------------------- preprocessing
def preprocess_recording(data, fs, ch_names, units="auto", min_regions=4, ch_pos=None):
    """Raw (channels, samples) recording -> epochs (n_epochs, n_good_channels, 1000) + site labels + report.

    ch_pos: optional {channel name: (x, y, z)} used only if the names do not identify scalp positions.
    """
    report = {"warnings": []}
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"expected a 2-D array (channels, samples), got shape {data.shape}")
    if ch_names is None:
        raise ValueError("ch_names is required so channels can be mapped to scalp regions "
                         "(e.g. ['Fp1', 'Fp2', 'C3', ...]); files loaded with MNE supply them automatically")
    ch_names = list(ch_names)
    if data.shape[0] != len(ch_names):
        if data.shape[1] == len(ch_names):
            data = data.T
            report["warnings"].append("input looked like (samples, channels); transposed it")
        else:
            raise ValueError(f"data has shape {data.shape} but {len(ch_names)} channel names were given")
    if fs is None or fs < MIN_FS:
        raise ValueError(f"sampling rate {fs} Hz is too low (need >= {MIN_FS:g} Hz to cover the 1-45 Hz range)")
    report.update(input_fs=float(fs), input_channels=len(ch_names), input_seconds=data.shape[1] / fs)

    keep, canon, dropped, report["channel_mapping"] = resolve_channels(ch_names, ch_pos, min_regions)
    data = data[keep]
    orig = [ch_names[i] for i in keep]
    report["channels_dropped"] = dropped
    if not np.isfinite(data).all():
        raise ValueError("data contains NaN/inf; crop or interpolate those segments first")

    data, report["units"] = to_microvolts(data, units)
    data = data - data.mean(axis=1, keepdims=True)                   # remove DC offset

    mains = detect_mains(data, fs)
    report["mains_hz"] = mains
    if mains:
        b, a = signal.iirnotch(mains, Q=30.0, fs=fs)
        data = signal.filtfilt(b, a, data, axis=-1)
    data = resample_to(data, fs)
    sos = signal.butter(4, BANDPASS, btype="bandpass", fs=TARGET_FS, output="sos")
    data = signal.sosfiltfilt(sos, data, axis=-1)

    bad = find_bad_channels(data)
    report["bad_channels"] = [o for o, b in zip(orig, bad) if b]
    data = data[~bad]
    canon = [c for c, b in zip(canon, bad) if not b]
    if not canon:
        raise ValueError("no usable channels left after bad-channel detection")
    report["channels_used"] = list(canon)

    regions = sorted({channel_region(c) for c in canon})
    report["regions_covered"] = regions
    if len(regions) < min_regions:
        raise ValueError(f"only {len(regions)} scalp regions covered ({regions}); need at least {min_regions} "
                         "for a trustworthy result")

    if len(canon) >= 4:
        data = data - data.mean(axis=0, keepdims=True)               # common average reference
        report["reference"] = "average"
    else:
        report["reference"] = "unchanged"
        report["warnings"].append("fewer than 4 usable channels: average reference skipped")

    epochs = make_epochs(data)
    ptp = np.ptp(epochs, axis=-1).max(axis=1)
    keep = ptp <= 3.0 * np.median(ptp) if len(ptp) >= 3 else np.ones(len(ptp), dtype=bool)
    report["n_epochs_total"], report["n_epochs_used"] = int(len(keep)), int(keep.sum())
    report["low_confidence"] = bool(keep.sum() < 3)
    if report["low_confidence"]:
        report["warnings"].append("fewer than 3 clean 4-s epochs: features are unstable; treat the result as low-confidence")
    if report["input_seconds"] < 30:
        report["warnings"].append("recording shorter than 30 s: subject-level estimate will be noisy")
    return epochs[keep], canon, report


# ----------------------------------------------------------------------------- features
def _feature_names():
    names = []
    for r in REGIONS:
        names += [f"{r}_{b}" for b in BANDS] + [f"{r}_logTBR"]
    names += [f"asym_alpha_{lobe}" for lobe in LOBES]
    names += [f"global_{b}" for b in BANDS] + ["global_logTBR"]
    return names


FEATURE_NAMES = _feature_names()       # 11 regions x 6 + 5 asymmetries + 6 global = 77


def epoch_features(epochs, canon_names, fs=TARGET_FS):
    """(n_epochs, n_channels, n_samples) -> (n_epochs, 77). Uncovered regions become NaN."""
    freqs, psd = signal.welch(epochs, fs=fs, nperseg=int(2 * fs), axis=-1)
    df = freqs[1] - freqs[0]

    def band(lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        return psd[..., m].sum(axis=-1) * df

    total = band(*BANDPASS) + 1e-12
    rel = np.stack([band(lo, hi) / total for lo, hi in BANDS.values()], axis=-1)   # (ep, ch, 5)
    regions = np.array([channel_region(c) for c in canon_names])
    eps = 1e-6

    def block(rr):                                   # (ep, 5) -> (ep, 6): 5 relative powers + log theta/beta
        tbr = np.log10((rr[:, 1] + eps) / (rr[:, 3] + eps))
        return np.column_stack([rr, tbr])

    feats, alpha = [], {}
    for r in REGIONS:
        idx = np.flatnonzero(regions == r)
        rr = rel[:, idx, :].mean(axis=1) if idx.size else np.full((rel.shape[0], 5), np.nan)
        feats.append(block(rr))
        alpha[r] = rr[:, 2]
    feats.append(np.column_stack([np.log10((alpha[f"{lobe}_R"] + eps) / (alpha[f"{lobe}_L"] + eps))
                                  for lobe in LOBES]))
    feats.append(block(rel.mean(axis=1)))
    return np.concatenate(feats, axis=1)


# ----------------------------------------------------------------------------- inference
def _read_eeglab_epochs(path, mne):
    """Epoched EEGLAB .set -> one pseudo-continuous array (epochs joined end-to-end) + a warning note."""
    ep = mne.io.read_epochs_eeglab(path, verbose="ERROR")
    picks = mne.pick_types(ep.info, eeg=True)
    if len(picks) == 0:
        picks = np.arange(len(ep.ch_names))
    x = ep.get_data(picks=picks)                           # (n_epochs, n_channels, n_times), volts
    x = x - x.mean(axis=-1, keepdims=True)                 # per-epoch DC removal so joins have no steps
    sfreq = float(ep.info["sfreq"])
    data = x.transpose(1, 0, 2).reshape(x.shape[1], -1)
    note = (f"epoched file: {x.shape[0]} epochs of {x.shape[2] / sfreq:.2f} s were joined end-to-end into "
            f"{data.shape[1] / sfreq:.0f} s. The joins are not continuous EEG and the model was trained on "
            "continuous 4 s windows, so interpret this result with extra caution")
    return data, sfreq, [ep.ch_names[i] for i in picks], note


def load_recording(recording, fs=None, ch_names=None, units="auto"):
    """Accept an array, a .npy path, or any file MNE can read (EDF/BDF/FIF/BrainVision/EEGLAB...).

    Returns (data, fs, ch_names, units, notes); notes are warnings to attach to the report.
    """
    if isinstance(recording, (str, Path)):
        path = Path(recording)
        if path.suffix.lower() == ".npy":
            recording = np.load(path)
        else:
            try:
                import mne
            except ImportError as exc:
                raise ImportError("reading EEG files needs MNE: pip install mne") from exc
            try:
                raw = mne.io.read_raw(path, preload=True, verbose="ERROR")
            except Exception as exc:
                msg = str(exc).lower()
                if path.suffix.lower() == ".set" and ("epoch" in msg or "trials" in msg):   # epoched EEGLAB file
                    try:
                        data, sfreq, names, note = _read_eeglab_epochs(path, mne)
                    except Exception as exc2:
                        raise ValueError(f"{path.name} is an epoched EEGLAB file and could not be read "
                                         f"({exc2}). Try exporting continuous data from EEGLAB.") from exc2
                    return data, sfreq, (ch_names or names), "V", [note]
                hint = " For .xdf use pyxdf and pass an array together with fs and ch_names." \
                    if path.suffix.lower() == ".xdf" else ""
                raise ValueError(f"could not read {path.name} with MNE ({exc}).{hint}") from exc
            picks = mne.pick_types(raw.info, eeg=True)
            if len(picks) == 0:
                picks = np.arange(len(raw.ch_names))
            names = [raw.ch_names[i] for i in picks]
            return raw.get_data(picks=picks), float(raw.info["sfreq"]), (ch_names or names), "V", []
    return np.asarray(recording, dtype=float), fs, ch_names, units, []


def format_report(res):
    r = res["report"]
    mains = f"{r['mains_hz']:g} Hz notch" if r["mains_hz"] else "no mains noise detected"
    lines = [
        f"Prediction: {res['prediction']} (confidence {res['confidence']:.1%}), "
        f"{r['n_epochs_used']}/{r['n_epochs_total']} epochs used",
        f"  input: {r['input_channels']} ch @ {r['input_fs']:g} Hz, {r['input_seconds']:.0f} s | units: {r['units']}",
        f"  {mains} | resampled to {TARGET_FS:g} Hz | band-pass {BANDPASS[0]:g}-{BANDPASS[1]:g} Hz | "
        f"reference: {r['reference']}",
        f"  channels used ({len(r['channels_used'])}): "
        + (f"{', '.join(r['channels_used'][:12])}, ... (10-10 sites assigned)" if len(r["channels_used"]) > 24
           else ", ".join(r["channels_used"])),
        f"  channel placement: {r.get('channel_mapping', 'channel names')}",
        f"  regions covered ({len(r['regions_covered'])}/{len(REGIONS)}): {', '.join(r['regions_covered'])}",
    ]
    if r["channels_dropped"]:
        lines.append(f"  ignored (non-scalp / unrecognised, {len(r['channels_dropped'])}): "
                     f"{', '.join(r['channels_dropped'])}")
    if r["bad_channels"]:
        lines.append(f"  excluded as bad: {', '.join(r['bad_channels'])}")
    lines += [f"  WARNING: {w}" for w in r["warnings"]]
    return "\n".join(lines)


def predict_dyslexia(recording, fs=None, ch_names=None, units="auto", model_dir=None, verbose=True):
    """Classify one recording of any length / montage / sampling rate.

    recording : array (channels, samples) | path to .npy | path to an EEG file readable by MNE
    fs, ch_names : required for arrays/.npy, read automatically from MNE-readable files
    Returns prediction, confidence (mean over epochs) and a report of every automatic decision.
    """
    model_dir = Path(model_dir) if model_dir else Path(__file__).resolve().parent
    with open(model_dir / "cleaning_model.pkl", "rb") as fh:
        cleaner = pickle.load(fh)
    with open(model_dir / "identification_model.pkl", "rb") as fh:
        clf = pickle.load(fh)

    data, fs, ch_names, units, notes = load_recording(recording, fs, ch_names, units)
    epochs, canon, report = preprocess_recording(data, fs, ch_names, units)
    report["warnings"] = notes + report["warnings"]
    proba = clf.predict_proba(cleaner.transform(epoch_features(epochs, canon)))
    mean_p = proba.mean(axis=0)
    classes = [int(c) for c in clf.classes_]
    best = int(np.argmax(mean_p))
    result = {
        "prediction": LABEL_NAMES[classes[best]],
        "confidence": float(mean_p[best]),
        "probabilities": {LABEL_NAMES[c]: float(p) for c, p in zip(classes, mean_p)},
        "epoch_probabilities_dyslexic": [round(float(p), 3) for p in proba[:, classes.index(1)]],
        "report": report,
    }
    if verbose:
        print(format_report(result))
    return result
