"""Reviewer comment 4: preprocessing and montage harmonisation on REAL public EEG (no dyslexia labels).

R1  audit of 13 real recordings in 11 vendor formats (MNE testing data, github.com/mne-tools/mne-testing-data):
    reading, channel mapping route and success, acceptance, detected units/mains/bad channels, epochs, feature
    completeness - with the released loader, and with electrode positions read from the file.
R2  harmonisation consistency on real signals: each accepted recording is transformed (relabelled, rescaled,
    resampled, mains added, reduced to sparse montages) and its features and model output are compared with
    the original. This tests engineering interoperability only; it says nothing about predictive validity.
"""
import pickle, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, mne
from scipy import signal
import eeg_adaptive as eeg
import revision_sim as rs

warnings.filterwarnings("ignore"); mne.set_log_level("ERROR")
# speed only: memoise MNE's standard-montage construction (identical results; the position route rebuilds
# every built-in montage on each call otherwise)
from functools import lru_cache
mne.channels.make_standard_montage = lru_cache(maxsize=None)(mne.channels.make_standard_montage)
D = Path(sys.argv[1]); RES = Path("results_r4_real"); RES.mkdir(exist_ok=True)
FILES = [  # path, reader, description
    ("MEG/sample/sample_audvis_trunc_raw.fif", mne.io.read_raw_fif, "Elekta/MNE sample, 60-ch EEG cap, labels 'EEG 0xx' + digitised positions"),
    ("EGI/test_egi.mff", mne.io.read_raw_egi, "EGI GSN HydroCel 129 (MFF)"),
    ("EEGLAB/test_raw.set", mne.io.read_raw_eeglab, "EEGLAB .set, 32 ch, labels 'EEG 0xx', no positions"),
    ("NihonKohden/MB0400FU.EEG", mne.io.read_raw_nihon, "Nihon Kohden clinical 10-20 (21E)"),
    ("NihonKohden/DA00100E.EEG", mne.io.read_raw_nihon, "Nihon Kohden clinical 10-20, 2 s"),
    ("nedf/testdata.nedf", mne.io.read_raw_nedf, "Neuroelectrics Enobio 32 ch (NEDF)"),
    ("curry/c,rfDC Curry 8.cdt", mne.io.read_raw_curry, "Compumedics Curry 8, bipolar-to-POz labels"),
    ("eximia/test_eximia.nxe", mne.io.read_raw_eximia, "Nexstim eXimia 60 ch"),
    ("GDF/test_gdf_2.20.gdf", mne.io.read_raw_gdf, "GDF 2.20, 64 ch 10-10"),
    ("antio/CA_208/test_CA_208.cnt", mne.io.read_raw_ant, "ANT Neuro eego (.cnt), 64 ch"),
    ("EDF/test_reduced.edf", mne.io.read_raw_edf, "EDF, BioSemi-style labels A1-D32"),
    ("BDF/test_bdf_stim_channel.bdf", mne.io.read_raw_bdf, "BioSemi BDF, 3 EEG ch"),
    ("Brainvision/test_NO.vhdr", mne.io.read_raw_brainvision, "BrainVision, 65 ch labelled 1-65, 0.4 s"),
]


def run_pipeline(data, fs, names, pos=None):
    try:
        ep, canon, rep = eeg.preprocess_recording(data, fs, names, ch_pos=pos)
        x = eeg.epoch_features(ep, canon)
        return dict(accepted=True, route=rep["channel_mapping"], mapped=len(canon) + len(rep["bad_channels"]),
                    used=len(canon), regions=len(rep["regions_covered"]), units=rep["units"].split()[0],
                    mains=rep["mains_hz"], bad=len(rep["bad_channels"]), epochs=rep["n_epochs_used"],
                    completeness=float(np.isfinite(x).all(axis=0).mean()), reason="", x=x, canon=canon,
                    low_conf=rep["low_confidence"])
    except ValueError as e:
        return dict(accepted=False, reason=str(e).split(".")[0][:90], x=None)


rows, keep = [], {}
for rel, reader, desc in FILES:
    path = D / rel
    row = dict(file=rel.split("/")[-1], format=desc)
    try:
        data, fs, names, units, notes = eeg.load_recording(path)          # released loader
        row["released_loader"] = "ok"
    except Exception as e:
        row["released_loader"] = "fails: " + str(e).split("(")[0][:60]
        data = None
    raw = reader(path, preload=True)
    picks = mne.pick_types(raw.info, eeg=True)
    picks = picks if len(picks) else np.arange(len(raw.ch_names))
    names = [raw.ch_names[i] for i in picks]
    data2 = raw.get_data(picks=picks)
    fs = float(raw.info["sfreq"])
    mon = raw.get_montage()
    pos = None
    if mon is not None:
        cp = mon.get_positions()["ch_pos"]
        pos = {n: cp[n] for n in names if n in cp and np.isfinite(cp[n]).all()} or None
    row.update(fs=fs, seconds=round(data2.shape[1] / fs, 1), eeg_channels=len(names), file_positions=0 if pos is None else len(pos))
    a = run_pipeline(data2, fs, names)                   # names only (as the released loader does)
    b = run_pipeline(data2, fs, names, pos) if pos else a
    for tag, r in [("names", a), ("names+pos", b)]:
        row[f"{tag}: accepted"] = r["accepted"]
        if r["accepted"]:
            row[f"{tag}: route"] = r["route"]; row[f"{tag}: mapped"] = f"{r['mapped']}/{len(names)}"
            row[f"{tag}: regions"] = r["regions"]; row[f"{tag}: completeness"] = round(r["completeness"], 3)
        else:
            row[f"{tag}: reason"] = r["reason"]
    if b["accepted"]:
        row.update(units=b["units"], mains=b["mains"], bad_channels=b["bad"], epochs=b["epochs"], low_confidence=b["low_conf"])
        keep[rel] = dict(data=data2, fs=fs, names=names, pos=pos, ref=b)
    rows.append(row)
r1 = pd.DataFrame(rows)
r1.to_csv(RES / "R1_real_file_audit.csv", index=False)
OUT = ["=== R1 Real-file audit (13 public recordings, 11 formats) ===", r1.to_string()]
print(OUT[-1], flush=True)

# ------------------------------------------------------------------ R2 harmonisation consistency on real signals
model = None
mp = Path("results_v3/cache")
# LR trained on the v2 main cohort (seed 0) - used only to measure output stability, not accuracy
cands = sorted(mp.glob("v2_nmm_0.60_s0_*.pkl"), key=lambda f: -f.stat().st_size)  # the full (n=360) cohort
if cands:
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    c = pickle.loads(cands[0].read_bytes())
    X = np.vstack([x for x in c["feats"].values() if x is not None])
    y = np.concatenate([[c["meta"].at[s, "label"]] * len(x) for s, x in c["feats"].items() if x is not None])
    model = Pipeline([("i", SimpleImputer(strategy="median", keep_empty_features=True)), ("s", StandardScaler()),
                      ("c", LogisticRegression(max_iter=3000, class_weight="balanced"))]).fit(X, y)

LEG = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
CONS = ["FP1", "FP2", "C3", "C4", "P7", "P8", "O1", "O2"]
rng = np.random.default_rng(0)


def transforms(rec):
    d, fs, names, pos = rec["data"], rec["fs"], rec["names"], rec["pos"]
    canon_of = {n: eeg.normalize_channel_name(n) for n in names}
    yield "relabel: 'EEG X-REF' style", d, fs, [f"EEG {n}-REF" for n in names], pos and {f"EEG {k}-REF": v for k, v in pos.items()}
    yield "relabel: upper case", d, fs, [n.upper() for n in names], pos and {k.upper(): v for k, v in pos.items()}
    yield "relabel: legacy T3-T6", d, fs, [LEG.get(canon_of[n], n) if canon_of[n] in LEG else n for n in names], pos
    yield "units: stored in microvolts", d * 1e6, fs, names, pos
    for new in (128, 512):
        if abs(new - fs) > 1:
            from fractions import Fraction
            fr = Fraction(new, int(round(fs))).limit_denominator(200)
            yield f"resample to {new} Hz", signal.resample_poly(d, fr.numerator, fr.denominator, axis=-1), \
                fs * fr.numerator / fr.denominator, names, pos
    t = np.arange(d.shape[1]) / fs
    yield "add 50 Hz mains (10 uV)", d + 10e-6 * np.sin(2 * np.pi * 50 * t), fs, names, pos
    used = rec["ref"]["canon"]
    site = {}
    for i, n in enumerate(names):
        c = canon_of[n]
        if c in CONS and c not in site:
            site[c] = i
    if len(site) >= 6:
        idx = sorted(site.values())
        yield f"consumer subset ({len(idx)} of the 8 sites)", d[idx], fs, [names[i] for i in idx], pos
    for k in range(5):
        idx = sorted(rng.choice(len(names), max(4, len(names) // 2), replace=False))
        yield f"random 50% channel subset #{k+1}", d[idx], fs, [names[i] for i in idx], pos and {names[i]: pos[names[i]] for i in idx if names[i] in pos}
    for k in range(5):
        idx = sorted(rng.choice(len(names), min(8, len(names)), replace=False))
        yield f"random 8-channel subset #{k+1}", d[idx], fs, [names[i] for i in idx], pos and {names[i]: pos[names[i]] for i in idx if names[i] in pos}


rows = []
for rel, rec in keep.items():
    x0 = np.nanmean(rec["ref"]["x"], axis=0)
    p0 = float(model.predict_proba(rec["ref"]["x"])[:, 1].mean()) if model is not None else np.nan
    for name, d, fs, names, pos in transforms(rec):
        r = run_pipeline(d, fs, names, pos)
        row = dict(file=rel.split("/")[-1], transform=name, accepted=r["accepted"])
        row["low_confidence_original"] = rec["ref"]["low_conf"]
        if r["accepted"]:
            x1 = np.nanmean(r["x"], axis=0)
            ok = np.isfinite(x0) & np.isfinite(x1)
            g = [eeg.FEATURE_NAMES.index(f"global_{b}") for b in ("delta", "theta", "alpha", "beta", "gamma")]
            row.update(completeness=float(np.isfinite(r["x"]).all(axis=0).mean()),
                       r_features=float(np.corrcoef(x0[ok], x1[ok])[0, 1]),
                       max_abs_diff_global_relpower=float(np.max(np.abs(x0[g] - x1[g]))),
                       abs_diff_p=abs(float(model.predict_proba(r["x"])[:, 1].mean()) - p0) if model is not None else np.nan)
        else:
            row["reason"] = r["reason"]
        rows.append(row)
    print("R2 done:", rel, flush=True)
r2 = pd.DataFrame(rows)
r2.to_csv(RES / "R2_real_harmonisation_raw.csv", index=False)
r2["family"] = r2["transform"].str.replace(r" #\d+", "", regex=True).str.replace(r"\(\d+ of the 8 sites\)", "", regex=True)
agg = r2.groupby("family").agg(n=("accepted", "size"), accepted=("accepted", "mean"),
                               completeness=("completeness", "mean"), r_features_median=("r_features", "median"),
                               r_features_min=("r_features", "min"),
                               global_relpower_maxdiff_median=("max_abs_diff_global_relpower", "median"),
                               model_output_abs_change_median=("abs_diff_p", "median"),
                               model_output_abs_change_max=("abs_diff_p", "max"))
agg.to_csv(RES / "R2_real_harmonisation.csv")
OUT += ["\n=== R2 Harmonisation consistency on real recordings (vs. the original recording) ===",
        f"recordings: {[k.split('/')[-1] for k in keep]}", agg.round(4).to_string()]
print("\n".join(OUT[-3:]), flush=True)
(RES / "summary_real.txt").write_text("\n".join(OUT))
