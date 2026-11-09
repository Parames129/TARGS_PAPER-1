"""Second-round revision experiments (reviewer comments 1-3 and Major Comment 2).

X1  effect sweep under the independent generator v3 ('nmm' and 'aperiodic' effects)
X2  interoperability under v3: leave-one-montage-out, leave-one-sampling-rate-out, unseen EGI-128
X3  cross-generator transfer v2 <-> v3
X4  operational limits of montage compatibility (mapping success, acceptance, feature completeness,
    prediction) for v2 and v3, on the same subjects re-analysed under each configuration
"""
import hashlib, json, os, pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import eeg_adaptive as eeg
import revision_sim as rs
import revision_sim_v3 as v3

QUICK = os.environ.get("QUICK") == "1"
N = 120 if QUICK else 360
SEEDS = [0] if QUICK else [0, 1, 2]
NJ = 2
LF = str(Path("leadfield_sample.npz").resolve())
RES = Path("results_v3"); CACHE = RES / "cache"; CACHE.mkdir(parents=True, exist_ok=True)
FS = [128, 250, 256, 500, 512, 1000]; DUR = [24, 32, 40]
V2_MAIN, V3_MAIN = 0.60, 0.12
VAR = "full|region_rel"
OUT = []


def log(title, obj):
    s = f"\n=== {title} ===\n" + (obj.to_string() if hasattr(obj, "to_string") else str(obj))
    print(s, flush=True); OUT.append(s)
    (RES / "summary_v3.txt").write_text("\n".join(OUT))


# ------------------------------------------------------------------ cohorts
def _worker(t):
    return v3.subject_worker_v3(t) if t["sim"] == "v3" else rs._subject_worker(t)


def cohort(seed, sim, eff, n=N, etype="nmm", kind=None, net="1020", fs=FS, dur=DUR, off=0, tag=""):
    key = hashlib.md5(json.dumps([seed, sim, eff, n, etype, kind, net, fs, dur, off, v3.SIM3, rs.SIM2],
                                 default=str).encode()).hexdigest()[:12]
    path = CACHE / f"{tag}{sim}_{etype}_{eff:.2f}_s{seed}_{key}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    rng = np.random.default_rng([seed, 7919])
    labels = np.array([0] * (n // 2) + [1] * (n - n // 2)); rng.shuffle(labels)
    tasks = []
    for i in range(n):
        if net == "egi":
            k, canon, sensors = "egi128", list(rs.BASE_CANON), list(rs.EGI_NAMES)
        else:
            k, canon = rs.random_montage(rng, kind); sensors = canon
        tasks.append(dict(sid=off + i, label=int(labels[i]), kind=k, canon=canon, sensors=sensors, net=net, sim=sim,
                          effect=eff, effect_type=etype, fs=int(rng.choice(fs)), dur=int(rng.choice(dur)),
                          plan={"full": ["region_rel"]}, lf_path=LF, p=rs.SIM2,
                          ss=np.random.SeedSequence([seed, 104729, i]).entropy))
    t0 = time.time()
    outs = Parallel(n_jobs=NJ)(delayed(_worker)(t) for t in tasks)
    meta = pd.DataFrame([o["meta"] for o in outs]).set_index("sid")
    feats = {o["meta"]["sid"]: o["feats"][VAR] for o in outs}
    c = dict(meta=meta, feats=feats)
    path.write_bytes(pickle.dumps(c))
    print(f"  built {tag}{sim}/{etype} eff={eff} seed={seed} n={n} {net} in {time.time()-t0:.0f}s", flush=True)
    return c


def model(name, seed=0):
    clf = (RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced", n_jobs=NJ, random_state=seed)
           if name == "RF" else LogisticRegression(max_iter=3000, class_weight="balanced"))
    return Pipeline([("i", SimpleImputer(strategy="median", keep_empty_features=True)), ("s", StandardScaler()), ("c", clf)])


def stack(c, sids):
    X, y, g = [], [], []
    for s in sids:
        x = c["feats"].get(s)
        if x is None:
            continue
        X.append(x); y += [int(c["meta"].at[s, "label"])] * len(x); g += [s] * len(x)
    return (np.vstack(X), np.array(y), np.array(g)) if X else (np.empty((0, 77)), np.array([]), np.array([]))


def fitpred(ctr, str_, cte, ste, name="RF", seed=0):
    X, y, _ = stack(ctr, str_)
    m = model(name, seed).fit(X, y)
    Xt, yt, gt = stack(cte, ste)
    p = pd.Series(np.nan, index=pd.Index(ste))
    if len(Xt):
        sp = pd.Series(m.predict_proba(Xt)[:, 1]).groupby(gt).mean()
        p.loc[sp.index] = sp.values
    return p


def oof(c, seed, name="RF"):
    meta = c["meta"]; sids = meta.index.to_numpy()
    p = pd.Series(np.nan, index=meta.index)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(sids, meta["label"]):
        pp = fitpred(c, sids[tr], c, sids[te], name, seed); p.loc[pp.index] = pp.values
    return p


def met(y, p):
    y = np.asarray(y, int); p = np.asarray(p, float); ok = np.isfinite(p)
    pi = np.where(ok, p, 0.5); pred = np.where(ok, (pi > 0.5).astype(int), 1 - y)
    return dict(n=len(y), coverage=ok.mean(), acc=accuracy_score(y, pred),
                auc=roc_auc_score(y, pi) if len(set(y)) == 2 else np.nan)


msd = lambda x: f"{np.nanmean(x):.3f} ± {np.nanstd(x, ddof=1) if len(x) > 1 else 0:.3f}"


def cohens_d(c):
    j = eeg.FEATURE_NAMES.index("global_logTBR")
    v = pd.Series({s: np.nanmean(x[:, j]) for s, x in c["feats"].items() if x is not None})
    y = c["meta"]["label"].loc[v.index]
    a, b = v[y == 1], v[y == 0]
    sp = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp


# ------------------------------------------------------------------ X1 effect sweep under v3
rows = []
for etype, grid in [("nmm", [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]), ("aperiodic", [0.0, 0.50, 1.00, 1.50])]:
    for eff in (grid if not QUICK else grid[::2]):
        for seed in SEEDS:
            c = cohort(seed, "v3", eff, etype=etype if eff > 0 else "nmm")   # eff = 0 is the same null cohort
            for name in ("RF", "LR"):
                m = met(c["meta"]["label"], oof(c, seed, name))
                rows.append(dict(effect_type=etype, eff=eff, seed=seed, model=name, **m, d_logTBR=cohens_d(c),
                                 refused=(c["meta"].get("error").notna().mean() if "error" in c["meta"] else 0.0)))
x1 = pd.DataFrame(rows); x1.to_csv(RES / "X1_v3_sweep_raw.csv", index=False)
log("X1 Independent generator v3: subject 5-fold CV (mean ± SD over seeds)",
    x1.groupby(["effect_type", "eff", "model"]).agg(acc=("acc", msd), auc=("auc", msd), d_logTBR=("d_logTBR", msd)))

# feature direction under v3 vs v2 (does the generator move the same features the same way?)
dirs = []
for sim, eff, et in [("v2", V2_MAIN, "nmm"), ("v3", V3_MAIN, "nmm"), ("v3", 1.00, "aperiodic")]:
    c = cohort(SEEDS[0], sim, eff, etype=et)
    F = pd.DataFrame({s: np.nanmean(x, 0) for s, x in c["feats"].items() if x is not None}).T
    F.columns = eeg.FEATURE_NAMES
    y = c["meta"]["label"].loc[F.index]
    for f in ["global_delta", "global_theta", "global_alpha", "global_beta", "global_gamma", "global_logTBR",
              "P_L_theta", "P_L_alpha", "T_L_theta", "O_L_alpha", "asym_alpha_P"]:
        a, b = F.loc[y == 1, f], F.loc[y == 0, f]
        sp = np.sqrt((a.var() + b.var()) / 2)
        dirs.append(dict(generator=f"{sim}/{et}/{eff}", feature=f, d=(a.mean() - b.mean()) / sp))
x1b = pd.DataFrame(dirs).pivot(index="feature", columns="generator", values="d")
x1b.to_csv(RES / "X1b_feature_effects.csv")
log("X1b Cohen's d (dyslexic - control) of key features under each generator", x1b.round(2))

# ------------------------------------------------------------------ X2 interoperability under v3
def heldout_vs_seen(c, mask, seed, name):
    meta = c["meta"]; r = np.random.default_rng(seed); out = []
    ks = meta.index[mask].to_numpy(); others = meta.index[~mask].to_numpy()
    if len(ks) < 6:
        return out
    perm = r.permutation(ks)
    for T in (perm[: len(perm) // 2], perm[len(perm) // 2:]):
        pool = np.setdiff1d(meta.index.to_numpy(), T)
        seen = r.choice(pool, size=min(len(others), len(pool)), replace=False)
        y = meta.loc[T, "label"]
        mu, ms = met(y, fitpred(c, others, c, T, name, seed)), met(y, fitpred(c, seen, c, T, name, seed))
        out.append(dict(unseen_acc=mu["acc"], seen_acc=ms["acc"], unseen_auc=mu["auc"], seen_auc=ms["auc"]))
    return out


rows = []
for seed in SEEDS:
    c = cohort(seed, "v3", V3_MAIN)
    meta = c["meta"]
    for name in ("RF", "LR"):
        for k in ["full19", "openbci8", "random"]:
            for r_ in heldout_vs_seen(c, (meta["kind"] == k).to_numpy(), seed, name):
                rows.append(dict(seed=seed, model=name, test="montage", held=k, **r_))
        for f in FS:
            for r_ in heldout_vs_seen(c, (meta["fs"] == f).to_numpy(), seed, name):
                rows.append(dict(seed=seed, model=name, test="sampling rate", held=str(f), **r_))
        egi = cohort(seed + 500, "v3", V3_MAIN, n=max(40, N // 4), net="egi", off=100000, tag="egi_")
        cv = met(meta["label"], oof(c, seed, name))
        mu = met(egi["meta"]["label"], fitpred(c, meta.index, egi, egi["meta"].index, name, seed))
        rows.append(dict(seed=seed, model=name, test="EGI-128", held="egi128", unseen_acc=mu["acc"],
                         seen_acc=cv["acc"], unseen_auc=mu["auc"], seen_auc=cv["auc"],
                         egi_mapped=egi["meta"]["n_mapped"].mean(), egi_regions=egi["meta"]["regions_used"].mean()))
x2 = pd.DataFrame(rows); x2.to_csv(RES / "X2_v3_interop_raw.csv", index=False)
x2["d_acc"] = x2.unseen_acc - x2.seen_acc
tab = x2.groupby(["model", "test", "held"]).agg(unseen_acc=("unseen_acc", msd), seen_acc=("seen_acc", msd),
                                                 d_acc=("d_acc", msd), unseen_auc=("unseen_auc", msd))
pv = {}
for key, d in x2[x2.test != "EGI-128"].groupby(["model", "test", "held"]):
    pv[key] = stats.ttest_rel(d.unseen_acc, d.seen_acc).pvalue if len(d) > 2 and d.d_acc.std() > 0 else np.nan
tab["p_paired"] = [pv.get(i, np.nan) for i in tab.index]
log("X2 Interoperability under v3 (same test subjects, size-matched training; mean ± SD over seeds x halves)", tab)
if "egi_mapped" in x2:
    log("X2b v3 EGI-128 mapping", x2[x2.test == "EGI-128"][["seed", "model", "egi_mapped", "egi_regions"]])

# ------------------------------------------------------------------ X3 cross-generator transfer
rows = []
for seed in SEEDS:
    # independent subjects per generator: cohorts that share a seed share their random subjects
    c2, c3 = cohort(seed, "v2", V2_MAIN), cohort(seed + 1000, "v3", V3_MAIN, off=200000)
    c3a = cohort(seed + 2000, "v3", 1.00, etype="aperiodic", off=300000)
    for name in ("RF", "LR"):
        for trn, ctr, tst, cte in [("v2", c2, "v3-nmm", c3), ("v3-nmm", c3, "v2", c2), ("v2", c2, "v3-aperiodic", c3a),
                                   ("v3-nmm", c3, "v3-aperiodic", c3a)]:
            m = met(cte["meta"]["label"], fitpred(ctr, ctr["meta"].index, cte, cte["meta"].index, name, seed))
            rows.append(dict(seed=seed, model=name, train=trn, test=tst, **m))
        for nm, cc in [("v2", c2), ("v3-nmm", c3), ("v3-aperiodic", c3a)]:
            m = met(cc["meta"]["label"], oof(cc, seed, name))
            rows.append(dict(seed=seed, model=name, train=nm, test=nm + " (5-fold CV)", **m))
x3 = pd.DataFrame(rows); x3.to_csv(RES / "X3_cross_generator_raw.csv", index=False)
log("X3 Cross-generator transfer (train on all subjects of one generator, test on another; within = 5-fold CV)",
    x3.groupby(["model", "train", "test"]).agg(acc=("acc", msd), auc=("auc", msd)))

# ------------------------------------------------------------------ X4 operational limits
CONFIGS = [("Full 10-20 (19 ch)", list(rs.BASE_CANON), "plain"),
           ("Legacy labels T3-T6 (19 ch)", list(rs.BASE_CANON), "legacy"),
           ("Reference-style labels 'EEG Fp1-REF'", list(rs.BASE_CANON), "ref"),
           ("Consumer 8 ch", list(rs.OPENBCI8_CANON), "plain")]
for k in [3, 4, 6, 8, 12, 16]:
    CONFIGS += [(f"Random {k} ch", k, "plain")]
CONFIGS += [("Generic labels Ch1-Ch8, no positions", list(rs.OPENBCI8_CANON), "generic")]


def x4_worker(task):
    rng = np.random.default_rng(task["ss"])
    if task["sim"] == "v3":
        sig, _ = v3.simulate_v3(task["label"], rs.BASE_CANON, task["fs"], task["dur"], rng, task["effect"], lf_path=LF)
    else:
        sig, _ = rs.simulate_v2(task["label"], rs.BASE_CANON, task["fs"], task["dur"], rng, task["effect"])
    out = []
    for cname, chans, style in task["configs"]:
        idx = [rs.BASE_CANON.index(c) for c in chans]
        if style == "generic":
            names = [f"Ch{i+1}" for i in range(len(idx))]
        else:
            names, _ = rs.style_names(chans, rng, style)
        mapped, accepted, compl, x = 0, False, 0.0, None
        try:
            keep, canon, _, _ = eeg.resolve_channels(names, None, 1)
            mapped = len(keep)
        except ValueError:
            mapped = 0
        try:
            ep, canon, rep = rs.prep(sig[idx], task["fs"], names, rs.PREP_CFGS["full"])
            x = rs.region_features(ep, canon); accepted = True
            compl = float(np.isfinite(x).all(axis=0).mean())
        except ValueError:
            pass
        out.append(dict(sid=task["sid"], label=task["label"], config=cname, n_ch=len(idx), mapped=mapped / len(idx),
                        accepted=accepted, completeness=compl, x=x))
    return out


rows = []
for sim, eff in [("v2", V2_MAIN), ("v3", V3_MAIN)]:
    for seed in SEEDS:
        train = cohort(seed, sim, eff)
        Xtr, ytr, _ = stack(train, train["meta"].index)
        mdl = {nm: model(nm, seed).fit(Xtr, ytr) for nm in ("RF", "LR")}
        r = np.random.default_rng([seed, 4242])
        n_te = 60 if QUICK else 120
        labels = r.permutation([0] * (n_te // 2) + [1] * (n_te - n_te // 2))
        tasks = []
        for i in range(n_te):
            cfg = []
            for cname, chans, style in CONFIGS:
                if isinstance(chans, int):
                    for d_ in range(3):
                        cfg.append((cname, [rs.BASE_CANON[j] for j in sorted(r.choice(19, chans, replace=False))], style))
                else:
                    cfg.append((cname, chans, style))
            tasks.append(dict(sim=sim, sid=400000 + i, label=int(labels[i]), fs=int(r.choice(FS)), dur=int(r.choice(DUR)),
                              effect=eff, configs=cfg, ss=np.random.SeedSequence([seed, 5151, i]).entropy))
        res = Parallel(n_jobs=NJ)(delayed(x4_worker)(t) for t in tasks)
        for subj in res:
            for o in subj:
                pr = {nm: (float(m.predict_proba(o["x"])[:, 1].mean()) if o["x"] is not None else np.nan)
                      for nm, m in mdl.items()}
                rows.append(dict(sim=sim, seed=seed, sid=o["sid"], label=o["label"], config=o["config"], n_ch=o["n_ch"],
                                 mapped=o["mapped"], accepted=o["accepted"], completeness=o["completeness"],
                                 p_RF=pr["RF"], p_LR=pr["LR"]))
x4 = pd.DataFrame(rows); x4.to_csv(RES / "X4_operational_limits_raw.csv", index=False)


def x4agg(d):
    acc = d[d.accepted]
    out = dict(n=len(d), channel_mapping=d.mapped.mean(), acceptance=d.accepted.mean(),
               feature_completeness=acc.completeness.mean() if len(acc) else np.nan)
    for nm in ("RF", "LR"):
        if len(acc) and acc.label.nunique() == 2:
            out[f"acc_{nm}_accepted"] = accuracy_score(acc.label, acc[f"p_{nm}"] > 0.5)
            out[f"auc_{nm}_accepted"] = roc_auc_score(acc.label, acc[f"p_{nm}"])
        out[f"acc_{nm}_ITT"] = met(d.label, d[f"p_{nm}"])["acc"]
    return pd.Series(out)


order = [c[0] for c in CONFIGS]
x4t = x4.groupby(["sim", "config"]).apply(x4agg)
if isinstance(x4t, pd.Series):
    x4t = x4t.unstack()
x4t = x4t.reset_index()
x4t["o"] = x4t.config.map({c: i for i, c in enumerate(order)})
x4t = x4t.sort_values(["sim", "o"]).drop(columns="o").set_index(["sim", "config"])
x4t.to_csv(RES / "X4_operational_limits.csv")
log("X4 Operational limits: channel-mapping success, recording acceptance, feature completeness (accepted), "
    "prediction (accepted and ITT); same test subjects re-analysed under each configuration", x4t.round(3))
print("DONE", flush=True)
