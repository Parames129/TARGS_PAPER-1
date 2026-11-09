"""Third-round revision analyses (reviewer 2).

A  reference choice with a recorded (Cz) reference, and the age / alpha-peak confound:
   IAF-adjusted bands and age regression inside the training folds          (major 5)
B  narrower volume conduction (lambda 0.20, 0.30 vs 0.45): unseen montages (E3) and electrode
   placement (E7)                                                           (major 4)
C  compositional features: drop one band per region; grouped permutation importance (major 6)
D  null cohorts: 20 per simulator                                           (null check)
E  mismatch effects: alpha-peak slowing, source coupling, effect between region boundaries
Usage: python run_round3.py A B C   (any subset)
"""
import hashlib, json, pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import eeg_adaptive as eeg
import revision_sim as rs
import round3_lib as L

PARTS = set(sys.argv[1:]) or set("ABCDE")
import os
QUICK = os.environ.get("QUICK") == "1"
N, NJ = (40 if QUICK else 360), 2
FS = [128, 250, 256, 500, 512, 1000]; DUR = [24, 32, 40]
RES = Path("results_r3_quick" if QUICK else "results_r3"); CACHE = RES / "cache"; CACHE.mkdir(parents=True, exist_ok=True)
FN = eeg.FEATURE_NAMES


def log(name, title, obj):
    s = f"\n=== {title} ===\n" + (obj.to_string() if hasattr(obj, "to_string") else str(obj))
    print(s, flush=True)
    with open(RES / f"summary_{name}.txt", "a") as fh:
        fh.write(s + "\n")


msd = lambda x: f"{np.nanmean(x):.3f} ± {np.nanstd(x, ddof=1) if len(x) > 1 else 0:.3f}"


# ------------------------------------------------------------------ generic cohort machinery
def build(worker, seed, tag, n=N, kind=None, extra=None, net="1020"):
    extra = extra or {}
    key = hashlib.md5(json.dumps([worker.__name__, seed, tag, n, kind, net, extra, rs.SIM2], default=str,
                                 sort_keys=True).encode()).hexdigest()[:12]
    path = CACHE / f"{tag}_s{seed}_{key}.pkl"
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
        tasks.append(dict(sid=i, label=int(labels[i]), kind=k, canon=canon, sensors=sensors, net=net,
                          fs=int(rng.choice(FS)), dur=int(rng.choice(DUR)), ss=np.random.SeedSequence([seed, 104729, i]).entropy,
                          **extra))
    t0 = time.time()
    outs = Parallel(n_jobs=NJ)(delayed(worker)(t) for t in tasks)
    meta = pd.DataFrame([o["meta"] for o in outs]).set_index("sid")
    c = dict(meta=meta, feats={k: {o["meta"]["sid"]: o["feats"].get(k) for o in outs} for k in outs[0]["feats"]})
    path.write_bytes(pickle.dumps(c))
    print(f"  built {tag} seed={seed} n={n} in {time.time()-t0:.0f}s", flush=True)
    return c


def model(name, seed=0):
    return (RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced", n_jobs=NJ, random_state=seed)
            if name == "RF" else LogisticRegression(max_iter=3000, class_weight="balanced"))


def stack(F, meta, sids, cols=None):
    X, y, g = [], [], []
    for s in sids:
        x = F.get(s)
        if x is None:
            continue
        X.append(x if cols is None else x[:, cols]); y += [int(meta.at[s, "label"])] * len(x); g += [s] * len(x)
    return np.vstack(X), np.array(y), np.array(g)


class Fitted:
    """median imputation -> optional age residualisation -> scaling -> classifier, all fit on training rows."""
    def __init__(self, name, seed, age_resid=False):
        self.name, self.seed, self.age_resid = name, seed, age_resid

    def fit(self, X, y, age=None):
        self.imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(X)
        X = self.imp.transform(X)
        if self.age_resid:
            A = np.column_stack([np.ones_like(age), age])
            self.beta = np.linalg.lstsq(A, X, rcond=None)[0]
            X = X - A @ self.beta
        self.sc = StandardScaler().fit(X)
        self.clf = model(self.name, self.seed).fit(self.sc.transform(X), y)
        return self

    def proba(self, X, age=None):
        X = self.imp.transform(X)
        if self.age_resid:
            X = X - np.column_stack([np.ones_like(age), age]) @ self.beta
        return self.clf.predict_proba(self.sc.transform(X))[:, 1]


def fitpred(F, meta, tr, te, name, seed, cols=None, age_resid=False, Fte=None, meta_te=None):
    Fte = F if Fte is None else Fte; meta_te = meta if meta_te is None else meta_te
    X, y, g = stack(F, meta, tr, cols)
    m = Fitted(name, seed, age_resid).fit(X, y, meta.loc[g, "age"].to_numpy() if age_resid else None)
    p = pd.Series(np.nan, index=pd.Index(te))
    te_ok = [s for s in te if Fte.get(s) is not None]
    if te_ok:
        Xt, _, gt = stack(Fte, meta_te, te_ok, cols)
        pp = m.proba(Xt, meta_te.loc[gt, "age"].to_numpy() if age_resid else None)
        sp = pd.Series(pp).groupby(gt).mean()
        p.loc[sp.index] = sp.values
    return p


def oof(F, meta, seed, name, cols=None, age_resid=False):
    sids = meta.index.to_numpy(); p = pd.Series(np.nan, index=meta.index)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(sids, meta["label"]):
        pp = fitpred(F, meta, sids[tr], sids[te], name, seed, cols, age_resid); p.loc[pp.index] = pp.values
    return p


def met(y, p):
    y = np.asarray(y, int); p = np.asarray(p, float); ok = np.isfinite(p)
    pi = np.where(ok, p, 0.5); pred = np.where(ok, (pi > 0.5).astype(int), 1 - y)
    return dict(coverage=ok.mean(), acc=accuracy_score(y, pred), auc=roc_auc_score(y, pi))


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); out = np.empty_like(p); m = len(p); run = 0
    for k, i in enumerate(o):
        run = max(run, min(1, (m - k) * p[i])); out[i] = run
    return out


# ================================================================== A  reference + age / IAF
def worker_A(t):
    rng = np.random.default_rng(t["ss"])
    canon = list(t["canon"]); sensors = canon + ([] if "CZ" in canon else ["CZ"])
    if t["sim"] == "v1":
        sig, info = rs.simulate_v1(t["label"], sensors, t["fs"], t["dur"], rng, t["effect"])
    else:
        sig, info = L.simulate_v2_mod(t["label"], sensors, t["fs"], t["dur"], rng, t["effect"])
    names_all, style = rs.style_names(sensors, rng)
    iz = sensors.index("CZ")
    m_idx = [sensors.index(c) for c in canon]
    feats, meta = {}, dict(sid=t["sid"], label=t["label"], kind=t["kind"], style=style,
                           **{k: v for k, v in info.items() if k in ("age", "alpha_pf", "eps_i", "gain")})
    # recordings as in the main design (no physical reference: an ideal 'infinity' reference)
    d0, n0 = sig[m_idx], [names_all[i] for i in m_idx]
    for ref in ("car", "none"):
        try:
            ep, cn = L.prep_ref(d0, t["fs"], n0, ref)
            fx, fa, iaf = L.features_fixed_and_iaf(ep, cn)
            feats[f"ideal_{ref}"] = fx
            if ref == "car":
                feats["ideal_car_iaf"] = fa; meta["iaf_est"] = iaf
        except ValueError:
            feats[f"ideal_{ref}"] = None
            if ref == "car":
                feats["ideal_car_iaf"] = None
    # recordings referenced to a physical Cz electrode (Cz itself becomes flat and is dropped)
    d1 = sig - sig[iz]
    keep = [i for i in m_idx if i != iz]
    d1, n1 = d1[keep], [names_all[i] for i in keep]
    for ref in ("none", "car", "median", "rest"):
        try:
            ep, cn = L.prep_ref(d1, t["fs"], n1, ref)
            feats[f"cz_{ref}"] = rs.region_features(ep, cn)
        except ValueError:
            feats[f"cz_{ref}"] = None
    return dict(meta=meta, feats=feats)


def part_A():
    rows, conf = [], []
    for sim, eff, seeds in [("v2", 0.60, range(5)), ("v1", 0.30, range(5))]:
        for seed in seeds:
            c = build(worker_A, seed, f"A_{sim}", extra=dict(sim=sim, effect=eff))
            meta = c["meta"]
            variants = ["ideal_car", "ideal_none", "cz_none", "cz_car", "cz_median", "cz_rest"]
            if sim == "v2":
                variants += ["ideal_car_iaf", "ideal_car+age", "ideal_car_iaf+age"]
            for v in variants:
                base, ar = v.replace("+age", ""), v.endswith("+age")
                for name in ("RF", "LR"):
                    p = oof(c["feats"][base], meta, seed, name, age_resid=ar)
                    m = met(meta["label"], p)
                    accs = {f"acc_{k}": accuracy_score(meta.label[meta.kind == k], (p[meta.kind == k].fillna(.5) > .5))
                            for k in ("full19", "openbci8", "random")}
                    rows.append(dict(sim=sim, seed=seed, variant=v, model=name, **m, **accs))
                    if sim == "v2" and base.startswith("ideal_car"):
                        for lab in (0, 1):
                            mm = (meta.label == lab) & p.notna()
                            conf.append(dict(seed=seed, variant=v, model=name, cls=lab,
                                             rho_age=stats.spearmanr(p[mm], meta.age[mm])[0],
                                             rho_iaf=stats.spearmanr(p[mm], meta.alpha_pf[mm])[0],
                                             rho_eps=stats.spearmanr(p[mm], meta.eps_i[mm])[0] if lab else np.nan))
            if sim == "v2":
                ok = meta.iaf_est.notna()
                conf.append(dict(seed=seed, variant="IAF estimate vs true", model="-", cls=-1,
                                 rho_age=np.corrcoef(meta.iaf_est[ok], meta.alpha_pf[ok])[0, 1],
                                 rho_iaf=np.mean(np.abs(meta.iaf_est[ok] - meta.alpha_pf[ok])), rho_eps=np.nan))
    a = pd.DataFrame(rows); a.to_csv(RES / "A_reference_age_raw.csv", index=False)
    cf = pd.DataFrame(conf); cf.to_csv(RES / "A_confound_raw.csv", index=False)
    tab = a.groupby(["sim", "model", "variant"]).agg(auc=("auc", msd), acc=("acc", msd), cov=("coverage", "mean"),
                                                     acc19=("acc_full19", "mean"), acc8=("acc_openbci8", "mean"),
                                                     accR=("acc_random", "mean"))
    # paired tests against the released pipeline variant (cz_car == ideal_car up to the dropped Cz)
    pv = []
    for (sim, mdl), d in a.groupby(["sim", "model"]):
        ref = d[d.variant == "cz_car"].set_index("seed")
        vs = [v for v in d.variant.unique() if v.startswith("cz_") and v != "cz_car"]
        ps = []
        for v in vs:
            x = d[d.variant == v].set_index("seed")
            ps.append(stats.ttest_rel(x.auc, ref.auc.loc[x.index]).pvalue)
        for v, p_, ph in zip(vs, ps, holm(ps)):
            x = d[d.variant == v].set_index("seed")
            pv.append(dict(sim=sim, model=mdl, variant=v, dAUC_vs_CAR=(x.auc - ref.auc.loc[x.index]).mean(),
                           dACC_vs_CAR=(x.acc - ref.acc.loc[x.index]).mean(), p=p_, p_holm=ph))
        r0 = d[d.variant == "ideal_car"].set_index("seed")
        for v in ["ideal_none", "ideal_car_iaf", "ideal_car+age", "ideal_car_iaf+age"]:
            if v in set(d.variant):
                x = d[d.variant == v].set_index("seed")
                pv.append(dict(sim=sim, model=mdl, variant=v + " vs ideal_car", dAUC_vs_CAR=(x.auc - r0.auc).mean(),
                               dACC_vs_CAR=(x.acc - r0.acc).mean(), p=stats.ttest_rel(x.auc, r0.auc).pvalue, p_holm=np.nan))
    log("A", "A1 Reference choice (subject 5-fold CV, mean ± SD over 5 cohorts). ideal_* = no physical reference "
        "(as in the main simulations); cz_* = recorded against a physical Cz electrode", tab)
    log("A", "A2 Paired differences (AUC/acc) vs CAR; Holm over the three cz_ alternatives", pd.DataFrame(pv).round(4))
    log("A", "A3 Confounds within class (Spearman rho of out-of-fold P(dyslexic) with age, true alpha peak, eps_i); "
        "last rows: IAF estimate vs true (r, MAE Hz)",
        cf.groupby(["variant", "model", "cls"])[["rho_age", "rho_iaf", "rho_eps"]].mean().round(3))


# ================================================================== B  narrower volume conduction
NAMED = {
    "full 19": list(rs.BASE_CANON),
    "consumer 8": list(rs.OPENBCI8_CANON),
    "posterior only": ["P7", "P3", "PZ", "P4", "P8", "O1", "O2"],
    "left hemi.+midline": ["FP1", "F7", "F3", "FZ", "T7", "C3", "CZ", "P7", "P3", "PZ", "O1"],
    "full - L posterior": [c for c in rs.BASE_CANON if c not in ("T7", "P7", "P3", "O1")],
    "right hemi.+midline": ["FP2", "F4", "F8", "FZ", "C4", "CZ", "T8", "P4", "P8", "PZ", "O2"],
    "fronto-central only": ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "C3", "CZ", "C4"],
}
for r_ in eeg.REGIONS:
    NAMED[f"drop {r_}"] = [c for c in rs.BASE_CANON if eeg.channel_region(c) != r_]


def heldout_vs_seen(F, meta, mask, seed, name):
    r = np.random.default_rng(seed); out = []
    ks = meta.index[mask].to_numpy(); others = meta.index[~mask].to_numpy()
    perm = r.permutation(ks)
    for T in (perm[: len(perm) // 2], perm[len(perm) // 2:]):
        pool = np.setdiff1d(meta.index.to_numpy(), T)
        seen = r.choice(pool, size=min(len(others), len(pool)), replace=False)
        y = meta.loc[T, "label"]
        mu, ms = met(y, fitpred(F, meta, others, T, name, seed)), met(y, fitpred(F, meta, seen, T, name, seed))
        out.append(dict(unseen_acc=mu["acc"], seen_acc=ms["acc"], unseen_auc=mu["auc"], seen_auc=ms["auc"]))
    return out


def std_worker(t):
    o = rs._subject_worker(t)
    return dict(meta=o["meta"], feats={"f": o["feats"]["full|region_rel"]})


def part_B():
    rows, rows7 = [], []
    for lam in (0.20, 0.30, 0.45):
        p = dict(rs.SIM2, vc_lambda=lam)
        for seed in range(3):
            ex = dict(sim="v2", effect=0.60, plan={"full": ["region_rel"]}, p=p)
            c = build(std_worker, seed, f"B_lam{lam:.2f}", extra=ex)
            F, meta = c["feats"]["f"], c["meta"]
            egi = build(std_worker, seed + 500, f"B_egi_lam{lam:.2f}", n=20 if QUICK else 90, net="egi", extra=ex)
            for name in ("RF", "LR"):
                cv = met(meta.label, oof(F, meta, seed, name))
                rows.append(dict(lam=lam, seed=seed, model=name, test="CV", held="-", unseen_acc=cv["acc"],
                                 seen_acc=np.nan, unseen_auc=cv["auc"], seen_auc=np.nan))
                for k in ("full19", "openbci8", "random"):
                    for r_ in heldout_vs_seen(F, meta, (meta.kind == k).to_numpy(), seed, name):
                        rows.append(dict(lam=lam, seed=seed, model=name, test="montage", held=k, **r_))
                mu = met(egi["meta"].label, fitpred(F, meta, meta.index, egi["meta"].index, name, seed,
                                                     Fte=egi["feats"]["f"], meta_te=egi["meta"]))
                rows.append(dict(lam=lam, seed=seed, model=name, test="EGI-128", held="egi", unseen_acc=mu["acc"],
                                 seen_acc=cv["acc"], unseen_auc=mu["auc"], seen_auc=cv["auc"]))
            # E7: 120 new subjects, recorded with 19 channels, re-analysed under every layout
            r = np.random.default_rng([seed, 4242])
            ne = 10 if QUICK else 60
            labels = r.permutation([0] * ne + [1] * ne)
            mons = list(NAMED.items())
            for k in (4, 6, 8, 12, 16):
                for d_ in range(3):
                    mons.append((f"random {k}", [rs.BASE_CANON[j] for j in sorted(r.choice(19, k, replace=False))]))
            tasks = [dict(sid=900000 + i, label=int(labels[i]), fs=int(r.choice(FS)), dur=int(r.choice(DUR)), effect=0.60,
                          montages=mons, p=p, ss=np.random.SeedSequence([seed, 5151, i]).entropy) for i in range(2 * ne)]
            res = Parallel(n_jobs=NJ)(delayed(rs._multi_montage_worker)(t) for t in tasks)
            X, y, _ = stack(F, meta, meta.index)
            for name in ("RF", "LR"):
                m = Fitted(name, seed).fit(X, y)
                for subj in res:
                    for o in subj:
                        pr = float(m.proba(o["x"]).mean()) if o["x"] is not None else np.nan
                        rows7.append(dict(lam=lam, seed=seed, model=name, sid=o["sid"], label=o["label"],
                                          montage=o["montage"], n_regions=o["n_regions"], p=pr))
            print(f"  B lam={lam} seed={seed} done", flush=True)
    b = pd.DataFrame(rows); b.to_csv(RES / "B_lambda_E3_raw.csv", index=False)
    b["d_acc"] = b.unseen_acc - b.seen_acc
    out = []
    for (lam, mdl, test, held), d in b.groupby(["lam", "model", "test", "held"]):
        pv = stats.ttest_rel(d.unseen_acc, d.seen_acc).pvalue if test == "montage" else np.nan
        out.append(dict(lam=lam, model=mdl, test=test, held=held, unseen_acc=msd(d.unseen_acc),
                        seen_acc=msd(d.seen_acc) if test != "CV" else "", d_acc=d.d_acc.mean(), unseen_auc=msd(d.unseen_auc), p=pv))
    out = pd.DataFrame(out)
    for (lam, mdl), idx in out[out.test == "montage"].groupby(["lam", "model"]).groups.items():
        out.loc[idx, "p_holm"] = holm(out.loc[idx, "p"].to_numpy())
    log("B", "B1 Volume-conduction width lambda: CV, leave-one-montage-out (same test subjects, size-matched training; "
        "3 cohorts x 2 halves) and unseen EGI-128", out.set_index(["lam", "model", "test", "held"]))
    e7 = pd.DataFrame(rows7); e7.to_csv(RES / "B_lambda_E7_raw.csv", index=False)
    e7["mfam"] = e7.montage.str.replace(r" \d+$", "", regex=True).where(~e7.montage.str.startswith("random"), e7.montage)

    def agg(d):
        acc_ = d[d.p.notna()]
        return pd.Series(dict(accepted=d.p.notna().mean(),
                              acc=accuracy_score(acc_.label, acc_.p > .5) if len(acc_) else np.nan,
                              auc=roc_auc_score(acc_.label, acc_.p) if acc_.label.nunique() == 2 else np.nan))
    t7 = e7.groupby(["model", "lam", "montage"]).apply(agg)
    if isinstance(t7, pd.Series):
        t7 = t7.unstack()
    t7.to_csv(RES / "B_lambda_E7.csv")
    log("B", "B2 Electrode placement under each lambda (accuracy/AUC on accepted subjects; 3 cohorts x 120 subjects)",
        t7.round(3).unstack("lam"))
    full = t7.xs("full 19", level="montage")["acc"]
    drops = t7[t7.index.get_level_values("montage").str.startswith("drop")]["acc"].unstack("montage")
    rng_ = (drops.sub(full, axis=0)).agg(["min", "max"], axis=1)
    log("B", "B3 Dropping one region: change in accuracy vs full 19 (min, max over the 11 regions)", rng_.round(3))


# ================================================================== C  compositional features
def part_C():
    cache = Path("results_v3/cache")
    REL = [i for i, f in enumerate(FN) if f.split("_")[-1] in ("delta", "theta", "alpha", "beta", "gamma")]
    DROPG = [i for i in range(len(FN)) if not FN[i].endswith("_gamma")]
    fam = lambda f: ("asym" if f.startswith("asym") else f.split("_")[-1])
    rows, imp = [], []
    for seed in range(3):
        f = sorted(cache.glob(f"v2_nmm_0.60_s{seed}_*.pkl"), key=lambda q: -q.stat().st_size)[0]
        c = pickle.loads(f.read_bytes()); F, meta = c["feats"], c["meta"]
        assert QUICK or len(meta) == 360
        for setname, cols in [("77 features (5 rel. powers per region)", None), ("drop gamma (4 rel. powers per region)", DROPG)]:
            names = FN if cols is None else [FN[i] for i in cols]
            fams = sorted(set(fam(n) for n in names))
            for name in ("RF", "LR"):
                m = met(meta.label, oof(F, meta, seed, name, cols=cols))
                rows.append(dict(seed=seed, set=setname, model=name, **m))
                # grouped permutation importance on held-out folds (epoch AUC drop, 5 repeats)
                sids = meta.index.to_numpy(); r = np.random.default_rng(seed)
                for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(sids, meta.label):
                    X, y, g = stack(F, meta, sids[tr], cols); mm = Fitted(name, seed).fit(X, y)
                    Xt, yt, gt = stack(F, meta, sids[te], cols); base = roc_auc_score(yt, mm.proba(Xt))
                    for fa in fams:
                        j = [k for k, n in enumerate(names) if fam(n) == fa]
                        drops = []
                        for _ in range(5):
                            Xp = Xt.copy(); Xp[:, j] = Xp[r.permutation(len(Xp))][:, j]
                            drops.append(base - roc_auc_score(yt, mm.proba(Xp)))
                        imp.append(dict(seed=seed, set=setname, model=name, family=fa, drop=np.mean(drops)))
        # how collinear are the relative powers? rank of the 5 per-region relative powers
        X, _, _ = stack(F, meta, meta.index)
        blk = X[:, [FN.index(f"P_L_{b}") for b in ("delta", "theta", "alpha", "beta", "gamma")]]
        blk = blk[np.isfinite(blk).all(1)]
        rows.append(dict(seed=seed, set="check: sum of P_L relative powers", model="-", coverage=np.nan,
                         acc=blk.sum(1).mean(), auc=blk.sum(1).std()))
    cc = pd.DataFrame(rows); cc.to_csv(RES / "C_compositional_raw.csv", index=False)
    log("C", "C1 Compositional features: CV with all 5 relative powers vs one band dropped per region",
        cc.groupby(["set", "model"]).agg(auc=("auc", msd), acc=("acc", msd)))
    im = pd.DataFrame(imp); im.to_csv(RES / "C_importance_raw.csv", index=False)
    log("C", "C2 Grouped permutation importance (held-out epoch AUC drop, mean over 3 cohorts)",
        im.groupby(["set", "model", "family"])["drop"].mean().unstack("family").round(4))


# ================================================================== D  null cohorts
def part_D():
    rows = []
    for sim in ("v1", "v2"):
        for seed in (range(100, 102) if QUICK else range(100, 120)):
            c = build(std_worker, seed, f"D_null_{sim}", extra=dict(sim=sim, effect=0.0, plan={"full": ["region_rel"]}, p=rs.SIM2))
            for name in ("RF", "LR"):
                rows.append(dict(sim=sim, seed=seed, model=name, **met(c["meta"].label, oof(c["feats"]["f"], c["meta"], seed, name))))
    d = pd.DataFrame(rows); d.to_csv(RES / "D_null_raw.csv", index=False)
    out = []
    for (sim, mdl), x in d.groupby(["sim", "model"]):
        m, s, n = x.auc.mean(), x.auc.std(ddof=1), len(x)
        h = stats.t.ppf(0.975, n - 1) * s / np.sqrt(n)
        out.append(dict(sim=sim, model=mdl, n_cohorts=n, auc_mean=m, auc_sd=s, ci_lo=m - h, ci_hi=m + h,
                        t=(m - 0.5) / (s / np.sqrt(n)), p_vs_half=stats.ttest_1samp(x.auc, 0.5).pvalue,
                        acc_mean=x.acc.mean(), frac_cohorts_auc_gt_0p55=(x.auc > 0.55).mean()))
    log("D", "D1 Null cohorts (eps = 0): subject 5-fold CV AUC over 20 independent cohorts per simulator",
        pd.DataFrame(out).round(4))


# ================================================================== E  mismatch effects
def worker_E(t):
    rng = np.random.default_rng(t["ss"])
    sig, info = L.simulate_v2_mod(t["label"], t["canon"], t["fs"], t["dur"], rng, t["effect"],
                                  mode=t["mode"], strength=t["strength"])
    names, _ = rs.style_names(t["canon"], rng)
    meta = dict(sid=t["sid"], label=t["label"], kind=t["kind"], age=info["age"], alpha_pf=info["alpha_pf"])
    try:
        ep, cn = L.prep_ref(sig, t["fs"], names, "car")
        fx, fa, iaf = L.features_fixed_and_iaf(ep, cn)
        return dict(meta=meta, feats={"fixed": fx, "iaf": fa})
    except ValueError:
        return dict(meta=meta, feats={"fixed": None, "iaf": None})


def part_E():
    conds = [("regional", 0.60, 0.0), ("boundary", 0.60, 0.0), ("alpha_slow", 0.0, 0.5), ("alpha_slow", 0.0, 1.0),
             ("alpha_slow", 0.0, 1.5), ("coupling", 0.0, 0.3), ("coupling", 0.0, 0.6)]
    rows = []
    for mode, eff, st in conds:
        for seed in range(3):
            c = build(worker_E, seed, f"E_{mode}_{eff:.2f}_{st:.2f}", extra=dict(mode=mode, effect=eff, strength=st))
            for fs_ in ("fixed", "iaf"):
                for name in ("RF", "LR"):
                    rows.append(dict(mode=mode, effect=eff, strength=st, seed=seed, bands=fs_, model=name,
                                     **met(c["meta"].label, oof(c["feats"][fs_], c["meta"], seed, name))))
    e = pd.DataFrame(rows); e.to_csv(RES / "E_mismatch_raw.csv", index=False)
    log("E", "E1 Mismatch effects (v2 background; subject 5-fold CV over 3 cohorts): fixed vs IAF-adjusted bands",
        e.groupby(["mode", "effect", "strength", "model", "bands"]).auc.agg(msd).unstack("bands"))


for part, fn in [("A", part_A), ("B", part_B), ("C", part_C), ("D", part_D), ("E", part_E)]:
    if part in PARTS:
        t0 = time.time(); fn(); print(f"PART {part} DONE in {time.time()-t0:.0f}s", flush=True)
