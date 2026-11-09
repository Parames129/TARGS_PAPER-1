"""Fourth-round revision analyses (reviewer 2, second report).

F  v3 (eta = 0.12): 10 cohorts. Leave-one-montage-out with the cohort as the unit (halves averaged),
   unseen EGI-128 against CV on all subjects and on the full-19-channel subset, and the dynamical regime
   of the effect-region neural masses (also at eta = 0.30, 3 cohorts).
G  v2 (eps = 0.60): the same EGI comparison over 10 cohorts.
H  v3 with the effect placed elsewhere: bilateral (16 regions) and left frontal (8 regions), 5 cohorts each.
Usage: python run_round4.py F G H
"""
import sys
PARTS = set(sys.argv[1:]) or set("FGH")
sys.argv = [sys.argv[0], "Z"]                  # import the round-3 machinery without running it
import run_round3 as r3                        # noqa: E402
import numpy as np, pandas as pd               # noqa: E402
from scipy import stats                        # noqa: E402
import revision_sim as rs                      # noqa: E402
import revision_sim_v3 as v3                   # noqa: E402

RES = r3.RES.parent / "results_r4"; RES.mkdir(exist_ok=True)
LF = "leadfield_sample.npz"
msd = r3.msd
FRONTAL_L = tuple(f"{x}-lh" for x in ("superiorfrontal", "rostralmiddlefrontal", "caudalmiddlefrontal", "parsopercularis",
                                      "parstriangularis", "parsorbitalis", "lateralorbitofrontal", "precentral"))
BILATERAL = tuple(v3.EFFECT_LABELS) + tuple(l.replace("-lh", "-rh") for l in v3.EFFECT_LABELS)


def log(name, title, obj):
    s = f"\n=== {title} ===\n" + (obj.to_string() if hasattr(obj, "to_string") else str(obj))
    print(s, flush=True)
    with open(RES / f"summary_{name}.txt", "a") as fh:
        fh.write(s + "\n")


def w3(t):
    o = v3.subject_worker_v3(t)
    return dict(meta=o["meta"], feats={"f": o["feats"]["full|region_rel"]})


def ex3(eff, labels=None):
    return dict(sim="v3", effect=eff, effect_type="nmm", plan={"full": ["region_rel"]}, lf_path=LF, p=rs.SIM2,
                **({"effect_labels": labels} if labels else {}))


def ex2():
    return dict(sim="v2", effect=0.60, plan={"full": ["region_rel"]}, p=rs.SIM2)


def egi_and_montage(c, egi, seed, rows, gen):
    F, meta = c["feats"]["f"], c["meta"]
    for name in ("RF", "LR"):
        p = r3.oof(F, meta, seed, name)
        cv = r3.met(meta.label, p)
        f19 = meta.kind == "full19"
        cv19 = r3.met(meta.label[f19], p[f19])
        eg = r3.met(egi["meta"].label, r3.fitpred(F, meta, meta.index, egi["meta"].index, name, seed,
                                                  Fte=egi["feats"]["f"], meta_te=egi["meta"]))
        rows.append(dict(gen=gen, seed=seed, model=name, test="EGI", held="egi", unseen_acc=eg["acc"], unseen_auc=eg["auc"],
                         cv_all_acc=cv["acc"], cv_all_auc=cv["auc"], cv_full19_acc=cv19["acc"], cv_full19_auc=cv19["auc"]))
        for k in ("full19", "openbci8", "random"):
            hv = r3.heldout_vs_seen(F, meta, (meta.kind == k).to_numpy(), seed, name)
            rows.append(dict(gen=gen, seed=seed, model=name, test="montage", held=k,
                             unseen_acc=np.mean([h["unseen_acc"] for h in hv]), seen_acc=np.mean([h["seen_acc"] for h in hv]),
                             unseen_auc=np.mean([h["unseen_auc"] for h in hv]), seen_auc=np.mean([h["seen_auc"] for h in hv])))


def summarise(rows, name):
    d = pd.DataFrame(rows)
    d.to_csv(RES / f"{name}_raw.csv", index=False)
    out = []
    for (gen, mdl), g in d.groupby(["gen", "model"]):
        m = g[g.test == "montage"]
        ps, recs = [], []
        for k, x in m.groupby("held"):
            dd = x.unseen_acc - x.seen_acc
            n = len(dd); h = stats.t.ppf(0.975, n - 1) * dd.std(ddof=1) / np.sqrt(n)
            pv = stats.ttest_rel(x.unseen_acc, x.seen_acc).pvalue
            ps.append(pv)
            recs.append(dict(gen=gen, model=mdl, row=f"held-out {k}", n_cohorts=n, unseen=msd(x.unseen_acc), seen=msd(x.seen_acc),
                             d_acc=dd.mean(), ci_lo=dd.mean() - h, ci_hi=dd.mean() + h, p=pv))
        for r, ph in zip(recs, r3.holm(ps)):
            r["p_holm"] = ph; out.append(r)
        e = g[g.test == "EGI"]
        for comp in ("cv_all", "cv_full19"):
            dd = e.unseen_acc - e[f"{comp}_acc"]
            n = len(dd); h = stats.t.ppf(0.975, n - 1) * dd.std(ddof=1) / np.sqrt(n)
            out.append(dict(gen=gen, model=mdl, row=f"EGI unseen vs {comp}", n_cohorts=n, unseen=msd(e.unseen_acc),
                            seen=msd(e[f"{comp}_acc"]), d_acc=dd.mean(), ci_lo=dd.mean() - h, ci_hi=dd.mean() + h,
                            p=stats.ttest_rel(e.unseen_acc, e[f"{comp}_acc"]).pvalue, p_holm=np.nan))
    log(name, f"{name}: cohort = unit of analysis (two halves averaged within a cohort); 95% CI over cohorts", pd.DataFrame(out).round(4))


def part_F():
    rows, diag = [], []
    for seed in range(10):
        c = r3.build(w3, seed, "F_v3_0.12", extra=ex3(0.12))
        egi = r3.build(w3, seed + 500, "F_v3egi_0.12", n=90, net="egi", extra=ex3(0.12))
        egi_and_montage(c, egi, seed, rows, "v3")
        diag.append(c["meta"].assign(seed=seed, eta=0.12))
        print(f"  F seed {seed} done", flush=True)
    for seed in range(3):
        c = r3.build(w3, seed, "F_v3_0.30", extra=ex3(0.30))
        diag.append(c["meta"].assign(seed=seed, eta=0.30))
    summarise(rows, "F")
    dg = pd.concat(diag)
    dg.to_csv(RES / "F_regime_raw.csv")
    cols = ["eff_peak_hz", "eff_frac_nonalpha", "eff_skew", "eff_frac_skew1", "eff_sd_ratio"]
    t = dg.groupby(["eta", "label"])[cols].agg(["median", "mean"]).round(3)
    log("F", "F2 Regime of the 8 effect-region neural masses (per subject: median peak frequency, share of effect columns "
        "with peak outside 7.5-13 Hz, median skewness, share |skew|>1, SD ratio effect/other regions)", t)
    sh = []
    for (eta, lab), g in dg.groupby(["eta", "label"]):
        sh.append(dict(eta=eta, label=lab, n=len(g),
                       share_subjects_peak_below_7p5=(g.eff_peak_hz < 7.5).mean(),
                       share_subjects_any_column_skew_gt1=(g.eff_frac_skew1 > 0).mean(),
                       share_subjects_majority_cols_nonalpha=(g.eff_frac_nonalpha > 0.5).mean(),
                       peak_shift_vs_controls=np.nan))
    sh = pd.DataFrame(sh)
    for eta in sh.eta.unique():
        c0 = dg[(dg.eta == eta) & (dg.label == 0)].eff_peak_hz.median()
        sh.loc[(sh.eta == eta) & (sh.label == 1), "peak_shift_vs_controls"] = dg[(dg.eta == eta) & (dg.label == 1)].eff_peak_hz.median() - c0
    log("F", "F3 Share of subjects out of the alpha regime", sh.round(3))
    # does the classifier output depend on regime exit, beyond the effect size?
    sub = dg[(dg.eta == 0.12) & (dg.label == 1)]
    log("F", "F4 Dyslexic subjects (eta=0.12): Spearman rho of eps_i with effect-column peak frequency and SD ratio",
        pd.Series(dict(rho_eps_peak=stats.spearmanr(sub.eps_i, sub.eff_peak_hz)[0],
                       rho_eps_sdratio=stats.spearmanr(sub.eps_i, sub.eff_sd_ratio)[0])).round(3))


def part_G():
    rows = []
    for seed in range(10):
        c = r3.build(r3.std_worker, seed, "G_v2_0.60", extra=ex2())
        egi = r3.build(r3.std_worker, seed + 500, "G_v2egi_0.60", n=90, net="egi", extra=ex2())
        egi_and_montage(c, egi, seed, rows, "v2")
    summarise(rows, "G")


def part_H():
    rows = []
    for loc, labels in [("left posterior (reference)", None), ("bilateral posterior", BILATERAL), ("left frontal", FRONTAL_L)]:
        for seed in range(5):
            tag = "F_v3_0.12" if labels is None else f"H_v3_{loc.split()[0]}"
            c = r3.build(w3, seed, tag, extra=ex3(0.12, labels))
            meta = c["meta"]
            for name in ("RF", "LR"):
                p = r3.oof(c["feats"]["f"], meta, seed, name)
                m = r3.met(meta.label, p)
                acc = {f"acc_{k}": r3.accuracy_score(meta.label[meta.kind == k], p[meta.kind == k].fillna(.5) > .5)
                       for k in ("full19", "openbci8", "random")}
                rows.append(dict(location=loc, seed=seed, model=name, **m, **acc))
    h = pd.DataFrame(rows); h.to_csv(RES / "H_location_raw.csv", index=False)
    log("H", "H1 v3 effect location (eta = 0.12; 5 cohorts; subject 5-fold CV)",
        h.groupby(["location", "model"]).agg(auc=("auc", msd), acc=("acc", msd), acc_full19=("acc_full19", "mean"),
                                             acc_consumer8=("acc_openbci8", "mean"), acc_random=("acc_random", "mean")).round(3))


for part, fn in [("G", part_G), ("F", part_F), ("H", part_H)]:
    if part in PARTS:
        import time
        t0 = time.time(); fn(); print(f"PART {part} DONE in {time.time()-t0:.0f}s", flush=True)
