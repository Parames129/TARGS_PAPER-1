import sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R, OUT = sys.argv[1], sys.argv[2]
plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False, "font.family": "serif",
                     "axes.linewidth": 0.6})
C_RF, C_LR = "#555555", "#C44E52"
x1 = pd.read_csv(f"{R}/X1_v3_sweep_raw.csv")
x3 = pd.read_csv(f"{R}/X3_cross_generator_raw.csv")
fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.0), gridspec_kw=dict(width_ratios=[1, 1, 1.35]))
for a, et, xl in [(ax[0], "nmm", "excitability effect η (v3)"), (ax[1], "aperiodic", "aperiodic-slope effect Δχ (v3)")]:
    d = x1[x1.effect_type == et]
    if et == "aperiodic":
        d = pd.concat([x1[(x1.effect_type == "nmm") & (x1.eff == 0)], d])
    for mdl, c, mk in [("RF", C_RF, "s"), ("LR", C_LR, "o")]:
        g = d[d.model == mdl].groupby("eff")["auc"]
        mu, sd = g.mean(), g.std()
        a.plot(mu.index, mu, marker=mk, ms=2.8, lw=1, color=c, label={"RF": "random forest", "LR": "logistic regression"}[mdl])
        a.fill_between(mu.index, mu - sd, mu + sd, color=c, alpha=0.18, lw=0)
    a.axhline(0.5, ls="--", lw=0.6, color="k")
    a.set(xlabel=xl, ylim=(0.35, 1.02))
ax[0].set_ylabel("subject ROC-AUC (5-fold CV)")
ax[0].legend(frameon=False, fontsize=6, loc="upper left")
order = [("v2", "v2 (5-fold CV)", "v2 → v2 (CV)"), ("v3-nmm", "v3-nmm (5-fold CV)", "v3-exc → v3-exc (CV)"),
         ("v2", "v3-nmm", "v2 → v3-exc"), ("v3-nmm", "v2", "v3-exc → v2"), ("v2", "v3-aperiodic", "v2 → v3-aper"),
         ("v3-nmm", "v3-aperiodic", "v3-exc → v3-aper")]
y = np.arange(len(order)); w = 0.38
for k, (mdl, c) in enumerate([("RF", C_RF), ("LR", C_LR)]):
    mu, sd = [], []
    for tr, te, _ in order:
        v = x3[(x3.model == mdl) & (x3.train == tr) & (x3.test == te)]["auc"]
        mu.append(v.mean()); sd.append(v.std())
    ax[2].barh(y + (k - 0.5) * w, mu, w, xerr=sd, color=c, error_kw=dict(lw=0.6, capsize=1.2))
ax[2].axvline(0.5, ls="--", lw=0.6, color="k")
ax[2].set_yticks(y); ax[2].set_yticklabels([o[2] for o in order], fontsize=6.3); ax[2].invert_yaxis()
ax[2].set(xlim=(0.3, 1.0), xlabel="subject ROC-AUC (train → test)")
for a, t, dx in zip(ax, "abc", (-0.22, -0.16, -0.45)):
    a.text(dx, 1.03, f"({t})", transform=a.transAxes, fontsize=8)
fig.tight_layout(pad=0.3, w_pad=0.8)
fig.savefig(f"{OUT}/fig_v3.pdf"); fig.savefig(f"{OUT}/fig_v3.png", dpi=300)
print("ok")
