import numpy as np, matplotlib, sys
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT=sys.argv[1]
plt.rcParams.update({"font.size":7.5,"axes.spines.top":False,"axes.spines.right":False,"font.family":"serif",
                     "axes.linewidth":0.6,"xtick.major.width":0.6,"ytick.major.width":0.6})
C1,C2,CG="#C44E52","#555555","#55A868"
def save(fig,n):
    fig.savefig(f"{OUT}/{n}.pdf"); fig.savefig(f"{OUT}/{n}.png",dpi=300)

# ---------- E1 effect-size sweep (values from results_summary.txt)
eps=[0,0.05,0.10,0.20,0.30,0.40,0.60,0.80]
v1_auc=[0.524,0.686,0.878,0.979,0.995,0.999,1.000,1.000]; v1_sd=[0.019,0.019,0.011,0.007,0.004,0.001,0,0]
v2_auc=[0.499,0.509,0.542,0.625,0.704,0.767,0.852,0.897]; v2_sd=[0.055,0.053,0.051,0.041,0.032,0.030,0.027,0.023]
v1_acc=[0.529,0.642,0.809,0.931,0.971,0.986,0.996,0.999]; v1_asd=[0.024,0.012,0.021,0.018,0.011,0.006,0.005,0.001]
v2_acc=[0.498,0.512,0.541,0.588,0.653,0.701,0.773,0.819]; v2_asd=[0.034,0.039,0.038,0.043,0.032,0.030,0.026,0.026]
v1_d=[-0.008,0.305,0.590,1.085,1.496,1.840,2.392,2.826]; v2_d=[-0.063,0.002,0.063,0.186,0.309,0.428,0.647,0.838]
fig,ax=plt.subplots(1,2,figsize=(3.5,1.75))
for y,s,c,m,l in [(v1_auc,v1_sd,C2,"s","v1 (idealized)"),(v2_auc,v2_sd,C1,"o","v2 (physiol. informed)")]:
    y,s=np.array(y),np.array(s)
    ax[0].plot(eps,y,marker=m,ms=2.8,color=c,lw=1,label=l); ax[0].fill_between(eps,y-s,y+s,color=c,alpha=0.18,lw=0)
ax[0].axhline(0.5,ls="--",lw=0.6,color="k"); ax[0].axvline(0.30,ls=":",lw=0.6,color="0.4")
ax[0].set(xlabel="simulated effect size ε",ylabel="subject ROC-AUC",ylim=(0.4,1.03))
ax[0].legend(frameon=False,fontsize=5.6,loc="lower right",bbox_to_anchor=(1.04,0.0),handlelength=1.2)
ax[1].plot(v1_d,v1_auc,"s",ms=2.8,color=C2); ax[1].plot(v2_d,v2_auc,"o",ms=2.8,color=C1)
ax[1].axhline(0.5,ls="--",lw=0.6,color="k")
ax[1].set(xlabel="Cohen's d (global log θ/β)",ylim=(0.4,1.03))
for a,t in zip(ax,"ab"): a.text(-0.22,1.02,f"({t})",transform=a.transAxes,fontsize=8)
fig.tight_layout(pad=0.3,w_pad=0.8); save(fig,"fig_sweep")

# ---------- E4 ablation, v2 (delta accuracy ITT, delta AUC)
abl=[("− channel-name normalisation",-0.169,0.019,-0.058),("− unit detection",0.003,0.003,-0.000),
     ("− mains notch",0.003,0.004,-0.001),("− bad-channel removal",-0.010,0.016,-0.011),
     ("− common average reference",0.007,0.011,0.007),("− epoch rejection",0.001,0.002,-0.000),
     ("absolute instead of relative power",-0.027,0.021,-0.039),("absolute power, no unit detection",-0.065,0.029,-0.075),
     ("channel-level (no region pooling)",-0.002,0.018,-0.008),("midline split F/C/P",0.006,0.007,-0.000),
     ("P7/P8 as temporal",-0.003,0.006,-0.002),("imputation + missingness indicators",-0.005,0.005,-0.006),
     ("HistGB, native NaN",0.002,0.018,-0.002),("spherical-spline interpolation",0.002,0.008,0.001),
     ("fixed 8-ch montage, relative power",-0.309,0.040,-0.097),("fixed 8-ch montage, absolute power",-0.327,0.047,-0.124),
     ("naive, fixed 8-ch",-0.578,0.036,-0.246),("naive, 19-ch + imputation",-0.237,0.019,-0.150)]
fig,ax=plt.subplots(figsize=(3.5,2.9))
y=np.arange(len(abl))
d=np.array([a[1] for a in abl]); s=np.array([a[2] for a in abl])
ax.barh(y,d,xerr=s,color=[CG if v>=0 else C1 for v in d],height=0.65,error_kw=dict(lw=0.6,capsize=1.2))
ax.axvline(0,color="k",lw=0.6); ax.set_yticks(y); ax.set_yticklabels([a[0] for a in abl],fontsize=6.3); ax.invert_yaxis()
for yy in (13.5,):
    ax.axhline(yy,color="0.7",lw=0.5,ls="--")
ax.set_xlabel("Δ subject accuracy vs. full (ITT)")
fig.tight_layout(pad=0.3); save(fig,"fig_ablation")

# ---------- E7 electrode placement
k=[4,6,8,10,12,14,16]; acc_k=[0.758,0.754,0.780,0.782,0.786,0.798,0.797]; itt_k=[0.486,0.741,0.779,0.782,0.786,0.798,0.797]
cov_k=[0.641,0.983,0.999,1,1,1,1]
reg=[4,5,6,7,8,9,10,11]; acc_r=[0.755,0.759,0.765,0.786,0.781,0.792,0.810,0.795]; n_r=[1545,1202,1508,1636,1860,1959,1612,600]
named=[("posterior only",0.803,0.872),("consumer 8-ch",0.797,0.879),("full 19",0.792,0.877),("left hemi.+midline",0.792,0.865),
       ("full − L posterior",0.758,0.824),("right hemi.+midline",0.747,0.805),("fronto-central only",0.697,0.788)]
fig,ax=plt.subplots(1,3,figsize=(7.16,1.8),gridspec_kw=dict(width_ratios=[1,1,1.25]))
ax[0].plot(k,acc_k,"o-",ms=3,color=C1,lw=1,label="processed"); ax[0].plot(k,itt_k,"s--",ms=2.6,color=C2,lw=0.9,label="ITT (refusals = errors)")
for kk,c in zip(k,cov_k):
    if c<0.9: ax[0].annotate(f"{c:.0%} usable",(kk,0.486),textcoords="offset points",xytext=(5,-1),fontsize=6)
ax[0].set(xlabel="channels (random subsets)",ylabel="subject accuracy",ylim=(0.45,0.85)); ax[0].legend(frameon=False,fontsize=6,loc="center right")
se=[np.sqrt(a*(1-a)/n)*1.96 for a,n in zip(acc_r,n_r)]
ax[1].errorbar(reg,acc_r,yerr=se,fmt="o-",ms=3,color=C1,lw=1,capsize=1.5,elinewidth=0.6)
ax[1].set(xlabel="scalp regions covered",ylim=(0.70,0.85))
yy=np.arange(len(named))
ax[2].barh(yy,[n[1] for n in named],color=[C1 if n[0]!="full 19" else C2 for n in named],height=0.6)
for i,n in enumerate(named): ax[2].text(n[1]+0.004,i,f"{n[1]:.3f} (AUC {n[2]:.3f})",va="center",fontsize=5.8)
ax[2].set_yticks(yy); ax[2].set_yticklabels([n[0] for n in named],fontsize=6.3); ax[2].invert_yaxis()
ax[2].set(xlim=(0.6,0.97),xlabel="subject accuracy")
for a,t in zip(ax,"abc"): a.text({"a":-0.24,"b":-0.30,"c":-0.45}[t],1.06,f"({t})",transform=a.transAxes,fontsize=8,va="bottom")
fig.tight_layout(pad=0.3,w_pad=0.9); save(fig,"fig_placement")

# ---------- E9 grouped permutation importance (band families), v1 vs v2
fam=["θ","α","log θ/β","γ","α asym.","β","δ"]
v1=[0.160,0.086,0.018,0.001,0.000,-0.000,0.001]; v1s=[0.029,0.022,0.006,0.001,0.000,0.000,0.001]
v2=[0.037,0.128,0.026,0.022,0.004,-0.001,-0.000]; v2s=[0.024,0.038,0.021,0.017,0.002,0.003,0.002]
fig,ax=plt.subplots(figsize=(3.5,1.55))
x=np.arange(len(fam)); w=0.38
ax.bar(x-w/2,v1,w,yerr=v1s,color=C2,label="v1 (idealized), ε=0.30",error_kw=dict(lw=0.6,capsize=1.2))
ax.bar(x+w/2,v2,w,yerr=v2s,color=C1,label="v2 (physiol. informed), ε=0.60",error_kw=dict(lw=0.6,capsize=1.2))
ax.set_xticks(x); ax.set_xticklabels(fam); ax.axhline(0,color="k",lw=0.5)
ax.set_ylabel("Δ held-out AUC"); ax.legend(frameon=False,fontsize=6)
fig.tight_layout(pad=0.3); save(fig,"fig_importance")
print("ok")
