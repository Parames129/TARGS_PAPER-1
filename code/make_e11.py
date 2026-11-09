import numpy as np, matplotlib, sys
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import eeg_adaptive as eeg, revision_sim as rs
OUT=sys.argv[1]
plt.rcParams.update({"font.size":8,"axes.spines.top":False,"axes.spines.right":False,"font.family":"serif"})
C_CTRL, C_DYS = "#4C72B0", "#C44E52"
KINDS=["full19","openbci8","random"]; tasks=[]
for sim, eps in [("v1",0.30),("v2",0.60)]:
    r=np.random.default_rng(11)
    for kind in KINDS:
        for lab in (0,1):
            for i in range(30):
                k,canon=rs.random_montage(r,kind)
                tasks.append(dict(sim=sim,effect=eps,kind=kind,label=lab,canon=canon,fs=250,dur=40,p=rs.SIM2,
                                  ss=np.random.SeedSequence([11,len(tasks)]).entropy))
out=Parallel(n_jobs=-1)(delayed(rs._psd_worker)(t) for t in tasks)
for t,o in zip(tasks,out): o["sim"]=t["sim"]
f=out[0]["freqs"]; fm=(f>=1)&(f<=45)
names={"full19":"19-channel 10–20","openbci8":"8-channel consumer","random":"random 8–18-channel subset"}
fig,axes=plt.subplots(2,3,figsize=(7.16,3.1),sharex=True)
for i,sim in enumerate(["v1","v2"]):
    for j,kind in enumerate(KINDS):
        ax=axes[i,j]
        for lab,col,nm in [(0,C_CTRL,"Control"),(1,C_DYS,"Dyslexic")]:
            S=np.array([o["glob"] for o in out if o["sim"]==sim and o["kind"]==kind and o["label"]==lab])
            ax.semilogy(f[fm],np.median(S,0)[fm],color=col,lw=1.2,label=nm)
            ax.fill_between(f[fm],np.percentile(S,25,0)[fm],np.percentile(S,75,0)[fm],color=col,alpha=0.18,lw=0)
        for lo,hi in eeg.BANDS.values(): ax.axvline(lo,color="0.88",lw=0.5,zorder=0)
        if i==0: ax.set_title(names[kind],fontsize=8)
        if j==0: ax.set_ylabel(("v1 (idealized)\nε = 0.30" if sim=="v1" else "v2 (physiol. informed)\nε = 0.60")+"\nrelative power",fontsize=7.5)
        if i==1: ax.set_xlabel("Frequency (Hz)")
axes[0,0].legend(frameon=False,fontsize=7)
fig.tight_layout(pad=0.4)
fig.savefig(OUT+"/fig_spectra.pdf"); fig.savefig(OUT+"/fig_spectra.png",dpi=300)
print("ok")
