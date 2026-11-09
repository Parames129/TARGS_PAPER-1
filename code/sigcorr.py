"""Similarity of the class-difference signatures of the generators (Section 4.5).

Cohen's d of every one of the 77 features (subject means, dyslexic minus control), averaged over cohorts, for v2
(eps = 0.60) and for v3 (excitability, aperiodic, frontal and bilateral effects); Pearson and Spearman correlation
between the d vectors, and the lead-field strength of the posterior and frontal effect regions.
Needs the cohort caches written by run_v3_experiments.py (results_v3/cache) and run_round4.py (results_r3/cache).
"""
import numpy as np, pickle, glob
from pathlib import Path
from scipy import stats
import revision_sim_v3 as v3, eeg_adaptive as eeg
def dvec(files, key=None):
    ds=[]
    for f in files:
        c=pickle.loads(Path(f).read_bytes()); F=c['feats'] if key is None else c['feats'][key]; meta=c['meta']
        X=[];y=[]
        for s,x in F.items():
            if x is None: continue
            X.append(np.nanmean(x,0)); y.append(int(meta.at[s,'label']))
        X=np.array(X); y=np.array(y); a,b=X[y==1],X[y==0]
        sp=np.sqrt((np.nanvar(a,0,ddof=1)+np.nanvar(b,0,ddof=1))/2); ds.append((np.nanmean(a,0)-np.nanmean(b,0))/sp)
    return np.nanmean(ds,0)
big=lambda pat:sorted(glob.glob(pat),key=lambda q:-Path(q).stat().st_size)[0]
d2=dvec([big(f'results_v3/cache/v2_nmm_0.60_s{s}_*.pkl') for s in range(3)])
d3=dvec([glob.glob(f'results_r3/cache/F_v3_0.12_s{s}_*.pkl')[0] for s in range(10)],'f')
da=dvec([big(f'results_v3/cache/v3_aperiodic_1.00_s{s}_*.pkl') for s in range(3)])
dfr=dvec([glob.glob(f'results_r3/cache/H_v3_left_s{s}_*.pkl')[0] for s in range(5)],'f')
dbi=dvec([glob.glob(f'results_r3/cache/H_v3_bilateral_s{s}_*.pkl')[0] for s in range(5)],'f')
for n,d in [('v3 exc',d3),('v3 aper',da),('v3 frontal',dfr),('v3 bilateral',dbi)]:
    o=np.isfinite(d2)&np.isfinite(d)
    print(n,o.sum(),'pearson %.3f spearman %.3f sign agree %.2f'%(stats.pearsonr(d2[o],d[o])[0],stats.spearmanr(d2[o],d[o])[0],np.mean(np.sign(d2[o])==np.sign(d[o]))))
fn=eeg.FEATURE_NAMES
for nm in ['P_L_theta','P_L_alpha','T_L_theta','O_L_alpha','global_alpha','global_theta','global_logTBR','global_delta','global_beta','global_gamma']:
    i=fn.index(nm); print(nm,'v2 %.2f v3 %.2f'%(d2[i],d3[i]))
print('v3 top',[(fn[i],round(d3[i],2),round(d2[i],2)) for i in np.argsort(-np.abs(d3))[:8]])
print('v2 top',[(fn[i],round(d2[i],2),round(d3[i],2)) for i in np.argsort(-np.abs(d2))[:8]])
print('max|d| v3 %.2f v2 %.2f frontal %.2f'%(np.nanmax(abs(d3)),np.nanmax(abs(d2)),np.nanmax(abs(dfr))))
L=np.load('leadfield_sample.npz',allow_pickle=True); G=L['base19_G']; labs=list(L['labels'])
fr=[f"{x}-lh" for x in ("superiorfrontal","rostralmiddlefrontal","caudalmiddlefrontal","parsopercularis","parstriangularis","parsorbitalis","lateralorbitofrontal","precentral")]
nrm=np.linalg.norm(G,axis=0); med=np.median(nrm)
for n,ss in [('post',list(v3.EFFECT_LABELS)),('frontal',fr)]:
    print(n,'mean LF norm / median of all 68: %.2f'%(np.mean([nrm[labs.index(l)] for l in ss])/med))
