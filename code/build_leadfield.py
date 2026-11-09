"""Label-level EEG lead fields from the MNE 'sample' subject (3-layer BEM, oct-4 source space, aparc labels)."""
import sys, glob, numpy as np, mne
mne.set_log_level("ERROR")
D = sys.argv[1]; OUT = sys.argv[2]
bem = mne.read_bem_solution(f"{D}/subjects/sample/bem/sample-1280-1280-1280-bem-sol.fif")
src = mne.read_source_spaces(f"{D}/subjects/sample/bem/sample-oct-4-src.fif")
trans = mne.read_trans(f"{D}/MEG/sample/sample_audvis_trunc-trans.fif")
labels = sorted([mne.read_label(f) for f in glob.glob(f"{D}/subjects/sample/label/aparc/*.label")], key=lambda l: l.name)
labels = [l for l in labels if "unknown" not in l.name and "corpuscallosum" not in l.name]
BASE = ["Fp1","Fp2","F7","F3","Fz","F4","F8","T7","C3","Cz","C4","T8","P7","P3","Pz","P4","P8","O1","O2"]
out = {}
for tag, names, mont in [("base19", BASE, "standard_1005"), ("egi128", None, "GSN-HydroCel-128")]:
    m = mne.channels.make_standard_montage(mont)
    ch = names or [c for c in m.ch_names if c.startswith("E")]
    info = mne.create_info(ch, 250.0, "eeg"); info.set_montage(m)
    fwd = mne.make_forward_solution(info, trans, src, bem, eeg=True, meg=False, mindist=5.0, n_jobs=2)
    fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True)
    G = fwd["sol"]["data"]                         # (n_sens, n_vertices) fixed normal orientation
    # vertex bookkeeping: map each label to its columns
    verts = [s["vertno"] for s in fwd["src"]]
    offs = [0, len(verts[0])]
    L = []
    for lab in labels:
        h = 0 if lab.hemi == "lh" else 1
        idx = np.flatnonzero(np.isin(verts[h], lab.vertices)) + offs[h]
        L.append(G[:, idx].sum(axis=1) if idx.size else np.zeros(G.shape[0]))
    out[f"{tag}_G"] = np.array(L).T                 # (n_sens, n_labels)
    out[f"{tag}_ch"] = np.array(ch)
    print(tag, out[f"{tag}_G"].shape, "zero labels:", int((np.abs(out[f'{tag}_G']).sum(0) == 0).sum()))
out["labels"] = np.array([l.name for l in labels])
np.savez(OUT, **out)
print(out["labels"])
