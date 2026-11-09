# Interoperable EEG-based Machine Learning for Dyslexia Screening

Code, signal generators and results for the article

> Shunmugam P., Mohamad Salleh N.H., Joseph A., Hasib T., Vengadasalam V.
> *Interoperable EEG-based Machine Learning for Dyslexia Screening: A Simulation-Based Methodological Validation
> Across Heterogeneous Electrode Configurations.* Neuroinformatics (under review).

The repository contains:

- an EEG pipeline that reads recordings with any electrode layout, sampling rate, unit or label style;
- three signal generators used to validate it;
- every analysis script reported in the article and its supplement (Online Resource 1).

> **Scope.** This is a simulation-based methodological validation. The pipeline is tested for engineering
> interoperability (also on real public recordings) and for predictive validity *in silico*. It is **not** a
> clinically validated dyslexia test, and its decision rule must be trained and validated on real, labeled EEG.

---

## Repository layout

```
├── code/                      pipeline, generators and analysis scripts (run from this folder)
│   ├── eeg_adaptive.py        the pipeline: channel mapping, preprocessing, 77 features, inference
│   ├── revision_sim.py        generators v1 (idealized) and v2 (physiologically informed)
│   ├── revision_sim_v3.py     generator v3 (Jansen–Rit neural masses + boundary-element head model)
│   ├── leadfield_sample.npz   v3 lead fields (19 10–20 sites and EGI-128, 68 cortical regions)
│   ├── build_leadfield.py     rebuilds leadfield_sample.npz from the MNE "sample" data set
│   ├── round3_lib.py          v2 mismatch effects, reference options (CAR, median, REST), IAF bands
│   ├── run_v3_experiments.py  v3 sweep, cross-generator transfer, v3 interoperability, operational limits
│   ├── run_real_data.py       13 real public recordings: file audit and harmonization consistency
│   ├── run_round3.py          E10–E15
│   ├── run_round4.py          E16–E18
│   ├── sigcorr.py             similarity of the v2 and v3 class-difference signatures
│   └── make_figs.py, make_e11.py, make_v3_fig.py   figures
├── notebooks/
│   ├── dyslexia_pipeline_adaptive.ipynb       builds the pipeline module and trains the reference models
│   ├── dyslexia_pipeline_adaptive-viz.ipynb   the same, with visualizations
│   └── revision_experiments.ipynb             main experiments E1–E9 (see the note on numbering below)
├── results/                   outputs reported in the article (CSV tables and text summaries)
├── requirements.txt           minimum versions
├── requirements-lock.txt      exact versions used for the analyses
├── environment.yml            conda environment built from the lock file
└── CITATION.cff
```

## Installation

```bash
conda env create -f environment.yml        # exact versions
conda activate eeg-dyslexia-pipeline
# or:  pip install -r requirements-lock.txt   (Python 3.13)
```

## Using the pipeline on your own recording

```python
import eeg_adaptive as eeg                     # run from code/, or add code/ to PYTHONPATH

data, fs, names, units, notes = eeg.load_recording("subject01.edf")      # any MNE-readable format
epochs, sites, report = eeg.preprocess_recording(data, fs, names, units)
X = eeg.epoch_features(epochs, sites)          # (n_epochs, 77) relative-power features
print(report)                                  # every automatic decision, warnings, low-confidence flag
```

A recording is analyzed only if all of the following hold:

1. Its channel labels map to 10–10 sites, or 3-D electrode positions are available. Positions come from the file or from a standard net recognized by its labels (at least 16 positioned channels).
2. At least 4 of the 11 scalp regions remain after bad-channel removal.
3. It contains at least one clean 4-s epoch. Recordings with fewer than 3 clean epochs are flagged as low-confidence.

`predict_dyslexia()` additionally needs the model files `cleaning_model.pkl` and `identification_model.pkl`, written
by `notebooks/dyslexia_pipeline_adaptive.ipynb`. Those models are trained on **simulated** data. They demonstrate the
interface and must not be used for screening.

## Reproducing the analyses

Run every script from the `code/` folder. Results are written to `code/results_*`, and generated cohorts are cached there.
Set `QUICK=1` for a short smoke test of `run_v3_experiments.py`, `run_round3.py` and `run_round4.py`. Run times below
are for a 2-core machine.

| Order | Command | Article | Notes |
|---|---|---|---|
| 1 | `notebooks/revision_experiments.ipynb` | E1–E9 (Sections 4.1–4.4, 4.9–4.11, 4.13–4.14) | 1.5–3 h on 8 cores |
| 2 | `python run_v3_experiments.py` | Section 4.5 (v3 sweep, transfer), Section 4.7 (operational limits) | about 1 h; writes `results_v3/` |
| 3 | `python run_real_data.py <path-to-mne-testing-data>` | Sections 4.7–4.8, Table S2 | needs step 2 (uses a cached v2 cohort) |
| 4 | `python run_round3.py A B C D E` | E10–E15 | A: E11 reference and E12 age/IAF; B: E13 volume conduction; C: E15 compositional features (needs step 2); D: E10 null cohorts; E: E14 mismatch effects |
| 5 | `python run_round4.py F G H` | E16–E18 | F: E16 v3 over 10 cohorts and E17 regime check; G: EGI comparison under v2; H: E18 effect location |
| 6 | `python sigcorr.py` | Section 4.5 (signature similarity) | needs steps 2 and 5 |
| 7 | `python make_figs.py <out>`, `python make_e11.py <out>`, `python make_v3_fig.py results_v3 <out>` | figures | |

**Numbering.** The notebook `revision_experiments.ipynb` uses its own experiment numbers. Its E1–E9 match the article.
Its E10–E13 (auto-detection counts, example spectra, EGI distance threshold, recording length) are reported in the
article without separate numbers. The article's E10–E18 are produced by `run_round3.py` and `run_round4.py`, as listed
above.

**Seeds.** Every cohort is generated from an integer seed, and every subject from the cohort seed and its index. Results
therefore do not depend on the number of parallel processes. Section 3.5 of the article lists the seeds of each analysis.

### Real recordings

The 13 test recordings come from the MNE testing-data repository and are not redistributed here. A sparse checkout
downloads only the files used:

```bash
git clone --filter=blob:none --no-checkout https://github.com/mne-tools/mne-testing-data.git realdata/mtd
cd realdata/mtd
git sparse-checkout set --no-cone /MEG/sample/sample_audvis_trunc_raw.fif /EGI/test_egi.mff/ \
    /EEGLAB/test_raw.set /EEGLAB/test_raw.fdt /NihonKohden/ /nedf/ /curry/ /eximia/ /GDF/test_gdf_2.20.gdf \
    /antio/CA_208/test_CA_208.cnt /EDF/test_reduced.edf /BDF/test_bdf_stim_channel.bdf \
    /Brainvision/test_NO.vhdr /Brainvision/test_NO.vmrk /Brainvision/test_NO.eeg
git checkout
cd ../../code && python run_real_data.py ../realdata/mtd
```

### v3 lead fields

`leadfield_sample.npz` is included. To rebuild it you need the MNE `sample` data set
(`mne.datasets.sample.data_path()`, about 1.5 GB):

```bash
python build_leadfield.py <path-to-MNE-sample-data> leadfield_sample.npz
```

## Results folder

| Folder | Contents |
|---|---|
| `results/main_experiments/` | summary of E1–E9 (notebook run) |
| `results/v3_generator/` | v3 sweep (X1), feature effects (X1b), interoperability (X2), cross-generator transfer (X3), operational limits (X4) |
| `results/real_recordings/` | real-file audit (R1) and harmonization consistency (R2), after the fix of prefixed net labels |
| `results/round3/` | A = E11/E12, B = E13, C = E15, D = E10, E = E14 |
| `results/round4/` | F = E16/E17, G = E16 (v2 EGI comparison), H = E18 |

## Citation

Please cite the article (see `CITATION.cff`). An archived version of this repository is available at Zenodo: 10.5281/zenodo.24077685.

## Funding

[To be added.]
