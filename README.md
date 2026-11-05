# CA2 — Physical Action Classification from EMG

Completed analysis of the supplied assignment, using the repository's data,
readme, reference paper, and EMGNET slides. Start with
[the PDF report](report/EMG_Classification_Report.pdf).

## Deliverables

- **Report:** [report/EMG_Classification_Report.pdf](report/EMG_Classification_Report.pdf),
  compiled with the actual `IEEEtran` journal class in two columns.
  Editable source: [report/EMG_Classification_Report.tex](report/EMG_Classification_Report.tex).
- **MATLAB:** [matlab/run_assignment.m](matlab/run_assignment.m) and
  [matlab/extract_trial_features.m](matlab/extract_trial_features.m).
- **Executed reference implementation:** [src/emg_assignment.py](src/emg_assignment.py).
- **Features and targets:** [artifacts/emg_features.mat](artifacts/emg_features.mat).
  `X`: 1200 × 48; `y`: 1200 × 1; `data=[X y]`: 1200 × 49.
- **Exact subsets and folds:** [artifacts/experiment_plan.mat](artifacts/experiment_plan.mat).
- **Results:** [artifacts/results/summary.json](artifacts/results/summary.json),
  [accuracy_curve.csv](artifacts/results/accuracy_curve.csv),
  [accuracy_runs.csv](artifacts/results/accuracy_runs.csv), confusion matrices,
  out-of-fold predictions, and `classification_results.mat`.
- **Figures:** [artifacts/figures](artifacts/figures), both vector PDF and PNG.
- **Submission bundle:** `EMG_CA2_Submission.zip` (report, source, outputs,
  and canonical input data; no environment, caches, logs in the binary acquisition format,
  or nested subject-3 duplicate).

The report's author line is neutral because a name/roll number was not supplied.
Edit [report/author.tex](report/author.tex) to add these details before submission.
The results do not need to be rerun to change that line.

## Measured results

| Experiment | Accuracy |
|---|---:|
| Walking versus Punching, repeated stratified CV | 93.08% |
| All 20 actions, repeated stratified CV | 39.86% |
| All 20 actions, leave-one-subject-out | 22.33% |
| Ten sampled two-class subsets, averaged over repeats | 90.08% |
| Ten sampled 19-class subsets, averaged over repeats | 40.51% |

The largest adjacent mean drop is **14.15 percentage points, from 2 to 3 classes**.
These are results for the prescribed mean/variance baseline, not a reproduction
of the reference paper's richer feature ensemble. The executed run took about
5.4 minutes including deliberate pauses. The report is six pages. Five unit tests,
result consistency checks, and an independent recomputation of 120 held-out
predictions passed; see [artifacts/validation.json](artifacts/validation.json).

## Run on the MacBook Air

The repository's `.venv` is already configured. From this repository:

```sh
.venv/bin/python src/emg_assignment.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python src/build_report.py
TECTONIC_CACHE_DIR="$PWD/.cache/tectonic" .tools/tectonic --keep-logs report/EMG_Classification_Report.tex
.venv/bin/python src/validate_results.py
.venv/bin/python src/package_submission.py
```

The numerical run forces BLAS/OpenMP to **one thread**, uses `n_jobs=1`, and
does not use a GPU, multiprocessing, or deep learning. It pauses 0.05 seconds
after each cross-validation evaluation and one second after every ten evaluations.
Completed class-count batches are cached in `artifacts/cache/`, with input/code/config
fingerprints to prevent accidental reuse after a change. Reruns use these checkpoints;
`--force` deliberately recomputes them. Output logs and saved results can be inspected
without rerunning any experiment. Memory-intensive raw-waveform distance matrices
are never constructed.

These settings keep computation modest on a fanless laptop, but temperature was
not measured: the sandbox blocked macOS thermal telemetry. Increase
`pause_seconds` or `batch_pause_seconds` in `config.json` if longer pauses are desired.
Do not run Python and MATLAB experiments simultaneously. Ten Monte Carlo repeats
are used, rather than the reference paper's 100, because the handout leaves this
parameter unspecified and ten provides the complete requested experiment at lower load.

To recreate the environment, use Python **3.9–3.12** (the pinned SciPy version is
not intended for Python 3.14). The completed run used `/usr/bin/python3`, version 3.9.6:

```sh
/usr/bin/python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Tectonic is a local downloaded compiler, excluded from the submission ZIP. Any
standard LaTeX installation with IEEEtran can compile the source instead:

```sh
cd report
pdflatex EMG_Classification_Report.tex
pdflatex EMG_Classification_Report.tex
```

The report generator uses saved measured results and does not retrain models.
The figures live in `artifacts/figures`; keep that directory beside `report`.

## MATLAB instructions

From MATLAB with this repository as the current folder:

```matlab
addpath('matlab');
run_assignment(false); % Load, inspect, plot, extract, label, and save arrays only
run_assignment;        % Also run the complete classification experiments
```

For a single computational thread from Terminal, where MATLAB is on `PATH`:

```sh
matlab -singleCompThread -batch "addpath('matlab'); run_assignment"
```

The code uses base MATLAB features including `readmatrix` (R2019a or later),
`jsondecode`, and local functions. It does not require the Statistics and Machine
Learning Toolbox. Euclidean kNN uses uniform voting, with smallest-label voting
ties. MATLAB writes to `artifacts/matlab/`, keeping its outputs separate.

**Execution status:** MATLAB/Octave was not installed on this computer, so the
MATLAB code was reviewed but not executed. All reported experiments were actually
run with Python. MATLAB checks its features against `artifacts/emg_features.mat`
and reuses the exact exported fold plan when present. If the plan is absent, it
generates combinations with `nchoosek` and random ranks with `randi`; language RNG
differences then give different numerical results. Exact distance ties at the kth
neighbor can also produce minor cross-implementation differences. The saved Python
outputs are not represented as MATLAB-generated outputs.

## Method and assumptions

- Use only `Data/sub1` through `Data/sub4`, directly under `Data`. The accidental
  `Data/sub3/sub3` directory is excluded; its 20 files match canonical hashes.
  The original directory is left untouched. There are **80 unique recordings**.
- Sort filenames within Normal (labels 1–10) and Aggressive (11–20); verify
  the mapping is identical for every subject. See `artifacts/class_labels.csv`.
- Split each actual file length into 15 exhaustive intervals using
  `floor(t * length / 15)`. Split each interval into three segments similarly.
  There are no supplied event markers: these are assumed trial intervals,
  not confirmed action-onset-aligned repetitions. Nothing is truncated or padded.
- Extract **mean and sample variance**, channel → segment → statistic order.
  `Nf = 8 × 3 × 2 = 48`. No extra filter is applied. Subject 2 is unfiltered per
  the readme; that acquisition difference is discussed in the report.
- Fit feature mean/std on training folds only. Use Euclidean **5-NN** with
  uniform votes, **10 stratified folds**, **10 repeats**, seed **2026**.
- For every `M = 2…19`, draw 10 combination ranks **with replacement**, following
  `randi` in the handout; use these same subsets across the repeats.
  The saved accuracy tensor has axes `[class count, subset, MC repeat]` and shape
  `18 × 10 × 10`. Every observation is predicted once per repeat.
- Binary example: Walking (9) vs Punching (17), fixed before results.
  Supplementary checks: all 20 classes and four-fold leave-one-subject-out.
- Define a descriptive rapid decline as an adjacent mean loss ≥5 percentage
  points; also report the largest loss. Changing subset composition means this
  is not a universal statistical change point.
- Trial-level folds share subjects/recordings across train and test. They must
  not be called subject-independent. The report separately gives LOSO results.
- Plot uncertainty bands are SD across subset means, **not confidence intervals**.
  Pooled confusion matrices contain repeated predictions, not independent samples.

## Assignment coverage

| Required task | Delivered implementation/output |
|---|---|
| Count folders, list both categories' text files, display all array shapes | MATLAB `dir`/`fprintf`; `artifacts/dataset_manifest.csv`; `artifacts/run.log` |
| Plot every channel for normal and aggressive examples | `raw_signals.pdf/png`; MATLAB raw Walking/Punching plots |
| Split 15 trials, then `N_seg` segments | Both feature implementations; `artifacts/trial_metadata.csv` |
| Extract/concatenate mean and variance for eight channels | 48-column `X` in `emg_features.mat` |
| Display feature vectors of two classes | `example_features.csv`, `example_features.pdf/png`; MATLAB `disp` |
| Labels 1–20 and combined inputs/outputs | `class_labels.csv`; `y` and 1200 × 49 `data` in MAT file |
| Binary cross-validation | Saved binary predictions, confusion matrix, report case study |
| 10 random combinations for every `M=2…19` and MC runs | `experiment_plan.mat`, 1800 rows in `accuracy_runs.csv` |
| kNN accuracy array, MC heatmap, mean accuracy curve | `classification_results.mat`, `mc_heatmap`, `accuracy_vs_classes` |
| Explain when accuracy declines rapidly | Measured adjacent-drop analysis in report and `summary.json` |
| Motivation, introduction, block diagram, dataset, methods, case studies, conclusion | IEEEtran PDF report (3–10 pages, checked automatically) |
| Linked code or appendix | Clickable relative code links and explicit paths in report; submission ZIP |

The IEEE journal class comes from [CTAN's IEEEtran package](https://ctan.org/pkg/ieeetran).
The two supplied reference PDFs are cited in the report. No numerical result is
copied from them and presented as an outcome of this implementation.
