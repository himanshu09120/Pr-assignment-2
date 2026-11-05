"""CA2: deterministic, single-thread EMG feature extraction and kNN analysis.

Run from any directory: python src/emg_assignment.py
Completed class-count batches are checkpointed; --force recomputes them.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import argparse
import hashlib
import json
import math
import platform
import time
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.io import savemat
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_info, threadpool_limits

CATEGORIES = ("Normal", "Aggressive")
CHANNELS = ("R biceps", "R triceps", "L biceps", "L triceps",
            "R thigh", "R hamstring", "L thigh", "L hamstring")
OUT = ROOT / "artifacts"
RESULTS = OUT / "results"
FIGURES = OUT / "figures"


def equal_slices(length, count):
    """Floor boundaries: contiguous, exhaustive, lengths differ by at most one."""
    if count < 1 or length < 2 * count:
        raise ValueError("Each segment needs at least two samples for sample variance")
    edges = np.arange(count + 1, dtype=np.int64) * length // count
    return [slice(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def trial_features(signal, n_trials=15, n_segments=3):
    if signal.ndim != 2 or signal.shape[1] != 8 or not np.isfinite(signal).all():
        raise ValueError("Expected a finite N x 8 signal")
    rows = []
    for trial_slice in equal_slices(len(signal), n_trials):
        trial = signal[trial_slice]
        # Flatten in channel -> segment -> statistic order, identical to MATLAB.
        values = np.empty((8, n_segments, 2))
        for s, window in enumerate(equal_slices(len(trial), n_segments)):
            values[:, s, 0] = trial[window].mean(axis=0)
            values[:, s, 1] = trial[window].var(axis=0, ddof=1)
        rows.append(values.reshape(-1))
    return np.asarray(rows)


def load_dataset(config):
    data_dir = ROOT / "Data"
    subjects = sorted(p for p in data_dir.iterdir()
                      if p.is_dir() and p.name.startswith("sub"))
    if [p.name for p in subjects] != ["sub1", "sub2", "sub3", "sub4"]:
        raise ValueError("Expected four top-level subject folders")
    mapping = []
    for category in CATEGORIES:
        names = sorted(p.stem for p in (subjects[0] / category / "txt").glob("*.txt"))
        if len(names) != 10:
            raise ValueError("Expected ten actions per category")
        for name in names:
            mapping.append({"label": len(mapping) + 1, "category": category, "action": name})
    class_map = pd.DataFrame(mapping)
    blocks, labels, groups, trials, manifest, metadata = [], [], [], [], [], []
    canonical = {}
    print(f"Top-level subject folders: {len(subjects)}", flush=True)
    for subject_id, subject in enumerate(subjects, 1):
        for category in CATEGORIES:
            actual = sorted(p.stem for p in (subject / category / "txt").glob("*.txt"))
            expected = class_map.loc[class_map.category == category, "action"].tolist()
            if actual != expected:
                raise ValueError(f"Inconsistent action names in {subject / category}")
        for row in mapping:
            path = subject / row["category"] / "txt" / (row["action"] + ".txt")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in canonical:
                raise ValueError(f"Unexpected duplicate canonical recording: {path}")
            canonical[digest] = str(path.relative_to(ROOT))
            signal = np.loadtxt(path)
            print(f"{path.relative_to(ROOT)}: {signal.shape}", flush=True)
            features = trial_features(signal, config["n_trials"], config["n_segments"])
            blocks.append(features)
            labels.extend([row["label"]] * len(features))
            groups.extend([subject_id] * len(features))
            trials.extend(range(1, len(features) + 1))
            manifest.append(dict(path=str(path.relative_to(ROOT)), subject=subject_id,
                                 **row, n_samples=len(signal), n_channels=signal.shape[1],
                                 minimum=float(signal.min()), maximum=float(signal.max()),
                                 fraction_abs_ge_4000=float(np.mean(np.abs(signal) >= 4000)),
                                 sha256=digest))
            for t, sl in enumerate(equal_slices(len(signal), config["n_trials"]), 1):
                metadata.append(dict(row=len(metadata) + 1, subject=subject_id,
                                     label=row["label"], trial=t, start_sample=sl.start + 1,
                                     end_sample=sl.stop, n_samples=sl.stop - sl.start,
                                     source=str(path.relative_to(ROOT))))
    excluded = []
    for path in sorted(data_dir.glob("sub*/**/txt/*.txt")):
        if str(path.relative_to(ROOT)) in canonical.values():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest not in canonical:
            raise ValueError(f"Unexpected non-duplicate nested recording: {path}")
        excluded.append(dict(excluded=str(path.relative_to(ROOT)),
                             canonical=canonical[digest], sha256=digest))
    pd.DataFrame(manifest).to_csv(OUT / "dataset_manifest.csv", index=False)
    pd.DataFrame(metadata).to_csv(OUT / "trial_metadata.csv", index=False)
    pd.DataFrame(excluded).to_csv(OUT / "excluded_duplicates.csv", index=False)
    class_map.to_csv(OUT / "class_labels.csv", index=False)
    X, y, subject_ids = np.vstack(blocks), np.asarray(labels), np.asarray(groups)
    assert X.shape == (1200, 16 * config["n_segments"])
    assert np.all(np.bincount(y)[1:] == 60)
    feature_names = [f"ch{c}_seg{s}_{stat}" for c in range(1, 9)
                     for s in range(1, config["n_segments"] + 1)
                     for stat in ("mean", "variance")]
    savemat(OUT / "emg_features.mat", dict(
        X=X, y=y[:, None], subject_id=subject_ids[:, None],
        trial_id=np.asarray(trials)[:, None], data=np.column_stack([X, y]),
        feature_names=np.array(feature_names, dtype=object),
        class_names=np.array(class_map.action.tolist(), dtype=object),
        n_segments=config["n_segments"]), do_compression=True)
    np.savez_compressed(OUT / "emg_features.npz", X=X, y=y, subject_id=subject_ids)
    # Two actual vectors, with every feature identified, as requested by task 2.
    indices = [np.flatnonzero((y == label) & (subject_ids == 1))[0]
               for label in config["binary_labels"]]
    pd.DataFrame(X[indices].T, index=feature_names,
                 columns=[class_map.action.iloc[label - 1]
                          for label in config["binary_labels"]]).to_csv(OUT / "example_features.csv")
    return X, y, subject_ids, class_map, pd.DataFrame(manifest)


def fold_ids(y, n_folds, seed):
    folds = np.zeros(len(y), dtype=np.int8)
    for fold, (_, test) in enumerate(StratifiedKFold(n_folds, shuffle=True,
                                                   random_state=seed).split(np.zeros(len(y)), y), 1):
        folds[test] = fold
    return folds


def standardize_train_test(train, test):
    mean = train.mean(axis=0)
    std = train.std(axis=0, ddof=1)
    std[std == 0] = 1
    return (train - mean) / std, (test - mean) / std


def evaluate(X, y, folds, n_neighbors):
    """Every observation is predicted once; scaler is fitted within each fold."""
    prediction = np.zeros_like(y)
    for fold in np.unique(folds):
        test = folds == fold
        train_X, test_X = standardize_train_test(X[~test], X[test])
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights="uniform",
                                     algorithm="brute", metric="euclidean", n_jobs=1)
        model.fit(train_X, y[~test])
        prediction[test] = model.predict(test_X)
    return float(accuracy_score(y, prediction)), prediction


def make_plan(y, config):
    rng = np.random.default_rng(config["seed"])
    counts = np.arange(2, 20)
    selected = np.zeros((18, config["n_combinations"], 20), dtype=np.int16)
    ranks = np.zeros((18, config["n_combinations"]), dtype=np.int32)
    folds = np.zeros((18, config["n_combinations"], config["n_mc"], len(y)), dtype=np.int8)
    for mi, m in enumerate(counts):
        combos = list(combinations(range(1, 21), int(m)))
        # Assignment specifies randi: ranks are sampled with replacement.
        ranks[mi] = rng.integers(1, len(combos) + 1, size=config["n_combinations"])
        for ci, rank in enumerate(ranks[mi]):
            labels = combos[rank - 1]
            selected[mi, ci, :m] = labels
            mask = np.isin(y, labels)
            for mc in range(config["n_mc"]):
                seed = config["seed"] + int(m) * 10000 + ci * 100 + mc
                folds[mi, ci, mc, mask] = fold_ids(y[mask], config["n_folds"], seed)
    binary_mask = np.isin(y, config["binary_labels"])
    binary = np.zeros((config["n_mc"], len(y)), dtype=np.int8)
    full = np.zeros_like(binary)
    for mc in range(config["n_mc"]):
        binary[mc, binary_mask] = fold_ids(y[binary_mask], config["n_folds"], config["seed"] + mc)
        full[mc] = fold_ids(y, config["n_folds"], config["seed"] + 200000 + mc)
    plan = dict(class_counts=counts, selected_labels=selected, combination_ranks=ranks,
                fold_ids=folds, binary_fold_ids=binary, full_fold_ids=full)
    savemat(OUT / "experiment_plan.mat", plan, do_compression=True)
    return plan


def save_figure(name):
    plt.savefig(FIGURES / (name + ".pdf"), bbox_inches="tight")
    plt.savefig(FIGURES / (name + ".png"), dpi=160, bbox_inches="tight")
    plt.close()


def figures(X, y, subjects, classes, manifest, config, acc, binary_cm, full_cm, loso):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(8, 2, figsize=(8.2, 8.0), sharex="col")
    for col, label in enumerate(config["binary_labels"]):
        row = manifest[(manifest.subject == 1) & (manifest.label == label)].iloc[0]
        signal = np.loadtxt(ROOT / row.path)
        for ch in range(8):
            axes[ch, col].plot(np.arange(len(signal)), signal[:, ch], linewidth=.4,
                               color=("#23688a", "#af5334")[col], rasterized=True)
            axes[ch, col].set_ylabel(f"C{ch + 1}", fontsize=8)
            axes[ch, col].tick_params(labelsize=7)
        axes[0, col].set_title(f"Subject 1: {row.action}")
        axes[-1, col].set_xlabel("Sample index (raw stored units)")
    fig.tight_layout(h_pad=.2)
    save_figure("raw_signals")

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
    for label, color in zip(config["binary_labels"], ("#23688a", "#af5334")):
        vector = X[np.flatnonzero((y == label) & (subjects == 1))[0]]
        for statistic, axis in enumerate(axes):
            axis.plot(np.arange(1, 25), vector[statistic::2], "o-", ms=2.3, lw=.7,
                      color=color, label=classes.action.iloc[label-1])
    for axis, title in zip(axes, ("Mean", "Sample variance")):
        axis.set_title(title); axis.set_xlabel("Channel/segment pair (1–24)")
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    save_figure("example_features")

    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    im = ax.imshow(binary_cm, cmap="Blues", vmin=0)
    for a in range(2):
        for b in range(2):
            ax.text(b, a, str(binary_cm[a,b]), ha="center", va="center")
    names = [classes.action.iloc[x-1] for x in config["binary_labels"]]
    ax.set(xticks=[0,1], yticks=[0,1], xticklabels=names, yticklabels=names,
           xlabel="Predicted", ylabel="Actual")
    fig.colorbar(im, ax=ax, label="Predictions across 10 repeats")
    fig.tight_layout(); save_figure("binary_confusion")

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    im = ax.imshow(acc.mean(axis=1).T * 100, aspect="auto", origin="lower",
                   extent=(1.5, 19.5, .5, config["n_mc"]+.5), cmap="viridis")
    ax.set(xticks=range(2,20), yticks=range(1,config["n_mc"]+1),
           xlabel="Number of classes M", ylabel="Monte Carlo repeat")
    fig.colorbar(im, ax=ax, label="Accuracy (%)")
    fig.tight_layout(); save_figure("mc_heatmap")

    fig, ax = plt.subplots(figsize=(6.3, 3.1))
    means, sd = acc.mean(axis=(1,2))*100, acc.mean(axis=2).std(axis=1,ddof=1)*100
    ax.plot(range(2,20), means, "o-", ms=3, color="#23688a", label="Mean over 10 subsets × 10 repeats")
    ax.fill_between(range(2,20), means-sd, means+sd, color="#23688a", alpha=.17,
                    label="±1 SD across subset means")
    ax.plot(range(2,20), 100/np.arange(2,20), "--", color="#999999", label="Balanced chance (1/M)")
    ax.set(xticks=range(2,20), xlabel="Number of classes M", ylabel="Accuracy (%)", ylim=(0,102))
    ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.tight_layout(); save_figure("accuracy_vs_classes")

    fig, ax = plt.subplots(figsize=(7, 5.7))
    im = ax.imshow(full_cm / full_cm.sum(axis=1, keepdims=True)*100, cmap="Blues", vmin=0, vmax=100)
    ax.set(xticks=range(20), yticks=range(20),
           xticklabels=[f"{i+1}" for i in range(20)], yticklabels=classes.action,
           xlabel="Predicted class label", ylabel="Actual action")
    ax.tick_params(labelsize=7)
    fig.colorbar(im, ax=ax, label="Row-normalized recall (%)")
    fig.tight_layout(); save_figure("multiclass_confusion")

    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    ax.bar(range(1,5), np.array(loso)*100, color="#af5334")
    ax.axhline(5, linestyle="--", color="#888888", label="20-class chance (5%)")
    ax.set(xticks=range(1,5), xlabel="Held-out subject", ylabel="Accuracy (%)", ylim=(0,100))
    ax.legend(fontsize=7); fig.tight_layout(); save_figure("subject_holdout")

    fig, ax = plt.subplots(figsize=(6.6, 2.6)); ax.axis("off")
    boxes = [(0.02,.66,"80 unique recordings\n4 subjects · 20 actions"),
             (.37,.66,"15 equal intervals/file\n3 segments/interval"),
             (.72,.66,"Mean + sample variance\n8 channels → 48 features"),
             (.72,.12,"Choose classes\nStratified 10-fold split"),
             (.37,.12,"Training-fold scaling\nEuclidean 5-NN"),
             (.02,.12,"Held-out predictions\nAccuracy + confusion")]
    for x,z,label in boxes:
        ax.text(x+.12,z+.08,label,ha="center",va="center",fontsize=8,
                bbox=dict(boxstyle="round,pad=.7",fc="#eaf1f5",ec="#23688a"),transform=ax.transAxes)
    for p,q in [((.28,.74),(.35,.74)),((.63,.74),(.70,.74)),
                ((.84,.56),(.84,.38)),((.70,.20),(.63,.20)),((.35,.20),(.28,.20))]:
        ax.annotate("",xy=q,xytext=p,xycoords="axes fraction",arrowprops=dict(arrowstyle="->"))
    save_figure("pipeline")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Ignore completed batch checkpoints")
    args = parser.parse_args()
    config = json.loads((ROOT / "config.json").read_text())
    for folder in (OUT, RESULTS, FIGURES, OUT / "cache"):
        folder.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with threadpool_limits(limits=1):
        X, y, subjects, classes, manifest = load_dataset(config)
        plan = make_plan(y, config)
        fingerprint = hashlib.sha256(X.tobytes()+y.tobytes()+json.dumps(config,sort_keys=True).encode()
                                     +Path(__file__).read_bytes()+sklearn.__version__.encode()).hexdigest()
        acc = np.empty((18, config["n_combinations"], config["n_mc"]))
        for mi, m in enumerate(range(2,20)):
            checkpoint = OUT / "cache" / f"{fingerprint[:16]}_M{m:02d}.npy"
            if checkpoint.exists() and not args.force:
                acc[mi] = np.load(checkpoint)
                print(f"M={m:2d}: restored checkpoint", flush=True)
                continue
            for ci in range(config["n_combinations"]):
                mask = np.isin(y, plan["selected_labels"][mi,ci,:m])
                for mc in range(config["n_mc"]):
                    folds = plan["fold_ids"][mi,ci,mc,mask]
                    acc[mi,ci,mc], _ = evaluate(X[mask],y[mask],folds,config["n_neighbors"])
                    time.sleep(config["pause_seconds"])
                time.sleep(config["batch_pause_seconds"])
            np.save(checkpoint, acc[mi])
            print(f"M={m:2d}: mean accuracy {acc[mi].mean()*100:.2f}% (single CPU thread)", flush=True)
        binary_mask = np.isin(y, config["binary_labels"])
        binary_scores, binary_f1, full_scores, full_f1 = [], [], [], []
        binary_cm, full_cm = np.zeros((2,2),int), np.zeros((20,20),int)
        predictions = []
        for mc in range(config["n_mc"]):
            for kind, mask, folds in (("binary",binary_mask,plan["binary_fold_ids"][mc,binary_mask]),
                                      ("full",np.ones(len(y),bool),plan["full_fold_ids"][mc])):
                score, pred = evaluate(X[mask], y[mask], folds, config["n_neighbors"])
                f1 = f1_score(y[mask],pred,average="macro",zero_division=0)
                if kind == "binary":
                    binary_scores.append(score); binary_f1.append(f1)
                    binary_cm += confusion_matrix(y[mask],pred,labels=config["binary_labels"])
                else:
                    full_scores.append(score); full_f1.append(f1)
                    full_cm += confusion_matrix(y[mask],pred,labels=range(1,21))
                predictions.extend(dict(experiment=kind,mc=mc+1,row=int(idx+1),fold=int(fold),
                                        actual=int(label),predicted=int(pr))
                                   for idx,fold,label,pr in zip(np.flatnonzero(mask),folds,y[mask],pred))
                time.sleep(config["pause_seconds"])
        loso, loso_cm = [], np.zeros((20,20),int)
        for subject in range(1,5):
            train_X, test_X = standardize_train_test(X[subjects!=subject],X[subjects==subject])
            model = KNeighborsClassifier(n_neighbors=config["n_neighbors"],algorithm="brute",n_jobs=1)
            pred = model.fit(train_X,y[subjects!=subject]).predict(test_X)
            loso.append(float(accuracy_score(y[subjects==subject],pred)))
            loso_cm += confusion_matrix(y[subjects==subject],pred,labels=range(1,21))
        pd.DataFrame(predictions).to_csv(RESULTS / "out_of_fold_predictions.csv",index=False)
        records = []
        for mi,m in enumerate(range(2,20)):
            for ci in range(config["n_combinations"]):
                for mc in range(config["n_mc"]):
                    records.append(dict(M=m,combination=ci+1,mc=mc+1,
                                        rank=int(plan["combination_ranks"][mi,ci]),
                                        labels=" ".join(map(str,plan["selected_labels"][mi,ci,:m])),
                                        accuracy=float(acc[mi,ci,mc])))
        pd.DataFrame(records).to_csv(RESULTS / "accuracy_runs.csv",index=False)
        means = acc.mean(axis=(1,2))
        curve = pd.DataFrame(dict(M=np.arange(2,20),mean_accuracy=means,
                                 sd_subset_mean=acc.mean(axis=2).std(axis=1,ddof=1),
                                 sd_mc_mean=acc.mean(axis=1).std(axis=1,ddof=1),
                                 chance=1/np.arange(2,20)))
        curve.to_csv(RESULTS / "accuracy_curve.csv",index=False)
        drops = means[:-1]-means[1:]
        worst = int(np.argmax(drops))
        # A stated descriptive threshold, not a statistically established change point.
        rapid = np.flatnonzero(drops >= .05)
        summary = dict(config=config,shape=list(X.shape),unique_files=len(manifest),
                       min_length=int(manifest.n_samples.min()),max_length=int(manifest.n_samples.max()),
                       total_time_samples=int(manifest.n_samples.sum()),
                       binary_mean=float(np.mean(binary_scores)),binary_sd=float(np.std(binary_scores,ddof=1)),
                       binary_macro_f1=float(np.mean(binary_f1)),binary_scores=binary_scores,
                       full_mean=float(np.mean(full_scores)),full_sd=float(np.std(full_scores,ddof=1)),
                       full_macro_f1=float(np.mean(full_f1)),full_scores=full_scores,
                       loso_scores=loso,loso_mean=float(np.mean(loso)),
                       largest_drop_from_M=worst+2,largest_drop_to_M=worst+3,
                       largest_drop_percentage_points=float(drops[worst]*100),
                       rapid_drop_threshold_pp=5,
                       first_rapid_drop_to_M=int(rapid[0]+3) if len(rapid) else None,
                       input_fingerprint=fingerprint,python=platform.python_version(),
                       platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__,
                       sklearn=sklearn.__version__,threadpools=threadpool_info())
        for name, cm in (("binary_confusion",binary_cm),("full_confusion",full_cm),("loso_confusion",loso_cm)):
            np.savetxt(RESULTS / (name+".csv"),cm,fmt="%d",delimiter=",")
        pd.DataFrame(dict(subject=range(1,5),accuracy=loso)).to_csv(RESULTS / "subject_holdout.csv",index=False)
        savemat(RESULTS / "classification_results.mat",dict(accuracy=acc,binary_scores=binary_scores,
                full_scores=full_scores,binary_confusion=binary_cm,full_confusion=full_cm,
                loso_scores=loso),do_compression=True)
        figures(X,y,subjects,classes,manifest,config,acc,binary_cm,full_cm,loso)
        summary["elapsed_seconds"] = time.monotonic()-start
        (RESULTS / "summary.json").write_text(json.dumps(summary,indent=2))
        print(json.dumps({k:summary[k] for k in ("binary_mean","full_mean","loso_mean","elapsed_seconds")},indent=2))


if __name__ == "__main__":
    main()
