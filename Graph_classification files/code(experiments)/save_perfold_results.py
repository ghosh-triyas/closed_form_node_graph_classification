"""
save_perfold_results.py

Utility: every experiment script we've built throughout this project
prints per-fold accuracy to the terminal but never SAVES it -- meaning
we cannot go back and run a significance test after the fact. This
module provides one function, `run_crossval_saving_folds`, that is a
drop-in replacement for the run_crossval / run_single patterns used
everywhere else: it does the SAME 10-fold CV, but additionally saves
the raw per-fold accuracy array to a .npy file, keyed by
(dataset_name, method_name), so it can be loaded later for a real
paired significance test.

IMPORTANT HONEST CONSTRAINT: a Wilcoxon signed-rank test requires
PAIRED per-fold scores for BOTH methods being compared. This is only
possible when:
  (a) comparing two configurations of OUR OWN method (we have full
      per-fold control), or
  (b) comparing against a baseline we ALSO ran ourselves under the
      identical 10-fold split (e.g. our own WL kernel implementation,
      see run_wl_kernel_baseline.py from earlier in this project).

It is NOT possible against literature-reported baselines (GCN, GIN,
GraphSAGE as cited from papers), since those give only mean +/- std,
never the individual fold scores. Significance testing against those
specific baselines cannot be done validly with any script -- this is
a fundamental data availability limit, not a coding problem. State
this explicitly in the paper rather than fabricate or approximate it.
"""
import numpy as np
import os

RESULTS_DIR = '/Users/X/Desktop/TopoSPR/paper_outputs/data/perfold_results'
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_fold_scores(dataset_name: str, method_name: str, fold_scores: list):
    """
    Save a list/array of per-fold accuracy scores (one float per CV
    fold, in [0,1] or percentage -- be consistent) to disk, keyed by
    dataset and method name.
    """
    fold_scores = np.asarray(fold_scores, dtype=np.float64)
    path = os.path.join(RESULTS_DIR, f'{dataset_name}__{method_name}.npy')
    np.save(path, fold_scores)
    print(f"  Saved {len(fold_scores)} fold scores to {path}")
    return path


def load_fold_scores(dataset_name: str, method_name: str) -> np.ndarray:
    path = os.path.join(RESULTS_DIR, f'{dataset_name}__{method_name}.npy')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No saved fold scores for {dataset_name}/{method_name} at "
            f"{path}. Run the corresponding experiment with "
            f"save_fold_scores() first.")
    return np.load(path)


def run_crossval_saving_folds(X, labels, classifier_fn, dataset_name,
                                method_name, n_folds=10, seed=42):
    """
    Generic 10-fold CV runner that saves per-fold scores to disk.

    classifier_fn(X_tr, y_tr, X_val, y_val, X_te, y_te) -> float
        Should fit on (X_tr,y_tr), select hyperparams on (X_val,y_val),
        and return test accuracy on (X_te,y_te). This lets this
        function work with ANY of our classifiers (KernelRidgeClassifier,
        MultiKernelRidgeClassifier, SVC for WL, etc.) via a small
        lambda wrapper at the call site.
    """
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_scores = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, labels)):
        n_val = max(1, len(tr_idx) // 10)
        val_idx, tr_idx2 = tr_idx[:n_val], tr_idx[n_val:]
        acc = classifier_fn(X[tr_idx2], labels[tr_idx2],
                             X[val_idx], labels[val_idx],
                             X[te_idx], labels[te_idx])
        fold_scores.append(acc)
        print(f"    Fold {fold+1:2d}: {acc*100:.2f}%")

    mean, std = np.mean(fold_scores)*100, np.std(fold_scores)*100
    print(f"  {method_name} on {dataset_name}: {mean:.2f}% +/- {std:.2f}%")
    save_fold_scores(dataset_name, method_name, fold_scores)
    return fold_scores
