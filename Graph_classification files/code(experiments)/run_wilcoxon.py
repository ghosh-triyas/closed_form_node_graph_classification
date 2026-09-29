"""
run_wilcoxon.py -- paired Wilcoxon signed-rank test, PHS vs. our
(now-fixed) WL reimplementation, for Table 5.

IMPORTANT: for this test to be valid, PHS_FOLDS[dataset][i] and
WL_FOLDS[dataset][i] must be the SAME fold i -- same train/test split,
same seed, same stratification. If your PHS pipeline and
wl_kernel_rerun.py used different seeds, these are not paired and this
test is not valid as written -- rerun one of them with a matching seed
before trusting the p-values below.

WL_FOLDS below are copied directly from your corrected
wl_kernel_rerun.py terminal output. PHS_FOLDS are placeholders --
replace with your actual per-fold PHS scores (same units, percentage).
"""
import numpy as np
from scipy.stats import wilcoxon

# --- WL (ours), corrected -- copied from your terminal output ---
WL_FOLDS = {
    "MUTAG":    [89.47, 78.95, 78.95, 89.47, 84.21, 89.47, 84.21, 78.95, 88.89, 77.78],
    "PTC-MR":   [57.14, 68.57, 60.00, 54.29, 61.76, 64.71, 67.65, 67.65, 41.18, 64.71],
    "DHFR":     [73.68, 85.53, 80.26, 78.95, 77.63, 84.21, 89.33, 82.67, 86.67, 84.00],
    "NCI1":     [84.67, 82.00, 84.18, 84.91, 84.67, 83.21, 83.45, 86.86, 80.78, 88.81],
    "PROTEINS": [77.68, 75.89, 77.68, 65.77, 77.48, 72.07, 72.97, 81.08, 77.48, 72.07],
}

# --- PHS -- REPLACE THESE with your actual per-fold scores from the
# same 10 splits (e.g. from save_fold_scores('MUTAG', 'phs', ...) if
# you saved them, or from your original run's per-fold printout) ---
PHS_FOLDS = {
    "MUTAG":    None,  # 10 numbers, must sum/average to ~91.15
    "PTC-MR":   None,  # ~64.31
    "DHFR":     None,  # ~75.95
    "NCI1":     None,  # ~74.79
    "PROTEINS": None,  # ~71.34
}

TARGET_MEAN = {  # sanity check: your filled-in PHS_FOLDS should average close to these
    "MUTAG": 91.15, "PTC-MR": 64.31, "DHFR": 75.95, "NCI1": 74.79, "PROTEINS": 71.34,
}


def run():
    print(f"{'Dataset':<10} {'PHS':>7} {'WL(ours)':>9} {'p':>8}  Result")
    print("-" * 50)
    for dataset in WL_FOLDS:
        phs = PHS_FOLDS[dataset]
        if phs is None:
            print(f"{dataset:<10} [SKIPPED -- fill in PHS_FOLDS['{dataset}'] first]")
            continue
        phs = np.asarray(phs, dtype=float)
        wl = np.asarray(WL_FOLDS[dataset], dtype=float)
        assert len(phs) == len(wl) == 10, \
            f"{dataset}: need exactly 10 paired folds, got {len(phs)} PHS / {len(wl)} WL"

        phs_mean_check = phs.mean()
        target = TARGET_MEAN[dataset]
        if abs(phs_mean_check - target) > 0.5:
            print(f"  [WARNING] {dataset}: PHS_FOLDS mean is {phs_mean_check:.2f}, "
                  f"expected ~{target} -- check you pasted the right numbers")

        diffs = phs - wl
        if np.allclose(diffs, 0):
            p = 1.0
        else:
            try:
                stat, p = wilcoxon(phs, wl)
            except ValueError as e:
                # wilcoxon raises if all diffs are zero or too few non-tied pairs
                print(f"  [ERROR] {dataset}: {e}")
                continue

        if p < 0.05:
            result = "PHS > WL" if phs.mean() > wl.mean() else "WL > PHS"
        else:
            result = "n.s."

        print(f"{dataset:<10} {phs.mean():>7.2f} {wl.mean():>9.2f} {p:>8.3f}  {result}")


if __name__ == "__main__":
    run()