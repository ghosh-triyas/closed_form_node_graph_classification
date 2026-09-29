"""
Multiclass Graph-SPR via One-vs-Rest.

Updated to pass edge_index and struct_start_idx through to GraphSPR.fit()
so the guided beam search has access to graph topology information.
"""

import numpy as np
from graph_spr import GraphSPR, build_design_matrix, ridge_solve


class MulticlassGraphSPR:
    """
    One-vs-Rest wrapper around GraphSPR.

    Usage:
        model = MulticlassGraphSPR(
            num_classes=7,
            mode='beam',        # 'beam' or 'greedy'
            beam_width=3,
            gamma=0.05,
            lam_nc=0.05,
        )
        model.fit(tilde_Theta, y, train_mask, val_mask,
                  edge_index=data.edge_index,
                  struct_start_idx=meta['struct_start_idx'])
        acc = model.evaluate(tilde_Theta, y, test_mask)
    """

    def __init__(self, num_classes,
                 K=4, r=0.5, max_paths=50,
                 max_support=1, max_freq=2,
                 lambda_grid=None, patience=5,
                 mode='beam',
                 beam_width=3,
                 gamma=0.05,
                 lam_nc=0.05,
                 crossover_interval=5,
                 verbose=True):

        self.num_classes = num_classes
        self.verbose     = verbose
        self.mode        = mode

        self.models = [
            GraphSPR(
                K=K, r=r,
                max_paths         = max_paths,
                max_support       = max_support,
                max_freq          = max_freq,
                lambda_grid       = lambda_grid,
                patience          = patience,
                mode              = mode,
                beam_width        = beam_width,
                gamma             = gamma,
                lam_nc            = lam_nc,
                crossover_interval= crossover_interval,
                verbose           = verbose,
            )
            for _ in range(num_classes)
        ]

    def fit(self, tilde_Theta, y, train_mask, val_mask,
            edge_index=None, struct_start_idx=None):
        """
        Train one binary classifier per class.

        Args:
            tilde_Theta:      [N, D] diffused angular coordinates ALL nodes
            y:                [N] integer class labels
            train_mask:       [N] bool
            val_mask:         [N] bool
            edge_index:       torch LongTensor [2, E]
                              Required for mode='beam'. Pass data.edge_index.
            struct_start_idx: int or None — first dim of structural block.
                              From meta['struct_start_idx'] returned by
                              build_diffused_angular_matrix.
        """
        for c in range(self.num_classes):
            if self.verbose:
                print(f"\n{'='*55}")
                print(f"Training class {c} vs rest  "
                      f"[mode={self.mode}]")
                print(f"{'='*55}")

            y_binary = (y == c).astype(np.int64)
            self.models[c].fit(
                tilde_Theta, y_binary, train_mask, val_mask,
                edge_index       = edge_index,
                struct_start_idx = struct_start_idx,
            )

        return self

    def predict_scores(self, tilde_Theta, mask=None):
        """Raw scores from all C classifiers. Shape [N_masked, C]."""
        scores = []
        for c in range(self.num_classes):
            raw = self.models[c].predict_raw(tilde_Theta, mask)
            scores.append(raw.reshape(-1, 1))
        return np.hstack(scores)

    def predict(self, tilde_Theta, mask=None):
        """Predicted class labels."""
        scores = self.predict_scores(tilde_Theta, mask)
        return np.argmax(scores, axis=1)

    def evaluate(self, tilde_Theta, y, mask):
        """Accuracy on masked nodes."""
        preds = self.predict(tilde_Theta, mask)
        return np.mean(preds == y[mask])

    def path_summary(self, struct_start_idx=None, feature_names=None):
        for c in range(self.num_classes):
            print(f"\nClass {c} model:")
            self.models[c].path_summary(struct_start_idx, feature_names)
