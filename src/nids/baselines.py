"""
Baseline NIDS Models and Statistical Comparative Testing.
Implements:
- Traffic-Only MLP (no text prototypes)
- Random Forest Baseline (pure NumPy or Scikit-Learn wrapper)
- McNemar's Paired Chi-Squared Test with continuity correction
"""

import math
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TrafficOnlyMLP(nn.Module):
    """
    Supervised MLP classifier trained solely on network traffic features.
    """

    def __init__(
        self,
        input_dim: int = 42,
        hidden_dim: int = 128,
        num_classes: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        reliability, pred = torch.max(probs, dim=-1)
        return pred, reliability, probs


class BaselineRandomForest:
    """
    Random Forest classifier baseline for tabular network flows.
    Uses sklearn if available, or a fast ensemble of randomized axis-aligned decision trees in NumPy.
    """

    def __init__(self, n_estimators: int = 50, max_depth: int = 12, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self._sklearn_rf = None
        self._trees = []
        self.num_classes = 10

        try:
            from sklearn.ensemble import RandomForestClassifier
            self._sklearn_rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1,
            )
        except ImportError:
            self._sklearn_rf = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self._sklearn_rf is not None:
            self._sklearn_rf.fit(X, y)
            self.num_classes = len(self._sklearn_rf.classes_)
            return self

        # Native fallback: Random decision forest ensemble
        rng = np.random.default_rng(self.random_state)
        self.num_classes = int(np.max(y) + 1)
        self._trees = []
        n_samples, n_features = X.shape

        for t in range(self.n_estimators):
            boot_idx = rng.integers(0, n_samples, size=n_samples)
            X_b = X[boot_idx]
            y_b = y[boot_idx]
            tree = self._build_tree(X_b, y_b, depth=0, rng=rng)
            self._trees.append(tree)
        return self

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int, rng: np.random.Generator):
        n_samples, n_features = X.shape
        counts = np.bincount(y, minlength=self.num_classes)
        prob = counts / max(1, n_samples)

        if depth >= self.max_depth or n_samples < 5 or np.max(counts) == n_samples:
            return {"type": "leaf", "prob": prob}

        # Subsample features
        feat_subsample = rng.choice(n_features, size=int(np.sqrt(n_features)), replace=False)
        best_feat = feat_subsample[0]
        best_thresh = np.median(X[:, best_feat])

        mask = X[:, best_feat] <= best_thresh
        if np.sum(mask) == 0 or np.sum(~mask) == 0:
            return {"type": "leaf", "prob": prob}

        left = self._build_tree(X[mask], y[mask], depth + 1, rng)
        right = self._build_tree(X[~mask], y[~mask], depth + 1, rng)
        return {"type": "node", "feat": best_feat, "thresh": best_thresh, "left": left, "right": right}

    def _predict_sample(self, tree: dict, x: np.ndarray) -> np.ndarray:
        if tree["type"] == "leaf":
            return tree["prob"]
        if x[tree["feat"]] <= tree["thresh"]:
            return self._predict_sample(tree["left"], x)
        return self._predict_sample(tree["right"], x)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._sklearn_rf is not None:
            return self._sklearn_rf.predict_proba(X)
        probs = np.zeros((len(X), self.num_classes))
        for tree in self._trees:
            t_probs = np.array([self._predict_sample(tree, x) for x in X])
            probs += t_probs
        probs /= len(self._trees)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=-1)


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> Tuple[float, float, Dict[str, int]]:
    """
    McNemar's test with continuity correction comparing two classification models:
    chi2 = (|b - c| - 1)^2 / (b + c)
    where:
      b: correct in A, incorrect in B
      c: incorrect in A, correct in B
    """
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    a = np.sum(correct_a & correct_b)
    b = np.sum(correct_a & ~correct_b)
    c = np.sum(~correct_a & correct_b)
    d = np.sum(~correct_a & ~correct_b)

    discordant = b + c
    if discordant == 0:
        return 0.0, 1.0, {"a": int(a), "b": int(b), "c": int(c), "d": int(d)}

    chi2 = (abs(b - c) - 1.0) ** 2 / discordant

    # Approximation of chi2 1-DOF survival function P(X >= chi2) = 2 * (1 - Phi(sqrt(chi2)))
    z = math.sqrt(chi2)
    # Complementary error function
    p_val = math.erfc(z / math.sqrt(2.0))

    return float(chi2), float(p_val), {"a": int(a), "b": int(b), "c": int(c), "d": int(d)}
