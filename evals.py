"""
Tools for evaluating molecule selection models, binary/ranking.
"""

import functools

import numpy as np
import pandas as pd
import sklearn.base
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
KS = [5, 10, 30]


def hits_at_k(y_true, y_scores, k):
    # Expects y_scores (raw probabilities/scores)
    if k <= 0:
        raise ValueError("k must be positive.")

    y_true_arr = np.asarray(y_true)
    y_scores_arr = np.asarray(y_scores)

    positive_sample_indices = np.where(y_true_arr == 1)[0]
    if len(positive_sample_indices) == 0:
        return 0.0

    hits = 0
    for i in positive_sample_indices:
        score_of_positive_sample = y_scores_arr[i]
        # Rank: 1 + number of items with score > score_of_positive_sample
        rank = np.sum(y_scores_arr > score_of_positive_sample) + 1
        if rank <= k:
            hits += 1

    return hits / len(positive_sample_indices)


def precision_at_k(y_true, y_scores, k):
    # Expects y_scores (raw probabilities/scores)
    if k <= 0:
        raise ValueError("k must be positive.")

    y_true_arr = np.asarray(y_true)
    y_scores_arr = np.asarray(y_scores)

    num_samples = len(y_true_arr)
    actual_k = min(k, num_samples)  # Cap k at the number of samples
    if actual_k == 0:
        return 0.0

    # Indices of top-k highest scores
    top_k_indices = np.argsort(y_scores_arr)[-actual_k:]

    true_positives_in_top_k = np.sum(y_true_arr[top_k_indices] == 1)
    return true_positives_in_top_k / actual_k


def mrr(y_true, y_scores):  # Mean Reciprocal Rank
    # Expects y_scores (raw probabilities/scores)
    y_true_arr = np.asarray(y_true)
    y_scores_arr = np.asarray(y_scores)

    positive_sample_indices = np.where(y_true_arr == 1)[0]
    if len(positive_sample_indices) == 0:
        return 0.0

    reciprocal_ranks = []
    for i in positive_sample_indices:
        score_of_positive_sample = y_scores_arr[i]
        # Rank: 1 + number of items with score > score_of_positive_sample
        rank = np.sum(y_scores_arr > score_of_positive_sample) + 1
        reciprocal_ranks.append(1.0 / rank)

    if not reciprocal_ranks:  # Should only happen if positive_sample_indices was empty
        return 0.0
    return np.mean(reciprocal_ranks)


def auprc(y_true, y_scores):  # Area Under Precision-Recall Curve
    # Expects y_scores (raw probabilities/scores)
    y_true_arr = np.asarray(y_true)
    y_scores_arr = np.asarray(y_scores)

    # AUPRC is typically not well-defined or is 0 if there are no positive samples.
    # sklearn's precision_recall_curve handles cases with one class gracefully.
    # If only negatives, precision is 0, recall is 0. AUC will be 0.
    # If only positives, precision is 1, recall is 1. AUC will be 1. (This case is less common for PR curve utility)
    if np.sum(y_true_arr == 1) == 0:  # No positive samples
        return 0.0  # Or np.nan, depending on preference. 0.0 is common.
    if len(np.unique(y_true_arr)) < 2 and np.all(
        y_true_arr == 1
    ):  # All positive samples
        return 1.0  # Or np.nan

    precision_points, recall_points, _ = precision_recall_curve(
        y_true_arr, y_scores_arr
    )
    # The AUC function expects recall (x-axis) then precision (y-axis)
    return auc(recall_points, precision_points)


metrics_dict = {
    "accuracy": accuracy_score,
    "balanced_accuracy": balanced_accuracy_score,
    "AUROC": roc_auc_score,
    "AUPRC": auprc,
    "precision": precision_score,
    "recall": recall_score,
    "mean_reciprocal_rank": mrr,
}
for k in KS:
    metrics_dict[f"hits_at_{k}"] = functools.partial(hits_at_k, k=k)
    metrics_dict[f"precision_at_{k}"] = functools.partial(precision_at_k, k=k)


def evaluate_predictions(y_true, y_pred) -> dict[str, float]:
    """Evaluate predictions using various metrics."""
    results = {}
    for metric_name, metric_fn in metrics_dict.items():
        results[metric_name] = metric_fn(y_true, y_pred)
    return results


def bootstrap_estimates(y_true, y_pred, metric_fn, n_iterations=1000, **metric_kwargs):
    """Bootstrap estimator with replacement for evaluating various metrics.

    This function applies bootstrap resampling with replacement to estimate a given
    metric function on the provided true labels and predictions. It is designed to be
    flexible and work with any metric function that accepts resampled y_true and y_pred
    (or data that can be resampled row-wise).

    """
    n_samples = y_true.shape[0]
    metric_values = []
    for _ in range(n_iterations):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_bootstrap = y_true[indices]
        y_pred_bootstrap = y_pred[indices]
        metric_value = metric_fn(y_true_bootstrap, y_pred_bootstrap, **metric_kwargs)
        metric_values.append(metric_value)
    return np.array(metric_values)


def boostrap_ci(y_true, y_pred, metric_fn, ci=0.95, n_iterations=1000, **metric_kwargs):
    """Calculate confidence intervals for a given metric using bootstrap resampling."""
    values = bootstrap_estimates(
        y_true, y_pred, metric_fn, n_iterations=n_iterations, **metric_kwargs
    )
    lower_percentile = (1 - ci) / 2 * 100
    upper_percentile = 100 - lower_percentile
    lower_bound = np.percentile(values, lower_percentile)
    upper_bound = np.percentile(values, upper_percentile)
    return np.mean(values), lower_bound, upper_bound


def get_baseline_models() -> dict[str, sklearn.base.BaseEstimator]:
    """Return a dictionary with baseline models for binary classification."""
    model_dict = {
        "stratified_dummy": DummyClassifier(strategy="stratified"),
        "most_frequent_dummy": DummyClassifier(strategy="most_frequent"),
        "uniform_dummy": DummyClassifier(strategy="uniform"),
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "decision_tree": DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=4, random_state=RANDOM_STATE
        ),
        "linear_svc": SVC(kernel="linear", random_state=RANDOM_STATE, probability=True),
    }
    return model_dict


class BinaryEvaluator:
    def __init__(self, X, y):

        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            raise TypeError("X must be a numpy array or pandas DataFrame")
        if not isinstance(y, (np.ndarray, pd.Series)):
            raise TypeError("y must be a numpy array or pandas Series")

        self.X = X.copy()
        self.y = y.copy()

        if not np.all(np.unique(self.y) == np.array([0, 1])):
            raise ValueError("Labels must be binary (0 or 1)")

        self.base_res = None

    def accuracy(self, yt, yp, thr=0.5):

        pred = (yp >= thr).astype(int)
        return accuracy_score(yt, pred)

    def balanced_accuracy(self, yt, yp, thr=0.5):

        pred = (yp >= thr).astype(int)
        return balanced_accuracy_score(yt, pred)

    def roc_auc(self, yt, yp):

        return roc_auc_score(yt, yp)

    def precision(self, yt, yp, thr=0.5):

        pred = (yp >= thr).astype(int)
        return precision_score(yt, pred, zero_division=0)

    def recall(self, yt, yp, thr=0.5):

        pred = (yp >= thr).astype(int)
        return recall_score(yt, pred)

    def hits_at_k(self, yt, yp, k):

        if k <= 0:
            raise ValueError("k must be positive")

        pos_idx = np.where(yt == 1)[0]
        if len(pos_idx) == 0:
            return 0.0

        hits = 0
        for i in pos_idx:
            ps = yp[i]
            r = np.sum(yp > ps) + 1
            hits += 1 if r <= k else 0

        return hits / len(pos_idx)

    def precision_at_k(self, yt, yp, k):

        if k <= 0:
            raise ValueError("k must be positive")
        if k > len(yt):
            raise ValueError("k cannot be larger than the number of samples")

        topk = np.argsort(yp)[-k:]
        tp = np.sum(yt[topk] == 1)
        return tp / k

    def mrr(self, yt, yp):

        pos_idx = np.where(yt == 1)[0]
        if len(pos_idx) == 0:
            return 0.0

        rr = []
        for i in pos_idx:
            ps = yp[i]
            r = np.sum(yp > ps) + 1
            rr.append(1.0 / r)
        return np.mean(rr)

    def CV_model(self, model):
        # model has to have methods fit and predict_proba
        metrics = [
            "accuracy",
            "balanced_accuracy",
            "roc_auc",
            "precision",
            "recall",
            "mrr",
        ]

        scr = {}
        for metric in metrics:
            scr[metric] = make_scorer(
                getattr(self, metric), response_method="predict_proba"
            )

        k_values = [5, 10, 30]
        for k in k_values:
            for metric in ["precision_at_k", "hits_at_k"]:
                metric_name = f"{metric}_{k}"
                scr[metric_name] = make_scorer(
                    lambda y_true, y_pred, m=metric, k=k: getattr(self, m)(
                        y_true, y_pred, k
                    ),
                    response_method="predict_proba",
                )

        cv_results = cross_validate(model, self.X, self.y, cv=5, scoring=scr)

        # Calculate mean and std for each metric
        return {
            "mean": {
                metric: cv_results[f"test_{metric}"].mean() for metric in scr.keys()
            },
            "std": {
                metric: cv_results[f"test_{metric}"].std() for metric in scr.keys()
            },
        }

    def compute_metrics(self, yt, yp, klist=[5, 10, 30], thr=0.5):

        pred = (yp >= thr).astype(int)
        m = {
            "accuracy": self.accuracy(yt, yp, thr),
            "balanced_accuracy": self.balanced_accuracy(yt, yp, thr),
            "roc_auc": self.roc_auc(yt, yp),
            "precision": self.precision(yt, yp, thr),
            "recall": self.recall(yt, yp, thr),
            "mean_reciprocal_rank": self.mrr(yt, yp),
            "positives": np.sum(yt == 1),
            "predicted_positives": np.sum(pred == 1),
        }

        klist = list(klist) + [np.sum(yt == 1)]
        for k in klist:
            m[f"hits_at_{k}"] = self.hits_at_k(yt, yp, k)
            m[f"precision_at_{k}"] = self.precision_at_k(yt, yp, k)

        return m

    def available_metrics(self):
        print("Available metrics:")
        print("- accuracy (with threshold)")
        print("- balanced_accuracy (with threshold)")
        print("- roc_auc")
        print("- precision (with threshold)")
        print("- recall (with threshold)")
        print("- hits_at_k (requires k parameter)")
        print("- precision_at_k (requires k parameter)")
        print("- mean_reciprocal_rank (mrr)")
        print("\nAdditional statistics:")
        print("- positives (number of positive samples)")
        print("- predicted_positives (number of predicted positive samples)")
