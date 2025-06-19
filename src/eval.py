from itertools import combinations
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm.auto import tqdm

from model import ParametricSurvivalModel


def t_auc(
    model: ParametricSurvivalModel,
    ts: np.ndarray,
    X: np.ndarray,
    T: np.ndarray,
    C: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Calculate the time-dependent AUC for survival data.

    At each time point t, each subject has either had an event, not yet had an event, or is censored.
    We don't try to predict the censoring times, so we only consider the uncensored subjects.
    A ParametricSurvivalModel cdf gives the probability of an event occurring before time t.
    The AUC is calculated as the area under the ROC curve for the predicted probabilities of the
    event occurring before time t, compared to the true event indicator (1 if event occurred, 0 if not).

    Args:
        model (ParametricSurvivalModel): The survival model to use for predictions.
        ts (np.ndarray): Time points at which to evaluate the AUC.
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        T (np.ndarray): True event times of shape (n_samples,).
        C (Optional[np.ndarray]): Censoring times of shape (n_samples,). If None, no censoring is considered.

    Returns:
        np.ndarray: Array of AUC values for each time point in `ts`.
    """
    n = X.shape[0]
    ts = np.tile(ts[:, np.newaxis], (1, n))
    params = model.param_mapping.map(X)
    d_pred = model.dist_type.cdf(ts, params)
    d_true = (ts > T).astype(int)
    if C is None:
        mask = np.ones_like(ts, dtype=bool)
    else:
        # Mask out censored subjects. If an event occurs before the censoring time, that subject is not censored.
        mask = (ts < C) | (T < C)
    aucs = np.array(
        [
            roc_auc_score(d_true[i][mask[i]], d_pred[i][mask[i]])
            if mask[i].sum() > 0
            else np.nan
            for i in range(len(ts))
        ]
    )
    return aucs


# Predictions:
#  - Y_pred: median survival time (uncensored)
#  - D_pred: Y_pred > C, or p(Y_pred > C) < 0.5
# Metrics:
#  - Binary classifcation metrics for D_pred vs D
#    - Accuracy, Precision, Recall, F1 Score
#    - ROC AUC
#  - Survival analysis metrics for Y_pred vs C
#    - Concordance index (C-index)
def binary_classification_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


def concordance_index(y_true, y_pred):
    """
    Calculate the concordance index (C-index) for survival data.
    C-index is the proportion of all pairs of subjects whose predicted survival times
    are correctly ordered.
    """
    # Ensure y_true and y_pred are 1D arrays
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Count concordant pairs
    concordant_pairs = 0
    total_pairs = 0

    n_pairs = len(y_true) * (len(y_true) - 1) // 2
    if n_pairs > 1e6:
        # Only take the first 1 million pairs for performance reasons
        n_pairs = 1e6
    pairs = combinations(range(len(y_true)), 2)
    for i, j in tqdm(pairs, desc="Calculating C-index", total=n_pairs):
        if total_pairs >= n_pairs:
            break
        if (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]) or (
            y_true[i] > y_true[j] and y_pred[i] > y_pred[j]
        ):
            concordant_pairs += 1
        total_pairs += 1

    return concordant_pairs / total_pairs if total_pairs > 0 else np.nan


def plot_roc(y_true, y_scores, title="ROC Curve"):
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name="ROC Curve",
            line=dict(color="blue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Guessing",
            line=dict(color="red", dash="dash"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=600,
        height=400,
    )
    return fig


# Plot the confusion matrix for training data
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(2, 2))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        ax=ax,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    return fig
