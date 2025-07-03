from itertools import combinations
from typing import Optional

import altair as alt
import numpy as np
import polars as pl
import torch as t
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm.auto import tqdm

from model import ParametricSurvivalModel
from vis import (
    plot_confusion_matrix,
    plot_likelihoods,
    plot_params,
    plot_params_3d,
    plot_params_by_d,
    plot_roc,
    plot_t_auc,
)


def show_evals(
    model: ParametricSurvivalModel,
    x_train: t.Tensor,
    y_train: t.Tensor,
    c_train: t.Tensor,
    d_train: t.Tensor,
    x_test: t.Tensor,
    y_test: t.Tensor,
    c_test: t.Tensor,
    d_test: t.Tensor,
):
    # Make predictions with the fitted model
    pred_df = model.predict(x_train, y_train, c_train, d_train)
    test_pred_df = model.predict(x_test, y_test, c_test, d_test)

    plot_likelihoods(test_pred_df).show()

    param_names = list(model.mapping.param_transforms.keys())
    plot_params(test_pred_df, param_names).show()
    plot_params_by_d(test_pred_df, param_names).show()
    plot_params_3d(test_pred_df, param_names).show()

    # # TODO plot average over class (either average the curve or the params)
    # plot_samples_by_d(test_df, model).show()

    # Calculate metrics for training data
    train_metrics = binary_classification_metrics(
        pred_df["D"].to_numpy(), pred_df["D_pred"].to_numpy() > 0.5
    )
    # Calculate metrics for test data
    test_metrics = binary_classification_metrics(
        test_pred_df["D"].to_numpy(), test_pred_df["D_pred"].to_numpy() > 0.5
    )
    # Calculate C-index for training data
    train_c_index = concordance_index(
        pred_df["Y"].to_numpy(), pred_df["Y_pred"].to_numpy()
    )
    # Calculate C-index for test data
    test_c_index = concordance_index(
        test_pred_df["Y"].to_numpy(), test_pred_df["Y_pred"].to_numpy()
    )
    # Calculate time-dependent AUC for test data
    ts = np.arange(200, 2500, 50)
    t_aucs = t_auc(
        model, ts, x_test.cpu().numpy(), y_test.cpu().numpy(), c_test.cpu().numpy()
    )
    aucs_df = pl.DataFrame({"Time": ts, "AUC": t_aucs.flatten()})

    print("Training Metrics:")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.4f}")
    print(f"  C-index: {train_c_index:.4f}")
    print("\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    print(f"  C-index: {test_c_index:.4f}")
    print("\nTime-dependent AUCs:")
    print(aucs_df)

    plot_roc(
        pred_df["D"].to_numpy(),
        pred_df["D_pred"].to_numpy(),
        title="ROC Curve (Train)",
    ).show()
    plot_roc(
        test_pred_df["D"].to_numpy(),
        test_pred_df["D_pred"].to_numpy(),
        title="ROC Curve (Test)",
    ).show()

    plot_t_auc(aucs_df).show()

    plot_confusion_matrix(
        pred_df["D"].to_numpy(),
        pred_df["D_pred"].to_numpy() > 0.5,
        title="Confusion Matrix (Train)",
    ).show()
    plot_confusion_matrix(
        test_pred_df["D"].to_numpy(),
        test_pred_df["D_pred"].to_numpy() > 0.5,
        title="Confusion Matrix (Test)",
    ).show()


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
    params = model.mapping.forward(t.tensor(X, device=model.device, dtype=t.float32))
    d_pred = (
        model.dist_type(**params).cdf(t.tensor(ts, device=model.device)).detach().cpu()
    )
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
