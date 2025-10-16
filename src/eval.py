# %%
from itertools import combinations
from typing import Optional

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm.auto import tqdm
import plotly.graph_objects as go
import matplotlib.pyplot as plt

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
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    c_train: torch.Tensor,
    d_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    c_test: torch.Tensor,
    d_test: torch.Tensor,
):
    # Make predictions with the fitted model
    pred_df = model.predict(x_train, y_train, c_train, d_train)
    test_pred_df = model.predict(x_test, y_test, c_test, d_test)

    plot_likelihoods(test_pred_df).show()

    param_names = list(model.mapping.param_transforms.keys())
    plot_params(test_pred_df, param_names).show()
    plot_params_by_d(test_pred_df, param_names).show()
    # plot_params_3d(test_pred_df, param_names).show()

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
    train_c_index = calculate_c_index(
        pred_df["Y"].to_numpy(), pred_df["Y_pred"].to_numpy(), pred_df["D"].to_numpy()
    )
    # Calculate C-index for test data
    test_c_index = calculate_c_index(
        test_pred_df["Y"].to_numpy(), test_pred_df["Y_pred"].to_numpy(), test_pred_df["D"].to_numpy()
    )
    # Calculate time-dependent AUC for test data
    ts = np.arange(200, 2500, 10)
    aucs_df = t_auc(
        model, ts, x_test.cpu().numpy(), y_test.cpu().numpy(), c_test.cpu().numpy()
    )
    aucs_df.plot(x="Time", y=["aucs", "aucs_without_c", "frac_censored", "frac_true_diagnosed (x100)", "frac_pred_diagnosed"])
    plt.xlabel("Time")
    plt.ylabel("AUC")
    plt.title("AUC vs Time")
    plt.show()


    print("Training Metrics:")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.4f}")
    print(f"  C-index: {train_c_index:.4f}")
    print("\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    print(f"  C-index: {test_c_index:.4f}")

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

    # plot_t_auc(aucs_df).show()

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


@torch.no_grad()
def t_auc(model, ts, x, y, c):
    mask = ts < c[:, None]
    d_true = (y[:, None] < ts) & (y < c)[:, None]

    model.eval()
    x_tensor = torch.tensor(x, device=model.device, dtype=torch.float32)
    p = model.mapping(x_tensor)
    ts_tensor = torch.tensor(
        np.minimum(ts[:, None], c[None, :]),
        device=model.device,
        dtype=torch.float32,
    )
    d_pred = model.dist_type(**p).cdf(ts_tensor).T.detach().cpu().numpy()

    aucs = np.array([roc_auc_score(d_true[:, i], d_pred[:, i]) for i in range(len(ts))])
    aucs_without_c = np.array(
        [
            roc_auc_score(d_true[:, i][mask[:, i]], d_pred[:, i][mask[:, i]])
            if np.sum(mask[:, i]) > 0
            else np.nan
            for i in range(len(ts))
        ]
    )
    frac_censored = np.mean(~mask & (d_true == 0), axis=0)
    frac_true_diagnosed = np.mean(d_true, axis=0)
    frac_pred_diagnosed = np.mean(d_pred > 0.5, axis=0)
    return pd.DataFrame({
        "Time": ts,
        "aucs": aucs,
        "aucs_without_c": aucs_without_c,
        "frac_censored": frac_censored,
        "frac_true_diagnosed (x100)": frac_true_diagnosed * 100,
        "frac_pred_diagnosed": frac_pred_diagnosed,
    })


def t_auc_old(
    model: ParametricSurvivalModel,
    ts: np.ndarray,
    X: np.ndarray,
    T: np.ndarray,
    C: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Calculate the time-dependent AUC for survival data.

    At each time point t, each subject is either resolved (diagnosed/censored) or unresolved (not yet d/c).
    A ParametricSurvivalModel cdf F(t) gives the probability of an event occurring before time t.
    The AUC is calculated as the area under the ROC curve for the predicted probabilities of the
    event occurring before time t, compared to the true event indicator (1 if event occurred, 0 if not).
    If the censoring time c < t, then we use the cdf value at c.

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

    x = torch.tensor(X, device=model.device, dtype=torch.float32)
    params = model.mapping(x)

    if C is None:
        C = np.full(n, np.inf)

    ts_tensor = torch.tensor(ts, device=model.device, dtype=torch.float32)
    d_pred = (
        model.dist_type(**params).cdf(ts_tensor).detach().cpu().numpy()
    )
    mask = ts < C
    d_pred = np.where(mask[None, :], d_pred, np.nan)
    

    d_true = T[:, None] < ts

    aucs = np.array(
        [
            roc_auc_score(d_true[:, i], d_pred[:, i])
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


def calculate_c_index(y_true, y_pred, d_true, n_samples=None) :
    """Calculate concordance index for survival predictions.
    Args:
        y_true (np.ndarray): True survival times.
        y_pred (np.ndarray): Predicted survival times.
        d (np.ndarray): Event indicators (1 if event occurred, 0 if censored).
    Returns:
        float: Concordance index.
    """
    n = len(y_true)
    permissible = 0
    concordant = 0

    i_d = np.where(d_true)[0]
    if len(i_d) == 0:
        return np.nan

    if n_samples is None:
        n_samples = i_d.shape[0] * (n - 1)

    while permissible < n_samples:
        i = np.random.choice(i_d)
        j = np.random.choice(n)
        if i == j:
            continue
        # You can only compare if at least one is uncensored and they have different survival times
        # Additionally, if one is censored, it must be censored after the other event
        # If it was censored before, we can't be sure of the ordering
        if y_true[i] == y_true[j]:
            continue
        if d_true[j] == 0 and y_true[j] < y_true[i]:
            continue
        permissible += 1
        if (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]) or (
            y_true[j] < y_true[i] and y_pred[j] < y_pred[i]
        ):
            concordant += 1

    return concordant / permissible


# %%
def visualise_predictions(c, y_true, y_pred):
    fig = go.Figure()

    df = pd.DataFrame({"c": c, "y_true": y_true, "y_pred": y_pred, "d_true": y_true < c, "d_pred": y_pred < c})

    marker = dict(
        opacity=0.7,
        # Predicted events are crosses, predicted censored are circles
        symbol=(df["y_pred"] < df["c"]).map({True: "x", False: "circle"}),
        # Correct predictions are green, incorrect are red
        color=(df["d_true"] == df["d_pred"]).map({True: "green", False: "red"}),
    )

    fig.add_trace(
        go.Scatter(x=df["y_true"], y=df["y_pred"], mode="markers", marker=marker)
    )

    # Add y=x line
    y_max = df[["y_true", "y_pred"]].max().max()
    fig.add_trace(go.Scatter(x=[0, y_max], y=[0, y_max], mode="lines", line=dict(color="black", dash="dash")))

    fig.update_layout(
        title="Predicted vs True Survival Times",
        xaxis_title="True Survival Time",
        yaxis_title="Predicted Survival Time",
        showlegend=False,
    )
    fig.show()


# %%
if __name__ == "__main__":
    # Evaluate some fake predictions
    # What's a prediction?
    # Each subject has a true survival time Y and a censoring time C
    # If Y <= C, the event is observed (D=1)
    # If Y > C, the event is censored (D=0)
    # We can either evaluate the binary classification of D vs D_pred
    # or the survival times Y vs Y_pred (only for uncensored subjects)
    n = 100
    c = np.random.uniform(500, 1500, size=n)
    y_true = np.random.exponential(scale=1000, size=n)
    y_true = np.minimum(y_true, c)
    d_true = y_true < c

    y_pred_perfect = y_true.copy()
    y_pred_random = np.random.exponential(scale=1000, size=n)
    y_pred_biased = y_true + np.random.normal(loc=200, scale=100, size=n)
    y_pred_noisy = y_true + np.random.normal(loc=0, scale=300, size=n)
    y_pred_worst = np.where(y_true < c, c, 0)
    y_pred_constant = np.full(n, np.median(y_true))

    y_preds = {
        "Perfect": y_pred_perfect,
        "Random": y_pred_random,
        "Biased": y_pred_biased,
        "Noisy": y_pred_noisy,
        "Worst": y_pred_worst,
        "Constant": y_pred_constant,
    }
    d_preds = {name: y_pred < c for name, y_pred in y_preds.items()}

    for name in y_preds.keys():
        print(f"\n{name} Predictions:")
        visualise_predictions(c, y_true, y_preds[name])

# %%
# X, C -> 0 < T < inf, Y = min(T, C), D = T < C
#
# Model choices
# 1. Predict T | X
# 2. Predict D | X, C
# 3. Predict Y | X, C (aWeibull)
# 
# 1. P(D) binary
# 2. P(D,C) at time u
#   - truth: uc (T<u, D=1), rc (C<u, D=0), lc (u<T<C, D=0)
#   - pred: uc (F(u)), 
