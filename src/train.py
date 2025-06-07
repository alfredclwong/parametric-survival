# %%
from itertools import combinations
from pathlib import Path
from dataclasses import dataclass
from typing import Callable

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import polars as pl
import seaborn as sns
from scipy.special import expit as sigmoid
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm.auto import tqdm

from config import EPS
from dist import ScaledExponential, ScaledWeibull, Weibull, Distribution
from mapping import ParamMapping
from mapping import LinearParamMapping
from model import ParametricSurvivalModel

# %%
ROOT_DIR = Path().cwd().parent
DATA_DIR = ROOT_DIR / "data"
PROCESSED_PARQUET_PATH = DATA_DIR / "dummy_processed.parquet"
# PROCESSED_PARQUET_PATH = DATA_DIR / "synth.parquet"
alt.data_transformers.enable("vegafusion")

# %%
def train_val_test_split(
    df: pl.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.1
):
    df = df.sample(fraction=1, shuffle=True)
    train_size = int(len(df) * train_ratio)
    val_size = int(len(df) * val_ratio)
    train_df = df.head(train_size)
    val_df = df.slice(train_size, val_size)
    test_df = df.tail(-train_size - val_size)
    return train_df, val_df, test_df


# Load the processed data and split into train and test sets
df = pl.read_parquet(PROCESSED_PARQUET_PATH).head(100_000)
X_cols = [col for col in df.columns if col.startswith("X_")]
df = df.sample(fraction=1, shuffle=True)
train_df, val_df, test_df = train_val_test_split(df)

# %%
@dataclass
class RunConfig:
    mapping: type[ParamMapping]
    dist_type: type[Distribution]
    bias: bool
    balance: bool
    param_transforms: dict[str, Callable]


def run(
    cfg: RunConfig,
):
    """
    End-to-end steps:
        - Generate X features iid standard
        - Generate param mapping weights
        - Generate params
        - Generate T from params
        - Generate C (independent)
        - Pre-process the X, T, C data to get Y, D
        - Fit the model to the data
        - Store test_df, pred_df, model weights (true and pred), metrics

    We pre-generate synthetic data with various hyperparameters and store these.
    The runs fit various models to the data and store the results.
    The results are then used to compare the models and hyperparameters.
    """
    pass


# %%

# %%
dist_type = ScaledWeibull
bias = True
balance = True
param_transforms = {
    # "scale": lambda x: 1 + 4999 * sigmoid(x),
    # "shape": lambda x: 0.5 + 4.5 * sigmoid(x),
    "scale": lambda x: 100 + 4900 * sigmoid(x),
    "shape": lambda x: 0.9 + 4.1 * sigmoid(x),
    "A": lambda x: np.clip(sigmoid(x), EPS, 1.0),
    "k": lambda x: np.clip(sigmoid(x) / 365, EPS, 1.0),
}


param_transforms = {
    k: v for k, v in param_transforms.items() if k in dist_type.param_names
}
linear_param_mapping = LinearParamMapping(
    param_transforms=param_transforms,
    n_features=len(X_cols),
    bias=bias,
)
model = ParametricSurvivalModel(
    dist_type=dist_type,
    param_mapping=linear_param_mapping,
)
x, t, y, c, d = (
    train_df[X_cols].to_numpy(),
    train_df["T"].to_numpy(),
    train_df["Y"].to_numpy(),
    train_df["C"].to_numpy(),
    train_df["D"].to_numpy(),
)
x_val, t_val, y_val, c_val, d_val = (
    val_df[X_cols].to_numpy(),
    val_df["T"].to_numpy(),
    val_df["Y"].to_numpy(),
    val_df["C"].to_numpy(),
    val_df["D"].to_numpy(),
)
x_test, t_test, y_test, c_test, d_test = (
    test_df[X_cols].to_numpy(),
    test_df["T"].to_numpy(),
    test_df["Y"].to_numpy(),
    test_df["C"].to_numpy(),
    test_df["D"].to_numpy(),
)
# Fit the model to the training data
history = model.fit(
    x,
    y,
    c,
    x_test=x_val,
    y_test=y_val,
    c_test=c_val,
    maxiter=500,
    balance=balance,
    patience=100,
)

# %%
pred_df = pl.DataFrame(
    {
        "C": c,
        "Y": y,
        "D": d,
        **model.param_mapping.map(x),
        "L": model.likelihoods(x, y, c),
        "D_pred": model.dist_type.cdf(c, model.param_mapping.map(x)).flatten(),
        "T_pred": model.median_survival_time(x),
    }
).with_columns(
    Y_pred=pl.min_horizontal(["T_pred", "C"]),
)

test_pred_df = pl.DataFrame(
    {
        "C": c_test,
        "T": t_test,
        "Y": y_test,
        "D": d_test,
        **model.param_mapping.map(x_test),
        "L": model.likelihoods(x_test, y_test, c_test),
        "T_pred": model.median_survival_time(x_test),
        "D_pred": model.dist_type.cdf(
            c_test, model.param_mapping.map(x_test)
        ).flatten(),
    }
).with_columns(
    Y_pred=pl.min_horizontal(["T_pred", "C"]),
)

# %%
# Plot the training and test loss history
history_df = pl.DataFrame(history, schema=["Step", "Metric", "Value"], orient="row")
history_df = history_df.with_columns(Value=pl.col("Value").clip(0.0, 10.0))
loss_chart = (
    alt.Chart(history_df)
    .mark_line()
    .encode(
        x=alt.X("Step", title="Step"),
        y=alt.Y("Value", title="Loss"),
        color=alt.Color("Metric:N", title="Metric"),
        tooltip=[
            alt.Tooltip("Step", title="Step"),
            alt.Tooltip("Metric", title="Metric"),
            alt.Tooltip("Value", title="Loss"),
        ],
    )
    .properties(title="Training and Test Loss History")
)
loss_chart.show()

# %%
likelihood_chart = (
    alt.Chart(pred_df)
    .mark_circle(size=10)
    .encode(
        x=alt.X("Y", title="Y"),
        y=alt.Y("L", title="Likelihood"),
        color=alt.Color("D:N", title="D"),
    )
    .properties(width=600, height=400)
    .facet(row=alt.Row("D:N", title="D"))
    .resolve_scale(y="independent")
    .properties(title="Likelihood of Y given C")
)

# vlines = (
#     alt.Chart(pl.DataFrame({"Y": np.arange(365, 2500, 365)}))
#     .mark_rule(color="red", strokeDash=[4,2])
#     .encode(x="Y:Q")
# )
# likelihood_chart += vlines

likelihood_chart.show()

# %%
# Plot the distribution of the mapped parameters, colored by the outcome D
param_names = list(param_transforms.keys())
param_charts = alt.vconcat(
    *[
        alt.Chart(pred_df)
        .mark_circle(size=10)
        .encode(
            x=alt.X(p0, title=p0),
            y=alt.Y(p1, title=p1),
            color=alt.Color("D:N", title="D"),
            tooltip=[
                alt.Tooltip(param_names[0], title=param_names[0]),
                alt.Tooltip(param_names[1], title=param_names[1]),
                alt.Tooltip("D", title="D"),
            ],
        )
        .properties(title="Parameter Distribution by D")
        .facet(column=alt.Column("D:N", title="D"))
        for p0, p1 in combinations(param_names, 2)
    ]
)
param_charts.show()

# %%
# Repeated histogram: columns by D, rows by params
n_rows = len(param_names)
n_cols = len(pred_df["D"].unique())
fig = sp.make_subplots(
    rows=n_rows, cols=n_cols, subplot_titles=[f"D={d}" for d in pred_df["D"].unique()]
)
for i, param in enumerate(param_names):
    for j, d in enumerate(pred_df["D"].unique()):
        param_data = pred_df.filter(pl.col("D") == d)[param].to_numpy()
        fig.add_trace(
            go.Histogram(
                x=param_data,
                name=f"{param} (D={d})",
                nbinsx=20,
                opacity=0.75,
            ),
            row=i + 1,
            col=j + 1,
        )
        fig.update_xaxes(title_text=param, row=i + 1, col=j + 1)
        fig.update_yaxes(title_text="Count", row=i + 1, col=j + 1)
fig.update_layout(
    title="Parameter Distribution by D",
    height=200 * n_rows,
    width=300 * n_cols,
    showlegend=False,
)
fig.show()

# %%
# Plot the 3D parameter distribution
param_df = pred_df.select(param_names + ["D"]).to_pandas()
fig = px.scatter_3d(
    param_df,
    x=param_names[0],
    y=param_names[1],
    z=param_names[2] if len(param_names) > 2 else None,
    color="D",
    title="3D Parameter Distribution by D",
)
fig.update_traces(marker=dict(size=1, opacity=0.5))
fig.update_layout(
    scene=dict(
        xaxis_title=param_names[0],
        yaxis_title=param_names[1],
        zaxis_title=param_names[2] if len(param_names) > 2 else None,
    )
)
fig.show()

# %%
n_samples = 5
Y = np.linspace(0, 2500, 101)
# Pick n_samples from each subset of test_df with D = True/False
samples_true = test_df.filter(pl.col("D")).sample(n_samples)
samples_false = test_df.filter(~pl.col("D")).sample(n_samples)
sample_dfs = [samples_true, samples_false]
sample_labels = [True, False]

# Add the set of C values to Y and sort
Y = np.sort(
    np.unique(
        np.concatenate([Y, samples_true["C"].to_numpy(), samples_false["C"].to_numpy()])
    )
)

# Create a DataFrame to hold the CDF, PDF, and likelihood for each sample
plot_data = []
for sample_df, label in zip(sample_dfs, sample_labels):
    for i in range(n_samples):
        sample = sample_df[i]
        params = model.param_mapping.map(sample[X_cols].to_numpy())
        cdf = model.dist_type.cdf(Y, params).flatten()
        pdf = model.dist_type.pdf(Y, params).flatten()
        x = np.tile(sample[X_cols].to_numpy().reshape(1, -1), (len(Y), 1))
        y = Y
        c = np.ones_like(Y) * sample["C"].item()
        likelihood = model.likelihoods(x, y, c).flatten()
        param_str = ", ".join(
            # f"{k}={v.item():.2f}" for k, v in params.items()
            f"{v.item():.3g}"
            for k, v in params.items()
        )
        data = {
            "T/Y": Y,
            "CDF": cdf,
            "PDF": pdf,
            "Likelihood": likelihood,
            "Sample": i + int(label) * n_samples,
            "D": label,
            "Params": [param_str] * len(Y),
        }
        plot_data.append(pl.DataFrame(data))
plot_df = pl.concat(plot_data)

# Use plotly to plot the CDF, PDF, and likelihood for each sample
# Rows: CDF, PDF, Likelihood
# Cols: D = True/False
fig = px.line(
    plot_df.to_pandas(),
    x="T/Y",
    y=["CDF", "PDF", "Likelihood"],
    color="Params",
    facet_col="D",
    facet_row="variable",
    title="Distribution of Samples by D",
)
fig.update_layout(
    height=700,
    width=800,
    legend_title=", ".join(param_names),
)
fig.update_yaxes(matches=None)  # Make y-axes independent between rows
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.show()


# %%
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


# Calculate metrics for training data
train_metrics = binary_classification_metrics(
    pred_df["D"].to_numpy(), pred_df["D_pred"].to_numpy() > 0.5
)
# Calculate metrics for test data
test_metrics = binary_classification_metrics(
    test_pred_df["D"].to_numpy(), test_pred_df["D_pred"].to_numpy() > 0.5
)
# Calculate C-index for training data
train_c_index = concordance_index(pred_df["Y"].to_numpy(), pred_df["Y_pred"].to_numpy())
# Calculate C-index for test data
test_c_index = concordance_index(
    test_pred_df["Y"].to_numpy(), test_pred_df["Y_pred"].to_numpy()
)
# Print the metrics
print("Training Metrics:")
for metric, value in train_metrics.items():
    print(f"  {metric}: {value:.4f}")
print(f"  C-index: {train_c_index:.4f}")
print("\nTest Metrics:")
for metric, value in test_metrics.items():
    print(f"  {metric}: {value:.4f}")
print(f"  C-index: {test_c_index:.4f}")

# %%
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


# Plot ROC curve for training data
train_roc_fig = plot_roc(
    pred_df["D"].to_numpy(), pred_df["D_pred"].to_numpy(), title="ROC Curve (Train)"
)
# Plot ROC curve for test data
test_roc_fig = plot_roc(
    test_pred_df["D"].to_numpy(),
    test_pred_df["D_pred"].to_numpy(),
    title="ROC Curve (Test)",
)
train_roc_fig.show()
test_roc_fig.show()

# %%
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
train_cm_fig = plot_confusion_matrix(
    pred_df["D"].to_numpy(),
    pred_df["D_pred"].to_numpy() > 0.5,
    title="Confusion Matrix (Train)",
)
test_cm_fig = plot_confusion_matrix(
    test_pred_df["D"].to_numpy(),
    test_pred_df["D_pred"].to_numpy() > 0.5,
    title="Confusion Matrix (Test)",
)
train_cm_fig.show()
test_cm_fig.show()

# %%
# # Save the model weights
# model_weights_path = DATA_DIR / f"{model.dist_type.__name__}_weights.npy"
# np.save(model_weights_path, model.param_mapping.get_weights())

# %%
