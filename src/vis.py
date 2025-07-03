from itertools import combinations

import torch as t
import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import polars as pl
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve

from model import ParametricSurvivalModel

alt.data_transformers.enable("vegafusion")


def plot_samples_by_d(df: pl.DataFrame, model: ParametricSurvivalModel):
    raise NotImplementedError


def plot_params_3d(pred_df, param_names):
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
    return fig


def plot_params_by_d(pred_df, param_names):
    # Repeated histogram: columns by D, rows by params
    n_rows = len(param_names)
    n_cols = len(pred_df["D"].unique())
    fig = sp.make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"D={d}" for d in pred_df["D"].unique()],
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
    return fig


def plot_params(pred_df, param_names):
    # Plot the distribution of the mapped parameters, colored by the outcome D
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
    return param_charts


def plot_likelihoods(pred_df):
    likelihood_chart = (
        alt.Chart(pred_df)
        .mark_circle(size=10)
        .encode(
            x=alt.X("Y", title="Y"),
            y=alt.Y("logL", title="logL"),
            color=alt.Color("D:N", title="D"),
        )
        .properties(width=600, height=400)
        .facet(row=alt.Row("D:N", title="D"))
        .resolve_scale(y="independent")
        .properties(title="log-likelihood of Y given C")
    )

    # vlines = (
    #     alt.Chart(pl.DataFrame({"Y": np.arange(365, 2500, 365)}))
    #     .mark_rule(color="red", strokeDash=[4,2])
    #     .encode(x="Y:Q")
    # )
    # likelihood_chart += vlines

    return likelihood_chart


def plot_loss_history(history, lim=10):
    """Plot the training and validation loss history using Altair.
    Args:
        history (dict): A dictionary containing the training and validation loss history.
            The keys should be "train" and "val", each containing a list of loss values.
    """
    # Flatten the data into a long format with columns: type, fold, value
    data = {k: np.array(v) for k, v in history.items() if k in ["train", "val"]}
    records = [
        {"step": i + 1, "type": k, "value": v[i]}
        for k, v in data.items()
        for i in range(len(v))
    ]
    history_df = pl.DataFrame(records)
    print(history_df)

    history_df = history_df.with_columns(
        pl.when(pl.col("value") > lim)
        .then(None)
        .otherwise(pl.col("value"))
        .alias("value")
    )
    chart = (
        alt.Chart(history_df)
        .mark_line()
        .encode(
            x=alt.X("step:Q", title="Step"),
            y=alt.Y("value:Q", title="Loss", scale=alt.Scale(domain=[0, lim])),
            color=alt.Color("type:N", title="Type"),
        )
        .properties(title="Training and Validation Loss History")
    )
    chart = chart.configure_axis(
        labelFontSize=12,
        titleFontSize=14,
    ).configure_title(fontSize=16)
    return chart


def plot_feature_histograms(df, features=None, maxbins=20, n_cols=5, subplot_size=120):
    # Plot a histogram of each feature using Altair
    charts = []
    if features is None:
        features = df.columns
    dtypes = df.select(pl.col(features)).dtypes
    for feature, dtype in zip(features, dtypes):
        is_categorical = dtype in [pl.Categorical, pl.String, pl.Boolean]
        if is_categorical:
            x = alt.X(feature).type("nominal")
        else:
            x = alt.X(feature).bin(maxbins=maxbins)
        chart = (
            alt.Chart(df.fill_nan(None))
            .mark_bar()
            .encode(x=x, y="count()")
            .properties(title=feature, width=subplot_size, height=subplot_size)
        )
        charts.append(chart)
    n_rows = (len(charts) + n_cols - 1) // n_cols
    charts = [
        alt.hconcat(*charts[i * n_cols : (i + 1) * n_cols]) for i in range(n_rows)
    ]
    return alt.vconcat(*charts).resolve_scale(x="shared", y="shared")


def plot_feature_histograms_pd(
    df, features=None, maxbins=20, n_cols=5, subplot_size=120
):
    # Plot a histogram of each feature using Altair
    charts = []
    if features is None:
        features = df.columns
    for feature in features:
        is_categorical = df[feature].dtype.name in ["category", "object", "bool"]
        if is_categorical:
            x = alt.X(feature).type("nominal")
        else:
            x = alt.X(feature).bin(maxbins=maxbins)
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(x=x, y="count()")
            .properties(title=feature, width=subplot_size, height=subplot_size)
        )
        charts.append(chart)
    n_rows = (len(charts) + n_cols - 1) // n_cols
    charts = [
        alt.hconcat(*charts[i * n_cols : (i + 1) * n_cols]) for i in range(n_rows)
    ]
    return alt.vconcat(*charts).resolve_scale(x="shared", y="shared")


def plot_roc(y_true, y_scores, title="ROC Curve"):
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


def plot_t_auc(aucs_df: pl.DataFrame):
    aucs_chart = (
        alt.Chart(aucs_df)
        .mark_line()
        .encode(x="Time", y="AUC")
        .properties(title="Time-dependent AUCs")
    )
    return aucs_chart


def plot_dist(dist: t.distributions.Distribution, x: t.Tensor):
    """
    Plot the PDF and CDF of a distribution.
    Args:
        dist (t.distributions.Distribution): The distribution to plot.
        x (t.Tensor): The range of x values to plot.
    Returns:
        alt.Chart: Altair chart with PDF and CDF.

    The dist may have multiple parameters, so we plot each parameter's PDF and CDF separately.
    """
    n_dist = dist.batch_shape[0] if dist.batch_shape else 1
    param_names = [k for k, _ in dist.arg_constraints.items() if k in dist.__dict__]
    param_values = [getattr(dist, name) for name in param_names]
    param_strs = [
        ", ".join(
            [
                f"{param_names[i][:3]}={param_values[i][j].item():.2f}"
                for i in range(len(param_names))
            ]
        )
        for j in range(n_dist)
    ]
    n = len(x)
    pdf = dist.log_prob(x).exp()  # Shape (n_dist, len(x))
    cdf = dist.cdf(x)  # Shape (n_dist, len(x))
    df = (
        pl.DataFrame(
            {
                "x": x.repeat(n_dist).flatten(),
                "pdf": pdf.flatten(),
                "cdf": cdf.flatten(),
                "params": [param_strs[j] for j in range(n_dist) for _ in range(n)],
            }
        )
        .with_columns(pl.when(pl.col("pdf") < 0.3).then("pdf").otherwise(None))
        .unpivot(index=["x", "params"], variable_name="type", value_name="y")
    )
    chart = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("x", title="x"),
            y=alt.Y("y:Q", title="y"),
            color=alt.Color("params", title="Parameters"),
            facet=alt.Facet("type", title=None, columns=2),
            tooltip=[
                alt.Tooltip("x", title="x"),
                alt.Tooltip("y", title="y"),
                alt.Tooltip("params", title="Parameters"),
            ],
        )
        .properties(
            title=f"{dist.__class__.__name__} Distribution",
        )
        .resolve_scale(
            y="independent",
        )
    )
    return chart
