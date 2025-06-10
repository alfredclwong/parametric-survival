import polars as pl
import altair as alt
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.express as px
import numpy as np
from itertools import combinations

alt.data_transformers.enable("vegafusion")


def plot_samples_by_d(df, model):
    X_cols = [col for col in df.columns if col.startswith("X_")]
    param_names = model.param_mapping.param_names

    n_samples = 5
    Y = np.linspace(0, 2500, 101)
    # Pick n_samples from each subset of test_df with D = True/False
    samples_true = df.filter(pl.col("D")).sample(n_samples)
    samples_false = df.filter(~pl.col("D")).sample(n_samples)
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

    return fig


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

    return likelihood_chart


def plot_loss_history(history_df):
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
    return loss_chart


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
