# %%
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import altair as alt
import numpy as np
import polars as pl
from scipy.special import expit as sigmoid

from config import EPS
from dist import Distribution, ScaledExponential, ScaledWeibull, Weibull
from mapping import LinearParamMapping, ParamMapping
from model import ParametricSurvivalModel
from vis import (
    plot_likelihoods,
    plot_loss_history,
    plot_params,
    plot_params_3d,
    plot_params_by_d,
    plot_samples_by_d,
)
from eval import (
    binary_classification_metrics,
    concordance_index,
    plot_confusion_matrix,
    plot_roc,
)

# %%
ROOT_DIR = Path().cwd().parent
DATA_DIR = ROOT_DIR / "data"
alt.data_transformers.enable("vegafusion")


# %%
def train_val_test_split(
    df: pl.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    split_d: bool = True,
):
    if split_d:
        # Ensure equal distribution of D in train, val, test sets
        d_true_df = df.filter(pl.col("D"))
        d_false_df = df.filter(~pl.col("D"))
        d_true_train_df, d_true_val_df, d_true_test_df = train_val_test_split(
            d_true_df, train_ratio=train_ratio, val_ratio=val_ratio, split_d=False
        )
        d_false_train_df, d_false_val_df, d_false_test_df = train_val_test_split(
            d_false_df, train_ratio=train_ratio, val_ratio=val_ratio, split_d=False
        )
        train_df = pl.concat([d_true_train_df, d_false_train_df], how="vertical")
        val_df = pl.concat([d_true_val_df, d_false_val_df], how="vertical")
        test_df = pl.concat([d_true_test_df, d_false_test_df], how="vertical")
        train_df = train_df.sample(fraction=1, shuffle=True)
        val_df = val_df.sample(fraction=1, shuffle=True)
        test_df = test_df.sample(fraction=1, shuffle=True)
        return train_df, val_df, test_df
    else:
        df = df.sample(fraction=1, shuffle=True)
        train_size = int(len(df) * train_ratio)
        val_size = int(len(df) * val_ratio)
        train_df = df.head(train_size)
        val_df = df.slice(train_size, val_size)
        test_df = df.tail(-train_size - val_size)
        return train_df, val_df, test_df


@dataclass
class RunConfig:
    synth_data_path: Path
    n_samples: int | tuple[int, int]
    mapping: type[ParamMapping]
    dist_type: type[Distribution]
    bias: bool
    balance: bool
    param_transforms: dict[str, Callable]


def run(cfg: RunConfig):
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
    # Load the synthetic data
    df = pl.read_parquet(cfg.synth_data_path)
    if cfg.n_samples is None:
        pass
    elif isinstance(cfg.n_samples, int):
        df = df.sample(n=cfg.n_samples, shuffle=True)
    elif isinstance(cfg.n_samples, tuple) and len(cfg.n_samples) == 2:
        df_d_true = df.filter(pl.col("D")).sample(n=cfg.n_samples[0], shuffle=True)
        df_d_false = df.filter(~pl.col("D")).sample(n=cfg.n_samples[1], shuffle=True)
        df = pl.concat([df_d_true, df_d_false], how="vertical").sample(
            fraction=1, shuffle=True
        )
    else:
        raise ValueError("n_samples must be an int or a tuple of two ints.")
    X_cols = [col for col in df.columns if col.startswith("X_")]
    print(df.select([col for col in df.columns if col not in X_cols]).describe())
    train_df, val_df, test_df = train_val_test_split(df)

    linear_param_mapping = LinearParamMapping(
        param_transforms=cfg.param_transforms,
        n_features=len(X_cols),
        bias=cfg.bias,
    )
    model = ParametricSurvivalModel(
        dist_type=cfg.dist_type,
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
        balance=cfg.balance,
        patience=100,
    )

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

    # Plot the training and test loss history
    history_df = pl.DataFrame(history, schema=["Step", "Metric", "Value"], orient="row")
    history_df = history_df.with_columns(Value=pl.col("Value").clip(0.0, 10.0))
    plot_loss_history(history_df).show()
    plot_likelihoods(test_pred_df).show()
    param_names = list(cfg.param_transforms.keys())
    plot_params(test_pred_df, param_names).show()
    plot_params_by_d(test_pred_df, param_names).show()
    plot_params_3d(test_pred_df, param_names).show()
    plot_samples_by_d(test_df, model).show()

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
    # Print the metrics
    print("Training Metrics:")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.4f}")
    print(f"  C-index: {train_c_index:.4f}")
    print("\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    print(f"  C-index: {test_c_index:.4f}")

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

    return model, history


# %%
cfg = RunConfig(
    # synth_data_path=DATA_DIR / "dummy_processed.parquet",
    # n_samples=None,
    synth_data_path=DATA_DIR / "synth.parquet",
    # n_samples=(10_000, 10_000),
    n_samples=100_000,
    mapping=LinearParamMapping,
    dist_type=ScaledWeibull,
    bias=True,
    balance=True,
    param_transforms={
        "A": lambda x: np.clip(sigmoid(x), EPS, 1.0),
        "scale": lambda x: 1 + 4900 * np.clip(sigmoid(x), EPS, 1.0),
        "shape": lambda x: 0.1 + 4.9 * np.clip(sigmoid(x), EPS, 1.0),
    },
)

model, history = run(cfg)

# %%
with pl.Config(tbl_rows=11):
    print(model.param_mapping.get_weights_df())

# %%
# # Save the model weights
# model_weights_path = DATA_DIR / f"{model.dist_type.__name__}_weights.npy"
# np.save(model_weights_path, model.param_mapping.get_weights())

# %%
