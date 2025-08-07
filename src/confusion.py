# %%
"""
Something that's good practice to do in the simulations is a confusion plot -
run a large number of these simulations, and plot change in estimated
concentration_0 as a function of ground-truth alpha, for example. For each
parameter pair, you get a correlation, that tells you how much that parameter
gets confused with another parameter. These correlations depend on the
'baseline' of all the parameter values, but can help see what's going on.
"""
# 1. Load synthetic data: X, T, C, Y, D
# 2. Load ground truth weights: W

# %%
import polars as pl
import torch as t
from torch.nn.functional import sigmoid
from typing import Callable
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path

from train import train_test_split
from model import ParametricSurvivalModel
from vis import plot_loss_history
from dist import AsymptoticWeibull
from mapping import ParamMappingConfig, ParamMapping
from model import TrainConfig
from config import EPS


# %%
def generate(
    n: int, n_features: int, param_transforms: dict[str, Callable]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    X = t.randn(n, n_features)
    X = t.cat([t.ones(n, 1), X], dim=1)
    X_cols = [f"X_{i}" for i in range(n_features + 1)]
    df = pl.DataFrame({col: X[:, i] for i, col in enumerate(X_cols)})

    mapping = ParamMapping(
        ParamMappingConfig(
            d_in=n_features + 1,
            d_hidden=[],
            param_transforms=param_transforms,
        )
    )
    W = mapping.weights_df.to_numpy()
    W = np.vstack([W[0], W[1:] * np.eye(W.shape[0] - 1, W.shape[1])])
    mapping.weights_df = pl.DataFrame(W, schema=mapping.weights_df.schema)
    P = mapping(X)

    c_dist = t.distributions.Weibull(scale=1000, concentration=1.5)
    c = c_dist.sample((n,)).detach().cpu().numpy().flatten()
    c = np.clip(c, 0, 3000)

    t_dist = AsymptoticWeibull(**P)
    _t = t_dist.sample((1,)).detach().cpu().numpy().flatten()
    _t = np.where(_t < c, _t, t.nan)

    df = df.with_columns(
        pl.Series("T", _t, dtype=pl.Float32),
        pl.Series("C", c, dtype=pl.Float32),
    )
    df = df.with_columns(
        pl.when(pl.col("T") < pl.col("C")).then(True).otherwise(False).alias("D")
    )
    df = df.with_columns(
        pl.when(pl.col("D")).then(pl.col("T")).otherwise(pl.col("C")).alias("Y")
    )

    return df, mapping.weights_df


def train(
    df: pl.DataFrame, param_transforms: dict[str, Callable], train_cfg: TrainConfig
) -> tuple[pl.DataFrame, pl.DataFrame, float]:
    if "X_0" not in df.columns:
        df = df.with_columns(pl.lit(1.0).alias("X_0"))
    x_cols = [col for col in df.columns if col.startswith("X_")]

    train_df, test_df = train_test_split(df, ratio=0.8)
    val_df, test_df = train_test_split(test_df, ratio=0.5)
    x_train, t_train, y_train, c_train, d_train = (
        train_df[x].to_torch() for x in [x_cols, "T", "Y", "C", "D"]
    )
    x_val, t_val, y_val, c_val, d_val = (
        val_df[x].to_torch() for x in [x_cols, "T", "Y", "C", "D"]
    )
    x_test, t_test, y_test, c_test, d_test = (
        test_df[x].to_torch() for x in [x_cols, "T", "Y", "C", "D"]
    )

    # Fit the model to the training data
    d_in = len(x_cols)
    mapping_cfg = ParamMappingConfig(
        d_in=d_in,
        d_hidden=[],
        param_transforms=param_transforms,
    )
    model = ParametricSurvivalModel(
        dist_type=AsymptoticWeibull,
        device="cpu",
        mapping_cfg=mapping_cfg,
    )

    history = model.fit(x_train, y_train, d_train, x_val, y_val, d_val, cfg=train_cfg)
    test_loss = (
        model.loss(x_test, y_test, d_test, balance=train_cfg.balance).mean().item()
    )
    weights_df = model.mapping.weights_df
    return history, weights_df, test_loss


def run_experiment(
    n: int,
    n_features: int,
    param_transforms: dict[str, Callable],
    train_cfg: TrainConfig,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    df, weights_df = generate(n, n_features, param_transforms)
    # plot_loss_history(history).show()
    history, pred_weights_df, test_loss = train(df, param_transforms, train_cfg)
    combined_weights_df = pl.concat(
        [
            weights_df.rename(lambda x: f"{x}_true"),
            pred_weights_df.rename(lambda x: f"{x}_pred"),
        ],
        how="horizontal",
    )
    combined_weights_df = combined_weights_df[sorted(combined_weights_df.columns)]
    return combined_weights_df, test_loss


# %%
n_experiments = 100
n = 10_000
n_features = 5
param_transforms = {
    "alpha": lambda x: t.clip(sigmoid(x), EPS, 1.0),
    "scale": lambda x: 10 * 365 * t.clip(sigmoid(x), EPS, 1.0),
    "concentration": lambda x: 5 * t.clip(sigmoid(x), EPS, 1.0),
}
train_cfg = TrainConfig(
    n_epochs=2000,
    learning_rate=2e-3,
    weight_decay=1e-6,
    balance=True,
    patience=100,
    batch_size=None,
    silent=True,
)

# %%
root_dir = Path(__file__).parent.parent
data_dir = root_dir / "data"
results_dir = data_dir / "results"
assert results_dir.exists(), f"Results directory {results_dir} does not exist"

# %%
n_batch = 10
for batch_idx in tqdm(range(0, n_experiments, n_batch), desc="Running experiments"):
    batch_results = []
    for i in tqdm(
        range(batch_idx, min(batch_idx + n_batch, n_experiments)),
        desc="Batch progress",
        leave=False,
    ):
        while True:
            try:
                combined_weights_df, test_loss = run_experiment(
                    n, n_features, param_transforms, train_cfg
                )
                if test_loss > 5:
                    raise AssertionError("Model failed to converge")
                break
            except Exception as e:
                print(f"Experiment {i} failed with error: {e}")
        combined_weights_df = combined_weights_df.with_columns(
            pl.lit(i).alias("run_id"),
            pl.lit(test_loss).alias("test_loss"),
        )
        batch_results.append(combined_weights_df)
    batch_df = pl.concat(batch_results)
    batch_df.write_parquet(
        results_dir / f"confusion_results_{batch_idx // n_batch}.parquet"
    )

# %%
