# %%
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import polars as pl
import torch as t
from torch.nn.functional import sigmoid

from config import EPS
from dist import AsymptoticWeibull
from eval import show_evals
from mapping import ParamMappingConfig
from model import ParametricSurvivalModel, TrainConfig
from vis import plot_loss_history

# %%
def train_test_split(
    df: pl.DataFrame,
    ratio: float = 0.8,
    split_d: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if split_d:
        # Ensure equal distribution of D in train, val, test sets
        d_true_df = df.filter(pl.col("D"))
        d_false_df = df.filter(~pl.col("D"))
        d_true_train_df, d_true_test_df = train_test_split(d_true_df, ratio, False)
        d_false_train_df, d_false_test_df = train_test_split(d_false_df, ratio, False)
        train_df = pl.concat([d_true_train_df, d_false_train_df], how="vertical")
        test_df = pl.concat([d_true_test_df, d_false_test_df], how="vertical")
        train_df = train_df.sample(fraction=1, shuffle=True)
        test_df = test_df.sample(fraction=1, shuffle=True)
        return train_df, test_df
    else:
        df = df.sample(fraction=1, shuffle=True)
        train_size = int(len(df) * ratio)
        train_df = df.head(train_size)
        test_df = df.tail(-train_size)
        return train_df, test_df


@dataclass
class RunConfig:
    synth_data_path: Path
    n_samples: Optional[int | tuple[int, int]]
    mapping_cfg: ParamMappingConfig
    dist_type: type[AsymptoticWeibull]
    param_transforms: dict[str, Callable]
    device: str
    train_cfg: TrainConfig


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
    df = df.with_columns([pl.col(pl.Float64).cast(pl.Float32)])
    # Add a column of ones for the intercept term called X_0
    if "X_0" not in df.columns:
        df = df.with_columns(pl.lit(1.0).alias("X_0"))
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
    x_cols = [col for col in df.columns if col.startswith("X_")]
    print(df.select([col for col in df.columns if col not in x_cols]).describe())

    # Train-test split
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
    model = ParametricSurvivalModel(
        dist_type=cfg.dist_type,
        mapping_cfg=cfg.mapping_cfg,
        device=cfg.device,
    )
    history = model.fit(x_train, y_train, d_train, x_val, y_val, d_val, cfg=cfg.train_cfg)
    plot_loss_history(history).show()

    show_evals(
        model,
        x_train=x_train,
        y_train=y_train,
        c_train=c_train,
        d_train=d_train,
        x_test=x_test,
        y_test=y_test,
        c_test=c_test,
        d_test=d_test,
    )

    return model


# %%
if __name__ == "__main__":
    ROOT_DIR = Path().cwd().parent
    DATA_DIR = ROOT_DIR / "data"

    n_samples = 100_000
    n_events = n_samples // 300
    n_features = 10
    # n_features = 40
    param_transforms = {
        "alpha": lambda x: t.clip(sigmoid(x), EPS, 1.0),
        "scale": lambda x: 10 * 365 * t.clip(sigmoid(x), EPS, 1.0),
        "concentration": lambda x: 5 * t.clip(sigmoid(x), EPS, 1.0),
        # "alpha": lambda x: t.ones_like(x),
        # "alpha": lambda x: 0.01 + 0.99 * t.clip(sigmoid(x), EPS, 1.0),
        # "scale": lambda x: 10 + 3640 * t.clip(sigmoid(x), EPS, 1.0),
        # "concentration": lambda x: 0.1 + 4.9 * t.clip(sigmoid(x), EPS, 1.0),
    }
    device = (
        "mps"
        if t.backends.mps.is_available()
        else "cuda"
        if t.cuda.is_available()
        else "cpu"
    )
    cfg = RunConfig(
        # synth_data_path=DATA_DIR / "dummy_processed.parquet",
        # n_samples=None,
        synth_data_path=DATA_DIR / "synth.parquet",
        n_samples=(n_events, n_samples - n_events),
        dist_type=AsymptoticWeibull,
        mapping_cfg=ParamMappingConfig(
            d_in=n_features + 1,
            d_hidden=[],
            param_transforms=param_transforms,
            # dropout=0.1,
        ),
        param_transforms=param_transforms,
        device=device,
        train_cfg=TrainConfig(
            n_epochs=2000,
            learning_rate=2e-3,
            weight_decay=1e-6,
            balance=True,
            batch_size=None,
            # Two interpretations of balance:
            # 1. Without balance, the loss is mostly determined by the D=0 class, which is much larger.
            # 2. Without balance, the D=1 class has a much larger loss, which can be targetted by the model.
        ),
    )
    model = run(cfg)

    with pl.Config(tbl_rows=11):
        print(model.mapping.weights_df)

# %%
    # # Save the model weights
    # model_weights_path = DATA_DIR / f"{cfg.synth_data_path.stem}_{model.dist_type.__name__}_weights.npy"
    # np.save(model_weights_path, model.param_mapping.get_weights())

# %%
