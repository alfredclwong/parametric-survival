# %%
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
from scipy.special import expit as sigmoid

from config import EPS
from dist import Distribution, ScaledExponential, ScaledWeibull, Weibull
from mapping import LinearParamMapping, ParamMapping
from process import process_df
from vis import plot_feature_histograms_pd, plot_feature_histograms

# %%
ROOT_DIR = Path().cwd().parent
DATA_DIR = ROOT_DIR / "data"


# %%
@dataclass
class SynthConfig:
    n_samples: int
    n_features: int
    noise: bool
    dist_type: type[Distribution]
    param_mapping_type: type[ParamMapping]
    param_transforms: dict[str, Callable]
    feature_importances: dict[str, np.ndarray]
    biases: dict[str, float]


def generate_synthetic_data(cfg: SynthConfig):
    data = np.random.randn(cfg.n_samples, cfg.n_features)
    X_cols = [f"X_{i + 1}" for i in range(cfg.n_features)]
    df = pl.DataFrame(data, schema=X_cols)
    mapping = LinearParamMapping(
        n_features=cfg.n_features,
        param_transforms=cfg.param_transforms,
        bias=True,
    )
    weights = mapping.get_weights(flatten=False)
    weights[0, :] = [cfg.biases[name] for name in mapping.param_names]
    weights[1:, :] *= np.array(
        [cfg.feature_importances[name] for name in mapping.param_names]
    ).T
    mapping.set_weights(weights)
    params = mapping.map(df[X_cols].to_numpy(), noise=cfg.noise)

    weights_df = mapping.get_weights_df()
    params_df = pl.DataFrame({name: pl.Series(param) for name, param in params.items()})

    t = cfg.dist_type.sample(1, params).flatten()
    c = Weibull.sample(
        cfg.n_samples, {"shape": np.array([1.5]), "scale": np.array([1000])}
    )
    t = np.clip(t, 0, 3000).flatten()
    c = np.clip(c, 0, 3000).flatten()
    t = np.where(t < c, t, np.nan)
    df = df.with_columns([pl.Series("T", t), pl.Series("C", c)])
    return df, weights_df, params_df


# %%
cfg = SynthConfig(
    n_samples=10_000_000,
    n_features=10,
    noise=True,
    dist_type=ScaledWeibull,
    param_mapping_type=LinearParamMapping,
    param_transforms={
        "A": lambda x: np.clip(sigmoid(x), EPS, 1.0),
        "scale": lambda x: 5000 * np.clip(sigmoid(x), EPS, 1.0),
        "shape": lambda x: 5 * np.clip(sigmoid(x), EPS, 1.0),
    },
    feature_importances={
        "A": np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0]) / 3 * 5,
        "scale": np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 0]) / 3,
        "shape": np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0]) / 3,
    },
    biases={
        "A": -8,  # Increasing A increases P(D)
        "scale": -0.5,  # Decreasing scale increases P(D)
        "shape": 0.2,  # Decreasing shape increases P(D)
    },
)

df, weights_df, params_df = generate_synthetic_data(cfg)
df, _ = process_df(df)

# print(df)
# with pl.Config(tbl_rows=cfg.n_features + 1):
#     print(weights_df)
# print(params_df)

plot_features = ["T", "C", "D", "Y"]
plot_feature_histograms(df, features=plot_features, n_cols=4).show()
plot_feature_histograms(params_df, n_cols=3).show()
pct_D = df.select(pl.col("D").mean()).item()
print(f"{pct_D:.2%}")

# %%
dummy_df = pl.read_parquet(DATA_DIR / "dummy_processed.parquet")
plot_feature_histograms(dummy_df, features=plot_features, n_cols=4).show()
pct_D = dummy_df.select(pl.col("D").mean()).item()
print(f"{pct_D:.2%}")

# %%
# df.write_parquet(DATA_DIR / "synth.parquet")
# weights_df.write_parquet(DATA_DIR / "synth_weights.parquet")
# params_df.write_parquet(DATA_DIR / "synth_params.parquet")

# %%
