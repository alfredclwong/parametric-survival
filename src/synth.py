# %%
from pathlib import Path
from dataclasses import dataclass
from typing import Callable

import numpy as np
import polars as pl
from scipy.special import expit as sigmoid

from config import EPS
from dist import Distribution, ScaledExponential, Weibull, ScaledWeibull
from mapping import LinearParamMapping, ParamMapping
from process import process_df, plot_feature_histograms

# %%
ROOT_DIR = Path().cwd().parent
DATA_DIR = ROOT_DIR / "data"
N_FEATURES = 10
N_RELEVANT_FEATURES = 5
N_SAMPLES = 1_000_000

# %%
@dataclass
class SynthConfig:
    n_features: int
    n_samples: int
    noise: bool
    dist_type: type[Distribution]
    param_mapping_type: type[ParamMapping]
    param_transforms: dict[str, Callable]
    feature_importances: dict[str, np.ndarray]
    biases: dict[str, float]

# %%
param_transforms = {
    "scale": lambda x: 100 + 4900 * sigmoid(x),
    "shape": lambda x: 0.9 + 4.1 * sigmoid(x),
    "A": lambda x: np.clip(sigmoid(x), EPS, 1.0),
    "k": lambda x: np.clip(sigmoid(x) / 365, EPS, 1.0),
}
biases = {
    "scale": -1.0,
    "shape": 0.5,
    "A": 1.0,
    "k": -0.1,
}
param_transforms = {
    k: v for k, v in param_transforms.items() if k in dist_type.param_names
}
biases = {k: v for k, v in biases.items() if k in dist_type.param_names}

# %%
def generate_synthetic_data(cfg: SynthConfig):
    data = np.random.randn(cfg.n_samples, cfg.n_features)
    X_cols = [f"X_{i + 1}" for i in range(cfg.n_features)]
    df = pl.DataFrame(data, schema=X_cols)
    mapping = LinearParamMapping(
        n_features=cfg.n_features,
        param_transforms=param_transforms,
        bias=any(cfg.biases.values()),
    )
    weights_dict = mapping.get_weights_dict()
    weights = weights.reshape(cfg.n_features + int(mapping.bias), -1)
    weights[cfg.n_relevant_features + int(mapping.bias) :] = 0
    if mapping.bias:
        weights[0] = np.array([biases.get(k, 0) for k in dist_type.param_names])
    mapping.set_weights(weights)
    params = mapping.map(df[X_cols].to_numpy())
    if noise:
        for k, f in param_transforms.items():
            params[k] *= 1 + 0.1 * np.random.randn(*params[k].shape)
    t = dist_type.sample(1, params).flatten()
    c = Weibull.sample(n_samples, {"shape": np.array([1.5]), "scale": np.array([1000])})
    t = np.clip(t, 0, 3000).flatten()
    c = np.clip(c, 0, 3000).flatten()
    t = np.where(t < c, t, np.nan)
    df = df.with_columns([pl.Series("T", t), pl.Series("C", c)])
    return df, mapping


# %%
dist_type = ScaledWeibull
df, mapping = generate_synthetic_data(
    N_FEATURES, N_SAMPLES, dist_type, N_RELEVANT_FEATURES
)
df, _ = process_df(df)
weights = mapping.get_weights().reshape(N_FEATURES + int(mapping.bias), -1)
rows = ["bias"] if mapping.bias else []
rows += [f"X_{i + 1}" for i in range(N_FEATURES)]
weights_df = pl.DataFrame(weights, schema=dist_type.param_names).with_columns(
    pl.Series("feature", rows)
)
weights_df

# %%
import altair as alt

alt.data_transformers.enable("vegafusion")
plot_feature_histograms(df)

# %%
df.write_parquet(DATA_DIR / "synth.parquet")
weights_df.write_parquet(DATA_DIR / "synth_weights.parquet")
