# %%
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch as t
from torch.nn.functional import sigmoid

from config import EPS
from dist import AsymptoticWeibull
from mapping import ParamMapping, ParamMappingConfig
from process import process_df
from vis import plot_feature_histograms

# %%
@dataclass
class SynthConfig:
    n_samples: int
    n_features: int
    noise: float  # multiplicative noise on the parameters = 1 + noise * N(0, 1)
    dist_type: type[t.distributions.Distribution]
    mapping_cfg: ParamMappingConfig
    feature_factors: np.ndarray
    biases: np.ndarray


def generate_synthetic_data(cfg: SynthConfig):
    assert len(cfg.mapping_cfg.d_hidden) == 0, "Hidden layers not supported"

    x = t.randn(cfg.n_samples, cfg.n_features)
    x = t.cat([t.ones(cfg.n_samples, 1), x], dim=1)  # add bias term
    x_cols = [f"X_{i}" for i in range(cfg.n_features + 1)]
    df = pl.DataFrame(x.numpy(), schema=x_cols)

    mapping = ParamMapping(cfg.mapping_cfg)
    for i in range(len(cfg.mapping_cfg.param_transforms)):
        mapping.mlp[-1].weight.data[i, 0] = cfg.biases[i]
        mapping.mlp[-1].weight.data[i, 1:] *= cfg.feature_factors[:, i]

    weights_df = mapping.weights_df

    params = mapping(x)
    params_df = pl.DataFrame(
        {name: pl.Series(param.detach().cpu()) for name, param in params.items()}
    )

    c_dist = t.distributions.Weibull(scale=1000, concentration=1.5)
    c = c_dist.sample((cfg.n_samples,)).detach().cpu().flatten()
    c = np.clip(c, 0, 3000)

    t_dist = cfg.dist_type(**params)
    _t = t_dist.sample((1,)).detach().cpu().flatten()
    _t = np.where(_t < c, _t, np.nan)

    df = df.with_columns([pl.Series("T", _t), pl.Series("C", c)])
    return df, weights_df, params_df


# %%
if __name__ == "__main__":
    ROOT_DIR = Path().cwd().parent
    DATA_DIR = ROOT_DIR / "data"

    param_transforms = {
        "alpha": lambda x: t.clip(sigmoid(x), EPS, 1.0),
        # "scale": lambda x: 100 * 365 * t.clip(sigmoid(x), EPS, 1.0),
        "scale": lambda x: 10 * 365 * t.clip(sigmoid(x), EPS, 1.0),
        "concentration": lambda x: 5 * t.clip(sigmoid(x), EPS, 1.0),
    }
    feature_importances = {
        "alpha": np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0]) / 3 * 5,
        "scale": np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 0]) / 3,
        "concentration": np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0]) / 3,
    }
    biases = {
        "alpha": -5,  # Increasing A increases P(D)
        "scale": 0.2,  # Decreasing scale increases P(D)
        # "scale": 15,  # Decreasing scale increases P(D)
        "concentration": -1.0,  # Decreasing shape increases P(D)
        # "concentration": -0.8,  # Decreasing shape increases P(D)
    }
    n_features = 10
    cfg = SynthConfig(
        n_samples=1_000_000,
        n_features=n_features,
        noise=0,
        dist_type=AsymptoticWeibull,
        # dist_type=t.distributions.Weibull,
        feature_factors=np.array(list(feature_importances.values())).T,
        biases=np.array(list(biases.values())),
        mapping_cfg=ParamMappingConfig(
            d_in=n_features + 1,
            d_hidden=[],
            param_transforms=param_transforms,
        ),
    )

    df, weights_df, params_df = generate_synthetic_data(cfg)
    df, _ = process_df(df)

    print(df)
    with pl.Config(tbl_rows=cfg.n_features + 1):
        print(weights_df)
    print(params_df)

# %%
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
    df.write_parquet(DATA_DIR / "synth.parquet")
    weights_df.write_parquet(DATA_DIR / "synth_weights.parquet")
    params_df.write_parquet(DATA_DIR / "synth_params.parquet")

# %%
