# %% dist.py
import numpy as np

from dist import Exponential, Weibull

exp_params = {"k": np.array([0.1, 1.0, 10.0])}
# print(Exponential.sample(n_samples=10, params=exp_params))
Exponential.plot(
    x=np.linspace(0, 5, 100),
    params=exp_params,
).show()

weibull_params = {
    "scale": np.array([1.0, 1.0, 1.0]),
    "shape": np.array([0.5, 1.0, 2.0]),
}
# print(Weibull.sample(n_samples=10, params=weibull_params))
Weibull.plot(
    x=np.linspace(0, 5, 100),
    params=weibull_params,
).show()

# %% mapping.py
from mapping import LinearParamMapping

n_features = 3
param_transforms = {
    "scale": lambda x: 1.0 + 9.0 * x,
    "shape": lambda x: 0.5 + 4.5 * x,
}
mapping = LinearParamMapping(
    n_features=n_features, param_transforms=param_transforms, bias=True
)
X = np.random.rand(2, n_features)
params = mapping.map(X)
print(mapping.get_weights_df())
print("X:", X)
print("Mapped parameters:", params)

# %%
# synth.py
import polars as pl
from scipy.special import expit as sigmoid

from config import EPS
from dist import ScaledWeibull
from process import process_df
from synth import SynthConfig, generate_synthetic_data
from vis import plot_feature_histograms

cfg = SynthConfig(
    n_samples=1_000_000,
    n_features=10,
    noise=0.1,
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
        # "k": np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1]) / 3,
    },
    biases={
        "A": 0,  # Increasing A increases P(D)
        "scale": -0.5,  # Decreasing scale increases P(D)
        "shape": -0.3,  # Decreasing shape increases P(D)
        # "k": 0,
    },
)

df, weights_df, params_df = generate_synthetic_data(cfg)
df, _ = process_df(df)

print(df)
with pl.Config(tbl_rows=cfg.n_features + 1):
    print(weights_df)
print(params_df)
pct_D = df.select(pl.col("D").mean()).item()
print(f"{pct_D:.2%}")

# %%
plot_features = ["T", "C", "D", "Y"]
plot_feature_histograms(df, features=plot_features, n_cols=4).show()
plot_feature_histograms(params_df, n_cols=3).show()

# %% model.py
from model import ParametricSurvivalModel
from vis import plot_likelihoods

mapping = LinearParamMapping(
    n_features=cfg.n_features,
    param_transforms=cfg.param_transforms,
    bias=True,
)
mapping.set_weights(weights_df.to_numpy().flatten())
model = ParametricSurvivalModel(
    dist_type=ScaledWeibull,
    param_mapping=mapping,
)
x_cols = df.columns[4:]
x = df.select(x_cols).to_numpy()
y = df["Y"].to_numpy()
c = df["C"].to_numpy()
params = model.param_mapping.map(x)
print("Parameters:", {k: v.shape for k, v in params.items()})
print(f"{x.shape=}, {y.shape=}, {c.shape=}")
like_df = pl.DataFrame(
    {
        "Y": y,
        "L": model.likelihoods(x, y, c, log=True).flatten(),
        "D": df["D"].to_numpy(),
    }
)
plot_likelihoods(like_df).show()

# %%
