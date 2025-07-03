# %%
import numpy as np
import polars as pl
import torch as t
from torch.nn.functional import sigmoid

from dist import AsymptoticWeibull
from mapping import ParamMapping, ParamMappingConfig
from model import ParametricSurvivalModel, TrainConfig
from process import process_df
from synth import SynthConfig, generate_synthetic_data
from vis import (
    plot_dist,
    plot_feature_histograms,
    plot_likelihoods,
    plot_params_3d,
    plot_params_by_d,
    plot_loss_history,
)
from train import train_test_split
from eval import show_evals
from config import EPS

# %% dist.py
# For distributions, we use Pytorch's built-in distributions.
# Since the Asymptotic Weibull distribution doesn't exist, we define our own.
# We inherit the Pytorch Weibull distribution and override select methods.
alpha = t.tensor([0.1, 0.4, 0.7, 1.0]).unsqueeze(1)
concentration = t.tensor([0.5, 1.0, 2.0, 5.0]).unsqueeze(1)
scale = t.full_like(concentration, 10.0)
weibull = t.distributions.Weibull(scale=scale, concentration=concentration)
asymptotic_weibull = AsymptoticWeibull(
    alpha=alpha, scale=scale, concentration=concentration
)
x = t.linspace(0, scale.max() * 3, 101)[1:]
plot_dist(weibull, x).properties(height=300, width=300).show()
plot_dist(asymptotic_weibull, x).properties(height=300, width=300).show()

# %% mapping.py
# See README for details on the parameter mapping.
n_samples = 2
n_features = 3
param_transforms = {
    "alpha": lambda x: t.clip(sigmoid(x), EPS, 1.0),
    "scale": lambda x: 3650 * t.clip(sigmoid(x), EPS, 1.0),
    "concentration": lambda x: 5 * t.clip(sigmoid(x), EPS, 1.0),
}
mapping_cfg = ParamMappingConfig(
    d_in=n_features + 1,
    d_hidden=[],
    param_transforms=param_transforms,
)
mapping = ParamMapping(mapping_cfg)
x = t.randn(n_samples, n_features)
x = t.cat([t.ones(x.shape[0], 1), x], dim=1)
with t.no_grad():
    params = mapping(x)
x_df = pl.DataFrame({f"x_{i}": x[:, i] for i in range(n_features + 1)})
params_df = pl.DataFrame(params)
print(x_df)
print(mapping.weights_df)

# %% synth.py
feature_factors = {
    "alpha": np.array([1, 0, 0]),
    "scale": np.array([0, 1, 0]),
    "concentration": np.array([0, 0, 1]),
}
biases = {
    "alpha": 0,  # Increasing alpha increases P(D)
    "scale": -0.5,  # Decreasing scale increases P(D)
    "concentration": -0.3,  # Decreasing shape increases P(D)
}
cfg = SynthConfig(
    n_samples=100_000,
    n_features=n_features,
    noise=0,
    dist_type=AsymptoticWeibull,
    mapping_cfg=mapping_cfg,
    feature_factors=np.array(list(feature_factors.values())).T,
    biases=np.array(list(biases.values())),
)
df, weights_df, params_df = generate_synthetic_data(cfg)
df, _ = process_df(df)
params_df = params_df.with_columns(
    pl.Series("D", df["D"].to_numpy()),
)

print(df)
with pl.Config(tbl_rows=cfg.n_features + 1):
    print(weights_df)
print(params_df)
pct_D = df.select(pl.col("D").mean()).item()
print(f"{pct_D:.2%}")
plot_feature_histograms(df, features=["T", "C", "D", "Y"], n_cols=4).show()
plot_feature_histograms(params_df, n_cols=3).show()
plot_params_3d(
    params_df,
    param_names=list(param_transforms.keys()),
).show()
plot_params_by_d(
    params_df,
    param_names=list(param_transforms.keys()),
).show()

# %% model.py
device = (
    "mps"
    if t.backends.mps.is_available()
    else "cuda"
    if t.cuda.is_available()
    else "cpu"
)
model = ParametricSurvivalModel(
    dist_type=AsymptoticWeibull,
    mapping_cfg=mapping_cfg,
    device=device,
)
model.mapping.weights_df = weights_df
x_cols = df.columns[4:]
x = df.select(x_cols).to_torch().to(device, dtype=t.float32)
y = df["Y"].to_torch().to(device, dtype=t.float32)
c = df["C"].to_torch().to(device, dtype=t.float32)
params = model.mapping(x)
like_df = pl.DataFrame(
    {
        "Y": y.detach().cpu(),
        "logL": model.log_likelihood(x, y, c).flatten().detach().cpu(),
        "D": df["D"].to_numpy(),
    }
)
print("Parameters:", {k: v.shape for k, v in params.items()})
print(f"{x.shape=}, {y.shape=}, {c.shape=}")
plot_likelihoods(like_df).show()

# %%
# train.py
train_df, val_df = train_test_split(df, ratio=0.8)
x_train, y_train, c_train, d_train = (
    train_df[x_cols].to_torch().to(device, dtype=t.float32),
    train_df["Y"].to_torch().to(device, dtype=t.float32),
    train_df["C"].to_torch().to(device, dtype=t.float32),
    train_df["D"].to_torch().to(device, dtype=t.bool),
)
x_val, y_val, c_val, d_val = (
    val_df[x_cols].to_torch().to(device, dtype=t.float32),
    val_df["Y"].to_torch().to(device, dtype=t.float32),
    val_df["C"].to_torch().to(device, dtype=t.float32),
    val_df["D"].to_torch().to(device, dtype=t.bool),
)
history = model.fit(
    x_train,
    y_train,
    d_train,
    x_val,
    y_val,
    d_val,
    cfg=TrainConfig(
        n_epochs=100,
        learning_rate=1e-3,
        weight_decay=0.0,
        balance=True,
    ),
)
plot_loss_history(history).show()
show_evals(
    model,
    x_train,
    y_train,
    c_train,
    d_train,
    x_val,
    y_val,
    c_val,
    d_val,
)

# %%
