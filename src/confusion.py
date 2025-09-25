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
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch as t
from torch.nn.functional import sigmoid
from typing import Callable

from tqdm.auto import tqdm

from train import train_test_split
from model import ParametricSurvivalModel
from vis import (
    plot_feature_histograms,
    plot_likelihoods,
    plot_loss_history,
    plot_params,
    plot_params_by_d,
)
from dist import AsymptoticWeibull
from mapping import ParamMappingConfig, ParamMapping
from model import TrainConfig
from config import EPS


# %%
def run(
    n: int, W_df: pl.DataFrame, param_transforms: dict[str, Callable]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Run the end-to-end simulation with n synthetic samples and a prescribed mapping to parameter space.

    Input features X are generated iid standard normal, and the mapping weights W_df are used in
    conjunction with the param_transforms to generate the parameters P. The survival times T and
    censoring times C are sampled from AsymptoticWeibull distributions. A linear parametric
    survival model is trained on the data, and the predicted weights are returned.

    Parameters
    ----------
    n : int
        Number of synthetic samples to generate.
    W_df : pl.DataFrame
        DataFrame containing the mapping weights.
    param_transforms : dict[str, Callable]
        Dictionary mapping parameter names to transformation functions.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame]
        DataFrame containing the synthetic data with features X, survival times T, censoring times C,
        and event indicators D, and a DataFrame containing the predicted weights.
    """
    n_features = W_df.shape[0]
    X = t.randn(n * 2, n_features)
    X[:, 0] = 1

    mapping = ParamMapping(
        ParamMappingConfig(
            d_in=n_features,
            d_hidden=[],
            param_transforms=param_transforms,
        )
    )
    mapping.weights_df = W_df
    P = mapping(X)

    C_dist = t.distributions.Weibull(scale=1000, concentration=1.5)
    C = C_dist.sample((X.shape[0],)).detach().cpu().flatten()
    C = t.clip(C, 0, 3000)

    T_dist = AsymptoticWeibull(**P)
    T = T_dist.sample((1,)).detach().cpu().flatten()
    T = t.where(T < C, T, t.nan)

    df = pl.DataFrame(
        {
            **{f"X_{i}": X[:, i] for i in range(X.shape[1])},
            "T": T,
            "C": C,
            "D": t.where(T < C, True, False),
            "Y": t.where(T < C, T, C),
        }
    )
    # plot_feature_histograms(df, features=["T", "C", "D", "Y"], n_cols=4).show()

    # params_df = pl.DataFrame(
    #     {name: pl.Series(P.detach().cpu()) for name, P in P.items()}
    # )
    # plot_feature_histograms(params_df, n_cols=3).show()
    device = (
        "mps"
        if t.backends.mps.is_available()
        else "cuda"
        if t.cuda.is_available()
        else "cpu"
    )
    train_cfg = TrainConfig(
        n_epochs=2000,
        learning_rate=5e-3,
        weight_decay=0,
        batch_size=None,
        patience=100,
        balance=True,
    )
    n_events = int(min(n // 300, df["D"].sum()))
    # n_samples = (n_events, n - n_events)
    n_samples = None
    print(
        f"n_samples: {n_samples}, n_d_true: {df['D'].sum()}, n_d_false: {n - df['D'].sum()}"
    )
    pred_W_df, history = predict_weights(df, train_cfg, n_samples, device)

    best_val_loss = min(history["val"])
    if best_val_loss > 5:
        raise ValueError(
            f"Model failed to converge, best val loss: {best_val_loss:.4f}."
        )

    true_weights_df = W_df.rename({col: f"{col[:-2]}_true" for col in W_df.columns})
    pred_weights_df_renamed = pred_W_df.rename(
        {col: f"{col[:-2]}_pred" for col in pred_W_df.columns}
    )
    combined_weights_df = pl.concat(
        [true_weights_df, pred_weights_df_renamed], how="horizontal"
    )
    combined_weights_df = combined_weights_df[sorted(combined_weights_df.columns)]

    return df, combined_weights_df


def predict_weights(
    df: pl.DataFrame, cfg: TrainConfig, n_samples: tuple[int, int], device: str
) -> tuple[pl.DataFrame, dict]:
    df = df.with_columns([pl.col(pl.Float64).cast(pl.Float32)])
    # Add a column of ones for the intercept term called X_0
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
    history = model.fit(x_train, y_train, d_train, x_val, y_val, d_val, cfg=cfg)
    # plot_loss_history(history).show()

    # test_pred_df = model.predict(x_test, y_test, c_test, d_test)

    # plot_likelihoods(test_pred_df).show()

    # param_names = list(model.mapping.param_transforms.keys())
    # plot_params(test_pred_df, param_names).show()
    # plot_params_by_d(test_pred_df, param_names).show()

    return model.mapping.weights_df, history


# %%
root_dir = Path().cwd().parent
data_dir = root_dir / "data"
W_df = pl.read_parquet(data_dir / "synth_AsymptoticWeibull_5f_weights.parquet")
W_df = W_df[:-1]
# W_df[0, 0] = 100.0
# W_df[1, 0] = 0.0
W_df

# %%
n = 100_000
param_transforms = {
    "alpha": lambda x: t.clip(sigmoid(x), EPS, 1.0),
    # "alpha": lambda x: t.ones_like(x),
    "scale": lambda x: 10 * 365 * t.clip(sigmoid(x), EPS, 1.0),
    "concentration": lambda x: 5 * t.clip(sigmoid(x), EPS, 1.0),
}

# %%
perturbed_params = [
    ("alpha_0", 0),
    ("alpha_0", 1),
    ("scale_0", 0),
    ("scale_0", 2),
    # ("concentration_0", 0),
    # ("concentration_0", 3),
]
perturbations = [-1, 0, 0.5, 1.0, 1.5, 2.0]
n_repeats = 5
all_results = []

for i, perturbed_param in enumerate(perturbed_params):
    results = []

    for perturbation in tqdm(perturbations, desc="Perturbations"):
        W_df_perturbed = W_df.clone()
        W_df_perturbed = W_df_perturbed.with_columns(
            pl.when(pl.arange(0, W_df.height) == perturbed_param[1])
            .then(pl.col(perturbed_param[0]) * perturbation)
            .otherwise(pl.col(perturbed_param[0]))
            .alias(perturbed_param[0])
        )
        display(W_df_perturbed)
        for _ in tqdm(range(n_repeats), desc="Repeats"):
            error_count = 0
            while error_count < 5:
                try:
                    df, pred_weights_df = run(n, W_df_perturbed, param_transforms)
                    break
                except Exception as e:
                    print(f"Error: {e}. Retrying...")
                    error_count += 1
            results.append((perturbation, pred_weights_df))
    all_results.append(results)

# %%
for i, perturbed_param in enumerate(perturbed_params):
    results = all_results[i]

    alpha_trues = [res[1][f"{perturbed_param[0][:-1]}true"][perturbed_param[1]] for res in results]
    alpha_trues = np.array(alpha_trues)
    pred_params = [
        ("alpha_pred", [0]),
        ("alpha_pred", [1]),
        ("alpha_pred", [2, 3, 4]),
        ("scale_pred", [0]),
        ("scale_pred", [2]),
        ("scale_pred", [1, 3, 4]),
        # ("concentration_pred", [0]),
        # ("concentration_pred", [3]),
        # ("concentration_pred", [1, 2, 4]),
    ]

    n_cols = 3
    n_rows = (len(pred_params) - 1) // n_cols + 1
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axs = axs.flatten() if n_rows > 1 else [axs]
    for j, pred_param in enumerate(pred_params):
        preds = [res[1][pred_param[0]][pred_param[1]] for res in results]
        preds = np.array(preds)
        # print(alpha_trues.shape, preds.shape)
        alpha_trues_tiled = np.tile(alpha_trues, (1, preds.shape[1]))
        ax = axs[j]
        # Compute correlation coefficient (r-value) and line of best fit
        x = np.array(alpha_trues_tiled).flatten()
        y = np.array(preds).flatten()
        if len(x) == len(y) and len(x) > 1:
            r = np.corrcoef(x, y)[0, 1]
            # Fit line of best fit
            m, b = np.polyfit(x, y, 1)
            x_fit = np.linspace(x.min(), x.max(), 100)
            y_fit = m * x_fit + b
            ax.plot(x_fit, y_fit, 'k--', label=f'Best fit (r={r:.2f})')
            ax.legend()

        ax.scatter(alpha_trues_tiled, preds)
        ax.set_xlabel("True Weights")
        ax.set_ylabel("Predicted Weights")
        _type = "control" if len(pred_param[1]) > 1 else "bias" if pred_param[1] == [0] else "weight"
        ax.set_title(f"({perturbed_param[0]}, {perturbed_param[1]}) vs ({pred_param[0]}, {_type})")
    plt.tight_layout()
    plt.show()

# %%
dfs = []
for i in range(len(all_results)):
    for j in range(len(all_results[i])):
        df = all_results[i][j][1]
        df = df.with_columns(
            pl.lit(perturbed_params[i][0]).alias("perturbed_param"),
            pl.lit(int(perturbed_params[i][1])).cast(pl.Float64).alias("perturbed_index"),
            pl.lit(all_results[i][j][0]).cast(pl.Float64).alias("perturbation"),
        )
        # display(df)
        dfs.append(df)
        # break
    # break
combined_df = pl.concat(dfs, how="vertical")
combined_df.write_parquet(data_dir / "perturbation_results.parquet")

# %%
combined_df