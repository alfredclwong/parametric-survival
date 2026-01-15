# %%
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

alt.data_transformers.enable("vegafusion")

# %%
ROOT_DIR = Path().cwd()
DATA_DIR = ROOT_DIR / "data"
deconf_path = DATA_DIR / "Tbrain_deconf.csv"
ukb_neuro_path = DATA_DIR / "ukb_neuro_data.csv"

# %%
deconf_df = pd.read_csv(deconf_path, index_col="eid")
ukb_neuro_df = pd.read_csv(ukb_neuro_path, index_col="ID")

# %%
age_imaging = ukb_neuro_df["ageImaging"]
age_initial = ukb_neuro_df["ageInitial"]
time_offset = (age_imaging - age_initial) * 365

# %%
ukb_neuro_df.loc[ukb_neuro_df["dem_timeto_diagn_orEnd"].isna() | (ukb_neuro_df["dem_timeto_diagn_orEnd"] < time_offset),
    ["dem_timeto_diagn_orEnd", "ageImaging", "ageInitial"]].join(
    time_offset.to_frame(name="time_offset")
).sort_values(by="time_offset")

# %%
# Infer censoring times C as max(*_timeto_diagn_orEnd) - time_offset across all diseases
all_diseases = [
    "dem",
    "delirium",
    "epilepsy",
    "migraine",
    "ms",
    "AD",
    "vascD",
    "FTD",
    "PDcheck",
]
c_df = pd.DataFrame(index=ukb_neuro_df.index)
for disease in all_diseases:
    diag_col = (
        "dementia"
        if disease == "dem"
        else "multiplesclerosis"
        if disease == "ms"
        else disease
    )
    c_df[f"C_{disease}"] = np.where(
        ukb_neuro_df[diag_col] == 0,
        ukb_neuro_df[f"{disease}_timeto_diagn_orEnd"],
        np.nan,
    )
c_df["C"] = c_df.max(axis=1) - time_offset
c_df

# %%
# Of the diagnosed cases, how many are pre-diagnosis at imaging time?
dem_ids = ukb_neuro_df.loc[ukb_neuro_df["dementia"] == 1].index
ep_ids = ukb_neuro_df.loc[ukb_neuro_df["epilepsy"] == 1].index
tmp_df = (
    ukb_neuro_df.loc[:, "dem_timeto_diagn_pastfuture"]
    < time_offset.loc[:]  # .fillna(0)
)
# tmp_df = (
#     ukb_neuro_df.loc[dem_ids, "dem_timeto_diagn_pastfuture"] <
#     time_offset.loc[dem_ids]#.fillna(0)
# )
dem_pred_diag_ids = tmp_df.loc[tmp_df].index
tmp_df = (
    ukb_neuro_df.loc[:, "epilepsyPlus_timeto_diagn_pastfuture"]
    < time_offset.loc[:]  # .fillna(0)
)
ep_pred_diag_ids = tmp_df.loc[tmp_df].index
print(
    f"Pre-diag dementia cases: {len(dem_pred_diag_ids)}/{len(dem_ids)}"
    f" ({len(dem_pred_diag_ids) / len(dem_ids):.2%})"
)
print(
    f"Pre-diag epilepsy cases: {len(ep_pred_diag_ids)}/{len(ep_ids)}"
    f" ({len(ep_pred_diag_ids) / len(ep_ids):.2%})"
)

# %%
tmp_df = ukb_neuro_df[["epilepsyPlus_timeto_diagn_pastfuture"]].join(
    time_offset.to_frame(name="time_offset")
)
tmp_df.loc[
    # ep_ids
    tmp_df.index.difference(ep_ids)
].loc[
    tmp_df["epilepsyPlus_timeto_diagn_pastfuture"] < tmp_df["time_offset"]
    # ].loc[
    #     tmp_df["epilepsyPlus_timeto_diagn_pastfuture"] > 0
]

# %%
no_imaging_ids = deconf_df.loc[deconf_df.isna().any(axis=1)].index
no_imaging_dem_ids = no_imaging_ids.intersection(dem_ids)
no_imaging_ep_ids = no_imaging_ids.intersection(ep_ids)
print(
    f"Total cases without any imaging data: {len(no_imaging_ids)}/{deconf_df.shape[0]}"
    f" ({len(no_imaging_ids) / deconf_df.shape[0]:.2%})"
)
print(
    f"Diagnosed dementia cases without any imaging data: {len(no_imaging_dem_ids)}/{len(dem_ids)}"
    f" ({len(no_imaging_dem_ids) / len(dem_ids):.2%})"
)
print(
    f"Diagnosed epilepsy cases without any imaging data: {len(no_imaging_ep_ids)}/{len(ep_ids)}"
    f" ({len(no_imaging_ep_ids) / len(ep_ids):.2%})"
)

# %%
# analyse partially incomplete imaging data
no_imaging_df = deconf_df.loc[deconf_df.isna().all(axis=1)]
partial_imaging_df = deconf_df.loc[
    deconf_df.isna().any(axis=1) & ~deconf_df.isna().all(axis=1)
]
pct_missing_df = partial_imaging_df.isna().mean(axis=1).sort_values(ascending=False)
print(
    f"Total cases with partial imaging data: {partial_imaging_df.shape[0]}/{deconf_df.shape[0]}"
    f" ({partial_imaging_df.shape[0] / deconf_df.shape[0]:.2%})"
)
print(
    f"Partial cases with <10% missing imaging data: {(pct_missing_df < 0.1).sum()}/{partial_imaging_df.shape[0]}"
    f" ({(pct_missing_df < 0.1).sum() / partial_imaging_df.shape[0]:.2%})"
)
print(
    f"Partial cases with >10% missing imaging data: {(pct_missing_df > 0.1).sum()}/{partial_imaging_df.shape[0]}"
    f" ({(pct_missing_df > 0.1).sum() / partial_imaging_df.shape[0]:.2%})"
)
pct_missing_df.loc[pct_missing_df > 0.1].plot.hist(
    bins=20, title="Histogram of % missing imaging features"
)

# %%
# Filter out incomplete imaging rows (any-NaN)
# Filter out prediag cases where (Plus)_timeto_diagn_pastfuture < time_offset := (ageImaging - ageInitial) * 365
# Calculate Y = _timeto_diagn_orEnd - time_offset
# Calculate C = max(*_timeto_diagn_orEnd) - time_offset
# Calculate D = Y < C (should match bool field - prediag cases)
# Calculate T = Y if D else nan (should match _timeto_diagn_pastfuture - time_offset where D is True)
diseases = ["epilepsyPlus", "dementia"]
short_names = ["epilepsyPlus", "dem"]
dfs = {}

# good_gmvols_df = deconf_df.dropna(axis=0, how="any")
# good_gmvols_df = deconf_df.dropna(axis=0, how="all")
good_gmvols_df = deconf_df.loc[deconf_df.isna().mean(axis=1) < 0.1]
print(f"Filtered gmvols: {good_gmvols_df.shape[0]}/{deconf_df.shape[0]}")

for disease, short_name in zip(diseases, short_names):
    timeto_diagn_pastfuture = ukb_neuro_df[f"{short_name}_timeto_diagn_pastfuture"]
    prediag = timeto_diagn_pastfuture < time_offset
    df = ukb_neuro_df.loc[~prediag]
    print(f"Filtered {disease} prediag cases: {df.shape[0]}/{ukb_neuro_df.shape[0]}")

    df = df.join(good_gmvols_df, how="inner")

    time_to_diagn_orEnd = ukb_neuro_df[
        f"{short_name.replace('Plus', '')}_timeto_diagn_orEnd"
    ]
    Y = time_to_diagn_orEnd - time_offset
    C = c_df.loc[Y.index, "C"]
    # D = Y < C
    D = timeto_diagn_pastfuture > time_offset
    Y = Y.where(D, C)  # or swap?
    T = Y.where(D, np.nan)

    # print(f"Filtered {disease} non-positive Y cases: {(Y > 0).sum()}/{Y.shape[0]}")
    # df = df.loc[Y > 0]

    df["Y"] = Y
    df["C"] = C
    df["D"] = D
    df["T"] = T

    # df = df[gmvols_complete_df.columns.tolist() + ["ageImaging", "Y", "C", "D", "T"]]
    dfs[disease] = df
    print(
        f"Final {disease} df has {df['D'].sum()}/{df.shape[0]} events ({df['D'].mean():.2%})"
    )

    # df.to_parquet(DATA_DIR / f"ukb_{disease}_clean.parquet")

# %%
# Check D vs bool field
for disease in diseases:
    df = dfs[disease]
    bool_field = df[disease.replace("Plus", "")]
    D = df["D"]
    mismatch_df = df[bool_field != D]
    short_name = "dem" if disease == "dementia" else disease
    print(
        mismatch_df[
            [
                disease.replace("Plus", ""),
                "D",
                "Y",
                "C",
                "T",
                f"{short_name}_timeto_diagn_pastfuture",
            ]
        ].join(time_offset.to_frame(name="time_offset"))
    )

# %%
# Check Y > 0
for disease in diseases:
    df = dfs[disease]
    negative_Y_df = df[df["Y"] <= 0]
    print(negative_Y_df)

# %%
negative_Y_ids = [
    1070615,
    1455065,
    1529464,
    1632235,
    1758851,
    1766332,
    1936025,
    2228630,
    2256316,
    2278569,
    2557363,
    2796725,
    3019763,
    3048716,
    3245824,
    3299388,
    3321114,
    3388907,
    3449402,
    3514316,
    3927012,
    3997138,
    4031489,
    4106075,
    4226623,
    4326913,
    4455944,
    4517094,
    4883471,
    4883929,
    5714738,
    5886492,
]
negative_Y_ids.sort(
    key=lambda x: ukb_neuro_df.loc[x, "ageImaging"] - ukb_neuro_df.loc[x, "ageInitial"]
)

# %%
years_to_imaging = ukb_neuro_df["ageImaging"] - ukb_neuro_df["ageInitial"]
ax = years_to_imaging.hist(bins=20, alpha=0.9, label="all", density=True)

# overlay only the entries that exist in the index
subset = years_to_imaging.loc[negative_Y_ids]
subset.hist(bins=20, ax=ax, color="C1", alpha=0.4, label="negative_Y_ids", density=True)

ax.legend()
ax.set_xlabel("Years to imaging")
ax

# %%
dfs["epilepsyPlus"].loc[
    # [1236945],
    # [1070615],
    negative_Y_ids,
    [
        "epilepsy",
        "epilepsy_timeto_diagn_orEnd",
        "epilepsy_timeto_imaging_pastfuture",
        "epilepsy_timeto_diagn_pastfuture",
        "epilepsyPlus_timeto_diagn_pastfuture",
        "dementia",
        "dem_timeto_diagn_orEnd",
        "dem_timeto_imaging_pastfuture",
        "dem_timeto_diagn_pastfuture",
        "ageImaging",
        "ageInitial",
        "Y",
        "C",
        "D",
        "T",
    ],
].T

# # %%
# dfs["dementia"].loc[
#     [1180224, 1312159],
#     [
#         "epilepsy",
#         "epilepsy_timeto_diagn_orEnd",
#         "epilepsy_timeto_imaging_pastfuture",
#         "epilepsy_timeto_diagn_pastfuture",
#         "epilepsyPlus_timeto_diagn_pastfuture",
#         "dementia",
#         "dem_timeto_diagn_orEnd",
#         "dem_timeto_imaging_pastfuture",
#         "dem_timeto_diagn_pastfuture",
#         "ageImaging",
#         "ageInitial",
#         "Y",
#         "C",
#         "D",
#         "T",
#     ],
# ].T

# %%
# df = dfs["dementia"].copy()
df = dfs["epilepsyPlus"].copy()

# Rename feature columns to X_*
x_cols = [col for col in df.columns if col not in "YCDT"]
df = df.rename(columns={col: f"X_{i}" for i, col in enumerate(x_cols)})
x_cols = [col for col in df.columns if col not in "YCDT"]

# train-val-test split (60-20-20) stratified on D
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["D"])
val_df, test_df = train_test_split(test_df, test_size=0.5, stratify=test_df["D"])

# Standardize features
scaler = StandardScaler()
x_train = scaler.fit_transform(train_df[x_cols])
x_val = scaler.transform(val_df[x_cols])
x_test = scaler.transform(test_df[x_cols])

train_df.loc[:, x_cols] = x_train
val_df.loc[:, x_cols] = x_val
test_df.loc[:, x_cols] = x_test

# %%
import torch
from torch.nn.functional import sigmoid

from config import EPS
from dist import AsymptoticWeibull
from model import ParametricSurvivalModel, ParamMappingConfig, TrainConfig

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
param_transforms = {
    "alpha": lambda x: torch.clip(sigmoid(x), EPS, 1.0),
    "scale": lambda x: 100 * 365 * torch.clip(sigmoid(x), EPS, 1.0),
    "concentration": lambda x: 5 * torch.clip(sigmoid(x), EPS, 1.0),
}
model = ParametricSurvivalModel(
    dist_type=AsymptoticWeibull,
    # dist_type=torch.distributions.Weibull,
    mapping_cfg=ParamMappingConfig(
        d_in=len(x_cols),
        d_hidden=[
            len(x_cols) * 2,
            len(x_cols) * 2,
            # len(x_cols) * 2,
        ],
        param_transforms=param_transforms,
    ),
    device=device,
)
x_train = torch.tensor(train_df[x_cols].values, device=device, dtype=torch.float32)
y_train = torch.tensor(train_df["Y"].values, device=device, dtype=torch.float32)
d_train = torch.tensor(train_df["D"].values, device=device, dtype=torch.bool)
x_val = torch.tensor(val_df[x_cols].values, device=device, dtype=torch.float32)
y_val = torch.tensor(val_df["Y"].values, device=device, dtype=torch.float32)
d_val = torch.tensor(val_df["D"].values, device=device, dtype=torch.bool)
x_test = torch.tensor(test_df[x_cols].values, device=device, dtype=torch.float32)
y_test = torch.tensor(test_df["Y"].values, device=device, dtype=torch.float32)
d_test = torch.tensor(test_df["D"].values, device=device, dtype=torch.bool)
train_cfg = TrainConfig(
    n_epochs=5000,
    # learning_rate=3e-5,
    learning_rate=5e-6,
    weight_decay=1e-4,
    balance=True,
    patience=200,
    batch_size=None,
)
history = model.fit(
    x_train,
    y_train,
    d_train,
    x_val,
    y_val,
    d_val,
    train_cfg,
)
px.line(history, y=["train", "val"], title="Training History").show()

# %%
from eval import show_evals

c_train = torch.tensor(train_df["C"].values, device=device, dtype=torch.float32)
c_test = torch.tensor(test_df["C"].values, device=device, dtype=torch.float32)
show_evals(model, x_train, y_train, c_train, d_train, x_test, y_test, c_test, d_test)

# %%
new_df = pd.read_csv(DATA_DIR / "dementia_data.csv")
new_df

# %%
common_ids = dfs["dementia"].index.intersection(new_df["eid"])
common_ids

# %%
new_with_imaging_df = new_df.loc[
    new_df["dem_date"].notna() & new_df["date_image"].notna()
]
new_with_imaging_df = new_with_imaging_df.loc[
    ~new_with_imaging_df["eid"].isin(dem_df.loc[dem_df["dementia"] == True].index)
]
new_with_imaging_df

# %%
new_with_imaging_df.loc[new_with_imaging_df["eid"].isin(deconf_df.index)]

# %%
new_with_imaging_df.loc[new_with_imaging_df["eid"].isin(common_ids)]

# %%
dem_df = dfs["dementia"]
