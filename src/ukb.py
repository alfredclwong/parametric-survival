# %%
from pathlib import Path
import pandas as pd
import altair as alt
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots

alt.data_transformers.enable("vegafusion")

# %%
DATA_DIR = Path().cwd().parent / "data"
deconf_path = DATA_DIR / "Tbrain_deconf.csv"
ukb_gmvols_path = DATA_DIR / "ukb_GMVols.csv"
ukb_neuro_path = DATA_DIR / "ukb_neuro_data.csv"
fields_path = DATA_DIR / "fieldsum.txt"

# %%
deconf_df = pd.read_csv(deconf_path, index_col="eid")
ukb_gmvols_df = pd.read_csv(ukb_gmvols_path, index_col="eid")
ukb_neuro_df = pd.read_csv(ukb_neuro_path, index_col="ID")
fields_df = pd.read_csv(fields_path, sep="\t", index_col="field_id")

# %%
ukb_neuro_df["epilepsy"].sum()

# %%
def get_field_desc(vol_col):
    field_id = int(vol_col.split("_")[0][1:])
    try:
        return fields_df.loc[field_id, "title"]
    except KeyError:
        return "N/A"


# %%
# ukb_gmvols -> deconf with some pre-processing
# (502486, 328) -> (497252, 153)
ukb_gmvols_df.shape, deconf_df.shape

# %%
# Unprocessed variables have fatter tails, especially to the right
# But maybe this is actually ok?
# We have left/right brain volumes, I think transforming to avg/diff would be good, can try later
mutual_cols = sorted(
    list(set(ukb_gmvols_df.columns).intersection(set(deconf_df.columns)))
)
n_rows = 6
bin = alt.Bin(maxbins=30)
charts = []
for i, col in enumerate(mutual_cols[:n_rows]):
    before = (
        alt.Chart(ukb_gmvols_df.reset_index())
        .mark_bar()
        .encode(
            x=alt.X(col, bin=bin),
            y="count()",
            tooltip=alt.Tooltip(col, bin=bin),
        )
        .properties(
            title=f"{col} = {get_field_desc(col)} (Before deconfounding)",
            width=400,
            height=100,
        )
    )
    after = (
        alt.Chart(deconf_df.reset_index())
        .mark_bar()
        .encode(
            x=alt.X(col, bin=bin),
            y="count()",
            tooltip=alt.Tooltip(col, bin=bin),
        )
        .properties(
            title=f"{col} = {get_field_desc(col)} (After deconfounding)",
            width=400,
            height=100,
        )
    )
    charts.append(alt.hconcat(before, after))
alt.vconcat(*charts).configure_title(anchor="start").show()

# %%
# age = ageInitial (baseline assessment)
# T = 0 at imaging
# T_i = XXX_timeto_imaging_pastfuture is time from imaging to diagnosis or censoring
# T_d = XXX_timeto_diagn_orEnd is time from baseline assessment to diagnosis or censoring
# ageImaging - ageInitial = time from baseline assessment to imaging (in years)
# So we should have:
# T_d - T_i = (ageImaging - ageInitial) * 365 (in days)
# however there are some errors
# conclusion: just use T_d - (ageImaging - ageInitial) * 365 as the time variable, ignore processed T_i
disease = "dem"  # 2 errors
disease = "epilepsyPlus"  # 100 errors
t_df = ukb_neuro_df[
    [
        f"{disease}_timeto_imaging_pastfuture",
        f"{disease.replace('Plus', '')}_timeto_diagn_orEnd",
        "ageImaging",
        "ageInitial",
    ]
].dropna()
t_df["ageDiffDays"] = (t_df["ageImaging"] - t_df["ageInitial"]) * 365
t_df["timeto_diff_days"] = t_df[f"{disease.replace('Plus', '')}_timeto_diagn_orEnd"] - t_df[f"{disease}_timeto_imaging_pastfuture"]
t_df["error_days"] = t_df["ageDiffDays"] - t_df["timeto_diff_days"]
t_df.loc[t_df["error_days"].abs() > 5]

# %%
all_diseases = ["dem", "delirium", "epilepsy", "migraine", "ms", "AD", "vascD", "FTD", "PDcheck"]
c_df = pd.DataFrame(index=ukb_neuro_df.index)
for disease in all_diseases:
    c_df[f"C_{disease}"] = np.where(
        ukb_neuro_df["dementia" if disease == "dem" else "multiplesclerosis" if disease == "ms" else disease] == 0,
        ukb_neuro_df[f"{disease}_timeto_diagn_orEnd"],
        np.nan,
    )
c_df["C"] = c_df.max(axis=1)
c_df["C"] -= (ukb_neuro_df["ageImaging"] - ukb_neuro_df["ageInitial"]) * 365
c_df

# %%
tmp_df = ukb_neuro_df[["epilepsy"]].copy()
tmp_df["epilepsy_timeto_diagn_orEnd"] = ukb_neuro_df["epilepsy_timeto_diagn_orEnd"].copy()
tmp_df["timeto_imaging"] = (ukb_neuro_df["ageImaging"] - ukb_neuro_df["ageInitial"]) * 365
tmp_df["Y"] = tmp_df["epilepsy_timeto_diagn_orEnd"] - tmp_df["timeto_imaging"]
tmp_df = tmp_df.join(c_df, how="left")
tmp_df["check"] = (tmp_df["Y"] < tmp_df["C"]) == tmp_df["epilepsy"]
tmp_df = tmp_df.loc[tmp_df[["Y", "C"]].notna().any(axis=1)]
tmp_df.loc[tmp_df["check"] == False]

# %%
ukb_neuro_df.loc[1024377, ["epilepsy", "epilepsy_timeto_diagn_orEnd", "dementia", "dem_timeto_diagn_orEnd", "ageImaging", "ageInitial"]]

# %%
ukb_neuro_df.loc[1010931, ["epilepsy", "epilepsy_timeto_diagn_orEnd", "dementia", "dem_timeto_diagn_orEnd", "ageImaging", "ageInitial"]]

# %%
diseases = {"dementia": "dem", "epilepsyPlus": "epilepsy"}
neuro_df = {}
neuro_df["ageImaging"] = ukb_neuro_df["ageImaging"].copy()
neuro_df["C"] = c_df["C"].copy()
for disease, short_name in diseases.items():
    neuro_df[f"D_{short_name}"] = ukb_neuro_df[disease.replace("Plus", "")].copy()
    neuro_df[f"Y_{short_name}"] = (
        ukb_neuro_df[f"{short_name}_timeto_diagn_orEnd"]
        - (ukb_neuro_df["ageImaging"] - ukb_neuro_df["ageInitial"]) * 365
    )
    neuro_df[f"T_{short_name}"] = np.where(neuro_df[f"D_{short_name}"] == 1, neuro_df[f"Y_{short_name}"], np.nan)
neuro_df = pd.DataFrame(neuro_df)
neuro_df

# %%
for disease in ["dem", "epilepsy"]:
    n = neuro_df.shape[0]
    d_sum = neuro_df[f"D_{disease}"].sum()
    postdiag = neuro_df[f"Y_{disease}"] > 0
    imaged = neuro_df[f"Y_{disease}"].notna()
    d_postdiag_sum = neuro_df.loc[postdiag, f"D_{disease}"].sum()
    d_imaged_sum = neuro_df.loc[imaged, f"D_{disease}"].sum()
    d_postdiag_imaged_sum = neuro_df.loc[postdiag & imaged, f"D_{disease}"].sum()
    print((postdiag==(imaged & postdiag)).all())  # all postdiag are imaged
    print(f"{disease}: n={n}, D sum={d_sum}, D postdiag sum={d_postdiag_sum}, D imaged sum={d_imaged_sum}, D postdiag & imaged sum={d_postdiag_imaged_sum}")

# %%
disease = "epilepsy"
df = neuro_df.join(deconf_df, how="left")

imaging_cols = deconf_df.columns.tolist()
surv_cols = list("TDY")
df = df[[
    *[f"{x}_{disease}" for x in surv_cols],
    "C",
    "ageImaging",
    *imaging_cols,
]]
df = df.rename(columns={f"{x}_{disease}": x for x in surv_cols})

print(f"Before filtering: {df.shape[0]} ({df['D'].sum()} events)")

# Filter to imaged events only
any_imaged = df["ageImaging"].notna()
df = df.loc[any_imaged]
print(f"After filtering to imaged events: {df.shape[0]} ({df['D'].sum()} events)")

# Filter to post-diagnosis events only
any_postdiag = df["Y"] > 0
df = df.loc[any_postdiag]
print(f"After filtering to post-diagnosis events: {df.shape[0]} ({df['D'].sum()} events)")

# Filter for complete deconf data
df = df.loc[df[deconf_df.columns].notna().all(axis=1)]
print(f"After filtering to complete deconf data: {df.shape[0]} ({df['D'].sum()} events)")

df

# %%
tmp_df = ukb_neuro_df.loc[1236945, ["epilepsy", "epilepsy_timeto_diagn_orEnd", "ageImaging", "ageInitial", "epilepsy_timeto_imaging_pastfuture"]]
tmp_df["Y"] = tmp_df["epilepsy_timeto_diagn_orEnd"] - (tmp_df["ageImaging"] - tmp_df["ageInitial"]) * 365
tmp_df

# %%
# check that Y < C where D == 1
invalid_y_c = df.loc[df["D"] != (df["Y"] < df["C"])]
valid_y_c = df.loc[df["D"] == (df["Y"] < df["C"])]
print(f"Number of invalid Y > C where D == 1: {invalid_y_c.shape[0]}")
invalid_y_c

# %%
valid_y_c

# %%
df.loc[2995308, ["D", "Y", "C", "ageImaging"]]

# %%
ukb_neuro_df.loc[2995308, ["epilepsy", "epilepsy_timeto_diagn_orEnd", "ageImaging", "ageInitial", "epilepsy_timeto_imaging_pastfuture"]]

# %%
# Pre-processing
# 1. Split train/val/test
# 2. Fit a StandardScaler on train, transform all
# 3. Save to parquet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=0,
    stratify=df["D"],
)
train_df, val_df = train_test_split(
    train_df, test_size=0.2, random_state=0, stratify=train_df["D"]
)
x_cols = ["ageImaging"] + imaging_cols
scaler = StandardScaler()
scaler.fit(train_df[x_cols])
train_df[x_cols] = scaler.transform(train_df[x_cols])
val_df[x_cols] = scaler.transform(val_df[x_cols])
test_df[x_cols] = scaler.transform(test_df[x_cols])

# # Add a constant column for bias term
# train_df["bias"] = 1.0
# val_df["bias"] = 1.0
# test_df["bias"] = 1.0
# x_cols = ["bias"] + x_cols

train_df.to_parquet(DATA_DIR / f"ukb_{disease}_train.parquet")
val_df.to_parquet(DATA_DIR / f"ukb_{disease}_val.parquet")
test_df.to_parquet(DATA_DIR / f"ukb_{disease}_test.parquet")
train_df

# %%
# Let's train a binary classifier on D using the x_cols (and C?) as features
from mlp import MLP, train_classifier
import torch

classifier = MLP(input_dim=len(x_cols), hidden_dims=[len(x_cols), len(x_cols)], output_dim=2)
x_train = torch.tensor(train_df[x_cols].values, dtype=torch.float32)
y_train = torch.tensor(train_df["D"].values, dtype=torch.long)
x_val = torch.tensor(val_df[x_cols].values, dtype=torch.float32)
y_val = torch.tensor(val_df["D"].values, dtype=torch.long)
optimizer = torch.optim.Adam(classifier.parameters(), lr=2e-3, weight_decay=1e-4)
n_epochs = 1000
history = train_classifier(classifier, x_train, y_train, x_val, y_val, optimizer, n_epochs, patience=200)
px.line(history, y=["train", "val"], title="Training History").show()

# %%
with torch.no_grad():
    x_test = torch.tensor(test_df[x_cols].values, dtype=torch.float32)
    y_test = torch.tensor(test_df["D"].values, dtype=torch.long)
    y_test_pred = classifier(x_test).softmax(dim=1)[:, 1]

from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_test.numpy(), y_test_pred.numpy())
print(f"Test ROC AUC: {roc_auc:.4f}")

# %%
pred_df = test_df[["D", "Y", "C"]].copy()
pred_df["D_pred"] = y_test_pred.numpy()

# %%
y_test_pred_hist = px.histogram(
    pred_df,
    x="D_pred",
    color="D",
    barmode="overlay",
    nbins=50,
    title="Predicted Probability of Event by True Event Status",
)
y_test_pred_hist.show()

# %%
pred_df.loc[pred_df["D"] == 1, "D_pred"].mean(), pred_df.loc[pred_df["D"] == 0, "D_pred"].mean()

# %%
pred_df.sort_values(["D", "Y"], ascending=[False, True]).head(20)

# %%
test_df.loc[2995308, ["T", "D", "Y", "C"]]

# %%
# use altair to plot y against log(D_pred)
# Calculate log(D_pred) and add as a new column
pred_df["log_D_pred"] = np.log(pred_df["D_pred"].clip(1e-8))
y_vs_dpred = (
    alt.Chart(pred_df.reset_index())
    .mark_circle(opacity=0.5
    ).encode(
        x="Y",
        y="log_D_pred",
        color="D:N",
        tooltip=["Y", "D", "D_pred", "log_D_pred", "C", "ID"],
    )
    .properties(
        title="Predicted Probability of Event vs. Time to Event/Censoring (D_pred log scale)",
        width=600,
        height=400,
    )
)
y_vs_dpred.show()

# %%
from model import ParametricSurvivalModel, ParamMappingConfig, TrainConfig
from dist import AsymptoticWeibull
from config import EPS
from torch.nn.functional import sigmoid
import torch

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
param_transforms = {
    # "alpha": lambda x: torch.clip(sigmoid(x), EPS, 1.0),
    "scale": lambda x: 100 * 365 * torch.clip(sigmoid(x), EPS, 1.0),
    "concentration": lambda x: 5 * torch.clip(sigmoid(x), EPS, 1.0),
}
model = ParametricSurvivalModel(
    # dist_type=AsymptoticWeibull,
    dist_type=torch.distributions.Weibull,
    mapping_cfg=ParamMappingConfig(
        d_in=len(x_cols),
        d_hidden=[
            len(x_cols) * 2,
            len(x_cols) * 2,
            len(x_cols) * 2,
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
    learning_rate=2e-5,
    weight_decay=1e-4,
    balance=False,
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
pred_df = model.predict(x_test, y_test, c_test, d_test, as_pl=False)
pred_df

# %%
pred_df.loc[pred_df["D"] == 1]

# %%
pred_df.loc[pred_df["D"] == 0].head(20)

# %%
from sklearn.metrics import brier_score_loss

@torch.no_grad()
def calculate_brier_score(
    model: ParametricSurvivalModel,
    x: torch.Tensor,
    y_true: np.ndarray,
    times: np.ndarray,
) -> pd.DataFrame:
    """Calculate Brier score at specified times.

    Returns:
        pd.DataFrame: DataFrame with columns 'time' and 'brier_score'.
    """
    model.eval()

    x = x.to(model.device, dtype=torch.float32)
    params = model.mapping(x)
    dist = model.dist_type(**params)
    ts = torch.tensor(times, device=model.device, dtype=torch.float32)
    d_pred = dist.cdf(ts[:, None]).cpu().numpy().T
    d_true = (y_true[:, None] <= times[None, :]).astype(int)
    print(d_pred.shape, d_true.shape)
    brier_scores = []
    for i, t in enumerate(times):
        bs = brier_score_loss(d_true[:, i], d_pred[:, i])
        brier_scores.append(bs)
    return pd.DataFrame({"time": times, "brier_score": brier_scores})

times = np.arange(0, 5000, 100)
brier_df = calculate_brier_score(model, x_test, y_test.cpu().numpy(), times)
quartiles = np.percentile(y_test.cpu().numpy(), [25, 50, 75])
fig = px.line(brier_df, x="time", y="brier_score", title="Brier Score over Time")
for q in quartiles:
    fig.add_vline(x=q, line_dash="dash", line_color="red")
fig.show()

# %%
test_df[x_cols].describe()
