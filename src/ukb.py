# %%
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots

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
# Notes: seems like deconf removed some extreme values, reducing 502486 to 497252
# Tbh the values didn't seem too extreme, fairly realistic tails
# But Xin probably knows better
# Also deconf removes some entire features, reducing 328 to 153
# We have left/right brain volumes, I think transforming to avg/diff would be good, can try later
# For now just normalise
mutual_cols = list(set(ukb_gmvols_df.columns).intersection(set(deconf_df.columns)))
n_cols = 5
fig = make_subplots(
    rows=n_cols,
    cols=2,
    subplot_titles=("Before deconfounding", "After deconfounding"),
)
fig.update_layout(height=300 * n_cols, width=900)
for i, col in enumerate(mutual_cols[:n_cols]):
    row = i + 1
    fig.add_trace(px.histogram(ukb_gmvols_df, x=col).data[0], row=row, col=1)
    fig.add_trace(px.histogram(deconf_df, x=col).data[0], row=row, col=2)
fig.show()

# %%
# [col for col in ukb_neuro_df.columns if "timeto_diagn" in col]
# disease = "dem"
disease = "epilepsy"

# %%
y = ukb_neuro_df[f"{disease}_timeto_diagn_orEnd"] - (ukb_neuro_df["ageImaging"] - ukb_neuro_df["age"]) * 365
y = y.dropna().round()
d = ukb_neuro_df.loc[y.index, "dementia" if disease == "dem" else "epilepsy"]
t = ukb_neuro_df.loc[y.index, f"{disease}_timeto_imaging_pastfuture"]
c = np.where(d == 1, y + 1, y)
y, t

# %%
df = pd.DataFrame({
    "Y": y,
    "D": d,
    "T": t,
    "C": c,
    "ageImaging": ukb_neuro_df.loc[y.index, "ageImaging"],
})
df = df.loc[df["Y"] >= 0]  # remove negative Y
df

# %%
import numpy as np
X = np.arange(9).reshape(3, 3)
print(X)
np.fill_diagonal(X, 0)
print(X)
print(X.max().max())

# %%
from plotly import graph_objects as go
ts = np.arange(0, 1, 0.01)
fig = go.Figure()
for k in [0.5, 1.0, 2.0]:
    h = k * ts ** (k - 1)
    fig.add_trace(go.Scatter(x=ts, y=h, mode="lines", name=f"k={k}"))
fig.update_layout(title="Weibull Hazard Functions", xaxis_title="Time (days)", yaxis_title="Hazard")
fig.show()

# %%
(df.loc[df["D"] == 1, "Y"] - df.loc[df["D"] == 1, "T"]).describe()

# %%
df.loc[(df["Y"] - df["T"]).abs() > 5]

# %%
weird_ids = df.loc[(df["Y"] - df["T"]).abs() > 5].index
weird_ids

# %%
ukb_neuro_df.loc[weird_ids, ["age", "ageInitial", "ageImaging", "dem_timeto_diagn_orEnd", "dem_timeto_imaging_pastfuture", "dementia"]]

# %%
df = df.drop(weird_ids)

# %%
# Notes: there are four points of interest on the time axis
# 1. Baseline assessment (ageAssessment)
# 2. Imaging visit (ageImaging)
# 3. Diagnosis of dementia (dem_timeto_diagn)
# 4. Right-censoring (dem_timeto_diagn_orEnd)
# For now, just use dem_timeto_diagn_orEnd as the time variable
# This is equivalent to Y = min(T, C) where T is time to dementia and C is time to censoring
# And D = 1(T <= C) is the event indicator

# %%
df = df.join(deconf_df, how="inner")
df

# %%
# Start off with only complete cases - can try imputation later
imaging_cols = [col for col in df.columns if col.startswith("x")]
complete_imaging = df[imaging_cols].notna().all(axis=1)
complete_imaging.mean()  # around 6% complete cases

# %%
# Pre-processing
# 1. Split train/val/test
# 2. Fit a StandardScaler on train, transform all
# 3. Save to parquet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

complete_imaging_df = df[complete_imaging].copy()
train_df, test_df = train_test_split(
    complete_imaging_df,
    test_size=0.2,
    random_state=0,
    stratify=complete_imaging_df["D"],
)
train_df, val_df = train_test_split(
    train_df, test_size=0.2, random_state=0, stratify=train_df["D"]
)
scaled_cols = ["ageImaging"] + imaging_cols
scaler = StandardScaler()
scaler.fit(train_df[scaled_cols])
train_df[scaled_cols] = scaler.transform(train_df[scaled_cols])
val_df[scaled_cols] = scaler.transform(val_df[scaled_cols])
test_df[scaled_cols] = scaler.transform(test_df[scaled_cols])

# Add a constant column for bias term
train_df["bias"] = 1.0
val_df["bias"] = 1.0
test_df["bias"] = 1.0

train_df.to_parquet(DATA_DIR / "ukb_dem_train.parquet")
val_df.to_parquet(DATA_DIR / "ukb_dem_val.parquet")
test_df.to_parquet(DATA_DIR / "ukb_dem_test.parquet")
train_df

# %%
train_df.loc[train_df["T"].notna()]

# %%
test_df.loc[test_df["T"].notna()]

# %%
train_df.describe()

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
    "alpha": lambda x: torch.clip(sigmoid(x), EPS, 1.0),
    "scale": lambda x: 10 * 365 * torch.clip(sigmoid(x), EPS, 1.0),
    "concentration": lambda x: 5 * torch.clip(sigmoid(x), EPS, 1.0),
}
model = ParametricSurvivalModel(
    dist_type=AsymptoticWeibull,
    # dist_type=torch.distributions.Weibull,
    mapping_cfg=ParamMappingConfig(
        d_in=len(scaled_cols) + 1,
        d_hidden=[len(scaled_cols) * 2] * 2,
        # d_hidden=[],
        param_transforms=param_transforms,
    ),
    device=device,
)
x_cols = ["bias"] + scaled_cols
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
    learning_rate=1e-5,
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
pred_df = model.predict(x_test, y_test, c_test, d_test)
pred_df = pred_df.to_pandas()
pred_df

# %%
pred_df.loc[pred_df["D"] == 1]

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
