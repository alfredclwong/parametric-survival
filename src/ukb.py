# %%
import pandas as pd
from pathlib import Path

# %%
root_dir = Path.cwd().parent
data_dir = root_dir / "data"
gm_vols_path = data_dir / "ukb_GMVols.csv"
neuro_data_path = data_dir / "ukb_neuro_data.csv"
field_sum_path = data_dir / "fieldsum.txt"

# %%
def print_nan_stats(df: pd.DataFrame): 
    isna = df.isna()
    print(f"{isna.sum().sum()}/{df.size} NaN vals ({isna.mean().mean():.2%})")
    print(f"{isna.all(axis=1).sum()}/{df.shape[0]} all-NaN rows ({isna.all(axis=1).mean():.2%})")
    print(f"{isna.all(axis=0).sum()}/{df.shape[1]} all-NaN cols ({isna.all(axis=0).mean():.2%})")
    print(f"{isna.any(axis=1).sum()}/{df.shape[0]} any-NaN rows ({isna.any(axis=1).mean():.2%})")
    print(f"{isna.any(axis=0).sum()}/{df.shape[0]} any-NaN cols ({isna.any(axis=0).mean():.2%})")

# %%
gm_vols = pd.read_csv(gm_vols_path, index_col="eid")
print("Raw gm_vols")
print_nan_stats(gm_vols)

non_repeat_cols = [col for col in gm_vols.columns if col.endswith("_2_0")]
gm_vols = gm_vols[non_repeat_cols]
print("Removed repeat image (_3_0) cols")
print_nan_stats(gm_vols)

gm_vols.dropna(axis=0, how="all", inplace=True)
print("Dropped NaN rows")
print_nan_stats(gm_vols)

# %%
field_sum = pd.read_csv(field_sum_path, delimiter="\t", index_col="field_id")
field_sum

# %%
col_ids = [int(col[1:-4]) for col in gm_vols.columns]
col_sums = [field_sum.loc[col_id, "title"] for col_id in col_ids]
list(zip(col_ids, col_sums))

# %%
gm_vols[gm_vols.isna().any(axis=1)]

# %%
gm_vols["x25001_2_0"] / gm_vols["x25002_2_0"] / gm_vols["x25000_2_0"]

# %%
neuro_data = pd.read_csv(neuro_data_path, index_col="ID")
neuro_data

# %%
neuro_cols = ["ageImaging", "sex", "epilepsy_timeto_imaging_pastfuture", "dem_timeto_imaging_pastfuture"]
neuro_data = neuro_data[neuro_cols]
neuro_data

# %%
[col for col in neuro_data.columns if "time" in col]

# %%
(neuro_data["epilepsy_timeto_imaging_pastfuture"] == neuro_data["dem_timeto_imaging_pastfuture"]).sum()

# %%
neuro_data["dem_timeto_diagn_orEnd"].isna().sum()
neuro_data["epilepsy_timeto_diagn_orEnd"].isna().sum()

# %%
[col for col in neuro_data.columns if "dem" in col]

# %%
((1 - neuro_data["dementia"]) * (1 - neuro_data["epilepsy"])).sum()

# %%
# number of not dem and not epilepsy
(neuro_data["dem_timeto_diagn_orEnd"] == neuro_data["epilepsy_timeto_diagn_orEnd"]).sum()
