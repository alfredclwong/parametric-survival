from pathlib import Path
from typing import Optional

import altair as alt
import joblib
import polars as pl
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


def row_filter(df, cond, desc, verbose=True):
    """Filter rows based on a condition and print the number of rows filtered."""
    before = df.height
    df_filtered = df.filter(cond)
    after = df_filtered.height
    if verbose:
        print(
            f"Filtered {before - after} rows ({(before - after) / before:.2%}) where {desc}."
        )
    return df_filtered


def process_df(
    df: pl.DataFrame, X_scalers: Optional[dict[str, StandardScaler]] = None
) -> tuple[pl.DataFrame, Optional[dict[str, StandardScaler]]]:
    # Process csv data and save as parquet
    # T = diagnosis time
    # C = censoring time
    # Y = min(T, C)
    # D = T < C
    # X = brain data features (X_1, X_2, ...)
    original_height = df.height
    X_cols = [col for col in df.columns if col.startswith("X_")]

    # Convert nan to null
    df = df.fill_nan(None)

    # Drop rows where T and C are both null
    df = row_filter(
        df, ~(pl.col("T").is_null() & pl.col("C").is_null()), "T and C are both null"
    )

    # Drop rows where T < 0, keeping null values
    df = row_filter(df, pl.col("T").is_null() | (pl.col("T") >= 0), "T < 0")

    # Drop rows where all X features are null
    df = row_filter(
        df,
        ~pl.all_horizontal([pl.col(col).is_null() for col in X_cols]),
        "all X features are null",
    )

    print(
        f"Remaining rows: {df.height}/{original_height} "
        f"({df.height / original_height:.2%})"
    )

    # Standardise X features
    if X_scalers is not None:
        for col in tqdm(X_cols, desc="Standardizing X features"):
            # Extract column as numpy array, fit and transform, then replace column
            values = df.select(col).to_numpy().reshape(-1, 1)
            if SCALERS_PATH.exists():
                scaled = X_scalers[col].transform(values)
            else:
                scaled = X_scalers[col].fit_transform(values)
            df = df.with_columns(pl.Series(col, scaled.flatten()))

    # Fill NaN values in X features with 0
    df = df.with_columns([pl.col(col).fill_nan(0) for col in X_cols])

    # Calculate Y and D
    df = df.with_columns(
        Y=pl.min_horizontal(["T", "C"]),
        D=(pl.col("T") < pl.col("C")).fill_null(False),
    )
    print("Added columns Y = min(T, C) and D = T < C.")

    # Reorder columns
    df = df.select(["T", "C", "Y", "D"] + X_cols)

    return df, X_scalers


def plot_feature_histograms(df, features=None, maxbins=20, n_cols=5, subplot_size=120):
    # Plot a histogram of each feature using Altair
    charts = []
    if features is None:
        features = df.columns
    dtypes = df.select(pl.col(features)).dtypes
    for feature, dtype in zip(features, dtypes):
        is_categorical = dtype in [pl.Categorical, pl.String, pl.Boolean]
        if is_categorical:
            x = alt.X(feature).type("nominal")
        else:
            x = alt.X(feature).bin(maxbins=maxbins)
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(x=x, y="count()")
            .properties(title=feature, width=subplot_size, height=subplot_size)
        )
        charts.append(chart)
    n_rows = (len(charts) + n_cols - 1) // n_cols
    charts = [
        alt.hconcat(*charts[i * n_cols : (i + 1) * n_cols]) for i in range(n_rows)
    ]
    return alt.vconcat(*charts).resolve_scale(x="shared", y="shared")


if __name__ == "__main__":
    alt.data_transformers.enable("vegafusion")
    alt.renderers.enable("browser")
    ROOT_DIR = Path().cwd()
    DATA_DIR = ROOT_DIR / "data"
    RAW_CSV_PATH = DATA_DIR / "dummy.csv"
    PROCESSED_PARQUET_PATH = DATA_DIR / "dummy_processed.parquet"
    SCALERS_PATH = DATA_DIR / "scalers.joblib"
    N_X_FEATURES = 40

    X_cols = [f"X_{i + 1}" for i in range(N_X_FEATURES)]
    csv_headers = ["T", "C"] + X_cols
    df = pl.read_csv(RAW_CSV_PATH, has_header=False, new_columns=csv_headers)
    if SCALERS_PATH.exists():
        X_scalers = joblib.load(SCALERS_PATH)
    else:
        X_scalers = {col: StandardScaler() for col in X_cols}
    processed_df, X_scalers = process_df(df, X_scalers)

    # PROCESSED_PARQUET_PATH.unlink(missing_ok=True)
    if PROCESSED_PARQUET_PATH.exists():
        existing_processed_df = pl.read_parquet(PROCESSED_PARQUET_PATH)
        assert (
            (existing_processed_df == processed_df).fill_null(True).to_numpy().all()
        ), (
            "Processed DataFrame does not match existing processed DataFrame."
            + "\nExisting DataFrame:\n"
            + str(processed_df)
            + "\nExisting Processed DataFrame:\n"
            + str(existing_processed_df)
        )
    else:
        df.write_parquet(PROCESSED_PARQUET_PATH)
        joblib.dump(X_scalers, DATA_DIR / "scalers.joblib")

    df = pl.read_parquet(PROCESSED_PARQUET_PATH)
    df.describe()
    df.plot.scatter(
        x="Y",
        y="C",
        color="D",
        tooltip=["Y", "C"],
    ).properties(
        title="Diagnosis/Censoring Times",
        width=600,
        height=400,
    )
    feature_histograms = plot_feature_histograms(df)
    feature_histograms.show()
