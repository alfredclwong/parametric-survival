# Generate visuals for README.md

# %%
from pathlib import Path
import numpy as np
import polars as pl

from vis import plot_feature_histograms

# %%
"""
0. data
    - dummy.parquet
1. dist.py
    - Contains the `Distribution` class and its subclasses for various survival distributions.
    - Implements methods for sampling, PDF, CDF, and median calculations.
2. mapping.py
    - Defines the `ParamMapping` class for mapping input features to distribution parameters.
    - Supports linear mappings and parameter transformations.
3. model.py
    - Implements the `ParametricSurvivalModel` class for modeling survival data.
    - Provides methods for fitting the model, predicting median survival time, and calculating negative log likelihood.
4. synth.py
    - Contains the `SynthConfig` class for generating synthetic survival data.
    - Implements a function to generate synthetic datasets based on specified configurations.
5. train.py
    - Defines the training process for the survival model.
    - Includes functions for training, validation, and evaluation of the model.
6. eval.py
    - Implements evaluation metrics for survival models, including binary classification metrics and concordance index.
    - Provides visualization functions for ROC curves and confusion matrices.
7. vis.py
    - Contains visualization functions for plotting feature histograms, model parameters, and evaluation metrics.
    - Utilizes Altair for interactive visualizations.
"""

# %%
ROOT_DIR = Path().cwd().parent
DOCS_DIR = ROOT_DIR / "docs"
DATA_DIR = ROOT_DIR / "data"

# %%
# 0. data
df = pl.read_parquet(DATA_DIR / "dummy_processed.parquet")
plot_feature_histograms(df, features=["T", "C", "D", "Y"]).save(DOCS_DIR / "tcdy.png")
df.select(["T", "C", "D" "Y"]).describe()

# %%
# 1. dist.py
x = np.linspace(0, 3000, 101)

weibull_chart = Weibull.plot(x, {
    "shape": np.array([0.5, 1.0, 2.0, 5.0]),
    "scale": np.array([1000, 1000, 1000, 1000])
})
weibull_chart.save(DOCS_DIR / "weibull.png")

scaled_weibull_chart = ScaledWeibull.plot(x, {
    "A": np.array([0.4, 0.6, 0.8, 1.0]),
    "shape": np.array([0.5, 1.0, 2.0, 5.0]),
    "scale": np.array([1000, 1000, 1000, 1000])
})
scaled_weibull_chart.save(DOCS_DIR / "scaled_weibull.png")

# %%
# 2. mapping.py
