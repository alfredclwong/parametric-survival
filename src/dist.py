from abc import ABC, abstractmethod
from collections.abc import Sequence
from scipy.stats import weibull_min

import altair as alt
import numpy as np
import polars as pl

from config import EPS


class Distribution(ABC):
    param_names: Sequence[str]

    @staticmethod
    def _check_params(params: dict[str, np.ndarray]):
        pass

    @staticmethod
    @abstractmethod
    def cdf(x: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def pdf(
        x: np.ndarray, params: dict[str, np.ndarray], log: bool = False
    ) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def median(params: dict[str, np.ndarray]) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def sample(n_samples: int, params: dict[str, np.ndarray]) -> np.ndarray:
        pass

    @classmethod
    def plot(
        cls, x: np.ndarray, params: dict[str, np.ndarray], max_y: float = 2.5
    ) -> alt.TopLevelMixin:
        cls._check_params(params)
        # x.shape: (n_samples,)
        # param.shape: (n_dists,)
        # tile x and params to match each other for vectorised computation
        n_samples = x.shape[0]
        n_dists = len(next(iter(params.values())))
        x = np.tile(x[:, np.newaxis], (1, n_dists))
        params = {
            k: np.tile(v[np.newaxis, :], (n_samples, 1)) for k, v in params.items()
        }

        y_cdf = cls.cdf(x, params)
        y_pdf = cls.pdf(x, params).clip(0, max_y)
        df = pl.DataFrame(
            {
                "x": x.flatten(),
                "cdf": y_cdf.flatten(),
                "pdf": y_pdf.flatten(),
                **{k: v.flatten() for k, v in params.items()},
            }
        )
        fstring = ", ".join(f"{k}={{}}" for k in params.keys())
        # fstring = f"{cls.__name__}({fstring})"
        df = df.with_columns(
            pl.format(
                fstring,
                *[pl.col(k).round_sig_figs(3) for k in params.keys()],
            ).alias("params")
        )
        cdf_chart = (
            alt.Chart(df)
            .mark_line()
            .encode(
                x=alt.X("x", title="x"),
                y=alt.Y("cdf", title="CDF"),
                color=alt.Color("params:N"),
            )
            .properties(title=f"{cls.__name__} CDF")
        )
        pdf_chart = (
            alt.Chart(df)
            .mark_line()
            .encode(
                x=alt.X("x", title="x"),
                y=alt.Y("pdf", title="PDF"),
                color=alt.Color("params:N"),
            )
            .properties(title=f"{cls.__name__} PDF")
        )
        return alt.hconcat(cdf_chart, pdf_chart)


class Weibull(Distribution):
    param_names = ["scale", "shape"]

    @staticmethod
    def _check_params(params: dict[str, np.ndarray]):
        if "scale" not in params or "shape" not in params:
            raise ValueError(
                "Weibull distribution requires 'scale' and 'shape' parameters."
            )
        if not np.all(params["scale"] > 0):
            raise ValueError("Weibull 'scale' parameter must be positive.")
        if not np.all(params["shape"] > 0):
            raise ValueError("Weibull 'shape' parameter must be positive.")

    @staticmethod
    def cdf(x: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
        Weibull._check_params(params)
        scale, shape = params["scale"], params["shape"]
        y = 1 - np.exp(-((x / scale) ** shape))
        return np.where(x >= 0, y, 0.0)

    @staticmethod
    def pdf(
        x: np.ndarray,
        params: dict[str, np.ndarray],
        log: bool = False,
        eps: float = EPS,
    ) -> np.ndarray:
        Weibull._check_params(params)
        scale, shape = params["scale"], params["shape"]
        if log:
            y = (
                np.log(shape + eps)
                - np.log(scale + eps)
                + (shape - 1) * (np.log(x + eps) - np.log(scale + eps))
                - ((x / (scale + eps)) ** shape)
            )
        else:
            y = (
                (shape / scale)
                * (x / scale) ** (shape - 1)
                * np.exp(-((x / scale) ** shape))
            )
        return np.where(
            x > 0, y, 0.0
        )  # define pdf(0) = 0, slightly different from wikipedia definition

    @staticmethod
    def median(params: dict[str, np.ndarray]) -> np.ndarray:
        Weibull._check_params(params)
        scale, shape = params["scale"], params["shape"]
        return scale * (np.log(2) ** (1 / shape + EPS))

    @staticmethod
    def sample(n_samples: int, params: dict[str, np.ndarray]) -> np.ndarray:
        # Return (n_samples, n_dists)
        Weibull._check_params(params)
        scale, shape = params["scale"], params["shape"]
        u = np.random.uniform(0, 1, size=(n_samples, len(scale)))
        x = scale * (-np.log(1 - u + EPS)) ** (1 / (shape + EPS))
        # x = np.array(
        #     [
        #         weibull_min.rvs(c=shape_i, scale=scale_i, size=n_samples)
        #         for scale_i, shape_i in zip(scale, shape)
        #     ]
        # ).T
        return x


class Exponential(Distribution):
    param_names = ["k"]

    @staticmethod
    def cdf(x: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
        k = params["k"]
        return Weibull.cdf(x, {"scale": 1 / (k + EPS), "shape": np.ones_like(x)})

    @staticmethod
    def pdf(
        x: np.ndarray, params: dict[str, np.ndarray], log: bool = False
    ) -> np.ndarray:
        k = params["k"]
        return Weibull.pdf(
            x, {"scale": 1 / (k + EPS), "shape": np.ones_like(x)}, log=log
        )

    @staticmethod
    def median(params: dict[str, np.ndarray]) -> np.ndarray:
        k = params["k"]
        return Weibull.median({"scale": 1 / (k + EPS), "shape": np.ones_like(k)})

    @staticmethod
    def sample(n_samples: int, params: dict[str, np.ndarray]) -> np.ndarray:
        k = params["k"]
        return Weibull.sample(
            n_samples, {"scale": 1 / (k + EPS), "shape": np.ones_like(k)}
        )


class ScaledExponential(Distribution):
    param_names = ["A", "k"]

    @staticmethod
    def _check_params(params: dict[str, np.ndarray]):
        if any(k not in params for k in ["A", "k"]):
            raise ValueError(
                "ScaledExponential distribution requires 'A' and 'k' parameters."
            )
        if not np.all(params["A"] > 0):
            raise ValueError("ScaledExponential 'A' parameter must be positive.")
        if not np.all(params["k"] > 0):
            raise ValueError("ScaledExponential 'k' parameter must be positive.")

    @staticmethod
    def cdf(x: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
        A, k = params["A"], params["k"]
        return A * Weibull.cdf(x, {"scale": 1 / (k + EPS), "shape": np.ones_like(x)})

    @staticmethod
    def pdf(
        x: np.ndarray, params: dict[str, np.ndarray], log: bool = False
    ) -> np.ndarray:
        A, k = params["A"], params["k"]
        y = Weibull.pdf(x, {"scale": 1 / (k + EPS), "shape": np.ones_like(x)}, log=log)
        return np.log(A + EPS) + y if log else A * y

    @staticmethod
    def median(params: dict[str, np.ndarray]) -> np.ndarray:
        ScaledExponential._check_params(params)
        A, k = params["A"], params["k"]
        return A * Weibull.median({"scale": 1 / (k + EPS), "shape": np.ones_like(k)})

    @staticmethod
    def sample(n_samples: int, params: dict[str, np.ndarray]) -> np.ndarray:
        ScaledExponential._check_params(params)
        A, k = params["A"], params["k"]
        x = Weibull.sample(
            n_samples, {"scale": 1 / (k + EPS), "shape": np.ones_like(k)}
        )  # shape: (n_samples, n_dists)
        u = np.random.uniform(0, 1, size=x.shape)
        x[u > A[np.newaxis, :]] = np.nan
        return x


class ScaledWeibull(Distribution):
    param_names = ["scale", "shape", "A"]

    @staticmethod
    def _check_params(params: dict[str, np.ndarray]):
        if any(k not in params for k in ["scale", "shape", "A"]):
            raise ValueError(
                "ScaledWeibull distribution requires 'scale', 'shape', and 'A' parameters."
            )
        if not np.all(params["scale"] > 0):
            raise ValueError("ScaledWeibull 'scale' parameter must be positive.")
        if not np.all(params["shape"] > 0):
            raise ValueError("ScaledWeibull 'shape' parameter must be positive.")
        if not np.all(params["A"] > 0):
            raise ValueError("ScaledWeibull 'A' parameter must be positive.")

    @staticmethod
    def cdf(x: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
        scale, shape, A = params["scale"], params["shape"], params["A"]
        return A * Weibull.cdf(x, {"scale": scale, "shape": shape})

    @staticmethod
    def pdf(
        x: np.ndarray,
        params: dict[str, np.ndarray],
        log: bool = False,
    ) -> np.ndarray:
        scale, shape, A = params["scale"], params["shape"], params["A"]
        y = Weibull.pdf(x, {"scale": scale, "shape": shape}, log=log)
        return np.log(A + EPS) + y if log else A * y

    @staticmethod
    def median(params: dict[str, np.ndarray]) -> np.ndarray:
        ScaledWeibull._check_params(params)
        scale, shape, A = params["scale"], params["shape"], params["A"]
        return A * Weibull.median({"scale": scale, "shape": shape})

    @staticmethod
    def sample(n_samples: int, params: dict[str, np.ndarray]) -> np.ndarray:
        ScaledWeibull._check_params(params)
        scale, shape, A = params["scale"], params["shape"], params["A"]
        x = Weibull.sample(n_samples, {"scale": scale, "shape": shape})
        u = np.random.uniform(0, 1, size=x.shape)
        x[u > A[np.newaxis, :]] = np.nan
        return x


if __name__ == "__main__":
    params = {
        "scale": np.array([1.0, 1.0, 1.0, 1.0, 0.9]),
        "shape": np.array([0.5, 1.0, 1.5, 5.0, 1.0]),
    }
    x_range = np.linspace(0, 5, 101)

    print({k: v.shape for k, v in params.items()})

    alt.renderers.enable("browser")
    alt.data_transformers.enable("vegafusion")
    # Weibull.plot(x_range, params).show()

    x = ScaledWeibull.sample(100000, params | {"A": np.array([0.2, 0.4, 0.6, 0.8, 1.0])})
    x_df = pl.DataFrame({"x": x[:, 2]}).fill_nan(None)
    alt.Chart(x_df).mark_bar().encode(
        alt.X("x", bin=alt.Bin(maxbins=50)),
        alt.Y("count()"),
    ).show()
