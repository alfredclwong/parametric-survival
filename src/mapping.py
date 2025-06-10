from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import polars as pl


class ParamMapping(ABC):
    n_features: int
    param_names: list[str]
    weights: np.ndarray
    transforms: list[Callable]

    def __init__(self, n_features: int, param_transforms: dict[str, Callable]):
        self.n_features = n_features
        self.param_names = list(param_transforms.keys())
        self.transforms = list(param_transforms.values())

    @abstractmethod
    def init_weights(self):
        raise NotImplementedError

    @abstractmethod
    def map(self, X: np.ndarray) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def get_weights(self, flatten: bool = True) -> np.ndarray:
        return self.weights.flatten() if flatten else self.weights

    def get_weights_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            {name: self.weights[:, i] for i, name in enumerate(self.param_names)}
        )

    def set_weights(self, weights: np.ndarray):
        self.weights = weights.reshape(self.weights.shape)


class LinearParamMapping(ParamMapping):
    def __init__(
        self, n_features: int, param_transforms: dict[str, Callable], bias: bool = True
    ):
        super().__init__(n_features, param_transforms)
        self.bias = bias
        self.init_weights()

    def init_weights(self):
        n_params = len(self.param_names)
        self.weights = np.random.randn(int(self.bias) + self.n_features, n_params)

    def map(self, X: np.ndarray, noise: float = 0.0) -> dict[str, np.ndarray]:
        if self.bias:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        param_values = np.dot(X, self.weights)
        param_values += noise * np.random.randn(*param_values.shape)
        params = {
            name: f(param_values[:, i])
            for i, (name, f) in enumerate(zip(self.param_names, self.transforms))
        }
        return params


class ConstantParamMapping(ParamMapping):
    def __init__(self, param_transforms: dict[str, Callable]):
        super().__init__(n_features=0, param_transforms=param_transforms)
        self.init_weights()

    def init_weights(self):
        self.weights = np.random.randn(len(self.param_names))

    def map(self, X: np.ndarray) -> dict[str, np.ndarray]:
        # X.shape: (n_samples, n_features)
        params = {
            name: f(np.full(X.shape[0], self.weights[i]))
            for i, (name, f) in enumerate(zip(self.param_names, self.transforms))
        }
        return params
