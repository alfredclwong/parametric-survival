from dataclasses import dataclass
from typing import Callable, Optional

import torch as t
import polars as pl


@dataclass(frozen=True)
class ParamMappingConfig:
    d_in: int
    d_hidden: list[int]
    param_transforms: dict[str, Callable]
    dropout: Optional[float] = None
    activation: Callable = t.nn.ReLU


class ParamMapping(t.nn.Module):
    def __init__(self, cfg: ParamMappingConfig):
        super().__init__()
        d_out = len(cfg.param_transforms)
        dims: list[int] = [cfg.d_in, *cfg.d_hidden, d_out]
        layers = []
        for i in range(len(dims) - 1):
            layer = t.nn.Linear(dims[i], dims[i + 1], bias=False)
            layers.append(layer)
            if i < len(dims) - 2:
                layers.append(cfg.activation())
        self.mlp = t.nn.Sequential(*layers)
        self.param_transforms = cfg.param_transforms
        self.init_weights()

    def init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, t.nn.Linear):
                t.nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    t.nn.init.zeros_(m.bias)

    def forward(self, x: t.Tensor) -> dict[str, t.Tensor]:
        x = self.mlp(x)
        params = {
            name: transform(x[:, i])
            for i, (name, transform) in enumerate(self.param_transforms.items())
        }
        return params

    @property
    def weights_df(self) -> pl.DataFrame:
        return pl.DataFrame({
            f"{param}_{layer}": pl.Series(self.mlp[layer].weight[i].detach().cpu().numpy().flatten())
            for i, param in enumerate(self.param_transforms.keys())
            for layer in range(len(self.mlp))
            if isinstance(self.mlp[layer], t.nn.Linear)
        })

    @weights_df.setter
    def weights_df(self, df: pl.DataFrame):
        for i, param in enumerate(self.param_transforms.keys()):
            for layer in range(len(self.mlp)):
                if isinstance(self.mlp[layer], t.nn.Linear):
                    self.mlp[layer].weight.data[i] = t.tensor(
                        df[f"{param}_{layer}"].to_numpy(),
                        dtype=t.float32,
                    ).view_as(self.mlp[layer].weight[i])
