from dataclasses import dataclass
from typing import Optional

import polars as pl
import torch as t
from tqdm.auto import tqdm

from config import EPS, INF
from mapping import ParamMapping, ParamMappingConfig


@dataclass(frozen=True)
class TrainConfig:
    n_epochs: int
    learning_rate: float
    weight_decay: float
    balance: bool
    batch_size: Optional[int] = None


class ParametricSurvivalModel(t.nn.Module):
    def __init__(
        self,
        dist_type: type[t.distributions.Distribution],
        mapping_cfg: ParamMappingConfig,
        device: str,
    ):
        super().__init__()
        self.dist_type = dist_type
        self.mapping = ParamMapping(mapping_cfg).to(device)
        self.device = device

    def get_dist(self, x: t.Tensor) -> t.distributions.Distribution:
        params = self.mapping(x)
        return self.dist_type(**params)

    def log_likelihood(self, x: t.Tensor, y: t.Tensor, d: t.Tensor) -> t.Tensor:
        """
        Compute the log-likelihood of the observed data under the model.
        Args:
            x (t.Tensor): Input features of shape (batch_size, d_in).
            y (t.Tensor): Observed outcomes of shape (batch_size,).
            d (t.Tensor): Diagnosis indicators of shape (batch_size,).
        Returns:
            t.Tensor: Log-likelihood of the observed data.
        """
        x = x.to(self.device, dtype=t.float32)
        y = y.to(self.device, dtype=t.float32)
        d = d.to(self.device, dtype=t.bool)
        dist = self.get_dist(x)
        p_event = dist.log_prob(y)
        p_censor = (1 - dist.cdf(y)).clip(min=EPS).log()
        # ll = t.where(d, p_event, p_censor)
        ll = d * p_event + (~d) * p_censor
        return ll

    def loss(
        self,
        x: t.Tensor,
        y: t.Tensor,
        d: t.Tensor,
        balance: bool,
    ) -> t.Tensor:
        ll = self.log_likelihood(x, y, d)
        if balance:
            loss_pos = ll[d == 1].nanmean()
            loss_neg = ll[d == 0].nanmean()
            return -(loss_pos + loss_neg) / 2
        else:
            return -ll.nanmean()

    def fit(
        self,
        x: t.Tensor,
        y: t.Tensor,
        d: t.Tensor,
        x_val: t.Tensor,
        y_val: t.Tensor,
        d_val: t.Tensor,
        cfg: TrainConfig,
    ) -> dict[str, list[float]]:
        """
        Fit the model using k-fold cross-validation.
        Args:
            x (t.Tensor): Input features of shape (n_samples, d_in).
            y (t.Tensor): Observed outcomes of shape (n_samples,).
            d (t.Tensor): Diagnosis indicators of shape (n_samples,).
            k_folds (int): Number of folds for cross-validation.
            n_epochs (int): Number of epochs for training.
        Returns:
            t.Tensor: Log-likelihood of the observed data for each fold.
        """
        x = x.to(self.device, dtype=t.float32)
        y = y.to(self.device, dtype=t.float32)
        d = d.to(self.device, dtype=t.bool)
        x_val = x_val.to(self.device, dtype=t.float32)
        y_val = y_val.to(self.device, dtype=t.float32)
        d_val = d_val.to(self.device, dtype=t.bool)

        self.mapping.init_weights()
        optimizer = t.optim.Adam(
            self.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        # optimizer = t.optim.SGD(
        #     self.parameters(),
        #     lr=cfg.learning_rate,
        #     weight_decay=cfg.weight_decay,
        #     momentum=0.9,
        # )

        history = {"train": [], "val": []}
        best = {"step": -1, "val_loss": INF, "state_dict": self.state_dict()}
        for epoch in (pbar := tqdm(range(cfg.n_epochs))):
            self.train()
            batch_size = len(x) if cfg.batch_size is None else cfg.batch_size
            loss = t.tensor(t.nan, device=self.device)
            for i in range(0, len(x), batch_size):
                x_batch = x[i : i + batch_size]
                y_batch = y[i : i + batch_size]
                d_batch = d[i : i + batch_size]
                loss = self.loss(x_batch, y_batch, d_batch, cfg.balance)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            self.eval()
            with t.no_grad():
                val_loss = self.loss(x_val, y_val, d_val, cfg.balance)
            if val_loss < best["val_loss"]:
                best = {
                    "step": epoch,
                    "val_loss": val_loss.item(),
                    "state_dict": self.state_dict(),
                }

            history["train"].append(loss.item())
            history["val"].append(val_loss.item())
            pbar.set_description(
                f"Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}"
            )

        print("Training complete.")
        print(f"Best validation loss: {best['val_loss']:.4f} at step {best['step']}")
        self.load_state_dict(best["state_dict"])

        return history

    def predict(
        self, x: t.Tensor, y: t.Tensor, c: t.Tensor, d: t.Tensor
    ) -> pl.DataFrame:
        x = x.to(self.device, dtype=t.float32)
        y = y.to(self.device, dtype=t.float32)
        c = c.to(self.device, dtype=t.float32)
        d = d.to(self.device, dtype=t.bool)

        with t.no_grad():
            params = self.mapping(x)
            df = pl.DataFrame(
                {
                    "Y": y.cpu(),
                    "C": c.cpu(),
                    "D": d.cpu(),
                    **{k: v.cpu() for k, v in params.items()},
                    "logL": self.log_likelihood(x, y, d).cpu(),
                    "D_pred": self.dist_type(**params).cdf(c).cpu(),
                    "T_pred": self.median_survival_time(x).cpu(),
                }
            ).with_columns(
                Y_pred=pl.min_horizontal(["T_pred", "C"]),
            )
        return df

    def median_survival_time(self, x: t.Tensor) -> t.Tensor:
        dist = self.get_dist(x)
        median = dist.icdf(t.tensor(0.5))
        return median
