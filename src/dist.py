import torch as t
from torch.distributions import constraints

from config import EPS


class AsymptoticWeibull(t.distributions.Weibull):
    arg_constraints = {
        "alpha": constraints.unit_interval,
        **t.distributions.Weibull.arg_constraints,
    }

    def __init__(self, alpha: t.Tensor, scale: t.Tensor, concentration: t.Tensor):
        self.alpha = alpha
        super().__init__(scale=scale, concentration=concentration)

    def log_prob(self, value: t.Tensor) -> t.Tensor:
        return super().log_prob(value) + t.log(self.alpha + EPS)

    def cdf(self, value: t.Tensor) -> t.Tensor:
        return super().cdf(value) * self.alpha

    def icdf(self, value: t.Tensor) -> t.Tensor:
        _icdf = super().icdf(value / self.alpha)
        if _icdf is None:
            raise NotImplementedError("Parent class does not implement icdf method.")
        mask = value > self.alpha
        _icdf[mask] = t.inf
        return _icdf

    def sample(self, sample_shape=None) -> t.Tensor:
        samples = super().sample(sample_shape or t.Size())
        if not isinstance(samples, t.Tensor):
            raise TypeError("Expected samples to be a torch.Tensor")
        mask = t.rand_like(samples) > self.alpha
        samples[mask] = t.inf
        return samples
