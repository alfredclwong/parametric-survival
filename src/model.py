# %%
from typing import Optional

import numpy as np
from scipy.optimize import minimize
from tqdm.auto import tqdm

from config import EPS
from dist import Distribution
from mapping import ParamMapping


class ParametricSurvivalModel:
    """
    Model censored clinical data (X, Y, C) using a parametric distribution.

    The model maps input features X to parameters of a distribution, which are
    used to predict the probability density function (PDF) of the outcome Y|C.

    The model is fitted to the training data by finding parameters which
    maximize the likelihood of the observed data under the assumed distribution.
    """

    def __init__(
        self,
        dist_type: type[Distribution],
        param_mapping: ParamMapping,
    ):
        self.dist_type = dist_type
        self.param_mapping = param_mapping

    def median_survival_time(self, x: np.ndarray) -> np.ndarray:
        params = self.param_mapping.map(x)
        return self.dist_type.median(params)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        c: np.ndarray,
        x_test=None,
        y_test=None,
        c_test=None,
        maxiter: int = 1000,
        balance: bool = False,
        patience: Optional[int] = None,
    ):
        test = x_test is not None and y_test is not None and c_test is not None
        pbar = tqdm(desc="Fitting model", total=maxiter)

        history = []
        best_test_loss = np.inf
        best_test_loss_step = -1
        best_weights = self.param_mapping.get_weights()
        patience_counter = 0

        def callback(weights):
            step = pbar.n + 1
            train_loss = self.neg_log_likelihood(weights, x, y, c, balance=balance)
            history.append((step, "train_loss", train_loss))
            test_loss = np.nan
            if test:
                test_loss = self.neg_log_likelihood(
                    weights, x_test, y_test, c_test, balance=balance
                )
                history.append((step, "test_loss", test_loss))
                nonlocal \
                    best_test_loss, \
                    patience_counter, \
                    best_weights, \
                    best_test_loss_step
                if test_loss < best_test_loss:
                    patience_counter = 0
                    best_weights = weights
                    best_test_loss = test_loss
                    best_test_loss_step = step
                else:
                    patience_counter += 1
                if patience is not None and patience_counter >= patience:
                    pbar.write(
                        f"Early stopping at step {step} with test loss {test_loss:.4f}"
                    )
                    pbar.close()
                    raise StopIteration("Early stopping triggered")
            pbar.update(1)
            pbar.set_postfix(
                {"train_loss": f"{train_loss:.4f}", "test_loss": f"{test_loss:.4f}"}
            )

        initial_weights = self.param_mapping.get_weights()
        result = minimize(
            self.neg_log_likelihood,
            initial_weights,
            args=(x, y, c, balance),
            options={"disp": True, "maxiter": maxiter},
            callback=callback,
        )
        pbar.close()

        if not result.success:
            print("Optimization failed: " + result.message)
        if test:
            print(f"Best test loss: {best_test_loss:.4f} at step {best_test_loss_step}")
            self.param_mapping.set_weights(best_weights)
        else:
            self.param_mapping.set_weights(result.x)
        return history

    def neg_log_likelihood(self, weights, x, y, c, balance=False, tol=EPS) -> float:
        self.param_mapping.set_weights(weights)  # TODO make a new instance instead?
        likelihoods = self.likelihoods(x, y, c, log=True)
        likelihoods = np.nan_to_num(likelihoods, neginf=-1e10)
        if balance:
            mask = np.isclose(y, c, atol=tol)
            censored_likelihoods = likelihoods[mask]
            observed_likelihoods = likelihoods[~mask]
            return float(
                -0.5 * (np.mean(censored_likelihoods) + np.mean(observed_likelihoods))
            )
        return -likelihoods.mean()

    def likelihoods(
        self,
        x: np.ndarray,
        y: np.ndarray,
        c: np.ndarray,
        eps: float = EPS,
        log: bool = False,
    ) -> np.ndarray:
        """
        x.shape: (n_samples, n_features)
        y.shape: (n_samples,)
        c.shape: (n_samples,)
        """
        params = self.param_mapping.map(x)

        p_observed = self.dist_type.pdf(y, params, log=log)
        p_censored = 1 - self.dist_type.cdf(y, params)
        p_censored = np.log(p_censored + eps) if log else p_censored

        likelihoods = np.full_like(y, dtype=float, fill_value=0 if log else 0)
        y_eq_c_mask = np.isclose(y, c, atol=eps)
        y_lt_c_mask = y < c
        likelihoods[y_eq_c_mask] = p_censored[y_eq_c_mask]
        likelihoods[y_lt_c_mask] = p_observed[y_lt_c_mask]
        return likelihoods


# %%
