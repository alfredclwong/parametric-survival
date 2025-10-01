# Technical Documentation

## dist.py
- Defines AsymptoticWeibull, which inherits from torch.distributions.Weibull
- $$F(t) = \alpha\left(1 - e^{-(t/\lambda)^k}\right)$$

## mapping.py
- Defines ParamMapping, which inherits from torch.nn.Module
- This takes as input a feature matrix X (n_samples, n_features)
- X is passed through an MLP and then some specified param_transforms, e.g. sigmoids
- Params are returned as a dict where each key is a vector (n_samples)

## model.py
- Combines a torch.distributions.Distribution and ParamMapping to create a ParametricSurvivalModel
- Implements a censoring-aware log-likelihood function
- Implements a class imbalance-aware loss function
- Implements a fit method with early stopping conditioned on val_loss

## synth.py
- Synthetic data generation

## train.py
- Train on a dataset

## eval.py
- A bunch of evals, notably t-auc and c-index

## confusion.py
- Experiment to investigate param recovery performance

## demo.py
- Demo of dist.py, mapping.py, model.py, synth.py, eval.py
