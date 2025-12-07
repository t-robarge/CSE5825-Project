# eval.py
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from model import GammaPoissonCoupledModel

try:
    from scipy.optimize import nnls
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ------------------- low-level NumPy metrics -------------------------

def _deviance_poisson_np(
    Y: np.ndarray,
    lam: np.ndarray,
    axis: int | None,
) -> np.ndarray:
    """
    Poisson deviance: D = 2 * sum [ y * log(y/λ) - (y - λ) ].

    If axis is None, returns scalar; else sums along that axis.
    """
    Y = np.asarray(Y, dtype=float)
    lam = np.asarray(lam, dtype=float)
    lam = np.where(lam <= 0.0, 1e-12, lam)

    mask = Y > 0
    term1 = np.zeros_like(Y)
    term1[mask] = Y[mask] * np.log(Y[mask] / lam[mask])

    dev = 2.0 * (term1 - (Y - lam))
    if axis is None:
        return dev.sum()
    else:
        return dev.sum(axis=axis)

def deviance_per_cell_Y(
    Y_df: pd.DataFrame,
    lam_df: pd.DataFrame,
) -> float:
    """
    Deviance per cell:

      D = Poisson deviance(Y, λ)
      D_per_cell = D / (n * L)

    where n = #samples, L = #consequences.
    """
    Y = Y_df.values.astype(float)
    lam = lam_df.values.astype(float)
    D = _deviance_poisson_np(Y, lam, axis=None)
    n, L = Y.shape
    return float(D / (n * L))

def _rmse_np(Y: np.ndarray, lam: np.ndarray) -> float:
    """Root mean squared error between Y and λ."""
    diff2 = (Y - lam) ** 2
    return float(np.sqrt(diff2.mean()))

def per_sample_rmse_Y(
    Y_df: pd.DataFrame,
    lam_df: pd.DataFrame,
) -> pd.Series:
    """
    Per-sample RMSE for Y:

      RMSE_i = sqrt( mean_l (Y_{i,l} - λ_{i,l})^2 )

    Returns
    -------
    Series indexed by sample_id.
    """
    Y = Y_df.values.astype(float)
    lam = lam_df.values.astype(float)
    diff2 = (Y - lam) ** 2
    rmse_per = np.sqrt(diff2.mean(axis=1))  # (n,)
    return pd.Series(rmse_per, index=Y_df.index, name="rmse_per_sample")

def relative_error_Y(
    Y_df: pd.DataFrame,
    lam_df: pd.DataFrame,
) -> Dict[str, object]:
    """
    Relative error metrics for Y:

      rel_err_{i,l} = |Y_{i,l} - λ_{i,l}| / (1 + Y_{i,l})

    Returns
    -------
    dict with:
      - "rel_err_df": DataFrame (samples × consequences)
      - "mean_rel_err": float (global mean over all cells)
      - "mean_rel_err_per_sample": Series (mean over consequences for each sample)
    """
    Y = Y_df.values.astype(float)
    lam = lam_df.values.astype(float)

    rel = np.abs(Y - lam) / (1.0 + Y)
    rel_df = pd.DataFrame(rel, index=Y_df.index, columns=Y_df.columns)

    mean_rel = float(rel.mean())
    mean_rel_per_sample = pd.Series(
        rel.mean(axis=1),
        index=Y_df.index,
        name="mean_rel_err_per_sample",
    )

    return {
        "rel_err_df": rel_df,
        "mean_rel_err": mean_rel,
        "mean_rel_err_per_sample": mean_rel_per_sample,
    }

# -------------------- model-based evaluation (Y) ---------------------

def evaluate_fit_Y(
    model: GammaPoissonCoupledModel,
    Y_df: pd.DataFrame | None = None,
) -> Dict[str, object]:
    """
    Evaluat2e fit for consequence counts Y using a fitted model.

    Parameters
    ----------
    model : GammaPoissonCoupledModel
    Y_df : DataFrame (samples × consequences), optional.
        If None, uses model.Y_df.

    Returns
    -------
    dict with:
      - "lambda_Y_df": DataFrame (samples × consequences)
      - "per_sample_deviance": Series indexed by sample_id
      - "global_deviance": float
      - "rmse": float
    """
    if Y_df is None:
        if model.Y_df is None:
            raise RuntimeError("Model has no Y_df stored; pass Y_df explicitly.")
        Y_df = model.Y_df

    lam_df = model.expected_lambda_Y_df()
    Y = Y_df.values
    lam = lam_df.values

    per_sample_dev = _deviance_poisson_np(Y, lam, axis=1)
    global_dev = float(per_sample_dev.sum())
    err = _rmse_np(Y, lam)

    per_sample_dev_series = pd.Series(per_sample_dev, index=Y_df.index)
    rmse_per_sample = per_sample_rmse_Y(Y_df, lam_df)
    dev_per_cell = deviance_per_cell_Y(Y_df, lam_df)
    rel_err_stats = relative_error_Y(Y_df, lam_df)

    return {
        "lambda_Y_df": lam_df,
        "per_sample_deviance": per_sample_dev_series,
        "global_deviance": global_dev,
        "deviance_per_cell": dev_per_cell,
        "rmse": err,
        "rmse_per_sample": rmse_per_sample,
        "mean_rel_err": rel_err_stats["mean_rel_err"],
        "mean_rel_err_per_sample": rel_err_stats["mean_rel_err_per_sample"],
    }




def predictive_loglik_Y(
    model: GammaPoissonCoupledModel,
    Y_df: pd.DataFrame | None,
) -> float:
    """
    Approximate predictive log-likelihood for Y under Poisson with λ_hat.

    Returns sum_{i,l} [ y_il * log λ_il - λ_il ] (omitting log(y!) constants).
    """
    if Y_df is None:
        if model.Y_df is None:
            raise RuntimeError("Model has no Y_df stored; pass Y_df explicitly.")
        Y_df = model.Y_df

    lam_df = model.expected_lambda_Y_df()
    Y = Y_df.values.astype(float)
    lam = lam_df.values.astype(float)
    lam = np.where(lam <= 0.0, 1e-12, lam)

    mask = Y > 0
    term = np.zeros_like(Y)
    term[mask] = Y[mask] * np.log(lam[mask])
    ll = term - lam
    return float(ll.sum())


# ------------------- posterior predictive (Y) ------------------------

def posterior_predictive_Y(
    model: GammaPoissonCoupledModel,
    num_draws: int = 1,
    random_state: int | None = None,
) -> Tuple[np.ndarray, pd.Index, pd.Index]:
    """
    Draw posterior predictive replicated consequence counts:

      Y_tilde(i,ℓ) ~ Poisson(λ_hat(i,ℓ))

    Returns
    -------
    Y_rep : np.ndarray, shape (num_draws, n, L)
    sample_ids : Index
    consequence_labels : Index
    """
    lam_df = model.expected_lambda_Y_df()
    lam = lam_df.values
    rng = np.random.default_rng(random_state)

    n, L = lam.shape
    Y_rep = np.zeros((num_draws, n, L), dtype=int)
    for d in range(num_draws):
        Y_rep[d] = rng.poisson(lam)

    return Y_rep, lam_df.index, lam_df.columns


# --------------------------- baselines -------------------------------

def global_consequence_baseline(
    Y_df: pd.DataFrame,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Global consequence baseline (no signatures).

    A_bar(ℓ) = sum_i Y(i,ℓ) / sum_{i,ℓ} Y(i,ℓ)
    λ_hat(i,ℓ) = u_i * A_bar(ℓ),  where u_i = sum_ℓ Y(i,ℓ).

    Returns
    -------
    A_bar : Series (consequences)
    lambda_Y_df : DataFrame (samples × consequences)
    """
    Y = Y_df.values.astype(float)
    n, L = Y.shape
    u = Y.sum(axis=1)
    total = Y.sum()
    if total == 0:
        A_bar_vals = np.ones(L) / L
    else:
        A_bar_vals = Y.sum(axis=0) / total

    A_bar = pd.Series(A_bar_vals, index=Y_df.columns, name="A_global")

    lam = u[:, None] * A_bar_vals[None, :]
    lambda_Y_df = pd.DataFrame(lam, index=Y_df.index, columns=Y_df.columns)
    return A_bar, lambda_Y_df


def shared_consequence_baseline(
    model: GammaPoissonCoupledModel,
    Y_df: pd.DataFrame | None = None,
    theta_df: pd.DataFrame | None = None,
    u: np.ndarray | None = None,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Shared consequence mixture baseline:

      Y(i,ℓ) ~ Poisson( u_i * sum_k theta(i,k) * A_shared(ℓ) )
      r_i = u_i * sum_k theta(i,k)
      A_shared(ℓ) = sum_i Y(i,ℓ) / sum_i r_i

    Returns
    -------
    A_shared : Series (consequences)
    lambda_Y_df : DataFrame (samples × consequences)
    """
    if Y_df is None:
        if model.Y_df is None:
            raise RuntimeError("Model has no Y_df stored; pass Y_df explicitly.")
        Y_df = model.Y_df

    if theta_df is None:
        if model.theta_df is None:
            raise RuntimeError("Model not fitted; theta undefined.")
        theta_df = model.theta_df

    if u is None:
        if model.u is None:
            raise RuntimeError("Model not fitted; u undefined.")
        u = model.u

    Y = Y_df.values.astype(float)
    theta_mean = theta_df  # n×K
    r = u * theta_mean.values.sum(axis=1)  # (n,)

    denom = r.sum()
    n, L = Y.shape
    if denom == 0:
        A_shared_vals = np.ones(L) / L
    else:
        numer = Y.sum(axis=0)  # (L,)
        A_shared_vals = numer / denom

    A_shared = pd.Series(A_shared_vals, index=Y_df.columns, name="A_shared")

    lam = r[:, None] * A_shared_vals[None, :]
    lambda_Y_df = pd.DataFrame(lam, index=Y_df.index, columns=Y_df.columns)
    return A_shared, lambda_Y_df


def nnls_factorization_baseline(
    model: GammaPoissonCoupledModel,
    Y_df: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Non-Bayesian NNLS factorization of Y given fixed theta:

      minimize  sum_{i,l} (Y_{i l} - u_i (θ A)_{i l})^2
      subject to A_{kl} ≥ 0, sum_l A_{kl} = 1.

    Implemented column-wise with NNLS, then row-normalized.
    """
    if not _HAS_SCIPY:
        raise RuntimeError("SciPy not available; install scipy to use NNLS baseline.")
    if model.u is None:
        raise RuntimeError("Model not fitted; u undefined.")

    if Y_df is None:
        if model.Y_df is None:
            raise RuntimeError("Model has no Y_df stored; pass Y_df explicitly.")
        Y_df = model.Y_df

    Y = Y_df.values.astype(float)           # n × L
    theta_mean = model.theta_mean           # n × K
    u = model.u                             # n

    # Design matrix for LS: (D_u Θ)
    Theta_tilde = theta_mean * u[:, None]   # n × K

    n, L = Y.shape
    K = theta_mean.shape[1]
    A = np.zeros((K, L))

    for l in range(L):
        # Solve: min ||Theta_tilde a_l - Y[:, l]||_2^2  s.t. a_l >= 0
        a_l, _ = nnls(Theta_tilde, Y[:, l])
        A[:, l] = a_l

    # Enforce row-sum=1 (approximation to the constrained LS)
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
    A_norm = A / row_sums

    A_nnls_df = pd.DataFrame(
        A_norm,
        index=model.signature_labels,
        columns=Y_df.columns,
    )

    lam = u[:, None] * (theta_mean @ A_norm)
    lambda_Y_df = pd.DataFrame(lam, index=Y_df.index, columns=Y_df.columns)

    return A_nnls_df, lambda_Y_df
