# simulate.py
from __future__ import annotations

import argparse
import json
import os
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

from model import GammaPoissonCoupledModel, GammaPoissonConfig
from eval import (
    evaluate_fit_Y,
    global_consequence_baseline,
    shared_consequence_baseline,
    nnls_factorization_baseline,
    _rmse_np,  # internal but handy
)


# ---------------------- labeled simulation ---------------------------

def simulate_coupled_data(
    n_samples: int,
    C_ctx_sig_df: pd.DataFrame,
    consequence_labels: List[str],
    random_state: Optional[int] = None,
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray
]:
    """
    Simulate X, Y from a synthetic coupled model using *real* C (contexts × signatures).

    Parameters
    ----------
    n_samples : int
    C_ctx_sig_df : DataFrame (contexts × signatures)
    consequence_labels : list of consequence type names
    random_state : int or None

    Returns
    -------
    X_df : samples × contexts
    Y_df : samples × consequences
    theta_true_df : samples × signatures
    A_true_df : signatures × consequences
    s : np.ndarray (n,) total SBS counts
    u : np.ndarray (n,) total consequence counts
    """
    rng = np.random.default_rng(random_state)

    contexts = list(C_ctx_sig_df.index)
    sigs = list(C_ctx_sig_df.columns)
    M = len(contexts)
    K = len(sigs)
    L = len(consequence_labels)

    sample_ids = [f"sample_{i}" for i in range(n_samples)]

    # True latent parameters
    theta_true = rng.gamma(shape=2.0, scale=1.0, size=(n_samples, K))
    A_true = rng.gamma(shape=2.0, scale=1.0, size=(K, L))
    A_true /= A_true.sum(axis=1, keepdims=True)

    # Burdens
    s = rng.integers(low=100, high=1000, size=n_samples).astype(float)
    u = rng.integers(low=100, high=1000, size=n_samples).astype(float)

    # C as signatures × contexts
    C_sig_ctx = C_ctx_sig_df.T.values.astype(float)  # K×M

    # SBS rates
    rate_X = (theta_true @ C_sig_ctx)   # n×M
    rate_X = s[:, None] * rate_X
    X = rng.poisson(rate_X)

    # consequence rates
    rate_Y = (theta_true @ A_true)      # n×L
    rate_Y = u[:, None] * rate_Y
    Y = rng.poisson(rate_Y)

    # Wrap into DataFrames
    X_df = pd.DataFrame(X, index=sample_ids, columns=contexts)
    Y_df = pd.DataFrame(Y, index=sample_ids, columns=consequence_labels)
    theta_true_df = pd.DataFrame(theta_true, index=sample_ids, columns=sigs)
    A_true_df = pd.DataFrame(A_true, index=sigs, columns=consequence_labels)

    return X_df, Y_df, theta_true_df, A_true_df, s, u


# ------------------------- CSV loaders -------------------------------

def load_X_csv(path: str) -> pd.DataFrame:
    """
    Load X from CSV.

    Expects:
      - first column: sample IDs (index)
      - header row: context / mutation-type labels
    """
    df = pd.read_csv(path, header=0, index_col=0)
    return df


def load_Y_csv(path: str) -> pd.DataFrame:
    """
    Load Y from CSV.

    Expects:
      - first column: sample IDs (index)
      - header row: consequence labels
    """
    df = pd.read_csv(path, header=0, index_col=0)
    return df


def load_C_txt(path: str) -> pd.DataFrame:
    """
    Load C (contexts × signatures) from txt.

    Expects:
      - first column: context labels (index)
      - header row: signature labels
    """
    df = pd.read_csv(path, header=0, sep='\s+', index_col=0)
    return df


# ------------------------------ CLI ---------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fit coupled Gamma–Poisson model to labeled X, Y, C CSVs."
    )
    parser.add_argument("--X_csv", required=True,
                        help="Path to X CSV (samples × contexts). First column = sample IDs.")
    parser.add_argument("--Y_csv", required=True,
                        help="Path to Y CSV (samples × consequences). First column = sample IDs.")
    parser.add_argument("--C_csv", required=True,
                        help="Path to C CSV (contexts × signatures). First column = context labels.")

    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--b", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_verbose", action="store_true")

    parser.add_argument("--outdir", type=str, default=None,
                        help="If set, save theta, A, lambda, and metrics here (CSV + JSON).")

    args = parser.parse_args()

    # Load labeled CSVs
    X_df = load_X_csv(args.X_csv)
    Y_df = load_Y_csv(args.Y_csv)
    C_df = load_C_txt(args.C_csv)

    print(f"Loaded X: {X_df.shape}, Y: {Y_df.shape}, C: {C_df.shape}")
    print(f"X samples: {len(X_df.index)}, contexts: {len(X_df.columns)}")
    print(f"Y consequences: {len(Y_df.columns)}")
    print(f"C contexts: {len(C_df.index)}, signatures: {len(C_df.columns)}")

    # Create config and model
    K = C_df.shape[1]
    cfg = GammaPoissonConfig(
        max_iter=args.max_iter,
        tol=args.tol,
        a=args.a,
        b=args.b,
        beta=args.beta,
        gamma=args.gamma,
        random_state=args.seed,
        verbose=not args.no_verbose,
    )
    model = GammaPoissonCoupledModel(K=K, config=cfg)

    # Fit
    model.fit(X_df, Y_df, C_df)

    # Evaluate
    metrics = evaluate_fit_Y(model)
    print("\n--- Coupled model ---")
    print(f"Global deviance: {metrics['global_deviance']:.3f}")
    print(f"RMSE: {metrics['rmse']:.4f}")

    # Baselines
    print("\n--- Baselines ---")
    A_bar, lam_global_df = global_consequence_baseline(Y_df)
    rmse_global = _rmse_np(Y_df.values, lam_global_df.values)
    print(f"Global baseline RMSE: {rmse_global:.4f}")

    A_shared, lam_shared_df = shared_consequence_baseline(model, Y_df)
    rmse_shared = _rmse_np(Y_df.values, lam_shared_df.values)
    print(f"Shared-A baseline RMSE: {rmse_shared:.4f}")

    try:
        A_nnls_df, lam_nnls_df = nnls_factorization_baseline(model, Y_df)
        rmse_nnls = _rmse_np(Y_df.values, lam_nnls_df.values)
        print(f"NNLS baseline RMSE: {rmse_nnls:.4f}")
        nnls_available = True
    except RuntimeError as e:
        print(f"NNLS baseline not available: {e}")
        A_nnls_df = None
        rmse_nnls = None
        nnls_available = False

    # Save outputs
    if args.outdir is not None:
        os.makedirs(args.outdir, exist_ok=True)

        # theta, A (means)
        model.theta_df.to_csv(os.path.join(args.outdir, "theta_mean.csv"))
        model.A_df.to_csv(os.path.join(args.outdir, "A_mean.csv"))
        model.A_row_normalized_df.to_csv(
            os.path.join(args.outdir, "A_row_normalized.csv")
        )

        # lambda estimates
        model.expected_lambda_Y_df().to_csv(
            os.path.join(args.outdir, "lambda_Y.csv")
        )
        model.expected_lambda_X_df().to_csv(
            os.path.join(args.outdir, "lambda_X.csv")
        )

        # Baselines
        A_bar.to_csv(os.path.join(args.outdir, "A_global_baseline.csv"))
        lam_global_df.to_csv(os.path.join(args.outdir, "lambda_Y_global_baseline.csv"))

        A_shared.to_csv(os.path.join(args.outdir, "A_shared_baseline.csv"))
        lam_shared_df.to_csv(
            os.path.join(args.outdir, "lambda_Y_shared_baseline.csv")
        )

        if nnls_available and A_nnls_df is not None:
            A_nnls_df.to_csv(os.path.join(args.outdir, "A_nnls_baseline.csv"))
            lam_nnls_df.to_csv(
                os.path.join(args.outdir, "lambda_Y_nnls_baseline.csv")
            )

        # Metrics JSON
        metrics_out = {
            "global_deviance": float(metrics["global_deviance"]),
            "rmse_coupled": float(metrics["rmse"]),
            "rmse_global_baseline": float(rmse_global),
            "rmse_shared_baseline": float(rmse_shared),
            "nnls_available": nnls_available,
        }
        if nnls_available and rmse_nnls is not None:
            metrics_out["rmse_nnls_baseline"] = float(rmse_nnls)

        with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
            json.dump(metrics_out, f, indent=2)

        print(f"\nSaved outputs to: {args.outdir}")


if __name__ == "__main__":
    main()
