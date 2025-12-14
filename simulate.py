# simulate.py
from __future__ import annotations

import argparse
import json
import os

from typing import Tuple, List
from sklearn.model_selection import StratifiedShuffleSplit 
import numpy as np
import pandas as pd

from model import GammaPoissonCoupledModel, GammaPoissonConfig
from eval import (
    evaluate_fit_Y,
    global_consequence_baseline,
    shared_consequence_baseline,
    nnls_factorization_baseline,
    nnls_factorization_baseline_predict,
    _rmse_np,              
    _deviance_poisson_np, 
)



# ---------------------- labeled simulation ---------------------------
def infer_theta_for_new_samples(
    model: GammaPoissonCoupledModel,
    X_df: pd.DataFrame,
    Y_df: pd.DataFrame,
    max_iter: int | None = None,
    tol: float | None = None,
) -> pd.DataFrame:
    """
    Infer theta for new samples given fixed A (from a fitted model).

    Parameters
    ----------
    model : fitted GammaPoissonCoupledModel
        Must already have A_mean, C_sig_ctx, config, etc.
    X_df : DataFrame (samples x contexts)
    Y_df : DataFrame (samples x consequences)
        Must have matching sample IDs.

    Returns
    -------
    theta_new_df : DataFrame (samples x signatures)
        Posterior means E_q[theta] for the new samples.
    """
    if not X_df.index.equals(Y_df.index):
        Y_df = Y_df.reindex(index=X_df.index)
        if Y_df.isnull().any().any():
            raise ValueError("Sample IDs in X_test and Y_test do not match and cannot be aligned.")

    # Align columns to model ordering
    if list(X_df.columns) != model.context_labels:
        X_df = X_df.reindex(columns=model.context_labels)
    if list(Y_df.columns) != model.consequence_labels:
        Y_df = Y_df.reindex(columns=model.consequence_labels)

    X = X_df.values.astype(float)   # (n, M)
    Y = Y_df.values.astype(float)   # (n, L)
    n, M = X.shape
    _, L = Y.shape

    K = model.K
    C_sig_ctx = model.C_sig_ctx          # (K, M)
    A_mean = model.A_mean                # (K, L)

    s = X.sum(axis=1)                    # SBS burden
    u = Y.sum(axis=1)                    # consequence burden

    cfg = model.config
    if max_iter is None:
        max_iter = cfg.max_iter
    if tol is None:
        tol = cfg.tol

    # Broadcast hyperparameters as in model.fit
    a_h = np.broadcast_to(cfg.a, (K,))
    b_h = np.broadcast_to(cfg.b, (K,))

    C_row_sums = C_sig_ctx.sum(axis=1)   # (K,)
    A_row_sums = A_mean.sum(axis=1)      # (K,)

    rng = np.random.default_rng(cfg.random_state)
    a_tilde = np.tile(a_h, (n, 1)) + rng.gamma(
        shape=1.0, scale=0.1, size=(n, K)
    )
    b_tilde = np.tile(b_h, (n, 1)) + 1.0

    prev_theta = a_tilde / b_tilde

    for it in range(max_iter):
        theta_mean = a_tilde / b_tilde   # (n, K)

        # r_X for test
        thetaC = theta_mean[:, :, None] * C_sig_ctx[None, :, :]  # n×K×M
        denom_X = thetaC.sum(axis=1, keepdims=True)              # n×1×M
        denom_X = np.where(denom_X == 0.0, 1.0, denom_X)
        r_X = thetaC / denom_X                                   # n×K×M
        X_hat_sum_m = (X[:, None, :] * r_X).sum(axis=2)          # n×K

        # r_Y for test
        thetaA = theta_mean[:, :, None] * A_mean[None, :, :]     # n×K×L
        denom_Y = thetaA.sum(axis=1, keepdims=True)              # n×1×L
        denom_Y = np.where(denom_Y == 0.0, 1.0, denom_Y)
        r_Y = thetaA / denom_Y                                   # n×K×L
        Y_hat = Y[:, None, :] * r_Y                              # n×K×L
        Y_hat_sum_l = Y_hat.sum(axis=2)                          # n×K

        # Gamma updates for theta (A fixed)
        a_tilde = a_h[None, :] + X_hat_sum_m + Y_hat_sum_l
        b_tilde = (
            b_h[None, :]
            + s[:, None] * C_row_sums[None, :]
            + u[:, None] * A_row_sums[None, :]
        )

        rel_change_theta = np.max(
            np.abs(theta_mean - prev_theta) / (1.0 + np.abs(prev_theta))
        )
        if rel_change_theta < tol:
            break
        prev_theta = theta_mean

    theta_mean = a_tilde / b_tilde
    theta_new_df = pd.DataFrame(
        theta_mean,
        index=X_df.index,
        columns=model.signature_labels,
    )
    test_u = Y.sum(axis=1)
    return theta_new_df, test_u

def simulate_coupled_data(
    n_samples: int,
    C_ctx_sig_df: pd.DataFrame,
    consequence_labels: List[str],
    random_state: int | None = None,
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray
]:
    """
    Simulate X, Y from a synthetic coupled model using C (contexts x signatures).

    Parameters
    ----------
    n_samples : int
    C_ctx_sig_df : DataFrame (contexts x signatures)
    consequence_labels : list of consequence type names
    random_state : int or None

    Returns
    -------
    X_df : samples x contexts
    Y_df : samples x consequences
    theta_true_df : samples x signatures
    A_true_df : signatures x consequences
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
    Load C (contexts x signatures) from txt.

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
                        help="Path to X CSV (samples x contexts). First column = sample IDs.")
    parser.add_argument("--Y_csv", required=True,
                        help="Path to Y CSV (samples x consequences). First column = sample IDs.")
    parser.add_argument("--C_csv", required=True,
                        help="Path to C CSV (contexts x signatures). First column = context labels.")
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--b", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_verbose", action="store_true")

    parser.add_argument("--outdir", type=str, default=None,
                        help="If set, save theta, A, lambda, and metrics here (CSV + JSON).")
    parser.add_argument(
        "--test_frac",
        type=float,
        default=0.0,
        help="Fraction of samples to hold out as a test set (0 = no split).",
    )

    args = parser.parse_args()

    # Load labeled CSVs
    X_df = load_X_csv(args.X_csv)
    Y_df = load_Y_csv(args.Y_csv)
    C_df = load_C_txt(args.C_csv)
    if args.X_csv == 'sbs96_by_tumor_with_cancer_type.csv':
        # keep only OV in X
        X_df = X_df[X_df['Cancer_Type'] == 'BRCA'].copy()

        # and subset Y to the same samples
        Y_df = Y_df.loc[X_df.index]
        # drop Cancer_Type column
        X_df = X_df.drop(columns=['Cancer_Type'])
    print(f"Loaded X: {X_df.shape}, Y: {Y_df.shape}, C: {C_df.shape}")
    print(f"X samples: {len(X_df.index)}, contexts: {len(X_df.columns)}")
    print(f"Y consequences: {len(Y_df.columns)}")
    print(f"C contexts: {len(C_df.index)}, signatures: {len(C_df.columns)}")

    if args.test_frac > 0.0:
        # total consequence burden per tumor
        burden = Y_df.sum(axis=1)  # pandas Series, index = sample IDs
        n_total = len(burden)

        # number of bins for stratification (at most 5, but not more than n_total)
        n_bins = min(5, n_total)

        strat_labels = None
        if n_bins >= 2:
            try:
                # quantile-based bins: 0, 1, ..., n_bins-1
                strat_labels = pd.qcut(
                    burden,
                    q=n_bins,
                    labels=False,
                    duplicates="drop"
                )
            except ValueError:
                strat_labels = None

        if strat_labels is not None:
            print(
                f"\nUsing stratified train/test split by total consequence burden "
                f"into {len(np.unique(strat_labels.dropna()))} strata."
            )
            sss = StratifiedShuffleSplit(
                n_splits=1,
                test_size=args.test_frac,
                random_state=args.seed,
            )
            all_idx = np.arange(n_total)
            (train_idx, test_idx), = sss.split(all_idx, strat_labels.values)

            all_ids = np.array(X_df.index)
            train_ids = all_ids[train_idx]
            test_ids = all_ids[test_idx]
        
        X_train = X_df.loc[train_ids]
        Y_train = Y_df.loc[train_ids]
        X_test = X_df.loc[test_ids]
        Y_test = Y_df.loc[test_ids]
    else:
        X_train, Y_train = X_df, Y_df
        X_test, Y_test = None, None
        train_ids = X_df.index
        test_ids = None
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
    model.fit(X_train, Y_train, C_df)

    # Evaluate
    metrics = evaluate_fit_Y(model)
    print("\n--- Coupled model ---")
    print(f"Global deviance: {metrics['global_deviance']:.3f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    if X_test is not None and Y_test is not None:
        theta_test_df, u_test = infer_theta_for_new_samples(model, X_test, Y_test)
        A_mean = model.A_df.values.astype(float)
        u_test = Y_test.sum(axis=1).values.astype(float)

        lam_test = u_test[:, None] * (theta_test_df.values @ A_mean)
        dev_test = _deviance_poisson_np(Y_test.values, lam_test, axis=None)
        rmse_test = _rmse_np(Y_test.values, lam_test)

        print("\n--- Coupled model (test set, held-out samples) ---")
        print(f"Test global deviance: {dev_test:.3f}")
        print(f"Test RMSE: {rmse_test:.4f}")

    # Baselines
    print("\n--- Baselines ---")
    A_bar, lam_global_df = global_consequence_baseline(Y_train)
    rmse_global = _rmse_np(Y_train.values, lam_global_df.values)
    print(f"Global baseline RMSE: {rmse_global:.4f}")


    try:
        A_nnls_df, lam_nnls_df = nnls_factorization_baseline(model, Y_train)
        rmse_nnls = _rmse_np(Y_train.values, lam_nnls_df.values)
        print(f"NNLS baseline RMSE: {rmse_nnls:.4f}")
        nnls_available = True
    except RuntimeError as e:
        print(f"NNLS baseline not available: {e}")
        A_nnls_df = None
        rmse_nnls = None
        nnls_available = False

    # test baselines on test set if available
    if X_test is not None and Y_test is not None:
        print("\n--- Baselines (test set) ---")

        # Global consequence baseline on test set
        A_bar_test, lam_global_test_df = global_consequence_baseline(Y_test)
        rmse_global_test = _rmse_np(Y_test.values, lam_global_test_df.values)
        print(f"Global baseline RMSE (test): {rmse_global_test:.4f}")

        # NNLS baseline on test set
        # Train side
        A_nnls_df, lambda_train_nnls = nnls_factorization_baseline(model, Y_train)

        # Test side (using same A_nnls_df)
        lambda_test_nnls = nnls_factorization_baseline_predict(
            model,      
            A_nnls_df,
            Y_df=Y_test,
            theta_df=theta_test_df,
            u=u_test,
        )
        rmse_nnls_test = _rmse_np(Y_test.values, lambda_test_nnls.values)
        print(f"NNLS baseline RMSE (test): {rmse_nnls_test:.4f}")
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


        if nnls_available and A_nnls_df is not None:
            A_nnls_df.to_csv(os.path.join(args.outdir, "A_nnls_baseline.csv"))
            lam_nnls_df.to_csv(
                os.path.join(args.outdir, "lambda_Y_nnls_baseline.csv")
            )
        # ELBO history
        pd.Series(model.elbo_history).to_csv(os.path.join(args.outdir, "elbo_history.csv"))
        # Metrics JSON
        metrics_out = {
            "global_deviance": float(metrics["global_deviance"]),
            "rmse_coupled": float(metrics["rmse"]),
            "rmse_global_baseline": float(rmse_global),
            "nnls_available": nnls_available,
        }
        if nnls_available and rmse_nnls is not None:
            metrics_out["rmse_nnls_baseline"] = float(rmse_nnls)

        with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
            json.dump(metrics_out, f, indent=2)

        print(f"\nSaved outputs to: {args.outdir}")


if __name__ == "__main__":
    main()
