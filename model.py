from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from scipy.special import digamma, gammaln, softmax


@dataclass
class GammaPoissonConfig:
    max_iter: int = 500
    tol: float = 1e-4          # relative change tolerance on parameters
    a: float = 1.0             # hyperparameters for theta (Gamma shape)
    b: float = 1.0             # hyperparameters for theta (Gamma rate)
    beta: float = 1.0          # hyperparameters for A (Gamma shape)
    gamma: float = 1.0         # hyperparameters for A (Gamma rate)
    random_state: int | None = 0
    verbose: bool = True


class GammaPoissonCoupledModel:
    """
    Coupled Gamma-Poisson factorization with labeled data.

    Inputs (all pandas DataFrames):
      X_df: samples x contexts (SBS96 mutation counts)
      Y_df: samples x consequences
      C_ctx_sig_df: contexts x signatures (fixed COSMIC signatures)

    Latent (learned):
      theta  (samples x signatures): tumor exposures
      A      (signatures x consequences): consequence weights
    """

    # --------------------------- init ---------------------------------

    def __init__(self, K: int, config: GammaPoissonConfig | None = None):
        self.K = K
        self.config = config or GammaPoissonConfig()
        self._rng = np.random.default_rng(self.config.random_state)

        # Variational parameters (numpy arrays)
        self.a_tilde: np.ndarray | None = None  # (n, K)
        self.b_tilde: np.ndarray | None = None  # (n, K)
        self.beta_tilde: np.ndarray | None = None  # (K, L)
        self.gamma_tilde: np.ndarray | None = None  # (K, L)

        # Observed / fixed data (DataFrames)
        self.X_df: pd.DataFrame | None = None           # samples × contexts
        self.Y_df: pd.DataFrame | None = None           # samples × consequences
        self.C_ctx_sig_df: pd.DataFrame | None = None   # contexts × signatures

        # Numeric versions for fast math
        self.X: np.ndarray | None = None       # (n, M)
        self.Y: np.ndarray | None = None       # (n, L)
        self.C_sig_ctx: np.ndarray | None = None  # (K, M) signatures × contexts

        # Row-sum scalings
        self.s: np.ndarray | None = None  # total SBS counts per sample (n,)
        self.u: np.ndarray | None = None  # total consequence counts per sample (n,)

        # Labels for interpretation
        self.sample_ids: List[str] | None = None
        self.context_labels: List[str] | None = None
        self.consequence_labels: List[str] | None = None
        self.signature_labels: List[str] | None = None

    # ---------------------- properties / accessors ---------------------

    @property
    def theta_mean(self) -> np.ndarray:
        """Posterior mean E[theta_ik] = a_tilde / b_tilde (numpy array n x K)."""
        if self.a_tilde is None or self.b_tilde is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.a_tilde / self.b_tilde

    @property
    def A_mean(self) -> np.ndarray:
        """Posterior mean E[A_kl] = beta_tilde / gamma_tilde (numpy array K x L)."""
        if self.beta_tilde is None or self.gamma_tilde is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.beta_tilde / self.gamma_tilde

    @property
    def theta_df(self) -> pd.DataFrame:
        """Posterior mean theta as DataFrame: samples x signatures."""
        if self.sample_ids is None or self.signature_labels is None:
            raise RuntimeError("Labels not set; fit the model first.")
        return pd.DataFrame(self.theta_mean,
                            index=self.sample_ids,
                            columns=self.signature_labels)

    @property
    def A_df(self) -> pd.DataFrame:
        """Posterior mean A as DataFrame: signatures x consequences."""
        if self.signature_labels is None or self.consequence_labels is None:
            raise RuntimeError("Labels not set; fit the model first.")
        return pd.DataFrame(self.A_mean,
                            index=self.signature_labels,
                            columns=self.consequence_labels)

    @property
    def A_row_normalized(self) -> np.ndarray:
        """Row-normalized A (numpy K x L)."""
        A = self.A_mean
        row_sums = A.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
        return A / row_sums

    @property
    def A_row_normalized_df(self) -> pd.DataFrame:
        """Row-normalized A as DataFrame (signatures x consequences)."""
        if self.signature_labels is None or self.consequence_labels is None:
            raise RuntimeError("Labels not set; fit the model first.")
        A_norm = self.A_row_normalized
        return pd.DataFrame(A_norm,
                            index=self.signature_labels,
                            columns=self.consequence_labels)

    # ----------------------------- fit --------------------------------

    def fit(
        self,
        X_df: pd.DataFrame,
        Y_df: pd.DataFrame,
        C_ctx_sig_df: pd.DataFrame,
    ) -> "GammaPoissonCoupledModel":
        """
        Run mean-field variational inference using the Poisson allocation trick.

        Parameters
        ----------
        X_df : DataFrame (samples x contexts)
        Y_df : DataFrame (samples x consequences)
        C_ctx_sig_df : DataFrame (contexts x signatures)
        """
        # Align sample IDs between X and Y
        if not X_df.index.equals(Y_df.index):
            # try to align
            Y_df = Y_df.reindex(index=X_df.index)
            if Y_df.isnull().any().any():
                raise ValueError("Sample IDs in X and Y do not match and cannot be aligned.")

        # Align contexts between X and C
        if not set(X_df.columns).issubset(set(C_ctx_sig_df.index)):
            missing = set(X_df.columns) - set(C_ctx_sig_df.index)
            raise ValueError(f"Contexts in X not found in C index: {sorted(missing)}")

        # Reorder contexts in C to match X's column order
        C_ctx_sig_df = C_ctx_sig_df.loc[X_df.columns]

        # Store DataFrames
        self.X_df = X_df.copy()
        self.Y_df = Y_df.copy()
        self.C_ctx_sig_df = C_ctx_sig_df.copy()

        # Store labels
        self.sample_ids = list(self.X_df.index)
        self.context_labels = list(self.X_df.columns)
        self.consequence_labels = list(self.Y_df.columns)
        self.signature_labels = list(self.C_ctx_sig_df.columns)
        if len(self.signature_labels) != self.K:
            raise ValueError(
                f"Provided K={self.K}, but C has {len(self.signature_labels)} signatures."
            )

        # Numeric arrays for math
        self.X = self.X_df.values.astype(float)           # (n, M)
        self.Y = self.Y_df.values.astype(float)           # (n, L)
        self.C_sig_ctx = self.C_ctx_sig_df.T.values.astype(float)  # (K, M)

        n, M = self.X.shape
        nY, L = self.Y.shape
        K, M_C = self.C_sig_ctx.shape

        if n != nY:
            raise ValueError("X and Y must have the same number of samples (rows).")
        if M != M_C:
            raise ValueError(f"X and C must have same #contexts. Got M={M}, M_C={M_C}.")
        if K != self.K:
            raise ValueError(f"C implies K={K} signatures, but model K={self.K}.")

        # Row-sum scalings
        self.s = self.X.sum(axis=1)  # SBS burden per sample
        self.u = self.Y.sum(axis=1)  # consequence burden per sample

        # Broadcast-able hyperparameters
        a_h = np.broadcast_to(self.config.a, (self.K,))
        b_h = np.broadcast_to(self.config.b, (self.K,))
        beta_h = np.broadcast_to(self.config.beta, (L,))
        gamma_h = np.broadcast_to(self.config.gamma, (L,))

        # store ELBO history if desired
        self.elbo_history = []

        # --------- initialize variational parameters (theta, A) --------

        # theta variational params
        self.a_tilde = np.tile(a_h, (n, 1)) + self._rng.gamma(
            shape=1.0, scale=0.1, size=(n, self.K)
        )
        self.b_tilde = np.tile(b_h, (n, 1)) + 1.0

        # A variational params
        self.beta_tilde = np.tile(beta_h, (self.K, 1)) + self._rng.gamma(
            shape=1.0, scale=0.1, size=(self.K, L)
        )

        # gamma_tilde initial using theta_mean and u
        theta_mean = self.theta_mean
        gamma_add = (self.u[:, None] * theta_mean).sum(axis=0)  # (K,)
        self.gamma_tilde = gamma_h[None, :] + gamma_add[:, None]

        C_row_sums = self.C_sig_ctx.sum(axis=1)  # (K,)

        prev_theta = self.theta_mean.copy()
        prev_A = self.A_mean.copy()
        prev_elbo = -np.inf

        # ----------------- VI update loop ------------------------------

        eps = 1e-12  # for safe logs

        for it in range(self.config.max_iter):
            # Current expectations
            theta_mean = self.theta_mean  # (n, K)
            A_mean = self.A_mean          # (K, L)

            # ---------- MFVI responsibilities using exp(E[log ·]) ----------
            E_log_theta = digamma(self.a_tilde) - np.log(self.b_tilde)          # (n,K)
            E_log_A     = digamma(self.beta_tilde) - np.log(self.gamma_tilde)  # (K,L)

            # X-channel: r_X(i,k,m) ∝ exp(E_log_theta(i,k)) * C(k,m)
            log_C = np.log(np.clip(self.C_sig_ctx, eps, None))                  # (K,M)
            logits_X = E_log_theta[:, :, None] + log_C[None, :, :]              # (n,K,M)
            r_X = softmax(logits_X, axis=1)                                     # softmax over k

            X_hat = self.X[:, None, :] * r_X                                    # (n,K,M)
            X_hat_sum_m = X_hat.sum(axis=2)                                     # (n,K)

            # Y-channel: r_Y(i,k,l) ∝ exp(E_log_theta(i,k) + E_log_A(k,l))
            logits_Y = E_log_theta[:, :, None] + E_log_A[None, :, :]            # (n,K,L)
            r_Y = softmax(logits_Y, axis=1)                                     # softmax over k

            Y_hat = self.Y[:, None, :] * r_Y                                    # (n,K,L)
            Y_hat_sum_l = Y_hat.sum(axis=2)                                     # (n,K)
            Y_hat_sum_i = Y_hat.sum(axis=0)                                     # (K,L)

            # ------- Update variational parameters -------
            # theta
            self.a_tilde = a_h[None, :] + X_hat_sum_m + Y_hat_sum_l             # (n,K)

            A_row_sums = A_mean.sum(axis=1)                                     # (K,)
            self.b_tilde = (
                b_h[None, :]
                + self.s[:, None] * C_row_sums[None, :]
                + self.u[:, None] * A_row_sums[None, :]
            )

            # A
            self.beta_tilde = beta_h[None, :] + Y_hat_sum_i                     # (K,L)

            theta_mean = self.theta_mean
            gamma_add = (self.u[:, None] * theta_mean).sum(axis=0)              # (K,)
            self.gamma_tilde = gamma_h[None, :] + gamma_add[:, None]            # (K,L)

            # ------- Convergence check (ELBO-based) -------
            if it % 10 == 0 or it == self.config.max_iter - 1:
                theta_now = self.theta_mean
                A_now = self.A_mean

                rel_change_theta = np.max(
                    np.abs(theta_now - prev_theta) / (1.0 + np.abs(prev_theta))
                )
                rel_change_A = np.max(
                    np.abs(A_now - prev_A) / (1.0 + np.abs(prev_A))
                )
                max_rel_change = max(rel_change_theta, rel_change_A)

                elbo_now = self._elbo_from_current_state(r_X, r_Y, X_hat, Y_hat)
                self.elbo_history.append(elbo_now)
                if self.config.verbose:
                    print(
                        f"[iter {it}] max rel change: {max_rel_change:.3e}, "
                        f"ELBO: {elbo_now:.6e}, ΔELBO={elbo_now - prev_elbo:.6e}"
                    )

                if elbo_now - prev_elbo < self.config.tol:
                    if self.config.verbose:
                        print(
                            f"Converged after {it} iterations: "
                            f"ΔELBO={elbo_now - prev_elbo:.3e} < {self.config.tol}"
                        )
                    break

                prev_elbo = elbo_now
                prev_theta = theta_now.copy()
                prev_A = A_now.copy()

        return self

    # ----------------- lambda (rate) computations ---------------------

    def expected_lambda_Y(self) -> np.ndarray:
        """
        Expected Poisson rate for Y (numpy): λ_hat(i,l) = u_i * sum_k theta(i,k)*A(k,l).
        Shape: (n, L).
        """
        if self.u is None:
            raise RuntimeError("Model not fitted; u undefined.")
        theta_mean = self.theta_mean        # n×K
        A_mean = self.A_mean                # K×L
        rates = theta_mean @ A_mean         # n×L
        return self.u[:, None] * rates

    def expected_lambda_Y_df(self) -> pd.DataFrame:
        """Expected λ_Y as DataFrame (samples x consequences)."""
        lam = self.expected_lambda_Y()
        return pd.DataFrame(lam,
                            index=self.sample_ids,
                            columns=self.consequence_labels)

    def expected_lambda_X(self) -> np.ndarray:
        """
        Expected Poisson rate for X (numpy):
          λ_hat(i,m) = s_i * sum_k theta(i,k) * C(k,m),
        where C is signatures x contexts (self.C_sig_ctx).
        Shape: (n, M).
        """
        if self.s is None or self.C_sig_ctx is None:
            raise RuntimeError("Model not fitted; s or C undefined.")
        theta_mean = self.theta_mean
        rates = theta_mean @ self.C_sig_ctx  # n×M
        return self.s[:, None] * rates

    def expected_lambda_X_df(self) -> pd.DataFrame:
        """Expected λ_X as DataFrame (samples x contexts)."""
        lam = self.expected_lambda_X()
        return pd.DataFrame(lam,
                            index=self.sample_ids,
                            columns=self.context_labels)

    # ------------------------------------------------------------ elbo ------------------------------------------------------------

    def _elbo_from_current_state(self, r_X, r_Y, X_hat, Y_hat) -> float:
        """
        Full ELBO (up to data-only constants) given current variational params
        and weights r_X, r_Y, expected allocations X_hat, Y_hat.
        """
        X = self.X.astype(float)        # (n,M)
        Y = self.Y.astype(float)        # (n,L)
        s = self.s                      # (n,)
        u = self.u                      # (n,)
        C = self.C_sig_ctx              # (K,M)

        a_t = self.a_tilde              # (n,K)
        b_t = self.b_tilde              # (n,K)
        beta_t = self.beta_tilde        # (K,L)
        gamma_t = self.gamma_tilde      # (K,L)

        theta_mean = a_t / b_t
        A_mean = beta_t / gamma_t

        E_log_theta = digamma(a_t) - np.log(b_t)
        E_log_A = digamma(beta_t) - np.log(gamma_t)

        # hyperparameters
        K = self.K
        _, L = A_mean.shape
        a_h = np.broadcast_to(self.config.a, (K,))
        b_h = np.broadcast_to(self.config.b, (K,))
        beta_h = np.broadcast_to(self.config.beta, (L,))
        gamma_h = np.broadcast_to(self.config.gamma, (L,))

        a0 = a_h[None, :]
        b0 = b_h[None, :]
        beta0 = beta_h[None, :]
        gamma0 = gamma_h[None, :]

        # --- theta prior + entropy ---
        log_p_theta = np.sum(
            (a0 - 1.0) * E_log_theta
            - b0 * theta_mean
            - gammaln(a0) + a0 * np.log(b0)
        )
        H_theta = np.sum(
            a_t - np.log(b_t) + gammaln(a_t) + (1.0 - a_t) * digamma(a_t)
        )

        # --- A prior + entropy ---
        log_p_A = np.sum(
            (beta0 - 1.0) * E_log_A
            - gamma0 * A_mean
            - gammaln(beta0) + beta0 * np.log(gamma0)
        )
        H_A = np.sum(
            beta_t - np.log(gamma_t) + gammaln(beta_t) + (1.0 - beta_t) * digamma(beta_t)
        )

        # --- X allocation / Poisson part ---
        log_s = np.log(np.where(s <= 0.0, 1e-12, s))
        log_C = np.log(np.where(C <= 0.0, 1e-12, C))

        term_X1 = (
            X_hat
            * (
                log_s[:, None, None]
                + log_C[None, :, :]
                + E_log_theta[:, :, None]
                - np.log(np.where(r_X <= 0.0, 1e-12, r_X))
            )
        ).sum()

        term_X2 = ((s[:, None] * theta_mean) @ C).sum()
        L_X = term_X1 - term_X2

        # --- Y allocation / Poisson part ---
        log_u = np.log(np.where(u <= 0.0, 1e-12, u))

        term_Y1 = (
            Y_hat
            * (
                log_u[:, None, None]
                + E_log_theta[:, :, None]
                + E_log_A[None, :, :]
                - np.log(np.where(r_Y <= 0.0, 1e-12, r_Y))
            )
        ).sum()

        term_Y2 = ((u[:, None] * theta_mean) @ A_mean).sum()
        L_Y = term_Y1 - term_Y2

        elbo = log_p_theta + H_theta + log_p_A + H_A + L_X + L_Y
        return float(elbo)
