#!/usr/bin/env python3
"""
Compute COSMIC SBS96 signature exposures via NNLS.

Inputs:
  - sbs96_by_tumor.csv  : samples × 96 SBS bins (columns like A[C>A]A, ...)
  - cosmic_sbs96.txt    : COSMIC SBS96 signatures (Type + SBS1..SBS99 etc.)

Outputs:
  - cosmic_exposures_counts.csv   : samples × signatures (mutation counts)
  - cosmic_exposures_simplex.csv  : samples × signatures (row-normalized)
"""

import numpy as np
import pandas as pd
from scipy.optimize import nnls

# ---- PATHS: edit as needed ----
SBS_MATRIX_PATH   = "sbs96_by_tumor.csv"
COSMIC_MATRIX_PATH = "COSMIC_v3.4_SBS_GRCh37.txt"  # the file with 'Type, SBS1, SBS2, ...'
OUT_COUNTS_PATH   = "cosmic_exposures_counts.csv"
OUT_SIMPLEX_PATH  = "cosmic_exposures_simplex.csv"

# ---- 1. Load sample × 96 SBS matrix ----
# rows: Tumor_Sample_Barcode
# cols: 96 SBS labels (e.g. A[C>A]A, A[C>A]C, ...)
X_df = pd.read_csv(SBS_MATRIX_PATH, index_col=0)
print(f"Loaded SBS matrix: {X_df.shape} (samples × 96)")

# Ensure numeric, no NaNs
X_df = X_df.apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)

# ---- 2. Load COSMIC SBS96 signature matrix ----
# Expect format:
# Type<TAB>SBS1<TAB>SBS2<...>
# A[C>A]A  0.000886157  ...
C_df = pd.read_csv(COSMIC_MATRIX_PATH, sep="\t")

# Set the 'Type' column as index (trinucleotide labels)
if "Type" not in C_df.columns:
    raise ValueError("Expected a 'Type' column in COSMIC file but didn't find it.")

C_df = C_df.set_index("Type")
print(f"Loaded COSMIC matrix: {C_df.shape} (96 × m signatures)")

# ---- 3. Align the 96 contexts with your SBS96 columns ----
contexts_sbs = list(X_df.columns)
contexts_cosmic = set(C_df.index)

missing_in_cosmic = [c for c in contexts_sbs if c not in contexts_cosmic]
if missing_in_cosmic:
    raise ValueError(
        "The following SBS96 contexts are in your matrix but not in COSMIC:\n"
        + ", ".join(missing_in_cosmic[:10])
        + (" ..." if len(missing_in_cosmic) > 10 else "")
    )

# Reorder COSMIC matrix to match your column order
C_aligned = C_df.loc[contexts_sbs]  # shape: (96, m)
print("Aligned COSMIC matrix shape:", C_aligned.shape)

# Convert to numpy
C = C_aligned.values.astype(float)   # A in NNLS
sig_names = list(C_aligned.columns)
n_signatures = C.shape[1]

# Optionally ensure each signature column sums to 1 (probability distribution)
col_sums = C.sum(axis=0, keepdims=True)
# Avoid divide-by-zero, but COSMIC should already be normalized
col_sums[col_sums == 0] = 1.0
C = C / col_sums

# ---- 4. Solve NNLS per sample:  min || C * e_i - x_i ||_2   s.t. e_i >= 0 ----
n_samples = X_df.shape[0]
exposures_counts = np.zeros((n_samples, n_signatures), dtype=float)

X_values = X_df.values  # n_samples × 96

print("Computing NNLS exposures for", n_samples, "samples and", n_signatures, "signatures...")

for i in range(n_samples):
    x_i = X_values[i, :]  # length 96
    e_i, _ = nnls(C, x_i)  # e_i: length m, nonnegative
    exposures_counts[i, :] = e_i
    if (i + 1) % 500 == 0:
        print(f"  done {i+1}/{n_samples} samples")

# ---- 5. Wrap into DataFrames and save ----
samples = X_df.index

E_counts_df = pd.DataFrame(exposures_counts, index=samples, columns=sig_names)
E_counts_df.to_csv(OUT_COUNTS_PATH)
print(f"Wrote counts-based exposures to {OUT_COUNTS_PATH} with shape {E_counts_df.shape}")

# Row-normalized exposures (simplex) for Bayesian model (θ_i)
row_sums = exposures_counts.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0  # avoid division by zero
theta = exposures_counts / row_sums

Theta_df = pd.DataFrame(theta, index=samples, columns=sig_names)
Theta_df.to_csv(OUT_SIMPLEX_PATH)
print(f"Wrote simplex-normalized exposures to {OUT_SIMPLEX_PATH} with shape {Theta_df.shape}")

print("Done.")
