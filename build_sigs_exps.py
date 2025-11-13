#!/usr/bin/env python3
import pandas as pd
import numpy as np
from SigProfilerAssignment import Analyzer as Analyze

IN_SBS_CSV   = "sbs96_by_tumor.csv"         # your sample × 96 matrix
MATRIX_TXT   = "sbs96_for_sigprofiler.txt"  # tab-delimited for cosmic_fit
OUTPUT_DIR   = "sigprofiler_assignment_out"

# 1. Load: rows = samples, cols = 96 SBS contexts
X = pd.read_csv(IN_SBS_CSV, index_col=0)
print(f"Loaded SBS matrix: {X.shape} (samples × 96)")

# Just in case: ensure numeric and no NaNs
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(0)
X = X.astype(np.int64)

# 2. Transpose to rows = channels, cols = samples
X_T = X.T.copy()  # (96, n_samples)

# 3. Build Channel column 1..96
channels = pd.Series(range(1, X_T.shape[0] + 1), name="Channel")

# Reset index so trinuc labels don't become a column
X_T_reset = X_T.reset_index(drop=True)

# Combine: first column = Channel, others = sample columns
matrix_df = pd.concat([channels, X_T_reset], axis=1)

# Column names: "Channel" + sample IDs (these are index of X)
matrix_df.columns = ["Channel"] + list(X.index)

# Final clean-up: ensure no NaNs at all and integer dtype
matrix_df = matrix_df.apply(pd.to_numeric, errors="coerce")
matrix_df = matrix_df.fillna(0)
matrix_df = matrix_df.astype(np.int64)

# Sanity checks
print("Any NaNs in matrix_df?", matrix_df.isna().any().any())
print("Dtypes:", matrix_df.dtypes.head())

# Write as tab-separated, no index
matrix_df.to_csv(MATRIX_TXT, sep="\t", index=False)
print(f"Wrote {MATRIX_TXT} with shape {matrix_df.shape}")

# Extra sanity: read back and check again
check = pd.read_csv(MATRIX_TXT, sep="\t")
print("Any NaNs after re-read?", check.isna().any().any())
print("Head of matrix file passed to cosmic_fit:")
print(check.head())

# 4. Run cosmic_fit
Analyze.cosmic_fit(
    samples=MATRIX_TXT,
    output=OUTPUT_DIR,
    input_type="matrix",
    genome_build="GRCh38",
    cosmic_version=3.4,
    collapse_to_SBS96=False,
    make_plots=False,
    export_probabilities=False,
    sample_reconstruction_plots=False,
    verbose=True,
    cpu=1,   # <-- add this
)
