#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

MAF_PATH   = "mc3.v0.2.8.PUBLIC.maf"
OUT_FX_CSV = "fx_by_tumor_sbs_filtered.csv"   # sample × Variant_Classification

# Only need these columns (plus those used in the filters)
usecols = [
    "Tumor_Sample_Barcode",
    "FILTER",
    "Variant_Type",
    "Reference_Allele",
    "Tumor_Seq_Allele2",
    "CONTEXT",
    "Variant_Classification",
]

df = pd.read_csv(MAF_PATH, sep="\t", comment="#", usecols=usecols, low_memory=False)

# ---- EXACT SAME FILTERS AS YOUR SBS96 CODE ----
df = df[
    (df["FILTER"] == "PASS") &
    (df["Variant_Type"] == "SNP") &
    df["Tumor_Sample_Barcode"].notna() &
    df["Reference_Allele"].notna() &
    df["Tumor_Seq_Allele2"].notna() &
    df["CONTEXT"].notna()
].copy()
# -----------------------------------------------

# Keep only rows with a valid Variant_Classification
df = df[df["Variant_Classification"].notna()].copy()
df["Tumor_Sample_Barcode"]   = df["Tumor_Sample_Barcode"].astype(str)
df["Variant_Classification"] = df["Variant_Classification"].astype(str)

# Build tumor_sample × Variant_Classification counts
fx_mat = (
    df.assign(n=1)
      .pivot_table(index="Tumor_Sample_Barcode",
                   columns="Variant_Classification",
                   values="n",
                   aggfunc="sum",
                   fill_value=0)
      .astype(int)
)

# Nice stable column order
fx_mat = fx_mat.reindex(sorted(fx_mat.columns), axis=1)

Path(OUT_FX_CSV).parent.mkdir(parents=True, exist_ok=True)
fx_mat.to_csv(OUT_FX_CSV)
print(f"Wrote {OUT_FX_CSV} with shape {fx_mat.shape}")

# Optional: quick summary of the most common classes
try:
    totals = fx_mat.sum(axis=0).sort_values(ascending=False)
    print("\nTop Variant_Classification by total SNP counts:")
    print(totals.head(10).to_string())
except Exception:
    pass
