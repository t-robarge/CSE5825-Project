#!/usr/bin/env python3
"""
Visualize tumor × SBS-96 matrix:
- Row-normalized heatmap
- Cohort SBS-96 bar plot
- PCA scatter
- UMAP scatter (if umap-learn is available)

Input: sbs96_by_tumor.csv  (rows=tumors, columns=96 bins)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Config ----------
CSV_PATH = "sbs96_by_tumor.csv"
OUTDIR = "sbs96_figs"
os.makedirs(OUTDIR, exist_ok=True)

# ---------- Load matrix ----------
def load_sbs96_matrix(path: str) -> pd.DataFrame:
    # Try reading with first column as index; fall back if needed
    try:
        df = pd.read_csv(path, index_col=0)
    except Exception:
        df = pd.read_csv(path)
        if df.columns[0].lower() in {"tumor_sample_barcode", "sample", "id"}:
            df = df.set_index(df.columns[0])
    # Make sure it’s numeric
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    return df

X = load_sbs96_matrix(CSV_PATH)
if X.shape[1] != 96:
    print(f"[warn] Expected 96 columns; found {X.shape[1]}. Continuing anyway.")

# ---------- Row-normalize ----------
row_sums = X.sum(axis=1).replace(0, np.nan)
Xrn = X.div(row_sums, axis=0).fillna(0).values  # numpy array for plotting convenience

# ---------- 1) Row-normalized heatmap ----------
plt.figure(figsize=(11, 6))
plt.imshow(Xrn, aspect="auto", interpolation="nearest")
plt.xlabel("SBS-96 bins (C>A, C>G, C>T, T>A, T>C, T>G; left/right=A,C,G,T)")
plt.ylabel("Tumor samples")
plt.title("Row-normalized SBS-96 heatmap")
plt.colorbar(label="Proportion")
plt.tight_layout()
heatmap_path = os.path.join(OUTDIR, "sbs96_heatmap_row_normalized.png")
plt.savefig(heatmap_path, dpi=200)
plt.show()
print(f"[saved] {heatmap_path}")

# ---------- 2) Cohort SBS-96 bar plot ----------
cohort = X.sum(axis=0).astype(float)
cohort = cohort / (cohort.sum() + 1e-12)

plt.figure(figsize=(11, 3.5))
plt.bar(range(cohort.shape[0]), cohort.values)
plt.xlabel("SBS-96 bin index (0..95)")
plt.ylabel("Proportion")
plt.title("Cohort SBS-96 profile (sum across tumors, normalized)")
plt.tight_layout()
bar_path = os.path.join(OUTDIR, "sbs96_cohort_barplot.png")
plt.savefig(bar_path, dpi=200)
plt.show()
print(f"[saved] {bar_path}")

# ---------- 3) PCA (2D) ----------
from sklearn.decomposition import PCA

# Use row-normalized for geometry
pca = PCA(n_components=2, random_state=0)
Zp = pca.fit_transform(Xrn)

plt.figure(figsize=(6, 5))
plt.scatter(Zp[:, 0], Zp[:, 1], s=10, alpha=0.8)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
plt.title("PCA of row-normalized SBS-96")
plt.tight_layout()
pca_path = os.path.join(OUTDIR, "sbs96_pca.png")
plt.savefig(pca_path, dpi=200)
plt.show()
print(f"[saved] {pca_path}")

# ---------- 4) UMAP (2D), if available ----------
try:
    import umap.umap_ as umap
    reducer = umap.UMAP(n_components=2, random_state=0)
    Zu = reducer.fit_transform(Xrn)

    plt.figure(figsize=(6, 5))
    plt.scatter(Zu[:, 0], Zu[:, 1], s=10, alpha=0.8)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title("UMAP of row-normalized SBS-96")
    plt.tight_layout()
    umap_path = os.path.join(OUTDIR, "sbs96_umap.png")
    plt.savefig(umap_path, dpi=200)
    plt.show()
    print(f"[saved] {umap_path}")
except Exception as e:
    print("[note] UMAP not installed or failed to run. Install via: pip install umap-learn")
    print(f"[detail] {e}")
