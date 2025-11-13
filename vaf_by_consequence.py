#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import textwrap
import os

def main():
    # --- Load MAF ---
    maf_path = "mc3.v0.2.8.PUBLIC.maf"
    if not os.path.exists(maf_path):
        raise FileNotFoundError(f"Could not find {maf_path}")

    maf = pd.read_csv(maf_path, sep="\t", comment="#", low_memory=False)

    # --- Ensure numeric depth/counts ---
    for c in ["t_depth", "t_alt_count", "n_depth", "n_alt_count"]:
        if c in maf.columns:
            maf[c] = pd.to_numeric(maf[c], errors="coerce")
        else:
            raise KeyError(f"Expected column '{c}' not found in MAF.")

    # --- Filter to PASS and compute VAFs ---
    if "FILTER" not in maf.columns:
        raise KeyError("Expected column 'FILTER' not found in MAF.")
    maf = maf.query("FILTER == 'PASS'").copy()

    maf["tumor_vaf"]  = maf["t_alt_count"] / maf["t_depth"]
    maf["normal_vaf"] = maf["n_alt_count"] / maf["n_depth"]

    # --- Basic QC: sufficient tumor depth, finite VAF ---
    maf_qc = maf[(maf["t_depth"] >= 30) & np.isfinite(maf["tumor_vaf"])].copy()

    # --- Require Variant_Classification column ---
    if "Variant_Classification" not in maf_qc.columns:
        raise KeyError("Expected column 'Variant_Classification' not found in MAF.")

    # --- Order classes by count (desc) so busy classes are grouped first ---
    vc_counts = maf_qc["Variant_Classification"].value_counts()
    ordered_classes = vc_counts.index.tolist()

    # Convert to categorical to control plotting order
    maf_qc["Variant_Classification"] = pd.Categorical(
        maf_qc["Variant_Classification"], categories=ordered_classes, ordered=True
    )

    # --- Plot: VAF by every variant classification ---
    n_classes = len(ordered_classes)
    # Dynamic width so labels have room; height fixed
    fig_w = max(10, 0.6 * n_classes)  # 0.6 inch per class, at least 10 inches
    fig_h = 6

    plt.figure(figsize=(fig_w, fig_h))
    ax = maf_qc.boxplot(
        column="tumor_vaf",
        by="Variant_Classification",
        rot=0,                 # we'll rotate manually for better control
        grid=False,
        showfliers=False,      # optional: hide extreme outliers for readability
        widths=0.6
    )

    # --- Improve labels: wrap + rotate + small font ---
    # Wrap each tick label to <= 12 chars per line (adjust as needed)
    wrapped = [textwrap.fill(str(lbl.get_text()), width=12) for lbl in ax.get_xticklabels()]
    ax.set_xticklabels(wrapped, rotation=35, ha="right", fontsize=8)

    ax.set_ylabel("Tumor VAF", fontsize=11)
    ax.set_xlabel("")  # the "by" add-on creates an x-label; remove it for cleanliness
    ax.set_title("Tumor VAF by Variant Classification", fontsize=13, pad=10)

    # Remove the automatic pandas suptitle
    plt.suptitle("")

    # Tighten layout and add extra bottom margin to avoid label clipping
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)

    # Higher DPI so text is crisp
    out_path = "vaf_by_variant_classification.png"
    plt.savefig(out_path, dpi=200)
    print(f"Saved figure to: {out_path}")

if __name__ == "__main__":
    main()
