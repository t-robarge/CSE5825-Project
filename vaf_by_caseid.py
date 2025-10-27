import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

MAF = "mc3.v0.2.8.PUBLIC.maf"   # <- your path
CASE_IDS = ["TCGA-02-0003","TCGA-02-0055","TCGA-02-2466"]     # <- put one or more case IDs here

# tumor sample-type prefixes (01=Primary Solid Tumor, 02=Recurrent, 06=Metastatic, 03=blood tumor)
TUMOR_TYPES = {"01","02","06","03"}

# --- load + basics ---
df = pd.read_csv(MAF, sep="\t", comment="#", low_memory=False)

# numeric counts
for c in ["t_depth","t_alt_count","n_depth","n_alt_count"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# add helpers
def case_of(barcode: str) -> str:
    try:
        a,b,c,*_ = barcode.split("-")
        return f"{a}-{b}-{c}"
    except Exception:
        return np.nan

def is_tumor_sample(barcode: str) -> bool:
    try:
        st = barcode.split("-")[3][:2]
        return st in TUMOR_TYPES
    except Exception:
        return False

df["case_id"] = df["Tumor_Sample_Barcode"].map(case_of)
df["tumor_vaf"] = df["t_alt_count"] / df["t_depth"]

# QC filters (tweak as needed)
qc = (
    (df["FILTER"] == "PASS") &
    (df["t_depth"] >= 30) &
    (df["t_alt_count"] >= 3) &
    df["Tumor_Sample_Barcode"].map(is_tumor_sample) &
    np.isfinite(df["tumor_vaf"])
)

df_qc = df[qc].copy()

# --- plot per case ---
for cid in CASE_IDS:
    sub = df_qc[df_qc["case_id"] == cid]
    if sub.empty:
        print(f"[warn] no PASS variants for case {cid} under current filters.")
        continue

    plt.figure(figsize=(6,4))
    plt.hist(sub["tumor_vaf"], bins=40, density=True)
    for x in [0.25, 0.5, 0.75]:
        plt.axvline(x, linestyle="--", linewidth=1)
    nvars = len(sub)
    title = f"{cid} — Tumor VAF (PASS, depth≥30, alt≥3)  [n={nvars}]"
    plt.title(title)
    plt.xlabel("Tumor VAF (t_alt_count / t_depth)")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.show()
