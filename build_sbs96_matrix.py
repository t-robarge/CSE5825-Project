#!/usr/bin/env python3
import pandas as pd
import numpy as np

MAF_PATH = "mc3.v0.2.8.PUBLIC.maf"
OUT_CSV  = "sbs96_by_tumor.csv"

# --- Helpers ---
COMP = {"A":"T","T":"A","C":"G","G":"C"}

def rc(seq: str) -> str:
    return "".join(COMP.get(b, "N") for b in seq[::-1])

def tri_from_context(ctx: str) -> str | None:
    """Extract central trinucleotide from MAF CONTEXT (handles odd-length contexts)."""
    if not isinstance(ctx, str) or len(ctx) < 3:
        return None
    i = len(ctx) // 2
    if i == 0 or i == len(ctx) - 1:
        return None
    return ctx[i-1:i+2].upper()

def norm_pyrimidine(ref: str, alt: str, tri: str) -> tuple[str, str, str] | None:
    """Normalize to pyrimidine (C/T) reference; reverse-complement if ref is A/G."""
    if not ref or not alt or not tri or len(tri) != 3:
        return None
    ref = ref.upper(); alt = alt.upper(); tri = tri.upper()
    if ref in ("C","T"):
        return ref, alt, tri
    elif ref in ("A","G"):
        return COMP.get(ref,"N"), COMP.get(alt,"N"), rc(tri)
    else:
        return None

def sbs96_label(ref: str, alt: str, tri: str) -> str:
    """Return canonical SBS-96 label like 'A[C>T]G' (left[ref>alt]right)."""
    left, mid, right = tri[0], tri[1], tri[2]
    return f"{left}[{ref}>{alt}]{right}"

def all_sbs96_columns() -> list[str]:
    """Canonical SBS-96 column order: substitutions in C>A, C>G, C>T, T>A, T>C, T>G;
       for each, left in A,C,G,T then right in A,C,G,T."""
    subs = [("C","A"), ("C","G"), ("C","T"), ("T","A"), ("T","C"), ("T","G")]
    bases = ["A","C","G","T"]
    cols = []
    for ref, alt in subs:
        for left in bases:
            for right in bases:
                cols.append(f"{left}[{ref}>{alt}]{right}")
    return cols

# --- Load MAF (only needed cols) ---
usecols = [
    "Tumor_Sample_Barcode", "FILTER", "Variant_Type",
    "Reference_Allele", "Tumor_Seq_Allele2", "CONTEXT"
]
df = pd.read_csv(MAF_PATH, sep="\t", comment="#", usecols=usecols, low_memory=False)

# Keep PASS SNPs with necessary fields
df = df[
    (df["FILTER"] == "PASS") &
    (df["Variant_Type"] == "SNP") &
    df["Tumor_Sample_Barcode"].notna() &
    df["Reference_Allele"].notna() &
    df["Tumor_Seq_Allele2"].notna() &
    df["CONTEXT"].notna()
].copy()

# Extract central trinucleotide from CONTEXT
df["tri"] = df["CONTEXT"].apply(tri_from_context)

# Normalize to pyrimidine convention
def to_bin(row):
    tri = row["tri"]
    ref = row["Reference_Allele"]
    alt = row["Tumor_Seq_Allele2"]
    norm = norm_pyrimidine(ref, alt, tri)
    if norm is None:
        return np.nan
    r, a, t = norm
    # ensure middle base matches the normalized ref
    if len(t) != 3 or t[1] != r or r not in ("C","T"):
        return np.nan
    if a not in ("A","C","G","T") or a == r:
        return np.nan
    return sbs96_label(r, a, t)

df["bin96"] = df.apply(to_bin, axis=1)
df = df.dropna(subset=["bin96"])

# Build tumor_sample × SBS-96 counts
mat = (df.assign(n=1)
         .pivot_table(index="Tumor_Sample_Barcode", columns="bin96",
                      values="n", aggfunc="sum", fill_value=0))

# Ensure all 96 columns exist and are in canonical order
all_cols = all_sbs96_columns()
for c in all_cols:
    if c not in mat.columns:
        mat[c] = 0
mat = mat.reindex(columns=all_cols).astype(int)

# Save
mat.to_csv(OUT_CSV)
print(f"Wrote {OUT_CSV} with shape {mat.shape}")

# (Optional) quick row-normalized heatmap preview (requires matplotlib)
try:
    import matplotlib.pyplot as plt
    X = mat.values.astype(float)
    X = X / (X.sum(axis=1, keepdims=True) + 1e-9)  # row-normalize exposures
    plt.figure(figsize=(10, 6))
    plt.imshow(X, aspect="auto", interpolation="nearest")
    plt.xlabel("SBS-96 bins (C>A, C>G, C>T, T>A, T>C, T>G)")
    plt.ylabel("Tumor sample")
    plt.title("Row-normalized SBS-96 exposures")
    plt.colorbar(label="Proportion")
    plt.tight_layout()
    plt.show()
except Exception:
    pass
