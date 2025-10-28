import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- load matrix ---
X = pd.read_csv("sbs96_by_tumor.csv", index_col=0).apply(pd.to_numeric, errors="coerce").fillna(0)

# Row-normalize for geometry
row_sums = X.sum(axis=1).replace(0, np.nan)
Xrn = X.div(row_sums, axis=0).fillna(0).values

# ---- OUTDIR ----
OUTDIR = "sbs_figs"
os.makedirs(OUTDIR, exist_ok=True)

# --- UMAP with cosine ---
try:
    from umap import UMAP  # modern umap-learn
except Exception:
    import umap.umap_ as umap
    UMAP = umap.UMAP

reducer = UMAP(n_components=2, metric="cosine", n_neighbors=30, min_dist=0.15, random_state=0)
Z = reducer.fit_transform(Xrn)

# --- helpers to pick bins ---
subs = [("C","A"),("C","G"),("C","T"),("T","A"),("T","C"),("T","G")]
bases = ["A","C","G","T"]

def index_from_label(lbl):
    left, rest = lbl.split("[")
    ref_alt, right = rest.split("]")
    ref, alt = ref_alt.split(">")
    block = subs.index((ref, alt))
    L = bases.index(left); R = bases.index(right)
    return block*16 + L*4 + R

# Map columns to 0..95 indices
try:
    col_idx = np.array([int(c) for c in X.columns])
except Exception:
    col_idx = np.array([index_from_label(c) for c in X.columns])

order = np.argsort(col_idx)
X = X.iloc[:, order]; col_idx = col_idx[order]
X_counts = X.values
burden = X_counts.sum(axis=1)

def block_mask(bid):  # 0:C>A,1:C>G,2:C>T,3:T>A,4:T>C,5:T>G
    return (col_idx // 16) == bid

def cpg_ct_mask():
    # C>T at CpG: block 2 (C>T), right base G (index 2)
    return (col_idx // 16 == 2) & ((col_idx % 4) == bases.index("G"))

def apobec_mask():
    # APOBEC proxy TpCpW across C>T and C>G blocks at T[C]W contexts
    block_ct = (col_idx // 16 == 2)
    block_cg = (col_idx // 16 == 1)
    within = col_idx % 16
    left_T = (within // 4) == bases.index("T")
    right_A = (within % 4) == bases.index("A")
    right_T = (within % 4) == bases.index("T")
    tcw = left_T & (right_A | right_T)
    return (block_ct | block_cg) & tcw

# Fractions
eps = 1e-12
frac_cpg_ct = X_counts[:, cpg_ct_mask()].sum(axis=1) / (burden + eps)
frac_tc      = X_counts[:, block_mask(4)].sum(axis=1)      / (burden + eps)  # T>C
frac_ca      = X_counts[:, block_mask(0)].sum(axis=1)      / (burden + eps)  # C>A
frac_cg      = X_counts[:, block_mask(1)].sum(axis=1)      / (burden + eps)  # C>G
frac_ta      = X_counts[:, block_mask(3)].sum(axis=1)      / (burden + eps)  # T>A  <-- NEW
frac_tg      = X_counts[:, block_mask(5)].sum(axis=1)      / (burden + eps)  # T>G  <-- NEW
apobec_score = X_counts[:, apobec_mask()].sum(axis=1)      / (burden + eps)

# --- plot helper that SAVES to sbs_figs ---
def umap_color(vals, title, fname, cmap="viridis"):
    plt.figure(figsize=(6,5))
    sc = plt.scatter(Z[:,0], Z[:,1], c=vals, s=8, cmap=cmap)
    plt.colorbar(sc, label=title)
    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
    plt.title(f"UMAP (colored by {title})")
    plt.tight_layout()
    out = os.path.join(OUTDIR, fname)
    plt.savefig(out, dpi=200)
    plt.show()
    print(f"[saved] {out}")

# Plots
umap_color(np.log10(burden+1), "log10 mutation burden",  "umap_burden.png")
umap_color(frac_cpg_ct,       "fraction C>T at CpG",     "umap_cpg_ct.png")
umap_color(frac_tc,           "fraction T>C",            "umap_tc.png")
umap_color(frac_ca,           "fraction C>A",            "umap_ca.png")
umap_color(frac_cg,           "fraction C>G",            "umap_cg.png")
umap_color(frac_ta,           "fraction T>A",            "umap_ta.png")   # <-- NEW
umap_color(frac_tg,           "fraction T>G",            "umap_tg.png")   # <-- NEW
umap_color(apobec_score,      "APOBEC proxy (TpCpW)",    "umap_apobec.png")
