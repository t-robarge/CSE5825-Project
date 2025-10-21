# --- CONFIG ---
MAF_PATH = "mc3.v0.2.8.PUBLIC.maf"   # or "mc3.v0.2.8.PUBLIC.maf.gz"
OUTPUT_DIR = "./mc3_plots"
CHUNKSIZE = 200_000  # adjust if you have more memory

import os, pandas as pd, numpy as np
import matplotlib.pyplot as plt
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Columns we need for all three plots
USECOLS = [
    "Hugo_Symbol",
    "Tumor_Sample_Barcode",
    "Variant_Classification",
    "Variant_Type",
    "Reference_Allele",
    "Tumor_Seq_Allele2",
]

# Variant classes considered "non-silent" for the gene plot
NONSILENT = {
    "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Ins", "Frame_Shift_Del",
    "In_Frame_Ins", "In_Frame_Del", "Splice_Site", "Translation_Start_Site",
    "Nonstop_Mutation"
}

# --- helpers for SBS-6 mapping ---
comp = str.maketrans({"A":"T","T":"A","C":"G","G":"C","a":"t","t":"a","c":"g","g":"c"})
def normalize_pyrimidine(ref, alt):
    """Return (ref, alt) normalized to pyrimidine ref (C/T) for SBS-6 counting."""
    if ref in ("C","T"):
        return ref, alt
    # complement both so ref is a pyrimidine
    return ref.translate(comp), alt.translate(comp)

def sbs6_class(ref, alt):
    """Map a single SNV to one of the 6 substitution classes or None if not SNV/invalid."""
    if ref is None or alt is None: 
        return None
    if len(ref) != 1 or len(alt) != 1 or ref == alt:
        return None
    ref = ref.upper(); alt = alt.upper()
    ref, alt = normalize_pyrimidine(ref, alt)
    pair = f"{ref}>{alt}"
    allowed = {"C>A","C>G","C>T","T>A","T>C","T>G"}
    return pair if pair in allowed else None

# --- accumulators ---
tmb_counts = {}                 # Tumor_Sample_Barcode -> mutation count (all calls)
gene_counts = {}                # Hugo_Symbol -> non-silent count
sbs6_counts = {"C>A":0,"C>G":0,"C>T":0,"T>A":0,"T>C":0,"T>G":0}

# --- streaming pass over the MAF ---
reader = pd.read_csv(
    MAF_PATH, sep="\t", comment="#", usecols=USECOLS, chunksize=CHUNKSIZE, low_memory=False
)

for i, chunk in enumerate(reader, 1):
    # 1) TMB per sample (count all calls; you could restrict to SNVs if you prefer)
    tmb = chunk.groupby("Tumor_Sample_Barcode").size()
    for k, v in tmb.items():
        tmb_counts[k] = tmb_counts.get(k, 0) + int(v)

    # 2) Top mutated genes (non-silent)
    ns = chunk[chunk["Variant_Classification"].isin(NONSILENT)]
    gene_ct = ns["Hugo_Symbol"].value_counts()
    for k, v in gene_ct.items():
        gene_counts[k] = gene_counts.get(k, 0) + int(v)

    # 3) SBS-6 spectrum (SNVs only; Variant_Type == 'SNP')
    snv = chunk[chunk["Variant_Type"] == "SNP"][["Reference_Allele","Tumor_Seq_Allele2"]].dropna()
    # vectorized mapping to SBS6
    refs = snv["Reference_Allele"].astype(str).str.upper()
    alts = snv["Tumor_Seq_Allele2"].astype(str).str.upper()

    # Normalize pyrimidine orientation
    ref_is_pyr = refs.isin(["C","T"])
    refs_norm = refs.where(ref_is_pyr, refs.map(lambda x: x.translate(comp)))
    alts_norm = alts.where(ref_is_pyr, alts.map(lambda x: x.translate(comp)))

    pairs = refs_norm + ">" + alts_norm
    valid = pairs.isin(sbs6_counts.keys())
    for k, v in pairs[valid].value_counts().items():
        sbs6_counts[k] += int(v)

    if i % 10 == 0:
        print(f"Processed {i*CHUNKSIZE:,} rows...")

# --- convert accumulators to DataFrames ---
tmb_df = pd.DataFrame({"Tumor_Sample_Barcode": list(tmb_counts.keys()), "mutations": list(tmb_counts.values())})
gene_df = pd.DataFrame({"Hugo_Symbol": list(gene_counts.keys()), "count": list(gene_counts.values())})
sbs6_df = pd.DataFrame({"substitution": list(sbs6_counts.keys()), "count": list(sbs6_counts.values())}).sort_values("substitution")

# --- PLOT 1: TMB histogram ---
plt.figure(figsize=(8,5))
plt.hist(tmb_df["mutations"], bins=60)
plt.xlabel("Mutations per tumor (all calls)")
plt.ylabel("Number of tumors")
plt.title("TCGA (MC3) — Tumor Mutation Burden (per sample)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "tmb_histogram.png"), dpi=200)
plt.close()

# --- PLOT 2: Top mutated genes (non-silent) ---
topN = 20
top_genes = gene_df.sort_values("count", ascending=False).head(topN)
plt.figure(figsize=(10,6))
plt.barh(top_genes["Hugo_Symbol"][::-1], top_genes["count"][::-1])
plt.xlabel("Non-silent mutation count")
plt.ylabel("Gene")
plt.title(f"TCGA (MC3) — Top {topN} Mutated Genes (non-silent)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top_genes_bar.png"), dpi=200)
plt.close()

# --- PLOT 3: SBS-6 substitution spectrum ---
order = ["C>A","C>G","C>T","T>A","T>C","T>G"]
sbs6_df = sbs6_df.set_index("substitution").loc[order].reset_index()
plt.figure(figsize=(7,4))
plt.bar(sbs6_df["substitution"], sbs6_df["count"])
plt.xlabel("Substitution class (SBS-6)")
plt.ylabel("Count")
plt.title("TCGA (MC3) — Substitution Spectrum (SBS-6)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sbs6_spectrum.png"), dpi=200)
plt.close()

print(f"Done. Saved plots in: {OUTPUT_DIR}")
