# ---- CONFIG ----
MAF_PATH = "mc3.v0.2.8.PUBLIC.maf"          # or .gz; pandas can read either
EXOME_BED = "gencode.v19.basic.exome.bed"   # set to None to use a fixed default
DEFAULT_EXOME_MB = 38.0                     # widely used exome size fallback

# TMB definition toggles
SNVS_ONLY = True            # True = only Variant_Type == 'SNP'
NON_SILENT_ONLY = True      # True = exclude 'Silent' etc.

# Plot settings
OUTPUT_PNG = "tmb_mut_per_mb.png"
BINS = 60
CAP_AT_PERCENTILE = 99.5    # cap x-axis at this percentile to avoid extreme tails
LOG_X = False               # set True if distribution is very wide

import pandas as pd, numpy as np, matplotlib.pyplot as plt, os

# ---- helper: compute callable exome size in Mb from BED ----
def exome_size_mb_from_bed(bed_path: str) -> float:
    total_bases = 0
    with open(bed_path, 'r') as f:
        for line in f:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3: 
                continue
            start, end = int(parts[1]), int(parts[2])
            total_bases += max(0, end - start)
    return total_bases / 1e6

if EXOME_BED and os.path.exists(EXOME_BED):
    EXOME_MB = exome_size_mb_from_bed(EXOME_BED)
else:
    EXOME_MB = DEFAULT_EXOME_MB

print(f"Using exome size: {EXOME_MB:.2f} Mb")

# ---- filters ----
NONSILENT = {
    "Missense_Mutation","Nonsense_Mutation","Frame_Shift_Ins","Frame_Shift_Del",
    "In_Frame_Ins","In_Frame_Del","Splice_Site","Translation_Start_Site","Nonstop_Mutation",
    # Optional: include "5'UTR","3'UTR","Intron" if you want broader non-silent; usually not for TMB
}

USECOLS = ["Tumor_Sample_Barcode","Variant_Type","Variant_Classification"]

# ---- streaming count of qualifying mutations per sample ----
counts = {}

reader = pd.read_csv(
    MAF_PATH, sep="\t", comment="#", usecols=USECOLS,
    chunksize=200_000, low_memory=False
)

for i, chunk in enumerate(reader, 1):
    df = chunk

    if SNVS_ONLY:
        df = df[df["Variant_Type"] == "SNP"]

    if NON_SILENT_ONLY:
        df = df[df["Variant_Classification"].isin(NONSILENT)]

    # groupby sample and add
    grp = df.groupby("Tumor_Sample_Barcode").size()
    for sid, n in grp.items():
        counts[sid] = counts.get(sid, 0) + int(n)

    if i % 10 == 0:
        print(f"Processed ~{i*200_000:,} rows")

# ---- compute TMB (mutations per Mb) ----
tmb_df = pd.DataFrame({
    "Tumor_Sample_Barcode": list(counts.keys()),
    "mutations": list(counts.values())
})
tmb_df["TMB_mut_per_Mb"] = tmb_df["mutations"] / EXOME_MB

# ---- plot ----
x = tmb_df["TMB_mut_per_Mb"].values
cap = np.percentile(x, CAP_AT_PERCENTILE)
plt.figure(figsize=(9,5))
plt.hist(x, bins=BINS, range=(0, cap))
plt.xlabel("Tumor mutation burden (mutations per Mb)")
plt.ylabel("Number of tumors")
plt.title(f"TCGA (MC3) — TMB (SNVs only: {SNVS_ONLY}, non-silent only: {NON_SILENT_ONLY})")
if LOG_X:
    plt.xscale("log")
    plt.xlabel("Tumor mutation burden (mut/Mb, log scale)")
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=200)
plt.close()

print(f"Saved: {OUTPUT_PNG}")
print(tmb_df.describe(percentiles=[0.5,0.9,0.95,0.99])[["TMB_mut_per_Mb"]])
