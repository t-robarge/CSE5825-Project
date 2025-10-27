import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# -------- CONFIG --------
MAF_PATH = "mc3.v0.2.8.PUBLIC.maf"  # Path to your decompressed MAF
CHUNKSIZE = 500_000  # Adjust based on memory (500k rows per chunk)

# Columns that actually exist in the MC3 public MAF
USECOLS = [
    "Hugo_Symbol",
    "Variant_Classification",
    "Variant_Type",
    "Tumor_Sample_Barcode"
]

# -------- 1. Initialize counters --------
total_rows = 0
mut_counts = defaultdict(int)
variant_class_counts = defaultdict(int)
variant_type_counts = defaultdict(int)
gene_counts = defaultdict(int)

# Per–cancer type aggregations
mutations_per_cancer = defaultdict(int)
samples_per_cancer = defaultdict(set)

# -------- 2. Process in chunks --------
for chunk in pd.read_csv(MAF_PATH, sep="\t", comment="#", usecols=USECOLS, chunksize=CHUNKSIZE, low_memory=False):
    total_rows += len(chunk)
    
    # Derive cancer type prefix from Tumor_Sample_Barcode (e.g., "TCGA-BRCA-...") → "BRCA"
    cancer_types = chunk["Tumor_Sample_Barcode"].str.extract(r"TCGA-([A-Z0-9]+)", expand=False).fillna("UNKNOWN")
    chunk["Cancer_Type"] = cancer_types

    # Per–sample mutation counts
    for sid, cnt in chunk["Tumor_Sample_Barcode"].value_counts().items():
        mut_counts[sid] += cnt

    # Variant classifications
    for vclass, cnt in chunk["Variant_Classification"].value_counts().items():
        variant_class_counts[vclass] += cnt

    # Variant types
    for vtype, cnt in chunk["Variant_Type"].value_counts().items():
        variant_type_counts[vtype] += cnt

    # Genes
    for gene, cnt in chunk["Hugo_Symbol"].value_counts().items():
        gene_counts[gene] += cnt

    # Per–cancer type
    for ctype, cnt in cancer_types.value_counts().items():
        mutations_per_cancer[ctype] += cnt

    # Unique samples per cancer type
    tmp = chunk[["Tumor_Sample_Barcode", "Cancer_Type"]].drop_duplicates()
    for ctype, grp in tmp.groupby("Cancer_Type"):
        samples_per_cancer[ctype].update(grp["Tumor_Sample_Barcode"].tolist())

# -------- 3. Summaries --------
n_mutations = total_rows
n_samples = len(mut_counts)
n_genes = len(gene_counts)

print(f"\n📊 MAF SUMMARY")
print(f"Total mutations: {n_mutations:,}")
print(f"Unique tumor samples: {n_samples:,}")
print(f"Unique genes mutated: {n_genes:,}\n")

# Variant Classifications
vc_df = pd.Series(variant_class_counts).sort_values(ascending=False)
print("Top Variant Classifications:")
print(vc_df.head(10), "\n")

# Variant Types
vt_df = pd.Series(variant_type_counts).sort_values(ascending=False)
print("Variant Types:")
print(vt_df, "\n")

# Genes
gene_df = pd.Series(gene_counts).sort_values(ascending=False)
print("Top 10 Most Mutated Genes:")
print(gene_df.head(10), "\n")

# Mutations per Sample
mut_per_sample = pd.Series(mut_counts).sort_values(ascending=False)
print("Mutation Counts per Sample (summary):")
print(mut_per_sample.describe())

# -------- 4. Cancer Type Summaries --------
mut_per_cancer_df = pd.Series(mutations_per_cancer, name="mutations").sort_values(ascending=False).to_frame()
tumors_per_cancer_df = pd.Series({k: len(v) for k, v in samples_per_cancer.items()},
                                 name="unique_tumors").sort_values(ascending=False).to_frame()

mut_per_cancer_df.to_csv("mutations_per_cancer_type.csv")
tumors_per_cancer_df.to_csv("tumors_per_cancer_type.csv")

print("\n📂 Saved per-cancer-type tables:")
print("mutations_per_cancer_type.csv")
print("tumors_per_cancer_type.csv")

# -------- 5. Visualizations --------
plt.figure(figsize=(8,4))
vc_df.head(10).plot(kind="barh", color="skyblue")
plt.title("Top Variant Classifications")
plt.xlabel("Count")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("variant_classifications.png", dpi=200)

plt.figure(figsize=(8,4))
gene_df.head(15).plot(kind="barh", color="salmon")
plt.title("Top Mutated Genes")
plt.xlabel("Mutations")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("top_genes.png", dpi=200)

plt.figure(figsize=(8,4))
plt.hist(mut_per_sample, bins=100, color="steelblue")
plt.title("Mutations per Sample")
plt.xlabel("# Mutations")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("mutations_per_sample_hist.png", dpi=200)

# Per–cancer type plots (top 20)
plt.figure(figsize=(9,5))
mut_per_cancer_df.head(20).iloc[::-1]["mutations"].plot(kind="barh", color="teal")
plt.title("Mutations per Cancer Type (Top 20)")
plt.xlabel("Total Mutations")
plt.tight_layout()
plt.savefig("mutations_per_cancer_top20.png", dpi=200)

plt.figure(figsize=(9,5))
tumors_per_cancer_df.head(20).iloc[::-1]["unique_tumors"].plot(kind="barh", color="orchid")
plt.title("Unique Tumors per Cancer Type (Top 20)")
plt.xlabel("# Unique Tumor Samples")
plt.tight_layout()
plt.savefig("tumors_per_cancer_top20.png", dpi=200)

print("\n✅ Saved visualizations:")
print("variant_classifications.png")
print("top_genes.png")
print("mutations_per_sample_hist.png")
print("mutations_per_cancer_top20.png")
print("tumors_per_cancer_top20.png")
