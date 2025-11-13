import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

maf = pd.read_csv("mc3.v0.2.8.PUBLIC.maf", sep="\t", comment="#", low_memory=False)

# QC + VAFs
for c in ["t_depth","t_alt_count","n_depth","n_alt_count"]:
    maf[c] = pd.to_numeric(maf[c], errors="coerce")

maf = maf.query("FILTER == 'PASS'").copy()
maf["tumor_vaf"]  = maf["t_alt_count"] / maf["t_depth"]
maf["normal_vaf"] = maf["n_alt_count"] / maf["n_depth"]
maf_qc = maf[(maf["t_depth"] >= 30) & np.isfinite(maf["tumor_vaf"])]

# 1) VAF histogram
plt.figure()
plt.hist(maf_qc["tumor_vaf"], bins=50, density=True)
for x in [0.25, 0.5, 0.75]:
    plt.axvline(x, linestyle="--")
plt.xlabel("Tumor VAF (t_alt / t_depth)")
plt.ylabel("Density")
plt.title("Tumor VAF distribution (PASS, depth≥30)")
plt.show()

# 2) VAF vs depth scatter
plt.figure()
plt.scatter(maf_qc["t_depth"], maf_qc["tumor_vaf"], s=4, alpha=0.5)
plt.xscale("log")
plt.xlabel("Tumor depth (log scale)")
plt.ylabel("Tumor VAF")
plt.title("VAF vs Depth")
plt.show()

# 3) VAF by consequence (top categories)
top_classes = maf_qc["Variant_Classification"].value_counts().head(6).index
sub = maf_qc[maf_qc["Variant_Classification"].isin(top_classes)]
# quick boxplot
plt.figure()
sub.boxplot(column="tumor_vaf", by="Variant_Classification", rot=45)
plt.ylabel("Tumor VAF")
plt.title("VAF by variant classification")
plt.suptitle("")
plt.savefig("vaf_by_consequence.png")
