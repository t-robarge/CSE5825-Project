import pandas as pd

# clinical file
clin = pd.read_csv("clinical_PANCAN_patient_with_followup.tsv", sep="\t",
                   encoding="latin1")
# tumor x mutation matrix
mut_matrix = pd.read_csv("sbs96_by_tumor.csv", index_col=0)
# keep only what we need
clin_small = clin[["bcr_patient_barcode", "acronym"]].drop_duplicates()
# bcr_patient_barcode: e.g. TCGA-OR-A5J1
# acronym: e.g. ACC, BRCA, LUAD, ...

# map sample IDs to cancer types
patient_to_acronym = dict(
    zip(clin_small["bcr_patient_barcode"], clin_small["acronym"])
)
# copy matrix to avoid modifying in place
annotated_matrix = mut_matrix.copy()
annotated_matrix["Cancer_Type"] = annotated_matrix.index.str.slice(0, 12).map(patient_to_acronym)

# save annotated matrix
annotated_matrix.to_csv("sbs96_by_tumor_with_cancer_type.csv")

#print cancer types