#!/usr/bin/env python3
"""
Lookup TCGA cancer type (project) from sample barcodes via the GDC API.

Usage:
  - Hardcode your barcodes in BARCADES below, or
  - run:  python gdc_case_lookup.py TCGA-02-0003-01A-01D-1490-08 TCGA-02-0003-10A-01D-1490-08
"""

import sys
import json
import requests
import pandas as pd

USECOLS = [
    "Hugo_Symbol",
    "Tumor_Sample_Barcode",
    "Variant_Classification",
    "Variant_Type",
    "Reference_Allele",
    "Tumor_Seq_Allele2",
]
CHUNKSIZE = 200_000
GDC_CASES = "https://api.gdc.cancer.gov/cases"
MAF_PATH = "mc3.v0.2.8.PUBLIC.maf"
# Minimal mapping for the "sample-type" code (the two digits at the start of field 4).
SAMPLE_TYPE_MAP = {
    "01": "Primary Solid Tumor",
    "02": "Recurrent Solid Tumor",
    "03": "Primary Blood Derived Cancer - Peripheral Blood",
    "06": "Metastatic",
    "10": "Blood Derived Normal",
    "11": "Solid Tissue Normal",
    # add more if you need them
}

def parse_case_id(barcode: str) -> str:
    """TCGA sample barcode → case/submitter ID (first 3 dash-separated fields)."""
    parts = barcode.split("-")
    if len(parts) < 3:
        raise ValueError(f"Not a valid TCGA barcode: {barcode}")
    return "-".join(parts[:3])

def parse_sample_type(barcode: str) -> str:
    """Extracts the 2-digit sample-type code and maps it to a label."""
    parts = barcode.split("-")
    if len(parts) < 4:
        return "Unknown sample type"
    code = parts[3][:2]
    return f"{code} ({SAMPLE_TYPE_MAP.get(code, 'Unknown')})"

def fetch_case_metadata(case_id: str) -> dict:
    """
    Query GDC Cases for a submitter_id (e.g., 'TCGA-02-0003').
    Returns a dict with key fields or {} if not found.
    """
    filters = {
        "op": "in",
        "content": {"field": "submitter_id", "value": [case_id]},
    }
    fields = ",".join([
        "submitter_id",
        "project.project_id",
        "project.name",
        "primary_site",
        "disease_type",
        "diagnoses.disease_type",
        "diagnoses.primary_diagnosis",
        "diagnoses.tumor_stage",
    ])
    params = {"filters": json.dumps(filters), "fields": fields, "format": "JSON", "size": 1}
    r = requests.get(GDC_CASES, params=params, timeout=20)
    r.raise_for_status()
    hits = r.json().get("data", {}).get("hits", [])
    return hits[0] if hits else {}

def main():
    barcodes = sys.argv[1:] or [
        "TCGA-02-0003-01A-01D-1490-08",
        "TCGA-02-0003-10A-01D-1490-08",
    ]
    reader = pd.read_csv(
    MAF_PATH, sep="\t", comment="#", usecols=USECOLS, chunksize=CHUNKSIZE, low_memory=False
)
    # Group barcodes by case id so we only call the API once per case
    by_case = {}
    for b in reader["Tumor_Sample_Barcode"]:
        cid = parse_case_id(b)
        by_case.setdefault(cid, []).append(b)

    for case_id, bcs in by_case.items():
        meta = fetch_case_metadata(case_id)
        if not meta:
            print(f"\nCase: {case_id}")
            print("  ⚠️  No case found in GDC.")
            for b in bcs:
                print(f"  - {b}  → sample type: {parse_sample_type(b)}")
            continue

        project_id = meta.get("project", {}).get("project_id")
        project_name = meta.get("project", {}).get("name")
        primary_site = meta.get("primary_site")
        disease_type = meta.get("disease_type")
        diagnoses = meta.get("diagnoses", []) or []
        diag_types = sorted({d.get("disease_type") for d in diagnoses if d.get("disease_type")})
        primary_dx = sorted({d.get("primary_diagnosis") for d in diagnoses if d.get("primary_diagnosis")})
        tumor_stages = sorted({d.get("tumor_stage") for d in diagnoses if d.get("tumor_stage")})

        print(f"\nCase: {case_id}")
        print(f"  Project (cancer type): {project_id or 'NA'}  —  {project_name or 'NA'}")
        print(f"  Primary site:         {primary_site or 'NA'}")
        print(f"  Disease type:         {disease_type or (', '.join(diag_types) if diag_types else 'NA')}")
        if primary_dx:
            print(f"  Primary diagnosis:    {', '.join(primary_dx)}")
        if tumor_stages:
            print(f"  Tumor stage(s):       {', '.join(tumor_stages)}")

        for b in bcs:
            print(f"  - {b}  → sample type: {parse_sample_type(b)}")

if __name__ == "__main__":
    main()
