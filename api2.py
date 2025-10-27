#!/usr/bin/env python3
"""
Scan a (large) MAF for tumor barcodes, map them to TCGA case IDs,
and fetch cancer-type metadata from the GDC Cases API.

Usage:
  python gdc_case_lookup_all.py  [--maf mc3.v0.2.8.PUBLIC.maf]
"""

import sys
import json
import math
import argparse
from collections import defaultdict
import requests
import pandas as pd

GDC_CASES = "https://api.gdc.cancer.gov/cases"
CHUNKSIZE = 200_000

SAMPLE_TYPE_MAP = {
    "01": "Primary Solid Tumor",
    "02": "Recurrent Solid Tumor",
    "03": "Primary Blood Derived Cancer - Peripheral Blood",
    "06": "Metastatic",
    "10": "Blood Derived Normal",
    "11": "Solid Tissue Normal",
    # add more if needed
}

def parse_case_id(barcode: str) -> str:
    parts = str(barcode).split("-")
    if len(parts) < 3:
        raise ValueError(f"Not a valid TCGA barcode: {barcode}")
    return "-".join(parts[:3])

def parse_sample_type(barcode: str) -> str:
    parts = str(barcode).split("-")
    if len(parts) < 4:
        return "Unknown sample type"
    code = parts[3][:2]
    return f"{code} ({SAMPLE_TYPE_MAP.get(code, 'Unknown')})"

def chunked(iterable, n):
    """Yield successive n-sized chunks from an iterable/list."""
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

def fetch_case_table(case_ids):
    """
    Batch query GDC Cases for many submitter_ids at once.
    Returns dict: case_id -> metadata dict.
    """
    out = {}
    if not case_ids:
        return out

    # GDC tolerates large 'value' lists, but keep batches moderate for URL size.
    for batch in chunked(case_ids, 100):
        filters = {"op":"in","content":{"field":"submitter_id","value": batch}}
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
        params = {
            "filters": json.dumps(filters),
            "fields": fields,
            "format": "JSON",
            "size": len(batch)
        }
        r = requests.get(GDC_CASES, params=params, timeout=30)
        r.raise_for_status()
        hits = r.json().get("data", {}).get("hits", [])
        for h in hits:
            cid = h.get("submitter_id")
            if cid:
                out[cid] = h
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maf", default="mc3.v0.2.8.PUBLIC.maf", help="Path to MAF (TSV)")
    args = ap.parse_args()

    maf_path = args.maf

    # 1) Scan the MAF in chunks to collect barcodes → case_ids
    by_case = defaultdict(set)  # case_id -> set of tumor barcodes
    # Read only the column we need here to save memory
    usecols = ["Tumor_Sample_Barcode"]
    for chunk in pd.read_csv(
        maf_path,
        sep="\t",
        comment="#",
        usecols=usecols,
        chunksize=CHUNKSIZE,
        low_memory=False
    ):
        # Drop NaNs just in case
        for b in chunk["Tumor_Sample_Barcode"].dropna().astype(str):
            try:
                cid = parse_case_id(b)
                by_case[cid].add(b)
            except Exception:
                # Ignore malformed barcodes
                continue

    case_ids = sorted(by_case.keys())
    if not case_ids:
        print("No TCGA tumor barcodes found in the MAF.")
        return

    # 2) Fetch metadata for all cases in batches
    meta_map = fetch_case_table(case_ids)
    # group by cancer types
    projects = set()

    # 3) Print a readable summary
    for case_id in case_ids:
        meta = meta_map.get(case_id, {})
        if not meta:
            print(f"\nCase: {case_id}")
            print("  ⚠️  No case found in GDC.")
            for b in sorted(by_case[case_id]):
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

        for b in sorted(by_case[case_id]):
            print(f"  - {b}  → sample type: {parse_sample_type(b)}")
        projects.add(project_name)
    print(projects)

if __name__ == "__main__":
    main()
