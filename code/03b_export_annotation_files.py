#!/usr/bin/env python3
"""Export the normalized counts for MapMyCells annotation."""

from __future__ import annotations

import argparse
import html
from datetime import datetime
from pathlib import Path


import anndata
import pandas as pd
import scanpy as sc

from config_utils import cohort_paths, load_cohort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a MapMyCells submission H5AD.")
    parser.add_argument("--config", required=True, type=Path, help="Cohort YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cohort = load_cohort(args.config)
    paths = cohort_paths(cohort)
    input_path = paths.output("03_normalize_dimred", "combined_norm_dimred.h5ad")
    export_dir = paths.output("04a_import_annotations", "mapmycells")
    export_dir.mkdir(parents=True, exist_ok=True)
    output_path = export_dir / "mapper_input_gzip2.h5ad"

    adata = sc.read_h5ad(input_path)
    if "counts" not in adata.layers:
        raise KeyError("Expected raw counts in adata.layers['counts'].")

    source_names = pd.Index(adata.var_names.astype(str))
    stripped_names = source_names.str.replace(r"(_mouse|_human)$", "", regex=True)
    mapper_adata = anndata.AnnData(
        X=adata.layers["counts"],
        obs=adata.obs[[]].copy(),
        var=adata.var[[]].copy(),
    )
    mapper_adata.var_names = stripped_names
    mapper_adata.var_names_make_unique()

    audit = pd.DataFrame(
        {
            "source_feature_name": source_names,
            "suffix_stripped_name": stripped_names,
            "exported_feature_name": mapper_adata.var_names.astype(str),
        }
    )
    audit["renamed_for_uniqueness"] = (
        audit["suffix_stripped_name"] != audit["exported_feature_name"]
    )
    audit_path = export_dir / "feature_mapping_audit.csv"
    audit.to_csv(audit_path, index=False)
    mapper_adata.write_h5ad(output_path, compression="gzip", compression_opts=4)

    summary = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Annotation export summary</title><style>
body{{font:15px/1.5 sans-serif;max-width:850px;margin:2rem auto;padding:0 1rem}}
code{{background:#eee;padding:.15rem}}</style></head><body>
<h1>{html.escape(str(cohort.get("cohort_name", "Cohort")))} annotation export</h1>
<p>Generated {datetime.now().isoformat(timespec="seconds")}.</p>
<ul><li>{mapper_adata.n_obs:,} cells</li><li>{mapper_adata.n_vars:,} features</li>
<li>{int(audit["renamed_for_uniqueness"].sum()):,} features renamed for uniqueness</li></ul>
<p>Submission: <code>{html.escape(output_path.name)}</code><br>
Feature audit: <code>{html.escape(audit_path.name)}</code></p></body></html>"""
    report_path = export_dir / "export_summary.html"
    report_path.write_text(summary, encoding="utf-8")

    print(f"Saved MapMyCells input: {output_path}")
    print(f"Saved feature mapping audit: {audit_path}")
    print(f"Saved export summary: {report_path}")


if __name__ == "__main__":
    main()
