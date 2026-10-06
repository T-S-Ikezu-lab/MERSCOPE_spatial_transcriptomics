#!/usr/bin/env python3
"""Import MapMyCells hierarchy assignments and optional coarse annotations."""

from __future__ import annotations

import argparse
import html
import json
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from config_utils import cohort_paths, load_cohort
from pipeline_utils import add_gene_metadata, add_mmc_hierarchy, clean_numbered_label, save_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import MapMyCells JSON labels into the normalized AnnData."
    )
    parser.add_argument("--config", required=True, type=Path, help="Cohort YAML config")
    return parser.parse_args()


def mapper_json_path(output_dir: Path, settings: dict, paths) -> Path:
    configured_path = settings.get("mapper_json_path")
    if configured_path:
        path = paths.resolve(configured_path)
        if not path.exists():
            raise FileNotFoundError(f"Configured MapMyCells JSON does not exist: {path}")
        return path

    mapper_dir = output_dir / "mapmycells"
    configured = settings.get("mapper_json_filename")
    if configured:
        path = mapper_dir / configured
        if not path.exists():
            raise FileNotFoundError(f"Configured MapMyCells JSON does not exist: {path}")
        return path
    candidates = sorted(mapper_dir.glob("*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No MapMyCells JSON found in {mapper_dir}. Set "
            "label_transfer.mapper_json_filename explicitly or place the result there."
        )
    return candidates[0]


def match_cell_ids(adata, results: dict) -> dict[str, str]:
    mapper_ids = set(results)
    overlap = mapper_ids.intersection(adata.obs_names)
    if overlap:
        print(f"Direct cell-ID overlap: {len(overlap):,} / {adata.n_obs:,}")
        return {cell_id: cell_id for cell_id in overlap}

    stripped = {}
    sample_ids = adata.obs["sample_id"].astype(str).unique()
    for obs_name in adata.obs_names:
        for sample_id in sample_ids:
            prefix = f"{sample_id}_"
            if obs_name.startswith(prefix):
                stripped[obs_name[len(prefix) :]] = obs_name
                break
        else:
            stripped[obs_name] = obs_name
    overlap = mapper_ids.intersection(stripped)
    if not overlap:
        raise ValueError(
            "No MapMyCells cell IDs match, either directly or after stripping "
            "sample prefixes."
        )
    print(f"Cell-ID overlap after stripping sample prefixes: {len(overlap):,}")
    return {cell_id: stripped[cell_id] for cell_id in overlap}


def transfer_class_labels(adata, mmc: dict, results: dict, mapper_to_adata: dict) -> None:
    tree = mmc["taxonomy_tree"]
    class_level = tree["hierarchy"][0]
    class_names = {
        node_id: metadata["name"]
        for node_id, metadata in tree["name_mapper"][class_level].items()
    }
    adata_to_mapper = {adata_id: mapper_id for mapper_id, adata_id in mapper_to_adata.items()}

    names, correlations, probabilities = [], [], []
    for obs_name in adata.obs_names:
        mapper_id = adata_to_mapper.get(obs_name)
        record = results.get(mapper_id, {}).get(class_level) if mapper_id else None
        if record:
            names.append(class_names.get(record["assignment"], "Unknown"))
            correlations.append(record["avg_correlation"])
            probabilities.append(record["bootstrapping_probability"])
        else:
            names.append("Unmapped")
            correlations.append(np.nan)
            probabilities.append(np.nan)
    adata.obs["class_name"] = pd.Categorical(names)
    adata.obs["class_avg_corr"] = correlations
    adata.obs["class_boot_prob"] = probabilities
    adata.obs["class_name_clean"] = (
        adata.obs["class_name"].astype("object").map(clean_numbered_label).astype("category")
    )
    add_mmc_hierarchy(adata, mmc, results, mapper_to_adata)


def assign_direct_coarse_labels(adata, workbook: Path) -> None:
    mapping = pd.read_excel(workbook)
    required = {"subclass", "class-APOE-cluster"}
    missing = required.difference(mapping.columns)
    if missing:
        raise ValueError(f"{workbook} is missing columns: {sorted(missing)}")
    lookup = dict(zip(mapping["subclass"], mapping["class-APOE-cluster"]))
    labels = adata.obs["mmc_subclass_name"].astype("object").map(lookup).fillna("other")
    adata.obs["cell_type"] = labels


def assign_hierarchical_coarse_labels(adata, workbook: Path) -> None:
    mapping = pd.read_excel(workbook)
    required = {"cell_group", "neurotransmitter", "class", "subclass"}
    missing = required.difference(mapping.columns)
    if missing:
        raise ValueError(f"{workbook} is missing columns: {sorted(missing)}")

    neuronal = mapping["cell_group"].str.contains("neuron", case=False, na=False)
    coarse = mapping["class"].astype("object").copy()
    coarse.loc[neuronal] = mapping.loc[neuronal, "neurotransmitter"]
    subclass_lookup = dict(zip(mapping["subclass"], coarse))
    class_lookup = dict(zip(mapping["class"], coarse))
    labels = adata.obs["mmc_subclass_name"].astype("object").map(subclass_lookup)
    labels = labels.fillna(adata.obs["class_name"].astype("object").map(class_lookup))
    labels = labels.fillna("Unclassified").map(clean_numbered_label)
    adata.obs["cell_type"] = pd.Categorical(labels)


def write_report(
    output_dir: Path,
    plots_dir: Path,
    cohort: dict,
    adata,
    threshold: float,
    mapper_path: Path,
    coarse_method: str,
) -> None:
    confidence = adata.obs["label_confidence"].value_counts()
    if "cell_type" in adata.obs:
        rows = "".join(
            f"<tr><td>{html.escape(str(label))}</td><td>{count:,}</td></tr>"
            for label, count in adata.obs["cell_type"].value_counts().items()
        )
        coarse_section = (
            "<h2>Coarse label counts</h2><table><thead><tr><th>Label</th>"
            f"<th>Cells</th></tr></thead><tbody>{rows}</tbody></table>"
        )
    else:
        coarse_section = "<h2>Coarse labels</h2><p>Not configured.</p>"
    svg = (plots_dir / "confidence_filtering.svg").read_text(encoding="utf-8")
    report = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Annotation evaluation</title><style>
body{{font:15px/1.45 sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem}}
table{{border-collapse:collapse}}th,td{{border:1px solid #bbb;padding:.35rem .6rem}}
svg{{max-width:100%;height:auto}}code{{background:#eee;padding:.15rem}}</style></head><body>
<h1>{html.escape(str(cohort.get("cohort_name", "Cohort")))} annotation evaluation</h1>
<p>Generated {datetime.now().isoformat(timespec="seconds")}. Source:
<code>{html.escape(mapper_path.name)}</code>.</p>
<ul><li>{adata.n_obs:,} cells; {adata.n_vars:,} features</li>
<li>Confidence threshold: {threshold:g}</li>
<li>High: {int(confidence.get("high", 0)):,}; low: {int(confidence.get("low", 0)):,};
unmapped: {int(confidence.get("unmapped", 0)):,}</li>
<li>Coarse mapping method: {html.escape(coarse_method)}</li></ul>
<h2>Confidence diagnostics</h2>{svg}
{coarse_section}</body></html>"""
    (output_dir / "annotation_report.html").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    cohort = load_cohort(args.config)
    paths = cohort_paths(cohort)
    output_dir = paths.output("04a_import_annotations")
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    input_path = paths.output("03_normalize_dimred", "combined_norm_dimred.h5ad")
    settings = cohort.get("label_transfer", {}) or {}
    mapper_path = mapper_json_path(output_dir, settings, paths)
    threshold = float(settings.get("confidence_threshold", 0.4))

    coarse_method = str(settings.get("coarse_mapping_method", "none")).lower()
    coarse_path = None
    if coarse_method != "none":
        coarse_name = settings.get("coarse_mapping_xlsx")
        if not coarse_name:
            raise KeyError(
                "label_transfer.coarse_mapping_xlsx is required when "
                "coarse_mapping_method is not 'none'."
            )
        coarse_path = paths.external_dir / coarse_name

    mapper_copy_dir = output_dir / "mapmycells"
    mapper_copy_dir.mkdir(parents=True, exist_ok=True)
    mapper_copy = mapper_copy_dir / mapper_path.name
    if mapper_path.resolve() != mapper_copy.resolve():
        shutil.copy2(mapper_path, mapper_copy)
    if coarse_path is not None:
        class_mapping_dir = output_dir / "class_mapping"
        class_mapping_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(coarse_path, class_mapping_dir / coarse_path.name)

    print(f"Input: {input_path}")
    print(f"MapMyCells JSON: {mapper_path}")
    print(
        f"Coarse mapping: {coarse_path}"
        if coarse_method != "none"
        else "Coarse mapping: disabled"
    )
    adata = sc.read_h5ad(input_path)
    add_gene_metadata(adata)
    with mapper_path.open(encoding="utf-8") as handle:
        mmc = json.load(handle)
    results = {record["cell_id"]: record for record in mmc["results"]}
    mapper_to_adata = match_cell_ids(adata, results)
    transfer_class_labels(adata, mmc, results, mapper_to_adata)

    mapped = adata.obs["class_avg_corr"].notna()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].hist(adata.obs.loc[mapped, "class_avg_corr"], bins=50, edgecolor="black")
    axes[0].axvline(threshold, color="red", linestyle="--", label=f"Threshold: {threshold:g}")
    axes[0].set(title="Class: average correlation", xlabel="Average correlation")
    axes[0].legend()
    axes[1].hist(adata.obs.loc[mapped, "class_boot_prob"], bins=50, edgecolor="black")
    axes[1].set(title="Class: bootstrap probability", xlabel="Bootstrap probability")
    for class_name in adata.obs.loc[mapped, "class_name"].unique():
        mask = mapped & (adata.obs["class_name"] == class_name)
        axes[2].scatter(
            adata.obs.loc[mask, "class_avg_corr"],
            adata.obs.loc[mask, "class_boot_prob"],
            s=1,
            alpha=0.3,
            label=class_name,
        )
    axes[2].axvline(threshold, color="red", linestyle="--")
    axes[2].set(
        title="Correlation vs bootstrap",
        xlabel="Average correlation",
        ylabel="Bootstrap probability",
    )
    axes[2].legend(markerscale=5, fontsize=6, bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    save_figure(fig, plots_dir, "confidence_filtering")

    low = adata.obs["class_avg_corr"] < threshold
    unmapped = adata.obs["class_avg_corr"].isna()
    adata.obs["label_confidence"] = "high"
    adata.obs.loc[low, "label_confidence"] = "low"
    adata.obs.loc[unmapped, "label_confidence"] = "unmapped"

    if coarse_method == "direct":
        assign_direct_coarse_labels(adata, coarse_path)
    elif coarse_method == "hierarchical":
        assign_hierarchical_coarse_labels(adata, coarse_path)
    elif coarse_method == "none":
        pass
    else:
        raise ValueError(
            "label_transfer.coarse_mapping_method must be direct, hierarchical, or none"
        )

    if "lognorm" in adata.layers:
        adata.X = adata.layers["lognorm"].copy()
    output_path = output_dir / "combined_annotated.h5ad"
    adata.write_h5ad(output_path)
    write_report(
        output_dir, plots_dir, cohort, adata, threshold, mapper_path, coarse_method
    )
    print(f"Saved {output_path}")
    print(f"Saved {output_dir / 'annotation_report.html'}")


if __name__ == "__main__":
    main()
