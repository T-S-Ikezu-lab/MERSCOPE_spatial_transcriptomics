"""Shared plotting, annotation, region-input, and pseudobulk helpers."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_figure(
    fig, plots_dir: Path, name: str, dpi: int = 150,
    *, svg_matches_png_dpi: bool = True,
) -> Path:
    png_path = plots_dir / f"{name}.png"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(
        plots_dir / f"{name}.svg",
        dpi=dpi if svg_matches_png_dpi else None,
        bbox_inches="tight",
    )
    plt.close(fig)
    return png_path


def clean_numbered_label(value):
    """Remove Allen-style numeric prefixes such as '30 Astro-Epen'."""
    if pd.isna(value):
        return value
    return re.sub(r"^\s*\d+\s+", "", str(value)).strip()


def safe_name(value):
    """Sanitize filenames identically to 05_run_edger.R to locate its outputs."""
    # Must match the R filename sanitizer, including collapsed punctuation runs.
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def add_gene_metadata(adata):
    """Keep suffixed feature names while adding display gene symbol and species."""
    feature_names = pd.Index(adata.var_names.astype(str), name="feature_name")
    species = pd.Series("unknown", index=feature_names, dtype="object")
    species[feature_names.str.endswith("_mouse")] = "mouse"
    species[feature_names.str.endswith("_human")] = "human"

    gene_symbol = feature_names.to_series().str.replace(r"(_mouse|_human)$", "", regex=True)

    adata.var["feature_name"] = feature_names.to_numpy()
    adata.var["gene_symbol"] = gene_symbol.to_numpy()
    adata.var["species"] = species.to_numpy()
    return adata


def keep_mouse_and_human_mapt(var):
    """Return a boolean gene mask: all mouse genes plus human MAPT."""
    species = var["species"] if "species" in var.columns else pd.Series("unknown", index=var.index)
    symbols = var["gene_symbol"] if "gene_symbol" in var.columns else pd.Series(var.index, index=var.index)
    return (species == "mouse") | ((species == "human") & (symbols.str.upper() == "MAPT"))


def add_mmc_hierarchy(adata, mmc, results, mapper_to_adata):
    """Transfer all MapMyCells hierarchy levels into mmc_* obs columns."""
    tree = mmc["taxonomy_tree"]
    hierarchy = tree["hierarchy"]
    adata_to_mapper = {v: k for k, v in mapper_to_adata.items()}

    for level in hierarchy:
        label = tree["hierarchy_mapper"][level].lower().replace(" ", "_")
        name_map = {
            node_id: tree["name_mapper"][level][node_id]["name"]
            for node_id in tree["name_mapper"][level]
        }

        names = []
        corr = []
        boot = []
        for obs_name in adata.obs_names:
            mapper_id = adata_to_mapper.get(obs_name)
            if mapper_id and mapper_id in results and level in results[mapper_id]:
                record = results[mapper_id][level]
                names.append(name_map.get(record["assignment"], "Unknown"))
                corr.append(record.get("avg_correlation", np.nan))
                boot.append(record.get("bootstrapping_probability", np.nan))
            else:
                names.append("Unmapped")
                corr.append(np.nan)
                boot.append(np.nan)

        adata.obs[f"mmc_{label}_name"] = pd.Categorical(names)
        adata.obs[f"mmc_{label}_corr"] = corr
        adata.obs[f"mmc_{label}_boot_prob"] = boot

    return adata


def aggregate_pseudobulk(
    adata,
    mask,
    gene_mask,
    group_map,
    region_name,
    sample_col="sample_id",
    class_col="cell_type",
    covariate_maps=None,
):
    """Aggregate raw counts to sample x cell type pseudobulk rows."""
    if "counts" not in adata.layers:
        raise KeyError("Expected raw counts in adata.layers['counts'].")

    obs = adata.obs.loc[mask, [sample_col, class_col]].copy()
    obs["group"] = obs[sample_col].map(group_map)
    obs = obs[obs["group"].notna()]

    X = adata.layers["counts"]
    if hasattr(X, "tocsr"):
        X = X.tocsr()

    rows = []
    row_ids = []
    meta = []
    obs_positions = np.flatnonzero(mask)
    grouped = obs.groupby([sample_col, "group", class_col], observed=True).indices

    for (sample_id, group, cell_type), rel_idx in grouped.items():
        abs_idx = obs_positions[np.asarray(rel_idx)]
        counts = X[abs_idx][:, gene_mask].sum(axis=0)
        counts = np.asarray(counts).ravel()
        pseudobulk_id = f"{region_name}__{sample_id}__{cell_type}"
        rows.append(counts)
        row_ids.append(pseudobulk_id)
        row_meta = {
            "pseudobulk_id": pseudobulk_id,
            "sample_id": sample_id,
            "group": group,
            "cell_type": cell_type,
            "region": region_name,
            "n_cells": int(len(abs_idx)),
            "library_size": int(counts.sum()),
        }
        for cov_name, cov_map in (covariate_maps or {}).items():
            row_meta[cov_name] = cov_map.get(sample_id)
        meta.append(row_meta)

    counts_df = pd.DataFrame(rows, index=row_ids, columns=adata.var_names[gene_mask])
    meta_df = pd.DataFrame(meta).set_index("pseudobulk_id") if meta else pd.DataFrame()
    return counts_df, meta_df


def configured_polygon_files(cohort, paths):
    """Resolve optional region JSON inputs from ``regions.files``."""
    files = (cohort.get("regions", {}) or {}).get("files", []) or []
    resolved = []
    for configured in files:
        path = paths.resolve(str(configured))
        if not path.exists():
            raise FileNotFoundError(f"Configured region input does not exist: {path}")
        candidates = path.rglob("*.json") if path.is_dir() else [path]
        resolved.extend(candidate for candidate in candidates if candidate.suffix == ".json")
    return sorted(set(resolved))
