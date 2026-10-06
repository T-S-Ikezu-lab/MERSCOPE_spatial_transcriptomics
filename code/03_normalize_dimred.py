#!/usr/bin/env python3
"""Normalize the combined cohort and fit its canonical dimensional reduction."""

from __future__ import annotations

import argparse
import html
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config_utils import cohort_paths, load_cohort
from pipeline_utils import save_figure


RESOLUTION_SWEEP = [0.3, 0.5, 0.8, 1.0, 1.5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize combined cells and run HVG, PCA, neighbors, UMAP, and Leiden."
    )
    parser.add_argument("--config", required=True, type=Path, help="Cohort YAML config")
    return parser.parse_args()


def inline_svg(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def group_order_from_config(cohort: dict) -> list[str]:
    """Reproduce the reporting workflow's configured comparison order."""
    ordered = []
    for comparison in cohort.get("comparisons", []) or []:
        if isinstance(comparison, (list, tuple)) and len(comparison) == 2:
            ordered.extend(str(group) for group in comparison)
    ordered = list(dict.fromkeys(ordered))
    if ordered:
        valid = set((cohort.get("group_map", {}) or {}).values())
        return [group for group in ordered if group in valid]
    return sorted({str(group) for group in (cohort.get("group_map", {}) or {}).values()})


def write_group_refit_umap(adata, cohort: dict, output_path: Path) -> None:
    """Fit report-only UMAP coordinates independently within each group."""
    max_cells_per_group = 120_000
    rng = np.random.default_rng(42)
    group_map = cohort.get("group_map", {}) or {}
    group_values = adata.obs["sample_id"].map(group_map).fillna("unknown").to_numpy()
    groups = [
        group
        for group in group_order_from_config(cohort)
        if np.any(group_values == group)
    ]
    if not groups:
        raise ValueError("No configured groups are represented in adata.obs['sample_id'].")

    frames = []
    for group in groups:
        group_idx = np.flatnonzero(group_values == group)
        if len(group_idx) > max_cells_per_group:
            keep_idx = rng.choice(
                group_idx, size=max_cells_per_group, replace=False
            )
        else:
            keep_idx = group_idx

        group_adata = adata[keep_idx].copy()
        if "X_pca" in group_adata.obsm and group_adata.obsm["X_pca"].shape[1] >= 2:
            use_pcs = min(30, group_adata.obsm["X_pca"].shape[1])
            n_neighbors = min(15, max(2, group_adata.n_obs - 1))
            sc.pp.neighbors(
                group_adata,
                use_rep="X_pca",
                n_pcs=use_pcs,
                n_neighbors=n_neighbors,
                random_state=0,
            )
        else:
            if "lognorm" in group_adata.layers:
                group_adata.X = group_adata.layers["lognorm"].copy()
            sc.pp.pca(
                group_adata,
                n_comps=min(30, group_adata.n_obs - 1, group_adata.n_vars - 1),
            )
            use_pcs = min(30, group_adata.obsm["X_pca"].shape[1])
            n_neighbors = min(15, max(2, group_adata.n_obs - 1))
            sc.pp.neighbors(
                group_adata,
                n_pcs=use_pcs,
                n_neighbors=n_neighbors,
                random_state=0,
            )
        sc.tl.umap(group_adata, random_state=0)
        frames.append(
            pd.DataFrame(
                {
                    "cell_id": group_adata.obs_names,
                    "UMAP1": group_adata.obsm["X_umap"][:, 0],
                    "UMAP2": group_adata.obsm["X_umap"][:, 1],
                    "group": group,
                }
            )
        )
        print(f"Computed separately fitted group UMAP: {group} (n={group_adata.n_obs:,})")

    pd.concat(frames, ignore_index=True)[
        ["cell_id", "UMAP1", "UMAP2", "group"]
    ].to_csv(output_path, index=False, compression="gzip")
    print(f"Saved report-only group UMAP coordinates: {output_path}")


def write_report(
    output_dir: Path,
    plots_dir: Path,
    cohort: dict,
    adata,
    n_hvg: int,
    n_pcs: int,
    n_neighbors: int,
    resolution: float,
    norm_method: str,
) -> None:
    group_map = cohort.get("group_map", {})
    counts = adata.obs["sample_id"].value_counts()
    rows = []
    for sample_id in sorted(adata.obs["sample_id"].astype(str).unique()):
        mask = adata.obs["sample_id"].astype(str) == sample_id
        rows.append(
            "<tr>"
            f"<td>{html.escape(sample_id)}</td>"
            f"<td>{html.escape(str(group_map.get(sample_id, 'unknown')))}</td>"
            f"<td>{int(counts[sample_id]):,}</td>"
            f"<td>{100 * counts[sample_id] / adata.n_obs:.2f}%</td>"
            f"<td>{adata.obs.loc[mask, 'n_transcripts'].median():.1f}</td>"
            "</tr>"
        )

    plot_names = [
        "per_sample_comparison",
        "normalization_distributions",
        "pca_variance_ratio",
        "pca_by_sample",
        "pseudobulk_pca_by_treatment",
        "pseudobulk_pca_by_experiment",
        "leiden_resolution_sweep",
        "umap_overview",
    ]
    figures = "\n".join(
        f"<section><h2>{html.escape(name.replace('_', ' ').title())}</h2>"
        f"{inline_svg(plots_dir / f'{name}.svg')}</section>"
        for name in plot_names
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Normalization evaluation</title>
<style>
body{{font:15px/1.45 sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#222}}
table{{border-collapse:collapse}}th,td{{border:1px solid #bbb;padding:.35rem .6rem;text-align:right}}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}
svg{{max-width:100%;height:auto}}section{{margin:2rem 0}}code{{background:#eee;padding:.15rem}}
</style></head><body>
<h1>{html.escape(str(cohort.get("cohort_name", "Cohort")))}</h1>
<p>Generated {datetime.now().isoformat(timespec="seconds")}. Canonical checkpoint:
<code>03_normalize_dimred/combined_norm_dimred.h5ad</code>.</p>
<ul><li>{adata.n_obs:,} cells; {adata.n_vars:,} features; {n_hvg:,} HVGs</li>
<li>Normalization: {html.escape(norm_method)}</li>
<li>{n_pcs} PCs; {n_neighbors} neighbors; Leiden resolution {resolution:g};
{adata.obs["leiden"].nunique()} clusters</li></ul>
<h2>Samples</h2><table><thead><tr><th>Sample</th><th>Group</th><th>Cells</th>
<th>Cohort</th><th>Median transcripts</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
{figures}</body></html>"""
    (output_dir / "normalization_report.html").write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    cohort = load_cohort(args.config)
    paths = cohort_paths(cohort)
    input_path = paths.output("02_combine_filtered", "all_samples_filtered.h5ad")
    output_dir = paths.output("03_normalize_dimred")
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    settings = (cohort.get("reporting", {}) or {}).get("visualizations", {}) or {}
    norm_method = settings.get("normalization_method", "Count norm (mean) + log1p")
    if norm_method != "Count norm (mean) + log1p":
        raise ValueError(
            "Only the normalization_method 'Count norm (mean) + log1p' "
            "is supported."
        )
    n_pcs = int(settings.get("n_pcs", 20))
    n_neighbors = int(settings.get("n_neighbors", 15))
    resolution = float(settings.get("leiden_resolution", 0.5))
    pseudobulk_top_genes = int(settings.get("pseudobulk_top_genes", 500))

    sc.settings.verbosity = 3
    sc.settings.set_figure_params(dpi=100, frameon=False)
    sns.set_style("whitegrid")

    print(f"Input:  {input_path}")
    print(f"Output: {output_dir}")
    adata_raw = sc.read_h5ad(input_path)
    adata_raw.layers["counts"] = adata_raw.X.copy()
    adata_raw.obs["n_transcripts"] = np.asarray(
        adata_raw.X.sum(axis=1)
    ).ravel()
    sample_ids = adata_raw.obs["sample_id"].unique().tolist()
    print(f"Loaded {adata_raw.n_obs:,} cells x {adata_raw.n_vars:,} genes")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sample_counts = adata_raw.obs["sample_id"].value_counts().reindex(sample_ids)
    axes[0].bar(range(len(sample_ids)), sample_counts.values, color="steelblue")
    axes[0].set_xticks(range(len(sample_ids)), sample_ids, rotation=45, ha="right")
    axes[0].set_ylabel("Number of cells")
    axes[0].set_title("Cell counts per sample")
    violin_data = [
        adata_raw.obs.loc[adata_raw.obs["sample_id"] == sid, "n_transcripts"]
        for sid in sample_ids
    ]
    axes[1].violinplot(violin_data, positions=range(len(sample_ids)), showmedians=True)
    axes[1].set_yscale("log")
    axes[1].set_xticks(range(len(sample_ids)), sample_ids, rotation=45, ha="right")
    axes[1].set_ylabel("Transcripts per cell")
    axes[1].set_title("Transcripts per cell by sample")
    fig.suptitle("Per-sample comparison after filtering")
    fig.tight_layout()
    save_figure(fig, plots_dir, "per_sample_comparison")

    mean_total = np.asarray(adata_raw.X.sum(axis=1)).ravel().mean()
    adata = adata_raw.copy()
    sc.pp.normalize_total(adata, target_sum=mean_total)
    sc.pp.log1p(adata)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for row, (matrix, title, color) in enumerate(
        [(adata_raw.layers["counts"], "Raw counts", "C0"), (adata.X, norm_method, "C1")]
    ):
        dense = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
        axes[row, 0].hist(dense.sum(axis=1), bins=80, color=color, alpha=0.8)
        axes[row, 0].set_title(f"{title} -- total per cell")
        nonzero = dense.ravel()
        nonzero = nonzero[nonzero > 0]
        if len(nonzero) > 500_000:
            nonzero = np.random.choice(nonzero, 500_000, replace=False)
        axes[row, 1].hist(nonzero, bins=100, color=color, alpha=0.8)
        axes[row, 1].set_title(f"{title} -- nonzero expression")
    fig.tight_layout()
    save_figure(fig, plots_dir, "normalization_distributions")

    adata.X = adata.layers["counts"].copy()
    sc.pp.highly_variable_genes(
        adata, n_top_genes=500, batch_key="sample_id", flavor="seurat_v3"
    )
    n_hvg = int(adata.var["highly_variable"].sum())

    adata.X = adata.layers["counts"].copy()
    sc.pp.normalize_total(adata, target_sum=mean_total)
    sc.pp.log1p(adata)
    adata.layers["lognorm"] = adata.X.copy()
    sc.pp.scale(adata, max_value=10)
    sc.pp.pca(adata, n_comps=50, use_highly_variable=True)

    sc.pl.pca_variance_ratio(adata, n_pcs=50, show=False, log=True)
    save_figure(plt.gcf(), plots_dir, "pca_variance_ratio")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sc.pl.pca(adata, color="sample_id", components="1,2", ax=axes[0], show=False)
    sc.pl.pca(adata, color="sample_id", components="3,4", ax=axes[1], show=False)
    fig.tight_layout()
    save_figure(fig, plots_dir, "pca_by_sample")

    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.umap(adata)
    sweep = list(RESOLUTION_SWEEP)
    if resolution not in sweep:
        sweep.append(resolution)
    for value in sweep:
        sc.tl.leiden(
            adata,
            resolution=value,
            key_added=f"leiden_{value}",
            flavor="igraph",
            n_iterations=2,
            directed=False,
        )
        print(f"Leiden {value:g}: {adata.obs[f'leiden_{value}'].nunique()} clusters")

    fig, axes = plt.subplots(1, len(RESOLUTION_SWEEP), figsize=(30, 5))
    for ax, value in zip(np.atleast_1d(axes), RESOLUTION_SWEEP):
        key = f"leiden_{value}"
        sc.pl.umap(
            adata,
            color=key,
            ax=ax,
            show=False,
            title=f"res={value} ({adata.obs[key].nunique()} clusters)",
            legend_loc="on data",
            legend_fontsize=5,
        )
    fig.suptitle("Leiden resolution sweep")
    fig.tight_layout()
    save_figure(fig, plots_dir, "leiden_resolution_sweep")

    selected_key = f"leiden_{resolution}"
    adata.obs["leiden"] = adata.obs[selected_key].copy()
    for value in sweep:
        if value != resolution:
            del adata.obs[f"leiden_{value}"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    sc.pl.umap(adata, color="leiden", ax=axes[0, 0], show=False, legend_loc="on data")
    sc.pl.umap(adata, color="sample_id", ax=axes[0, 1], show=False)
    sc.pl.umap(adata, color="n_transcripts", ax=axes[1, 0], show=False)
    sc.pl.umap(adata, color="n_genes_by_counts", ax=axes[1, 1], show=False)
    fig.suptitle(
        f"UMAP overview -- {norm_method}; {n_pcs} PCs, {n_neighbors} neighbors, "
        f"Leiden {resolution:g}"
    )
    fig.tight_layout()
    save_figure(fig, plots_dir, "umap_overview")

    counts = adata.layers["counts"]
    counts = counts.tocsr() if hasattr(counts, "tocsr") else counts
    pb_samples = pd.Index(sorted(adata.obs["sample_id"].unique()))
    pb = np.zeros((len(pb_samples), adata.n_vars), dtype=np.float64)
    sample_values = adata.obs["sample_id"].values
    for index, sample_id in enumerate(pb_samples):
        pb[index] = np.asarray(counts[np.flatnonzero(sample_values == sample_id)].sum(axis=0)).ravel()
    pb = pd.DataFrame(pb, index=pb_samples, columns=adata.var_names)
    logcpm = np.log1p(pb.div(pb.sum(axis=1), axis=0) * 1e6)
    top_genes = logcpm.var().nlargest(min(pseudobulk_top_genes, adata.n_vars)).index
    pb_scaled = StandardScaler().fit_transform(logcpm[top_genes])
    pb_pca = PCA(n_components=min(10, len(pb_samples) - 1, len(top_genes))).fit(pb_scaled)
    scores = pb_pca.transform(pb_scaled)
    variance = pb_pca.explained_variance_ratio_ * 100
    group_map = cohort.get("group_map", {}) or {}
    experiment_map = (
        (cohort.get("covariates", {}) or {}).get("experiment", {}) or {}
    )
    pb_labels = {
        "treatment": [group_map.get(sample, "unknown") for sample in pb_samples],
        "experiment": [experiment_map.get(sample, "unknown") for sample in pb_samples],
    }
    for label_name, labels in pb_labels.items():
        fig, ax = plt.subplots(figsize=(8, 7))
        for label in sorted(set(labels)):
            mask = np.asarray(labels) == label
            ax.scatter(
                scores[mask, 0], scores[mask, 1], s=140, label=label,
                edgecolor="black",
            )
        for sample, x, y in zip(pb_samples, scores[:, 0], scores[:, 1]):
            ax.annotate(
                sample, (x, y), fontsize=7, xytext=(4, 4),
                textcoords="offset points",
            )
        ax.set(
            xlabel=f"PC1 ({variance[0]:.1f}%)",
            ylabel=f"PC2 ({variance[1]:.1f}%)",
        )
        ax.set_title(f"Sample-level pseudobulk PCA (by {label_name})")
        ax.legend(title=label_name)
        fig.tight_layout()
        save_figure(fig, plots_dir, f"pseudobulk_pca_by_{label_name}")

    adata.X = adata.layers["lognorm"].copy()
    output_path = output_dir / "combined_norm_dimred.h5ad"
    adata.write_h5ad(output_path)
    write_group_refit_umap(adata, cohort, output_dir / "group_refit_umap.csv.gz")
    write_report(
        output_dir, plots_dir, cohort, adata, n_hvg, n_pcs, n_neighbors,
        resolution, norm_method
    )
    print(f"Saved {output_path}")
    print(f"Saved {output_dir / 'normalization_report.html'}")


if __name__ == "__main__":
    main()
