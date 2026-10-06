#!/usr/bin/env python3
"""Select polygon ROIs, persist cell membership, and generate region QC."""

from __future__ import annotations

import argparse
import html
import json
import shutil
from datetime import datetime
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.path import Path as MplPath

from config_utils import cohort_paths, load_cohort
from pipeline_utils import configured_polygon_files, save_figure as shared_save_figure


# ROI SVG raster layers must retain Matplotlib's default save DPI.
save_figure = partial(shared_save_figure, svg_matches_png_dpi=False)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select cells inside configured ROI polygons and generate region QC."
    )
    parser.add_argument("--config", required=True, type=Path, help="Cohort YAML config")
    parser.add_argument(
        "--cell-type-level",
        default="coarse",
        help="'coarse' uses obs['cell_type']; any other value names an obs column.",
    )
    return parser.parse_args(argv)


def safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    )


def load_polygons(
    polygon_files: list[Path], known_samples: set[str]
) -> dict[str, tuple[Path, np.ndarray]]:
    """Load exactly one valid polygon JSON per represented sample."""
    polygons: dict[str, tuple[Path, np.ndarray]] = {}
    for polygon_file in polygon_files:
        try:
            payload = json.loads(polygon_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not read ROI JSON {polygon_file}: {error}") from error
        if not isinstance(payload, dict):
            raise ValueError(f"ROI JSON root must be an object: {polygon_file}")

        sample_id = str(payload.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError(f"ROI JSON is missing sample_id: {polygon_file}")
        if sample_id not in known_samples:
            raise ValueError(
                f"ROI JSON {polygon_file} references unknown sample_id {sample_id!r}"
            )
        if sample_id in polygons:
            previous = polygons[sample_id][0]
            raise ValueError(
                "Multiple ROI JSON files were found for sample "
                f"{sample_id!r}; expected exactly one canonical file:\n"
                f"  - {previous}\n  - {polygon_file}"
            )

        try:
            vertices = np.asarray(payload.get("vertices", []), dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"ROI vertices must be numeric [x, y] pairs: {polygon_file}"
            ) from error
        if vertices.ndim != 2 or vertices.shape[1:] != (2,) or len(vertices) < 3:
            raise ValueError(
                f"ROI JSON must contain at least three [x, y] vertices: {polygon_file}"
            )
        if not np.isfinite(vertices).all():
            raise ValueError(f"ROI vertices contain nonfinite coordinates: {polygon_file}")
        polygons[sample_id] = (polygon_file, vertices)
    return polygons


def build_roi_mask(adata, polygons: dict[str, tuple[Path, np.ndarray]]) -> np.ndarray:
    if "spatial" not in adata.obsm:
        raise KeyError("Expected spatial coordinates in adata.obsm['spatial'].")
    if "sample_id" not in adata.obs:
        raise KeyError("Expected sample IDs in adata.obs['sample_id'].")

    mask = np.zeros(adata.n_obs, dtype=bool)
    sample_values = adata.obs["sample_id"].astype(str).to_numpy()
    coordinates = np.asarray(adata.obsm["spatial"])
    for sample_id, (_, vertices) in polygons.items():
        sample_positions = np.flatnonzero(sample_values == sample_id)
        mask[sample_positions] = MplPath(vertices).contains_points(
            coordinates[sample_positions]
        )
    return mask


def copy_polygon_inputs(
    polygons: dict[str, tuple[Path, np.ndarray]], destination: Path
) -> None:
    for sample_id, (source, _) in polygons.items():
        sample_dir = destination / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, sample_dir / source.name)


def composition_tables(
    adata,
    samples: list[str],
    class_col: str,
    roi_mask: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample_values = adata.obs["sample_id"].astype(str).to_numpy()
    class_values = (
        adata.obs[class_col].astype("object").fillna("Unclassified").astype(str)
    )
    labels = sorted(class_values.unique())
    roi_counts = pd.DataFrame(0, index=samples, columns=labels, dtype=int)
    whole_counts = pd.DataFrame(0, index=samples, columns=labels, dtype=int)

    for sample_id in samples:
        sample_mask = sample_values == sample_id
        whole = class_values.iloc[np.flatnonzero(sample_mask)].value_counts()
        roi = class_values.iloc[np.flatnonzero(sample_mask & roi_mask)].value_counts()
        whole_counts.loc[sample_id, whole.index] = whole.astype(int)
        roi_counts.loc[sample_id, roi.index] = roi.astype(int)

    roi_totals = roi_counts.sum(axis=1)
    whole_totals = whole_counts.sum(axis=1)
    roi_percent = roi_counts.div(roi_totals.replace(0, np.nan), axis=0) * 100
    whole_percent = whole_counts.div(whole_totals.replace(0, np.nan), axis=0) * 100

    n_labels = max(len(labels), 1)
    roi_smoothed = roi_counts.add(0.5).div(roi_totals + 0.5 * n_labels, axis=0)
    whole_smoothed = whole_counts.add(0.5).div(
        whole_totals + 0.5 * n_labels, axis=0
    )
    enrichment = np.log2(roi_smoothed / whole_smoothed)
    enrichment.loc[roi_totals == 0, :] = np.nan

    records = []
    for sample_id in samples:
        for label in labels:
            records.append(
                {
                    "sample_id": sample_id,
                    "cell_type": label,
                    "roi_cells": int(roi_counts.loc[sample_id, label]),
                    "roi_percent": roi_percent.loc[sample_id, label],
                    "whole_sample_cells": int(whole_counts.loc[sample_id, label]),
                    "whole_sample_percent": whole_percent.loc[sample_id, label],
                    "log2_roi_enrichment": enrichment.loc[sample_id, label],
                }
            )
    return pd.DataFrame(records), roi_percent, enrichment


def write_region_qc(
    output_dir: Path,
    cohort: dict,
    adata,
    polygons: dict[str, tuple[Path, np.ndarray]],
    roi_mask: np.ndarray,
    class_col: str,
) -> None:
    plots_dir = output_dir / "plots"
    tables_dir = output_dir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    samples = [str(sample) for sample in cohort["samples"]]
    sample_values = adata.obs["sample_id"].astype(str).to_numpy()
    spatial = np.asarray(adata.obsm["spatial"])
    counts = adata.layers["counts"]
    transcripts = (
        adata.obs["n_transcripts"].to_numpy()
        if "n_transcripts" in adata.obs
        else np.asarray(counts.sum(axis=1)).ravel()
    )
    genes = (
        adata.obs["n_genes_by_counts"].to_numpy()
        if "n_genes_by_counts" in adata.obs
        else np.asarray((counts > 0).sum(axis=1)).ravel()
    )

    reporting = cohort.get("reporting", {}) or {}
    palettes = reporting.get("palettes", {}) or {}
    visualizations = reporting.get("visualizations", {}) or {}
    minimum_heatmap_cells = int(
        visualizations.get("region_qc_min_cell_type_cells", 10)
    )
    fixed_colors = palettes.get("cell_type_fixed", {}) or {}
    color_cycle = palettes.get("cell_type_cycle", []) or list(plt.cm.tab20.colors)
    class_values = adata.obs[class_col].astype("object").fillna("Unclassified").astype(str)
    labels = sorted(class_values.unique())
    palette = {
        label: matplotlib.colors.to_hex(
            fixed_colors.get(label, color_cycle[index % len(color_cycle)])
        )
        for index, label in enumerate(labels)
    }

    summary_rows = []
    sample_plot_paths = {}
    warnings = []
    for sample_id in samples:
        sample_mask = sample_values == sample_id
        sample_positions = np.flatnonzero(sample_mask)
        inside = roi_mask[sample_positions]
        coordinates = spatial[sample_positions]
        polygon = polygons.get(sample_id)
        status = "ok"
        if polygon is None:
            status = "missing polygon"
            warnings.append(f"{sample_id}: no canonical polygon JSON")
        elif not inside.any():
            status = "zero cells inside"
            warnings.append(f"{sample_id}: polygon contains zero cells")

        absolute_inside = sample_positions[inside]
        n_total = len(sample_positions)
        n_inside = int(inside.sum())
        pct_inside = 100 * n_inside / n_total if n_total else np.nan
        summary_rows.append(
            {
                "sample_id": sample_id,
                "group": cohort["group_map"].get(sample_id, "unknown"),
                "status": status,
                "total_cells": n_total,
                "roi_cells": n_inside,
                "roi_percent": pct_inside,
                "n_vertices": len(polygon[1]) if polygon else 0,
                "median_transcripts_in_roi": (
                    float(np.median(transcripts[absolute_inside])) if n_inside else np.nan
                ),
                "median_genes_in_roi": (
                    float(np.median(genes[absolute_inside])) if n_inside else np.nan
                ),
            }
        )
        if polygon is None or not len(coordinates):
            continue

        fig, axis = plt.subplots(figsize=(6, 6))
        axis.scatter(
            coordinates[~inside, 0],
            coordinates[~inside, 1],
            color="#d9d9d9",
            s=0.4,
            alpha=0.4,
            rasterized=True,
        )
        inside_types = class_values.iloc[absolute_inside].to_numpy()
        for label in labels:
            label_mask = inside_types == label
            if label_mask.any():
                axis.scatter(
                    coordinates[inside][label_mask, 0],
                    coordinates[inside][label_mask, 1],
                    color=palette[label],
                    s=1.2,
                    alpha=0.75,
                    rasterized=True,
                )
        vertices = polygon[1]
        closed = np.vstack([vertices, vertices[0]])
        axis.plot(closed[:, 0], closed[:, 1], color="lime", linewidth=1.5)
        axis.set(
            title=f"{sample_id} — {n_inside:,} cells inside ({pct_inside:.1f}%)",
            xticks=[],
            yticks=[],
        )
        axis.set_aspect("equal")
        axis.invert_yaxis()
        sample_plot_paths[sample_id] = save_figure(
            fig, plots_dir, safe_name(sample_id), dpi=180
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = tables_dir / "region_qc_summary.csv"
    summary.to_csv(summary_path, index=False)

    x = np.arange(len(summary))
    group_colors = palettes.get("groups", {}) or {}
    bar_colors = [group_colors.get(group, "#4575b4") for group in summary["group"]]
    fig, axes = plt.subplots(
        2, 1, figsize=(max(9, len(summary) * 0.55), 8), sharex=True
    )
    axes[0].bar(x, summary["roi_cells"], color=bar_colors)
    axes[0].set_ylabel("Cells inside ROI")
    axes[1].bar(x, summary["roi_percent"], color=bar_colors)
    axes[1].set(ylabel="Cells inside ROI (%)", xticks=x)
    axes[1].set_xticklabels(summary["sample_id"], rotation=60, ha="right")
    fig.tight_layout()
    counts_plot = save_figure(fig, plots_dir, "roi_cell_count_and_percent")

    composition, roi_percent, enrichment = composition_tables(
        adata, samples, class_col, roi_mask
    )
    composition["group"] = composition["sample_id"].map(cohort["group_map"])
    composition_path = tables_dir / "roi_cell_type_composition.csv"
    composition.to_csv(composition_path, index=False)

    stacked_plot = None
    if roi_percent.notna().any(axis=None):
        order = roi_percent.mean(axis=0).sort_values(ascending=False).index
        plotted = roi_percent.loc[:, order].fillna(0)
        fig, axis = plt.subplots(figsize=(max(9, len(samples) * 0.6), 6))
        bottom = np.zeros(len(plotted))
        for label in plotted.columns:
            values = plotted[label].to_numpy()
            axis.bar(
                np.arange(len(plotted)),
                values,
                bottom=bottom,
                color=palette[label],
                label=label,
                width=0.85,
            )
            bottom += values
        axis.set(
            ylabel="ROI cells (%)",
            xlabel="Sample",
            title=f"Cell-type composition inside ROI ({class_col})",
            xticks=np.arange(len(plotted)),
            ylim=(0, 100),
        )
        axis.set_xticklabels(plotted.index, rotation=60, ha="right")
        axis.legend(
            bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=8
        )
        fig.tight_layout()
        stacked_plot = save_figure(fig, plots_dir, "roi_cell_type_composition")

    heatmap_path = None
    eligible_labels = (
        composition.groupby("cell_type", observed=True)["roi_cells"].sum()
        >= minimum_heatmap_cells
    )
    eligible_labels = eligible_labels[eligible_labels].index
    heatmap = enrichment.loc[:, enrichment.columns.intersection(eligible_labels)]
    heatmap = heatmap.loc[heatmap.notna().any(axis=1)]
    if not heatmap.empty:
        heatmap = heatmap.loc[
            :, heatmap.var(axis=0).sort_values(ascending=False).index
        ]
        finite = np.abs(heatmap.to_numpy()[np.isfinite(heatmap.to_numpy())])
        color_limit = max(1.0, float(np.quantile(finite, 0.95))) if len(finite) else 1.0
        fig, axis = plt.subplots(
            figsize=(
                max(8, heatmap.shape[1] * 0.65),
                max(4, heatmap.shape[0] * 0.4),
            )
        )
        image = axis.imshow(
            heatmap, aspect="auto", cmap="RdBu_r", vmin=-color_limit, vmax=color_limit
        )
        axis.set(
            xticks=np.arange(heatmap.shape[1]),
            yticks=np.arange(heatmap.shape[0]),
            xticklabels=heatmap.columns,
            yticklabels=heatmap.index,
            title=f"ROI enrichment relative to matched whole sample ({class_col})",
        )
        axis.tick_params(axis="x", rotation=60)
        fig.colorbar(image, ax=axis, label="log2 enrichment (ROI / whole sample)")
        fig.tight_layout()
        heatmap_path = save_figure(fig, plots_dir, "roi_cell_type_enrichment")

    legend_labels = sorted(
        composition.loc[composition["roi_cells"] > 0, "cell_type"].unique()
    )
    legend = "".join(
        f'<span class="legend-item"><span class="swatch" style="background:{palette[label]}"></span>'
        f"{html.escape(label)}</span>"
        for label in legend_labels
    )
    table_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row.sample_id))}</td>"
        f"<td>{html.escape(str(row.group))}</td>"
        f"<td>{html.escape(str(row.status))}</td>"
        f"<td>{int(row.total_cells):,}</td>"
        f"<td>{int(row.roi_cells):,}</td>"
        f"<td>{row.roi_percent:.1f}%</td>"
        f"<td>{int(row.n_vertices):,}</td>"
        f"<td>{row.median_transcripts_in_roi:,.1f}</td>"
        f"<td>{row.median_genes_in_roi:,.1f}</td>"
        "</tr>"
        for row in summary.itertuples()
    )
    warning_html = (
        "<ul>"
        + "".join(f"<li>{html.escape(warning)}</li>" for warning in warnings)
        + "</ul>"
        if warnings
        else "<p>No validation warnings.</p>"
    )
    cards = []
    for sample_id in samples:
        path = sample_plot_paths.get(sample_id)
        if path:
            cards.append(
                f'<div class="card"><h3>{html.escape(sample_id)}</h3>'
                f'<img src="plots/{path.name}"></div>'
            )
        else:
            status = summary.loc[summary["sample_id"] == sample_id, "status"].iloc[0]
            cards.append(
                f'<div class="card"><h3>{html.escape(sample_id)}</h3>'
                f'<p class="warning">{html.escape(status)}</p></div>'
            )

    composition_html = ""
    if stacked_plot:
        composition_html += (
            '<h2>ROI cell-type composition</h2>'
            "<p>The stacked bars show absolute composition within each ROI.</p>"
            f'<img class="cohort-plot" src="plots/{stacked_plot.name}">'
        )
    if heatmap_path:
        composition_html += (
            "<h2>ROI cell-type enrichment</h2>"
            "<p>Values compare each ROI with its matched whole sample. Zero means no "
            "enrichment; positive and negative values indicate enrichment and depletion. "
            f"Cell types with fewer than {minimum_heatmap_cells:,} ROI cells across the "
            "cohort are omitted from this heatmap but retained in the downloadable table.</p>"
            f'<img class="cohort-plot" src="plots/{heatmap_path.name}">'
        )

    region_name = str((cohort.get("regions", {}) or {}).get("name", "subset"))
    report = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Region QC</title><style>
body{{font:14px/1.45 sans-serif;max-width:1400px;margin:2rem auto;padding:0 1rem;color:#222}}
table{{border-collapse:collapse;font-size:12px}}th,td{{border:1px solid #ccc;padding:.35rem;text-align:right}}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2),th:nth-child(3),td:nth-child(3){{text-align:left}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:1rem}}
.card{{border:1px solid #ddd;border-radius:6px;padding:.5rem;text-align:center}}.card img{{max-width:100%}}
.cohort-plot{{max-width:100%}}.warning{{color:#a00}}.legend-item{{display:inline-block;margin:.25rem .8rem .25rem 0}}
.swatch{{display:inline-block;width:12px;height:12px;margin-right:.3rem;border:1px solid #999;vertical-align:middle}}
</style></head><body>
<h1>{html.escape(region_name.replace("_", " ").title())} region QC</h1>
<p>Generated {datetime.now().isoformat(timespec="seconds")}. Grey points are cells
outside the region; colored points are cells inside it. Polygon boundaries are green.</p>
<p><a href="tables/{summary_path.name}">Download summary CSV</a> ·
<a href="tables/{composition_path.name}">Download cell-type composition CSV</a></p>
<h2>Validation warnings</h2>{warning_html}
<h2>Per-sample metrics</h2><table><thead><tr>
<th>Sample</th><th>Group</th><th>Status</th><th>Total cells</th><th>ROI cells</th>
<th>ROI %</th><th>Vertices</th><th>Median transcripts</th><th>Median genes</th>
</tr></thead><tbody>{table_rows}</tbody></table>
<h2>ROI size</h2><img class="cohort-plot" src="plots/{counts_plot.name}">
{composition_html}
<h2>Cell-type legend</h2><div>{legend}</div>
<h2>Per-sample polygon review</h2><div class="grid">{''.join(cards)}</div>
</body></html>"""
    report_path = output_dir / "region_qc_report.html"
    report_path.write_text(report, encoding="utf-8")
    print(f"Saved {report_path}")


def main(argv=None) -> None:
    args = parse_args(argv)
    cohort = load_cohort(args.config)
    paths = cohort_paths(cohort)
    input_path = paths.output("04a_import_annotations", "combined_annotated.h5ad")
    output_dir = paths.output("04b_select_rois")
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(input_path)
    class_col = "cell_type" if args.cell_type_level == "coarse" else args.cell_type_level
    if class_col not in adata.obs:
        raise KeyError(
            f"Cell-type column {class_col!r} is not present in adata.obs. "
            f"Available columns: {sorted(adata.obs.columns)}"
        )
    if "counts" not in adata.layers:
        raise KeyError("Expected raw counts in adata.layers['counts'].")

    configured_files = (cohort.get("regions", {}) or {}).get("files", []) or []
    polygon_files = configured_polygon_files(cohort, paths)
    if configured_files and not polygon_files:
        raise FileNotFoundError(
            "Configured region inputs contained no JSON files."
        )
    known_samples = {str(sample) for sample in cohort["samples"]}
    polygons = load_polygons(polygon_files, known_samples)
    roi_mask = build_roi_mask(adata, polygons)
    adata.obs["in_roi"] = roi_mask
    adata.uns["roi_selection"] = {
        "region_name": str((cohort.get("regions", {}) or {}).get("name", "subset")),
        "source_files": [str(path) for path in polygon_files],
        "n_selected_cells": int(roi_mask.sum()),
    }

    annotated_output = output_dir / "combined_annotated_with_rois.h5ad"
    roi_output = output_dir / "roi_cells.h5ad"
    adata.write_h5ad(annotated_output)
    adata[roi_mask].copy().write_h5ad(roi_output)
    copy_polygon_inputs(polygons, output_dir / "region_inputs")
    if polygons:
        write_region_qc(output_dir, cohort, adata, polygons, roi_mask, class_col)
    else:
        print("No region JSON inputs configured; wrote an empty in_roi mask.")
    print(f"Selected {int(roi_mask.sum()):,} / {adata.n_obs:,} cells")
    print(f"Saved {annotated_output}")
    print(f"Saved {roi_output}")


if __name__ == "__main__":
    main()
