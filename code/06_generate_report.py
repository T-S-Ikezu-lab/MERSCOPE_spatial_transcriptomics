#!/usr/bin/env python3
"""Gather pipeline outputs into static cohort tables, plots, and an HTML report."""

import argparse
import itertools
import math
import random
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape
from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests

from config_utils import cohort_paths, load_cohort
sc.settings.verbosity = 1
sns.set_style("whitegrid")
sc.settings.set_figure_params(dpi=100, frameon=False)


def make_contrasting_cycle(colors):
    half = len(colors) // 2
    result = []
    for index in range(half):
        result.extend((colors[index], colors[index + half]))
    if len(colors) % 2:
        result.append(colors[-1])
    return result


def build_palette(labels, colors, fixed_colors, seed=42):
    """Assign configured fixed colors and deterministic fallback colors."""
    reserved = {color.upper() for color in fixed_colors.values()}
    base = [color for color in make_contrasting_cycle(colors)
            if color.upper() not in reserved]
    if not base:
        raise ValueError("reporting.palettes.cell_type_cycle must contain an unreserved color")
    random.Random(seed).shuffle(base)

    mapping = {}
    other_index = 0
    for label in labels:
        if label in fixed_colors:
            mapping[label] = fixed_colors[label]
            continue
        color = base[other_index % len(base)]
        round_index = other_index // len(base)
        other_index += 1
        if round_index:
            rgb = np.array(mcolors.to_rgb(color))
            rgb = rgb * 0.72 if round_index % 2 else 1 - (1 - rgb) * 0.72
            color = mcolors.to_hex(np.clip(rgb, 0, 1))
        mapping[label] = color
    return mapping


def group_order_from_config(cohort):
    """Preserve comparison order, then append any remaining configured groups."""
    ordered = []
    for comparison in cohort.get("comparisons", []) or []:
        if isinstance(comparison, (list, tuple)) and len(comparison) == 2:
            ordered.extend(str(group) for group in comparison)
    ordered.extend(str(group) for group in (cohort.get("group_map", {}) or {}).values())
    return list(dict.fromkeys(ordered))


def write_cluster_spatial_plots(adata, groups, palette, plots_dir):
    """Write per-sample and cohort-composite spatial plots for each coarse type."""
    separate_dir = plots_dir / "spatial_by_cluster" / "separate"
    composite_dir = plots_dir / "spatial_by_cluster" / "composite"
    separate_dir.mkdir(parents=True, exist_ok=True)
    composite_dir.mkdir(parents=True, exist_ok=True)

    coarse_values = adata.obs["cell_type"]
    if isinstance(coarse_values.dtype, pd.CategoricalDtype):
        coarse_order = [
            label for label in coarse_values.cat.categories
            if (coarse_values == label).any()
        ]
    else:
        coarse_order = sorted(coarse_values.dropna().unique())

    samples_by_group = {
        group: sorted(
            adata.obs.loc[adata.obs["group"] == group, "sample_id"].unique()
        )
        for group in groups
    }
    spatial = adata.obsm["spatial"]
    sample_values = adata.obs["sample_id"].values
    coarse_array = coarse_values.values
    per_sample = {
        sample_id: (
            spatial[sample_values == sample_id],
            coarse_array[sample_values == sample_id],
        )
        for sample_id in adata.obs["sample_id"].unique()
    }

    def slug(value):
        return (
            str(value).lower().replace("-", "_").replace(" ", "_").replace("/", "_")
        )

    def draw(axis, coordinates, labels, cell_type, color):
        axis.scatter(
            coordinates[:, 0], coordinates[:, 1], c="#d9d9d9", s=0.6,
            alpha=0.35, linewidths=0, rasterized=True,
        )
        selected = labels == cell_type
        if selected.any():
            axis.scatter(
                coordinates[selected, 0], coordinates[selected, 1], c=color,
                s=1.4, alpha=0.9, linewidths=0, rasterized=True,
            )
        axis.set_aspect("equal")
        axis.invert_yaxis()
        axis.set_xticks([])
        axis.set_yticks([])
        return int(selected.sum())

    for cell_type in coarse_order:
        color = palette[cell_type]
        for group in groups:
            for sample_id in samples_by_group[group]:
                coordinates, labels = per_sample[sample_id]
                figure, axis = plt.subplots(figsize=(6, 6))
                n_selected = draw(
                    axis, coordinates, labels, cell_type, color
                )
                axis.set_title(
                    f"{cell_type} — {sample_id}\n"
                    f"{group} (n={n_selected:,} / {len(labels):,})",
                    fontsize=10,
                )
                figure.tight_layout()
                figure.savefig(
                    separate_dir / f"spatial_{slug(cell_type)}__{slug(sample_id)}.png",
                    dpi=180,
                    bbox_inches="tight",
                )
                figure.savefig(
                    separate_dir / f"spatial_{slug(cell_type)}__{slug(sample_id)}.svg",
                    format="svg",
                    bbox_inches="tight",
                )
                plt.close(figure)

    max_columns = max(len(samples) for samples in samples_by_group.values())
    for cell_type in coarse_order:
        figure, axes = plt.subplots(
            len(groups),
            max_columns,
            figsize=(4.4 * max_columns, 4.6 * len(groups)),
            squeeze=False,
        )
        for row, group in enumerate(groups):
            group_samples = samples_by_group[group]
            for column in range(max_columns):
                axis = axes[row][column]
                if column >= len(group_samples):
                    axis.axis("off")
                    continue
                sample_id = group_samples[column]
                coordinates, labels = per_sample[sample_id]
                n_selected = draw(
                    axis, coordinates, labels, cell_type, palette[cell_type]
                )
                axis.set_title(f"{sample_id} (n={n_selected:,})", fontsize=9)
            axes[row][0].set_ylabel(
                group, fontsize=12, fontweight="bold", labelpad=12
            )
        figure.suptitle(
            f"Spatial distribution — {cell_type} (all samples; rows = group)",
            fontsize=15,
        )
        figure.tight_layout(rect=[0, 0, 1, 0.98])
        figure.savefig(
            composite_dir / f"spatial_composite_{slug(cell_type)}.png",
            dpi=180,
            bbox_inches="tight",
        )
        figure.savefig(
            composite_dir / f"spatial_composite_{slug(cell_type)}.svg",
            format="svg",
            bbox_inches="tight",
        )
        plt.close(figure)

    print(
        f"Wrote per-cluster spatial plots for {len(coarse_order)} cell types "
        f"under {separate_dir.parent}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Cohort YAML file")
    return parser.parse_args()


def run(args):
    cohort = load_cohort(args.config)
    paths = cohort_paths(cohort)
    group_map = cohort.get("group_map", {}) or {}
    reporting = cohort.get("reporting", {}) or {}
    palettes = reporting.get("palettes", {}) or {}
    visualizations = reporting.get("visualizations", {}) or {}

    roi_annotated_path = paths.output(
        "04b_select_rois", "combined_annotated_with_rois.h5ad"
    )
    annotated_path = paths.output("04a_import_annotations", "combined_annotated.h5ad")
    normalized_path = paths.output("03_normalize_dimred", "combined_norm_dimred.h5ad")
    INPUT_PATH = next(
        path
        for path in (roi_annotated_path, annotated_path, normalized_path)
        if path.exists()
    )
    OUTPUT_DIR = paths.output("06_generate_report")
    PLOTS_DIR = OUTPUT_DIR / "plots"
    for directory in (OUTPUT_DIR, PLOTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    NORM_METHOD = visualizations.get("normalization_method", "Count norm (mean) + log1p")
    N_PCS = int(visualizations.get("n_pcs", 20))
    N_NEIGHBORS = int(visualizations.get("n_neighbors", 15))
    LEIDEN_RESOLUTION = float(visualizations.get("leiden_resolution", 0.5))
    N_TOP_GENES = int(visualizations.get("pseudobulk_top_genes", 500))
    GROUP_ORDER = reporting.get("group_order") or group_order_from_config(cohort)
    CELL_TYPE_COLORS = palettes.get("cell_type_fixed", {}) or {}
    CELL_TYPE_CYCLE = palettes.get("cell_type_cycle", [
        "#FFD600", "#39FF14", "#00FFB3", "#00FFD5", "#00E5FF", "#00B0FF",
        "#2979FF", "#651FFF", "#AA00FF", "#F5009D", "#FF4081", "#FF5252",
    ])
    GROUP_PALETTE = palettes.get("groups", {}) or {}
    GROUP_PALETTE.setdefault("unknown", "#9E9E9E")

    def save_fig(fig, name, dpi=150):
        fig.savefig(PLOTS_DIR / f"{name}.png", dpi=dpi, bbox_inches="tight")
        fig.savefig(PLOTS_DIR / f"{name}.svg", format="svg", bbox_inches="tight")
        plt.close(fig)

    print(f"Config: {args.config.resolve()}")
    print(f"Input: {INPUT_PATH}")
    print(f"Output: {OUTPUT_DIR}")

    adata = sc.read_h5ad(INPUT_PATH)

    if "counts" not in adata.layers:
        raise KeyError("Expected raw counts in adata.layers['counts'].")
    if "X_umap" not in adata.obsm:
        raise KeyError("Expected UMAP coordinates in adata.obsm['X_umap'].")
    if "spatial" not in adata.obsm:
        raise KeyError("Expected spatial coordinates in adata.obsm['spatial'].")
    if "sample_id" not in adata.obs:
        raise KeyError("Expected sample identifiers in adata.obs['sample_id'].")
    if "n_transcripts" not in adata.obs:
        adata.obs["n_transcripts"] = np.asarray(
            adata.layers["counts"].sum(axis=1)
        ).ravel()

    CELLTYPE_RESOLUTIONS = {}
    if "cell_type" in adata.obs:
        CELLTYPE_RESOLUTIONS["coarse"] = {
            "column": "cell_type", "label": "Coarse cell type"
        }
    atlas_col = next(
        (column for column in ("class_name_clean", "class_name") if column in adata.obs),
        None,
    )
    if atlas_col:
        CELLTYPE_RESOLUTIONS["atlas"] = {
            "column": atlas_col, "label": "Atlas cell type"
        }
    resolution_keys = list(CELLTYPE_RESOLUTIONS)

    adata.obs["group"] = adata.obs["sample_id"].map(group_map).fillna("unknown")
    groups = [g for g in GROUP_ORDER if (adata.obs["group"] == g).any()]
    if not groups:
        raise ValueError(
            "No configured groups are present in the annotated data. "
            "Check group_map and reporting.group_order."
        )

    n_unknown = int((adata.obs["group"] == "unknown").sum())
    if n_unknown:
        print(f"Warning: {n_unknown:,} cells have group='unknown' (sample not in group_map).")

    print(f"Loaded: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
    print(f"Groups: {groups}")
    print("Resolutions:")
    for k in resolution_keys:
        print(f"  {k}: {CELLTYPE_RESOLUTIONS[k]['column']}")
    print(adata.obs["group"].value_counts().to_string())

    subset_mask = (
        adata.obs["in_roi"].astype(bool).to_numpy()
        if "in_roi" in adata.obs
        else np.zeros(adata.n_obs, dtype=bool)
    )
    n_subset = int(subset_mask.sum())
    region_name = str((cohort.get("regions", {}) or {}).get("name", "subset"))
    print(f"Selected region '{region_name}': {n_subset:,} cells")
    if n_subset == 0:
        print("No cells were selected by step 04b; region views will be skipped.")

    REGIONS = {"whole_brain": {"label": "Whole brain", "mask": None}}
    if n_subset > 0:
        REGIONS[region_name] = {
            "label": region_name.replace("_", " ").title(),
            "mask": subset_mask,
        }
    region_keys = list(REGIONS)


    def region_mask(region_key):
        """Boolean mask over `adata.obs` rows for a region (whole brain = every cell)."""
        m = REGIONS[region_key]["mask"]
        return np.ones(adata.n_obs, dtype=bool) if m is None else m


    RES_FILE_SLUG = {"coarse": "cell_type", "atlas": "atlas_class"}

    CELLTYPE_VIEWS = {}
    for region_key in region_keys:
        for res_key in resolution_keys:
            CELLTYPE_VIEWS[f"{res_key}_{region_key}"] = {
                "res_key": res_key,
                "region_key": region_key,
                "column": CELLTYPE_RESOLUTIONS[res_key]["column"],
                "res_label": CELLTYPE_RESOLUTIONS[res_key]["label"],
                "region_label": REGIONS[region_key]["label"],
                "label": f"{CELLTYPE_RESOLUTIONS[res_key]['label']} — {REGIONS[region_key]['label']}",
                "file_slug": f"{RES_FILE_SLUG[res_key]}_{region_key}",
                "plot_slug": f"{res_key}_{region_key}",
            }
    view_keys = list(CELLTYPE_VIEWS)


    CELLTYPE_ORDERS = {}
    CELLTYPE_PALETTES = {}
    for res_key in resolution_keys:
        col = CELLTYPE_RESOLUTIONS[res_key]["column"]
        vals = adata.obs[col]
        if isinstance(vals.dtype, pd.CategoricalDtype):
            base_order = [c for c in vals.cat.categories if (vals == c).any()]
        else:
            base_order = sorted(vals.dropna().unique())
        totals = (
            pd.crosstab(adata.obs["group"], vals)
            .reindex(index=groups, columns=base_order, fill_value=0)
            .sum(axis=0)
        )
        CELLTYPE_ORDERS[res_key] = totals.sort_values(ascending=False).index.tolist()
        CELLTYPE_PALETTES[res_key] = build_palette(CELLTYPE_ORDERS[res_key], CELL_TYPE_CYCLE, CELL_TYPE_COLORS)

    for vk, v in CELLTYPE_VIEWS.items():
        print(f"View {vk:<24} {v['label']:<44} n={int(region_mask(v['region_key']).sum()):,}")

    umap = adata.obsm["X_umap"]
    spatial = adata.obsm["spatial"]

    refit_path = paths.output("03_normalize_dimred", "group_refit_umap.csv.gz")
    group_refit_umap = {}
    if refit_path.exists():
        refit_df = pd.read_csv(refit_path, dtype={"cell_id": str, "group": str})
        for group, group_df in refit_df.groupby("group", observed=True, sort=False):
            coords = group_df.set_index("cell_id")[["UMAP1", "UMAP2"]]
            coords.index.name = None
            group_refit_umap[str(group)] = coords
        print(f"Loaded separately fitted group UMAP coordinates: {refit_path}")

    celltype_results = {}

    for view_key, view in CELLTYPE_VIEWS.items():
        res_key = view["res_key"]
        col = view["column"]
        label = view["label"]
        slug = view["plot_slug"]
        is_whole_brain = view["region_key"] == "whole_brain"

        vmask = region_mask(view["region_key"])
        obs_v = adata.obs.loc[vmask]
        view_names = set(obs_v.index)
        vals = obs_v[col]
        obs_type = vals.values
        obs_group_v = obs_v["group"].values
        umap_v = umap[vmask]
        spatial_v = spatial[vmask]
        x_view, y_view = umap_v[:, 0], umap_v[:, 1]


        cell_type_palette = CELLTYPE_PALETTES[res_key]
        ct_counts = pd.crosstab(obs_v["group"], vals).reindex(
            index=groups,
            columns=CELLTYPE_ORDERS[res_key],
            fill_value=0,
        )
        view_totals = ct_counts.sum(axis=0)
        cell_type_order = view_totals[view_totals > 0].sort_values(ascending=False).index.tolist()
        ct_counts = ct_counts.reindex(columns=cell_type_order)
        ct_props = (
            ct_counts.div(ct_counts.sum(axis=1).replace(0, np.nan), axis=0)
            .fillna(0)
            * 100
        )
        group_cell_counts = {g: int(ct_counts.loc[g].sum()) for g in groups}

        composition_table = []
        for ct in cell_type_order:
            row = {"cell_type": ct}
            row.update({g: float(ct_props.loc[g, ct]) for g in groups})
            composition_table.append(row)

        fig, ax = plt.subplots(figsize=(10, 6))
        bottom = np.zeros(len(groups))
        for ct in cell_type_order:
            vals_ct = ct_props[ct].values
            ax.bar(groups, vals_ct, bottom=bottom, label=ct, color=cell_type_palette[ct], width=0.72)
            bottom += vals_ct
        ax.set_ylabel("Cell-type proportion (%)")
        ax.set_title(f"Cell-type composition by group ({label})")
        ax.set_ylim(0, 100)
        ax.legend(title=col, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, fontsize=7)
        fig.tight_layout()
        comp_name = f"celltype_composition_by_group_{slug}"
        save_fig(fig, comp_name)

        ncol_legend = 1 if len(cell_type_order) <= 24 else 2
        legend_right = 0.83 if ncol_legend == 1 else 0.70
        legend_handles = [Patch(facecolor=cell_type_palette[ct], label=ct) for ct in cell_type_order]

        fig, axes = plt.subplots(1, len(groups), figsize=(7 * len(groups), 8.6), sharex=True, sharey=True)
        axes = np.atleast_1d(axes)
        for ax, grp in zip(axes, groups):
            gmask = obs_group_v == grp
            ax.scatter(x_view, y_view, c="#d9d9d9", s=1, alpha=0.12, linewidths=0, rasterized=True)
            gx, gy = umap_v[gmask, 0], umap_v[gmask, 1]
            gt = obs_type[gmask]
            for ct in cell_type_order:
                cmask = gt == ct
                if cmask.any():
                    ax.scatter(gx[cmask], gy[cmask], c=cell_type_palette[ct], s=2,
                               alpha=0.75, linewidths=0, rasterized=True)
            ax.set_title(f"{grp} (n={int(gmask.sum()):,})")
            ax.set_xlabel("UMAP1")
        axes[0].set_ylabel("UMAP2")
        fig.suptitle(f"UMAP by group ({label})", fontsize=14)
        fig.legend(handles=legend_handles, title=col, loc="center left", bbox_to_anchor=(legend_right + 0.01, 0.5),
                   frameon=True, ncol=ncol_legend, fontsize=7, title_fontsize=8, borderaxespad=0.0)
        fig.tight_layout(rect=[0, 0.02, legend_right, 0.97])
        umap_name = f"umap_celltype_by_group_{slug}"
        save_fig(fig, umap_name)

        umap_refit_path = None
        if group_refit_umap and all(group in group_refit_umap for group in groups):
            refit_coords = {}
            for group in groups:
                coords_df = group_refit_umap[group]
                if not is_whole_brain:
                    coords_df = coords_df[coords_df.index.isin(view_names)]
                refit_coords[group] = coords_df

            fig, axes = plt.subplots(1, len(groups), figsize=(7 * len(groups), 8.6))
            axes = np.atleast_1d(axes)
            for ax, group in zip(axes, groups):
                coords_df = refit_coords[group]
                stype = vals.loc[coords_df.index].values
                for cell_type in cell_type_order:
                    selected = stype == cell_type
                    if selected.any():
                        xy = coords_df.loc[selected, ["UMAP1", "UMAP2"]].values
                        ax.scatter(
                            xy[:, 0], xy[:, 1], c=cell_type_palette[cell_type],
                            s=2, alpha=0.75, linewidths=0, rasterized=True,
                        )
                ax.set_title(f"{group} (re-fit, n={len(coords_df):,})")
                ax.set_xlabel("UMAP1")
            axes[0].set_ylabel("UMAP2")
            fig.suptitle(f"UMAP by group, fitted separately ({label})", fontsize=14)
            fig.legend(
                handles=legend_handles, title=col, loc="center left",
                bbox_to_anchor=(legend_right + 0.01, 0.5), frameon=True,
                ncol=ncol_legend, fontsize=7, title_fontsize=8,
                borderaxespad=0.0,
            )
            fig.tight_layout(rect=[0, 0.02, legend_right, 0.97])
            umap_refit_name = f"umap_celltype_by_group_refit_{slug}"
            save_fig(fig, umap_refit_name)
            umap_refit_path = f"plots/{umap_refit_name}.png"

        spatial_by_group = {}
        for grp in groups:
            group_samples = sorted(obs_v.loc[obs_v["group"] == grp, "sample_id"].unique())
            n = len(group_samples)
            ncols = min(3, max(1, n))
            nrows = math.ceil(n / ncols)
            nrows = max(1, nrows)
            fig, axes = plt.subplots(nrows, ncols, figsize=(6.6 * ncols, 5.8 * nrows))
            axes = np.atleast_1d(axes).ravel()

            if not group_samples:
                axes[0].text(
                    0.5, 0.5, "No cells in this region", ha="center", va="center",
                    transform=axes[0].transAxes,
                )
                axes[0].axis("off")

            for ax, sid in zip(axes, group_samples):
                smask = (obs_v["sample_id"].values == sid)
                coords = spatial_v[smask]
                stype = obs_type[smask]
                for ct in cell_type_order:
                    cmask = stype == ct
                    if cmask.any():
                        ax.scatter(coords[cmask, 0], coords[cmask, 1], c=cell_type_palette[ct],
                                   s=0.8 if is_whole_brain else 2.5, alpha=0.65, linewidths=0, rasterized=True)
                ax.set_title(f"{sid} (n={int(smask.sum()):,})", fontsize=10)
                ax.set_aspect("equal")
                ax.invert_yaxis()
                ax.set_xticks([])
                ax.set_yticks([])

            for ax in axes[len(group_samples):]:
                ax.axis("off")

            fig.suptitle(f"Spatial distribution by cell type — {grp} ({label})", fontsize=14)
            fig.legend(handles=legend_handles, title=col, loc="center left", bbox_to_anchor=(legend_right + 0.01, 0.5),
                       frameon=True, ncol=ncol_legend, fontsize=7, title_fontsize=8, borderaxespad=0.0)
            fig.tight_layout(rect=[0, 0.02, legend_right, 0.97])
            grp_slug = grp.lower().replace("-", "_").replace(" ", "_")
            spatial_name = f"spatial_celltype_{grp_slug}_{slug}"
            save_fig(fig, spatial_name, dpi=180)
            spatial_by_group[grp] = f"plots/{spatial_name}.png"

        celltype_results[view_key] = {
            "label": label,
            "res_key": res_key,
            "res_label": view["res_label"],
            "region_key": view["region_key"],
            "region_label": view["region_label"],
            "column": col,
            "n_cells": int(vmask.sum()),
            "composition_table": composition_table,
            "group_cell_counts": group_cell_counts,
            "plots": {
                "composition": f"plots/{comp_name}.png",
                "umap_by_group": f"plots/{umap_name}.png",
                "umap_by_group_refit": umap_refit_path,
                "spatial_by_group": spatial_by_group,
            },
        }
        print(f"Generated group-distribution plots for view: {view_key}")

    print("\nGenerated group-distribution plots for views:", view_keys)


    for view_key, view in CELLTYPE_VIEWS.items():
        res_key = view["res_key"]
        col = view["column"]
        label = view["label"]
        slug = view["plot_slug"]

        vmask = region_mask(view["region_key"])
        obs_v = adata.obs.loc[vmask]
        vals = obs_v[col]
        obs_type = vals.values
        umap_v = umap[vmask]

        cell_type_palette = CELLTYPE_PALETTES[res_key]
        present = pd.Series(obs_type).value_counts()
        cell_type_order = [ct for ct in CELLTYPE_ORDERS[res_key] if present.get(ct, 0) > 0]

        ncol_legend = 1 if len(cell_type_order) <= 24 else 2
        legend_right = 0.80 if ncol_legend == 1 else 0.68

        fig, ax = plt.subplots(figsize=(11, 9))
        for ct in cell_type_order:
            cmask = obs_type == ct
            ax.scatter(umap_v[cmask, 0], umap_v[cmask, 1], c=cell_type_palette[ct],
                       s=1.5, alpha=0.7, linewidths=0, rasterized=True)

        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_title(f"UMAP — all cells ({label})\nn = {int(vmask.sum()):,} cells", fontsize=13)
        legend_handles = [Patch(facecolor=cell_type_palette[ct], label=ct) for ct in cell_type_order]
        fig.legend(handles=legend_handles, title=col, loc="center left",
                   bbox_to_anchor=(legend_right + 0.01, 0.5), frameon=True,
                   ncol=ncol_legend, fontsize=7, title_fontsize=8, borderaxespad=0.0)
        fig.tight_layout(rect=[0, 0.02, legend_right, 0.96])
        overall_name = f"umap_overall_{slug}"
        save_fig(fig, overall_name, dpi=200)

        celltype_results[view_key]["plots"]["umap_overall"] = f"plots/{overall_name}.png"
        print(f"Overall UMAP written for view: {view_key} ({len(cell_type_order)} labels)")

    print("\nOverall (all-cell) UMAPs written for views:", view_keys)

    if "coarse" in CELLTYPE_PALETTES:
        write_cluster_spatial_plots(
            adata,
            groups,
            CELLTYPE_PALETTES["coarse"],
            PLOTS_DIR,
        )

    TABLES_DIR = OUTPUT_DIR / "tables"
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


    def _stars(q):
        if q is None or np.isnan(q):
            return ""
        if q < 0.001:
            return "***"
        if q < 0.01:
            return "**"
        if q < 0.05:
            return "*"
        return ""


    def write_proportions_by_group_csv(label_col, file_slug, cell_mask):
        """% composition of `label_col` within each group (rows=group, cols=label)."""
        valid = adata.obs[cell_mask].dropna(subset=["group", label_col]).copy()
        valid = valid[valid["group"].isin(groups)]
        prop_table = (
            pd.crosstab(valid["group"], valid[label_col], normalize="index")
            .reindex(index=groups)
            .fillna(0)
            * 100
        )
        label_order = prop_table.mean(axis=0).sort_values(ascending=False).index.tolist()
        prop_table = prop_table[label_order]
        prop_table.index.name = "group"
        out = TABLES_DIR / f"{file_slug}_proportions_by_group.csv"
        prop_table.to_csv(out)
        print(f"Saved: {out}")
        return f"tables/{out.name}"


    def run_pairwise_proportion_stats(label_col, file_slug, label_noun, region_noun,
                                      cell_mask, min_plot_pct=1.0):
        """Pairwise Welch t-tests on arcsine-sqrt per-sample proportions (BH-FDR)."""
        cohort_name = cohort.get("cohort_name", "")

        df = adata.obs.loc[cell_mask, ["sample_id", label_col]].copy()
        df["sample_id"] = df["sample_id"].astype(str)
        df["group"] = df["sample_id"].map(group_map)
        df = df.dropna(subset=["sample_id", "group", label_col])
        df["group"] = df["group"].astype(str)

        sample_counts = pd.crosstab(df["sample_id"], df[label_col])
        sample_prop = sample_counts.div(sample_counts.sum(axis=1), axis=0)
        sample_group = (
            df[["sample_id", "group"]].drop_duplicates("sample_id")
            .set_index("sample_id")["group"].reindex(sample_prop.index)
        )

        present_groups = [g for g in groups if (sample_group == g).any()]
        group_samples = {g: sample_group.index[sample_group == g].tolist() for g in present_groups}

        print("\nSamples per group:")
        for g in present_groups:
            print(f"  {g}: n={len(group_samples[g])}")

        comparisons = cohort.get("comparisons", None)
        if not comparisons:
            comparisons = list(itertools.combinations(present_groups, 2))
        comparisons = [(a, b) for a, b in comparisons if a in present_groups and b in present_groups]

        transformed = np.arcsin(np.sqrt(sample_prop.clip(0, 1)))
        label_order = sample_prop.mean(axis=0).sort_values(ascending=False).index.tolist()

        rows = []
        for group1, group2 in comparisons:
            samples1, samples2 = group_samples[group1], group_samples[group2]
            for lab in label_order:
                x1 = transformed.loc[samples1, lab].values
                x2 = transformed.loc[samples2, lab].values
                p1 = sample_prop.loc[samples1, lab].values
                p2 = sample_prop.loc[samples2, lab].values
                sd1, sd2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
                if np.isclose(sd1, 0) and np.isclose(sd2, 0):
                    if np.isclose(x1.mean(), x2.mean()):
                        t_stat, p_value = 0.0, 1.0
                    else:
                        t_stat, p_value = np.nan, np.nan
                else:
                    res = scipy_stats.ttest_ind(x1, x2, equal_var=False, alternative="two-sided")
                    t_stat, p_value = res.statistic, res.pvalue
                mean1, mean2 = 100 * p1.mean(), 100 * p2.mean()
                rows.append({
                    label_col: lab,
                    "comparison": f"{group1}_vs_{group2}",
                    "group1": group1, "group2": group2,
                    "n_group1": len(p1), "n_group2": len(p2),
                    "mean_pct_group1": mean1, "mean_pct_group2": mean2,
                    "difference_pct_points": mean1 - mean2,
                    "welch_t": t_stat, "p_value": p_value,
                })

        results = pd.DataFrame(
            rows,
            columns=[
                label_col,
                "comparison",
                "group1",
                "group2",
                "n_group1",
                "n_group2",
                "mean_pct_group1",
                "mean_pct_group2",
                "difference_pct_points",
                "welch_t",
                "p_value",
            ],
        )
        results["q_value"] = np.nan
        for comparison, sub in results.groupby("comparison"):
            valid = sub["p_value"].notna()
            if valid.sum() == 0:
                continue
            qvals = multipletests(sub.loc[valid, "p_value"], method="fdr_bh")[1]
            results.loc[sub.loc[valid].index, "q_value"] = qvals
        results["significant_FDR05"] = results["q_value"] < 0.05

        stats_file = TABLES_DIR / f"{file_slug}_pairwise_proportion_stats.csv"
        prop_file = TABLES_DIR / f"{file_slug}_proportions_per_sample.csv"
        results.round(5).to_csv(stats_file, index=False)
        (100 * sample_prop).round(4).to_csv(prop_file)
        print(f"Saved: {stats_file}")
        print(f"Saved: {prop_file}")

        group_means = pd.DataFrame(
            {g: 100 * sample_prop.loc[group_samples[g]].mean(axis=0) for g in present_groups}
        )
        plot_labels = (
            group_means.max(axis=1).loc[lambda x: x >= min_plot_pct]
            .sort_values(ascending=False).index.tolist()
        )

        if not plot_labels or not comparisons:
            return {
                "pairwise_stats": f"tables/{stats_file.name}",
                "per_sample": f"tables/{prop_file.name}",
            }, []

        long_prop = (
            (100 * sample_prop[plot_labels]).reset_index()
            .melt(id_vars="sample_id", var_name=label_col, value_name="pct")
        )
        long_prop["group"] = long_prop["sample_id"].map(sample_group.to_dict())
        palette = {g: GROUP_PALETTE.get(g, "#9E9E9E") for g in present_groups}

        stats_plots = []
        for group1, group2 in comparisons:
            comparison = f"{group1}_vs_{group2}"
            dat = long_prop[long_prop["group"].isin([group1, group2])].copy()
            fig_width = max(12, 0.9 * len(plot_labels))
            fig, ax = plt.subplots(figsize=(fig_width, 7))
            sns.barplot(data=dat, x=label_col, y="pct", hue="group",
                        hue_order=[group1, group2], order=plot_labels,
                        palette=palette, errorbar=None, alpha=0.50, ax=ax)
            sns.stripplot(data=dat, x=label_col, y="pct", hue="group",
                          hue_order=[group1, group2], order=plot_labels,
                          palette=palette, dodge=True, edgecolor="black",
                          linewidth=0.6, size=6, ax=ax)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[:2], labels[:2], title="Group",
                      bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)

            stat_sub = results[results["comparison"] == comparison].set_index(label_col)
            global_max = max(dat["pct"].max(), 1)
            pad = max(global_max * 0.035, 0.15)
            for i, lab in enumerate(plot_labels):
                q = stat_sub.loc[lab, "q_value"]
                lab_max = dat.loc[dat[label_col] == lab, "pct"].max()
                if np.isnan(q):
                    txt = "q=NA"
                else:
                    star = _stars(q)
                    txt = f"q={q:.2g}" + (f" {star}" if star else "")
                ax.text(i, lab_max + pad, txt, ha="center", va="bottom", fontsize=8, rotation=90)

            ax.set_xlabel(label_noun.capitalize())
            ax.set_ylabel(f"Proportion within sample ({region_noun}, %)")
            title = f"{group1} vs {group2} — {region_noun}\nPairwise Welch tests; q = BH-FDR adjusted p-value"
            if cohort_name:
                title += f"\n{cohort_name}"
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=45)
            for lbl in ax.get_xticklabels():
                lbl.set_ha("right")
            ax.set_ylim(0, global_max + 6 * pad)
            fig.tight_layout()
            plot_name = f"{file_slug}_proportions_{group1}_vs_{group2}_stats"
            save_fig(fig, plot_name, dpi=180)
            stats_plots.append({"title": f"{group1} vs {group2}", "path": f"plots/{plot_name}.png"})
            print(f"Saved: {PLOTS_DIR / (plot_name + '.png')}")

        csvs = {
            "pairwise_stats": f"tables/{stats_file.name}",
            "per_sample": f"tables/{prop_file.name}",
        }
        return csvs, stats_plots


    for view_key, view in CELLTYPE_VIEWS.items():
        print(f"\n{'=' * 70}\n{view['label']}\n{'=' * 70}")
        col = view["column"]
        slug = view["file_slug"]
        vmask = region_mask(view["region_key"])
        by_group_csv = write_proportions_by_group_csv(col, slug, vmask)
        csvs, stats_plots = run_pairwise_proportion_stats(
            col, slug, view["res_label"], view["region_label"], vmask, min_plot_pct=1.0
        )
        csvs["proportions_by_group"] = by_group_csv
        celltype_results[view_key]["csvs"] = csvs
        celltype_results[view_key]["stats_plots"] = stats_plots

    print("\nProportion CSVs + pairwise stats written for:", view_keys)


    def write_raw_counts_csv(label_col, file_slug, cell_mask):
        """Number of cells per sample x cell type (rows=sample, cols=cell type)."""
        valid = adata.obs[cell_mask].dropna(subset=[label_col]).copy()
        valid["sample_id"] = valid["sample_id"].astype(str)
        counts = pd.crosstab(valid["sample_id"], valid[label_col])
        counts = counts.reindex(columns=counts.sum(axis=0).sort_values(ascending=False).index)
        counts.insert(0, "group", counts.index.map(group_map))
        counts.index.name = "sample_id"
        out = TABLES_DIR / f"{file_slug}_cell_counts_per_sample.csv"
        counts.to_csv(out)
        print(f"Saved: {out}")
        return f"tables/{out.name}"


    for view_key, view in CELLTYPE_VIEWS.items():
        vmask = region_mask(view["region_key"])
        raw_csv = write_raw_counts_csv(view["column"], view["file_slug"], vmask)
        celltype_results[view_key].setdefault("csvs", {})["raw_counts"] = raw_csv

    print("\nRaw cell-count CSVs written for:", view_keys)

    n_total = adata.n_obs
    sample_counts = adata.obs["sample_id"].value_counts()
    samples_summary = []
    for sid in sorted(adata.obs["sample_id"].unique()):
        mask = adata.obs["sample_id"] == sid
        samples_summary.append({
            "sample_id": sid,
            "treatment": group_map.get(sid, "unknown"),
            "n_cells": int(sample_counts[sid]),
            "pct_total": 100.0 * sample_counts[sid] / n_total,
            "median_transcripts": float(adata.obs.loc[mask, "n_transcripts"].median()),
        })

    n_hvg = int(adata.var["highly_variable"].sum()) if "highly_variable" in adata.var else 0
    n_clusters = int(adata.obs["leiden"].nunique()) if "leiden" in adata.obs else 0

    STEP03_REL = "../03_normalize_dimred/plots"
    plots = {
        "per_sample_comparison":       f"{STEP03_REL}/per_sample_comparison.png",
        "normalization_distributions": f"{STEP03_REL}/normalization_distributions.png",
        "leiden_resolution_sweep":     f"{STEP03_REL}/leiden_resolution_sweep.png",
        "umap_overview":               f"{STEP03_REL}/umap_overview.png",
        "pseudobulk_pca_treatment":    f"{STEP03_REL}/pseudobulk_pca_by_treatment.png",
        "pseudobulk_pca_experiment":   f"{STEP03_REL}/pseudobulk_pca_by_experiment.png",
    }

    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).parent / "templates")),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("pipeline_report.html.j2")

    html = template.render(
        cohort_name=cohort.get("cohort_name", "Cohort"),
        generated_at=datetime.now().strftime("%B %d, %Y"),
        n_cells=adata.n_obs,
        n_genes=adata.n_vars,
        n_samples=len(samples_summary),
        norm_method=NORM_METHOD,
        n_hvg=n_hvg,
        n_pcs=N_PCS,
        n_neighbors=N_NEIGHBORS,
        leiden_resolution=LEIDEN_RESOLUTION,
        n_clusters=n_clusters,
        n_top_genes=N_TOP_GENES,
        samples=samples_summary,
        groups=groups,
        view_keys=view_keys,
        region_keys=region_keys,
        celltype_results=celltype_results,
        plots=plots,
        config_yaml=yaml.safe_dump(cohort, sort_keys=False, default_flow_style=False),
    )

    report_path = OUTPUT_DIR / "pipeline_report.html"
    report_path.write_text(html)
    print(f"Report written: {report_path}")

def main():
    run(parse_args())


if __name__ == "__main__":
    main()
