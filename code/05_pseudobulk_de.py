#!/usr/bin/env python3
"""Run sample-level pseudobulk edgeR DE with all genes and an exploratory-only marker-filtered view whose p-values are not calibrated because the same data defines markers and tests DE.
Write tables, plots, an Excel workbook, and regional HTML reports under 05_pseudobulk_de/."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter

from config_utils import cohort_paths, load_cohort
from pipeline_utils import (
    add_gene_metadata,
    aggregate_pseudobulk,
    keep_mouse_and_human_mapt,
    safe_name,
)


def run_r(cmd, check=True):
    """Run Rscript directly from PATH and report command failures."""
    result = subprocess.run([str(x) for x in cmd], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[R] command failed (exit {result.returncode}):")
        print(" ".join(map(str, cmd)))
        print("STDOUT:\n" + result.stdout)
        print("STDERR:\n" + result.stderr)
        if check:
            raise subprocess.CalledProcessError(result.returncode, cmd)
    return result


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path, help="Path to cohort YAML")
    ap.add_argument("--cell-type-level", default="coarse",
                    help="'coarse' (default, uses obs['cell_type'] and the configured coarse classes) "
                         "or any obs column name (e.g. 'atlas_class') to test at a different resolution.")
    ap.add_argument("--significance-metric", choices=["fdr", "pvalue"], default=None,
                    help="Which column the --alpha cutoff is applied to when calling genes "
                         "significant: 'fdr' (Benjamini-Hochberg adjusted p, the default) or "
                         "'pvalue' (edgeR's raw p-value). Overrides config `de_params.significance_metric`. "
                         "edgeR always reports both columns; this only changes which one gates significance.")
    ap.add_argument("--alpha", type=float, default=None,
                    help="Significance cutoff applied to the chosen metric "
                         "(default 0.05, or config `de_params.alpha`).")
    ap.add_argument("--lfc", type=float, default=None,
                    help="|log2FC| cutoff for 'significant' (default 0.25, or config `de_params.lfc`).")
    ap.add_argument("--min-cells", type=int, default=30, help="Min cells per pseudobulk sample (default 30).")
    ap.add_argument("--no-min-cells-filter", action="store_true",
                    help="Disable the per-pseudobulk-sample min-cells threshold (keep every sample "
                         "regardless of cell count; equivalent to --min-cells 0). edgeR still needs "
                         ">=2 samples per group, so --min-samples is unaffected.")
    ap.add_argument("--min-samples", type=int, default=2, help="Min samples per group (default 2).")
    ap.add_argument("--covariates", default="none",
                    help="Comma-separated covariate name(s) to add as blocking factors in the "
                         "edgeR design (e.g. 'experiment'). Must be defined in the config's "
                         "`covariates:` section. Default 'none'. The design becomes "
                         "~ 0 + group + <covariates>; treatment contrasts are still over group only. "
                         "A covariate that is constant within a given region x cell_type block is "
                         "dropped for that model to avoid a rank-deficient design.")
    ap.add_argument("--run-label", default=None,
                    help="Optional label to namespace outputs so repeated runs do not overwrite each "
                         "other. When set, ALL outputs go under outputs/05_pseudobulk_de/<run-label>/ "
                         "instead of the step top level. Use e.g. --run-label with_experiment when "
                         "passing --covariates experiment.")
    ap.add_argument("--skip-subset", action="store_true",
                    help="Only run whole_brain even if polygon files exist.")
    args = ap.parse_args(argv)
    if args.no_min_cells_filter:


        args.min_cells = 0
    return args


def load_inputs(args):
    """Load config and the annotated h5ad; validate; return a context dict."""

    if not args.config.exists():
        sys.exit(f"[error] config not found: {args.config}")
    cohort = load_cohort(args.config)


    de_params = cohort.get("de_params", {}) or {}
    if args.alpha is None:
        args.alpha = float(de_params.get("alpha", 0.05))
    if args.lfc is None:
        args.lfc = float(de_params.get("lfc", 0.25))
    if args.significance_metric is None:
        args.significance_metric = str(de_params.get("significance_metric", "fdr")).lower()
    if args.significance_metric not in ("fdr", "pvalue"):
        sys.exit(f"[error] significance_metric must be 'fdr' or 'pvalue', got "
                 f"{args.significance_metric!r}.")

    args.sig_col = "PValue" if args.significance_metric == "pvalue" else "FDR"
    args.metric_axis_label = "p-value" if args.significance_metric == "pvalue" else "FDR"
    args.metric_prose = ("raw p-value (PValue column)" if args.significance_metric == "pvalue"
                         else "FDR (Benjamini-Hochberg adjusted p-value)")

    here = Path(__file__).resolve().parent
    paths = cohort_paths(cohort)
    root = paths.project_root


    step_root = paths.output("05_pseudobulk_de")
    if args.run_label:
        step_root = step_root / args.run_label
    ctx = {
        "cohort": cohort,
        "paths": paths,
        "root": root,
        "output_dir": paths.output_dir,
        "step_root": step_root,
        "results_dir": step_root / "tables",
        "plots_dir": step_root / "plots",
        "input_h5ad": paths.output(
            "04b_select_rois", "combined_annotated_with_rois.h5ad"
        ),
        "group_map": cohort["group_map"],
        "comparisons": [tuple(c) for c in cohort["comparisons"]],
        "covariate_maps": cohort.get("covariates", {}) or {},
    }
    ctx["results_dir"].mkdir(parents=True, exist_ok=True)
    ctx["plots_dir"].mkdir(parents=True, exist_ok=True)
    level_name = "coarse" if args.cell_type_level == "coarse" else args.cell_type_level
    ctx["level_name"] = level_name
    (ctx["plots_dir"] / "volcano_panels").mkdir(parents=True, exist_ok=True)
    (ctx["plots_dir"] / "volcano_individual").mkdir(parents=True, exist_ok=True)

    edger_script = here / "05_run_edger.R"
    ctx["edger_script"] = edger_script

    adata = sc.read_h5ad(ctx["input_h5ad"])
    add_gene_metadata(adata)
    if "in_roi" not in adata.obs:
        sys.exit(
            "[error] obs['in_roi'] is missing. Run 04b_select_rois.py before step 05."
        )


    if args.cell_type_level == "coarse":
        class_col = "cell_type"
    else:
        class_col = args.cell_type_level
    if class_col not in adata.obs.columns:
        sys.exit(f"[error] --cell-type-level '{class_col}' is not an obs column. "
                 f"Available: {sorted(adata.obs.columns)}")

    present = [str(c) for c in pd.unique(adata.obs[class_col].astype(str))]
    configured_cell_types = (cohort.get("de_params", {}) or {}).get("cell_types")
    if configured_cell_types:
        cell_type_set = [
            str(cell_type) for cell_type in configured_cell_types
            if str(cell_type) in present
        ]
    else:
        cell_type_set = present

    required_obs = {"sample_id", "label_confidence", class_col}
    missing = required_obs - set(adata.obs.columns)
    if missing:
        sys.exit(f"[error] missing obs columns: {sorted(missing)}")
    if "counts" not in adata.layers:
        sys.exit("[error] adata.layers['counts'] missing; pseudobulk DE needs raw counts.")
    if adata.var_names.has_duplicates:
        sys.exit("[error] adata.var_names has duplicates.")

    gene_mask = keep_mouse_and_human_mapt(adata.var).to_numpy()
    base_mask = (
        (adata.obs["label_confidence"] == "high")
        & adata.obs[class_col].astype(str).isin([str(c) for c in cell_type_set])
        & adata.obs["sample_id"].isin(ctx["group_map"])
    ).to_numpy()

    if not base_mask.any():
        sys.exit(
            "[error] no cells passed the base filter (label_confidence=='high' & "
            f"obs['{class_col}'] in cell_type_set & sample_id in group_map). "
            f"cell_type_set={cell_type_set}. "
            f"Present obs['{class_col}'] values: {sorted(present)}. "
            "Check that --cell-type-level matches how step 04 labelled cells."
        )

    ctx.update({
        "adata": adata,
        "class_col": class_col,
        "cell_type_set": cell_type_set,
        "gene_mask": gene_mask,
        "base_mask": base_mask,
        "roi_mask": adata.obs["in_roi"].astype(bool).to_numpy(),


        "safe_to_cell_type": {safe_name(c): str(c) for c in cell_type_set},
    })
    print(f"[load] {adata.n_obs:,} cells x {adata.n_vars:,} features")
    print(f"[scope] level={args.cell_type_level} (obs['{class_col}']) | "
          f"cell types={cell_type_set}")
    print(f"[scope] cells in base_mask: {base_mask.sum():,} | genes: {int(gene_mask.sum())}")
    return ctx


def build_pseudobulk(ctx, args):
    """Aggregate whole_brain and the optional step-04b ROI."""
    adata = ctx["adata"]
    base_mask = ctx["base_mask"]
    class_col = ctx["class_col"]

    all_counts, all_meta = [], []
    wc, wm = aggregate_pseudobulk(
        adata, mask=base_mask, gene_mask=ctx["gene_mask"],
        group_map=ctx["group_map"], region_name="whole_brain", class_col=class_col,
        covariate_maps=ctx["covariate_maps"])
    all_counts.append(wc)
    all_meta.append(wm)

    regions = ["whole_brain"]
    if not args.skip_subset:
        sub_mask = ctx["roi_mask"]
        if sub_mask.any():
            region_name = str(
                (ctx["cohort"].get("regions", {}) or {}).get("name", "subset")
            )
            sc_, sm_ = aggregate_pseudobulk(
                adata, mask=base_mask & sub_mask, gene_mask=ctx["gene_mask"],
                group_map=ctx["group_map"], region_name=region_name, class_col=class_col,
                covariate_maps=ctx["covariate_maps"])
            all_counts.append(sc_)
            all_meta.append(sm_)
            regions.append(region_name)
            print(f"[subset] {region_name}: {int(sub_mask.sum()):,} cells inside polygons")
        else:
            print("[subset] step 04b selected no ROI cells — only whole_brain will be tested.")

    counts_df = pd.concat(all_counts, axis=0)
    metadata_df = pd.concat(all_meta, axis=0)
    counts_df.to_csv(ctx["results_dir"] / "pseudobulk_counts.csv")
    metadata_df.to_csv(ctx["results_dir"] / "pseudobulk_metadata.csv")

    ctx["counts_df"] = counts_df
    ctx["metadata_df"] = metadata_df
    ctx["regions"] = regions
    print(f"[pseudobulk] rows: {counts_df.shape[0]:,} | features: {counts_df.shape[1]:,} | regions: {regions}")
    return ctx


def run_edger(ctx, args):
    """edgeR dependency check, then one QLF model per region x cell_type."""

    results_dir = ctx["results_dir"]
    plots_dir = ctx["plots_dir"]
    metadata_df = ctx["metadata_df"]
    comparisons = ctx["comparisons"]

    comparisons_arg = ";".join(f"{g1},{g2}" for g1, g2 in comparisons)
    covariates_arg = args.covariates.strip() if args.covariates else "none"
    if not covariates_arg:
        covariates_arg = "none"
    if covariates_arg.lower() != "none":
        requested = [c for c in covariates_arg.split(",") if c]
        defined = set(ctx.get("covariate_maps", {}))
        undefined = [c for c in requested if c not in defined]
        if undefined:
            sys.exit(f"[error] --covariates {undefined} not defined in the config's "
                     f"`covariates:` section. Defined: {sorted(defined)}")
        missing = [c for c in requested if c not in metadata_df.columns]
        if missing:
            sys.exit(f"[error] --covariates column(s) not in pseudobulk metadata: {missing}. "
                     f"Available: {sorted(metadata_df.columns)}")
        print(f"[edgeR] covariates: {covariates_arg}")

    check = run_r(
        ["Rscript", "-e",
         "cat(R.version.string, '\\n'); if (!requireNamespace('edgeR', quietly=TRUE)) quit(status=10)"],
        check=False)
    if check.returncode != 0:
        sys.exit("[error] Rscript/edgeR unavailable. Pseudobulk CSVs were written; "
                 "load R + edgeR and rerun.")
    print("[edgeR] available: " + check.stdout.strip())

    edger_dir = results_dir / "edger"
    edger_plot_dir = plots_dir / "edger_diagnostics"
    edger_dir.mkdir(parents=True, exist_ok=True)
    edger_plot_dir.mkdir(parents=True, exist_ok=True)

    skip_rows, out_files = [], []
    for region in metadata_df["region"].unique():
        region_meta = metadata_df[metadata_df["region"] == region]
        for cell_type in ctx["cell_type_set"]:
            class_meta = region_meta[region_meta["cell_type"] == cell_type]
            class_meta = class_meta[class_meta["n_cells"] >= args.min_cells]
            sample_counts = class_meta["group"].value_counts()
            runnable = [
                (g1, g2) for g1, g2 in comparisons
                if sample_counts.get(g1, 0) >= args.min_samples
                and sample_counts.get(g2, 0) >= args.min_samples
            ]
            if not runnable:
                skip_rows.append({"region": region, "cell_type": cell_type,
                                  "comparison": "all",
                                  "reason": "no_comparison_has_enough_samples",
                                  "n_samples": int(len(class_meta))})
                continue
            cmd = [
                "Rscript", str(ctx["edger_script"]),
                "--counts", str(results_dir / "pseudobulk_counts.csv"),
                "--metadata", str(results_dir / "pseudobulk_metadata.csv"),
                "--comparisons", comparisons_arg,
                "--region", region,
                "--cell-type", cell_type,
                "--min-cells", str(args.min_cells),
                "--covariates", covariates_arg,
                "--out-dir", str(edger_dir),
                "--plot-dir", str(edger_plot_dir),
            ]
            print(f"[edgeR] {region} / {cell_type}")
            run_r(cmd, check=True)
            safe_ct = safe_name(cell_type)
            for g1, g2 in comparisons:
                f = edger_dir / f"{region}__{safe_ct}__{g1}_vs_{g2}.csv"
                if f.exists():
                    out_files.append(f)

    pd.DataFrame(skip_rows).to_csv(results_dir / "edger_skipped_tests.csv", index=False)
    ctx["edger_dir"] = edger_dir
    ctx["edger_plot_dir"] = edger_plot_dir
    ctx["out_files"] = out_files
    print(f"[edgeR] result files: {len(out_files)}")
    return ctx


def combine_results(ctx):
    adata = ctx["adata"]
    counts_df = ctx["counts_df"]
    metadata_df = ctx["metadata_df"]
    results_dir = ctx["results_dir"]

    gene_info = adata.var.loc[ctx["gene_mask"], ["feature_name", "gene_symbol", "species"]] \
        .drop_duplicates("feature_name")

    cpm_rows = []
    for (region, cell_type, group), meta_sub in metadata_df.groupby(
            ["region", "cell_type", "group"], observed=True):
        sub = counts_df.loc[meta_sub.index]
        cpm = sub.div(sub.sum(axis=1).replace(0, np.nan), axis=0) * 1e6
        mean_cpm = cpm.mean(axis=0).rename("mean_cpm").reset_index().rename(columns={"index": "feature_name"})
        mean_cpm["region"] = region
        mean_cpm["cell_type"] = cell_type
        mean_cpm["group"] = group
        cpm_rows.append(mean_cpm)
    cpm_summary = pd.concat(cpm_rows, ignore_index=True) if cpm_rows else pd.DataFrame(
        columns=["feature_name", "region", "cell_type", "group", "mean_cpm"])
    cpm_summary.to_csv(results_dir / "pseudobulk_mean_cpm_by_group.csv", index=False)

    rows = []
    for f in ctx["out_files"]:
        df = pd.read_csv(f)
        region, cell_type, comp = f.stem.split("__", 2)


        cell_type = ctx.get("safe_to_cell_type", {}).get(cell_type, cell_type)
        df["region"] = region
        df["cell_type"] = cell_type
        df["comparison"] = comp
        rows.append(df)
    if rows:
        de_all = pd.concat(rows, ignore_index=True)
    else:
        de_all = pd.DataFrame(columns=["feature_name", "region", "cell_type", "comparison",
                                       "group1", "group2", "logFC", "PValue", "FDR"])
    de_all = de_all.merge(gene_info, on="feature_name", how="left")
    if len(cpm_summary):
        de_all = de_all.merge(
            cpm_summary.rename(columns={"group": "group1", "mean_cpm": "mean_cpm_group1"}),
            on=["region", "cell_type", "group1", "feature_name"], how="left")
        de_all = de_all.merge(
            cpm_summary.rename(columns={"group": "group2", "mean_cpm": "mean_cpm_group2"}),
            on=["region", "cell_type", "group2", "feature_name"], how="left")
    de_all.to_csv(results_dir / "pseudobulk_de_all_genes.csv", index=False)
    ctx["de_all"] = de_all
    print(f"[combine] de_all rows: {len(de_all):,}")
    return ctx


def de_count_summary(de_tbl, alpha, lfc, sig_col="FDR"):
    if not len(de_tbl):
        return pd.DataFrame(columns=["region", "cell_type", "comparison",
                                     "n_tested", "n_sig", "n_up", "n_down"])

    def _counts(df):
        tested = df[df[sig_col].notna()]
        sig = tested[(tested[sig_col] < alpha) & (tested["logFC"].abs() > lfc)]
        return pd.Series({
            "n_tested": int(len(tested)),
            "n_sig": int(len(sig)),
            "n_up": int((sig["logFC"] > 0).sum()),
            "n_down": int((sig["logFC"] < 0).sum()),
            "min_p": float(tested[sig_col].min()) if len(tested) else np.nan,
        })
    return (de_tbl.groupby(["region", "cell_type", "comparison"], observed=True)
            .apply(_counts, include_groups=False).reset_index())


def marker_filter(ctx, args):
    """Recompute class-enriched markers within EACH region at the active level, then restrict that region's DE results to marker (cell_type, feature) pairs."""
    adata = ctx["adata"]
    base_mask = ctx["base_mask"]
    class_col = ctx["class_col"]
    de_all = ctx["de_all"]
    results_dir = ctx["results_dir"]


    region_masks = {"whole_brain": base_mask}
    subset_regions = [region for region in ctx["regions"] if region != "whole_brain"]
    if subset_regions:
        region_masks[subset_regions[0]] = base_mask & ctx["roi_mask"]

    marker_frames = []
    for region, mask in region_masks.items():
        if not mask.any():
            continue
        view = adata[mask, ctx["gene_mask"]].copy()

        view.obs["_class"] = view.obs[class_col].astype(str)


        # Scanpy's Wilcoxon test requires at least two cells per group.
        marker_min_cells = max(args.min_cells, 2)
        group_sizes = view.obs["_class"].value_counts()
        keep_groups = group_sizes[group_sizes >= marker_min_cells].index
        dropped = sorted(set(group_sizes.index) - set(keep_groups))
        if dropped:
            print(f"[marker] {region}: dropping {len(dropped)} group(s) with "
                  f"<{marker_min_cells} cells before marker test: {dropped}")
            view = view[view.obs["_class"].isin(keep_groups)].copy()


        # Remove unused categorical levels before rank_genes_groups.
        view.obs["_class"] = view.obs["_class"].astype(str)

        if view.obs["_class"].nunique() < 2:
            print(f"[marker] {region}: <2 cell types with >={args.min_cells} "
                  f"cells, skipping marker filter for this region.")
            continue
        sc.tl.rank_genes_groups(view, groupby="_class", method="wilcoxon", pts=True)
        rgg = view.uns["rank_genes_groups"]
        for grp in list(rgg["names"].dtype.names):
            block = pd.DataFrame({
                "region": region,
                "cell_type": grp,
                "feature_name": rgg["names"][grp],
                "logfoldchanges": rgg["logfoldchanges"][grp],
                "pvals_adj": rgg["pvals_adj"][grp],
                "scores": rgg["scores"][grp],
            })
            pts = rgg["pts"][grp] if "pts" in rgg else None
            block["pts"] = block["feature_name"].map(pts) if pts is not None else np.nan
            marker_frames.append(block)

    if not marker_frames:
        de_all["enriched_marker"] = False
        ctx["de_all"] = de_all
        de_all.to_csv(results_dir / "pseudobulk_de_all_genes.csv", index=False)
        ctx["de_marker"] = de_all.iloc[0:0].copy()
        ctx["marker_warning"] = MARKER_WARNING
        print("[marker] no markers computed; enriched_marker column set all-False.")
        return ctx

    markers = pd.concat(marker_frames, ignore_index=True)
    markers.to_csv(results_dir / "class_enriched_markers.csv", index=False)

    marker_features = markers[
        (markers["pvals_adj"] < args.alpha)
        & (markers["logfoldchanges"] > 0)
        & (markers["pts"] > 0.05)
    ].copy()
    marker_features.to_csv(results_dir / "class_enriched_marker_features_used.csv", index=False)


    marker_keys = marker_features[["region", "cell_type", "feature_name"]].assign(enriched_marker=True)
    de_all = de_all.merge(marker_keys, on=["region", "cell_type", "feature_name"], how="left")
    de_all["enriched_marker"] = de_all["enriched_marker"].fillna(False).astype(bool)
    de_all.to_csv(results_dir / "pseudobulk_de_all_genes.csv", index=False)
    ctx["de_all"] = de_all


    de_marker = de_all[de_all["enriched_marker"]].copy()
    de_marker.to_csv(results_dir / "pseudobulk_de_class_marker_genes.csv", index=False)

    ctx["de_marker"] = de_marker
    ctx["marker_warning"] = MARKER_WARNING
    print(f"[marker] class markers: {len(marker_features):,} | filtered DE rows: {len(de_marker):,}")
    return ctx


MARKER_WARNING = (
    "Class-marker-restricted results are EXPLORATORY ONLY. The same expression data "
    "both defines the markers (cell-type-vs-rest Wilcoxon) and is then tested for "
    "treatment differential expression, which is statistically circular ('double "
    "dipping'). Do not treat these p-values / FDR as calibrated. The all-genes "
    "results are the primary, statistically valid result."
)


def _safe(s):
    """Make a string safe for a flat filename."""
    return "".join(c if (c.isalnum() or c in "-_") else "-" for c in str(s))


def _draw_volcano(axis, frame, title, args, max_labels=18):
    frame = frame[
        frame["logFC"].notna() & frame[args.sig_col].notna()
    ].copy()
    if frame.empty:
        axis.text(0.5, 0.5, "No tested genes", ha="center", va="center")
        axis.set_title(title)
        return False

    frame["_y"] = -np.log10(frame[args.sig_col].clip(lower=1e-300))
    significant = (
        (frame[args.sig_col] < args.alpha)
        & (frame["logFC"].abs() > args.lfc)
    )
    up = significant & (frame["logFC"] > 0)
    down = significant & (frame["logFC"] < 0)
    axis.scatter(frame.loc[~significant, "logFC"], frame.loc[~significant, "_y"],
                 color="#bdbdbd", s=12, alpha=0.7)
    axis.scatter(frame.loc[up, "logFC"], frame.loc[up, "_y"],
                 color="#d73027", s=18, alpha=0.85)
    axis.scatter(frame.loc[down, "logFC"], frame.loc[down, "_y"],
                 color="#4575b4", s=18, alpha=0.85)

    labels = frame.loc[significant].nlargest(max_labels, "_y")
    for _, row in labels.iterrows():
        axis.annotate(
            str(row.get("gene_symbol", row.get("feature_name", ""))),
            (row["logFC"], row["_y"]),
            fontsize=6,
            xytext=(3, 3),
            textcoords="offset points",
        )
    axis.axhline(-np.log10(args.alpha), color="#666666", ls="--", lw=0.7)
    axis.axvline(args.lfc, color="#666666", ls="--", lw=0.7)
    axis.axvline(-args.lfc, color="#666666", ls="--", lw=0.7)
    axis.set_title(f"{title}\n({int(significant.sum())} sig)", fontsize=10)
    return True


def _save_static_figure(fig, path, dpi=180):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    plt.close(fig)


def _plot_volcano(frame, path, title, args):
    fig, axis = plt.subplots(figsize=(6, 5))
    made = _draw_volcano(axis, frame, title, args)
    if made:
        axis.set_xlabel("log2 fold change")
        axis.set_ylabel(f"-log10 {args.metric_axis_label}")
        fig.tight_layout()
        _save_static_figure(fig, path)
    else:
        plt.close(fig)
    return made


def _plot_volcano_panels(de_table, output_dir, cell_types, args):
    saved = []
    for (region, comparison), frame in de_table.groupby(
        ["region", "comparison"], observed=True
    ):
        present = [cell_type for cell_type in cell_types
                   if cell_type in set(frame["cell_type"])]
        if not present:
            continue
        n_columns = min(3, len(present))
        n_rows = int(np.ceil(len(present) / n_columns))
        fig, axes = plt.subplots(
            n_rows, n_columns,
            figsize=(4.2 * n_columns, 3.8 * n_rows),
            squeeze=False,
        )
        for axis, cell_type in zip(axes.ravel(), present):
            _draw_volcano(
                axis,
                frame[frame["cell_type"] == cell_type],
                cell_type,
                args,
            )
            axis.set_xlabel("log2 fold change")
            axis.set_ylabel(f"-log10 {args.metric_axis_label}")
        for axis in axes.ravel()[len(present):]:
            axis.axis("off")
        fig.suptitle(f"{region}: {comparison}", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.99])
        path = output_dir / region / f"volcano_panel_{comparison}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        _save_static_figure(fig, path, dpi=200)
        saved.append(path)
    return saved


def make_volcanoes(ctx, args):
    """Write primary and exploratory marker-filtered volcano PNG/SVG files under the step plots folder."""
    ind_dir = ctx["plots_dir"] / "volcano_individual"
    pan_dir = ctx["plots_dir"] / "volcano_panels"
    cto = ctx["cell_type_set"]

    genesets = [("all_genes", ctx["de_all"])]
    if len(ctx.get("de_marker", [])):
        genesets.append(("markers", ctx["de_marker"]))

    individual, panels = {}, {}
    for set_name, de_tbl in genesets:
        if de_tbl is None or not len(de_tbl):
            continue


        ind_paths = []
        if set_name == "all_genes":
            for (region, cell_type, comp), df in de_tbl.groupby(
                    ["region", "cell_type", "comparison"], observed=True):
                out = ind_dir / f"{_safe(region)}__{_safe(cell_type)}__{_safe(comp)}.png"
                title = f"{region} | {cell_type} | {comp}"
                if _plot_volcano(df, out, title, args):
                    ind_paths.append(out)
        individual[set_name] = ind_paths


        with tempfile.TemporaryDirectory() as tmp:
            staged = _plot_volcano_panels(de_tbl, Path(tmp), cto, args)
            pan_paths = []
            for sp in staged:
                region = sp.parent.name
                comp = sp.stem.replace("volcano_panel_", "")
                dest = pan_dir / f"{_safe(region)}__{set_name}__{_safe(comp)}.png"
                shutil.copyfile(sp, dest)
                shutil.copyfile(sp.with_suffix(".svg"), dest.with_suffix(".svg"))
                pan_paths.append(dest)
        panels[set_name] = pan_paths

    ctx["individual_volcanoes"] = individual
    ctx["panel_volcanoes"] = panels
    n_ind = sum(len(v) for v in individual.values())
    n_pan = sum(len(v) for v in panels.values())
    print(f"[volcano] individual plot pairs: {n_ind} | panel plot pairs: {n_pan}")
    return ctx


_XLSX_KEY_COLS = ["gene_symbol", "feature_name", "cell_type", "comparison",
                  "logFC", "FDR", "PValue", "passed_edger_filter", "enriched_marker"]

_XLSX_COL_DOC = [
    ("gene_symbol", "Gene symbol (human-readable)."),
    ("feature_name", "Panel feature ID for the gene (matches the MERSCOPE panel)."),
    ("cell_type", "Cell type the test was run in (one edgeR model per cell type)."),
    ("comparison", "Treatment contrast, e.g. ASO-Ttbk1_vs_dPBS."),
    ("logFC", "log2 fold change, group1 vs group2 (positive = up in group1)."),
    ("FDR", "Benjamini-Hochberg adjusted p-value. Primary significance column."),
    ("PValue", "Raw p-value from edgeR's quasi-likelihood F-test."),
    ("passed_edger_filter",
     "TRUE if this gene passes edgeR's expression filter (filterByExpr) for this "
     "cell type / region. The filter is NOT applied — every gene is tested — so "
     "this flag lets you exclude low-expression genes downstream if desired. "
     "Low-expression genes (FALSE) have noisier, less reliable statistics."),
    ("enriched_marker",
     "TRUE if this gene is a class-enriched marker of this cell type in this "
     "region (cell-type-vs-rest Wilcoxon). Filtering enriched_marker = TRUE "
     "reproduces the EXPLORATORY marker-filtered volcano panels. See caveat below."),
    ("logCPM", "Average log2 CPM across samples (edgeR abundance)."),
    ("F", "edgeR quasi-likelihood F statistic."),
    ("region", "Tissue region: whole_brain or the configured polygon region."),
    ("group1", "First group in the contrast (numerator)."),
    ("group2", "Second group in the contrast (denominator)."),
    ("n_samples_group1", "Number of pseudobulk samples (mice) in group1."),
    ("n_samples_group2", "Number of pseudobulk samples (mice) in group2."),
    ("n_cells_group1", "Total cells aggregated into group1 pseudobulks."),
    ("n_cells_group2", "Total cells aggregated into group2 pseudobulks."),
    ("mean_cpm_group1", "Mean per-sample CPM in group1."),
    ("mean_cpm_group2", "Mean per-sample CPM in group2."),
    ("species", "Gene species annotation (mouse / human MAPT)."),
    ("status", "edgeR test status (tested / skipped reason)."),
]


def write_excel_de_table(ctx, args):
    """Write one Excel workbook with a README sheet and one sheet per region."""

    de_all = ctx["de_all"]
    out = ctx["step_root"] / f"DE_results_{ctx['level_name']}.xlsx"

    wb = Workbook()


    ws = wb.active
    ws.title = "README"
    head = Font(name="Arial", bold=True, size=12)
    sub = Font(name="Arial", bold=True, size=10)
    body = Font(name="Arial", size=10)
    wrap = Alignment(wrap_text=True, vertical="top")

    ws["A1"] = "Pseudobulk differential expression — results table"
    ws["A1"].font = head
    min_cells_txt = (f"Pseudobulk samples with fewer than {args.min_cells} cells are dropped."
                     if args.min_cells > 0
                     else "Per-sample cell-count filter DISABLED (--no-min-cells-filter): "
                          "every pseudobulk sample is kept regardless of cell count.")
    rows = [
        ("", ""),
        ("Cell-type level", ctx["level_name"]),
        ("Test", "edgeR quasi-likelihood F-test (glmQLFit + glmQLFTest, robust); "
                 "TMM normalization. All genes are tested; edgeR's expression "
                 "filter (filterByExpr) is recorded in passed_edger_filter but "
                 "NOT applied, so low-expression genes are flagged, not dropped."),
        ("Replication unit", "One pseudobulk sample per mouse (sample_id) per cell type "
                             "per region; raw counts are summed across that cell's cells."),
        ("Genes tested", "All mouse genes plus human MAPT. Blank control probes and other "
                         "human genes are excluded before testing."),
        ("Cells included", "Only cells with label_confidence == 'high', whose cell type is one "
                           "of the tested classes, and whose sample_id is assigned a group in "
                           "the cohort config are aggregated. All other cells are excluded."),
        ("Design", ("~ 0 + group" if (not args.covariates or args.covariates.lower() == "none")
                    else f"~ 0 + group + {args.covariates} (covariates are blocking factors; "
                         f"treatment contrasts are over group only, and any covariate constant "
                         f"within a region x cell_type block is dropped for that model)")),
        ("Sample filters", f"{min_cells_txt} A comparison is run for a cell type only when BOTH "
                           f"groups have at least {args.min_samples} pseudobulk samples "
                           f"(min samples/group = {args.min_samples}); otherwise it is skipped "
                           f"and logged to edger_skipped_tests.csv."),
        ("Significance thresholds", f"A gene is called significant when {args.metric_prose} < "
                                    f"{args.alpha} AND |log2FC| > {args.lfc}. These thresholds "
                                    f"define significance only; no genes are removed from the "
                                    f"table by them. Both PValue and FDR are reported for every "
                                    f"gene regardless of which one gates significance."),
        ("Sheets", "One sheet per region (whole_brain, plus any polygon ROI such as "
                   "configured subset). Each row is one gene tested in one cell type for one contrast."),
        ("", ""),
        ("IMPORTANT — enriched_marker caveat", ""),
    ]
    r = 3
    for k, v in rows:
        ws.cell(r, 1, k).font = sub if k else body
        c = ws.cell(r, 2, v)
        c.font = body
        c.alignment = wrap
        r += 1
    ws.cell(r, 2, MARKER_WARNING).font = body
    ws.cell(r, 2).alignment = wrap
    r += 2

    ws.cell(r, 1, "Column").font = sub
    ws.cell(r, 2, "Meaning").font = sub
    r += 1
    for col, desc in _XLSX_COL_DOC:
        ws.cell(r, 1, col).font = body
        d = ws.cell(r, 2, desc)
        d.font = body
        d.alignment = wrap
        r += 1
    ws.column_dimensions["A"].width = 26
    ws.column_dimensions["B"].width = 96


    hdr_font = Font(name="Arial", bold=True, color="FFFFFF")
    hdr_fill = PatternFill("solid", start_color="4575B4")

    regions = list(ctx.get("regions", []))
    if len(de_all):
        for rg in de_all["region"].unique():
            if rg not in regions:
                regions.append(rg)
    if not regions:
        regions = ["whole_brain"]

    for region in regions:
        sub_df = de_all[de_all["region"] == region].copy() if len(de_all) else de_all.iloc[0:0].copy()
        ordered = [c for c in _XLSX_KEY_COLS if c in sub_df.columns]
        ordered += [c for c in sub_df.columns if c not in ordered]
        sub_df = sub_df[ordered] if len(ordered) else sub_df
        if "FDR" in sub_df.columns:
            sub_df = sub_df.sort_values(["cell_type", "comparison", "FDR"],
                                        na_position="last")

        sheet_name = region[:31] or "region"
        sh = wb.create_sheet(sheet_name)
        sh.append(list(sub_df.columns))
        for cell in sh[1]:
            cell.font = hdr_font
            cell.fill = hdr_fill
        for _, row in sub_df.iterrows():
            sh.append([_xlsx_cell(v) for v in row.tolist()])
        sh.freeze_panes = "A2"
        for j, col in enumerate(sub_df.columns, start=1):
            width = 11
            if col in ("gene_symbol", "cell_type", "comparison", "feature_name"):
                width = 18
            sh.column_dimensions[get_column_letter(j)].width = width

    wb.save(out)
    ctx["excel_path"] = out
    print(f"[excel] DE table -> {out}")
    return ctx


def _xlsx_cell(v):
    """Coerce a pandas value into something openpyxl will write cleanly."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if np.isnan(v) else float(v)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _img_tag(path, rel_to=None, width="100%"):
    """Reference an image with a path relative to the report."""
    path = Path(path)
    if not path.exists():
        return f'<p class="missing">[missing: {path.name}]</p>'
    src = path.name
    if rel_to is not None:
        try:
            src = path.relative_to(rel_to).as_posix()
        except ValueError:
            src = path.name
    return (f'<img src="{src}" '
            f'style="max-width:{width};border:1px solid #ddd;border-radius:6px;margin:6px 0;">')


def _df_html(df, max_rows=300):
    if df is None or not len(df):
        return '<p class="missing">[no rows]</p>'
    return df.head(max_rows).to_html(index=False, classes="tbl", border=0,
                                     float_format=lambda x: f"{x:.4g}")


_REPORT_CSS = """
body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; max-width: 1100px;
       margin: 24px auto; padding: 0 18px; color: #222; line-height: 1.5; }
h1 { border-bottom: 2px solid #4575b4; padding-bottom: 6px; }
h2 { margin-top: 34px; color: #2b4a7a; border-bottom: 1px solid #e3e3e3; padding-bottom: 4px; }
.note { background:#fffbe6; border-left:4px solid #d4a017; padding:10px 14px; border-radius:0 6px 6px 0; }
.missing { color:#b00; font-style:italic; }
table.tbl { border-collapse: collapse; font-size: 13px; margin: 8px 0; }
table.tbl th, table.tbl td { border: 1px solid #ddd; padding: 4px 8px; text-align: right; }
table.tbl th { background:#eef3f9; }
code { background:#f0f0f0; padding:1px 4px; border-radius:3px; }
"""


def build_region_report(ctx, args, region, index):
    """Write one HTML report for a region at the step output root."""
    de_all = ctx["de_all"]
    step_root = ctx["step_root"]
    panel_paths = ctx.get("panel_volcanoes", {})

    de_all_r = de_all[de_all["region"] == region] if len(de_all) else de_all

    sum_all = de_count_summary(de_all_r, args.alpha, args.lfc, sig_col=args.sig_col)

    if len(sum_all):
        sum_all = sum_all[sum_all["n_sig"] > 0].reset_index(drop=True)

    def _for_region(set_name):
        return [p for p in panel_paths.get(set_name, [])
                if Path(p).name.startswith(f"{_safe(region)}__{set_name}__")]

    all_panels = sorted(_for_region("all_genes"))

    excel_name = ctx["excel_path"].name if ctx.get("excel_path") else f"DE_results_{ctx['level_name']}.xlsx"


    min_cells_txt = ("per-sample cell filter disabled (every pseudobulk sample kept)"
                     if args.min_cells == 0
                     else f"min cells/pseudobulk = {args.min_cells}")
    if not args.covariates or args.covariates.lower() == "none":
        design_txt = "~ 0 + group"
    else:
        design_txt = (f"~ 0 + group + {args.covariates} (covariates are blocking factors; "
                      f"treatment contrasts are over group only)")


    pvalue_warning = ""
    if args.significance_metric == "pvalue":
        pvalue_warning = (
            '<p class="note"><b>Significance is based on the RAW p-value, not FDR.</b> '
            f'Genes are called significant at PValue &lt; {args.alpha}, which does NOT '
            'correct for multiple testing across the thousands of genes tested per cell '
            'type. Expect false positives (roughly '
            f'{args.alpha:.0%} of tested genes by chance alone). The FDR column is still '
            'reported for every gene in the results table — use it to gauge how many of '
            'these calls survive multiple-testing correction.</p>')

    sections = [pvalue_warning + f"""
<h2>Overview</h2>
<p>Pseudobulk differential expression for region <b>{region}</b>, cell-type level
<code>{args.cell_type_level}</code>. Test: edgeR quasi-likelihood F-test (glmQLFit +
glmQLFTest, robust), TMM normalization, design <code>{design_txt}</code>. Every gene
is tested; edgeR's expression filter (filterByExpr) is recorded per gene but not
applied.</p>
<p><b>Cells included:</b> only cells with <code>label_confidence == 'high'</code>, whose
cell type is one of the tested classes, and whose sample is assigned a group in the
cohort config. One pseudobulk sample is built per mouse per cell type by summing raw
counts.</p>
<p><b>Genes tested:</b> all mouse genes plus human MAPT (Blank control probes and other
human genes are excluded).</p>
<p><b>Sample filters:</b> {min_cells_txt}; a comparison runs for a cell type only when
both groups have at least {args.min_samples} pseudobulk samples (min samples/group =
{args.min_samples}).</p>
<p><b>Significance thresholds:</b> a gene is called significant when {args.metric_prose}
&lt; {args.alpha} and |log2FC| &gt; {args.lfc} (these define significance only; no genes
are removed from the table by them).</p>
<p>Full numeric results for every gene are in
<a href="{excel_name}"><code>{excel_name}</code></a> (sheet <b>{region[:31]}</b>).
Individual per-cell-type volcano PNGs are in
<code>plots/volcano_individual/</code> (named
<code>{region}__&lt;cell_type&gt;__&lt;comparison&gt;.png</code>).</p>
"""]

    sections.append("<h2>DE count summary — all genes</h2>"
                    "<p>Only cell type / comparison combinations with at least one "
                    "significant gene are listed.</p>" + _df_html(sum_all))
    sections.append("<h2>Volcano panels — all genes (primary result)</h2>"
                    "<p>One figure per treatment comparison; each subplot is a cell type.</p>"
                    + ("".join(_img_tag(p, rel_to=step_root) for p in all_panels) if all_panels
                       else '<p class="missing">[no all-genes panels for this region]</p>'))

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Pseudobulk DE — {region}</title><style>{_REPORT_CSS}</style></head><body>
<h1>Pseudobulk DE — {region}</h1>
<p style="color:#666;">Level {args.cell_type_level}</p>
{''.join(sections)}
<hr style="margin-top:32px;border:none;border-top:1px solid #ddd;">
<p style="color:#888;font-size:12px;">Generated {datetime.now():%Y-%m-%d}</p>
</body></html>"""

    out = step_root / f"{index:02d}_REPORT_{region}_{ctx['level_name']}.html"
    out.write_text(html, encoding="utf-8")
    print(f"[report] {region} -> {out}")
    return out


def build_reports(ctx, args):
    paths = []
    for i, region in enumerate(ctx["regions"], start=1):
        paths.append(build_region_report(ctx, args, region, i))
    ctx["report_paths"] = paths
    return ctx


def main(argv=None):
    args = parse_args(argv)
    ctx = load_inputs(args)
    ctx = build_pseudobulk(ctx, args)
    ctx = run_edger(ctx, args)
    ctx = combine_results(ctx)


    summary = de_count_summary(ctx["de_all"], args.alpha, args.lfc, sig_col=args.sig_col)
    summary.to_csv(ctx["results_dir"] / "de_count_summary.csv", index=False)
    ctx["de_summary"] = summary

    ctx = marker_filter(ctx, args)
    ctx = make_volcanoes(ctx, args)
    ctx = write_excel_de_table(ctx, args)
    ctx = build_reports(ctx, args)

    print("\n" + "=" * 64)
    print("DONE")
    print("=" * 64)
    print(f"  regions tested : {ctx['regions']}")
    print(f"  covariates     : {args.covariates}")
    print(f"  run label      : {args.run_label or '(none)'}")
    print(f"  all-genes DE   : {len(ctx['de_all']):,} rows")
    print(f"  filtered DE    : {len(ctx.get('de_marker', [])):,} rows")
    print(f"  outputs        : {ctx['step_root']}")
    print(f"  excel table    : {ctx.get('excel_path')}")
    for p in ctx["report_paths"]:
        print(f"  report         : {p}")
    print("=" * 64)


    print("\n" + "=" * 64)
    print(f"SIGNIFICANT DE genes ({args.metric_axis_label} < {args.alpha}, |log2FC| > {args.lfc}) "
          f"— summed across cell types")
    print("=" * 64)
    summary = ctx.get("de_summary")
    if summary is None or not len(summary):
        print("  (no DE results)")
    else:
        agg = (summary.groupby(["region", "comparison"], observed=True)[["n_sig", "n_up", "n_down"]]
               .sum().astype(int).reset_index())
        for region in ctx["regions"]:
            region_rows = agg[agg["region"] == region]
            if not len(region_rows):
                continue
            print(f"  {region}:")
            for _, row in region_rows.iterrows():
                print(f"    {row['comparison']:<28} "
                      f"{row['n_sig']:>4} sig  (up {row['n_up']}, down {row['n_down']})")
    print("=" * 64)


if __name__ == "__main__":
    main()
