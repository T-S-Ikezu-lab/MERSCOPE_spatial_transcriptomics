#!/usr/bin/env python3
"""Apply per-sample spatial masks and cell QC, then render a cohort report."""

import argparse
import gc
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import tifffile as tiff
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape
from config_utils import cohort_paths, load_cohort

sns.set_style("whitegrid")
sc.settings.set_figure_params(dpi=100, frameon=False)


# Report-flag thresholds (not filters).
FLAG_LOW_RETENTION = 0.50
FLAG_VERY_LOW_RETENTION = 0.25
FLAG_HIGH_MERQUACO = 0.30
FLAG_MEDIAN_RATIO = 0.5
FLAG_MAD_MULTIPLIER = 2.0
DOUBLET_MULTIPLIER = 3.0


def setup_dirs(paths):
    dirs = {
        "preprocessed": paths.output("00_preprocessing"),
        "filtered_h5ad": paths.output("01_qc_cohort", "filtered"),
        "report": paths.output("01_qc_cohort"),
        "plots": paths.output("01_qc_cohort", "plots"),
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def sample_paths(sid, preprocessed_dir):
    out = Path(preprocessed_dir) / sid
    return {
        "h5ad": out / f"{sid}_raw.h5ad",
        "damage_mask": out / "00_merquaco" / "damage_mask.tiff",
        "lifting_mask": out / "00_merquaco" / "detachment_mask.tiff",
        "lifting_mask_edited": out / "lifting_mask_edited.tiff",
    }


def get_thresholds(sid, filter_params, per_sample=None):
    """Lower-bound thresholds only. Upper bounds are tracked as flags, not filters."""
    t = {
        "min_counts": filter_params.get("min_counts", 10),
        "min_genes": filter_params.get("min_genes", 0),
        "min_volume": filter_params.get("min_volume", 100),
    }
    if per_sample and sid in per_sample:
        t.update(per_sample[sid])
    return t


def cells_in_mask(adata, mask):
    """Map cells to mask pixels assuming the mask spans exactly the cell-coordinate bounding box."""
    xs = adata.obsm["spatial"][:, 0]
    ys = adata.obsm["spatial"][:, 1]
    x_scale = mask.shape[1] / (xs.max() - xs.min())
    y_scale = mask.shape[0] / (ys.max() - ys.min())
    x_idx = np.clip(np.round((xs - xs.min()) * x_scale).astype(int), 0, mask.shape[1] - 1)
    y_idx = np.clip(np.round((ys - ys.min()) * y_scale).astype(int), 0, mask.shape[0] - 1)
    return mask[y_idx, x_idx] != 0


def process_sample(sid, paths, thresholds, dirs, edited_mask_used_log):
    """Load one sample, apply masks + filters, write filtered h5ad, generate per-sample plots, return summary dict."""
    print(f"\n[{sid}]")

    if not paths["h5ad"].exists():
        print(f"  SKIP: h5ad not found at {paths['h5ad']}")
        return None

    adata = sc.read_h5ad(paths["h5ad"])
    n_loaded = adata.n_obs
    print(f"  Loaded: {n_loaded:,} cells x {adata.n_vars:,} genes")


    damage_mask = tiff.imread(str(paths["damage_mask"]))
    if paths["lifting_mask_edited"].exists():
        lifting_mask = tiff.imread(str(paths["lifting_mask_edited"]))
        edited = True
        edited_mask_used_log.append(sid)


        if paths["lifting_mask"].exists():
            raw_mtime = paths["lifting_mask"].stat().st_mtime
            edit_mtime = paths["lifting_mask_edited"].stat().st_mtime
            if raw_mtime > edit_mtime:
                raw_date = datetime.fromtimestamp(raw_mtime).strftime("%Y-%m-%d %H:%M")
                edit_date = datetime.fromtimestamp(edit_mtime).strftime("%Y-%m-%d %H:%M")
                print(f"  WARNING: raw lifting mask ({raw_date}) is newer than "
                      f"edited mask ({edit_date}). Edit may be stale — consider re-editing.")
                stale_edit_warning = f"Raw mask ({raw_date}) newer than edit ({edit_date})"
            else:
                stale_edit_warning = None
        else:
            stale_edit_warning = None
    else:
        lifting_mask = tiff.imread(str(paths["lifting_mask"]))
        edited = False
        stale_edit_warning = None

    in_damage = cells_in_mask(adata, damage_mask)
    in_lifting = cells_in_mask(adata, lifting_mask)
    merquaco_removed = in_damage | in_lifting
    n_damage = int(in_damage.sum())
    n_lifting = int(in_lifting.sum())
    n_merquaco = int(merquaco_removed.sum())

    print(f"  MerQuaCo: damage={n_damage:,}, lifting={n_lifting:,}, "
          f"union={n_merquaco:,} ({100*n_merquaco/n_loaded:.1f}%)")


    sc.pp.calculate_qc_metrics(adata, percent_top=None, inplace=True)


    counts_fail = (adata.obs["total_counts"] < thresholds["min_counts"]).to_numpy()
    genes_fail = (adata.obs["n_genes_by_counts"] < thresholds["min_genes"]).to_numpy()
    volume_fail = (adata.obs["volume"] < thresholds["min_volume"]).to_numpy()
    qc_fail_any = counts_fail | genes_fail | volume_fail
    n_qc_independent = int(qc_fail_any.sum())


    kept = (~merquaco_removed) & (~qc_fail_any)
    n_kept = int(kept.sum())
    pct_retained = 100 * n_kept / n_loaded

    print(f"  QC fail (independent of MerQuaCo): "
          f"counts<{thresholds['min_counts']}={int(counts_fail.sum()):,}, "
          f"genes<{thresholds['min_genes']}={int(genes_fail.sum()):,}, "
          f"vol<{thresholds['min_volume']}={int(volume_fail.sum()):,}, "
          f"any={n_qc_independent:,}")
    print(f"  Final: {n_kept:,} cells retained ({pct_retained:.1f}%)")


    adata_filtered = adata[kept].copy()
    out_h5ad = dirs["filtered_h5ad"] / f"{sid}_filtered.h5ad"
    adata_filtered.write_h5ad(out_h5ad)
    print(f"  Wrote: {out_h5ad}")


    plot_spatial_4panel(adata, merquaco_removed, qc_fail_any, kept, sid, dirs["plots"])
    plot_qc_distributions(adata, kept, thresholds, sid, dirs["plots"])

    plot_mask_overlay(adata, in_damage, in_lifting, sid, dirs["plots"])
    plot_cells_by_count(adata, sid, dirs["plots"])


    kept_obs = adata.obs.loc[kept]
    summary = {
        "sample_id": sid,
        "n_loaded": int(n_loaded),
        "n_damage": n_damage,
        "n_lifting": n_lifting,
        "n_merquaco_union": n_merquaco,
        "n_qc_counts": int(counts_fail.sum()),
        "n_qc_genes": int(genes_fail.sum()),
        "n_qc_volume": int(volume_fail.sum()),
        "n_qc_removed": n_qc_independent,
        "n_kept": n_kept,
        "pct_retained": round(pct_retained, 2),
        "pct_merquaco_removed": round(100 * n_merquaco / n_loaded, 2),
        "median_total_counts": float(kept_obs["total_counts"].median()),
        "median_n_genes": float(kept_obs["n_genes_by_counts"].median()),
        "median_volume": float(kept_obs["volume"].median()),
        "p99_total_counts": float(kept_obs["total_counts"].quantile(0.99)),
        "thresholds": thresholds,
        "edited_mask_used": edited,
        "stale_edit_warning": stale_edit_warning,
        "plot_spatial_4panel": f"plots/{sid}_spatial_4panel.png",
        "plot_qc_distributions": f"plots/{sid}_qc_distributions.png",
    }


    summary["_kept_total_counts"] = kept_obs["total_counts"].to_numpy()


    del adata, adata_filtered, in_damage, in_lifting, merquaco_removed
    del counts_fail, genes_fail, volume_fail, qc_fail_any, kept
    gc.collect()

    return summary


def _save(fig, path, dpi=120):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    fig.savefig(Path(path).with_suffix(".svg"), format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_spatial_4panel(adata, merquaco_removed, qc_removed, kept, sid, plots_dir):
    """Single color per panel. Shared axes. Gray context dots in removed panels."""
    xs = adata.obsm["spatial"][:, 0]
    ys = adata.obsm["spatial"][:, 1]

    fig, axes = plt.subplots(1, 4, figsize=(22, 6), sharex=True, sharey=True)
    fig.suptitle(f"Spatial QC — {sid}", fontsize=13)


    sc1 = axes[0].scatter(xs, ys, c=adata.obs["total_counts"], cmap="viridis",
                          s=0.5, alpha=0.6, rasterized=True)
    axes[0].set_title(f"All loaded cells\n(n={adata.n_obs:,})")
    plt.colorbar(sc1, ax=axes[0], fraction=0.04, pad=0.02, label="Total counts")


    axes[1].scatter(xs, ys, c="lightgray", s=0.4, alpha=0.4, rasterized=True)
    axes[1].scatter(xs[merquaco_removed], ys[merquaco_removed],
                    c="#c0392b", s=0.6, alpha=0.7, rasterized=True)
    axes[1].set_title(f"Removed by MerQuaCo\n(n={int(merquaco_removed.sum()):,})")


    axes[2].scatter(xs, ys, c="lightgray", s=0.4, alpha=0.4, rasterized=True)
    axes[2].scatter(xs[qc_removed], ys[qc_removed],
                    c="#e67e22", s=0.6, alpha=0.7, rasterized=True)
    axes[2].set_title(f"Flagged by QC thresholds\n(n={int(qc_removed.sum()):,}, may overlap MerQuaCo)")


    sc4 = axes[3].scatter(xs[kept], ys[kept],
                          c=adata.obs.loc[kept, "total_counts"], cmap="viridis",
                          s=0.5, alpha=0.6, rasterized=True)
    axes[3].set_title(f"Final retained\n(n={int(kept.sum()):,})")
    plt.colorbar(sc4, ax=axes[3], fraction=0.04, pad=0.02, label="Total counts")

    for ax in axes:
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    plt.tight_layout()
    _save(fig, plots_dir / f"{sid}_spatial_4panel.png")


def plot_qc_distributions(adata, kept, thresholds, sid, plots_dir):
    """Compact: 3 histograms (top) + 3 scatters (bottom)."""
    obs_all = adata.obs
    obs_kept = adata.obs.loc[kept]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"QC distributions — {sid} "
                 f"({adata.n_obs:,} → {int(kept.sum()):,}, "
                 f"{100*int(kept.sum())/adata.n_obs:.1f}% retained)",
                 fontsize=12)

    hist_panels = [
        ("total_counts", thresholds["min_counts"], "Total counts"),
        ("n_genes_by_counts", thresholds["min_genes"], "Genes detected"),
        ("volume", thresholds["min_volume"], "Volume (µm³)"),
    ]
    for ax, (col, cutoff, xlabel) in zip(axes[0], hist_panels):
        ax.hist(obs_all[col], bins=80, alpha=0.5, color="steelblue", label=f"All (n={adata.n_obs:,})")
        ax.hist(obs_kept[col], bins=80, alpha=0.7, color="forestgreen", label=f"Kept (n={int(kept.sum()):,})")
        if cutoff > 0:
            ax.axvline(cutoff, color="red", ls="--", lw=1, label=f"≥ {cutoff}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Cells")
        ax.legend(fontsize=8)

    scatter_panels = [
        ("total_counts", "n_genes_by_counts", "Total counts", "Genes"),
        ("volume", "total_counts", "Volume", "Total counts"),
        ("volume", "n_genes_by_counts", "Volume", "Genes"),
    ]
    for ax, (xc, yc, xl, yl) in zip(axes[1], scatter_panels):
        ax.scatter(obs_all[xc], obs_all[yc], s=1, alpha=0.1, color="steelblue", rasterized=True)
        ax.scatter(obs_kept[xc], obs_kept[yc], s=1, alpha=0.15, color="forestgreen", rasterized=True)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)

    plt.tight_layout()
    _save(fig, plots_dir / f"{sid}_qc_distributions.png")


def plot_mask_overlay(adata, in_damage, in_lifting, sid, plots_dir):
    """Diagnostic 1x2: cells colored by damage / lifting flag separately. Written to disk only; not included in the report."""
    xs = adata.obsm["spatial"][:, 0]
    ys = adata.obsm["spatial"][:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    fig.suptitle(f"MerQuaCo masks — {sid}", fontsize=12)

    for ax, mask_bool, label, color in [
        (axes[0], in_damage, "Damage", "#c0392b"),
        (axes[1], in_lifting, "Lifting", "#2980b9"),
    ]:
        ax.scatter(xs[~mask_bool], ys[~mask_bool], c="lightgray", s=0.4, alpha=0.4, rasterized=True)
        ax.scatter(xs[mask_bool], ys[mask_bool], c=color, s=0.5, alpha=0.7, rasterized=True)
        n = int(mask_bool.sum())
        ax.set_title(f"{label}: {n:,} cells ({100*n/adata.n_obs:.1f}%)")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    plt.tight_layout()
    _save(fig, plots_dir / f"{sid}_mask_overlay.png")


def plot_cells_by_count(adata, sid, plots_dir):
    """Diagnostic 2x3: spatial scatter of cells with exactly 0, 1, 2, 3, 4, 5 transcripts. Useful for identifying spatial structure in the low-count tail (e.g., regions where segmentation is producing many empty/near-empty cells). Computed on pre-filter cells. Written to disk only."""
    xs = adata.obsm["spatial"][:, 0]
    ys = adata.obsm["spatial"][:, 1]
    counts = adata.obs["total_counts"].to_numpy()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
    fig.suptitle(f"Cells by exact transcript count — {sid}", fontsize=12)
    axes = axes.flatten()

    for ax, val in zip(axes, [0, 1, 2, 3, 4, 5]):
        mask = counts == val
        n = int(mask.sum())

        ax.scatter(xs, ys, c="lightgray", s=0.3, alpha=0.3, rasterized=True)
        if n > 0:
            ax.scatter(xs[mask], ys[mask], c="#8e44ad", s=2.0, alpha=0.7, rasterized=True)
        ax.set_title(f"total_counts == {val}  (n={n:,})")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    plt.tight_layout()
    _save(fig, plots_dir / f"{sid}_cells_by_count.png")


def mad(arr):
    arr = np.asarray(arr)
    return float(np.median(np.abs(arr - np.median(arr))))


def compute_cohort_flags(summaries):
    """Add 'flags' (list of strings) and 'status' (ok/warn/fail) to each summary."""
    if not summaries:
        return

    medians_counts = np.array([s["median_total_counts"] for s in summaries])
    medians_volume = np.array([s["median_volume"] for s in summaries])
    p99_counts = np.array([s["p99_total_counts"] for s in summaries])

    cohort_median_counts = float(np.median(medians_counts))
    cohort_median_volume = float(np.median(medians_volume))
    cohort_p99_counts = float(np.median(p99_counts))

    mad_counts = mad(medians_counts) or 1.0
    mad_volume = mad(medians_volume) or 1.0

    for s in summaries:
        flags = []
        status = "ok"

        retained = s["pct_retained"] / 100
        if retained < FLAG_VERY_LOW_RETENTION:
            flags.append(f"Very low retention ({s['pct_retained']:.0f}%)")
            status = "fail"
        elif retained < FLAG_LOW_RETENTION:
            flags.append(f"Low retention ({s['pct_retained']:.0f}%)")
            status = "warn" if status == "ok" else status

        if s["pct_merquaco_removed"] / 100 > FLAG_HIGH_MERQUACO:
            flags.append(f"High MerQuaCo removal ({s['pct_merquaco_removed']:.0f}%)")
            status = "warn" if status == "ok" else status

        if s["median_total_counts"] < FLAG_MEDIAN_RATIO * cohort_median_counts:
            flags.append(f"Low median counts ({s['median_total_counts']:.0f} vs cohort {cohort_median_counts:.0f})")
            status = "warn" if status == "ok" else status

        if abs(s["median_total_counts"] - cohort_median_counts) > FLAG_MAD_MULTIPLIER * mad_counts:
            flags.append("Counts median is cohort outlier (>2 MAD)")
            status = "warn" if status == "ok" else status

        if abs(s["median_volume"] - cohort_median_volume) > FLAG_MAD_MULTIPLIER * mad_volume:
            flags.append("Volume median is cohort outlier (>2 MAD)")
            status = "warn" if status == "ok" else status


        doublet_threshold = DOUBLET_MULTIPLIER * cohort_p99_counts
        n_doublets = int((s["_kept_total_counts"] > doublet_threshold).sum())
        s["n_potential_doublets"] = n_doublets
        s["doublet_threshold"] = round(doublet_threshold, 0)
        if n_doublets > 0.01 * s["n_kept"]:
            flags.append(f"{n_doublets:,} potential doublets (>{doublet_threshold:.0f} counts)")
            status = "warn" if status == "ok" else status


        if s.get("stale_edit_warning"):
            flags.append(f"Stale edited mask: {s['stale_edit_warning']}")
            status = "warn" if status == "ok" else status

        s["flags"] = flags
        s["status"] = status

    return {
        "cohort_median_counts": cohort_median_counts,
        "cohort_median_volume": cohort_median_volume,
        "cohort_p99_counts": cohort_p99_counts,
    }


def plot_cohort_stacked_bar(summaries, plots_dir):
    """Export the cohort disposition plot as PNG and SVG."""
    sids = [s["sample_id"] for s in summaries]
    damage = np.array([s["n_damage"] for s in summaries])

    lifting_only = np.array([s["n_merquaco_union"] - s["n_damage"] for s in summaries])
    qc_counts = np.array([s["n_qc_counts"] for s in summaries])
    qc_volume = np.array([s["n_qc_volume"] for s in summaries])

    qc_other = np.array([
        max(0, s["n_qc_removed"] - s["n_qc_counts"] - s["n_qc_volume"])
        for s in summaries
    ])
    kept = np.array([s["n_kept"] for s in summaries])

    fig, ax = plt.subplots(figsize=(max(8, len(sids) * 0.7), 6))
    bottom = np.zeros(len(sids))
    layers = [
        (damage, "#c0392b", "MerQuaCo: damage"),
        (lifting_only, "#2980b9", "MerQuaCo: lifting"),
        (qc_counts, "#e67e22", "QC: low counts"),
        (qc_volume, "#f1c40f", "QC: low volume"),
        (qc_other, "#95a5a6", "QC: other"),
        (kept, "#27ae60", "Kept"),
    ]
    for arr, color, label in layers:
        ax.bar(sids, arr, bottom=bottom, color=color, label=label, edgecolor="white", linewidth=0.5)
        bottom += arr

    ax.set_ylabel("Cells")
    ax.set_title("Cohort: cell disposition per sample")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=9)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    png_out = plots_dir / "cohort_stacked_bar.png"
    svg_out = plots_dir / "cohort_stacked_bar.svg"
    fig.savefig(png_out, dpi=120, bbox_inches="tight")
    fig.savefig(svg_out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"SVG:    {svg_out}")
    return "plots/cohort_stacked_bar.png"


def plot_cohort_violins(summaries, plots_dir):
    """REMOVED per user request — kept as stub in case it's wanted back later."""
    return None


def render_report(summaries, cohort_stats, cohort_plots, config, args, dirs,
                  edited_mask_used_log, template_dir):
    """Render the single cohort HTML report."""
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("cohort_qc_report.html.j2")


    summaries_clean = []
    for s in summaries:
        s2 = {k: v for k, v in s.items() if not k.startswith("_")}
        summaries_clean.append(s2)

    html = template.render(
        cohort_name=config.get("cohort_name", "Cohort"),
        generated_at=datetime.now().strftime("%B %d, %Y"),
        summaries=summaries_clean,
        cohort_stats=cohort_stats,
        cohort_plots=cohort_plots,
        config_yaml=yaml.safe_dump(config, sort_keys=False, default_flow_style=False),
        edited_mask_samples=edited_mask_used_log,
        flag_settings={
            "low_retention_pct": int(FLAG_LOW_RETENTION * 100),
            "very_low_retention_pct": int(FLAG_VERY_LOW_RETENTION * 100),
            "high_merquaco_pct": int(FLAG_HIGH_MERQUACO * 100),
            "mad_multiplier": FLAG_MAD_MULTIPLIER,
            "doublet_multiplier": DOUBLET_MULTIPLIER,
        },
    )

    out = dirs["report"] / "cohort_qc_report.html"
    out.write_text(html)
    print(f"\nReport: {out}")


    df = pd.DataFrame([
        {k: v for k, v in s.items()
         if not k.startswith("_") and not isinstance(v, (list, dict))}
        for s in summaries
    ])
    csv_out = dirs["report"] / "cohort_summary.csv"
    df.to_csv(csv_out, index=False)
    print(f"CSV:    {csv_out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to cohort.yaml")
    parser.add_argument("--samples", default=None,
                        help="Comma-separated subset of samples to process "
                             "(default: all samples in cohort.yaml)")
    parser.add_argument("--template-dir", default=None,
                        help="Directory containing cohort_qc_report.html.j2 "
                             "(default: ./templates next to this script)")
    args = parser.parse_args()

    config = load_cohort(args.config)
    paths = cohort_paths(config)
    all_samples = config["samples"]
    filter_params = config.get("filter_params", {})
    per_sample_overrides = config.get("per_sample_thresholds", {})

    samples = (
        [s.strip() for s in args.samples.split(",")]
        if args.samples else all_samples
    )
    invalid = [s for s in samples if s not in all_samples]
    if invalid:
        print(f"WARNING: not in cohort.yaml: {invalid}")
    samples = [s for s in samples if s in all_samples]

    template_dir = (
        Path(args.template_dir) if args.template_dir
        else Path(__file__).parent / "templates"
    )
    if not (template_dir / "cohort_qc_report.html.j2").exists():
        print(f"ERROR: template not found at {template_dir}/cohort_qc_report.html.j2")
        sys.exit(1)

    dirs = setup_dirs(paths)

    print("=" * 70)
    print(f"Cohort QC: {config.get('cohort_name', 'cohort')}")
    print(f"Samples: {len(samples)}")
    print(f"Output:  {dirs['report']}")
    print("=" * 70)

    summaries = []
    edited_mask_used_log = []
    for sid in samples:
        paths = sample_paths(sid, dirs["preprocessed"])
        thresholds = get_thresholds(sid, filter_params, per_sample_overrides)
        result = process_sample(sid, paths, thresholds, dirs, edited_mask_used_log)
        if result is not None:
            summaries.append(result)

    if not summaries:
        print("ERROR: no samples processed.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("Computing cohort flags + plots")
    print("=" * 70)
    cohort_stats = compute_cohort_flags(summaries)
    cohort_plots = {
        "stacked_bar": plot_cohort_stacked_bar(summaries, dirs["plots"]),
    }

    render_report(summaries, cohort_stats, cohort_plots, config, args, dirs,
                  edited_mask_used_log, template_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
