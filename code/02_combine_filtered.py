#!/usr/bin/env python3
"""Combine filtered samples into a raw-count AnnData object for normalization."""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import scipy.sparse
import scanpy as sc
from anndata.experimental import concat_on_disk

from config_utils import cohort_paths, load_cohort


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to cohort.yaml")
    parser.add_argument("--samples", default=None,
                        help="Comma-separated subset of samples to combine "
                             "(default: all samples in cohort.yaml)")
    parser.add_argument("--output", default=None,
                        help="Output h5ad path (default: "
                             "{project_root}/outputs/02_combine_filtered/all_samples_filtered.h5ad)")
    args = parser.parse_args()

    config = load_cohort(args.config)
    paths = cohort_paths(config)
    all_samples = config["samples"]

    samples = (
        [s.strip() for s in args.samples.split(",")]
        if args.samples else all_samples
    )
    invalid = [s for s in samples if s not in all_samples]
    if invalid:
        print(f"WARNING: not in cohort.yaml: {invalid}")
    samples = [s for s in samples if s in all_samples]

    filtered_dir = paths.output("01_qc_cohort", "filtered")
    if args.output:
        output_path = Path(args.output)
    else:
        checkpoint_dir = paths.output("02_combine_filtered")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        output_path = checkpoint_dir / "all_samples_filtered.h5ad"

    print("=" * 70)
    print(f"Combining {len(samples)} samples")
    print(f"Source:  {filtered_dir}")
    print(f"Output:  {output_path}")
    print("=" * 70)

    adatas = []
    for sid in samples:
        h5ad_path = filtered_dir / f"{sid}_filtered.h5ad"
        if not h5ad_path.exists():
            print(f"  SKIP {sid}: not found at {h5ad_path}")
            continue

        ad = sc.read_h5ad(h5ad_path)


        # Sample identity must come from the filename, not potentially stale obs.
        ad.obs["sample_id"] = sid

        ad.obs_names = [f"{sid}_{x}" for x in ad.obs_names]
        print(f"  {sid}: {ad.n_obs:,} cells x {ad.n_vars:,} genes")
        adatas.append((sid, ad))

    if not adatas:
        print("ERROR: no samples loaded.")
        sys.exit(1)


    in_files = {}
    for sid, ad in adatas:


        tmp_path = output_path.parent / f"_tmp_{sid}.h5ad"
        ad.write_h5ad(tmp_path)
        in_files[sid] = str(tmp_path)

    del adatas
    gc.collect()

    print(f"\nConcatenating ({len(in_files)} samples) directly to disk...")
    if output_path.exists():
        output_path.unlink()
    concat_on_disk(
        in_files,
        out_file=str(output_path),
        join="inner",
        merge="same",
        label="sample_id",
    )


    for tmp in in_files.values():
        Path(tmp).unlink(missing_ok=True)


    print("Converting .X to float32 CSR sparse...")
    combined = sc.read_h5ad(output_path)
    needs_convert = (
        not scipy.sparse.issparse(combined.X)
        or combined.X.dtype != np.float32
    )
    if needs_convert:
        combined.X = scipy.sparse.csr_matrix(combined.X, dtype=np.float32)
        combined.write_h5ad(output_path)
        print("  Converted and re-saved as float32 sparse.")
    else:
        print("  Already sparse, no conversion needed.")


    print(f"Combined: {combined.n_obs:,} cells x {combined.n_vars:,} genes")


    assert "sample_id" in combined.obs.columns, "sample_id missing from obs"
    assert "spatial" in combined.obsm, "spatial coords missing from obsm"
    n_nan = int(np.isnan(combined.obsm["spatial"]).any(axis=1).sum())
    if n_nan > 0:
        print(f"WARNING: {n_nan} cells have NaN spatial coordinates")

    print("\nPer-sample breakdown:")
    print(combined.obs["sample_id"].value_counts().to_string())

    print(f"\nWrote: {output_path}")
    print(f"  {combined.n_obs:,} cells x {combined.n_vars:,} genes")
    print("  .X contains raw counts. Normalization is deferred to step 03.")

    del combined
    gc.collect()

    print("\nDone.")


if __name__ == "__main__":
    main()
