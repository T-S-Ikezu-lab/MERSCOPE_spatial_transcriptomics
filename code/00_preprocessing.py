"""Load MERSCOPE data, build AnnData, and run MerQuaCo QC."""

import argparse
import scanpy as sc
import pandas as pd
import numpy as np

from config_utils import cohort_paths, load_cohort


parser = argparse.ArgumentParser(description="Preprocess a single MERSCOPE sample.")
parser.add_argument("sample_id", help="Sample ID (must be listed in cohort.yaml)")
parser.add_argument("--config", required=True, help="Path to cohort config")
args = parser.parse_args()

SAMPLE_ID = args.sample_id

try:
    import merquaco
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "Step 00 requires MerQuaCo. Install it from its source repository "
        "into the same Python environment before running preprocessing."
    ) from error


print(f"Loading config: {args.config}")
cohort = load_cohort(args.config)
paths = cohort_paths(cohort)
prep = cohort["preprocessing"]


if SAMPLE_ID not in cohort["samples"]:
    print(f"WARNING: {SAMPLE_ID} is not listed in cohort.yaml samples. Proceeding anyway.")


DATA_DIR   = paths.sample_data_dir(SAMPLE_ID)
DAPI_DIR   = paths.dapi_dir(SAMPLE_ID)
OUTPUT_DIR = paths.output("00_preprocessing", SAMPLE_ID)


CELL_BY_GENE         = DATA_DIR / prep.get("cell_by_gene_filename", "cell_by_gene.csv")
CELL_METADATA        = DATA_DIR / prep.get("cell_metadata_filename", "cell_metadata_reseg.csv")
DETECTED_TRANSCRIPTS = DATA_DIR / prep.get("detected_transcripts_filename", "detected_transcripts_reseg.csv")
DAPI_IMG             = DAPI_DIR / prep["dapi_filename"]


if prep.get("codebook_path"):
    CODE_BOOK = paths.resolve(prep["codebook_path"])
else:
    CODE_BOOK = paths.external_dir / prep["codebook_filename"]

BLANK_GENE_PREFIX       = prep["blank_gene_prefix"]
ILASTIK_PATH            = prep["ilastik_path"]
VENTRICLE_GENES         = prep["ventricle_genes"]
RUN_PIXEL_CLASSIFICATION = prep.get("run_pixel_classification", True)
RUN_DROPOUT             = prep.get("run_dropout", True)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Sample ID:  {SAMPLE_ID}")
print(f"Data dir:   {DATA_DIR}")
print(f"Output dir: {OUTPUT_DIR}")


print("\n--- Loading raw data ---")

cell_by_gene = pd.read_csv(CELL_BY_GENE, index_col=0)
print(f"  cell_by_gene shape: {cell_by_gene.shape}")

cell_metadata = pd.read_csv(CELL_METADATA, index_col=0)
print(f"  cell_metadata shape: {cell_metadata.shape}")

detected_transcripts = pd.read_csv(DETECTED_TRANSCRIPTS)
print(f"  detected_transcripts shape: {detected_transcripts.shape}")


all_columns = cell_by_gene.columns.tolist()
blank_cols = [c for c in all_columns if c.startswith(BLANK_GENE_PREFIX)]
gene_cols = [c for c in all_columns if not c.startswith(BLANK_GENE_PREFIX)]

print(f"  Total columns: {len(all_columns)}")
print(f"  Real genes:    {len(gene_cols)}")
print(f"  Blank barcodes: {len(blank_cols)}")


print("\n--- Building AnnData ---")

adata = sc.AnnData(
    X=cell_by_gene[gene_cols].values,
    obs=cell_metadata.loc[cell_by_gene.index].copy(),
    var=pd.DataFrame(index=gene_cols),
)


adata.obsm["blank_counts"] = cell_by_gene[blank_cols].values
adata.uns["blank_gene_names"] = blank_cols


adata.obsm["spatial"] = adata.obs[["center_x", "center_y"]].values.astype(np.float64)


adata.obs_names = [str(x) for x in adata.obs_names]
adata.var_names_make_unique()


adata.obs["sample_id"] = SAMPLE_ID


n_cells, n_genes = adata.shape
spatial = adata.obsm["spatial"]
print(f"  Cells:  {n_cells:,}")
print(f"  Genes:  {n_genes:,}")
print(f"  Blanks: {len(blank_cols)}")
print(f"  X range: [{spatial[:, 0].min():.1f}, {spatial[:, 0].max():.1f}]")
print(f"  Y range: [{spatial[:, 1].min():.1f}, {spatial[:, 1].max():.1f}]")


adata_path = OUTPUT_DIR / f"{SAMPLE_ID}_raw.h5ad"
print(f"\n--- Saving AnnData to {adata_path} ---")
adata.write_h5ad(adata_path)
print("  Done.")


print("\n--- Running MerQuaCo QC ---")

MERQUACO_OUTPUT = OUTPUT_DIR / "00_merquaco"
exp = merquaco.experiment.Experiment(
    transcripts_input=detected_transcripts,
    codebook_input=str(CODE_BOOK),
    dapi_high_res_image_path=str(DAPI_IMG),
    ilastik_program_path=ILASTIK_PATH,
    output_dir=str(MERQUACO_OUTPUT),
    ventricle_genes_list=VENTRICLE_GENES,
)

exp.run_all_qc(
    run_pixel_classification=RUN_PIXEL_CLASSIFICATION,
    run_dropout=RUN_DROPOUT,
    plot_figures=True,
    save_metrics=True,
)

print("\n--- All done ---")
print(f"  AnnData saved to:    {adata_path}")
print(f"  MerQuaCo outputs in: {MERQUACO_OUTPUT}")
