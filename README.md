# MERSCOPE Spatial Transcriptomics Workflow

A config-driven, script-per-step pipeline for MERSCOPE cohorts: per-sample QC
with MerQuaCo, cohort filtering, normalization and clustering, MapMyCells label
import, polygon ROI selection, pseudobulk edgeR differential expression, and a
static HTML report.

## Setup

- Python 3.11; Python 3.10-3.12 is expected to work.
- R >= 4.3 with edgeR and limma:
  `BiocManager::install(c("edgeR", "limma"))`. `Rscript` must be on `PATH`;
  on HPC systems, activate the appropriate R environment first.
- MerQuaCo (step 00 only) is not on PyPI; install from its source repository
  into the same Python environment.
- ilastik (step 00 only); set its executable path in the config.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

Copy [cohort_config.example.yaml](cohort_config.example.yaml) outside the
repository, edit the paths, and write a sample sheet beside it. Run from the
repository root:

```bash
CONFIG=../configs/my_cohort.yaml
bash code/00_run_preprocessing.sh -c "$CONFIG" -j 4
python code/01_qc_cohort.py --config "$CONFIG"
python code/02_combine_filtered.py --config "$CONFIG"
python code/03_normalize_dimred.py --config "$CONFIG"
python code/03b_export_annotation_files.py --config "$CONFIG"
# Submit mapper_input_gzip2.h5ad to MapMyCells and save the returned JSON as configured.
python code/04a_import_annotations.py --config "$CONFIG"
python code/04b_select_rois.py --config "$CONFIG" --cell-type-level class_name
python code/05_pseudobulk_de.py --config "$CONFIG" --cell-type-level class_name
python code/06_generate_report.py --config "$CONFIG"
```

Steps 04a and 04b supply `obs["label_confidence"]` and `obs["in_roi"]`
required by step 05; run both before 05/06 for the complete workflow.
The example disables coarse mapping, so steps 04b/05 use `class_name`.
When coarse mapping is enabled, omit `--cell-type-level class_name` to use
the default `coarse` level, which requires `obs["cell_type"]`.

## Pipeline steps

| Step | Purpose | Main output |
|---|---|---|
| 00 | Build per-sample AnnData and run MerQuaCo | `00_preprocessing/<sample>/` |
| 01 | Apply masks and cell QC | `01_qc_cohort/` |
| 02 | Combine filtered samples | `02_combine_filtered/all_samples_filtered.h5ad` |
| 03 | Normalize, PCA, neighbors, UMAP, and Leiden | `03_normalize_dimred/combined_norm_dimred.h5ad` |
| 03b | Export an optional MapMyCells submission | `04a_import_annotations/mapmycells/` |
| 04a | Import MapMyCells and optional coarse labels | `04a_import_annotations/combined_annotated.h5ad` |
| 04b | Select polygon ROIs and generate region QC | `04b_select_rois/combined_annotated_with_rois.h5ad` |
| 05 | Run edgeR pseudobulk DE | `05_pseudobulk_de/` |
| 06 | Gather results into static tables, plots, and HTML | `06_generate_report/` |

## Configuration

Use one YAML per cohort, based on
[cohort_config.example.yaml](cohort_config.example.yaml).
It defines paths, comparisons, preprocessing/filtering, annotation, regions,
DE settings, and reporting. Path templates support `{project_root}`,
`{external_dir}`, and `{sample_id}`. Sample-sheet CSV paths are relative to the
YAML file; required columns are `sample_id`, `group`, and `include`:

```csv
sample_id,group,include
sample_1,control,true
sample_2,treatment,true
```

Included sample IDs must be unique. `true`, `yes`, `y`, and `1` include rows;
`false`, `no`, `n`, and `0` exclude them. Additional populated columns become
sample-level covariates and must have values for every included sample.
Configs without a sample sheet may define `samples`, `group_map`, and
`covariates` directly.

## Inputs

- Per-sample MERSCOPE cell-by-gene, cell metadata, detected transcripts, DAPI
  mosaic, and codebook files, with filenames/paths set in the config.
- MapMyCells result JSON, supplied through `label_transfer.mapper_json_path`
  or `mapper_json_filename` in the step-03b export folder.
- Optional ROI polygon JSON, one file per sample:

```json
{"sample_id": "sample_01", "vertices": [[0, 0], [100, 0], [100, 100]]}
```

Vertices must use the same coordinate units as `obsm["spatial"]`. Any tool
that writes this JSON can be used. Duplicate files for a sample are an error.
`regions.files` accepts individual JSON files or recursively searched directories.
With no configured polygons, step 04b writes an empty `in_roi` mask and
steps 05/06 produce whole-dataset outputs.

## Outputs

Results are stored under the configured `paths.output_dir`, in the step folders
listed above. The config and sample sheet are copied there for provenance.
Static plots are saved as PNG and SVG; steps 05/06 produce DE tables, an Excel
workbook (step 05), and HTML reports. Keep the complete output directory.

## Analysis assumptions

- DE tests all `*_mouse` features plus human `MAPT`
  (`keep_mouse_and_human_mapt`); feature names carry `_mouse`/`_human` suffixes.
- `coarse_mapping_method: direct` requires workbook columns `subclass` and
  `class-APOE-cluster`; `hierarchical` requires `cell_group`, `neurotransmitter`,
  `class`, and `subclass`.
- QC report-flag thresholds are module constants in
  [01_qc_cohort.py](code/01_qc_cohort.py), not config settings.

## Running on a cluster

Each step is an independent command, so wrap it in your scheduler of choice.
Run step 00 one sample per job with
`python code/00_preprocessing.py <sample_id> --config ...`.
For concurrent step-00 jobs, give each its own `TMPDIR` and `XDG_CACHE_HOME`
to avoid ilastik cache collisions.

## License
See [LICENSE](LICENSE).
