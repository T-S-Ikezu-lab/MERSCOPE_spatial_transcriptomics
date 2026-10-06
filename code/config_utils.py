"""Load cohort YAML and optional sample sheets, and resolve shared pipeline paths."""

from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path

import yaml


_DEFAULT_PATHS = {
    "sample_data_dir": "{project_root}/data/resegmentation/{sample_id}",
    "dapi_dir": "{project_root}/data/raw/{sample_id}",
    "output_dir": "{project_root}/outputs",
    "external_dir": "{project_root}/data/external",
}


class CohortPaths:
    """Resolved filesystem locations for a cohort."""

    def __init__(self, cohort: dict):
        paths_cfg = dict(_DEFAULT_PATHS)
        paths_cfg.update(cohort.get("paths", {}) or {})

        project_root = (cohort.get("paths", {}) or {}).get("project_root") or cohort.get(
            "project_root"
        )
        if not project_root:
            raise KeyError(
                "cohort config must define 'paths.project_root' "
                "(or a top-level 'project_root')."
            )
        self.project_root = Path(project_root)


        # Resolve external_dir first because other path templates can reference it.
        self.external_dir = Path(
            paths_cfg["external_dir"].format(project_root=self.project_root)
        )
        self.output_dir = Path(
            paths_cfg["output_dir"].format(
                project_root=self.project_root, external_dir=self.external_dir
            )
        )
        self._sample_data_tmpl = paths_cfg["sample_data_dir"]
        self._dapi_tmpl = paths_cfg["dapi_dir"]

    def _fill(self, tmpl: str, sample_id: str | None = None) -> Path:
        return Path(
            tmpl.format(
                project_root=self.project_root,
                external_dir=self.external_dir,
                sample_id="" if sample_id is None else sample_id,
            )
        )

    def sample_data_dir(self, sample_id: str) -> Path:
        """Directory holding a sample's cell_by_gene / metadata / transcripts CSVs."""
        return self._fill(self._sample_data_tmpl, sample_id)

    def dapi_dir(self, sample_id: str) -> Path:
        """Directory holding a sample's DAPI mosaic image."""
        return self._fill(self._dapi_tmpl, sample_id)

    def output(self, *parts) -> Path:
        """Join path components under the cohort's output directory."""
        return self.output_dir.joinpath(*parts)

    def resolve(self, template: str) -> Path:
        """Resolve an arbitrary config path template ({project_root}/{external_dir})."""
        return self._fill(template, None)


def _load_sample_sheet(cohort: dict, config_path: Path) -> None:
    configured = cohort.get("sample_sheet")
    if not configured:
        return

    sheet_path = Path(configured)
    if not sheet_path.is_absolute():
        sheet_path = config_path.parent / sheet_path
    if not sheet_path.exists():
        raise FileNotFoundError(f"Sample sheet does not exist: {sheet_path}")

    with sheet_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "group", "include"}
    missing = required - set(rows[0] if rows else [])
    if missing:
        raise ValueError(
            f"{sheet_path} is missing required columns: {sorted(missing)}"
        )

    true_values = {"1", "true", "yes", "y"}
    false_values = {"0", "false", "no", "n"}
    for row_number, row in enumerate(rows, start=2):
        value = str(row["include"]).strip().lower()
        if value not in true_values | false_values:
            raise ValueError(
                f"{sheet_path}:{row_number} has invalid include value "
                f"{row['include']!r}"
            )
    included = [
        row for row in rows if str(row["include"]).strip().lower() in true_values
    ]
    samples = [str(row["sample_id"]).strip() for row in included]
    if not samples:
        raise ValueError(f"{sheet_path} contains no included samples")
    if any(not sample for sample in samples):
        raise ValueError(f"{sheet_path} has an included row with no sample_id")
    if len(samples) != len(set(samples)):
        raise ValueError(f"{sheet_path} contains duplicate included sample IDs")
    if any(not str(row["group"]).strip() for row in included):
        raise ValueError(f"{sheet_path} has an included sample with no group")

    cohort["samples"] = samples
    cohort["group_map"] = {
        str(row["sample_id"]).strip(): str(row["group"]).strip()
        for row in included
    }
    covariates = {}
    for column in rows[0]:
        if column in required:
            continue
        values = [str(row.get(column, "")).strip() for row in included]
        if not any(values):
            continue
        if not all(values):
            raise ValueError(
                f"{sheet_path} has incomplete values in covariate column '{column}'"
            )
        covariates[column] = dict(zip(samples, values))
    cohort["covariates"] = covariates


def load_cohort(path) -> dict:
    """Load a cohort YAML and expand its optional CSV sample sheet."""
    config_path = Path(path).resolve()
    with config_path.open() as handle:
        cohort = yaml.safe_load(handle)
    if not isinstance(cohort, dict):
        raise ValueError(f"Cohort config must contain a YAML mapping: {config_path}")
    _load_sample_sheet(cohort, config_path)
    paths = CohortPaths(cohort)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    _copy_provenance_file(config_path, paths.output("analysis_config.yaml"))
    configured_sheet = cohort.get("sample_sheet")
    if configured_sheet:
        sheet_path = Path(configured_sheet)
        if not sheet_path.is_absolute():
            sheet_path = config_path.parent / sheet_path
        _copy_provenance_file(sheet_path, paths.output(sheet_path.name))
    return cohort


def _copy_provenance_file(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        return
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def cohort_paths(cohort: dict) -> CohortPaths:
    """Build a :class:`CohortPaths` from a loaded cohort dict."""
    return CohortPaths(cohort)
