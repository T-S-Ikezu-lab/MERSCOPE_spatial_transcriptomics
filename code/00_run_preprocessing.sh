#!/usr/bin/env bash
# Run 00_preprocessing.py on all samples in parallel.
#
# Usage:
#   bash code/00_run_preprocessing.sh -c ../configs/cohort.yaml
#   bash code/00_run_preprocessing.sh -c ../configs/cohort.yaml -j 2
set -euo pipefail

MAX_JOBS=14
CONFIG=""

while getopts "j:c:" opt; do
  case $opt in
    j) MAX_JOBS=$OPTARG ;;
    c) CONFIG=$OPTARG ;;
    *) echo "Usage: $0 -c CONFIG [-j NUM_PARALLEL_JOBS]" >&2; exit 1 ;;
  esac
done

if [[ -z "$CONFIG" ]]; then
  echo "ERROR: a cohort config is required (pass one with -c)" >&2
  exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config not found: $CONFIG (pass one with -c)" >&2
  exit 1
fi

LOG_DIR="logs/preprocessing"
mkdir -p "$LOG_DIR"

mapfile -t SAMPLES < <(
  python -c '
import sys
sys.path.insert(0, "code")
from config_utils import load_cohort
cfg = load_cohort(sys.argv[1])
for s in cfg.get("samples", []):
    print(s)
' "$CONFIG"
)

if (( ${#SAMPLES[@]} == 0 )); then
  echo "ERROR: no samples found under 'samples:' in $CONFIG" >&2
  exit 1
fi

echo "=== Running preprocessing: ${#SAMPLES[@]} samples, max $MAX_JOBS parallel ==="

# Failures are collected and reported at the end.
FAIL_FILE="$(mktemp)"
trap 'rm -f "$FAIL_FILE"' EXIT

printf '%s\n' "${SAMPLES[@]}" | xargs -P "$MAX_JOBS" -I {} bash -c '
  sample="$1"; config="$2"; logdir="$3"; failfile="$4"
  echo "[$(date "+%H:%M:%S")] Started  $sample"
  if python code/00_preprocessing.py "$sample" --config "$config" \
       &> "$logdir/${sample}.log"; then
    echo "[$(date "+%H:%M:%S")] OK       $sample"
  else
    code=$?
    echo "[$(date "+%H:%M:%S")] FAILED   $sample (exit $code, see $logdir/${sample}.log)" >&2
    echo "$sample" >> "$failfile"
  fi
' _ {} "$CONFIG" "$LOG_DIR" "$FAIL_FILE"

echo "[$(date '+%H:%M:%S')] All samples processed. Logs in $LOG_DIR/"

if [[ -s "$FAIL_FILE" ]]; then
  echo "=== $(wc -l < "$FAIL_FILE") sample(s) FAILED ===" >&2
  sort "$FAIL_FILE" >&2
  exit 1
fi

echo "[$(date '+%H:%M:%S')] All ${#SAMPLES[@]} samples completed successfully."
