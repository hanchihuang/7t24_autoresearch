#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 7 ]]; then
  cat <<'EOF'
Usage:
  scripts/new_task.sh <project_dir> <goal> <metric> <direction> <verify> <guard> <run_tag> [scope] [prompt]

Example:
  scripts/new_task.sh \
    /abs/path/to/project \
    "Improve benchmark pass rate beyond 0.72" \
    benchmark_pass_rate \
    higher \
    "python3 run_eval.py --samples 30" \
    "python3 -m py_compile main.py" \
    benchmark_run_20260330 \
    "/abs/path/to/project/src,/abs/path/to/project/tests" \
    "Continue iterating on the benchmark task with scout-first protocol."
EOF
  exit 1
fi

PROJECT_DIR="$1"
GOAL="$2"
METRIC="$3"
DIRECTION="$4"
VERIFY="$5"
GUARD="$6"
RUN_TAG="$7"
SCOPE="${8:-$PROJECT_DIR}"
PROMPT="${9:-$GOAL}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_PATH="${PROJECT_DIR}/autoresearch-launch.json"

python3 "${ROOT_DIR}/scripts/init_generic_task.py" \
  --project-dir "${PROJECT_DIR}" \
  --output "${OUTPUT_PATH}" \
  --goal "${GOAL}" \
  --metric "${METRIC}" \
  --direction "${DIRECTION}" \
  --verify "${VERIFY}" \
  --guard "${GUARD}" \
  --run-tag "${RUN_TAG}" \
  --scope "${SCOPE}" \
  --prompt "${PROMPT}"

echo
echo "Generated launch file:"
echo "  ${OUTPUT_PATH}"
echo
echo "Optional next-direction controls:"
echo "  edit ${OUTPUT_PATH} and set:"
echo "    config.max_auto_direction_replans"
echo "    config.preferred_direction_families"
echo "    config.banned_direction_families"
echo
echo "Next step:"
echo "  python3 ${ROOT_DIR}/scripts/autoresearch_runtime_ctl.py start --repo ${PROJECT_DIR}"
