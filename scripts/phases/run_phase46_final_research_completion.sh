#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
ARGS=("src/phase46_final_research_completion.py")

if [[ "${1:-}" == "--dry-run" ]]; then
  ARGS+=("--dry-run")
fi

"${PYTHON}" "${ARGS[@]}"
