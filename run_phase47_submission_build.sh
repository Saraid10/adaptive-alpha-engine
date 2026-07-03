#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
ARGS=("src/phase47_submission_build.py")

if [[ "${1:-}" == "--dry-run" ]]; then
  ARGS+=("--dry-run")
fi

"${PYTHON}" "${ARGS[@]}"
