#!/usr/bin/env bash
# =============================================================================
# Hy-Embodied-0.5-VLA — minimal single-GPU proof-of-concept entrypoint.
#
# Loads the released UMI checkpoint and reconstructs real UMI action chunks via
# flow matching, scoring predicted vs ground-truth actions against trivial
# baselines. See poc/run_poc.py for details. Writes EVAL.md + results.json +
# per_sample.jsonl under .openresearch/artifacts/.
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

# --- 1. uv ---
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# --- 2. env ---
# --no-install-project: the project's pyproject points `license` at a missing
# `License.txt`, so building the hy-vla wheel fails. We don't need it installed;
# `hy_vla` is a plain package dir at the repo root, imported via PYTHONPATH.
uv sync --no-install-project

# --- 3. run ---
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
exec .venv/bin/python poc/run_poc.py
