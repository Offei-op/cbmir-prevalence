#!/usr/bin/env bash
# Run from repository root after reviewing the files. Requires authenticated gh.
set -euo pipefail
command -v gh >/dev/null || { echo 'Install GitHub CLI and run gh auth login first.' >&2; exit 1; }
gh auth status
if [ ! -d .git ]; then
  git init -b main
fi
git add README.md pyproject.toml .gitignore src configs notebooks docs tests scripts
if ! git diff --cached --quiet; then
  git commit -m "Organize CBMIR prevalence experiment and document methodology"
fi
# Private by default for work in progress. No raw data or trained artifacts included.
gh repo create Offei-op/cbmir-prevalence --private --source=. --remote=origin --push --description 'Chest X-ray CBMIR evaluation under decreasing relevant-image prevalence'
