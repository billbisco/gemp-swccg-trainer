#!/usr/bin/env bash
# Unattended LINEAR vs LINEAR overnight loop.
# Self-play uses one weights file on both seats. The promotion gate plays the
# candidate against the previous kept AckbarBot linear pack (both seats).
# No Beginner opponent. Does not set a shuffle seed. Does not write champs/_promoted.
# --init is used only when current.linear.json is missing or unusable.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/linear-overnight
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
exec python3 -m trainer.improve.linear_overnight \
  --rounds 40 \
  --hours 8 \
  --games 8 \
  --max-batches 2 \
  --gate-per-seat 4 \
  --max-decisions 8000 \
  --max-millis 180000 \
  --lr 0.1 \
  --init zeros \
  --out-dir runs/linear-overnight \
  "$@"
