#!/usr/bin/env bash
# Unattended LINEAR vs LINEAR loop. Runs until the process is killed.
# Self-play uses the kept pack (current.linear.json) on both seats.
# The promotion gate is life-force only, both seats, LINEAR vs the previous
# kept linear pack (current.linear.json). Not HEURISTIC, not YodaBot/ADVANCED,
# not Beginner. Decision jsonl is a small sample; csv/json stay.
# --rounds 0 and --hours 0 disable the round count and the hour budget.
# Per-game max-decisions / max-millis still stop a single stuck game.
# --init is used only when current.linear.json is missing or unusable.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/linear-overnight
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
exec python3 -m trainer.improve.linear_overnight \
  --rounds 0 \
  --hours 0 \
  --games 8 \
  --max-batches 2 \
  --gate-per-seat 4 \
  --max-decisions 8000 \
  --max-millis 180000 \
  --lr 0.1 \
  --init zeros \
  --out-dir runs/linear-overnight \
  "$@"
