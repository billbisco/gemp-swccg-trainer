# WC96 batch-50 + improve/gate report

**Date:** 2026-10-02 (America/Caracas)  
**Honesty:** Collection ≠ learning. Upgrade only because the live gate passed.

## 1. Collect (50 games)

| Field | Value |
|-------|-------|
| Matchup | WC96 P-ANH 1996 Dark vs Light, Beginner vs Beginner |
| Format | `premiere_anh` |
| Gym | `swccg-gemp` `feature/headless-bot-vs-bot` `HeadlessBotVsBotBatch` |
| Finished / errors | **50 / 0** |
| Stop reason | `target_games_reached_50` (requested N=50) |
| Dark wins / WR | 17 / **34%** |
| Light wins / WR | 33 / **66%** |
| Avg decisions | **1330** (median 1150) |
| Avg duration | **4995 ms** (median 2981 ms) |
| Wall (batch play) | ~250 s (~4.2 min) |

### Artifact paths

- CSV: `/workspace/gemp-swccg-trainer/runs/wc96-batch-50/games.csv`
- Traces JSONL: `/workspace/gemp-swccg-trainer/runs/wc96-batch-50/traces/batch.jsonl` (66502 lines)
- Replays xml.gz: `/workspace/gemp-swccg-trainer/runs/wc96-batch-50/replays/game-0001` … `game-0050` (all 50 kept; ~4.4 MB)
- Summary JSON: `/workspace/gemp-swccg-trainer/runs/wc96-batch-50/summary.json`
- Maven log: `/workspace/gemp-swccg-trainer/runs/wc96-batch-50/maven.log`

Side WR skew is **deck asymmetry** under identical Beginner policies, not evidence of learning.

## 2. Improve (attempted)

- Module: `trainer/improve/search.py` (Gaussian mutate + Advanced-blend prior + trace keyword proxy ranker)
- Gym enablement (needed for live eval): `ConfigurableHeuristicAi` + `headless.dark.weights` / `headless.light.weights`
- Candidate selected: `blend-adv45` (Beginner weights blended ~45% toward Advanced action priors)
- Candidate weights: `runs/wc96-batch-50/improve/candidate-weights.json`

## 3. Gate (16 live games)

| Seat | Candidate WR | Baseline side WR (from collect) | Δ |
|------|--------------|----------------------------------|---|
| Dark (cand vs BEGINNER Light) | **0.50** (4/8) | 0.34 | +0.16 |
| Light (BEGINNER Dark vs cand) | **0.75** (6/8) | 0.66 | +0.09 |
| Combined | **0.625** (10/16) | expected 0.50 | +0.125 |

- **Gate: PASS** (side Δ≥0.05 and combined ≥ expected+0.05)
- Elo-ish est vs equal prior: ~1589
- Caveat: N=8/side is small; treat as a soft promote, re-gate with larger N before trusting heavily.
- Gate CSVs: `runs/wc96-batch-50/improve/gate/gate-cand-dark.csv`, `gate-cand-light.csv`

## 4. Champ decision

- **Upgraded champ:** `champs/heuristic-v1-wc96-batch50-cand1/` (`policyKind=heuristic.v1`)
- Baseline Beginner **replaced as champ pointer** for this run (pack marked `gatePassed=true`)
- Pack files: `manifest.json`, `weights.json`, `NOTES.md`, `gate.json`

## 5. What was NOT done

- No RL / imitation net
- Hall CHAMP load still needs gym Hall wiring (pack is data-only; headless can load via weights path)
