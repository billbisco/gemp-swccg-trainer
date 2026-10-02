# WC96 parallel Java gym workers

**Date:** 2026-10-02 (America/Caracas)  
**Change:** collect/gate now shard games across concurrent `HeadlessBotVsBotBatch` JVM
processes (direct CLI classpath — not more Grok bots). Replay/traces off.

## Bench (24 WC96 games, champ weights as Dark vs BEGINNER Light)

| Mode | Workers | Wall | Games/hour |
|------|---------|------|------------|
| Prior Maven serial collect (50/300s) | 1 | — | **~600** |
| Direct CLI serial | 1 | 101.3 s | **852** |
| Direct CLI parallel | **6** | 46.4 s | **1860** |
| Direct CLI parallel | 8 | 44.3 s | **1951** |

**Chosen default:** `GEMP_GYM_WORKERS=6` (~3.1× vs prior Maven serial; 8w only +5% more).

## Notes

- Classpath from surefire `classPathUrl.*` snapshot → `runs/wc96-loop/gym.classpath`
- Merge shard CSVs for honest gates
- Future benchmark (not integrated): Peter Soetens browser model ~90% vs Advanced
