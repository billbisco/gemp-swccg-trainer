# gemp-swccg-trainer

Local self-play **collect + watch** UI for Star Wars CCG bots on GEMP.

> **Day-1 scope:** Start / Pause / Watch last game / Export champ stub.  
> The UI still collects games. Keyword mutate stays a baseline. The first **linear.v1** self-play step is `trainer.improve.linear_selfplay` (candidate only — it does not promote).

Companion to [`swccg-gemp`](https://github.com/billbisco/swccg-gemp) thin gym (`feature/headless-bot-vs-bot`). Architecture: see `/workspace/docs/swccg-bot-trainer-architecture.md` (local) / project docs.

## Learned policy (not trained)

Ratified design: [`docs/design-learned-policy-v1.md`](docs/design-learned-policy-v1.md) (2026-10-02).

AckbarBot's next strategy is a hybrid information set (`float32[128]` packed summary + sparse bags), self-play, and side-split win rate plus Life Force differential. Keyword mutate/blend (`heuristic.v1`) stays the baseline backend. It is not the training path for this design. **No learned pack has been promoted.** The first self-play step writes a candidate under `runs/linear-selfplay/` and stops there.

Schema stubs (contracts only):

- `schemas/information-set.v1.schema.json` — InformationSetV1; rejects `opponentDeckPriorKnown`
- `schemas/feature_layout_v1.json` — frozen 128-d layout, `seenHistoryCap` 64
- `schemas/experience.v1.schema.json` — step / episode lines; episode requires both final LFs
- `schemas/soft-metagame-belief.v1.schema.json` — separate soft belief store; must not be seeded from the exact opposing list
- `schemas/linear.v1.weights.schema.json` — gym-cli `LinearPolicyAi` pack (`W` over packed 128 + optional bag hash + actionFeat 24). Not for Hall.

### Run the first linear self-play step

Does not promote, does not set a shuffle seed, and does not start the keyword loop.

```bash
cd /workspace/gemp-swccg-trainer
PYTHONPATH=. python3 -m trainer.improve.linear_selfplay --dry-run
# 2 games as Dark and 2 as Light vs Beginner (default init is tiebreak):
PYTHONPATH=. python3 -m trainer.improve.linear_selfplay --live --games 2 --opponent BEGINNER --both-seats-vs-beginner
```

`--dry-run` fits the update on synthetic outcomes (no JVM). `--live` plays WC96 headless `LINEAR` vs `BEGINNER` (or `LINEAR`, which does not finish: tiebreak passes every phase and the game hits maxDecisions with life force unchanged) and reads FEATURES traces. Output: `runs/linear-selfplay/<stamp>/candidate.linear.json` plus win rate and mean life-force differential. `--gate` can measure N=2 vs Beginner and still does not copy into `champs/`.

Default `--init tiebreak` sets pass and integerNorm to +0.05 (everything else 0). An all-zero pack is legal but livelocks on WC96 Activate: ties keep the first candidate, which is Activate 0 Force, forever. `--init zeros` and `--init small-random` are also available. Gym shuffle seeds are not set.

Honest limit: greedy `linear.v1` adds `packed[128]` to every action, so those weights cannot change the choice. The update is a softmax surrogate of REINFORCE on action features 0..22, with one episode return (win + clipped LF diff) shared by every aligned decision. If options cannot be rebuilt, it nudges pass / integer / index from the side-split return — a stub, not a tactic.

Gym FEATURES traces (`-Dheadless.traceLevel=FEATURES`) already embed `state.packed`. This step does not change the gym.


## Quick start

```bash
cd /workspace/gemp-swccg-trainer   # or your clone path
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m trainer.app
```

Open in a browser on the **same machine running the trainer**:

- **http://127.0.0.1:8765/** (or the host you bound; default `0.0.0.0:8765`)
- **http://127.0.0.1:8765/progress** — live WC96 loop monitor (refreshes every 30 seconds)

`127.0.0.1` is local to the machine running `trainer.app`; a browser on another
computer cannot use the box's localhost. Run the trainer locally on that computer
(or use an explicitly configured network address).

### Windows quick start

From a clone of this repository, double-click `start-trainer.bat` (or run it from
Command Prompt). It creates `.venv`, installs the requirements on first run, and
starts the local server. Then open **http://127.0.0.1:8765/progress** in that same
Windows browser. The monitor reads `runs/wc96-loop/` in that local clone, so a
Windows copy will only show a loop started on Windows.

## UI controls

| Control | Behavior |
|---------|----------|
| **Start** | Spawns a background collect loop (`auto` → Maven headless if available, else **mock**) |
| **Pause** | Stops scheduling new games |
| **Watch last game** | Summary + JSONL decision timeline + paths to `runs/.../traces` and `runs/.../replays` |
| **Export champ** | Writes a schema-valid `heuristic.v1` **stub** zip under `champs/` |

## Decks

Default configured names (Librarian sample decks) — **playable headless now**:

- `P-ANH 1996 World Champion (Dark)` → `decks/wc96-anh-dark.txt` (`headless.decks=wc96`, format `premiere_anh`)
- `P-ANH 1996 World Champion (Light)` → `decks/wc96-anh-light.txt`
- Also available: Open 40 Beginner (`headless.decks=open40`, format `open`)

## GEMP replay (xml.gz)

Headless games can write **real** GEMP replays via `HeadlessReplayWriter` (same `xml.gz` event
format as Hall `GameRecorder`, **no** DB history row).

Committed sample (Watch fallback):

```
runs/_sample/wc96-replay/
  meta.json
  ~OzzelBot/z2lvxyskmomnws1c.xml.gz
  ~AckbarBot/m8pchsostwv5wq7g.xml.gz
  game-0001.jsonl
  README.md
```

**Open in a local GEMP Replay viewer**

1. Copy into your GEMP `application.root/replays/`:

```bash
GEMP_APP_ROOT=/path/to/gemp/app
mkdir -p "$GEMP_APP_ROOT/replays/~OzzelBot" "$GEMP_APP_ROOT/replays/~AckbarBot"
cp runs/_sample/wc96-replay/~OzzelBot/z2lvxyskmomnws1c.xml.gz "$GEMP_APP_ROOT/replays/~OzzelBot/"
cp runs/_sample/wc96-replay/~AckbarBot/m8pchsostwv5wq7g.xml.gz "$GEMP_APP_ROOT/replays/~AckbarBot/"
```

2. Open `game.html?replayId=~OzzelBot$z2lvxyskmomnws1c` (Dark POV) or
   `game.html?replayId=~AckbarBot$m8pchsostwv5wq7g` (Light POV).

Fresh Maven WC96 demo output (gitignored): `runs/wc96-demo/replays/game-0001/`.

## Runners

| Mode | What it does |
|------|----------------|
| `auto` (default) | Use Maven headless when available; else mock |
| `maven` | Force Maven (falls back to mock on failure) |
| `mock` | Writes plausible JSONL under `runs/<id>/traces/` so the UI works without Java |

Maven WC96 + replay (acceptance path):

```bash
cd /workspace/swccg-gemp/src
mvn -pl gemp-swccg-server -am -DfailIfNoTests=false \
  -Dtest=HeadlessBotVsBotBatchTest#wc96Anh_beginnerVsBeginner_writesReplay \
  -Dheadless.wc96Replay=true -Dheadless.games=1 \
  -Dheadless.replay.dir=/workspace/gemp-swccg-trainer/runs/wc96-demo/replays \
  test
```

## Layout

```
gemp-swccg-trainer/
├── trainer.toml          # parallelism, gym path, decks, UI port
├── trainer/app.py        # FastAPI entry
├── trainer/progress.py   # live WC96 progress reader
├── trainer/workers/      # Start/Pause orchestrator
├── trainer/export/       # heuristic.v1 stub exporter
├── ui/                   # vanilla SPA
├── schemas/              # champ + trace stubs
├── decks/                # WC96 + Open40 contents
├── runs/                 # job artifacts (traces, metrics, replays/)
├── scripts/write_status.py # optional STATUS.json refresh helper
└── champs/               # exported packs
```

## API

- `GET /api/health`
- `GET /api/status`
- `POST /api/start?mode=auto|maven|mock`
- `POST /api/pause`
- `GET /api/watch`
- `POST /api/export`
- `GET /api/config`
- `GET /api/progress` — live WC96 loop state; also refreshes `runs/wc96-loop/STATUS.json`

## Notes

- Trainer is **separate** from the GEMP war.
- Replay `xml.gz` is not produced until gym `HeadlessReplayWriter` lands; Watch still works via JSONL + folder path.
- No secrets in this repo. Keep `runs/` local (gitignored except sample).

## Improve / gate (minimal)

Collection ≠ learning. After a collect batch:

```bash
PYTHONPATH=. python3 -c "from trainer.improve.search import random_search, gate_candidate, export_candidate_champ"
# see docs/wc96-batch-50-REPORT.md for a worked WC96 example
```

Live eval needs gym `ConfigurableHeuristicAi` + `-Dheadless.dark.weights=` / `-Dheadless.light.weights=` on `feature/headless-bot-vs-bot`.
Promoted packs (if gate passes) land under `champs/_promoted/`.

