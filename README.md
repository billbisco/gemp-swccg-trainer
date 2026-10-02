# gemp-swccg-trainer

Local self-play **collect + watch** UI for Star Wars CCG bots on GEMP.

> **Day-1 scope:** Start / Pause / Watch last game / Export champ stub.  
> **Learning does NOT happen yet** — the improve stage is later. This UI orchestrates game collection and lets you inspect JSONL decision timelines offline.

Companion to [`swccg-gemp`](https://github.com/billbisco/swccg-gemp) thin gym (`feature/headless-bot-vs-bot`). Architecture: see `/workspace/docs/swccg-bot-trainer-architecture.md` (local) / project docs.

## Quick start

```bash
cd /workspace/gemp-swccg-trainer   # or your clone path
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m trainer.app
```

Open in a browser:

- **http://127.0.0.1:8765/** (or the host you bound; default `0.0.0.0:8765`)

## UI controls

| Control | Behavior |
|---------|----------|
| **Start** | Spawns a background collect loop (`auto` → Maven headless if available, else **mock**) |
| **Pause** | Stops scheduling new games |
| **Watch last game** | Summary + JSONL decision timeline + paths to `runs/.../traces` and `runs/.../replays` |
| **Export champ** | Writes a schema-valid `heuristic.v1` **stub** zip under `champs/` |

## Decks

Default configured names (Librarian sample decks):

- `P-ANH 1996 World Champion (Dark)` → `decks/wc96-anh-dark.txt`
- `P-ANH 1996 World Champion (Light)` → `decks/wc96-anh-light.txt`

**Playable today via Maven headless spike:** Open 40 Beginner Dark/Light (hardcoded in `HeadlessBotVsBotRunner`). WC96 contents are staged for future `file:` deck loading in gym-cli.

## Runners

| Mode | What it does |
|------|----------------|
| `auto` (default) | Use Maven `HeadlessBotVsBotBatchTest` when `swccg-gemp` headless sources/classes exist; else mock |
| `maven` | Force Maven (falls back to mock on failure) |
| `mock` | Writes plausible JSONL under `runs/<id>/traces/` so the UI works without Java |

Maven invocation (when used) runs from the GEMP reactor `src/` (or a detached worktree of `feature/headless-bot-vs-bot`):

```bash
mvn -pl gemp-swccg-server -am \
  -Dtest=HeadlessBotVsBotBatchTest#batchSelfPlay_writesCsv \
  -Dheadless.games=1 -Dheadless.traces=true ...
```

## Layout

```
gemp-swccg-trainer/
├── trainer.toml          # parallelism, gym path, decks, UI port
├── trainer/app.py        # FastAPI entry
├── trainer/workers/      # Start/Pause orchestrator
├── trainer/export/       # heuristic.v1 stub exporter
├── ui/                   # vanilla SPA
├── schemas/              # champ + trace stubs
├── decks/                # WC96 + Open40 contents
├── runs/                 # job artifacts (traces, metrics, replays/)
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

## Notes

- Trainer is **separate** from the GEMP war.
- Replay `xml.gz` is not produced until gym `HeadlessReplayWriter` lands; Watch still works via JSONL + folder path.
- No secrets in this repo. Keep `runs/` local (gitignored except sample).
