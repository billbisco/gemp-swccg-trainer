"""FastAPI local UI for SWCCG bot self-play collect + watch (day-1)."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from trainer.export.champ import export_stub_champ
from trainer.workers.orchestrator import ORCHESTRATOR, ROOT

UI_DIR = ROOT / "ui"
CHAMPS_DIR = ROOT / "champs"

app = FastAPI(title="GEMP SWCCG Trainer", version="0.1.0")


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "service": "gemp-swccg-trainer",
        "stage": "collect+watch",
        "learning": False,
        "note": "Learning does NOT happen until improve stage; this UI is collect + watch.",
    }


@app.get("/api/status")
def status():
    return ORCHESTRATOR.snapshot()


@app.post("/api/start")
def start(mode: str | None = Query(default=None, description="auto|maven|mock")):
    return ORCHESTRATOR.start(mode=mode)


@app.post("/api/pause")
def pause():
    return ORCHESTRATOR.pause()


@app.get("/api/watch")
def watch():
    return ORCHESTRATOR.last_game_payload()


@app.post("/api/export")
def export_champ():
    snap = ORCHESTRATOR.snapshot()
    result = export_stub_champ(
        CHAMPS_DIR,
        deck_hints=[
            snap.get("decks", {}).get("configured_dark") or "P-ANH 1996 World Champion (Dark)",
            snap.get("decks", {}).get("configured_light") or "P-ANH 1996 World Champion (Light)",
        ],
    )
    return result


@app.get("/api/config")
def config():
    cfg_path = ROOT / "trainer.toml"
    return {
        "root": str(ROOT),
        "runs": str(ROOT / "runs"),
        "decks": str(ROOT / "decks"),
        "champs": str(CHAMPS_DIR),
        "trainer_toml": cfg_path.read_text() if cfg_path.exists() else "",
    }


@app.get("/")
def index():
    index_path = UI_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>UI missing</h1><p>Expected ui/index.html</p>", status_code=500)
    return FileResponse(index_path)


if UI_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")


def main():
    import uvicorn

    host = "0.0.0.0"
    port = 8765
    # Prefer trainer.toml
    try:
        import tomllib
    except ImportError:
        tomllib = None  # type: ignore
    cfg_path = ROOT / "trainer.toml"
    if tomllib and cfg_path.exists():
        cfg = tomllib.loads(cfg_path.read_text())
        host = cfg.get("ui", {}).get("host", host)
        port = int(cfg.get("ui", {}).get("port", port))
    print(f"GEMP SWCCG Trainer UI → http://{host}:{port}/")
    print("Open in browser: http://127.0.0.1:%s/" % port)
    print("Learning does NOT run yet — collect + watch only.")
    uvicorn.run("trainer.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
