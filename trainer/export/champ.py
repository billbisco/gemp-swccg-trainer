"""Export schema-valid heuristic.v1 stub champ packs (day-1)."""
from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# Baseline BeginnerAi-ish keyword weights (illustrative stub matching architecture doc).
BEGINNER_WEIGHTS = {
    "base": "BEGINNER",
    "actionWeights": [
        {"k": "force drain", "w": 120},
        {"k": "initiate battle", "w": 110},
        {"k": "battle", "w": 70},
        {"k": "weapon", "w": 40},
        {"k": "fire", "w": 35},
        {"k": "deploy", "w": 60},
        {"k": "play", "w": 35},
        {"k": "move", "w": 35},
        {"k": "activate", "w": 50},
        {"k": "retrieve", "w": 30},
        {"k": "draw", "w": 25},
        {"k": "steal", "w": 25},
        {"k": "capture", "w": 25},
        {"k": "react", "w": 20},
        {"k": "take into hand", "w": 30},
    ],
    "actionPenalties": [
        {"k": "pass", "w": -120},
        {"k": "forfeit", "w": -80},
        {"k": "lose", "w": -45},
        {"k": "place in lost pile", "w": -70},
        {"k": "place in used pile", "w": -35},
        {"k": "return to hand", "w": -30},
        {"k": "sacrifice", "w": -80},
        {"k": "revert", "w": -40},
    ],
    "choiceWeights": [
        {"k": "draw", "w": 40},
        {"k": "retrieve", "w": 35},
        {"k": "deploy", "w": 30},
        {"k": "battle destiny", "w": 35},
        {"k": "weapon destiny", "w": 35},
        {"k": "activate", "w": 30},
        {"k": "force drain", "w": 40},
        {"k": "initiate", "w": 30},
        {"k": "capture", "w": 20},
        {"k": "steal", "w": 20},
        {"k": "use", "w": 5},
        {"k": "yes", "w": 5},
    ],
    "choicePenalties": [
        {"k": "lose", "w": -45},
        {"k": "forfeit", "w": -60},
        {"k": "lost pile", "w": -50},
        {"k": "used pile", "w": -30},
        {"k": "return to hand", "w": -25},
        {"k": "neither", "w": -25},
        {"k": "cancel", "w": -20},
        {"k": "pass", "w": -30},
    ],
    "cardHints": [
        "pilot",
        "weapon",
        "character",
        "starship",
        "vehicle",
        "droid",
        "alien",
        "jedi",
        "sith",
        "effect",
        "interrupt",
        "location",
        "site",
        "system",
    ],
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def export_stub_champ(
    champs_dir: Path,
    *,
    champ_id: str | None = None,
    display_name: str | None = None,
    parent_id: str = "builtin-BEGINNER",
    deck_hints: list[str] | None = None,
) -> dict:
    """Write champ dir + zip; return manifest + paths + hash."""
    champs_dir = Path(champs_dir)
    champs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    champ_id = champ_id or f"stub-{stamp}"
    display_name = display_name or f"Beginner-Stub-{stamp}"
    pack_dir = champs_dir / champ_id
    pack_dir.mkdir(parents=True, exist_ok=True)

    weights_path = pack_dir / "weights.json"
    weights_path.write_text(json.dumps(BEGINNER_WEIGHTS, indent=2) + "\n")
    weights_hash = _sha256_file(weights_path)

    notes = pack_dir / "NOTES.md"
    notes.write_text(
        "# Champ stub (day-1)\n\n"
        "Schema-valid `heuristic.v1` baseline weights.\n"
        "Learning / improve stage is NOT active yet — this is collect+watch UI export only.\n"
    )

    manifest = {
        "manifestVersion": 1,
        "id": champ_id,
        "displayName": display_name,
        "policyKind": "heuristic.v1",
        "gymApiVersion": "0.0.0-day1",
        "createdAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "parentId": parent_id,
        "elo": None,
        "ladderId": None,
        "deckHints": deck_hints
        or [
            "P-ANH 1996 World Champion (Dark)",
            "P-ANH 1996 World Champion (Light)",
        ],
        "files": {"weights": "weights.json"},
        "hash": {"alg": "sha256", "value": weights_hash},
        "signature": None,
        "stub": True,
        "note": "Day-1 stub; improve stage not yet wired.",
    }
    manifest_path = pack_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    zip_path = champs_dir / f"{champ_id}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in ("manifest.json", "weights.json", "NOTES.md"):
            zf.write(pack_dir / name, arcname=f"{champ_id}/{name}")

    zip_hash = _sha256_file(zip_path)
    return {
        "id": champ_id,
        "dir": str(pack_dir.resolve()),
        "zip": str(zip_path.resolve()),
        "weightsSha256": weights_hash,
        "zipSha256": zip_hash,
        "manifest": manifest,
    }
