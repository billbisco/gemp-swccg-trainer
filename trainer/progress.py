"""Live WC96 loop progress for the local monitoring page."""
from __future__ import annotations

import ast
import json
import os
import re
import tempfile
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "wc96-loop"
STATUS_PATH = RUN_DIR / "STATUS.json"
LOG_PATH = RUN_DIR / "loop.stdout.log"
PID_PATH = RUN_DIR / "loop.pid"
CURRENT_PATH = ROOT / "champs" / "_promoted" / "CURRENT"

_ROUND_RE = re.compile(r"^round(\d+)(?:-|$)")
_GAMES_PER_HOUR_RE = re.compile(r"(?:gamesPerHour|games/hour)\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)", re.I)


def _tail(path: Path, limit: int = 120) -> list[str]:
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return list(deque((line.rstrip("\n") for line in handle), maxlen=limit))
    except OSError:
        return []


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, TypeError):
        return None


def _collect_from_line(line: str) -> dict[str, Any] | None:
    marker = "[collect] done:"
    if marker not in line:
        return None
    try:
        value = ast.literal_eval(line.split(marker, 1)[1].strip())
        return value if isinstance(value, dict) else None
    except (SyntaxError, ValueError):
        return None


def _pid_state() -> tuple[int | None, bool]:
    try:
        pid = int(PID_PATH.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None, False
    if pid <= 0:
        return pid, False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return pid, False
    except PermissionError:
        return pid, True
    return pid, True


def _current_champ() -> str | None:
    try:
        value = CURRENT_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


def _latest_round() -> dict[str, Any] | None:
    if not RUN_DIR.is_dir():
        return None
    folders: list[tuple[int, Path]] = []
    for path in RUN_DIR.iterdir():
        if not path.is_dir():
            continue
        match = _ROUND_RE.match(path.name)
        if match:
            folders.append((int(match.group(1)), path))
    if not folders:
        return None
    # mtime handles repeated attempts of the same round while it is running.
    _, folder = max(folders, key=lambda item: (item[1].stat().st_mtime, item[0], item[1].name))
    match = _ROUND_RE.match(folder.name)
    number = int(match.group(1)) if match else None

    collect_summary: dict[str, Any] | None = None
    summary_files = sorted(
        folder.glob("**/*.summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if summary_files:
        collect_summary = _read_json(summary_files[0])

    return {
        "number": number,
        "folder": folder.name,
        "path": str(folder),
        "collect": collect_summary,
        "result": _read_json(folder / "round_result.json"),
    }


def _last_gate_or_promote(lines: list[str]) -> str | None:
    for line in reversed(lines):
        if any(marker in line for marker in ("[PROMOTE]", "[no-promote]", "[gate]")):
            return line
    return None


def _write_status(status: dict[str, Any]) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    # Replace atomically so a browser never sees half-written JSON.
    fd, temp_name = tempfile.mkstemp(prefix="STATUS.", suffix=".tmp", dir=RUN_DIR)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(status, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temp_name, STATUS_PATH)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def build_progress(*, write_status: bool = True) -> dict[str, Any]:
    """Compute progress from live files, optionally refreshing STATUS.json."""
    lines = _tail(LOG_PATH)
    pid, alive = _pid_state()
    current = _current_champ()
    latest_round = _latest_round()

    last_collect: dict[str, Any] | None = None
    for line in reversed(lines):
        last_collect = _collect_from_line(line)
        if last_collect is not None:
            break
    if last_collect is None and latest_round:
        last_collect = latest_round.get("collect")

    games_per_hour = None
    if last_collect:
        games_per_hour = last_collect.get("gamesPerHour")
    if games_per_hour is None:
        for line in reversed(lines):
            match = _GAMES_PER_HOUR_RE.search(line)
            if match:
                games_per_hour = float(match.group(1))
                break

    status: dict[str, Any] = {
        "updatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "loopAlive": alive,
        "loopPid": pid,
        "currentChamp": current,
        "lastCollect": last_collect,
        "lastCollectWR": {
            "dark": last_collect.get("darkWinRate") if last_collect else None,
            "light": last_collect.get("lightWinRate") if last_collect else None,
        },
        "lastPromoteGateLine": _last_gate_or_promote(lines),
        "gamesPerHour": games_per_hour,
        "roundNumber": latest_round.get("number") if latest_round else None,
        "latestRound": latest_round,
        "tail": lines[-40:],
        "paths": {
            "status": str(STATUS_PATH),
            "log": str(LOG_PATH),
            "pid": str(PID_PATH),
            "current": str(CURRENT_PATH),
        },
    }
    if write_status:
        _write_status(status)
    return status
