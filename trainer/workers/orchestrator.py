"""Day-1 training orchestrator: Start/Pause background collect loop.

Tries real Maven headless batch (Open 40 beginner decks on feature/headless-bot-vs-bot).
Falls back to a mock runner that writes plausible JSONL + summary so the UI works offline.
Learning / improve is NOT performed here — collect + watch only.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class GameSummary:
    game_index: int
    game_id: str
    winner: str | None
    finished: bool
    dark_lf: int
    light_lf: int
    dark_turns: int
    light_turns: int
    decisions: int
    elapsed_ms: int
    dark_ai: str
    light_ai: str
    mode: str  # "maven" | "mock"
    notes: str = ""
    traces_path: str = ""
    replay_dir: str = ""


@dataclass
class RunState:
    status: str = "idle"  # idle|running|paused|error
    run_id: str | None = None
    run_dir: str | None = None
    mode: str = "auto"  # auto|maven|mock
    actual_mode: str | None = None
    games_completed: int = 0
    dark_wins: int = 0
    light_wins: int = 0
    unfinished: int = 0
    started_at: float | None = None
    last_error: str | None = None
    last_game: dict | None = None
    notes: list[str] = field(default_factory=list)
    pause_requested: bool = False


class Orchestrator:
    def __init__(self, root: Path | None = None):
        self.root = Path(root or ROOT)
        self.state = RunState()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._cfg = self._load_toml()

    def _load_toml(self) -> dict:
        path = self.root / "trainer.toml"
        text = path.read_text() if path.exists() else ""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore
        return tomllib.loads(text) if text.strip() else {}

    def snapshot(self) -> dict:
        with self._lock:
            s = asdict(self.state)
            elapsed = 0.0
            if self.state.started_at and self.state.status == "running":
                elapsed = time.time() - self.state.started_at
            elif self.state.started_at:
                elapsed = max(0.0, time.time() - self.state.started_at)
            games = max(1, self.state.games_completed)
            s["games_per_hour"] = (
                round(self.state.games_completed / (elapsed / 3600.0), 1)
                if elapsed > 1
                else None
            )
            total_decided = self.state.dark_wins + self.state.light_wins
            s["dark_win_rate"] = (
                round(self.state.dark_wins / total_decided, 3) if total_decided else None
            )
            s["light_win_rate"] = (
                round(self.state.light_wins / total_decided, 3) if total_decided else None
            )
            s["decks"] = {
                "configured_dark": self._cfg.get("decks", {}).get("dark_name"),
                "configured_light": self._cfg.get("decks", {}).get("light_name"),
                "playable_note": (
                    "Maven headless supports headless.decks=wc96 (P-ANH 1996) with "
                    "premiere_anh + HeadlessReplayWriter xml.gz under runs/.../replays/."
                ),
                "sample_replay_dir": str(self.root / "runs" / "_sample" / "wc96-replay"),
            }
            return s

    def start(self, mode: str | None = None) -> dict:
        with self._lock:
            if self.state.status == "running":
                already = True
            else:
                already = False
                self._cfg = self._load_toml()
                requested = (mode or self._cfg.get("gym", {}).get("runner") or "auto").lower()
                run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
                run_dir = self.root / "runs" / run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "replays").mkdir(exist_ok=True)
                (run_dir / "traces").mkdir(exist_ok=True)
                self.state = RunState(
                    status="running",
                    run_id=run_id,
                    run_dir=str(run_dir),
                    mode=requested,
                    started_at=time.time(),
                    pause_requested=False,
                    notes=[],
                )
                self._thread = threading.Thread(
                    target=self._loop, args=(run_dir, requested), daemon=True, name="trainer-collect"
                )
                self._thread.start()
        return self.snapshot()

    def pause(self) -> dict:
        with self._lock:
            if self.state.status == "running":
                self.state.pause_requested = True
                self.state.status = "paused"
                self.state.notes.append("Pause requested — no new games will start.")
        return self.snapshot()

    def last_game_payload(self) -> dict:
        with self._lock:
            last = self.state.last_game
            run_dir = self.state.run_dir
        if not last and run_dir:
            # Try load from disk
            summary_path = Path(run_dir) / "last_game.json"
            if summary_path.exists():
                last = json.loads(summary_path.read_text())
        if not last:
            # Fall back to committed WC96 sample replay + traces
            sample_dir = self.root / "runs" / "_sample" / "wc96-replay"
            sample_last = sample_dir / "last_game.json"
            sample_traces = sample_dir / "game-0001.jsonl"
            legacy = self.root / "runs" / "_sample" / "sample-traces.jsonl"
            if sample_last.exists():
                last = json.loads(sample_last.read_text())
                return {
                    "available": True,
                    "summary": last,
                    "timeline": self._read_jsonl(sample_traces, limit=500) if sample_traces.exists() else [],
                    "replay_dir": str(sample_dir),
                    "traces_path": str(sample_traces) if sample_traces.exists() else None,
                    "run_dir": str(sample_dir),
                    "message": "Showing committed WC96 sample (no live run yet). Click Start for a fresh game.",
                }
            return {
                "available": False,
                "message": "No games yet. Click Start, or browse sample traces.",
                "sample_traces_path": str(legacy) if legacy.exists() else None,
                "timeline": self._read_jsonl(legacy, limit=80) if legacy.exists() else [],
                "replay_dir": None,
            }
        traces = Path(last.get("traces_path") or "")
        timeline = self._read_jsonl(traces, limit=500) if traces.exists() else []
        return {
            "available": True,
            "summary": last,
            "timeline": timeline,
            "replay_dir": last.get("replay_dir"),
            "traces_path": last.get("traces_path"),
            "run_dir": run_dir,
        }

    @staticmethod
    def _read_jsonl(path: Path, limit: int = 500) -> list[dict]:
        rows: list[dict] = []
        try:
            with path.open() as f:
                for i, line in enumerate(f):
                    if i >= limit:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        rows.append({"raw": line, "parseError": True})
        except OSError:
            pass
        return rows

    def _loop(self, run_dir: Path, requested: str) -> None:
        actual = self._resolve_mode(requested)
        with self._lock:
            self.state.actual_mode = actual
            if actual == "mock":
                self.state.notes.append(
                    "Running MOCK collect loop (Maven headless unavailable or runner=mock). "
                    "Writes plausible JSONL — not real GEMP games."
                )
            else:
                decks = self._cfg.get("gym", {}).get("decks", "wc96")
                self.state.notes.append(
                    f"Running MAVEN headless batch (decks={decks}, replay="
                    f"{self._cfg.get('gym', {}).get('replay', True)})."
                )
        game_index = 0
        try:
            while True:
                with self._lock:
                    if self.state.pause_requested or self.state.status != "running":
                        self.state.status = "paused"
                        break
                game_index += 1
                if actual == "maven":
                    summary = self._run_maven_game(run_dir, game_index)
                else:
                    summary = self._run_mock_game(run_dir, game_index)
                self._record_game(summary)
                # Brief pause so UI can poll
                time.sleep(0.2 if actual == "mock" else 0.5)
        except Exception as e:  # noqa: BLE001
            with self._lock:
                self.state.status = "error"
                self.state.last_error = str(e)
                self.state.notes.append(f"Orchestrator error: {e}")

    def _resolve_mode(self, requested: str) -> str:
        if requested == "mock":
            return "mock"
        if requested in ("maven", "auto"):
            if self._maven_available():
                return "maven"
            if requested == "maven":
                with self._lock:
                    self.state.notes.append(
                        "Maven headless requested but unavailable — falling back to mock."
                    )
            return "mock"
        return "mock"

    def _maven_available(self) -> bool:
        gym = self._cfg.get("gym", {})
        gemp = Path(gym.get("gemp_repo", "/workspace/swccg-gemp"))
        if not (gemp / "src" / "pom.xml").exists():
            return False
        if shutil.which("mvn") is None:
            return False
        # Compiled test classes OR source on headless branch
        test_classes = (
            gemp
            / "src"
            / "gemp-swccg-server"
            / "target"
            / "test-classes"
            / "com"
            / "gempukku"
            / "swccgo"
            / "ai"
            / "HeadlessBotVsBotBatchTest.class"
        )
        src_java = (
            gemp
            / "src"
            / "gemp-swccg-server"
            / "src"
            / "test"
            / "java"
            / "com"
            / "gempukku"
            / "swccgo"
            / "ai"
            / "HeadlessBotVsBotBatchTest.java"
        )
        # Also accept if git has the branch (we can checkout worktree later)
        has_branch = (gemp / ".git" / "refs" / "heads" / "feature" / "headless-bot-vs-bot").exists()
        return test_classes.exists() or src_java.exists() or has_branch

    def _ensure_headless_sources(self, gemp: Path) -> bool:
        """Checkout feature/headless-bot-vs-bot into a disposable worktree if needed."""
        src = (
            gemp
            / "src"
            / "gemp-swccg-server"
            / "src"
            / "test"
            / "java"
            / "com"
            / "gempukku"
            / "swccgo"
            / "ai"
            / "HeadlessBotVsBotBatchTest.java"
        )
        if src.exists():
            return True
        worktree = Path("/tmp/swccg-gemp-headless-wt")
        if (worktree / "src" / "gemp-swccg-server" / "src" / "test" / "java" / "com" / "gempukku" / "swccgo" / "ai" / "HeadlessBotVsBotBatchTest.java").exists():
            return True
        try:
            if worktree.exists():
                shutil.rmtree(worktree, ignore_errors=True)
            subprocess.run(
                ["git", "worktree", "add", "--detach", str(worktree), "feature/headless-bot-vs-bot"],
                cwd=str(gemp),
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
            with self._lock:
                self.state.notes.append(f"Could not prepare headless worktree: {e}")
            return False

    def _run_maven_game(self, run_dir: Path, game_index: int) -> GameSummary:
        gym = self._cfg.get("gym", {})
        gemp = Path(gym.get("gemp_repo", "/workspace/swccg-gemp"))
        # Prefer worktree if current branch lacks sources
        src_root = gemp / "src"
        wt = Path("/tmp/swccg-gemp-headless-wt")
        if not (
            src_root
            / "gemp-swccg-server"
            / "src"
            / "test"
            / "java"
            / "com"
            / "gempukku"
            / "swccgo"
            / "ai"
            / "HeadlessBotVsBotBatchTest.java"
        ).exists():
            if self._ensure_headless_sources(gemp):
                src_root = wt / "src"

        traces_path = run_dir / "traces" / f"game-{game_index:04d}.jsonl"
        csv_path = run_dir / "traces" / f"game-{game_index:04d}.csv"
        replay_dir = run_dir / "replays"
        log_path = run_dir / "maven.log"

        dark = gym.get("dark_ai", "BEGINNER")
        light = gym.get("light_ai", "BEGINNER")
        max_millis = int(gym.get("max_millis", 180000))

        decks = gym.get("decks", "wc96")
        fmt = gym.get("format", "premiere_anh" if decks == "wc96" else "open")
        want_replay = bool(gym.get("replay", True))
        cmd = [
            "mvn",
            "-pl",
            "gemp-swccg-server",
            "-am",
            "-DfailIfNoTests=false",
            "-Dtest=HeadlessBotVsBotBatchTest#batchSelfPlay_writesCsv",
            f"-Dheadless.games=1",
            f"-Dheadless.dark={dark}",
            f"-Dheadless.light={light}",
            f"-Dheadless.decks={decks}",
            f"-Dheadless.format={fmt}",
            f"-Dheadless.csv={csv_path}",
            f"-Dheadless.maxMillis={max_millis}",
            "-Dheadless.traces=true",
            f"-Dheadless.traces.path={traces_path}",
            "-Dheadless.verbose=false",
        ]
        if want_replay:
            cmd.append("-Dheadless.replay=true")
            cmd.append(f"-Dheadless.replay.dir={replay_dir}")
        cmd.append("test")
        # Note: Maven property paths for traces may be relative to module cwd.
        # Also write absolute via symlink after run if needed.
        env = os.environ.copy()
        started = time.time()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(src_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=max_millis / 1000.0 + 300,
            )
            log_path.write_text(
                f"$ {' '.join(cmd)}\n\n=== stdout ===\n{proc.stdout[-20000:]}\n\n=== stderr ===\n{proc.stderr[-20000:]}\n"
            )
            if proc.returncode != 0:
                # Fall back to mock for this game so UI keeps working
                with self._lock:
                    self.state.notes.append(
                        f"Maven game {game_index} failed (rc={proc.returncode}); writing mock game instead. See maven.log."
                    )
                return self._run_mock_game(run_dir, game_index, note="maven-failed-fallback")

            # Maven may write traces under module target/ if property path was relative.
            # Search for newest jsonl if expected path missing.
            if not traces_path.exists():
                candidates = list((src_root / "gemp-swccg-server" / "target").glob("**/*decision*.jsonl"))
                candidates += list(run_dir.glob("**/*.jsonl"))
                if candidates:
                    newest = max(candidates, key=lambda p: p.stat().st_mtime)
                    shutil.copy2(newest, traces_path)

            winner, finished, dark_lf, light_lf, decisions, ds_turn, ls_turn = self._parse_outcome(
                traces_path, csv_path
            )
            elapsed = int((time.time() - started) * 1000)
            return GameSummary(
                game_index=game_index,
                game_id=f"maven-{run_dir.name}-{game_index}",
                winner=winner,
                finished=finished,
                dark_lf=dark_lf,
                light_lf=light_lf,
                dark_turns=ds_turn,
                light_turns=ls_turn,
                decisions=decisions,
                elapsed_ms=elapsed,
                dark_ai=dark,
                light_ai=light,
                mode="maven",
                notes=f"decks={decks} format={fmt} replay={want_replay}",
                traces_path=str(traces_path),
                replay_dir=str(replay_dir / f"game-{game_index:04d}") if want_replay else str(replay_dir),
            )
        except subprocess.TimeoutExpired:
            with self._lock:
                self.state.notes.append(f"Maven game {game_index} timed out; mock fallback.")
            return self._run_mock_game(run_dir, game_index, note="maven-timeout-fallback")

    def _parse_outcome(
        self, traces_path: Path, csv_path: Path
    ) -> tuple[str | None, bool, int, int, int, int, int]:
        winner = None
        finished = False
        dark_lf = 0
        light_lf = 0
        decisions = 0
        ds_turn = 0
        ls_turn = 0
        if csv_path.exists():
            lines = csv_path.read_text().strip().splitlines()
            if len(lines) >= 2:
                # Positional prefix is stable. Later columns (format, decks, error,
                # darkLifeForce, lightLifeForce) are read by name when the header has them.
                header = lines[0].split(",")
                parts = lines[-1].split(",")
                if len(parts) >= 9:
                    winner = parts[3] or None
                    finished = bool(winner) and not (parts[9] if len(parts) > 9 else "")
                    try:
                        decisions = int(parts[4] or 0) + int(parts[5] or 0)
                        ds_turn = int(parts[6] or 0)
                        ls_turn = int(parts[7] or 0)
                    except ValueError:
                        pass
                    def named(col: str):
                        if col not in header:
                            return None
                        i = header.index(col)
                        if i >= len(parts) or parts[i] == "":
                            return None
                        return parts[i]
                    try:
                        dlf = named("darkLifeForce")
                        llf = named("lightLifeForce")
                        if dlf is not None:
                            dark_lf = int(dlf)
                        if llf is not None:
                            light_lf = int(llf)
                    except ValueError:
                        pass
        if traces_path.exists():
            last = None
            with traces_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        last = json.loads(line)
                        decisions = max(decisions, int(last.get("decisionIndex") or 0))
                    except json.JSONDecodeError:
                        continue
            if last:
                dark_lf = int(last.get("darkLF") or 0)
                light_lf = int(last.get("lightLF") or 0)
                if last.get("side") == "DARK":
                    ds_turn = max(ds_turn, int(last.get("turn") or 0))
                else:
                    ls_turn = max(ls_turn, int(last.get("turn") or 0))
                if last.get("type") == "outcome":
                    winner = last.get("winner") or winner
                    finished = bool(last.get("finished", finished))
        if winner:
            finished = True
        return winner, finished, dark_lf, light_lf, decisions, ds_turn, ls_turn

    def _run_mock_game(
        self, run_dir: Path, game_index: int, note: str = ""
    ) -> GameSummary:
        traces_path = run_dir / "traces" / f"game-{game_index:04d}.jsonl"
        replay_dir = run_dir / "replays"
        game_id = f"mock-{uuid.uuid4()}"
        dark_ai = self._cfg.get("gym", {}).get("dark_ai", "BEGINNER")
        light_ai = self._cfg.get("gym", {}).get("light_ai", "BEGINNER")
        # Simulate a short game with LF race ending
        decisions = random.randint(80, 220)
        dark_lf = 40
        light_lf = 40
        rows: list[dict] = []
        header = {
            "type": "header",
            "schemaVersion": 1,
            "gameId": game_id,
            "gameIndex": game_index,
            "mode": "mock",
            "note": "MOCK game — not a real GEMP match. Day-1 UI collect stub.",
            "darkDeck": "P-ANH 1996 World Champion (Dark) [staged; not executed]",
            "lightDeck": "P-ANH 1996 World Champion (Light) [staged; not executed]",
            "ts": int(time.time() * 1000),
        }
        rows.append(header)
        phases = ["PLAY_STARTING_CARDS", "ACTIVATE", "CONTROL", "DEPLOY", "BATTLE", "MOVE", "DRAW"]
        for i in range(1, decisions + 1):
            side = "DARK" if i % 2 else "LIGHT"
            player = "~OzzelBot" if side == "DARK" else "~AckbarBot"
            if i % 17 == 0:
                if side == "DARK":
                    light_lf = max(0, light_lf - random.randint(1, 3))
                else:
                    dark_lf = max(0, dark_lf - random.randint(1, 3))
            rows.append(
                {
                    "ts": int(time.time() * 1000) + i,
                    "type": "decision",
                    "decisionIndex": i,
                    "gameId": game_id,
                    "gameIndex": game_index,
                    "playerId": player,
                    "side": side,
                    "aiSkill": dark_ai if side == "DARK" else light_ai,
                    "turn": 1 + i // 30,
                    "phase": phases[min(len(phases) - 1, i // 12)],
                    "decisionType": "CARD_ACTION_CHOICE" if i % 5 else "MULTIPLE_CHOICE",
                    "decisionId": i,
                    "decisionText": "Mock decision — select action",
                    "options": {"items": [{"text": "OK"}, {"text": "Pass"}]},
                    "optionCount": 2,
                    "chosen": "0",
                    "accepted": True,
                    "darkLF": dark_lf,
                    "lightLF": light_lf,
                    "darkHand": random.randint(0, 8),
                    "lightHand": random.randint(0, 8),
                }
            )
            if dark_lf == 0 or light_lf == 0:
                decisions = i
                break
        if dark_lf == 0 and light_lf == 0:
            winner = random.choice(["~OzzelBot", "~AckbarBot"])
        elif light_lf == 0:
            winner = "~OzzelBot"
        elif dark_lf == 0:
            winner = "~AckbarBot"
        else:
            # Force a winner for mock completeness
            winner = "~OzzelBot" if dark_lf >= light_lf else "~AckbarBot"
            if winner == "~OzzelBot":
                light_lf = 0
            else:
                dark_lf = 0
        outcome = {
            "type": "outcome",
            "schemaVersion": 1,
            "gameId": game_id,
            "gameIndex": game_index,
            "finished": True,
            "winner": winner,
            "darkLF": dark_lf,
            "lightLF": light_lf,
            "decisionCount": decisions,
            "mode": "mock",
            "note": note or "mock",
        }
        rows.append(outcome)
        with traces_path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        # Mock does not write xml.gz — point at committed WC96 sample for Watch.
        sample = self.root / "runs" / "_sample" / "wc96-replay"
        (replay_dir / "README.txt").write_text(
            "MOCK games do not write GEMP xml.gz.\n"
            f"See real sample at {sample}\n"
            f"Decision JSONL: {traces_path}\n"
        )
        elapsed = random.randint(800, 3500)
        return GameSummary(
            game_index=game_index,
            game_id=game_id,
            winner=winner,
            finished=True,
            dark_lf=dark_lf,
            light_lf=light_lf,
            dark_turns=1 + decisions // 30,
            light_turns=1 + decisions // 30,
            decisions=decisions,
            elapsed_ms=elapsed,
            dark_ai=dark_ai,
            light_ai=light_ai,
            mode="mock",
            notes=note or "MOCK collect — learning does not happen until improve stage.",
            traces_path=str(traces_path),
            replay_dir=str(replay_dir),
        )

    def _record_game(self, summary: GameSummary) -> None:
        with self._lock:
            self.state.games_completed += 1
            if summary.finished and summary.winner:
                if "Ozzel" in summary.winner or summary.winner.endswith("Dark"):
                    self.state.dark_wins += 1
                elif "Ackbar" in summary.winner or "Yoda" in summary.winner or "Light" in summary.winner:
                    self.state.light_wins += 1
                else:
                    # Heuristic: dark seat id vs light
                    if summary.winner.startswith("~O"):
                        self.state.dark_wins += 1
                    else:
                        self.state.light_wins += 1
            else:
                self.state.unfinished += 1
            payload = asdict(summary)
            self.state.last_game = payload
            if self.state.run_dir:
                Path(self.state.run_dir, "last_game.json").write_text(json.dumps(payload, indent=2) + "\n")
                metrics = Path(self.state.run_dir, "metrics.jsonl")
                with metrics.open("a") as f:
                    f.write(
                        json.dumps(
                            {
                                "ts": time.time(),
                                "games_completed": self.state.games_completed,
                                "dark_wins": self.state.dark_wins,
                                "light_wins": self.state.light_wins,
                                "game": payload,
                            }
                        )
                        + "\n"
                    )


ORCHESTRATOR = Orchestrator()
