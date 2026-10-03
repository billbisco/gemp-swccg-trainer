"""Overnight LINEAR vs LINEAR self-play loop for AckbarBot.

Uses ``trainer.improve.linear_selfplay`` for the pack, the gym batch, and the
update. Self-play is LINEAR vs LINEAR with one ``--linear-weights`` file on
both seats. This loop is the unattended part: many rounds, random shuffles,
a head-to-head gate, and a training-champ pointer.

What it does not do
-------------------
* Does not distill YodaBot / AdvancedAi.
* Does not run the keyword mutate loop (``trainer.improve.search`` / ``loop``).
* Does not write under ``champs/`` and never touches
  ``champs/_promoted/heuristic-v1-wc96-r08-blend-adv25``.
* Does not pass a gym shuffle seed. Strength numbers are not from a fixed seed.
* Does not gate against BEGINNER.

Promotion
---------
After each self-play update, the candidate plays the previous kept AckbarBot
linear pack (``current.linear.json`` before the swap). Candidate Dark vs kept
Light, then candidate Light vs kept Dark. Different packs use
``--dark-weights`` / ``--light-weights``. WC96, premiere_anh, no shuffle seed.
It replaces the training champ only when BOTH seats are not worse than an
even split: win rate >= 0.5 and mean life-force differential >= 0. Equality
is not worse. A missing seat or metric does not promote. The pointer is
``runs/linear-overnight/CURRENT.json`` plus ``current.linear.json``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from trainer.improve.linear_selfplay import (
    DARK_PLAYER,
    LIGHT_PLAYER,
    REPO,
    find_classpath,
    games_from_jsonl,
    load_pack,
    make_pack,
    run_live_batch,
    update_from_games,
    write_pack,
)

PROMOTION_RULE = (
    "Promote to the training champ only when the candidate is not worse than the "
    "previous kept AckbarBot linear pack head-to-head on both seats "
    "(candidate Dark vs kept Light, and candidate Light vs kept Dark), "
    "WC96 premiere_anh, no shuffle seed. Not worse means win rate >= 0.5 AND "
    "mean life-force differential >= 0 on each seat. Equality counts as not worse. "
    "A missing seat or metric does not promote. Unfinished maxDecisions/maxMillis "
    "games are skipped. No Beginner opponent. Never writes champs/_promoted."
)

# Head-to-head bar. The kept pack is the opponent, so "not worse than the kept
# pack" is an even split, not the kept pack's older gate numbers.
EVEN_BAR = {
    "asDark": {"winRate": 0.5, "meanLfDiff": 0.0},
    "asLight": {"winRate": 0.5, "meanLfDiff": 0.0},
}

CHAMPS = REPO / "champs"
FORBIDDEN = CHAMPS / "_promoted"


def log(msg: str) -> None:
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{stamp}] {msg}", flush=True)


def run_limits(rounds: int, hours: float) -> tuple[int | None, float | None]:
    """Stop bounds. None means that stop is off (run until the process is killed).

    ``rounds == 0`` does not stop after a round count. ``hours <= 0`` does not
    stop after a wall-clock budget. Per-game maxDecisions / maxMillis are not
    these bounds.
    """
    if rounds < 0:
        raise ValueError("rounds must be >= 0 (0 = until killed)")
    last_round = None if rounds == 0 else rounds
    hour_cap = None if hours <= 0 else hours
    return last_round, hour_cap


def next_round_index(kept_round: int, existing: list[int]) -> int:
    """Resume after the kept pointer and after any round directory already on disk."""
    start = int(kept_round or 0) + 1
    if existing:
        start = max(start, max(existing) + 1)
    return start


def assert_safe_out_dir(out_dir: Path) -> Path:
    resolved = out_dir.resolve()
    champs = CHAMPS.resolve()
    forbidden = FORBIDDEN.resolve()
    if resolved == champs or champs in resolved.parents:
        raise SystemExit(f"refusing to write under champs/: {resolved}")
    if resolved == forbidden or forbidden in resolved.parents:
        raise SystemExit(f"refusing to write under champs/_promoted: {resolved}")
    return resolved


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_pid(pid_path: Path) -> None:
    if pid_path.is_file():
        try:
            old = int(pid_path.read_text(encoding="utf-8").strip())
        except ValueError:
            old = 0
        if old and old != os.getpid() and pid_alive(old):
            log(f"loop already running pid={old}; not starting another")
            raise SystemExit(0)
    pid_path.write_text(str(os.getpid()) + "\n", encoding="utf-8")


def csv_finished_indexes(csv_path: Path) -> set[int]:
    """Game indexes whose CSV row finished with a real winner and no error.

    maxDecisions / maxMillis stops are written as the error cell. Those rows
    are not training games.
    """
    found: set[int] = set()
    if not csv_path.is_file():
        return found
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (row.get("error") or "").strip():
                continue
            winner = row.get("winner") or ""
            if winner not in (DARK_PLAYER, LIGHT_PLAYER):
                continue
            try:
                found.add(int(row["gameIndex"]))
            except (KeyError, TypeError, ValueError):
                continue
    return found


def select_finished(games: list[dict[str, Any]], indexes: set[int]) -> list[dict[str, Any]]:
    """Keep games that both the trace outcome and the CSV call finished."""
    kept: list[dict[str, Any]] = []
    for game in games:
        outcome = game.get("outcome") or {}
        if outcome.get("finished") is False or outcome.get("cancelled") is True:
            continue
        stopper = str(outcome.get("stopper") or "")
        if "maxDecisions" in stopper or "maxMillis" in stopper:
            continue
        winner = str(outcome.get("winner") or "")
        if winner not in (DARK_PLAYER, LIGHT_PLAYER):
            continue
        try:
            idx = int(outcome.get("gameIndex"))
        except (TypeError, ValueError):
            continue
        if idx not in indexes:
            continue
        kept.append(game)
    return kept


def seat_not_worse(cand: dict[str, Any], base: dict[str, Any], min_games: int) -> tuple[bool, str]:
    games = int(cand.get("games") or 0)
    if games < min_games:
        return False, f"finished games {games} < {min_games}"
    reasons: list[str] = []
    for key in ("winRate", "meanLfDiff"):
        c = cand.get(key)
        b = base.get(key)
        if c is None or b is None:
            return False, f"missing {key}"
        if float(c) + 1e-9 < float(b):
            reasons.append(f"{key} {float(c):.6g} < {float(b):.6g}")
    if reasons:
        return False, "; ".join(reasons)
    return True, "not worse on win rate and mean LF differential"


def should_promote(
    cand: dict[str, Any],
    base: dict[str, Any],
    min_games: int,
) -> tuple[bool, dict[str, str]]:
    """True only when both seats are not worse on both metrics."""
    detail: dict[str, str] = {}
    ok = True
    for seat in ("asDark", "asLight"):
        good, why = seat_not_worse(cand.get(seat) or {}, base.get(seat) or {}, min_games)
        detail[seat] = why
        ok = ok and good
    return ok, detail


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_pointer(out_dir: Path, pack_path: Path, payload: dict[str, Any]) -> None:
    """Copy weights to current.linear.json and atomically write CURRENT.json."""
    assert_safe_out_dir(out_dir)
    current_weights = out_dir / "current.linear.json"
    pack = load_pack(pack_path)
    write_pack(current_weights, pack)
    body = dict(payload)
    body["weights"] = "current.linear.json"
    body["promotionRule"] = PROMOTION_RULE
    body["updatedAt"] = datetime.now().astimezone().isoformat(timespec="seconds")
    _write_json(out_dir / "CURRENT.json", body)


def play_batch(
    *,
    classpath: str,
    weights: Path,
    out_dir: Path,
    label: str,
    games: int,
    dark: str,
    light: str,
    max_millis: int,
    max_decisions: int,
    dark_weights: Path | None = None,
    light_weights: Path | None = None,
) -> dict[str, Any]:
    dark_w = dark_weights or weights
    light_w = light_weights or weights
    same = dark_w.resolve() == light_w.resolve()
    wdesc = str(dark_w.resolve()) if same else f"dark={dark_w.resolve()} light={light_w.resolve()}"
    log(
        f"play {label} games={games} dark={dark} light={light} "
        f"decks=wc96 format=premiere_anh maxDecisions={max_decisions} "
        f"maxMillis={max_millis} seed=NONE weights={wdesc}"
    )
    meta = run_live_batch(
        classpath=classpath,
        games=games,
        dark=dark,
        light=light,
        weights=weights,
        csv_path=out_dir / f"{label}.csv",
        traces_path=out_dir / f"{label}.jsonl",
        log_path=out_dir / f"{label}.log",
        max_millis=max_millis,
        max_decisions=max_decisions,
        dark_weights=dark_weights,
        light_weights=light_weights,
    )
    log(f"play {label} exit={meta.get('exit')} ok={meta.get('ok')}")
    return meta


def collect_finished(
    *,
    classpath: str,
    weights: Path,
    out_dir: Path,
    games: int,
    max_millis: int,
    max_decisions: int,
    max_batches: int,
) -> tuple[list[dict[str, Any]], int, int]:
    """Play LINEAR vs LINEAR until ``games`` finished, or ``max_batches``."""
    finished: list[dict[str, Any]] = []
    played = 0
    batches = 0
    while len(finished) < games and batches < max_batches:
        batches += 1
        need = games - len(finished)
        label = f"selfplay-b{batches}"
        meta = play_batch(
            classpath=classpath,
            weights=weights,
            out_dir=out_dir,
            label=label,
            games=need,
            dark="LINEAR",
            light="LINEAR",
            max_millis=max_millis,
            max_decisions=max_decisions,
        )
        if not meta.get("ok"):
            raise RuntimeError(f"self-play jvm failed label={label} exit={meta.get('exit')}")
        played += need
        indexes = csv_finished_indexes(out_dir / f"{label}.csv")
        got = select_finished(games_from_jsonl(out_dir / f"{label}.jsonl"), indexes)
        log(f"{label} finished {len(got)}/{need} (csv finished indexes={sorted(indexes)})")
        finished.extend(got)
    return finished[:games], played, batches


def measure_seat_csv(csv_path: Path, side: str) -> dict[str, Any]:
    """Score one side of a finished gym CSV.

    ``side`` is ``DARK`` or ``LIGHT``. Rows with an error (including
    maxDecisions / maxMillis) or without a real winner are skipped. Life-force
    differential is that side's life force minus the other side's.
    """
    if side not in ("DARK", "LIGHT"):
        raise ValueError(f"side must be DARK or LIGHT, got {side!r}")
    player = DARK_PLAYER if side == "DARK" else LIGHT_PLAYER
    games = 0
    wins = 0
    lf_sum = 0.0
    lf_n = 0
    if csv_path.is_file():
        with csv_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if (row.get("error") or "").strip():
                    continue
                if (row.get("winner") or "") not in (DARK_PLAYER, LIGHT_PLAYER):
                    continue
                games += 1
                wins += int(row.get("winner") == player)
                try:
                    dlf = int(row["darkLifeForce"])
                    llf = int(row["lightLifeForce"])
                except (KeyError, TypeError, ValueError):
                    continue
                if dlf < 0 or llf < 0:
                    continue
                lf_sum += float(dlf - llf) if side == "DARK" else float(llf - dlf)
                lf_n += 1
    return {
        "games": games,
        "wins": wins,
        "winRate": (wins / games) if games else None,
        "meanLfDiff": (lf_sum / lf_n) if lf_n else None,
    }


def gate_vs_kept(
    *,
    classpath: str,
    cand_weights: Path,
    kept_weights: Path,
    gate_dir: Path,
    per_seat: int,
    max_millis: int,
    max_decisions: int,
) -> dict[str, Any]:
    """Candidate LINEAR pack vs the previous kept LINEAR pack, both seats.

    No Beginner. Each seat is a separate batch so the two packs stay on
    ``--dark-weights`` / ``--light-weights``.
    """
    gate_dir.mkdir(parents=True, exist_ok=True)
    # (label, candidate side, dark weights, light weights)
    seats = (
        ("as-dark", "DARK", cand_weights, kept_weights),
        ("as-light", "LIGHT", kept_weights, cand_weights),
    )
    measured: dict[str, dict[str, Any]] = {}
    for label, side, dark_w, light_w in seats:
        meta = play_batch(
            classpath=classpath,
            weights=cand_weights,
            out_dir=gate_dir,
            label=label,
            games=per_seat,
            dark="LINEAR",
            light="LINEAR",
            max_millis=max_millis,
            max_decisions=max_decisions,
            dark_weights=dark_w,
            light_weights=light_w,
        )
        if not meta.get("ok"):
            return {"ok": False, "label": label, "exit": meta.get("exit"), "opponent": "previous-kept-linear"}
        measured[side] = measure_seat_csv(gate_dir / f"{label}.csv", side)
    as_dark = measured["DARK"]
    as_light = measured["LIGHT"]
    games = as_dark["games"] + as_light["games"]
    wins = as_dark["wins"] + as_light["wins"]
    lf_parts = []
    for seat in (as_dark, as_light):
        if seat["meanLfDiff"] is not None and seat["games"]:
            lf_parts.append((seat["meanLfDiff"], seat["games"]))
    mean_lf = None
    if lf_parts:
        mean_lf = sum(m * n for m, n in lf_parts) / sum(n for _, n in lf_parts)
    meas = {
        "ok": True,
        "opponent": "previous-kept-linear",
        "candidateWeights": cand_weights.name,
        "keptWeights": kept_weights.name,
        "candidateGames": games,
        "candidateWins": wins,
        "candidateWinRate": (wins / games) if games else None,
        "meanCandidateLfDiff": mean_lf,
        "asDark": as_dark,
        "asLight": as_light,
    }
    log(
        "gate LINEAR vs previous kept LINEAR "
        f"pooled WR={meas.get('candidateWinRate')} "
        f"meanLF={meas.get('meanCandidateLfDiff')} "
        f"dark={as_dark} light={as_light}"
    )
    return meas


def run_baseline(args: argparse.Namespace, classpath: str, out_dir: Path) -> dict[str, Any]:
    """Write an initial pack. No Beginner games. The first gate is a later candidate vs this file."""
    del classpath  # baseline does not play
    pack = make_pack(args.init)
    baseline_weights = out_dir / "baseline.linear.json"
    write_pack(baseline_weights, pack)
    log(f"baseline init={args.init} -> {baseline_weights.name} (no Beginner gate)")
    pointer = {
        "schema": "linear-overnight-current.v1",
        "role": "baseline",
        "round": 0,
        "init": args.init,
        "promoted": False,
        "metrics": None,
        "comparedTo": None,
        "note": (
            "Initial linear pack. No Beginner gate. A candidate promotes only when "
            "it is not worse than this pack head-to-head on both seats."
        ),
    }
    write_pointer(out_dir, baseline_weights, pointer)
    log("wrote baseline CURRENT.json")
    return json.loads((out_dir / "CURRENT.json").read_text(encoding="utf-8"))


def kept_pack_status(out_dir: Path) -> tuple[dict[str, Any] | None, str]:
    """Load the previous kept pack. Weights must be a usable linear.v1 file."""
    current_path = out_dir / "CURRENT.json"
    weights_path = out_dir / "current.linear.json"
    if not weights_path.is_file():
        return None, "no current.linear.json"
    try:
        load_pack(weights_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return None, f"unusable current.linear.json: {exc}"
    if not current_path.is_file():
        return {"round": 0, "role": "baseline", "metrics": None}, "weights only"
    try:
        kept = json.loads(current_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"round": 0, "role": "baseline", "metrics": None}, f"weights ok, CURRENT.json unusable ({exc})"
    if not isinstance(kept, dict):
        return {"round": 0, "role": "baseline", "metrics": None}, "weights ok, CURRENT.json not an object"
    return kept, "ok"


def one_round(
    args: argparse.Namespace,
    classpath: str,
    out_dir: Path,
    rnd: int,
    kept: dict[str, Any],
) -> dict[str, Any]:
    round_dir = out_dir / "rounds" / f"r{rnd:03d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    champ_path = out_dir / "current.linear.json"
    # Snapshot only. The JVM and the update both start from the kept pack
    # itself (current.linear.json), not from a zeros reinit and not from a
    # file whose name is init.linear.json.
    init_path = round_dir / "init.linear.json"
    write_pack(init_path, load_pack(champ_path))
    log(
        f"round {rnd}: self-play loads kept pack {champ_path.resolve()} "
        f"(not a zeros reinit; {init_path.name} is only a snapshot)"
    )
    min_games = max(1, args.gate_per_seat // 2)
    report: dict[str, Any] = {
        "round": rnd,
        "promoted": False,
        "promotionRule": PROMOTION_RULE,
    }
    finished, played, batches = collect_finished(
        classpath=classpath,
        weights=champ_path,
        out_dir=round_dir,
        games=args.games,
        max_millis=args.max_millis,
        max_decisions=args.max_decisions,
        max_batches=args.max_batches,
    )
    report["selfPlayPlayed"] = played
    report["selfPlayBatches"] = batches
    report["selfPlayFinished"] = len(finished)
    if not finished:
        report["skipped"] = "no finished LINEAR vs LINEAR games (maxDecisions/maxMillis skipped)"
        log(f"round {rnd}: no finished self-play games; weights unchanged")
        _write_json(round_dir / "report.json", report)
        return report

    pack = load_pack(champ_path)
    stats = update_from_games(pack, finished, args.lr)
    pack["trainer"] = {
        "name": "linear_overnight",
        "updateRule": stats["updateRule"],
        "source": "live",
        "opponent": "LINEAR",
        "bothSeats": True,
        "round": rnd,
        "finishedGames": len(finished),
        "promoted": False,
        "distilledFrom": None,
    }
    cand_path = round_dir / "candidate.linear.json"
    write_pack(cand_path, pack)
    report["update"] = stats
    log(
        f"round {rnd}: update {stats['updateRule']} "
        f"finished={len(finished)} decisionsUsed={stats['decisionsUsed']} "
        f"skipped={stats['decisionsSkipped']}"
    )
    kept_weights = out_dir / "current.linear.json"
    meas = gate_vs_kept(
        classpath=classpath,
        cand_weights=cand_path,
        kept_weights=kept_weights,
        gate_dir=round_dir / "gate",
        per_seat=args.gate_per_seat,
        max_millis=args.max_millis,
        max_decisions=args.max_decisions,
    )
    report["gate"] = meas
    report["gateOpponent"] = str(kept_weights)
    if not meas.get("ok"):
        report["skipped"] = "gate jvm failed; kept previous champ"
        log(f"round {rnd}: gate failed; kept previous champ")
        _write_json(round_dir / "report.json", report)
        return report
    ok, detail = should_promote(meas, EVEN_BAR, min_games)
    report["promoteDetail"] = detail
    report["comparedToRound"] = kept.get("round")
    if not ok:
        report["skipped"] = "gate not better-or-equal on both seats"
        log(f"round {rnd}: KEEP previous champ ({detail})")
        _write_json(round_dir / "report.json", report)
        return report
    pack["trainer"]["promoted"] = True
    write_pack(cand_path, pack)
    pointer = {
        "schema": "linear-overnight-current.v1",
        "role": "training-champ",
        "round": rnd,
        "init": args.init,
        "promoted": True,
        "metrics": meas,
        "comparedTo": {"round": kept.get("round"), "role": kept.get("role")},
        "updateRule": stats["updateRule"],
        "selfPlayFinished": len(finished),
        "note": "Training champ only. Not copied to champs/_promoted.",
    }
    write_pointer(out_dir, cand_path, pointer)
    report["promoted"] = True
    log(f"round {rnd}: PROMOTED training champ ({detail})")
    _write_json(round_dir / "report.json", report)
    return report


def run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Unattended LINEAR vs LINEAR overnight loop. Does not promote into champs/.")
    parser.add_argument("--rounds", type=int, default=0, help="0 = run until killed (no round cap)")
    parser.add_argument("--hours", type=float, default=0.0, help="<=0 = no hour cap")
    parser.add_argument("--games", type=int, default=8, help="finished LINEAR vs LINEAR games per round")
    parser.add_argument("--max-batches", type=int, default=2, help="self-play batches per round if games do not finish")
    parser.add_argument("--gate-per-seat", type=int, default=4)
    parser.add_argument("--max-decisions", type=int, default=8000)
    parser.add_argument("--max-millis", type=int, default=180_000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--init", choices=("zeros", "small-random", "tiebreak"), default="zeros")
    parser.add_argument("--classpath-file", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=REPO / "runs" / "linear-overnight")
    parser.add_argument("--fresh", action="store_true", help="ignore an existing CURRENT.json and start at the baseline")
    args = parser.parse_args(argv)

    if args.rounds < 0 or args.games < 1 or args.gate_per_seat < 1:
        parser.error("rounds must be >= 0 (0 = until killed); games and gate-per-seat must be >= 1")
    if args.max_decisions < 1 or args.max_millis < 1:
        parser.error("caps must be >= 1")

    out_dir = assert_safe_out_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    acquire_pid(out_dir / "loop.pid")

    classpath = find_classpath(args.classpath_file)
    if not classpath:
        log("no gym classpath")
        raise SystemExit(1)

    last_round, hour_cap = run_limits(args.rounds, args.hours)
    deadline = None if hour_cap is None else time.time() + hour_cap * 3600.0
    log(
        f"start rounds={'until-killed' if last_round is None else last_round} "
        f"hours={'none' if deadline is None else hour_cap} games={args.games} "
        f"gatePerSeat={args.gate_per_seat} maxDecisions={args.max_decisions} "
        f"init={args.init} pid={os.getpid()} out={out_dir}"
    )
    log("no shuffle seed; WC96; premiere_anh; self-play LINEAR vs LINEAR same weights file")
    log("gate opponent: previous kept AckbarBot linear pack, both seats; no Beginner")
    log(PROMOTION_RULE)

    current_path = out_dir / "CURRENT.json"
    if args.fresh:
        kept = run_baseline(args, classpath, out_dir)
        start_round = 1
    else:
        kept, status = kept_pack_status(out_dir)
        if kept is None:
            log(f"no usable CURRENT pack ({status}); starting from init={args.init}")
            kept = run_baseline(args, classpath, out_dir)
            start_round = 1
        else:
            existing: list[int] = []
            rounds_root = out_dir / "rounds"
            if rounds_root.is_dir():
                for child in rounds_root.iterdir():
                    name = child.name
                    if child.is_dir() and name.startswith("r") and name[1:].isdigit():
                        existing.append(int(name[1:]))
            start_round = next_round_index(int(kept.get("round") or 0), existing)
            log(
                f"resume from round {kept.get('round')} role={kept.get('role')} "
                f"next={start_round} status={status} weights=current.linear.json "
                f"(stored metrics are not the gate bar)"
            )

    consecutive_failures = 0
    rnd = start_round
    while True:
        if last_round is not None and rnd > last_round:
            log(f"stop: completed {last_round} rounds")
            break
        if deadline is not None and time.time() >= deadline:
            log(f"stop: hour cap reached before round {rnd}")
            break
        try:
            report = one_round(args, classpath, out_dir, rnd, kept)
        except Exception as exc:
            consecutive_failures += 1
            log(f"round {rnd} FAILED {type(exc).__name__}: {exc}")
            traceback.print_exc()
            if consecutive_failures >= 3:
                log("stop: 3 consecutive round failures")
                raise SystemExit(1)
            rnd += 1
            continue
        consecutive_failures = 0
        if report.get("promoted"):
            kept = json.loads(current_path.read_text(encoding="utf-8"))
        rnd += 1
    log("loop exit")


def main() -> None:
    try:
        run()
    except KeyboardInterrupt:
        log("interrupted")
        raise SystemExit(130)


if __name__ == "__main__":
    main()
