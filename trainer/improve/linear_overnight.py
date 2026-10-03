"""Overnight LINEAR vs LINEAR self-play loop for AckbarBot.

Uses ``trainer.improve.linear_selfplay`` for the pack, the gym batch, and the
update. That module already plays LINEAR vs LINEAR (same ``--linear-weights``
file on both seats). This loop is the unattended part: many rounds, random
shuffles, a Beginner gate, and a training-champ pointer.

What it does not do
-------------------
* Does not distill YodaBot / AdvancedAi.
* Does not run the keyword mutate loop (``trainer.improve.search`` / ``loop``).
* Does not write under ``champs/`` and never touches
  ``champs/_promoted/heuristic-v1-wc96-r08-blend-adv25``.
* Does not pass a gym shuffle seed. Strength numbers are not from a fixed seed.

Promotion
---------
After each self-play update, the candidate plays BEGINNER on both seats.
It replaces the training champ only when BOTH seats are not worse than the
previous kept pack (or the stored baseline) on win rate AND on mean
life-force differential. Equality is not worse. A missing seat or metric
does not promote. The pointer is ``runs/linear-overnight/CURRENT.json``
plus ``current.linear.json``.
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
    _concat_csv,
    candidate_measurement,
    find_classpath,
    games_from_jsonl,
    load_pack,
    make_pack,
    run_live_batch,
    update_from_games,
    write_pack,
)

PROMOTION_RULE = (
    "Promote to the training champ only when both seats (Dark and Light vs BEGINNER) "
    "are not worse than the previous kept pack on win rate AND on mean life-force "
    "differential. Equality counts as not worse. A missing seat or metric does not "
    "promote. Unfinished maxDecisions/maxMillis games are skipped. "
    "Never writes champs/_promoted."
)

CHAMPS = REPO / "champs"
FORBIDDEN = CHAMPS / "_promoted"


def log(msg: str) -> None:
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{stamp}] {msg}", flush=True)


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
) -> dict[str, Any]:
    log(
        f"play {label} games={games} dark={dark} light={light} "
        f"decks=wc96 format=premiere_anh maxDecisions={max_decisions} "
        f"maxMillis={max_millis} seed=NONE weights={weights.name}"
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


def gate_vs_beginner(
    *,
    classpath: str,
    weights: Path,
    gate_dir: Path,
    per_seat: int,
    max_millis: int,
    max_decisions: int,
) -> dict[str, Any]:
    gate_dir.mkdir(parents=True, exist_ok=True)
    csvs: list[Path] = []
    for dark, light, label in (
        ("LINEAR", "BEGINNER", "as-dark"),
        ("BEGINNER", "LINEAR", "as-light"),
    ):
        meta = play_batch(
            classpath=classpath,
            weights=weights,
            out_dir=gate_dir,
            label=label,
            games=per_seat,
            dark=dark,
            light=light,
            max_millis=max_millis,
            max_decisions=max_decisions,
        )
        if not meta.get("ok"):
            return {"ok": False, "label": label, "exit": meta.get("exit")}
        csvs.append(gate_dir / f"{label}.csv")
    _concat_csv(csvs, gate_dir / "games.csv")
    meas = candidate_measurement(gate_dir / "games.csv")
    meas["ok"] = True
    log(
        "gate LINEAR vs BEGINNER "
        f"pooled WR={meas.get('candidateWinRate')} "
        f"meanLF={meas.get('meanCandidateLfDiff')} "
        f"dark={meas.get('asDark')} light={meas.get('asLight')}"
    )
    return meas


def run_baseline(args: argparse.Namespace, classpath: str, out_dir: Path) -> dict[str, Any]:
    pack = make_pack(args.init)
    baseline_weights = out_dir / "baseline.linear.json"
    write_pack(baseline_weights, pack)
    log(f"baseline init={args.init} -> {baseline_weights.name}")
    meas = gate_vs_beginner(
        classpath=classpath,
        weights=baseline_weights,
        gate_dir=out_dir / "baseline-gate",
        per_seat=args.gate_per_seat,
        max_millis=args.max_millis,
        max_decisions=args.max_decisions,
    )
    if not meas.get("ok"):
        raise RuntimeError(f"baseline gate failed: {meas}")
    pointer = {
        "schema": "linear-overnight-current.v1",
        "role": "baseline",
        "round": 0,
        "init": args.init,
        "promoted": False,
        "metrics": meas,
        "comparedTo": None,
        "note": "Stored baseline (initial pack vs BEGINNER, both seats). Training champ until a candidate is not worse on both seats.",
    }
    write_pointer(out_dir, baseline_weights, pointer)
    log("wrote baseline CURRENT.json")
    return json.loads((out_dir / "CURRENT.json").read_text(encoding="utf-8"))


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
    init_path = round_dir / "init.linear.json"
    write_pack(init_path, load_pack(champ_path))
    min_games = max(1, args.gate_per_seat // 2)
    report: dict[str, Any] = {
        "round": rnd,
        "promoted": False,
        "promotionRule": PROMOTION_RULE,
    }
    finished, played, batches = collect_finished(
        classpath=classpath,
        weights=init_path,
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

    pack = load_pack(init_path)
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
    meas = gate_vs_beginner(
        classpath=classpath,
        weights=cand_path,
        gate_dir=round_dir / "gate",
        per_seat=args.gate_per_seat,
        max_millis=args.max_millis,
        max_decisions=args.max_decisions,
    )
    report["gate"] = meas
    if not meas.get("ok"):
        report["skipped"] = "gate jvm failed; kept previous champ"
        log(f"round {rnd}: gate failed; kept previous champ")
        _write_json(round_dir / "report.json", report)
        return report
    ok, detail = should_promote(meas, kept.get("metrics") or {}, min_games)
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
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--hours", type=float, default=8.0)
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

    if args.rounds < 1 or args.games < 1 or args.gate_per_seat < 1:
        parser.error("rounds, games, and gate-per-seat must be >= 1")
    if args.max_decisions < 1 or args.max_millis < 1:
        parser.error("caps must be >= 1")

    out_dir = assert_safe_out_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    acquire_pid(out_dir / "loop.pid")

    classpath = find_classpath(args.classpath_file)
    if not classpath:
        log("no gym classpath")
        raise SystemExit(1)

    deadline = time.time() + args.hours * 3600.0
    log(
        f"start rounds={args.rounds} hours={args.hours} games={args.games} "
        f"gatePerSeat={args.gate_per_seat} maxDecisions={args.max_decisions} "
        f"init={args.init} pid={os.getpid()} out={out_dir}"
    )
    log("no shuffle seed; WC96; premiere_anh; LINEAR vs LINEAR same weights")
    log(PROMOTION_RULE)

    current_path = out_dir / "CURRENT.json"
    weights_path = out_dir / "current.linear.json"
    if args.fresh or not (current_path.is_file() and weights_path.is_file()):
        kept = run_baseline(args, classpath, out_dir)
        start_round = 1
    else:
        kept = json.loads(current_path.read_text(encoding="utf-8"))
        start_round = int(kept.get("round") or 0) + 1
        log(f"resume from round {kept.get('round')} role={kept.get('role')} next={start_round}")

    consecutive_failures = 0
    for rnd in range(start_round, args.rounds + 1):
        if time.time() >= deadline:
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
            continue
        consecutive_failures = 0
        if report.get("promoted"):
            kept = json.loads(current_path.read_text(encoding="utf-8"))
    else:
        log(f"stop: completed {args.rounds} rounds")
    if time.time() >= deadline:
        log("stop: hour cap")
    log("loop exit")


def main() -> None:
    try:
        run()
    except KeyboardInterrupt:
        log("interrupted")
        raise SystemExit(130)


if __name__ == "__main__":
    main()
