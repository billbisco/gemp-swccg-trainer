"""Minimal heuristic.v1 weight mutator + Maven gate eval.

Collection ≠ learning. A candidate is only an "upgraded champ" if the gate
reports a higher win rate vs builtin BEGINNER on WC96 (as Dark and/or Light).
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Baseline BeginnerAi keyword weights (must match BeginnerAi.java / export stub).
BEGINNER_WEIGHTS: dict[str, Any] = {
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
        "pilot", "weapon", "character", "starship", "vehicle", "droid", "alien",
        "jedi", "sith", "effect", "interrupt", "location", "site", "system",
    ],
}

_WEIGHT_KEYS = ("actionWeights", "actionPenalties", "choiceWeights", "choicePenalties")


def mutate_weights(
    base: dict[str, Any] | None = None,
    *,
    rng: random.Random | None = None,
    sigma: float = 12.0,
    frac: float = 0.35,
    nudge_toward_advanced: bool = True,
) -> dict[str, Any]:
    """Gaussian-perturb a random subset of keyword weights.

    When nudge_toward_advanced is True, also gently mix a few action weights
    toward known AdvancedAi values (force drain / battle / deploy) so the
    search has a directional prior — still a heuristic search, not imitation.
    """
    rng = rng or random.Random()
    out = copy.deepcopy(base or BEGINNER_WEIGHTS)
    advanced_nudge = {
        "force drain": 160,
        "initiate battle": 150,
        "battle": 100,
        "deploy": 90,
        "activate": 70,
        "move": 50,
    }
    for key in _WEIGHT_KEYS:
        rows = out[key]
        n = max(1, int(math.ceil(len(rows) * frac)))
        idxs = rng.sample(range(len(rows)), n)
        for i in idxs:
            w = float(rows[i]["w"])
            w = w + rng.gauss(0.0, sigma)
            if nudge_toward_advanced and key == "actionWeights":
                k = rows[i]["k"]
                if k in advanced_nudge and rng.random() < 0.4:
                    # Blend 15–40% toward Advanced prior
                    blend = rng.uniform(0.15, 0.40)
                    w = (1 - blend) * w + blend * advanced_nudge[k]
            # Keep penalties negative-ish and rewards positive-ish
            if "Penalties" in key and w > -5:
                w = -abs(w) - 5
            if "Penalties" not in key and w < 0:
                w = abs(w)
            rows[i]["w"] = int(round(w))
    return out


def score_from_traces_proxy(traces_path: Path, weights: dict[str, Any]) -> float:
    """Cheap offline proxy: average keyword-score of chosen action text on winning side.

    Not a substitute for live gate games — used only to rank random mutants before
    spending Maven eval budget.
    """
    if not traces_path.is_file():
        return 0.0

    def score_text(text: str) -> float:
        t = (text or "").lower()
        s = 0.0
        for key in _WEIGHT_KEYS:
            for row in weights[key]:
                if row["k"].lower() in t:
                    s += float(row["w"])
        return s

    total = 0.0
    n = 0
    # Sample up to 8k lines for speed
    with traces_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 8000:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            parts = [str(obj.get("decisionText") or "")]
            chosen = obj.get("chosen")
            opts = (obj.get("options") or {}).get("items") or []
            if isinstance(chosen, str) and chosen.isdigit() and opts:
                idx = int(chosen)
                if 0 <= idx < len(opts):
                    parts.append(str(opts[idx].get("text") or ""))
            elif isinstance(chosen, str):
                parts.append(chosen)
            for item in opts[:6]:
                parts.append(str(item.get("text") or ""))
            blob = " ".join(parts)
            sc = score_text(blob)
            total += sc
            n += 1
    return total / max(1, n)


def random_search(
    *,
    traces_path: Path | None,
    n_candidates: int = 12,
    seed: int = 42,
    sigma: float = 12.0,
) -> list[dict[str, Any]]:
    """Generate mutants and rank by optional trace proxy (highest first)."""
    rng = random.Random(seed)
    ranked: list[dict[str, Any]] = []
    for i in range(n_candidates):
        w = mutate_weights(BEGINNER_WEIGHTS, rng=rng, sigma=sigma)
        proxy = score_from_traces_proxy(traces_path, w) if traces_path else 0.0
        ranked.append({"id": f"mut-{i:02d}", "weights": w, "proxyScore": proxy})
    ranked.sort(key=lambda r: r["proxyScore"], reverse=True)
    return ranked


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def export_candidate_champ(
    champs_dir: Path,
    weights: dict[str, Any],
    *,
    champ_id: str,
    display_name: str,
    notes: str,
    parent_id: str = "builtin-BEGINNER",
    gate: dict[str, Any] | None = None,
    stub: bool = False,
) -> dict[str, Any]:
    champs_dir = Path(champs_dir)
    champs_dir.mkdir(parents=True, exist_ok=True)
    pack_dir = champs_dir / champ_id
    pack_dir.mkdir(parents=True, exist_ok=True)

    weights_path = pack_dir / "weights.json"
    weights_path.write_text(json.dumps(weights, indent=2) + "\n")
    weights_hash = _sha256_file(weights_path)

    (pack_dir / "NOTES.md").write_text(notes if notes.endswith("\n") else notes + "\n")
    if gate is not None:
        (pack_dir / "gate.json").write_text(json.dumps(gate, indent=2) + "\n")

    manifest = {
        "manifestVersion": 1,
        "id": champ_id,
        "displayName": display_name,
        "policyKind": "heuristic.v1",
        "gymApiVersion": "0.0.0-day1",
        "createdAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "parentId": parent_id,
        "elo": gate.get("eloEst") if gate else None,
        "ladderId": None,
        "deckHints": [
            "P-ANH 1996 World Champion (Dark)",
            "P-ANH 1996 World Champion (Light)",
        ],
        "files": {"weights": "weights.json"},
        "hash": {"alg": "sha256", "value": weights_hash},
        "signature": None,
        "stub": stub,
        "note": "heuristic weight-search candidate" if not stub else "stub",
        "gatePassed": bool(gate and gate.get("passed")),
    }
    (pack_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    import zipfile

    zip_path = champs_dir / f"{champ_id}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in ("manifest.json", "weights.json", "NOTES.md"):
            zf.write(pack_dir / name, arcname=f"{champ_id}/{name}")
        if (pack_dir / "gate.json").exists():
            zf.write(pack_dir / "gate.json", arcname=f"{champ_id}/gate.json")

    return {
        "id": champ_id,
        "dir": str(pack_dir.resolve()),
        "zip": str(zip_path.resolve()),
        "weightsSha256": weights_hash,
        "zipSha256": _sha256_file(zip_path),
        "manifest": manifest,
    }


def evaluate_via_maven(
    *,
    gemp_src: Path,
    run_dir: Path,
    dark_weights: Path | None,
    light_weights: Path | None,
    games: int,
    label: str,
    max_millis: int = 300_000,
) -> dict[str, Any]:
    """Run HeadlessBotVsBotBatchTest with optional weight overlays; parse CSV."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / f"{label}.csv"
    log_path = run_dir / f"{label}.maven.log"
    cmd = [
        "mvn",
        "-pl",
        "gemp-swccg-server",
        "-am",
        "-DfailIfNoTests=false",
        "-Dtest=HeadlessBotVsBotBatchTest#batchSelfPlay_writesCsv",
        f"-Dheadless.games={games}",
        "-Dheadless.dark=BEGINNER",
        "-Dheadless.light=BEGINNER",
        "-Dheadless.decks=wc96",
        "-Dheadless.format=premiere_anh",
        "-Dheadless.traces=false",
        "-Dheadless.replay=false",
        f"-Dheadless.csv={csv_path}",
        f"-Dheadless.maxMillis={max_millis}",
        "-Dheadless.verbose=false",
        "test",
    ]
    if dark_weights is not None:
        cmd.append(f"-Dheadless.dark.weights={dark_weights}")
    if light_weights is not None:
        cmd.append(f"-Dheadless.light.weights={light_weights}")

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(gemp_src),
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0
    log_path.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr)
    stats = summarize_csv(csv_path)
    stats["label"] = label
    stats["mavenExit"] = proc.returncode
    stats["wallSec"] = round(elapsed, 2)
    stats["csv"] = str(csv_path)
    stats["log"] = str(log_path)
    return stats


def summarize_csv(csv_path: Path) -> dict[str, Any]:
    dark_wins = light_wins = errors = 0
    decisions: list[int] = []
    durations: list[float] = []
    rows = 0
    if not csv_path.is_file():
        return {
            "games": 0,
            "darkWins": 0,
            "lightWins": 0,
            "errors": 0,
            "darkWinRate": None,
            "lightWinRate": None,
            "avgDecisions": None,
            "avgDurationMs": None,
        }
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows += 1
            err = (row.get("error") or "").strip()
            if err:
                errors += 1
                continue
            winner = row.get("winner") or ""
            if winner == "~OzzelBot":
                dark_wins += 1
            elif winner == "~AckbarBot":
                light_wins += 1
            try:
                decisions.append(int(row["darkDecisions"]) + int(row["lightDecisions"]))
            except (KeyError, ValueError):
                pass
            try:
                durations.append(float(row["elapsedMs"]))
            except (KeyError, ValueError):
                pass
    finished = dark_wins + light_wins
    return {
        "games": rows,
        "finished": finished,
        "darkWins": dark_wins,
        "lightWins": light_wins,
        "errors": errors,
        "darkWinRate": (dark_wins / finished) if finished else None,
        "lightWinRate": (light_wins / finished) if finished else None,
        "avgDecisions": (sum(decisions) / len(decisions)) if decisions else None,
        "avgDurationMs": (sum(durations) / len(durations)) if durations else None,
    }


def elo_from_score(score: float, n: int, prior: float = 1500.0) -> float:
    """Crude Elo-ish from win rate vs equal baseline (expected 0.5)."""
    if n <= 0:
        return prior
    # logistic: score = 1/(1+10^((opp-elo)/400)); invert vs opp=prior
    score = min(max(score, 1e-3), 1 - 1e-3)
    return prior + 400.0 * math.log10(score / (1 - score))


def gate_candidate(
    *,
    gemp_src: Path,
    work_dir: Path,
    candidate_weights: Path,
    games_per_side: int = 8,
    baseline_dark_rate: float = 0.5,
    baseline_light_rate: float = 0.5,
) -> dict[str, Any]:
    """Gate: candidate as Dark vs BEGINNER Light, then BEGINNER Dark vs candidate Light.

    Pass if candidate improves on the collect-batch side baseline rates (decks are
    asymmetric — raw 0.5 is the wrong bar for WC96 Dark). Fail closed otherwise.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    as_dark = evaluate_via_maven(
        gemp_src=gemp_src,
        run_dir=work_dir,
        dark_weights=candidate_weights,
        light_weights=None,
        games=games_per_side,
        label="gate-cand-dark",
    )
    as_light = evaluate_via_maven(
        gemp_src=gemp_src,
        run_dir=work_dir,
        dark_weights=None,
        light_weights=candidate_weights,
        games=games_per_side,
        label="gate-cand-light",
    )

    # Candidate wins: as Dark → darkWins; as Light → lightWins
    cand_wins = (as_dark.get("darkWins") or 0) + (as_light.get("lightWins") or 0)
    total = (as_dark.get("finished") or 0) + (as_light.get("finished") or 0)
    rate = (cand_wins / total) if total else 0.0
    dark_side_rate = as_dark.get("darkWinRate")
    light_side_rate = as_light.get("lightWinRate")

    expected = games_per_side * baseline_dark_rate + games_per_side * baseline_light_rate
    expected_rate = (expected / total) if total else 0.5
    dark_delta = (dark_side_rate - baseline_dark_rate) if dark_side_rate is not None else None
    light_delta = (light_side_rate - baseline_light_rate) if light_side_rate is not None else None
    passed = False
    reason = ""
    if total < max(4, games_per_side):
        reason = f"insufficient finished games ({total})"
    else:
        side_ok = (
            (dark_delta is not None and dark_delta >= 0.05)
            or (light_delta is not None and light_delta >= 0.05)
        )
        combined_ok = rate >= expected_rate + 0.05
        if side_ok and combined_ok:
            passed = True
            reason = (
                f"improved vs side baselines "
                f"(dark {dark_side_rate} vs {baseline_dark_rate}, "
                f"light {light_side_rate} vs {baseline_light_rate}; "
                f"combined {rate:.3f} vs expected {expected_rate:.3f})"
            )
        else:
            reason = (
                f"failed gate: dark {dark_side_rate} (base {baseline_dark_rate}, d={dark_delta}), "
                f"light {light_side_rate} (base {baseline_light_rate}, d={light_delta}), "
                f"combined {rate:.3f} vs expected {expected_rate:.3f}+0.05"
            )

    return {
        "passed": passed,
        "reason": reason,
        "candidateWins": cand_wins,
        "totalFinished": total,
        "combinedWinRate": rate,
        "expectedWinRate": expected_rate if total else None,
        "baselineDarkRate": baseline_dark_rate,
        "baselineLightRate": baseline_light_rate,
        "darkDelta": dark_delta,
        "lightDelta": light_delta,
        "eloEst": elo_from_score(rate, total) if total else None,
        "asDark": as_dark,
        "asLight": as_light,
        "gamesPerSide": games_per_side,
        "stopReason": "gate_eval_complete",
    }
