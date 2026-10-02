"""WC96 continuous collect → improve → gate → promote loop.

Throughput bias: CSV primary; traces optional; xml.gz replay OFF by default.
Honest gates only. Collection ≠ learning.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from trainer.improve.search import (
    BEGINNER_WEIGHTS,
    evaluate_via_maven,
    export_candidate_champ,
    gate_candidate,
    mutate_weights,
    score_from_traces_proxy,
    summarize_csv,
    elo_from_score,
)

ROOT = Path("/workspace/gemp-swccg-trainer")
GEMP_SRC = Path("/workspace/swccg-gemp/src")
CHAMPS = ROOT / "champs"
PROMOTED = CHAMPS / "_promoted"
DOCS = ROOT / "docs"
RUNS = ROOT / "runs" / "wc96-loop"

# Peter Soetens note (future benchmark only — do not merge):
# ~90% vs Advanced on ~380 random legal decks; browser-side model; ~800k games/day.
FUTURE_BENCHMARK = "Peter Soetens browser model ~90% vs Advanced (~380 decks); not integrated yet."


def ts() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")


def load_weights(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def current_champ_id() -> str:
    pointer = PROMOTED / "CURRENT"
    if pointer.is_file():
        return pointer.read_text().strip()
    # fallback: only promoted pack
    kids = [p.name for p in PROMOTED.iterdir() if p.is_dir()]
    if kids:
        return sorted(kids)[-1]
    return "heuristic-v1-wc96-batch50-cand1"


def champ_weights_path(champ_id: str) -> Path:
    for base in (PROMOTED / champ_id, CHAMPS / champ_id):
        w = base / "weights.json"
        if w.is_file():
            return w
    raise FileNotFoundError(champ_id)


def set_current(champ_id: str) -> None:
    PROMOTED.mkdir(parents=True, exist_ok=True)
    (PROMOTED / "CURRENT").write_text(champ_id + "\n")


def blend_advanced(base: dict[str, Any], frac: float, rng: random.Random) -> dict[str, Any]:
    out = copy.deepcopy(base)
    advanced = {
        "force drain": 160,
        "initiate battle": 150,
        "battle": 100,
        "deploy": 90,
        "activate": 70,
        "move": 50,
        "weapon": 55,
        "fire": 50,
    }
    for row in out["actionWeights"]:
        if row["k"] in advanced:
            b = frac * rng.uniform(0.8, 1.2)
            b = min(max(b, 0.05), 0.9)
            row["w"] = int(round((1 - b) * row["w"] + b * advanced[row["k"]]))
    return out


def search_from_champ(
    base_weights: dict[str, Any],
    *,
    traces_path: Path | None,
    n_candidates: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    ranked: list[dict[str, Any]] = []
    # Always include directed blends
    for i, frac in enumerate((0.25, 0.40, 0.55, 0.70)):
        w = blend_advanced(base_weights, frac, rng)
        proxy = score_from_traces_proxy(traces_path, w) if traces_path else 0.0
        ranked.append({"id": f"blend-adv{int(frac*100)}", "weights": w, "proxyScore": proxy})
    for i in range(max(0, n_candidates - 4)):
        w = mutate_weights(base_weights, rng=rng, sigma=14.0, frac=0.40)
        proxy = score_from_traces_proxy(traces_path, w) if traces_path else 0.0
        ranked.append({"id": f"mut-{i:02d}", "weights": w, "proxyScore": proxy})
    ranked.sort(key=lambda r: r["proxyScore"], reverse=True)
    return ranked


def collect_selfplay(
    *,
    run_dir: Path,
    dark_w: Path | None,
    light_w: Path | None,
    games: int,
    label: str,
    traces: bool = False,
) -> dict[str, Any]:
    """CSV collect; replay always off. Traces off by default for throughput."""
    run_dir.mkdir(parents=True, exist_ok=True)
    # evaluate_via_maven already sets traces=false replay=false
    stats = evaluate_via_maven(
        gemp_src=GEMP_SRC,
        run_dir=run_dir,
        dark_weights=dark_w,
        light_weights=light_w,
        games=games,
        label=label,
    )
    # optional: if traces wanted, caller can re-run; keep default off
    (run_dir / f"{label}.summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def gate_vs_opponent(
    *,
    work_dir: Path,
    candidate_weights: Path,
    opponent_weights: Path | None,
    games_per_side: int,
    label_prefix: str,
    baseline_dark_rate: float,
    baseline_light_rate: float,
) -> dict[str, Any]:
    """Candidate as Dark vs opp Light; opp Dark vs candidate Light."""
    work_dir.mkdir(parents=True, exist_ok=True)
    as_dark = evaluate_via_maven(
        gemp_src=GEMP_SRC,
        run_dir=work_dir,
        dark_weights=candidate_weights,
        light_weights=opponent_weights,
        games=games_per_side,
        label=f"{label_prefix}-cand-dark",
    )
    as_light = evaluate_via_maven(
        gemp_src=GEMP_SRC,
        run_dir=work_dir,
        dark_weights=opponent_weights,
        light_weights=candidate_weights,
        games=games_per_side,
        label=f"{label_prefix}-cand-light",
    )
    cand_wins = (as_dark.get("darkWins") or 0) + (as_light.get("lightWins") or 0)
    total = (as_dark.get("finished") or 0) + (as_light.get("finished") or 0)
    rate = (cand_wins / total) if total else 0.0
    dark_side = as_dark.get("darkWinRate")
    light_side = as_light.get("lightWinRate")
    expected = games_per_side * baseline_dark_rate + games_per_side * baseline_light_rate
    expected_rate = (expected / total) if total else 0.5
    dark_delta = (dark_side - baseline_dark_rate) if dark_side is not None else None
    light_delta = (light_side - baseline_light_rate) if light_side is not None else None
    # vs equal-strength champ baseline rates should be ~0.5/0.5 if same policy both sides;
    # for BEGINNER use collect side rates; for champ-vs-champ use 0.5/0.5
    side_ok = (
        (dark_delta is not None and dark_delta >= 0.05)
        or (light_delta is not None and light_delta >= 0.05)
    )
    combined_ok = rate >= expected_rate + 0.05
    passed = bool(total >= max(4, games_per_side) and side_ok and combined_ok)
    reason = (
        f"{'PASS' if passed else 'FAIL'}: dark {dark_side} (base {baseline_dark_rate}, d={dark_delta}), "
        f"light {light_side} (base {baseline_light_rate}, d={light_delta}), "
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
        "opponent": str(opponent_weights) if opponent_weights else "builtin-BEGINNER",
    }


def git_push_trainer(message: str) -> str | None:
    """Commit + push trainer changes. Returns commit URL or None."""
    try:
        subprocess.run(["git", "add", "champs", "docs", "trainer"], cwd=ROOT, check=False)
        # only stage report docs + champs packs (not huge runs)
        st = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
        if not st.stdout.strip():
            return None
        subprocess.run(["git", "add", "-A", "champs", "docs", "trainer"], cwd=ROOT, check=False)
        # unstage accidental large artifacts
        subprocess.run(["git", "reset", "HEAD", "--", "runs/", "trainer-server.log", "trainer-server.pid"], cwd=ROOT, check=False)
        c = subprocess.run(
            ["git", "commit", "-m", message],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if c.returncode != 0:
            return None
        subprocess.run(["git", "push", "origin", "HEAD"], cwd=ROOT, check=False, capture_output=True)
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return f"https://github.com/billbisco/gemp-swccg-trainer/commit/{sha}"
    except Exception as e:
        return f"push-error:{e}"


def write_report(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n")


def run_round(
    *,
    round_idx: int,
    collect_games: int,
    gate_n: int,
    n_search: int,
    max_live_cands: int,
    baseline_dark: float,
    baseline_light: float,
) -> dict[str, Any]:
    champ_id = current_champ_id()
    champ_w = champ_weights_path(champ_id)
    base = load_weights(champ_w)
    round_dir = RUNS / f"round{round_idx:02d}-{ts()}"
    round_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== ROUND {round_idx} champ={champ_id} dir={round_dir}", flush=True)

    # 1) Collect self-play with champ weights both sides (throughput: no replay)
    print(f"[collect] {collect_games} games champ vs champ …", flush=True)
    collect = collect_selfplay(
        run_dir=round_dir / "collect",
        dark_w=champ_w,
        light_w=champ_w,
        games=collect_games,
        label="collect-champ-v-champ",
    )
    print(f"[collect] done: {collect}", flush=True)

    # Side rates from equal-policy self-play should be near deck asymmetry
    # Use BEGINNER baseline for gate-vs-beginner; 0.5 for gate-vs-champ
    traces = None  # throughput: no traces this round

    # 2) Search mutants from champ base
    ranked = search_from_champ(base, traces_path=traces, n_candidates=n_search, seed=42 + round_idx)
    (round_dir / "search_ranked.json").write_text(
        json.dumps([{"id": r["id"], "proxyScore": r["proxyScore"]} for r in ranked], indent=2) + "\n"
    )

    # 3) Live-gate top candidates (vs BEGINNER + vs champ)
    best: dict[str, Any] | None = None
    tried: list[dict[str, Any]] = []
    for cand in ranked[:max_live_cands]:
        cid = cand["id"]
        wpath = round_dir / f"cand-{cid}-weights.json"
        wpath.write_text(json.dumps(cand["weights"], indent=2) + "\n")
        print(f"[gate] {cid} vs BEGINNER N={gate_n}/side …", flush=True)
        g_beg = gate_vs_opponent(
            work_dir=round_dir / "gate" / cid / "vs-beginner",
            candidate_weights=wpath,
            opponent_weights=None,
            games_per_side=gate_n,
            label_prefix="vsbeg",
            baseline_dark_rate=baseline_dark,
            baseline_light_rate=baseline_light,
        )
        print(f"[gate] {cid} vs BEGINNER → {g_beg['passed']} {g_beg['combinedWinRate']}", flush=True)
        print(f"[gate] {cid} vs champ {champ_id} N={gate_n}/side …", flush=True)
        # Equal-policy WC96 side rates ≈ deck asymmetry (not 0.5/0.5).
        # Prefer this-round collect rates when available; else BEGINNER collect priors.
        ch_base_dark = float(collect.get("darkWinRate") or baseline_dark)
        ch_base_light = float(collect.get("lightWinRate") or baseline_light)
        g_ch = gate_vs_opponent(
            work_dir=round_dir / "gate" / cid / "vs-champ",
            candidate_weights=wpath,
            opponent_weights=champ_w,
            games_per_side=gate_n,
            label_prefix="vschamp",
            baseline_dark_rate=ch_base_dark,
            baseline_light_rate=ch_base_light,
        )
        print(f"[gate] {cid} vs champ → {g_ch['passed']} {g_ch['combinedWinRate']}", flush=True)
        entry = {"id": cid, "vsBeginner": g_beg, "vsChamp": g_ch, "weightsPath": str(wpath)}
        tried.append(entry)
        # Promote only if beats BEGINNER bar AND beats (or ties-strong) previous champ
        # Honest: need vsBeginner pass AND vsChamp combined >= 0.55 (clear edge) OR vsChamp pass
        if g_beg.get("passed") and (g_ch.get("passed") or (g_ch.get("combinedWinRate") or 0) >= 0.55):
            best = entry
            break
        # Keep best-so-far by vsChamp WR if beginner-pass
        if g_beg.get("passed"):
            if best is None or (g_ch.get("combinedWinRate") or 0) > (best["vsChamp"].get("combinedWinRate") or 0):
                best = entry

    (round_dir / "gate_tried.json").write_text(json.dumps(tried, indent=2, default=str) + "\n")

    result: dict[str, Any] = {
        "round": round_idx,
        "parentChamp": champ_id,
        "collect": collect,
        "promoted": False,
        "newChamp": None,
        "best": best,
        "futureBenchmarkNote": FUTURE_BENCHMARK,
        "dir": str(round_dir),
    }

    # 4) Promote only on honest double-gate
    if best and best["vsBeginner"].get("passed") and (
        best["vsChamp"].get("passed") or (best["vsChamp"].get("combinedWinRate") or 0) >= 0.55
    ):
        new_id = f"heuristic-v1-wc96-r{round_idx:02d}-{best['id']}"
        notes = (
            f"# {new_id}\n\n"
            f"Parent: `{champ_id}`\n"
            f"Collect: {collect_games} champ-vs-champ WC96 (replay off)\n"
            f"vs BEGINNER: {best['vsBeginner'].get('reason')}\n"
            f"vs champ: {best['vsChamp'].get('reason')}\n"
            f"\nHonesty: promoted only after live gates. Collection ≠ learning.\n"
            f"\nFuture benchmark (not used): {FUTURE_BENCHMARK}\n"
        )
        gate_blob = {
            "vsBeginner": best["vsBeginner"],
            "vsChamp": best["vsChamp"],
            "passed": True,
        }
        pack = export_candidate_champ(
            CHAMPS,
            load_weights(Path(best["weightsPath"])),
            champ_id=new_id,
            display_name=f"WC96-Heuristic-R{round_idx}",
            notes=notes,
            parent_id=champ_id,
            gate=gate_blob,
        )
        dest = PROMOTED / new_id
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(pack["dir"], dest)
        set_current(new_id)
        result["promoted"] = True
        result["newChamp"] = new_id
        result["pack"] = pack
        report = (
            f"# WC96 loop round {round_idx} PROMOTE\n\n"
            f"- Parent: `{champ_id}`\n"
            f"- New: `{new_id}`\n"
            f"- Collect finished={collect.get('finished')} darkWR={collect.get('darkWinRate')} lightWR={collect.get('lightWinRate')}\n"
            f"- vs BEGINNER combined WR={best['vsBeginner'].get('combinedWinRate')} (N={gate_n}/side)\n"
            f"- vs champ combined WR={best['vsChamp'].get('combinedWinRate')} (N={gate_n}/side)\n"
            f"- Replay: off (throughput). CSV under `{round_dir}`.\n"
            f"- Future benchmark note: {FUTURE_BENCHMARK}\n"
        )
        write_report(DOCS / f"wc96-loop-round{round_idx:02d}-REPORT.md", report)
        url = git_push_trainer(f"Promote {new_id} (WC96 loop round {round_idx})")
        result["commitUrl"] = url
        print(f"[PROMOTE] {new_id} {url}", flush=True)
        # Larger confirmation vs BEGINNER (soft gates have been wrong before).
        try:
            confirm_n = max(gate_n, 20)
            print(f"[confirm] re-gate {new_id} vs BEGINNER N={confirm_n}/side …", flush=True)
            conf = gate_vs_opponent(
                work_dir=round_dir / "confirm",
                candidate_weights=Path(pack["dir"]) / "weights.json",
                opponent_weights=None,
                games_per_side=confirm_n,
                label_prefix="confirm",
                baseline_dark_rate=baseline_dark,
                baseline_light_rate=baseline_light,
            )
            (round_dir / "confirm.json").write_text(json.dumps(conf, indent=2, default=str) + "\n")
            result["confirm"] = {k: conf.get(k) for k in ("passed", "combinedWinRate", "reason", "darkDelta", "lightDelta")}
            print(f"[confirm] passed={conf.get('passed')} WR={conf.get('combinedWinRate')}", flush=True)
            if not conf.get("passed"):
                print("[confirm] WARNING: soft promote did not hold at larger N — keep searching", flush=True)
        except Exception as ce:
            print(f"[confirm] skipped: {ce}", flush=True)

    else:
        report = (
            f"# WC96 loop round {round_idx} — no promote\n\n"
            f"- Champ remains: `{champ_id}`\n"
            f"- Collect: {json.dumps(collect)}\n"
            f"- Best tried: {json.dumps(best, default=str)[:2000] if best else None}\n"
            f"- Honest fail-closed. Replay off.\n"
            f"- Future benchmark note: {FUTURE_BENCHMARK}\n"
        )
        write_report(DOCS / f"wc96-loop-round{round_idx:02d}-REPORT.md", report)
        url = git_push_trainer(f"WC96 loop round {round_idx} report (no promote)")
        result["commitUrl"] = url
        print(f"[no-promote] champ stays {champ_id}", flush=True)

    (round_dir / "round_result.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    return result


def regate_champ(*, games_per_side: int, baseline_dark: float, baseline_light: float) -> dict[str, Any]:
    champ_id = current_champ_id()
    champ_w = champ_weights_path(champ_id)
    out = RUNS / f"regate-{ts()}"
    print(f"[regate] {champ_id} vs BEGINNER N={games_per_side}/side → {out}", flush=True)
    g = gate_vs_opponent(
        work_dir=out,
        candidate_weights=champ_w,
        opponent_weights=None,
        games_per_side=games_per_side,
        label_prefix="regate",
        baseline_dark_rate=baseline_dark,
        baseline_light_rate=baseline_light,
    )
    (out / "regate.json").write_text(json.dumps(g, indent=2, default=str) + "\n")
    report = (
        f"# WC96 re-gate {champ_id}\n\n"
        f"- N={games_per_side}/side vs builtin BEGINNER\n"
        f"- passed={g.get('passed')}\n"
        f"- combined WR={g.get('combinedWinRate')} expected={g.get('expectedWinRate')}\n"
        f"- darkDelta={g.get('darkDelta')} lightDelta={g.get('lightDelta')}\n"
        f"- reason: {g.get('reason')}\n"
        f"- Soft prior gate was N=8; this is the larger confirmation.\n"
        f"- Future benchmark (not used): {FUTURE_BENCHMARK}\n"
    )
    write_report(DOCS / "wc96-regate-REPORT.md", report)
    url = git_push_trainer(f"WC96 re-gate {champ_id} N={games_per_side}/side")
    g["champId"] = champ_id
    g["commitUrl"] = url
    g["dir"] = str(out)
    print(f"[regate] done passed={g.get('passed')} WR={g.get('combinedWinRate')} {url}", flush=True)
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--collect-games", type=int, default=50)
    ap.add_argument("--gate-n", type=int, default=12, help="games per side for gates")
    ap.add_argument("--regate-n", type=int, default=25, help="games per side for initial re-gate")
    ap.add_argument("--skip-regate", action="store_true")
    ap.add_argument("--n-search", type=int, default=16)
    ap.add_argument("--max-live-cands", type=int, default=2)
    ap.add_argument("--baseline-dark", type=float, default=0.34)
    ap.add_argument("--baseline-light", type=float, default=0.66)
    ap.add_argument("--start-round", type=int, default=1)
    args = ap.parse_args()

    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / ".gitkeep").touch()
    # ensure CURRENT pointer
    if not (PROMOTED / "CURRENT").exists():
        set_current(current_champ_id())

    status_path = RUNS / "loop_status.json"
    history: list[Any] = []

    if not args.skip_regate:
        reg = regate_champ(
            games_per_side=args.regate_n,
            baseline_dark=args.baseline_dark,
            baseline_light=args.baseline_light,
        )
        history.append({"phase": "regate", **{k: reg.get(k) for k in ("passed", "combinedWinRate", "champId", "commitUrl", "reason")}})
        status_path.write_text(json.dumps({"history": history, "updatedAt": ts()}, indent=2) + "\n")

    for r in range(args.start_round, args.start_round + args.rounds):
        t0 = time.time()
        try:
            res = run_round(
                round_idx=r,
                collect_games=args.collect_games,
                gate_n=args.gate_n,
                n_search=args.n_search,
                max_live_cands=args.max_live_cands,
                baseline_dark=args.baseline_dark,
                baseline_light=args.baseline_light,
            )
        except Exception as e:
            print(f"[ERROR] round {r}: {e}", flush=True)
            history.append({"phase": "round", "round": r, "error": str(e)})
            status_path.write_text(json.dumps({"history": history, "updatedAt": ts()}, indent=2) + "\n")
            continue
        history.append(
            {
                "phase": "round",
                "round": r,
                "promoted": res.get("promoted"),
                "newChamp": res.get("newChamp"),
                "parentChamp": res.get("parentChamp"),
                "commitUrl": res.get("commitUrl"),
                "wallSec": round(time.time() - t0, 1),
            }
        )
        status_path.write_text(
            json.dumps(
                {
                    "history": history,
                    "currentChamp": current_champ_id(),
                    "updatedAt": ts(),
                    "futureBenchmarkNote": FUTURE_BENCHMARK,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
