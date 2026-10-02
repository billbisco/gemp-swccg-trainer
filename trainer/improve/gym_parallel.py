"""Parallel HeadlessBotVsBotBatch JVM workers (direct java, no per-batch Maven).

Throughput path: shard N games across W processes, merge CSVs.
Replay off by default. Not for spawning more Grok bots — Java gym only.
"""
from __future__ import annotations

import csv
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from trainer.improve.search import summarize_csv

GEMP_SRC = Path("/workspace/swccg-gemp/src")
DEFAULT_CP_FILE = Path("/workspace/gemp-swccg-trainer/runs/wc96-loop/gym.classpath")
MAIN = "com.gempukku.swccgo.ai.HeadlessBotVsBotBatch"


def load_classpath(cp_file: Path = DEFAULT_CP_FILE) -> str:
    if not cp_file.is_file():
        raise FileNotFoundError(f"missing classpath file: {cp_file}")
    return cp_file.read_text().strip()


def _split_games(total: int, workers: int) -> list[int]:
    workers = max(1, min(workers, total))
    base, rem = divmod(total, workers)
    return [base + (1 if i < rem else 0) for i in range(workers) if base + (1 if i < rem else 0) > 0]


def _run_shard(
    *,
    classpath: str,
    games: int,
    csv_path: Path,
    log_path: Path,
    dark_weights: Path | None,
    light_weights: Path | None,
    max_millis: int,
    shard_id: int,
) -> dict[str, Any]:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "java",
        "-Xms256m",
        "-Xmx1024m",
        "-cp",
        classpath,
        MAIN,
        f"--games={games}",
        "--dark=BEGINNER",
        "--light=BEGINNER",
        "--decks=wc96",
        "--format=premiere_anh",
        "--no-traces",
        "--no-replay",
        "--quiet",
        f"--maxMillis={max_millis}",
        f"--csv={csv_path.resolve()}",
    ]
    if dark_weights is not None:
        cmd.append(f"--dark-weights={Path(dark_weights).resolve()}")
    if light_weights is not None:
        cmd.append(f"--light-weights={Path(light_weights).resolve()}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(GEMP_SRC / "gemp-swccg-server"), capture_output=True, text=True)
    elapsed = time.time() - t0
    log_path.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr)
    return {
        "shard": shard_id,
        "games": games,
        "exit": proc.returncode,
        "wallSec": round(elapsed, 2),
        "csv": str(csv_path),
        "ok": proc.returncode == 0 and csv_path.is_file(),
    }


def merge_csvs(shard_csvs: list[Path], out_csv: Path) -> int:
    """Concatenate shard CSVs with a single header. Returns data row count."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = None
    rows: list[list[str]] = []
    for p in shard_csvs:
        if not p.is_file():
            continue
        with p.open(encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            h = next(reader, None)
            if h is None:
                continue
            if header is None:
                header = h
            for row in reader:
                if row:
                    rows.append(row)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if header:
            w.writerow(header)
        w.writerows(rows)
    return len(rows)


def evaluate_parallel(
    *,
    run_dir: Path,
    dark_weights: Path | None,
    light_weights: Path | None,
    games: int,
    label: str,
    workers: int = 6,
    max_millis: int = 300_000,
    cp_file: Path = DEFAULT_CP_FILE,
) -> dict[str, Any]:
    """Run W concurrent JVM batch shards; merge to label.csv; return summarize_csv stats."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    workers = int(os.environ.get("GEMP_GYM_WORKERS", workers))
    chunks = _split_games(games, workers)
    classpath = load_classpath(cp_file)
    shard_dir = run_dir / f"{label}.shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    shard_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(chunks)) as ex:
        futs = {}
        for i, n in enumerate(chunks):
            csv_path = shard_dir / f"shard-{i:02d}.csv"
            log_path = shard_dir / f"shard-{i:02d}.log"
            fut = ex.submit(
                _run_shard,
                classpath=classpath,
                games=n,
                csv_path=csv_path,
                log_path=log_path,
                dark_weights=dark_weights,
                light_weights=light_weights,
                max_millis=max_millis,
                shard_id=i,
            )
            futs[fut] = i
        for fut in as_completed(futs):
            shard_results.append(fut.result())
    shard_results.sort(key=lambda r: r["shard"])

    out_csv = run_dir / f"{label}.csv"
    shard_csvs = [Path(r["csv"]) for r in shard_results]
    merge_csvs(shard_csvs, out_csv)
    stats = summarize_csv(out_csv)
    wall = time.time() - t0
    stats["label"] = label
    stats["mavenExit"] = 0 if all(r.get("ok") for r in shard_results) else 1
    stats["wallSec"] = round(wall, 2)
    stats["csv"] = str(out_csv)
    stats["log"] = str(shard_dir)
    stats["workers"] = len(chunks)
    stats["shardGames"] = chunks
    stats["shards"] = shard_results
    stats["gamesPerHour"] = round((stats.get("finished") or 0) / wall * 3600, 1) if wall > 0 else None
    stats["engine"] = "parallel-jvm-cli"
    (run_dir / f"{label}.parallel.json").write_text(
        __import__("json").dumps(
            {
                "workers": len(chunks),
                "chunks": chunks,
                "wallSec": stats["wallSec"],
                "gamesPerHour": stats["gamesPerHour"],
                "shards": shard_results,
            },
            indent=2,
        )
        + "\n"
    )
    return stats
