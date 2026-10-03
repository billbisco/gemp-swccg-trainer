"""AckbarBot linear.v1 seeds from YodaBot / AdvancedAi keyword scores.

YodaBot is the hall id ``~YodaBot`` (``HallServer.AI_ADVANCED_ID``). That id
constructs ``AdvancedAi``. Keyword numbers are the arrays in
``AdvancedAi.java``, not BeginnerAi and not ``getPassPenalty`` (320), which
is a separate subtraction and not a keyword weight.

``LinearPolicyAi`` action features (24), in order:

0 pass flag
1 integer value normalized to [0, 1]
2 index normalized across candidates
3..18 text-hash one-hot (16 buckets)
19..22 blueprintId hash one-hot (4 buckets)
23 constant 1

Those hash buckets are not keyword identities. A keyword is copied only when
its name is one of those features. Anything else stays 0. Packed weights, bag
weights, and bias stay 0.

The only name match is action-penalty ``pass`` = -160 onto feature 0. The
choice-table ``pass`` = -40 is a second table and there is no second pass
feature, so it is not added. Sign stays negative, the same as Yoda.

Any non-zero action weight disables ``LinearPolicyAi``'s all-zero anti-stall
prior. Every positive scale of this pack is the same greedy policy: pass
scores below 0 and every other action scores 0, so the earliest legal
candidate wins. Scales are still gated separately because the gym shuffle is
unseeded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from trainer.improve.linear_overnight import (
    EVEN_BAR,
    gate_vs_kept,
    log,
    should_promote,
    write_pointer,
)
from trainer.improve.linear_selfplay import (
    ACTION_FEAT_DIM,
    AF_BP_BUCKETS,
    AF_BP_HASH,
    AF_INDEX,
    AF_INTEGER,
    AF_ONES,
    AF_PASS,
    AF_TEXT_BUCKETS,
    AF_TEXT_HASH,
    BAG_HASH_DIM,
    PACKED_DIM,
    REPO,
    W_LEN,
    find_classpath,
    make_pack,
    write_pack,
)

SEED_SCHEMA = "ackbar-seed.v1"
SOURCE = "yoda-advanced-keywords"
SCALES = (0.0, 0.125, 0.25, 0.5, 1.0, 1.25, 1.5, 2.0)
# Overnight script: --gate-per-seat 4, and should_promote min_games is half of that.
GATE_PER_SEAT = 4
MIN_FINISHED = GATE_PER_SEAT // 2

# AdvancedAi.ACTION_WEIGHTS then ACTION_PENALTIES. Order matches the Java arrays.
YODA_ACTION_KEYWORDS: tuple[tuple[str, int], ...] = (
    ("force drain", 160),
    ("initiate battle", 150),
    ("battle", 100),
    ("weapon", 50),
    ("fire", 45),
    ("deploy", 90),
    ("play", 40),
    ("move", 50),
    ("activate", 70),
    ("retrieve", 40),
    ("draw", 35),
    ("steal", 35),
    ("capture", 35),
    ("download", 45),
    ("search", 30),
    ("react", 30),
    ("cancel", 30),
    ("take into hand", 35),
    ("pass", -160),
    ("forfeit", -100),
    ("lose", -60),
    ("place in lost pile", -90),
    ("place in used pile", -40),
    ("return to hand", -25),
    ("sacrifice", -120),
    ("revert", -60),
)

# AdvancedAi.CHOICE_WEIGHTS then CHOICE_PENALTIES. No choice-only features exist
# in the 24, so these stay unmapped. Choice "pass" is not added onto feature 0.
YODA_CHOICE_KEYWORDS: tuple[tuple[str, int], ...] = (
    ("draw", 60),
    ("retrieve", 45),
    ("deploy", 40),
    ("battle destiny", 50),
    ("weapon destiny", 50),
    ("activate", 40),
    ("force drain", 60),
    ("initiate", 40),
    ("capture", 30),
    ("steal", 30),
    ("download", 30),
    ("use", 10),
    ("yes", 10),
    ("lose", -55),
    ("forfeit", -70),
    ("lost pile", -60),
    ("used pile", -35),
    ("return to hand", -25),
    ("neither", -30),
    ("cancel", -30),
    ("pass", -40),
)

# Feature index -> the keyword name that feature actually is.
# Only the pass flag has a keyword name. Hash buckets do not.
ACTION_FEATURE_KEYWORD: dict[int, str] = {
    AF_PASS: "pass",
}


def action_feature_names() -> list[str]:
    names = ["pass", "integerNorm", "indexNorm"]
    names.extend(f"textHash{i}" for i in range(AF_TEXT_BUCKETS))
    names.extend(f"blueprintHash{i}" for i in range(AF_BP_BUCKETS))
    names.append("ones")
    if len(names) != ACTION_FEAT_DIM:
        raise RuntimeError(f"feature name count {len(names)} != {ACTION_FEAT_DIM}")
    if names[AF_PASS] != "pass" or names[AF_INTEGER] != "integerNorm":
        raise RuntimeError("feature order drifted from LinearPolicyAi")
    if names[AF_INDEX] != "indexNorm" or names[AF_TEXT_HASH] != "textHash0":
        raise RuntimeError("feature order drifted from LinearPolicyAi")
    if names[AF_BP_HASH] != "blueprintHash0" or names[AF_ONES] != "ones":
        raise RuntimeError("feature order drifted from LinearPolicyAi")
    return names


def yoda_action_base() -> list[float]:
    """Keyword scores aligned onto the 24 action weights. Unmapped stay 0."""
    weights = [0.0] * ACTION_FEAT_DIM
    for keyword, score in YODA_ACTION_KEYWORDS:
        matched = [idx for idx, name in ACTION_FEATURE_KEYWORD.items() if name == keyword]
        if len(matched) != 1:
            continue
        weights[matched[0]] = float(score)
    return weights


def unmapped_keywords() -> list[dict[str, Any]]:
    """Keywords with no action feature of the same name.

    Choice-table ``pass`` is listed even though the action table mapped
    ``pass``: there is only one pass feature, and the choice weight is not
    added to it.
    """
    rows: list[dict[str, Any]] = []
    for table, keywords in (
        ("action", YODA_ACTION_KEYWORDS),
        ("choice", YODA_CHOICE_KEYWORDS),
    ):
        for keyword, score in keywords:
            if table == "action" and keyword in ACTION_FEATURE_KEYWORD.values():
                continue
            rows.append({"table": table, "keyword": keyword, "score": score})
    return rows


def features_without_keyword() -> list[dict[str, Any]]:
    names = action_feature_names()
    out = []
    for idx, name in enumerate(names):
        if idx in ACTION_FEATURE_KEYWORD:
            continue
        out.append({"index": idx, "feature": name})
    return out


def seed_meta(scale: float) -> dict[str, Any]:
    return {
        "schema": SEED_SCHEMA,
        "source": SOURCE,
        "scale": float(scale),
        "packed": "zeros",
        "bag": "zeros",
        "action": "yoda * scale",
    }


def make_seed_pack(scale: float) -> dict[str, Any]:
    """linear.v1 pack. Packed and bag stay 0. Bias 0. Action is yoda * scale."""
    if scale < 0:
        raise ValueError("scale must be >= 0")
    pack = make_pack("zeros")
    base = PACKED_DIM + BAG_HASH_DIM
    aligned = yoda_action_base()
    for i, value in enumerate(aligned):
        pack["W"][base + i] = float(value) * float(scale)
    pack["bias"] = 0.0
    pack["seed"] = seed_meta(scale)
    if len(pack["W"]) != W_LEN:
        raise RuntimeError("W length drifted")
    if any(pack["W"][i] != 0.0 for i in range(base)):
        raise RuntimeError("packed or bag weight was not left at 0")
    return pack


def scale_filename(scale: float) -> str:
    text = f"{float(scale):.3f}".rstrip("0").rstrip(".")
    if text == "-0":
        text = "0"
    return f"scale-{text}.linear.json"


def write_seed_packs(seeds_dir: Path) -> list[Path]:
    seeds_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for scale in SCALES:
        path = seeds_dir / scale_filename(scale)
        write_pack(path, make_seed_pack(scale))
        paths.append(path)
    alignment = {
        "source": "AdvancedAi.java via HallServer ~YodaBot",
        "actionFeatures": action_feature_names(),
        "mapped": [
            {
                "feature": "pass",
                "index": AF_PASS,
                "table": "action",
                "keyword": "pass",
                "yoda": -160,
                "note": "AdvancedAi.ACTION_PENALTIES. Choice-table pass -40 is not added.",
            }
        ],
        "unmappedKeywords": unmapped_keywords(),
        "featuresWithoutKeyword": features_without_keyword(),
        "notUsed": {
            "getPassPenalty": 320,
            "why": "AdvancedAi overrides HeuristicAiBase.getPassPenalty (200) to 320. It is not a KeywordWeight.",
        },
        "scales": list(SCALES),
        "policyNote": (
            "Positive scales share one greedy policy. Pass is strictly negative "
            "and every other action weight is 0, so non-pass actions tie at 0 "
            "and the earliest candidate wins. The zeros anti-stall prior is off "
            "whenever any weight is non-zero."
        ),
    }
    (seeds_dir / "alignment.json").write_text(
        json.dumps(alignment, indent=2) + "\n", encoding="utf-8"
    )
    return paths


def beats_zeros(meas: dict[str, Any]) -> tuple[bool, dict[str, str]]:
    """Same bar as the overnight gate against an even split."""
    if not meas.get("ok"):
        return False, {"gate": "jvm failed or metrics missing"}
    return should_promote(meas, EVEN_BAR, MIN_FINISHED)


def pooled_lf(meas: dict[str, Any]) -> float | None:
    value = meas.get("meanCandidateLfDiff")
    if value is None:
        return None
    return float(value)


def pick_winner(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Seed that beats zeros on both seats. Ties: higher pooled mean LF, then lower scale.

    Scale 0 is not a row. If none qualify, the kept pack is zeros.
    """
    both = [row for row in rows if row.get("beatsBoth")]
    if not both:
        return {"kept": "zeros", "scale": 0.0, "reason": "no seed beat zeros on both seats"}
    both.sort(key=lambda row: (-float(row["pooledMeanLfDiff"]), float(row["scale"])))
    best = both[0]
    return {
        "kept": "seed",
        "scale": best["scale"],
        "path": best["path"],
        "pooledMeanLfDiff": best["pooledMeanLfDiff"],
        "reason": "beat zeros on both seats; best pooled mean LF diff",
    }


def run_ladder(
    *,
    out_dir: Path,
    classpath: str,
    max_millis: int = 180_000,
    max_decisions: int = 8000,
) -> dict[str, Any]:
    """Gate each non-zero seed against the scale-0 zeros pack. Write the kept pack."""
    seeds_dir = out_dir / "seeds"
    paths = write_seed_packs(seeds_dir)
    by_scale = {float(scale): path for scale, path in zip(SCALES, paths)}
    zeros = by_scale[0.0]
    rows: list[dict[str, Any]] = []
    for scale in SCALES:
        if scale == 0.0:
            log("scale 0 is the zeros control; not a candidate")
            continue
        gate_dir = seeds_dir / "gates" / scale_filename(scale).replace(".linear.json", "")
        log(f"gate scale={scale} vs zeros both seats games={GATE_PER_SEAT} seed=NONE")
        meas = gate_vs_kept(
            classpath=classpath,
            cand_weights=by_scale[scale],
            kept_weights=zeros,
            gate_dir=gate_dir,
            per_seat=GATE_PER_SEAT,
            max_millis=max_millis,
            max_decisions=max_decisions,
        )
        ok, detail = beats_zeros(meas)
        pooled = pooled_lf(meas)
        row = {
            "scale": scale,
            "path": str(by_scale[scale]),
            "beatsBoth": ok,
            "promoteDetail": detail,
            "pooledMeanLfDiff": pooled,
            "asDark": meas.get("asDark"),
            "asLight": meas.get("asLight"),
            "gateOk": bool(meas.get("ok")),
        }
        rows.append(row)
        log(
            f"scale={scale} beatsBoth={ok} dark={meas.get('asDark')} "
            f"light={meas.get('asLight')} pooledLF={pooled} detail={detail}"
        )
    winner = pick_winner(rows)
    if winner["kept"] == "seed":
        kept_path = Path(winner["path"])
        note = (
            f"Ackbar seed ladder kept scale {winner['scale']} "
            "(Yoda action pass * scale on feature 0 only). "
            "Beat the zeros pack on both seats."
        )
        init = f"ackbar-seed-scale-{winner['scale']}"
    else:
        kept_path = zeros
        note = (
            "Ackbar seed ladder kept the all-zeros pack. "
            "No non-zero Yoda-aligned seed beat zeros on both seats."
        )
        init = "zeros"
    pointer = {
        "schema": "linear-overnight-current.v1",
        "role": "ackbar-seed" if winner["kept"] == "seed" else "baseline",
        "round": 0,
        "init": init,
        "promoted": winner["kept"] == "seed",
        "metrics": {
            "ladder": rows,
            "winner": winner,
        },
        "comparedTo": {"pack": "seeds/scale-0.linear.json", "role": "zeros-control"},
        "note": note,
    }
    write_pointer(out_dir, kept_path, pointer)
    report = {
        "winner": winner,
        "rows": rows,
        "unmappedKeywords": unmapped_keywords(),
        "mapped": {"pass": -160, "feature": 0},
        "keptWeights": str((out_dir / "current.linear.json").resolve()),
        "note": note,
    }
    (seeds_dir / "ladder-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    log(note)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="AckbarBot Yoda-keyword seed ladder vs zeros.")
    parser.add_argument("--out-dir", type=Path, default=REPO / "runs" / "linear-overnight")
    parser.add_argument("--max-millis", type=int, default=180_000)
    parser.add_argument("--max-decisions", type=int, default=8000)
    parser.add_argument("--write-only", action="store_true", help="write seed packs and stop")
    args = parser.parse_args(argv)
    if args.write_only:
        write_seed_packs(args.out_dir / "seeds")
        return
    classpath = find_classpath(None)
    if not classpath:
        raise SystemExit("no gym classpath")
    run_ladder(
        out_dir=args.out_dir,
        classpath=classpath,
        max_millis=args.max_millis,
        max_decisions=args.max_decisions,
    )


if __name__ == "__main__":
    main()
