"""First linear.v1 self-play step for AckbarBot.

Fits a gym-cli ``linear.v1`` pack (the JSON ``LinearPolicyAi`` loads). Does not
promote a champ, does not distill from YodaBot/AdvancedAi, and does not touch
the keyword mutate loop.

Update rule
-----------
``LinearPolicyAi`` scores an action as
``bias + dot(W, concat(packed[128], bagHash[bagHashDim], actionFeat[24]))``
and plays the greedy argmax. ``packed`` and ``bagHash`` are state features:
they are identical for every legal action, so their weights (and ``bias``,
and action feature 23 which is the constant 1) cannot change the choice.
Only action features 0..22 move the greedy policy.

FEATURES traces include ``state.packed`` and ``chosen``, plus a capped option
list, but not the action-feature matrix and not a per-decision advantage.
When the logged options are complete enough to rebuild the candidate list
(same action features as ``LinearPolicyAi.actionFeatures``, including Java
``String.hashCode`` buckets), this step applies a softmax surrogate of
REINFORCE on those action features, with the episode return (side win +
clipped life-force differential) shared by every decision of that seat.
The softmax is a training surrogate only. The gym policy stays greedy.

That gradient is zero for packed, bag, bias, and the constant-ones feature.
Those entries stay at their init (zeros, unless ``--init small-random``).

If no decision can be aligned, the step falls back to an episode-level stub:
it nudges pass / integerNorm / indexNorm by the side-split return gap.
Those three signs are arbitrary probes, not tactics. The stub exists so a
candidate file is still written. It is not a policy gradient.

Gym shuffle seeds are never set. Strength numbers are not from a fixed seed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA = "linear.v1"
FEATURE_SCHEMA_VERSION = 1
PACKED_DIM = 128
BAG_HASH_DIM = 16  # LinearPolicyAi.DEFAULT_BAG_HASH_DIM / zeros()
ACTION_FEAT_DIM = 24
W_LEN = PACKED_DIM + BAG_HASH_DIM + ACTION_FEAT_DIM  # 168

AF_PASS = 0
AF_INTEGER = 1
AF_INDEX = 2
AF_TEXT_HASH = 3
AF_TEXT_BUCKETS = 16
AF_BP_HASH = 19
AF_BP_BUCKETS = 4
AF_ONES = 23
INTEGER_ENUM_CAP = 24

DARK_PLAYER = "~OzzelBot"
LIGHT_PLAYER = "~AckbarBot"
LF_SCALE = 30.0
LF_MIX = 0.25
WEIGHT_CLIP = 1.0

GEMP_SERVER = Path("/workspace/swccg-gemp/src/gemp-swccg-server")
DEFAULT_CP_FILE = Path("/workspace/gemp-swccg-trainer/runs/wc96-loop/gym.classpath")
MAIN = "com.gempukku.swccgo.ai.HeadlessBotVsBotBatch"
REPO = Path("/workspace/gemp-swccg-trainer")

LIMITATION = (
    "Greedy linear.v1 adds packed[128] and bagHash to every legal action, so those "
    "weights cannot change the action. This step updates action features 0..22 only, "
    "via a softmax surrogate of REINFORCE (the deployed policy is still greedy argmax, "
    "so this is not the gradient of the policy that plays). The same episode return "
    "(win + clipped LF differential) is copied onto every aligned decision — there is "
    "no per-decision advantage. Decisions whose option list is truncated or not "
    "rebuildable are skipped. If none align, pass/integer/index are nudged by the "
    "side-split return gap; that nudge is an arbitrary stub, not a learned tactic. "
    "Not distilled from YodaBot/AdvancedAi. Not promoted."
)

UPDATE_REINFORCE = "softmax-surrogate-reinforce-action-features"
UPDATE_STUB = "episode-return-stub-pass-integer-index"


def java_string_hash(value: str) -> int:
    """Java ``String.hashCode`` for BMP strings (SWCCG labels are BMP)."""
    h = 0
    for ch in value:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    if h >= 0x80000000:
        h -= 0x100000000
    return h


def bucket(value: str, buckets: int) -> int:
    if buckets <= 1:
        return 0
    mod = java_string_hash(value) % buckets
    return mod + buckets if mod < 0 else mod


def is_pass_text(text: str | None) -> bool:
    if not text:
        return False
    t = text.lower().strip()
    return t == "pass" or t.startswith("pass ") or t.endswith(" pass") or " pass " in t


def action_features(
    text: str,
    blueprint_id: str,
    passed: bool,
    integer_norm: float,
    index_norm: float,
) -> list[float]:
    """Match ``LinearPolicyAi.actionFeatures``."""
    feat = [0.0] * ACTION_FEAT_DIM
    feat[AF_PASS] = 1.0 if passed or is_pass_text(text) else 0.0
    feat[AF_INTEGER] = float(integer_norm)
    feat[AF_INDEX] = float(index_norm)
    hashed = (text or "").lower()
    feat[AF_TEXT_HASH + bucket(hashed, AF_TEXT_BUCKETS)] = 1.0
    if blueprint_id:
        feat[AF_BP_HASH + bucket(blueprint_id, AF_BP_BUCKETS)] = 1.0
    feat[AF_ONES] = 1.0
    return feat


def _java_round(value: float) -> int:
    if value >= 0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def integer_samples(min_v: int, max_v: int, preferred: int) -> list[int]:
    """Match ``LinearPolicyAi.integerSamples`` (LinkedHashSet insertion order)."""
    span = max_v - min_v
    if span <= INTEGER_ENUM_CAP:
        return list(range(min_v, max_v + 1))
    values: list[int] = []

    def add(v: int) -> None:
        if v not in values:
            values.append(v)

    add(min_v)
    add(max_v)
    if preferred < min_v:
        preferred = min_v
    elif preferred > max_v:
        preferred = max_v
    add(preferred)
    for k in range(1, 7):
        add(min_v + _java_round(span * (k / 7.0)))
    return values


def _as_int(value: Any, default: int) -> int:
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, list):
        if not value:
            return default
        value = value[0]
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, list):
        value = value[0] if value else False
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() == "true"


def _norm(index: int, count: int) -> float:
    if count <= 1:
        return 0.0
    return index / float(count - 1)


def make_pack(init: str = "zeros", rng: random.Random | None = None) -> dict[str, Any]:
    """JSON object ``LinearPolicyAi.fromJson`` accepts.

    ``zeros`` matches ``LinearPolicyAi.zeros()`` (bagHashDim 16, bias 0, W all 0).
    Default gym LINEAR applies its own anti-stall prior when W is all zeros or
    omitted (real actions 0, non-zero integer up to +0.05, pass -0.05, activate-0
    / zero-integer -0.10), so a zeros pack no longer needs an external tiebreak
    file to avoid the activate-0 livelock. ``tiebreak`` is the older hand-built
    near-zero pack: pass +0.05 and integerNorm +0.05 stored in W. Those non-zero
    entries disable the Java prior. ``small-random`` draws N(0, 0.01) on action
    features 0..22 only. Packed, bag, bias, and the constant-ones feature stay 0
    because they do not change argmax.
    """
    weights = [0.0] * W_LEN
    base = PACKED_DIM + BAG_HASH_DIM
    if init == "small-random":
        draw = rng if rng is not None else random.SystemRandom()
        for j in range(AF_ONES):
            weights[base + j] = float(draw.gauss(0.0, 0.01))
    elif init == "tiebreak":
        # Documented probe, not a learned tactic. See module docstring.
        weights[base + AF_PASS] = 0.05
        weights[base + AF_INTEGER] = 0.05
    elif init != "zeros":
        raise ValueError(f"unknown init {init!r}")
    return {
        "schema": SCHEMA,
        "featureSchemaVersion": FEATURE_SCHEMA_VERSION,
        "packedDim": PACKED_DIM,
        "bagHashDim": BAG_HASH_DIM,
        "actionFeatDim": ACTION_FEAT_DIM,
        "bias": 0.0,
        "W": weights,
    }


def pack_weights(pack: dict[str, Any]) -> list[float]:
    w = pack.get("W")
    if not isinstance(w, list) or len(w) != W_LEN:
        raise ValueError(f"W must have length {W_LEN}")
    if pack.get("schema") != SCHEMA:
        raise ValueError("schema must be linear.v1")
    if int(pack.get("packedDim", -1)) != PACKED_DIM:
        raise ValueError("packedDim must be 128")
    if int(pack.get("bagHashDim", -1)) != BAG_HASH_DIM:
        raise ValueError("bagHashDim must be 16 for this trainer")
    if int(pack.get("actionFeatDim", -1)) != ACTION_FEAT_DIM:
        raise ValueError("actionFeatDim must be 24")
    return [float(x) for x in w]


def write_pack(path: Path, pack: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Round so the file stays readable; Java loads these as floats.
    out = dict(pack)
    out["W"] = [round(float(x), 6) for x in pack_weights(pack)]
    out["bias"] = float(out.get("bias") or 0.0)
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


def load_pack(path: Path) -> dict[str, Any]:
    pack = json.loads(path.read_text(encoding="utf-8"))
    pack_weights(pack)
    return pack


def _softmax(scores: list[float], temperature: float = 1.0) -> list[float]:
    temp = temperature if temperature > 0 else 1.0
    peak = max(scores)
    exps = [math.exp((s - peak) / temp) for s in scores]
    total = sum(exps)
    if total <= 0:
        n = len(scores)
        return [1.0 / n] * n
    return [e / total for e in exps]


def _action_slice_grad(feats: list[list[float]], chosen: int, weights: list[float]) -> list[float]:
    """d log softmax(scores) / d actionFeat weights, at ``weights``.

    Packed and bag dots are constant across actions, so they cancel. Scoring
    with the action slice alone matches the full concatenated score.
    """
    base = PACKED_DIM + BAG_HASH_DIM
    scores = []
    for feat in feats:
        scores.append(sum(weights[base + j] * feat[j] for j in range(ACTION_FEAT_DIM)))
    pi = _softmax(scores, 1.0)
    grad = [0.0] * ACTION_FEAT_DIM
    for j in range(AF_ONES):  # AF_ONES is 1 on every action; gradient is identically 0
        expected = sum(pi[b] * feats[b][j] for b in range(len(feats)))
        grad[j] = feats[chosen][j] - expected
    return grad


def _options_of(step: dict[str, Any]) -> dict[str, Any]:
    opts = step.get("options")
    return opts if isinstance(opts, dict) else {}


def _items_of(step: dict[str, Any]) -> list[dict[str, Any]]:
    items = _options_of(step).get("items")
    if not isinstance(items, list):
        return []
    return [it for it in items if isinstance(it, dict)]


def candidates_from_step(step: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    """Rebuild LinearPolicyAi candidates from a FEATURES/COMPACT trace step.

    Returns (candidates, status) where status is ``ok``, ``partial`` (truncated
    or otherwise not the full legal set), or ``skip`` (cannot rebuild).
    A candidate dict has raw/text/blueprint/passed/integer_norm/index_norm.
    """
    decision = str(step.get("decisionType") or "")
    opts = _options_of(step)
    items = _items_of(step)
    state = step.get("state") if isinstance(step.get("state"), dict) else {}
    truncated = bool(opts.get("truncated"))
    option_count = step.get("optionCount")
    try:
        option_count_i = int(option_count) if option_count is not None else len(items)
    except (TypeError, ValueError):
        option_count_i = len(items)

    def item_cand(i: int, n: int, raw: str, passed: bool = False, integer_norm: float = 0.0) -> dict[str, Any]:
        it = items[i] if 0 <= i < len(items) else {}
        text = str(it.get("text") or "")
        bp = str(it.get("blueprintId") or "")
        return {
            "raw": raw,
            "text": text,
            "blueprint": bp,
            "passed": passed or is_pass_text(text),
            "integer_norm": integer_norm,
            "index_norm": _norm(i, n),
        }

    cands: list[dict[str, Any]] = []
    status = "ok"

    if decision == "EMPTY":
        cands.append({
            "raw": "pass", "text": "Pass", "blueprint": "", "passed": True,
            "integer_norm": 0.0, "index_norm": 0.0,
        })
    elif decision == "MULTIPLE_CHOICE":
        if not items or truncated or (option_count_i > len(items)):
            return [], "partial" if items else "skip"
        for i in range(len(items)):
            cands.append(item_cand(i, len(items), str(i)))
    elif decision in ("ACTION_CHOICE", "CARD_ACTION_CHOICE"):
        if truncated or (items and option_count_i > len(items)):
            return [], "partial"
        n = len(items)
        for i in range(n):
            cands.append(item_cand(i, max(n, 1), str(i)))
        no_pass = _as_bool(opts.get("noPass"))
        if decision == "CARD_ACTION_CHOICE" and not no_pass:
            cands.append({
                "raw": "", "text": "Pass", "blueprint": "", "passed": True,
                "integer_norm": 0.0, "index_norm": 1.0,
            })
        if not cands:
            raw = "0" if decision == "ACTION_CHOICE" else ""
            cands.append({
                "raw": raw, "text": "Pass", "blueprint": "", "passed": True,
                "integer_norm": 0.0, "index_norm": 0.0,
            })
    elif decision == "INTEGER":
        if "min" in opts or "max" in opts:
            min_v = _as_int(opts.get("min"), 0)
            max_v = _as_int(opts.get("max"), min_v)
            preferred = _as_int(opts.get("defaultValue"), min_v)
        elif state.get("isActivateDecision"):
            min_v = _as_int(state.get("activateMin"), 0)
            max_v = _as_int(state.get("activateMax"), min_v)
            preferred = min_v
        else:
            return [], "skip"
        if max_v < min_v:
            max_v = min_v
        values = integer_samples(min_v, max_v, preferred)
        for i, value in enumerate(values):
            integer_norm = 0.0 if max_v == min_v else (value - min_v) / float(max_v - min_v)
            cands.append({
                "raw": str(value),
                "text": f"INTEGER {value}",
                "blueprint": "",
                "passed": False,
                "integer_norm": integer_norm,
                "index_norm": _norm(i, len(values)),
            })
    elif decision in ("CARD_SELECTION", "ARBITRARY_CARDS"):
        if truncated or "selectableCount" in opts:
            return [], "partial"
        selectable = opts.get("selectable")
        if isinstance(selectable, list) and len(selectable) not in (0, len(items)):
            return [], "partial"
        kept: list[int] = []
        for i in range(len(items)):
            if isinstance(selectable, list) and i < len(selectable) and not _as_bool(selectable[i]):
                continue
            if items[i].get("cardId") is None and not items[i].get("text"):
                continue
            kept.append(i)
        min_v = _as_int(opts.get("min"), 0)
        max_v = _as_int(opts.get("max"), len(items) if items else 0)
        if max_v < min_v:
            max_v = min_v
        if min_v == 0:
            cands.append({
                "raw": "", "text": "Pass", "blueprint": "", "passed": True,
                "integer_norm": 0.0, "index_norm": 0.0,
            })
        if not kept:
            status = "ok" if cands else "skip"
        elif min_v <= 1 and max_v >= 1:
            for order, i in enumerate(kept):
                it = items[i]
                raw = str(it.get("cardId") or "")
                text = str(it.get("text") or "")
                bp = str(it.get("blueprintId") or "")
                cands.append({
                    "raw": raw, "text": text, "blueprint": bp, "passed": is_pass_text(text),
                    "integer_norm": 0.0, "index_norm": _norm(order, len(kept)),
                })
        else:
            take = min(max(min_v, 1), min(max_v, len(kept)))
            raws = []
            texts = []
            for k in range(take):
                it = items[kept[k]]
                raws.append(str(it.get("cardId") or ""))
                texts.append(str(it.get("text") or ""))
            cands.append({
                "raw": ",".join(raws),
                "text": " ".join(texts),
                "blueprint": "",
                "passed": False,
                "integer_norm": 0.0,
                "index_norm": 0.0,
            })
        if option_count_i > len(items) and items:
            return [], "partial"
    else:
        return [], "skip"

    if len(cands) < 2:
        # No alternative: softmax gradient is identically 0.
        return [], "skip"
    return cands, status


def _packed_ok(step: dict[str, Any]) -> bool:
    state = step.get("state")
    if not isinstance(state, dict):
        return False
    packed = state.get("packed")
    if not isinstance(packed, list) or len(packed) != PACKED_DIM:
        return False
    return True


def _match_chosen(cands: list[dict[str, Any]], chosen: Any) -> int | None:
    raw = "" if chosen is None else str(chosen)
    if len(raw) >= 160:
        # Trace chosen is truncated at 160 chars; do not pretend it matched.
        return None
    for i, cand in enumerate(cands):
        if cand["raw"] == raw:
            return i
    return None


def aligned_decisions(step: dict[str, Any]) -> tuple[list[list[float]], int] | None:
    """Action-feature matrix and chosen index, or None if this step cannot train."""
    if step.get("accepted") is False:
        return None
    if str(step.get("aiSkill") or "") != "LINEAR":
        return None
    if not _packed_ok(step):
        return None
    cands, status = candidates_from_step(step)
    if status != "ok" or len(cands) < 2:
        return None
    chosen = _match_chosen(cands, step.get("chosen"))
    if chosen is None:
        return None
    feats = [
        action_features(c["text"], c["blueprint"], c["passed"], c["integer_norm"], c["index_norm"])
        for c in cands
    ]
    return feats, chosen


def seat_return(side: str, outcome: dict[str, Any]) -> tuple[float, float | None, float]:
    """Return (reward, lf_diff_from_seat, win_term).

    reward = win (+1/-1/0) + LF_MIX * clip((ownLF-oppLF)/30, -1, 1).
    Unfinished / missing LF contributes no life-force term.
    """
    winner = str(outcome.get("winner") or "")
    if winner == DARK_PLAYER:
        win = 1.0 if side == "DARK" else -1.0
    elif winner == LIGHT_PLAYER:
        win = 1.0 if side == "LIGHT" else -1.0
    else:
        win = 0.0
    dlf = outcome.get("darkLifeForce", outcome.get("darkLF"))
    llf = outcome.get("lightLifeForce", outcome.get("lightLF"))
    try:
        dlf_i = int(dlf)
        llf_i = int(llf)
    except (TypeError, ValueError):
        return win, None, win
    if dlf_i < 0 or llf_i < 0:
        return win, None, win
    diff = (dlf_i - llf_i) if side == "DARK" else (llf_i - dlf_i)
    lf_term = max(-1.0, min(1.0, diff / LF_SCALE))
    return win + LF_MIX * lf_term, float(diff), win


def _side_of(step: dict[str, Any]) -> str | None:
    side = str(step.get("side") or "").upper()
    if side in ("DARK", "LIGHT"):
        return side
    player = str(step.get("playerId") or "")
    if player == DARK_PLAYER:
        return "DARK"
    if player == LIGHT_PLAYER:
        return "LIGHT"
    return None


def update_from_games(
    pack: dict[str, Any],
    games: list[dict[str, Any]],
    lr: float,
) -> dict[str, Any]:
    """One batch update. Mutates ``pack['W']`` and returns a report dict."""
    weights = pack_weights(pack)
    seat_games = 0
    decisions_used = 0
    decisions_seen = 0
    decisions_skipped = 0
    delta = [0.0] * ACTION_FEAT_DIM
    dark_rewards: list[float] = []
    light_rewards: list[float] = []
    dark_lf: list[float] = []
    light_lf: list[float] = []
    dark_wins = 0
    light_wins = 0
    finished = 0

    for game in games:
        outcome = game.get("outcome") or {}
        winner = str(outcome.get("winner") or "")
        if outcome.get("finished") is False or winner not in (DARK_PLAYER, LIGHT_PLAYER):
            continue
        if winner in (DARK_PLAYER, LIGHT_PLAYER):
            finished += 1
            if winner == DARK_PLAYER:
                dark_wins += 1
            else:
                light_wins += 1
        for side in ("DARK", "LIGHT"):
            reward, lf_diff, _win = seat_return(side, outcome)
            if side == "DARK":
                dark_rewards.append(reward)
                if lf_diff is not None:
                    dark_lf.append(lf_diff)
            else:
                light_rewards.append(reward)
                if lf_diff is not None:
                    light_lf.append(lf_diff)
            grads: list[list[float]] = []
            for step in game.get("steps") or []:
                if _side_of(step) != side:
                    continue
                if str(step.get("aiSkill") or "") != "LINEAR":
                    continue
                decisions_seen += 1
                aligned = aligned_decisions(step)
                if aligned is None:
                    decisions_skipped += 1
                    continue
                feats, chosen = aligned
                grads.append(_action_slice_grad(feats, chosen, weights))
            if not grads:
                continue
            seat_games += 1
            decisions_used += len(grads)
            mean = [0.0] * ACTION_FEAT_DIM
            for g in grads:
                for j in range(ACTION_FEAT_DIM):
                    mean[j] += g[j] / len(grads)
            for j in range(AF_ONES):
                delta[j] += reward * mean[j]

    rule = UPDATE_STUB
    if seat_games:
        rule = UPDATE_REINFORCE
        base = PACKED_DIM + BAG_HASH_DIM
        for j in range(AF_ONES):
            weights[base + j] += lr * (delta[j] / seat_games)
            if weights[base + j] > WEIGHT_CLIP:
                weights[base + j] = WEIGHT_CLIP
            elif weights[base + j] < -WEIGHT_CLIP:
                weights[base + j] = -WEIGHT_CLIP
    else:
        # Episode stub. Signs are probes, not tactics. See LIMITATION.
        dark_m = sum(dark_rewards) / len(dark_rewards) if dark_rewards else 0.0
        light_m = sum(light_rewards) / len(light_rewards) if light_rewards else 0.0
        gap = dark_m - light_m
        base = PACKED_DIM + BAG_HASH_DIM
        probes = {AF_PASS: -0.05 * gap, AF_INTEGER: 0.05 * gap, AF_INDEX: 0.05 * gap}
        for j, step_size in probes.items():
            weights[base + j] += lr * step_size
            weights[base + j] = max(-WEIGHT_CLIP, min(WEIGHT_CLIP, weights[base + j]))

    pack["W"] = weights
    return {
        "updateRule": rule,
        "limitation": LIMITATION,
        "seatGamesUpdated": seat_games,
        "decisionsSeen": decisions_seen,
        "decisionsUsed": decisions_used,
        "decisionsSkipped": decisions_skipped,
        "finished": finished,
        "darkWins": dark_wins,
        "lightWins": light_wins,
        "darkWinRate": (dark_wins / finished) if finished else None,
        "lightWinRate": (light_wins / finished) if finished else None,
        "meanDarkReturn": (sum(dark_rewards) / len(dark_rewards)) if dark_rewards else None,
        "meanLightReturn": (sum(light_rewards) / len(light_rewards)) if light_rewards else None,
        "meanDarkLfDiff": (sum(dark_lf) / len(dark_lf)) if dark_lf else None,
        "meanLightLfDiff": (sum(light_lf) / len(light_lf)) if light_lf else None,
        "lr": lr,
        "promoted": False,
    }


def games_from_jsonl(path: Path) -> list[dict[str, Any]]:
    games: dict[Any, dict[str, Any]] = {}
    order: list[Any] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = row.get("gameIndex", row.get("gameId"))
            if key not in games:
                games[key] = {"steps": [], "outcome": None, "header": None}
                order.append(key)
            kind = row.get("type")
            if kind == "outcome":
                games[key]["outcome"] = row
            elif kind == "header":
                games[key]["header"] = row
            elif kind == "step" or row.get("decisionType"):
                games[key]["steps"].append(row)
    return [games[k] for k in order]


def synthetic_games(n: int = 4) -> list[dict[str, Any]]:
    """Tiny fixture so the update math runs without the JVM.

    Dark plays a non-pass action and wins; Light plays Pass and loses.
    These are not SWCCG games. Packed is zeros of length 128 so the
    FEATURES gate (packed present) is satisfied. Life force is invented.
    """
    packed = [0.0] * PACKED_DIM
    games = []
    for i in range(n):
        dlf = 20 + i
        llf = 5
        dark_step = {
            "type": "step",
            "gameIndex": i + 1,
            "side": "DARK",
            "playerId": DARK_PLAYER,
            "aiSkill": "LINEAR",
            "accepted": True,
            "decisionType": "ACTION_CHOICE",
            "chosen": "0",
            "optionCount": 2,
            "options": {
                "items": [
                    {"text": "Fire laser", "blueprintId": "1_2"},
                    {"text": "Pass", "blueprintId": ""},
                ]
            },
            "state": {"packed": packed, "schemaVersion": 1},
        }
        light_step = {
            "type": "step",
            "gameIndex": i + 1,
            "side": "LIGHT",
            "playerId": LIGHT_PLAYER,
            "aiSkill": "LINEAR",
            "accepted": True,
            "decisionType": "ACTION_CHOICE",
            "chosen": "1",
            "optionCount": 2,
            "options": {
                "items": [
                    {"text": "Fire laser", "blueprintId": "1_2"},
                    {"text": "Pass", "blueprintId": ""},
                ]
            },
            "state": {"packed": list(packed), "schemaVersion": 1},
        }
        games.append({
            "header": {"type": "header", "gameIndex": i + 1, "darkAi": "LINEAR", "lightAi": "LINEAR"},
            "steps": [dark_step, light_step],
            "outcome": {
                "type": "outcome",
                "gameIndex": i + 1,
                "winner": DARK_PLAYER,
                "finished": True,
                "darkLifeForce": dlf,
                "lightLifeForce": llf,
            },
        })
    return games


def find_classpath(explicit: Path | None = None) -> str | None:
    if explicit is not None:
        if not explicit.is_file():
            return None
        return explicit.read_text(encoding="utf-8").strip()
    env = os.environ.get("GEMP_CLASSPATH_FILE")
    path = Path(env) if env else DEFAULT_CP_FILE
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def run_live_batch(
    *,
    classpath: str,
    games: int,
    dark: str,
    light: str,
    weights: Path,
    csv_path: Path,
    traces_path: Path,
    log_path: Path,
    max_millis: int,
    max_decisions: int,
) -> dict[str, Any]:
    """One JVM batch. No shuffle seed is passed. FEATURES traces via -D."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "java",
        "-Xms256m",
        "-Xmx2048m",
        "-Dheadless.traceLevel=FEATURES",
        "-cp",
        classpath,
        MAIN,
        f"--games={games}",
        f"--dark={dark}",
        f"--light={light}",
        f"--linear-weights={weights.resolve()}",
        "--decks=wc96",
        "--format=premiere_anh",
        "--traces",
        f"--tracesPath={traces_path.resolve()}",
        "--no-replay",
        "--quiet",
        f"--maxMillis={max_millis}",
        f"--maxDecisions={max_decisions}",
        f"--csv={csv_path.resolve()}",
    ]
    timeout = max(60, int(games * (max_millis / 1000.0) + 90))
    proc = subprocess.run(
        cmd,
        cwd=str(GEMP_SERVER),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    log_path.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr, encoding="utf-8")
    return {
        "exit": proc.returncode,
        "timeoutSec": timeout,
        "csv": str(csv_path),
        "traces": str(traces_path),
        "log": str(log_path),
        "ok": proc.returncode in (0, 2) and csv_path.is_file() and traces_path.is_file(),
    }


def candidate_measurement(csv_path: Path) -> dict[str, Any]:
    """Win rate and mean LF differential from the LINEAR seat only.

    When both seats are LINEAR this is not a strength measurement against an
    opponent; ``vsOther`` stays empty and the caller should use side-split rates.
    """
    per_seat = {
        "DARK": {"games": 0, "wins": 0, "lf": []},
        "LIGHT": {"games": 0, "wins": 0, "lf": []},
    }
    if not csv_path.is_file():
        return {"candidateGames": 0, "candidateWinRate": None, "meanCandidateLfDiff": None}
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (row.get("error") or "").strip():
                continue
            dark_ai = (row.get("darkAi") or "").upper()
            light_ai = (row.get("lightAi") or "").upper()
            if dark_ai == "LINEAR" and light_ai == "LINEAR":
                continue
            if "darkLifeForce" not in row or "lightLifeForce" not in row:
                continue
            try:
                dlf = int(row["darkLifeForce"])
                llf = int(row["lightLifeForce"])
            except (TypeError, ValueError):
                continue
            winner = row.get("winner") or ""
            if dark_ai == "LINEAR":
                seat = per_seat["DARK"]
                seat["games"] += 1
                seat["wins"] += int(winner == DARK_PLAYER)
                if dlf >= 0 and llf >= 0:
                    seat["lf"].append(float(dlf - llf))
            elif light_ai == "LINEAR":
                seat = per_seat["LIGHT"]
                seat["games"] += 1
                seat["wins"] += int(winner == LIGHT_PLAYER)
                if dlf >= 0 and llf >= 0:
                    seat["lf"].append(float(llf - dlf))
    games = per_seat["DARK"]["games"] + per_seat["LIGHT"]["games"]
    wins = per_seat["DARK"]["wins"] + per_seat["LIGHT"]["wins"]
    lfs = per_seat["DARK"]["lf"] + per_seat["LIGHT"]["lf"]

    def seat_row(side: str) -> dict[str, Any]:
        s = per_seat[side]
        return {
            "games": s["games"],
            "wins": s["wins"],
            "winRate": (s["wins"] / s["games"]) if s["games"] else None,
            "meanLfDiff": (sum(s["lf"]) / len(s["lf"])) if s["lf"] else None,
        }

    return {
        "candidateGames": games,
        "candidateWins": wins,
        "candidateWinRate": (wins / games) if games else None,
        "meanCandidateLfDiff": (sum(lfs) / len(lfs)) if lfs else None,
        "asDark": seat_row("DARK"),
        "asLight": seat_row("LIGHT"),
    }


def summarize_batch_csv(csv_path: Path) -> dict[str, Any]:
    """Side-split win rate and mean life-force differential from a gym CSV."""
    dark_wins = light_wins = errors = rows = 0
    dark_lf_diffs: list[float] = []
    if not csv_path.is_file():
        return {"games": 0, "finished": 0, "errors": 0}
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows += 1
            if (row.get("error") or "").strip():
                errors += 1
                continue
            winner = row.get("winner") or ""
            if winner == DARK_PLAYER:
                dark_wins += 1
            elif winner == LIGHT_PLAYER:
                light_wins += 1
            if "darkLifeForce" in row and "lightLifeForce" in row:
                try:
                    dlf = int(row["darkLifeForce"])
                    llf = int(row["lightLifeForce"])
                except (TypeError, ValueError):
                    continue
                if dlf >= 0 and llf >= 0:
                    dark_lf_diffs.append(float(dlf - llf))
    finished = dark_wins + light_wins
    out: dict[str, Any] = {
        "games": rows,
        "finished": finished,
        "errors": errors,
        "darkWins": dark_wins,
        "lightWins": light_wins,
        "darkWinRate": (dark_wins / finished) if finished else None,
        "lightWinRate": (light_wins / finished) if finished else None,
        "meanDarkLfDiff": (sum(dark_lf_diffs) / len(dark_lf_diffs)) if dark_lf_diffs else None,
        "meanLightLfDiff": (-sum(dark_lf_diffs) / len(dark_lf_diffs)) if dark_lf_diffs else None,
    }
    return out


def _print_report(report: dict[str, Any]) -> None:
    print("linear.v1 self-play")
    print(f"  live: {report.get('live')}  source: {report.get('source')}")
    print(f"  update: {report.get('updateRule')}")
    print(f"  candidate: {report.get('candidate')}")
    print(f"  promoted: {report.get('promoted')}")
    if report.get("source") != "live":
        print("  note: win rate below is NOT a gym measurement")
    print(f"  games finished: {report.get('finished')}  errors: {report.get('errors')}")
    print(f"  dark-seat WR: {report.get('darkWinRate')}  light-seat WR: {report.get('lightWinRate')}")
    print(f"  mean LF diff dark-minus-light: {report.get('meanDarkLfDiff')}")
    print(f"  mean LF diff light-minus-dark: {report.get('meanLightLfDiff')}")
    cand = report.get("candidateMeasurement") or {}
    if cand.get("candidateGames"):
        print(
            f"  LINEAR seat WR: {cand.get('candidateWinRate')} "
            f"({cand.get('candidateWins')}/{cand.get('candidateGames')})"
        )
        print(f"  mean LF diff from LINEAR seat: {cand.get('meanCandidateLfDiff')}")
        print(f"  as Dark: {cand.get('asDark')}")
        print(f"  as Light: {cand.get('asLight')}")
    print(f"  decisions used/skipped: {report.get('decisionsUsed')}/{report.get('decisionsSkipped')}")
    if report.get("liveError"):
        print(f"  live error: {report.get('liveError')}")
    if report.get("gate"):
        print(f"  gate (not promoted): {json.dumps(report.get('gate'))}")
    print("  limitation:")
    print("   " + LIMITATION)


def run(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="One linear.v1 self-play update. Does not promote. Does not set a gym seed.",
    )
    parser.add_argument("--games", type=int, default=4, help="headless games when --live (default 4)")
    parser.add_argument("--opponent", choices=("LINEAR", "BEGINNER"), default="BEGINNER",
                        help="BEGINNER finishes games. LINEAR vs LINEAR with tiebreak passes every phase and does not finish")
    parser.add_argument("--both-seats-vs-beginner", action="store_true",
                        help="with BEGINNER, also run the pack as Light (doubles games)")
    parser.add_argument("--init", choices=("zeros", "small-random", "tiebreak"), default="tiebreak",
                        help="zeros uses the gym anti-stall prior (no external tiebreak file); tiebreak bakes pass/integer +0.05 into W")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--live", action="store_true", help="run headless gym games (random shuffle, no seed)")
    parser.add_argument("--dry-run", action="store_true", help="synthetic outcomes only; no JVM")
    parser.add_argument("--gate", action="store_true",
                        help="optional N=2 vs Beginner measurement; never promotes")
    parser.add_argument("--max-millis", type=int, default=120_000)
    parser.add_argument("--max-decisions", type=int, default=4000,
                        help="cap a livelock before the JVM runs out of heap")
    parser.add_argument("--classpath-file", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.games < 1:
        parser.error("--games must be >= 1")
    if args.dry_run and args.live:
        parser.error("pass only one of --dry-run and --live")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.out_dir or (REPO / "runs" / "linear-selfplay" / stamp)
    run_dir.mkdir(parents=True, exist_ok=True)

    pack = make_pack(args.init)
    init_path = run_dir / "init.linear.json"
    write_pack(init_path, pack)

    classpath = find_classpath(args.classpath_file)
    want_live = args.live or (not args.dry_run and classpath is not None)
    if args.dry_run:
        want_live = False

    live_meta: dict[str, Any] | None = None
    live_error: str | None = None
    games: list[dict[str, Any]]
    source = "synthetic"

    if want_live:
        if not classpath:
            live_error = "no gym classpath (runs/wc96-loop/gym.classpath or GEMP_CLASSPATH_FILE)"
            games = synthetic_games(args.games)
            source = "synthetic-fallback"
        else:
            try:
                games, live_meta = _play_live(args, classpath, run_dir, init_path)
                source = "live"
            except (OSError, subprocess.SubprocessError, json.JSONDecodeError, RuntimeError) as exc:
                live_error = f"{type(exc).__name__}: {exc}"
                games = synthetic_games(args.games)
                source = "synthetic-fallback"
    else:
        games = synthetic_games(args.games)
        syn_path = run_dir / "synthetic.jsonl"
        _write_synthetic_jsonl(syn_path, games)

    # Reload init so a failed partial write cannot leak into the update.
    pack = load_pack(init_path)
    stats = update_from_games(pack, games, args.lr)
    pack["trainer"] = {
        "name": "linear_selfplay",
        "updateRule": stats["updateRule"],
        "limitation": LIMITATION,
        "source": source,
        "opponent": args.opponent,
        "init": args.init,
        "gamesRequested": args.games,
        "promoted": False,
        "distilledFrom": None,
    }
    cand_path = run_dir / "candidate.linear.json"
    write_pack(cand_path, pack)

    csv_stats: dict[str, Any] = {}
    if live_meta and live_meta.get("csv") and Path(live_meta["csv"]).is_file():
        csv_stats = summarize_batch_csv(Path(live_meta["csv"]))

    report: dict[str, Any] = {
        **stats,
        "live": source == "live",
        "source": source,
        "liveError": live_error,
        "candidate": str(cand_path),
        "init": str(init_path),
        "runDir": str(run_dir),
        "opponent": args.opponent,
        "errors": (csv_stats.get("errors") if csv_stats else 0),
        "promoted": False,
    }
    # Prefer gym CSV for the strength printout when it has rows.
    if live_meta and live_meta.get("csv"):
        report["candidateMeasurement"] = candidate_measurement(Path(live_meta["csv"]))
    if csv_stats.get("finished"):
        report["finished"] = csv_stats["finished"]
        report["darkWins"] = csv_stats["darkWins"]
        report["lightWins"] = csv_stats["lightWins"]
        report["darkWinRate"] = csv_stats["darkWinRate"]
        report["lightWinRate"] = csv_stats["lightWinRate"]
        report["meanDarkLfDiff"] = csv_stats["meanDarkLfDiff"]
        report["meanLightLfDiff"] = csv_stats["meanLightLfDiff"]
        report["errors"] = csv_stats.get("errors")
    if live_meta:
        report["jvm"] = {k: live_meta[k] for k in ("exit", "log", "traces", "csv") if k in live_meta}

    if args.gate:
        report["gate"] = _maybe_gate(classpath, cand_path, run_dir, args.max_millis)

    (run_dir / "update_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _print_report(report)
    return report


def _write_synthetic_jsonl(path: Path, games: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for game in games:
            if game.get("header"):
                handle.write(json.dumps(game["header"]) + "\n")
            for step in game.get("steps") or []:
                handle.write(json.dumps(step) + "\n")
            if game.get("outcome"):
                handle.write(json.dumps(game["outcome"]) + "\n")


def _play_live(
    args: argparse.Namespace,
    classpath: str,
    run_dir: Path,
    weights: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Play on WC96. Opponent LINEAR uses one batch. BEGINNER is Dark=LINEAR unless both seats."""
    batches: list[tuple[str, str, str]] = []
    if args.opponent == "LINEAR":
        batches.append(("LINEAR", "LINEAR", "selfplay"))
    else:
        batches.append(("LINEAR", "BEGINNER", "vs-beginner-as-dark"))
        if args.both_seats_vs_beginner:
            batches.append(("BEGINNER", "LINEAR", "vs-beginner-as-light"))

    merged: list[dict[str, Any]] = []
    last_meta: dict[str, Any] = {}
    csvs: list[Path] = []
    for dark, light, label in batches:
        csv_path = run_dir / f"{label}.csv"
        traces_path = run_dir / f"{label}.jsonl"
        log_path = run_dir / f"{label}.log"
        meta = run_live_batch(
            classpath=classpath,
            games=args.games,
            dark=dark,
            light=light,
            weights=weights,
            csv_path=csv_path,
            traces_path=traces_path,
            log_path=log_path,
            max_millis=args.max_millis,
            max_decisions=args.max_decisions,
        )
        last_meta = meta
        if not meta["ok"]:
            tail = ""
            if log_path.is_file():
                text = log_path.read_text(encoding="utf-8", errors="replace")
                tail = text[-1500:]
            raise RuntimeError(f"jvm exit {meta['exit']} label={label} tail={tail}")
        merged.extend(games_from_jsonl(traces_path))
        csvs.append(csv_path)
    if len(csvs) > 1:
        _concat_csv(csvs, run_dir / "games.csv")
        last_meta = dict(last_meta)
        last_meta["csv"] = str(run_dir / "games.csv")
    return merged, last_meta


def _concat_csv(parts: list[Path], dest: Path) -> None:
    header = None
    rows: list[list[str]] = []
    for part in parts:
        with part.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            h = next(reader, None)
            if h is None:
                continue
            if header is None:
                header = h
            for row in reader:
                if row:
                    rows.append(row)
    with dest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if header:
            writer.writerow(header)
        writer.writerows(rows)


def _maybe_gate(classpath: str | None, weights: Path, run_dir: Path, max_millis: int) -> dict[str, Any]:
    """N=2 vs Beginner (1 as Dark, 1 as Light). Measurement only — never promotes."""
    if not classpath:
        return {"ran": False, "reason": "no classpath", "promoted": False}
    gate_dir = run_dir / "gate-beginner"
    gate_dir.mkdir(parents=True, exist_ok=True)
    pieces = []
    for dark, light, label in (
        ("LINEAR", "BEGINNER", "as-dark"),
        ("BEGINNER", "LINEAR", "as-light"),
    ):
        meta = run_live_batch(
            classpath=classpath,
            games=1,
            dark=dark,
            light=light,
            weights=weights,
            csv_path=gate_dir / f"{label}.csv",
            traces_path=gate_dir / f"{label}.jsonl",
            log_path=gate_dir / f"{label}.log",
            max_millis=max_millis,
            max_decisions=4000,
        )
        # Gate does not need FEATURES traces; leave the files, do not train on them.
        stats = summarize_batch_csv(gate_dir / f"{label}.csv") if meta["ok"] else {"ok": False, "exit": meta["exit"]}
        stats["label"] = label
        stats["ok"] = meta["ok"]
        pieces.append(stats)
    return {"ran": True, "nPerSeat": 1, "nTotal": 2, "promoted": False, "seats": pieces}


def main() -> None:
    try:
        run()
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
