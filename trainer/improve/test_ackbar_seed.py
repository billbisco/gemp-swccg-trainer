"""Yoda keyword alignment onto linear.v1 action weights. No JVM."""
from __future__ import annotations

import unittest
from pathlib import Path

from trainer.improve.ackbar_seed import (
    ACTION_FEATURE_KEYWORD,
    SCALES,
    YODA_ACTION_KEYWORDS,
    YODA_CHOICE_KEYWORDS,
    action_feature_names,
    features_without_keyword,
    make_seed_pack,
    pick_winner,
    unmapped_keywords,
    yoda_action_base,
)
from trainer.improve.linear_selfplay import (
    ACTION_FEAT_DIM,
    AF_KIND,
    AF_ONES,
    AF_PASS,
    BAG_HASH_DIM,
    KIND_KEYWORDS,
    PACKED_DIM,
    W_LEN,
    pack_weights,
)


class AlignmentTest(unittest.TestCase):
    def test_feature_order_matches_linear_policy(self) -> None:
        names = action_feature_names()
        self.assertEqual(len(names), ACTION_FEAT_DIM)
        self.assertEqual(ACTION_FEAT_DIM, 50)
        self.assertEqual(names[0], "pass")
        self.assertEqual(names[1], "integerNorm")
        self.assertEqual(names[2], "indexNorm")
        self.assertEqual(names[3:19], [f"textHash{i}" for i in range(16)])
        self.assertEqual(names[19:23], [f"blueprintHash{i}" for i in range(4)])
        self.assertEqual(names[23], "ones")
        self.assertEqual(names[AF_KIND:], [f"kind:{name}" for name in KIND_KEYWORDS])

    def test_every_action_keyword_maps_and_pass_stays_negative(self) -> None:
        base = yoda_action_base()
        self.assertEqual(base[AF_PASS], 0.0)
        self.assertEqual(base[AF_ONES], 0.0)
        self.assertEqual(sum(1 for v in base if v != 0.0), len(KIND_KEYWORDS))
        self.assertEqual(dict(YODA_ACTION_KEYWORDS)["pass"], -160)
        self.assertEqual(base[AF_KIND + KIND_KEYWORDS.index("pass")], -160.0)
        self.assertEqual(base[AF_KIND + KIND_KEYWORDS.index("force drain")], 160.0)
        self.assertEqual(base[AF_KIND + KIND_KEYWORDS.index("sacrifice")], -120.0)
        self.assertEqual(set(ACTION_FEATURE_KEYWORD.values()), set(KIND_KEYWORDS))
        self.assertLess(base[AF_KIND + KIND_KEYWORDS.index("pass")], 0.0)

    def test_scores_match_advanced_ai_java(self) -> None:
        java = Path("/workspace/swccg-gemp/src/gemp-swccg-server/src/main/java/com/gempukku/swccgo/ai/models/AdvancedAi.java")
        source = java.read_text(encoding="utf-8")
        def grab(const: str) -> list[tuple[str, int]]:
            start = source.index(f"{const} = new KeywordWeight[]")
            body = source[start:source.index("};", start)]
            return [(kw, int(score)) for kw, score in __import__("re").findall(r'new KeywordWeight\("([^"]+)", (-?\d+)\)', body)]
        self.assertEqual(list(YODA_ACTION_KEYWORDS), grab("ACTION_WEIGHTS") + grab("ACTION_PENALTIES"))
        self.assertEqual(list(YODA_CHOICE_KEYWORDS), grab("CHOICE_WEIGHTS") + grab("CHOICE_PENALTIES"))

    def test_unmapped_keywords_are_the_choice_table(self) -> None:
        rows = unmapped_keywords()
        keys = {(row["table"], row["keyword"]) for row in rows}
        self.assertNotIn(("action", "pass"), keys)
        self.assertNotIn(("action", "force drain"), keys)
        self.assertIn(("choice", "pass"), keys)
        self.assertIn(("choice", "yes"), keys)
        self.assertEqual(len([r for r in rows if r["table"] == "action"]), 0)
        self.assertEqual(len([r for r in rows if r["table"] == "choice"]), len(YODA_CHOICE_KEYWORDS))

    def test_hash_and_norm_features_stay_zero(self) -> None:
        bare = features_without_keyword()
        indexes = {row["index"] for row in bare}
        self.assertEqual(indexes, set(range(AF_KIND)))
        self.assertEqual(len(bare), AF_ONES + 1)
        self.assertNotIn(AF_KIND, indexes)

    def test_scale_multiplies_every_kind_and_nothing_else(self) -> None:
        pack = make_seed_pack(1.25)
        weights = pack_weights(pack)
        base = PACKED_DIM + BAG_HASH_DIM
        self.assertEqual(pack["schema"], "linear.v1")
        self.assertEqual(pack["actionFeatDim"], ACTION_FEAT_DIM)
        self.assertEqual(pack["bias"], 0.0)
        self.assertEqual(pack["seed"]["schema"], "ackbar-seed.v1")
        self.assertEqual(pack["seed"]["source"], "yoda-advanced-keywords")
        self.assertEqual(pack["seed"]["scale"], 1.25)
        self.assertEqual(pack["seed"]["packed"], "zeros")
        self.assertEqual(pack["seed"]["bag"], "zeros")
        self.assertEqual(pack["seed"]["action"], "yoda * scale")
        self.assertEqual(len(weights), W_LEN)
        self.assertTrue(all(v == 0.0 for v in weights[:base]))
        self.assertEqual(weights[base + AF_PASS], 0.0)
        self.assertEqual(weights[base + AF_ONES], 0.0)
        self.assertEqual(weights[base + AF_KIND + KIND_KEYWORDS.index("pass")], -200.0)
        self.assertEqual(weights[base + AF_KIND + KIND_KEYWORDS.index("force drain")], 200.0)
        self.assertTrue(all(v == 0.0 for i in range(AF_KIND) for v in [weights[base + i]]))

    def test_scale_zero_is_all_zeros(self) -> None:
        pack = make_seed_pack(0.0)
        self.assertTrue(all(v == 0.0 for v in pack_weights(pack)))
        self.assertEqual(pack["seed"]["scale"], 0.0)

    def test_scales_cover_the_agreed_ladder(self) -> None:
        self.assertEqual(SCALES, (0.0, 0.125, 0.25, 0.5, 1.0, 1.25, 1.5, 2.0))


class WinnerTest(unittest.TestCase):
    def test_none_keeps_zeros(self) -> None:
        rows = [
            {"scale": 1.0, "beatsBoth": False, "pooledMeanLfDiff": 5.0, "path": "a"},
            {"scale": 2.0, "beatsBoth": False, "pooledMeanLfDiff": 9.0, "path": "b"},
        ]
        winner = pick_winner(rows)
        self.assertEqual(winner["kept"], "zeros")
        self.assertEqual(winner["scale"], 0.0)

    def test_both_seats_picks_better_pooled_lf(self) -> None:
        rows = [
            {"scale": 0.5, "beatsBoth": True, "pooledMeanLfDiff": 1.0, "path": "half"},
            {"scale": 2.0, "beatsBoth": True, "pooledMeanLfDiff": 3.5, "path": "two"},
            {"scale": 1.0, "beatsBoth": False, "pooledMeanLfDiff": 100.0, "path": "no"},
        ]
        winner = pick_winner(rows)
        self.assertEqual(winner["kept"], "seed")
        self.assertEqual(winner["scale"], 2.0)
        self.assertEqual(winner["path"], "two")

    def test_lf_tie_keeps_lower_scale(self) -> None:
        rows = [
            {"scale": 1.5, "beatsBoth": True, "pooledMeanLfDiff": 2.0, "path": "high"},
            {"scale": 0.25, "beatsBoth": True, "pooledMeanLfDiff": 2.0, "path": "low"},
        ]
        winner = pick_winner(rows)
        self.assertEqual(winner["scale"], 0.25)


if __name__ == "__main__":
    unittest.main()
