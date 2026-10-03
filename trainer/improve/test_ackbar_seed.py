"""Yoda keyword alignment onto linear.v1 action weights. No JVM."""
from __future__ import annotations

import unittest

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
    AF_PASS,
    BAG_HASH_DIM,
    PACKED_DIM,
    W_LEN,
    pack_weights,
)


class AlignmentTest(unittest.TestCase):
    def test_feature_order_matches_linear_policy(self) -> None:
        names = action_feature_names()
        self.assertEqual(len(names), 24)
        self.assertEqual(names[0], "pass")
        self.assertEqual(names[1], "integerNorm")
        self.assertEqual(names[2], "indexNorm")
        self.assertEqual(names[3:19], [f"textHash{i}" for i in range(16)])
        self.assertEqual(names[19:23], [f"blueprintHash{i}" for i in range(4)])
        self.assertEqual(names[23], "ones")

    def test_only_pass_keyword_maps_and_stays_negative(self) -> None:
        base = yoda_action_base()
        self.assertEqual(base[AF_PASS], -160.0)
        self.assertEqual(sum(1 for v in base if v != 0.0), 1)
        self.assertLess(base[AF_PASS], 0.0)
        action_pass = dict(YODA_ACTION_KEYWORDS)["pass"]
        self.assertEqual(action_pass, -160)
        self.assertEqual(ACTION_FEATURE_KEYWORD, {AF_PASS: "pass"})

    def test_unmapped_keywords_are_named(self) -> None:
        rows = unmapped_keywords()
        keys = {(row["table"], row["keyword"]) for row in rows}
        self.assertNotIn(("action", "pass"), keys)
        self.assertIn(("choice", "pass"), keys)
        self.assertIn(("action", "force drain"), keys)
        self.assertIn(("action", "sacrifice"), keys)
        self.assertIn(("choice", "yes"), keys)
        mapped_action = {kw for kw, _ in YODA_ACTION_KEYWORDS} - {"pass"}
        self.assertEqual(mapped_action, {row["keyword"] for row in rows if row["table"] == "action"})
        self.assertEqual(len([r for r in rows if r["table"] == "choice"]), len(YODA_CHOICE_KEYWORDS))

    def test_hash_and_norm_features_stay_zero(self) -> None:
        bare = features_without_keyword()
        indexes = {row["index"] for row in bare}
        self.assertEqual(indexes, set(range(ACTION_FEAT_DIM)) - {AF_PASS})
        self.assertEqual(len(bare), 23)

    def test_scale_multiplies_pass_only(self) -> None:
        pack = make_seed_pack(1.25)
        weights = pack_weights(pack)
        base = PACKED_DIM + BAG_HASH_DIM
        self.assertEqual(pack["schema"], "linear.v1")
        self.assertEqual(pack["bias"], 0.0)
        self.assertEqual(pack["seed"]["schema"], "ackbar-seed.v1")
        self.assertEqual(pack["seed"]["source"], "yoda-advanced-keywords")
        self.assertEqual(pack["seed"]["scale"], 1.25)
        self.assertEqual(pack["seed"]["packed"], "zeros")
        self.assertEqual(pack["seed"]["bag"], "zeros")
        self.assertEqual(pack["seed"]["action"], "yoda * scale")
        self.assertEqual(len(weights), W_LEN)
        self.assertTrue(all(v == 0.0 for v in weights[:base]))
        self.assertEqual(weights[base + AF_PASS], -200.0)
        self.assertTrue(all(v == 0.0 for i, v in enumerate(weights[base:]) if i != AF_PASS))

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
