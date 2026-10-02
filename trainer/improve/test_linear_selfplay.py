"""Unit tests for the linear.v1 self-play update. No JVM."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from trainer.improve.linear_selfplay import (
    AF_INDEX,
    AF_INTEGER,
    AF_ONES,
    AF_PASS,
    BAG_HASH_DIM,
    PACKED_DIM,
    UPDATE_REINFORCE,
    UPDATE_STUB,
    W_LEN,
    action_features,
    bucket,
    integer_samples,
    java_string_hash,
    make_pack,
    pack_weights,
    synthetic_games,
    update_from_games,
    write_pack,
)


class LinearSelfPlayTest(unittest.TestCase):
    def test_java_hash_matches_jshell(self) -> None:
        # Checked against jshell String.hashCode on 2026-10-02.
        self.assertEqual(java_string_hash("pass"), 3433489)
        self.assertEqual(java_string_hash("integer 3"), 492449233)
        self.assertEqual(java_string_hash(""), 0)
        self.assertEqual(bucket("pass", 16), 3433489 % 16)

    def test_zeros_pack_matches_loader_contract(self) -> None:
        pack = make_pack("zeros")
        self.assertEqual(pack["schema"], "linear.v1")
        self.assertEqual(pack["featureSchemaVersion"], 1)
        self.assertEqual(pack["packedDim"], 128)
        self.assertEqual(pack["bagHashDim"], 16)
        self.assertEqual(pack["actionFeatDim"], 24)
        self.assertEqual(pack["bias"], 0.0)
        weights = pack_weights(pack)
        self.assertEqual(len(weights), W_LEN)
        self.assertEqual(W_LEN, PACKED_DIM + BAG_HASH_DIM + 24)
        self.assertTrue(all(v == 0.0 for v in weights))

    def test_action_features_pass_and_ones(self) -> None:
        feat = action_features("Pass", "", True, 0.0, 1.0)
        self.assertEqual(feat[AF_PASS], 1.0)
        self.assertEqual(feat[AF_ONES], 1.0)
        self.assertEqual(sum(feat[3:19]), 1.0)

    def test_integer_samples_wide_span_includes_ends(self) -> None:
        values = integer_samples(0, 100, 7)
        self.assertEqual(values[0], 0)
        self.assertIn(100, values)
        self.assertIn(7, values)
        self.assertLessEqual(len(values), 9)

    def test_reinforce_moves_action_slice_not_packed(self) -> None:
        pack = make_pack("zeros")
        report = update_from_games(pack, synthetic_games(4), lr=0.1)
        self.assertEqual(report["updateRule"], UPDATE_REINFORCE)
        self.assertGreater(report["decisionsUsed"], 0)
        weights = pack["W"]
        self.assertTrue(all(v == 0.0 for v in weights[: PACKED_DIM + BAG_HASH_DIM]))
        base = PACKED_DIM + BAG_HASH_DIM
        # Synthetic: Dark fires (non-pass) and wins; Light passes and loses.
        # Both seat returns push the pass weight down.
        self.assertLess(weights[base + AF_PASS], 0.0)
        self.assertEqual(weights[base + AF_ONES], 0.0)
        self.assertFalse(report["promoted"])
        # Dark won every synthetic game.
        self.assertEqual(report["darkWinRate"], 1.0)
        self.assertGreater(report["meanDarkLfDiff"], 0)

    def test_stub_when_traces_have_no_aligned_decisions(self) -> None:
        pack = make_pack("zeros")
        games = [
            {
                "steps": [],
                "outcome": {
                    "winner": "~OzzelBot",
                    "darkLifeForce": 30,
                    "lightLifeForce": 0,
                },
            }
        ]
        report = update_from_games(pack, games, lr=0.1)
        self.assertEqual(report["updateRule"], UPDATE_STUB)
        base = PACKED_DIM + BAG_HASH_DIM
        weights = pack["W"]
        moved = [AF_PASS, AF_INTEGER, AF_INDEX]
        for j in range(24):
            if j in moved:
                self.assertNotEqual(weights[base + j], 0.0)
            else:
                self.assertEqual(weights[base + j], 0.0)
        self.assertTrue(all(v == 0.0 for v in weights[:base]))

    def test_write_roundtrip(self) -> None:
        pack = make_pack("zeros")
        path = Path("/tmp/linear-v1-roundtrip.json")
        write_pack(path, pack)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(loaded["W"]), W_LEN)
        self.assertEqual(loaded["schema"], "linear.v1")



class CandidateMeasurementTest(unittest.TestCase):
    def test_linear_seat_only(self) -> None:
        from trainer.improve.linear_selfplay import candidate_measurement
        path = Path("/tmp/linear-meas.csv")
        path.write_text(
            "gameIndex,darkAi,lightAi,winner,error,darkLifeForce,lightLifeForce\n"
            "1,LINEAR,BEGINNER,~AckbarBot,,0,44\n"
            "1,BEGINNER,LINEAR,~AckbarBot,,0,36\n",
            encoding="utf-8",
        )
        stats = candidate_measurement(path)
        self.assertEqual(stats["candidateGames"], 2)
        self.assertEqual(stats["candidateWins"], 1)
        self.assertEqual(stats["candidateWinRate"], 0.5)
        self.assertEqual(stats["asDark"]["meanLfDiff"], -44.0)
        self.assertEqual(stats["asLight"]["meanLfDiff"], 36.0)


if __name__ == "__main__":
    unittest.main()
