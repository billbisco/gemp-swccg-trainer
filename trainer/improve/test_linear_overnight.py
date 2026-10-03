"""Gate math for the overnight loop. No JVM."""
from __future__ import annotations

import unittest
from pathlib import Path

from trainer.improve.linear_overnight import (
    EVEN_BAR,
    FORBIDDEN,
    PROMOTION_RULE,
    assert_safe_out_dir,
    measure_seat_csv,
    seat_not_worse,
    select_finished,
    should_promote,
)
from trainer.improve.linear_selfplay import DARK_PLAYER, LIGHT_PLAYER, linear_weight_cli


def _seat(games, wins, lf):
    return {
        "games": games,
        "wins": wins,
        "winRate": (wins / games) if games else None,
        "meanLfDiff": lf,
    }


def _meas(dark, light):
    return {"asDark": dark, "asLight": light}


class PromotionTest(unittest.TestCase):
    def test_equal_both_seats_promotes(self) -> None:
        base = _meas(_seat(4, 2, 1.0), _seat(4, 1, -2.0))
        ok, detail = should_promote(base, base, min_games=2)
        self.assertTrue(ok)
        self.assertIn("not worse", detail["asDark"])
        self.assertIn("not worse", detail["asLight"])

    def test_one_seat_winrate_drop_rejects(self) -> None:
        base = _meas(_seat(4, 2, 1.0), _seat(4, 2, 1.0))
        cand = _meas(_seat(4, 1, 5.0), _seat(4, 3, 5.0))
        ok, detail = should_promote(cand, base, min_games=2)
        self.assertFalse(ok)
        self.assertIn("winRate", detail["asDark"])

    def test_lf_drop_rejects_even_if_winrate_holds(self) -> None:
        base = _meas(_seat(4, 2, 3.0), _seat(4, 2, 3.0))
        cand = _meas(_seat(4, 2, 3.0), _seat(4, 2, 2.5))
        ok, _detail = should_promote(cand, base, min_games=2)
        self.assertFalse(ok)

    def test_both_metrics_up_on_both_seats_promotes(self) -> None:
        base = _meas(_seat(4, 1, -1.0), _seat(4, 1, -1.0))
        cand = _meas(_seat(4, 2, 0.0), _seat(4, 3, 4.0))
        ok, _detail = should_promote(cand, base, min_games=2)
        self.assertTrue(ok)

    def test_missing_metric_does_not_promote(self) -> None:
        base = _meas(_seat(4, 1, None), _seat(4, 1, 1.0))
        cand = _meas(_seat(4, 2, 1.0), _seat(4, 2, 1.0))
        ok, _detail = should_promote(cand, base, min_games=2)
        self.assertFalse(ok)

    def test_too_few_finished_gate_games(self) -> None:
        good, why = seat_not_worse(_seat(1, 1, 10.0), _seat(4, 0, -10.0), min_games=2)
        self.assertFalse(good)
        self.assertIn("finished games", why)

    def test_refuses_champs_promoted_dir(self) -> None:
        with self.assertRaises(SystemExit):
            assert_safe_out_dir(FORBIDDEN / "heuristic-v1-wc96-r08-blend-adv25")
        with self.assertRaises(SystemExit):
            assert_safe_out_dir(Path("/workspace/gemp-swccg-trainer/champs"))


class FinishedFilterTest(unittest.TestCase):
    def test_skips_max_decisions(self) -> None:
        games = [
            {
                "outcome": {
                    "gameIndex": 1,
                    "finished": False,
                    "winner": "",
                    "stopper": "maxDecisions=8000",
                    "darkLifeForce": 20,
                    "lightLifeForce": 39,
                }
            },
            {
                "outcome": {
                    "gameIndex": 2,
                    "finished": True,
                    "winner": DARK_PLAYER,
                    "stopper": "",
                    "darkLifeForce": 10,
                    "lightLifeForce": 0,
                }
            },
            {
                "outcome": {
                    "gameIndex": 3,
                    "finished": True,
                    "winner": LIGHT_PLAYER,
                    "stopper": "maxMillis",
                }
            },
        ]
        kept = select_finished(games, {2})
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["outcome"]["winner"], DARK_PLAYER)



class HeadToHeadGateTest(unittest.TestCase):
    def test_even_split_promotes(self) -> None:
        cand = _meas(_seat(4, 2, 0.0), _seat(4, 2, 0.0))
        ok, detail = should_promote(cand, EVEN_BAR, min_games=2)
        self.assertTrue(ok)
        self.assertIn("not worse", detail["asDark"])
        self.assertIn("not worse", detail["asLight"])

    def test_below_even_winrate_rejects(self) -> None:
        cand = _meas(_seat(4, 1, 1.0), _seat(4, 2, 1.0))
        ok, detail = should_promote(cand, EVEN_BAR, min_games=2)
        self.assertFalse(ok)
        self.assertIn("winRate", detail["asDark"])

    def test_negative_lf_rejects(self) -> None:
        cand = _meas(_seat(4, 2, 0.0), _seat(4, 3, -0.1))
        ok, _detail = should_promote(cand, EVEN_BAR, min_games=2)
        self.assertFalse(ok)

    def test_rule_names_kept_pack_not_beginner_opponent(self) -> None:
        self.assertIn("previous kept AckbarBot", PROMOTION_RULE)
        self.assertIn("No Beginner", PROMOTION_RULE)
        self.assertNotIn("vs BEGINNER", PROMOTION_RULE)

    def test_same_file_is_one_flag_split_files_are_per_seat(self) -> None:
        same = linear_weight_cli(Path("/tmp/current.linear.json"))
        self.assertEqual(same, ["--linear-weights=/tmp/current.linear.json"])
        split = linear_weight_cli(
            Path("/tmp/candidate.linear.json"),
            dark_weights=Path("/tmp/candidate.linear.json"),
            light_weights=Path("/tmp/current.linear.json"),
        )
        self.assertEqual(
            split,
            [
                "--dark-weights=/tmp/candidate.linear.json",
                "--light-weights=/tmp/current.linear.json",
            ],
        )
        self.assertFalse(any("BEGINNER" in arg for arg in split))

    def test_measure_seat_scores_candidate_side_and_skips_caps(self) -> None:
        path = Path("/tmp/gate-seat.csv")
        path.write_text(
            "gameIndex,darkAi,lightAi,winner,error,darkLifeForce,lightLifeForce\n"
            "1,LINEAR,LINEAR,~OzzelBot,,10,4\n"
            "2,LINEAR,LINEAR,~AckbarBot,,0,6\n"
            "3,LINEAR,LINEAR,,maxDecisions=8000,1,1\n",
            encoding="utf-8",
        )
        dark = measure_seat_csv(path, "DARK")
        light = measure_seat_csv(path, "LIGHT")
        self.assertEqual(dark["games"], 2)
        self.assertEqual(dark["wins"], 1)
        self.assertEqual(dark["winRate"], 0.5)
        self.assertEqual(dark["meanLfDiff"], 0.0)  # (10-4) + (0-6) = 0
        self.assertEqual(light["wins"], 1)
        self.assertEqual(light["meanLfDiff"], 0.0)


if __name__ == "__main__":
    unittest.main()
