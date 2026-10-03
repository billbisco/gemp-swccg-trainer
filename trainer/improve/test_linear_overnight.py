"""Gate math for the overnight loop. No JVM."""
from __future__ import annotations

import unittest
from pathlib import Path

from trainer.improve.linear_overnight import (
    FORBIDDEN,
    assert_safe_out_dir,
    seat_not_worse,
    select_finished,
    should_promote,
)
from trainer.improve.linear_selfplay import DARK_PLAYER, LIGHT_PLAYER


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


if __name__ == "__main__":
    unittest.main()
