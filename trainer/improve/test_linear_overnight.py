"""Gate math for the overnight loop. No JVM."""
from __future__ import annotations

import unittest
from pathlib import Path

from trainer.improve.linear_overnight import (
    EVEN_BAR,
    FORBIDDEN,
    KEYWORD_PACK_REL,
    KEYWORD_SEAT,
    PROMOTION_RULE,
    YODA_SEAT,
    assert_safe_out_dir,
    decide_keep,
    ensure_scale1_seed,
    gate_weight_pair,
    holds_real_keywords,
    learned_opponent_bar,
    measure_seat_csv,
    next_round_index,
    run_limits,
    seat_beats,
    seat_not_worse,
    select_finished,
    should_promote,
)
from trainer.improve.ackbar_seed import make_seed_pack
from trainer.improve.linear_selfplay import make_pack
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

    def test_winrate_drop_does_not_reject_when_lf_holds(self) -> None:
        base = _meas(_seat(4, 2, 1.0), _seat(4, 2, 1.0))
        cand = _meas(_seat(4, 1, 5.0), _seat(4, 3, 5.0))
        ok, detail = should_promote(cand, base, min_games=2)
        self.assertTrue(ok)
        self.assertNotIn("winRate", detail["asDark"])

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

    def test_scores_cap_as_loss_for_deciding_seat(self) -> None:
        games = [
            {
                "outcome": {
                    "gameIndex": 4,
                    "finished": False,
                    "winner": "",
                    "stopper": "maxDecisions=8000",
                    "decidingPlayer": DARK_PLAYER,
                    "darkLifeForce": 22,
                    "lightLifeForce": 18,
                }
            }
        ]
        kept = select_finished(games, {4})
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["outcome"]["winner"], LIGHT_PLAYER)
        self.assertEqual(kept[0]["outcome"]["darkLifeForce"], 0)
        self.assertEqual(kept[0]["outcome"]["lightLifeForce"], 30)
        # Real board life force is not the score, and two different caps match.
        self.assertNotEqual(22, kept[0]["outcome"]["darkLifeForce"])



class HeadToHeadGateTest(unittest.TestCase):
    def test_even_split_promotes(self) -> None:
        cand = _meas(_seat(4, 2, 0.0), _seat(4, 2, 0.0))
        ok, detail = should_promote(cand, EVEN_BAR, min_games=2)
        self.assertTrue(ok)
        self.assertIn("not worse", detail["asDark"])
        self.assertIn("not worse", detail["asLight"])

    def test_below_even_winrate_still_keeps_when_lf_holds(self) -> None:
        cand = _meas(_seat(4, 1, 1.0), _seat(4, 2, 1.0))
        ok, detail = should_promote(cand, EVEN_BAR, min_games=2)
        self.assertTrue(ok)
        self.assertNotIn("winRate", detail["asDark"])

    def test_negative_lf_rejects(self) -> None:
        cand = _meas(_seat(4, 2, 0.0), _seat(4, 3, -0.1))
        ok, _detail = should_promote(cand, EVEN_BAR, min_games=2)
        self.assertFalse(ok)

    def test_rule_is_life_force_vs_previous_linear_pack(self) -> None:
        self.assertIn("previous kept linear pack", PROMOTION_RULE)
        self.assertIn("current.linear.json", PROMOTION_RULE)
        self.assertIn("seat LINEAR", PROMOTION_RULE)
        self.assertIn("not worse", PROMOTION_RULE)
        self.assertIn("Win rate is logged, not a bar", PROMOTION_RULE)
        self.assertIn("no YodaBot/ADVANCED gate", PROMOTION_RULE)
        self.assertIn("no HEURISTIC gate", PROMOTION_RULE)
        self.assertIn("not Beginner", PROMOTION_RULE)
        self.assertIn("zeros pack is not a gate", PROMOTION_RULE)
        self.assertIn("deciding seat", PROMOTION_RULE)
        self.assertIn("0 vs 30", PROMOTION_RULE)
        self.assertNotIn("heuristic-v1-wc96-r08-blend-adv25", PROMOTION_RULE)
        self.assertNotIn("vs BEGINNER", PROMOTION_RULE)
        self.assertEqual(YODA_SEAT, "ADVANCED")
        self.assertEqual(KEYWORD_SEAT, "HEURISTIC")
        self.assertTrue(str(KEYWORD_PACK_REL).endswith(
            "champs/_promoted/heuristic-v1-wc96-r08-blend-adv25/weights.json"
        ))

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
            "gameIndex,darkAi,lightAi,winner,error,darkLifeForce,lightLifeForce,decidingPlayer\n"
            "1,LINEAR,LINEAR,~OzzelBot,,10,4,\n"
            "2,LINEAR,LINEAR,~AckbarBot,,0,6,\n"
            "3,LINEAR,LINEAR,,maxDecisions=8000,1,1,\n"
            "4,LINEAR,LINEAR,,maxMillis=180000,40,10,~OzzelBot\n",
            encoding="utf-8",
        )
        dark = measure_seat_csv(path, "DARK")
        light = measure_seat_csv(path, "LIGHT")
        # Row 3 has no deciding seat, so it still drops.
        # Row 4: Dark was deciding, so Dark loses with LF 0 vs 30 (not the raw 40 vs 10).
        self.assertEqual(dark["games"], 3)
        self.assertEqual(dark["wins"], 1)
        self.assertEqual(dark["winRate"], 1 / 3)
        self.assertEqual(dark["meanLfDiff"], -10.0)  # (6 + -6 + -30) / 3
        self.assertEqual(light["wins"], 2)
        self.assertEqual(light["meanLfDiff"], 10.0)


class OpponentGateTest(unittest.TestCase):
    def _both(self, wins, lf, games=4):
        return _meas(_seat(games, wins, lf), _seat(games, wins, lf))

    def _meas_opponents(self, yoda, keyword):
        return {"ok": True, "opponents": {"yoda": yoda, "keyword": keyword}}

    def test_first_keep_requires_positive_lf_vs_keyword_only(self) -> None:
        meas = self._meas_opponents(self._both(0, -5.0), self._both(0, 2.0))
        ok, detail = decide_keep(meas, None, min_games=2)
        self.assertTrue(ok)
        self.assertEqual(set(detail), {"keyword.asDark", "keyword.asLight"})
        self.assertTrue(all("mean LF" in why for why in detail.values()))

    def test_first_keep_tie_is_not_enough(self) -> None:
        tied = self._meas_opponents(self._both(4, 9.0), self._both(2, 0.0))
        ok, detail = decide_keep(tied, None, min_games=2)
        self.assertFalse(ok)
        self.assertTrue(any("meanLfDiff" in why for why in detail.values()))
        self.assertFalse(any("winRate" in why for why in detail.values()))

    def test_first_keep_fails_if_one_keyword_seat_does_not_beat(self) -> None:
        yoda = self._both(3, 1.0)
        keyword = _meas(_seat(4, 0, 1.0), _seat(4, 0, 0.0))
        ok, detail = decide_keep(self._meas_opponents(yoda, keyword), None, min_games=2)
        self.assertFalse(ok)
        self.assertIn("meanLfDiff", detail["keyword.asLight"])

    def test_equality_with_previous_learned_pack_keeps(self) -> None:
        block = self._both(1, -2.0)
        meas = self._meas_opponents(block, block)
        ok, detail = decide_keep(meas, meas["opponents"], min_games=2)
        self.assertTrue(ok)
        self.assertTrue(all("not worse" in why for why in detail.values()))

    def test_worse_keyword_lf_rejects_even_if_yoda_block_is_worse(self) -> None:
        prev = self._meas_opponents(self._both(2, 1.0), self._both(2, 1.0))
        cand = self._meas_opponents(self._both(0, -9.0), _meas(_seat(4, 2, 1.0), _seat(4, 2, 0.5)))
        ok, detail = decide_keep(cand, prev["opponents"], min_games=2)
        self.assertFalse(ok)
        self.assertIn("meanLfDiff", detail["keyword.asLight"])
        self.assertNotIn("yoda.asLight", detail)

    def test_too_few_finished_games_does_not_keep(self) -> None:
        block = self._both(2, 5.0, games=1)
        ok, detail = decide_keep(self._meas_opponents(block, block), None, min_games=2)
        self.assertFalse(ok)
        self.assertIn("finished games", detail["keyword.asDark"])

    def test_learned_bar_ignores_zeros_baseline(self) -> None:
        self.assertIsNone(learned_opponent_bar({"role": "baseline", "promoted": False, "metrics": None}))
        self.assertIsNone(learned_opponent_bar({"role": "ackbar-seed", "promoted": False}))
        good, why = seat_beats(_seat(4, 2, 0.0), EVEN_BAR["asDark"], min_games=2)
        self.assertFalse(good)
        self.assertIn("<=", why)

    def test_advanced_flag_is_linear_weights_only(self) -> None:
        cand = Path("/tmp/cand.linear.json")
        dark, light = gate_weight_pair(cand, "ADVANCED", None, "DARK")
        flags = linear_weight_cli(cand, dark_weights=dark, light_weights=light)
        self.assertEqual(flags, ["--linear-weights=/tmp/cand.linear.json"])
        self.assertFalse(any("BEGINNER" in arg or "zeros" in arg for arg in flags))

    def test_heuristic_flag_puts_keyword_file_on_opponent_seat(self) -> None:
        cand = Path("/tmp/cand.linear.json")
        keyword = Path("/tmp/heuristic-weights.json")
        dark, light = gate_weight_pair(cand, "HEURISTIC", keyword, "LIGHT")
        flags = linear_weight_cli(cand, dark_weights=dark, light_weights=light)
        self.assertEqual(
            flags,
            [
                "--dark-weights=/tmp/heuristic-weights.json",
                "--light-weights=/tmp/cand.linear.json",
            ],
        )

    def test_scale1_seed_holds_keywords_and_pass_only_does_not(self) -> None:
        seed = make_seed_pack(1.0)
        self.assertTrue(holds_real_keywords(seed))
        self.assertFalse(holds_real_keywords(make_pack("zeros")))
        self.assertFalse(holds_real_keywords(make_pack("tiebreak")))
        pass_only = make_pack("zeros")
        from trainer.improve.linear_selfplay import AF_KIND as kind
        from trainer.improve.linear_selfplay import KIND_KEYWORDS as names
        from trainer.improve.linear_selfplay import PACKED_DIM, BAG_HASH_DIM
        pass_only["W"][PACKED_DIM + BAG_HASH_DIM + kind + names.index("pass")] = -160.0
        self.assertFalse(holds_real_keywords(pass_only))

    def test_missing_scale1_is_rewritten_from_advanced_scores(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path, note = ensure_scale1_seed(Path(tmp))
            self.assertTrue(path.is_file())
            self.assertIn("rewrote", note)
            pack = __import__("json").loads(path.read_text(encoding="utf-8"))
            self.assertTrue(holds_real_keywords(pack))


if __name__ == "__main__":
    unittest.main()


class LoopBoundTest(unittest.TestCase):
    def test_zero_rounds_and_hours_do_not_stop(self) -> None:
        last, hours = run_limits(0, 0.0)
        self.assertIsNone(last)
        self.assertIsNone(hours)
        last, hours = run_limits(0, -1.0)
        self.assertIsNone(last)
        self.assertIsNone(hours)

    def test_positive_caps_still_parse(self) -> None:
        self.assertEqual(run_limits(40, 8.0), (40, 8.0))

    def test_resume_skips_existing_round_dirs(self) -> None:
        self.assertEqual(next_round_index(0, list(range(1, 41))), 41)
        self.assertEqual(next_round_index(7, []), 8)
