# heuristic.v1 candidate from WC96 batch-50

## Collect
- 50 WC96 Beginner vs Beginner games under `runs/wc96-batch-50/`
- Dark WR=0.340, Light WR=0.660 (deck asymmetry; not learning)
- avg decisions=1330.0, avg duration ms=4995.1
- stopReason=target_games_reached_50

## Improve
- Method: random Gaussian mutate of Beginner weights + directed Advanced blend candidate
- Selected candidate id: `blend-adv45`
- **Collection ≠ learning** — mutants ranked by a cheap trace keyword proxy then live-gated in the JVM.

## Gate
- Candidate as Dark vs builtin BEGINNER Light (8 games)
- Builtin BEGINNER Dark vs candidate as Light (8 games)
- Baseline side rates from collect batch used as the bar (not raw 0.5)
- **passed=True**
- reason: improved vs side baselines (dark 0.5 vs 0.34, light 0.75 vs 0.66; combined 0.625 vs expected 0.500)
- combined WR=0.625, expected=0.5, eloEst=1588.7394998465425
- darkDelta=0.15999999999999998, lightDelta=0.08999999999999997

## Honesty
If gate failed, baseline Beginner remains the champ. This pack is a candidate artifact only.
