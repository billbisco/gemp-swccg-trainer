# WC96 re-gate heuristic-v1-wc96-batch50-cand1

- N=25/side vs builtin BEGINNER
- passed=False
- combined WR=0.46 expected=0.5
- darkDelta=-0.020000000000000018 lightDelta=-0.06000000000000005
- reason: FAIL: dark 0.32 (base 0.34, d=-0.020000000000000018), light 0.6 (base 0.66, d=-0.06000000000000005), combined 0.460 vs expected 0.500+0.05
- Soft prior gate was N=8; this is the larger confirmation.
- Future benchmark (not used): Peter Soetens browser model ~90% vs Advanced (~380 decks); not integrated yet.

## Decision

Soft N=8 promote did **not** hold at N=25/side. Pack remains search seed /
CURRENT pointer for continuity, but is **not** treated as proven > BEGINNER.
Round improve must beat builtin BEGINNER (side baselines 0.34/0.66) to promote.
Replay stays off for throughput.
