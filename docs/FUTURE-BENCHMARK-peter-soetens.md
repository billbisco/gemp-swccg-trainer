# Future benchmark (not integrated)

Peter Soetens (Slack): browser-side model ~90% win rate vs Advanced on ~380
random legal decks; ~800k games/day; beta PR expected this weekend; LLM path
abandoned.

**Do not merge yet.** Use as an external quality ceiling / throughput reference
for Bill's WC96 heuristic ladder once a PR exists. Our current loop stays on
`heuristic.v1` weight search + honest Maven gates (CSV primary, xml.gz replay
off/sparse for throughput).
