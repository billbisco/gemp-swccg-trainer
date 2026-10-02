# Sample WC96 headless replay (real GEMP xml.gz)

Completed **P-ANH 1996 World Champion (Dark)** vs **(Light)** BeginnerAi headless game
(`format=premiere_anh`) with `HeadlessReplayWriter` output.

| Field | Value |
|-------|-------|
| Winner | `~OzzelBot` (Dark) |
| Decks | P-ANH 1996 World Champion Dark / Light |
| Format | `premiere_anh` |
| Decisions | ~1031 |
| Elapsed | ~8.0 s |
| Dark replay | `~OzzelBot/z2lvxyskmomnws1c.xml.gz` |
| Light replay | `~AckbarBot/m8pchsostwv5wq7g.xml.gz` |

## Open in local GEMP Replay viewer

1. Copy into your GEMP `application.root`:

```bash
GEMP_APP_ROOT=/path/to/gemp/app   # directory that contains replays/
mkdir -p "$GEMP_APP_ROOT/replays/~OzzelBot" "$GEMP_APP_ROOT/replays/~AckbarBot"
cp runs/_sample/wc96-replay/~OzzelBot/z2lvxyskmomnws1c.xml.gz \
   "$GEMP_APP_ROOT/replays/~OzzelBot/"
cp runs/_sample/wc96-replay/~AckbarBot/m8pchsostwv5wq7g.xml.gz \
   "$GEMP_APP_ROOT/replays/~AckbarBot/"
```

2. With local GEMP web up, open:

- Dark POV: `game.html?replayId=~OzzelBot$z2lvxyskmomnws1c`
- Light POV: `game.html?replayId=~AckbarBot$m8pchsostwv5wq7g`

No history DB row is required — `ReplayRequestHandler` serves the file directly.

## Live demo path (local, gitignored)

After a Maven WC96 run the fresh copy also lands at:

`runs/wc96-demo/replays/game-0001/`
