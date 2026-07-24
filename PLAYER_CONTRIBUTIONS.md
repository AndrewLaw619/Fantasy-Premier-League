# Attributing team strength scores to individual players

`player_contributions.py` answers the question "which players are driving this
team's attack / defence / overall score, and by how much?"

## What the team scores actually are

`teams.csv` carries six numbers per team:

| Column | Meaning |
| --- | --- |
| `strength_attack_home` / `_away` | attacking rating |
| `strength_defence_home` / `_away` | defensive rating |
| `strength_overall_home` / `_away` | overall rating |

They sit on a roughly 950–1400 scale and move in steps of 5. They are set
editorially by FPL and updated during the season; they are **not** computed from
player data. So there is no formula to invert and no direct decomposition — a
team's rating is one opaque integer, and there are only 20 of them per season.

## The approach

Rebuild each score from player-level gameweek data, check how much of the
published score the rebuild explains, then decompose the rebuild — which is
additive over players by construction — and price each player's share back in
strength points.

Pooling seasons naively destroys the fit, because FPL re-bases the ladder from
one season to the next. The slope is therefore fitted on seasons demeaned within
season, then the level is anchored to the season being analysed.

How well the rebuild tracks the published scores (2022-23 to 2025-26):

| Score | Driver | Within-season R² |
| --- | --- | --- |
| overall | xG difference per match | **0.82** |
| attack | xG per match | 0.52 |
| defence | xGC per match | 0.47 |

The overall rating is essentially expected goal difference on a rescaled axis.
The attack and defence legs are noisier — FPL's split does not cleanly separate
the two the way the underlying data does.

## Attack: exact, no model needed

xG and xA are owned by individuals, so a team's attacking rate is literally the
sum of its players' rates. The split is arithmetic, not inference.

Credit is measured **above positional replacement level** — the 20th percentile
per-90 rate among players at that position with 450+ minutes — so a forward does
not outrank a defender purely by playing further up the pitch.

## Defence: ridge adjusted plus-minus

Every player on the pitch shares the same conceded-xG reading, so per-player
defensive rates are nearly identical within a squad. Ranking on them directly
just ranks minutes played; in testing that put Salah third among Liverpool's
defensive contributors.

The fix is an adjusted plus-minus fitted at fixture level: one row per team per
fixture, target = xG that team generated, regressors = every player's on-pitch
fraction, entered as an attacking effect for the team in possession and a
defensive effect for the team facing them, plus a home term. Ridge handles the
collinearity, with the penalty chosen by cross-validation over fixtures.

The coefficients are additive: intercept plus the eleven on-pitch coefficients
reconstruct a team's conceded xG for a match, which is what makes the split
legitimate rather than a heuristic.

## Reading the output

Attributed points are measured above a replacement-level baseline, so each
team's published score splits into a squad floor plus what individuals added:

```
=== Liverpool ===  (2024-25)
  published attack 1340 = 891 baseline + 449 attributed to players
  Mohamed Salah    141 attack points  (31% of everything Liverpool's players added)
  Luis Díaz         62
  Szoboszlai        45
```

## Known limits

- **Teammates who never rotate are not separable.** A back four that starts
  every match gives the model no variation to work with, and ridge shrinks them
  toward a common value. Rotation, injury and substitution are what identify
  individual defensive effects.
- **Attack and defence points are not like-for-like.** Attacking credit is
  measured directly; defensive credit is shrunk by regularisation, so defensive
  totals are systematically smaller. Compare within a dimension, using the
  `*_share_pct` columns, not across them.
- **Only 2022-23 onward.** FPL did not publish `expected_*` columns before then.
- **Attacking credit misses build-up play.** xG+xA rewards the shooter and the
  assister and nobody else. `data/<season>/understat/` carries `xGChain` and
  `xGBuildup`, which credit every player in the possession — the natural next
  upgrade, gated on joining Understat IDs to FPL IDs (see `match_ids` in
  `understat.py`).

## Usage

```bash
python player_contributions.py --season 2024-25
python player_contributions.py --season 2024-25 --team Arsenal
python player_contributions.py --season 2024-25 --out contributions.csv
```
