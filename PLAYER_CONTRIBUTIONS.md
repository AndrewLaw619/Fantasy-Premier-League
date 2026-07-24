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
one season to the next. Both the blend weight and the slope are therefore fitted
on seasons demeaned within season, then the level is anchored to the season being
analysed.

## Realised output, not just expected

The published scores respond to what teams actually did, not only to how good
their chances were. Fitting the weight on realised output rather than assuming
it beats using expected output alone on every score:

| Score | Driver per match | Weight on realised | R² blended | R² expected only |
| --- | --- | --- | --- | --- |
| overall | goal difference | 0.52 | **0.89** | 0.82 |
| attack | goal involvements | 0.38 | 0.61 | 0.52 |
| defence | goals conceded | 0.52 | 0.51 | 0.47 |

So a striker who scores twenty and a striker who scores none separate on the
realised half of the blend, not only on chance quality. In 2024-25 that moves
Salah — 29 goals from 24.7 xG, 18 assists from 9.0 xA — from 80 to 95 attacking
points, while Cameron Archer's 2 goals from 5.7 xG drop him from +2.7 to -3.3.

The overall rating is essentially blended goal difference on a rescaled axis.
The attack and defence legs are noisier — FPL's split does not cleanly separate
the two the way the underlying data does.

## Attack: exact, no model needed

Goals, assists, xG and xA are all owned by individuals, so a team's blended
involvement rate is literally the sum of its players' rates. The split is
arithmetic, not inference. Involvements rather than goals is what makes this
exact — a team's involvement total is the sum of its players' involvements,
where a goals-only driver would need an apportionment step for assists.

Credit is measured **above positional replacement level** — the 20th percentile
per-90 rate among players at that position with 450+ minutes — so a forward does
not outrank a defender purely by playing further up the pitch.

## Defence: ridge adjusted plus-minus

Every player on the pitch shares the same conceded reading, so per-player
defensive rates are nearly identical within a squad. Ranking on them directly
just ranks minutes played; in testing that put Salah third among Liverpool's
defensive contributors.

The fix is an adjusted plus-minus fitted at fixture level: one row per team per
fixture, target = blended goals and xG that team generated, regressors = every
player's on-pitch fraction, entered as an attacking effect for the team in
possession and a defensive effect for the team facing them, plus a home term.
Ridge handles the collinearity, with the penalty chosen by cross-validation over
fixtures.

The coefficients are additive: intercept plus the eleven on-pitch coefficients
reconstruct what a team conceded in a match, which is what makes the split
legitimate rather than a heuristic.

## Overall: calibrated, not summed

`overall_points` is not `attack_points + defence_points` — those two sit on
different ladders and adding them would be meaningless. Attacking and defensive
contributions are both in goal-difference units before scaling, so the overall
score is priced from their sum with the overall slope, against the
best-fitting calibration of the three (R² = 0.89).

## Reading the output

Attributed points are measured above a replacement-level baseline, so each
team's published score splits into a squad floor plus what individuals added:

```
=== Liverpool ===  (2024-25)
  published overall 1360 = 1086 baseline + 274 attributed to players
                        GI/90  xGI/90   attack   defence   overall
  Mohamed Salah          1.25    0.90     94.6       5.1      83.3
  Luis Díaz              0.75    0.61     39.5      10.6      38.2
  Szoboszlai             0.58    0.47     29.2       8.0      28.3
  Alexander-Arnold       0.38    0.34     23.2      12.9      25.1
```

The `GI/90` against `xGI/90` gap is the finishing premium the blend picks up.

## Known limits

- **Teammates who never rotate are not separable.** A back four that starts
  every match gives the model no variation to work with, and ridge shrinks them
  toward a common value. Rotation, injury and substitution are what identify
  individual defensive effects.
- **Attack and defence points are not like-for-like.** Attacking credit is
  measured directly; defensive credit is shrunk by regularisation, so defensive
  totals are systematically smaller. Compare within a dimension, using the
  `*_share_pct` columns, not across them. `overall_points` is the column to use
  when you want one number per player.
- **Realised output is noisier than expected output.** The fitted weights sit
  near the middle for a reason: goals reflect finishing skill and luck in
  unknown proportion. A single season of over-performance is partly the latter,
  so these are contribution shares for a season that happened, not forecasts.
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
