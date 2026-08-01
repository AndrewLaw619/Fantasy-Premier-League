# Season simulation engine — implementation plan

Simulate a Premier League season repeatedly, from historical team results and
player performances, to produce distributions of FPL points per player.

---

## 1. The architecture question: top-down or bottom-up

Your instinct is bottom-up, and it is right about the *causal* direction. But
taken literally — simulate each player's events independently, then add them up
— it breaks in three ways:

- **Scorelines stop being coherent.** Independent per-player goal draws do not
  sum to a plausible team score, and nothing stops both teams "winning."
- **Clean sheets contradict themselves.** A defender's clean sheet and the goals
  conceded drawn for his teammate are the same event. Drawn separately they
  disagree.
- **Teammate correlation vanishes.** A 4–0 win means several teammates return
  together. Independent draws produce that only by coincidence, so the *spread*
  of a squad's joint outcomes — exactly what matters for captaincy and
  differentials — comes out far too narrow.

Pure top-down has the opposite flaw: simulate the result then allocate, and the
players no longer determine the outcome, which is the thing you actually want to
vary.

**Recommendation: hierarchical — causally bottom-up, structurally top-down
within each match.**

```
players on the pitch  ->  team attack/defence rates for THIS fixture
                      ->  coherent scoreline drawn from those rates
                      ->  goals and assists allocated back to those players
                      ->  peripheral events, bonus, points
```

Players drive the result (bottom-up, as you wanted); the match is drawn once as
a joint object so everything downstream is mutually consistent. Swap a striker
out and the team's scoring rate falls, the scoreline distribution shifts, and
every teammate's returns move with it.

This also reuses what we already built: the ridge APM in `player_contributions.py`
produces exactly the needed object — team conceded/created rates as an
**additive function of who is on the pitch**, intercept plus the eleven on-pitch
coefficients.

---

## 2. Layer stack

Each layer conditions on the one above — the "events stacking on top of each
other" you described.

| Layer | Produces | Method | Data |
| --- | --- | --- | --- |
| 0 | availability, minutes | 3-state Markov + injury hazard | `starts`, `minutes`, `chance_of_playing_*`, `news` |
| 1 | fixture team rates λ_home, λ_away | lineup-additive ratings + opponent + home | APM ratings, `teams.csv` |
| 2 | scoreline | bivariate Poisson, Dixon-Coles low-score correction | 10 seasons of results |
| 3 | goals, assists → players | multinomial on xG / xA shares | per-90 player rates |
| 4 | saves, cards, penalties, own goals, defensive contributions | per-player count models | `gws/merged_gw.csv` |
| 5 | BPS → bonus 3/2/1 | BPS from simulated events, ranked within match | `bps`, fixture `stats` blob |
| 6 | FPL points | rules engine | validated, see §4 |

### Layer 0 — availability and minutes

**This is the highest-leverage component and the one public FPL models most
often skip.** A player's points variance is dominated by whether they play
2,900 minutes or 900, not by their per-90 rate.

Model each player-gameweek as one of `absent / cameo / start`, as a Markov chain
(last week's state predicts this week's strongly), with:
- an injury hazard that enters `absent` and holds for a drawn spell length
- a rotation propensity per player, fitted from start-sequence runs
- fixture congestion as a covariate (midweek European football, `kickoff_time`)
- `chance_of_playing_next_round` and `news` as the in-season override

### Layer 1 — fixture team rates

Dixon-Coles form, with lineup dependence:

```
λ_home = exp( attack(home XI) - defence(away XI) + home_advantage )
```

where `attack(XI)` is the sum of on-pitch player attack coefficients — already
what the APM returns. Home advantage measured at **+0.19 xG** in the earlier
work. Promoted teams need a prior (see §6).

### Layer 2 — scoreline

Independent Poisson under-predicts 0–0, 1–0, 0–1 and 1–1. Apply the standard
Dixon-Coles τ correction on those four cells, plus a correlation term. Fit by
maximum likelihood on ~3,800 historical matches. Optionally add time-decay
weighting so recent seasons count more.

### Layer 3 — allocating goals and assists

Given the team scored *k*: draw scorers multinomially with weights equal to each
on-pitch player's share of the team's xG. Then per goal, draw whether it was
assisted (measure the base rate from the fixture `stats` blobs), and if so draw
the assister from xA shares, excluding the scorer.

This is the step that makes team quality flow to players correctly, and it
matches what the transfer work found: **a player carries their share of chances;
the team supplies the conversion.**

### Layer 4 — peripherals

- **Saves** — Poisson, mean tied to the opponent's shot volume, so saves and
  goals conceded stay correlated (a keeper facing a barrage gets both).
- **Penalties** — use `penalties_order`, which is in `players_raw.csv`. Penalty
  duty is a step function that no rate model recovers; it must be explicit.
- **Cards** — Bernoulli/Poisson per 90 with team and (if sourced) referee effects.
- **Defensive contribution** — the new 2025-26 term, and a threshold effect, so
  the *distribution* matters more than the mean. Model CBIT/CBIRT per 90 as
  negative binomial (over-dispersed relative to Poisson) and apply the threshold
  per draw. Fitting a mean and comparing it to the threshold would be wrong.

### Layer 5 — bonus

BPS is a deterministic function of match events. Compute it from the simulated
events, rank within the match, award 3/2/1 with FPL's tie rules. Validate
against the real `bps` column, which is per player per match in the data.

---

## 3. Directory layout

```
season_sim/
  PLAN.md
  season_sim/
    scoring.py        # FPL rules -> points
    ratings.py        # player ratings, reusing player_contributions
    availability.py   # minutes and injury model
    match.py          # one fixture -> events
    season.py         # fixture list -> N simulated seasons
    calibrate.py      # fit every parameter from history
    backtest.py       # holdout evaluation and reliability
    cli.py
  tests/
  reports/            # calibration output, figures
```

---

## 4. Scoring rules — already verified

Rather than trust a published summary, the rules were **reverse-engineered from
2025-26 results and reconstruct `total_points` exactly** (every residual is 0,
or exactly the +2 defensive-contribution award):

| Event | Points |
| --- | --- |
| played 1–59 min / 60+ min | 1 / 2 |
| goal — GK, DEF / MID / FWD | 6 / 5 / 4 |
| assist | 3 |
| clean sheet — GK, DEF / MID | 4 / 1 |
| every 3 saves | 1 |
| penalty saved / missed | 5 / −2 |
| every 2 goals conceded (GK, DEF) | −1 |
| yellow / red | −1 / −3 |
| own goal | −2 |
| bonus | 3 / 2 / 1 |
| **defensive contribution** (new 2025-26) | **2** |

Defensive contribution thresholds, confirmed against the data:

- **DEF: 10+** where the stat is CBIT (clearances + blocks + interceptions + tackles)
- **MID and FWD: 12+** where the stat is CBIRT (the above plus recoveries)
- **GK: not eligible** — zero awards in the season

Sanity note: the GK goal value was never exercised by that check, because no
goalkeeper scored in 2025-26. It is set to 6, matching DEF.

The exact reconstruction means `scoring.py` can ship first with a test that
replays every historical gameweek and asserts a zero residual. Everything above
it is then measured against a known-correct scorer.

---

## 5. Backtesting

Hold out whole seasons. Never tune on the season being scored.

**Baselines to beat** — a simulator that cannot beat these is not earning its cost:
1. last season's points per 90, carried forward
2. **FPL's own `xP` column**, which is already in `merged_gw.csv` — a genuine,
   non-trivial baseline sitting in the data
3. season-long average by position and price

**Metrics, by level:**

| Level | Metric |
| --- | --- |
| match | ranked probability score on 1X2; log-loss on exact scoreline |
| team | Brier score on clean sheets; calibration of goals conceded |
| player-gameweek | log-loss on returned/blanked; reliability diagram of P(goal) |
| player-season | coverage — does the 50/80/95 band contain the actual total? |
| ranking | Spearman correlation of projected against actual season points |

**Calibration is the thing to check, not accuracy.** A simulator whose 80% bands
cover 55% of outcomes is worse than useless for FPL decisions, even with a good
mean. Every reported interval gets a coverage check, exactly as the transfer
projection does now (nominal 80% covering 78% measured).

Guard against the obvious traps: no using a season's own team ratings to
simulate that season, no fitting the DC threshold on the test year, and promoted
teams scored separately since they have no prior PL data.

---

## 6. Known gaps and risks

| Risk | Mitigation |
| --- | --- |
| **xG data only from 2022-23** — ~3.5 seasons for anything player-level | Team-level layers use all 10 seasons of results; player layers accept the shorter window and shrink hard |
| **Defensive contribution is one season old** and is a large scoring term | Flag DC-derived projections as low-confidence; re-fit as 2026-27 accrues |
| **Promoted teams have no PL history** | Prior from historical promoted-team performance, widened; do not pretend to precision |
| **New signings from abroad have no PL data** | Reuse the transfer-projection shrinkage; for non-PL arrivals, position-and-price prior only |
| **Penalty and set-piece duty are step functions** | Model explicitly from `penalties_order`, not implicitly through rates |
| **Managers are an FPL asset since 2024-25** | Out of scope for v1 — state it rather than silently omitting |
| **The clustering work is not in this repo** | See below |

**On the clustering:** you mentioned it as a starting point, but the branch is
identical to `master` apart from our own commits — the clustering artefacts are
not here. If you point me at them, the natural use is as a **shrinkage prior**:
a player with thin data borrows the rate distribution of his cluster rather than
the league-wide positional mean. That is a real improvement for new signings and
low-minute players, which are the two weakest cases above.

---

## 7. Phasing

Each phase ends with a backtest, not a demo.

| Phase | Deliverable | Done when |
| --- | --- | --- |
| 0 | data layer + `scoring.py` | replays every historical gameweek at zero residual |
| 1 | match engine, team level | beats a bookmaker-free baseline on RPS for held-out seasons |
| 2 | goal/assist allocation | player-gameweek return probabilities are calibrated |
| 3 | peripherals, BPS, bonus | simulated BPS ranks reproduce actual bonus awards |
| 4 | availability and minutes | simulated minutes distributions match held-out actuals |
| 5 | full season, N=10k | season-total coverage bands verified across all players |
| 6 | FPL outputs | captaincy, differentials, price-change interaction |

Phases 0–2 are the spine. Phase 4 will move the numbers most.

---

## 8. Decisions I need from you

1. **Scope of v1** — every player, or a filtered universe (say, 2%+ ownership or
   900+ minutes last season)? Affects runtime a lot.
2. **Simulation count** — 10k season sims is roughly the point where tail
   estimates settle; 100k is better for differentials and much slower. Preference?
3. **Managers** — in or out for v1? I have them out above.
4. **The clustering artefacts** — where do they live?
5. **Target output** — a projected points table, or the full joint distribution
   retained so squad-level questions ("what is P(my XI beats the field this
   week)") can be asked later? The second costs storage but is much more useful,
   and is hard to retrofit.
