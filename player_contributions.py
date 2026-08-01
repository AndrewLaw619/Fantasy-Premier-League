"""Attribute FPL team strength scores to the individual players who drive them.

FPL publishes six numbers per team in ``teams.csv`` -- ``strength_attack_*``,
``strength_defence_*`` and ``strength_overall_*``.  Those numbers are editorial
integers set by the FPL team; they are not computed from player data, so they
cannot be decomposed directly.  What *can* be done is:

1.  Rebuild each score from player-level gameweek data and check how much of the
    published score that reconstruction explains (``calibrate``).
2.  Decompose the reconstruction, which is additive over players by
    construction, and express each player's share back in strength points.

Attacking output (xG, xA) is owned by individuals, so its decomposition is
exact.  Defensive output is a shared team outcome, so it is split with a ridge
regularised adjusted plus-minus fitted at fixture level.

Usage::

    python player_contributions.py --season 2024-25
    python player_contributions.py --season 2024-25 --team Arsenal
    python player_contributions.py --season 2024-25 --out contributions.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

# Seasons where FPL supplies expected_* columns in the gameweek data.
XG_SEASONS = ['2022-23', '2023-24', '2024-25', '2025-26']

# A player has to clear this many minutes before a replacement-level baseline or
# an APM coefficient is worth reading.
MIN_MINUTES = 450

# Strength points a squad needs to have generated above league average before
# splitting that credit between its players says anything.
MIN_SHARE_BASE = 5.0

NUMERIC_COLS = [
    'minutes', 'expected_goals', 'expected_assists', 'expected_goals_conceded',
    'goals_scored', 'assists',
]

# Candidate weights on realised output when blending it with expected output.
BLEND_GRID = np.arange(0.0, 1.001, 0.02)


def load_gameweeks(season, data_dir='data'):
    """Load merged gameweek data for a season with the numeric columns coerced."""
    path = os.path.join(data_dir, season, 'gws', 'merged_gw.csv')
    gw = pd.read_csv(path)
    for col in NUMERIC_COLS:
        gw[col] = pd.to_numeric(gw.get(col), errors='coerce').fillna(0)
    return gw


def load_web_names(season, gw, data_dir='data'):
    """Map each player's full name to the short name FPL displays.

    Truncating a full name to its last word mangles the ones that need help most
    -- "Diogo Teixeira da Silva" and "Gabriel Martinelli Silva" both collapse to
    "Silva".  ``players_raw.csv`` already carries FPL's own display name.
    """
    raw = pd.read_csv(os.path.join(data_dir, season, 'players_raw.csv'))
    short = raw.set_index('id').web_name
    element = gw.groupby('name').element.first()
    return element.map(short).fillna(pd.Series(element.index, index=element.index))


def load_team_strength(season, data_dir='data'):
    """Load teams.csv and average the home/away legs of each strength score."""
    teams = pd.read_csv(os.path.join(data_dir, season, 'teams.csv'))
    return pd.DataFrame({
        'team': teams['name'],
        'fpl_attack': (teams.strength_attack_home + teams.strength_attack_away) / 2,
        'fpl_defence': (teams.strength_defence_home + teams.strength_defence_away) / 2,
        'fpl_overall': (teams.strength_overall_home + teams.strength_overall_away) / 2,
    })


def team_season_rates(gw):
    """Per-match attacking and defensive rates for every team in a season.

    ``expected_goals_conceded`` is recorded once per on-pitch player, so the
    eleven readings for a match sum to eleven times the team's conceded xG.
    """
    agg = gw.groupby('team').agg(
        xG=('expected_goals', 'sum'),
        xA=('expected_assists', 'sum'),
        xGC=('expected_goals_conceded', 'sum'),
        goals=('goals_scored', 'sum'),
        assists=('assists', 'sum'),
        conceded=('goals_conceded', 'sum'),
        minutes=('minutes', 'sum'),
    ).reset_index()
    agg['matches'] = agg.team.map(gw.groupby('team').GW.nunique())

    # Goal involvements rather than goals: a team's involvement total is the
    # exact sum of its players' involvements, which is what makes the split
    # below arithmetic instead of an apportionment.
    agg['GI_per_match'] = (agg.goals + agg.assists) / agg.matches
    agg['xGI_per_match'] = (agg.xG + agg.xA) / agg.matches
    agg['GC_per_match'] = agg.conceded / 11 / agg.matches
    agg['xGC_per_match'] = agg.xGC / 11 / agg.matches
    return agg


def blended_driver(frame, weight, kind):
    """Blend realised and expected output, weighting realised output by ``weight``.

    Expected output measures how good the chances were; realised output measures
    what the team actually did with them.  The published scores respond to both,
    so neither alone is the right driver.
    """
    if kind == 'attack':
        return weight * frame.GI_per_match + (1 - weight) * frame.xGI_per_match
    if kind == 'defence':
        return weight * frame.GC_per_match + (1 - weight) * frame.xGC_per_match
    return (weight * (frame.GI_per_match - frame.GC_per_match)
            + (1 - weight) * (frame.xGI_per_match - frame.xGC_per_match))


DRIVERS = [('fpl_attack', 'attack'),
           ('fpl_defence', 'defence'),
           ('fpl_overall', 'overall')]


def _within_season_fit(pooled, driver, score):
    """Regress a score on a driver using within-season variation only."""
    dx = driver - driver.groupby(pooled.season).transform('mean')
    dy = pooled[score] - pooled.groupby('season')[score].transform('mean')
    slope = float(np.dot(dx, dy) / np.dot(dx, dx))
    r2 = float(np.corrcoef(dx, dy)[0, 1] ** 2)
    return slope, r2


def pooled_team_seasons(seasons=None, data_dir='data'):
    """Team-season rates joined to the published strength scores, all seasons."""
    seasons = seasons or XG_SEASONS
    frames = []
    for name in seasons:
        rates = team_season_rates(load_gameweeks(name, data_dir))
        merged = rates.merge(load_team_strength(name, data_dir), on='team')
        merged['season'] = name
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)


def best_blend_weight(pooled, score, kind):
    """The weight on realised output that best explains a published score."""
    scored = [(_within_season_fit(pooled, blended_driver(pooled, w, kind), score)[1], w)
              for w in BLEND_GRID]
    return max(scored)[1]


def calibrate(season, seasons=None, data_dir='data'):
    """Fit published FPL strength scores onto team rates rebuilt from players.

    Two things are fitted per score.  First the blend weight: how much the
    published score responds to what teams actually did versus how good their
    chances were.  Then the slope, which converts one unit of the winning driver
    into FPL strength points and so puts a player's share of that driver on the
    same scale as the published score.

    FPL re-bases the strength ladder between seasons, so both are fitted on
    seasons demeaned within season and the level is anchored to the requested
    season afterwards.

    Returns ``{score: (baseline, slope, weight, within_r2, season_r2, n)}``.
    """
    pooled = pooled_team_seasons(seasons, data_dir)

    fits = {}
    for score, kind in DRIVERS:
        weight = best_blend_weight(pooled, score, kind)
        driver = blended_driver(pooled, weight, kind)
        slope, within_r2 = _within_season_fit(pooled, driver, score)

        target = pooled.season == season
        season_r2 = float(np.corrcoef(driver[target], pooled.loc[target, score])[0, 1] ** 2)
        baseline = float(pooled.loc[target, score].mean() - slope * driver[target].mean())
        fits[score] = (baseline, slope, float(weight), within_r2, season_r2, len(pooled))
    return fits


def replacement_levels(players, quantile=0.20):
    """Positional replacement level for attacking rate, in involvements per 90.

    A player's contribution is measured against what a freely available squad
    player at the same position produces, not against zero -- otherwise every
    forward outranks every defender by construction.
    """
    regulars = players[players.minutes >= MIN_MINUTES]
    return regulars.groupby('position').blended_90.quantile(quantile)


def attack_contributions(gw, attack_slope, weight):
    """Exact additive split of each team's attacking rate across its players.

    Goals, assists, xG and xA are all owned by individuals, so a team's blended
    involvement rate is literally the sum of its players' rates and no model is
    needed.  A striker who scores twenty and a striker who scores none separate
    here on the realised half of the blend, not just on chance quality.
    """
    players = gw.groupby(['team', 'name', 'position']).agg(
        minutes=('minutes', 'sum'),
        xG=('expected_goals', 'sum'),
        xA=('expected_assists', 'sum'),
        goals=('goals_scored', 'sum'),
        assists=('assists', 'sum'),
    ).reset_index()
    players = players[players.minutes > 0].copy()
    players['nineties'] = players.minutes / 90
    players['GI'] = players.goals + players.assists
    players['xGI'] = players.xG + players.xA
    players['blended'] = weight * players.GI + (1 - weight) * players.xGI
    players['GI_90'] = players.GI / players.nineties
    players['xGI_90'] = players.xGI / players.nineties
    players['blended_90'] = players.blended / players.nineties

    matches = gw.groupby('team').GW.nunique()
    players['matches'] = players.team.map(matches)

    replacement = replacement_levels(players)
    players['replacement_90'] = players.position.map(replacement).fillna(0)

    # Share of the team's own attacking output.
    players['attack_share_pct'] = (
        100 * players.blended / players.groupby('team').blended.transform('sum'))

    # Rate above replacement, expressed per team-match then priced in strength
    # points via the calibration slope.
    above = (players.blended_90 - players.replacement_90) * players.nineties
    players['attack_above_repl'] = above
    players['attack_per_match'] = above / players.matches
    players['attack_points'] = attack_slope * players.attack_per_match
    return players.drop(columns=['GI', 'xGI', 'blended'])


def _fixture_design(gw, weight):
    """Build the fixture-level design matrix for the adjusted plus-minus fit.

    One row per team per fixture.  The target is the blend of goals and xG that
    team generated; the regressors are the on-pitch fraction of every player,
    entered once as an attacking effect for the team in possession and once as a
    defensive effect for the team facing them.
    """
    played = gw[gw.minutes > 0].copy()
    played['pid'] = played.team + '|' + played['name']

    sides = played.groupby(['fixture', 'team', 'was_home']).agg(
        xG=('expected_goals', 'sum'), goals=('goals_scored', 'sum')).reset_index()
    sides['scored'] = weight * sides.goals + (1 - weight) * sides.xG
    paired = sides.merge(sides[['fixture', 'team']].rename(columns={'team': 'opponent'}),
                         on='fixture')
    paired = paired[paired.team != paired.opponent].reset_index(drop=True)

    players = sorted(played.pid.unique())
    index = {pid: i for i, pid in enumerate(players)}
    n = len(players)

    lineups = played.groupby(['fixture', 'team'])[['pid', 'minutes']].apply(
        lambda d: list(zip(d.pid, d.minutes)), include_groups=False).to_dict()

    rows, cols, vals = [], [], []
    for r, (fixture, team, opponent) in enumerate(
            zip(paired.fixture, paired.team, paired.opponent)):
        for pid, minutes in lineups.get((fixture, team), []):
            rows.append(r)
            cols.append(index[pid])
            vals.append(min(minutes, 90) / 90)
        for pid, minutes in lineups.get((fixture, opponent), []):
            rows.append(r)
            cols.append(n + index[pid])
            vals.append(min(minutes, 90) / 90)

    x = sparse.csr_matrix((vals, (rows, cols)), shape=(len(paired), 2 * n))
    home = paired.was_home.astype(float).to_numpy().reshape(-1, 1)
    x = sparse.hstack([x, sparse.csr_matrix(home)]).tocsr()
    return x, paired.scored.to_numpy(), players


def _select_alpha(x, y, alphas, seed=0):
    """Pick the ridge penalty by held-out error on fixtures."""
    folds = KFold(n_splits=5, shuffle=True, random_state=seed)
    best, best_error = alphas[0], np.inf
    for alpha in alphas:
        errors = []
        for train, test in folds.split(np.arange(len(y))):
            model = Ridge(alpha=alpha).fit(x[train], y[train])
            errors.append(np.mean((y[test] - model.predict(x[test])) ** 2))
        error = np.mean(errors)
        if error < best_error:
            best, best_error = alpha, error
    return best, best_error


def defence_contributions(gw, defence_slope, weight, alphas=None, seed=0):
    """Split each team's conceded rate across players via ridge APM.

    Every player on the pitch shares the same conceded reading, so the only way
    to separate teammates is the variation created by rotation, injury and
    substitution.  Ridge handles the collinearity that leaves behind; players
    who never rotate stay poorly identified and are shrunk toward the league
    baseline.  Coefficients are additive: the intercept plus the eleven on-pitch
    coefficients reconstruct what a team conceded in that match.
    """
    alphas = alphas or [25, 50, 100, 200, 400, 800]
    x, y, players = _fixture_design(gw, weight)
    alpha, cv_error = _select_alpha(x, y, alphas, seed)
    model = Ridge(alpha=alpha).fit(x, y)

    n = len(players)
    ratings = pd.DataFrame({
        'pid': players,
        'apm_attack_90': model.coef_[:n],
        'apm_defence_90': model.coef_[n:2 * n],
    })
    ratings[['team', 'name']] = ratings.pid.str.split('|', n=1, expand=True)

    minutes = gw[gw.minutes > 0].copy()
    minutes['pid'] = minutes.team + '|' + minutes['name']
    ratings['minutes'] = ratings.pid.map(minutes.groupby('pid').minutes.sum())
    ratings['matches'] = ratings.team.map(gw.groupby('team').GW.nunique())

    # Minutes share of a full team-match, so contributions sum to the team rate.
    share = ratings.minutes / (ratings.matches * 90)
    ratings['defence_conceded_per_match'] = ratings.apm_defence_90 * share

    # A lower conceded coefficient is better, so credit is measured against the
    # league-average on-pitch player.
    league_average = np.average(ratings.apm_defence_90,
                                weights=ratings.minutes.clip(lower=1))
    saved = (league_average - ratings.apm_defence_90) * share
    ratings['defence_per_match'] = saved
    ratings['defence_points'] = defence_slope * -saved

    # Share of the defensive credit its own squad generated, so defenders are
    # ranked against teammates rather than against the attacking scale.  A squad
    # with almost nobody above league average has no meaningful split to report,
    # so the share is left blank rather than magnified out of rounding noise.
    positive = ratings.defence_points.clip(lower=0)
    squad_total = positive.groupby(ratings.team).transform('sum')
    ratings['defence_share_pct'] = np.where(
        squad_total >= MIN_SHARE_BASE, 100 * positive / squad_total, np.nan)

    return ratings.drop(columns=['pid']), {'alpha': alpha, 'cv_mse': cv_error,
                                           'r2': model.score(x, y),
                                           'home_effect': model.coef_[-1]}


def build_report(season, data_dir='data', calibration=None):
    """Per-player attacking and defensive contributions for one season."""
    gw = load_gameweeks(season, data_dir)
    fits = calibration or calibrate(season, data_dir=data_dir)
    _, attack_slope, attack_weight, _, _, _ = fits['fpl_attack']
    _, defence_slope, defence_weight, _, _, _ = fits['fpl_defence']
    overall_slope = fits['fpl_overall'][1]

    attack = attack_contributions(gw, attack_slope, attack_weight)
    defence, apm_info = defence_contributions(gw, defence_slope, defence_weight)

    report = attack.merge(
        defence[['team', 'name', 'apm_attack_90', 'apm_defence_90',
                 'defence_per_match', 'defence_points', 'defence_share_pct']],
        on=['team', 'name'], how='left')
    for col in ['defence_points', 'defence_per_match']:
        report[col] = report[col].fillna(0)

    # The overall score is calibrated on goal difference, so a player's overall
    # contribution is their attacking and defensive contributions in those same
    # units, priced with the overall slope rather than by adding two scores that
    # sit on different ladders.
    report['overall_points'] = overall_slope * (report.attack_per_match
                                                + report.defence_per_match)
    report['web_name'] = report['name'].map(load_web_names(season, gw, data_dir))
    report['season'] = season

    report = report.sort_values(['team', 'overall_points'], ascending=[True, False])
    return report, fits, apm_info


def team_accounting(report, season, fits, data_dir='data'):
    """Reconcile attributed player points against each team's published score.

    Attributed points are measured above a replacement-level baseline, so the
    residual is the part of the published score that the squad's floor -- rather
    than any individual -- accounts for.
    """
    published = load_team_strength(season, data_dir)
    totals = report.groupby('team').agg(
        attack_attributed=('attack_points', 'sum'),
        defence_attributed=('defence_points', 'sum'),
        overall_attributed=('overall_points', 'sum')).reset_index()
    totals = totals.merge(published, on='team', how='left')
    for score in ['attack', 'defence', 'overall']:
        totals[f'{score}_baseline'] = totals[f'fpl_{score}'] - totals[f'{score}_attributed']
    return totals.sort_values('overall_attributed', ascending=False)


def print_summary(report, fits, apm_info, season, team=None, top=10, data_dir='data'):
    """Print the calibration, then the biggest contributors."""
    labels = {'fpl_attack': ('attack ', 'involvements '),
              'fpl_defence': ('defence', 'conceded     '),
              'fpl_overall': ('overall', 'difference   ')}

    print(f'\nSeason {season}')
    print('\nHow much of the published FPL score the player data rebuilds')
    print('  score     driver per match  realised wt  within-R2  season-R2  points per unit')
    for score, (_, slope, weight, within_r2, season_r2, n) in fits.items():
        name, driver = labels[score]
        print(f'  {name}   {driver}        {weight:.2f}        {within_r2:.2f}       '
              f'{season_r2:.2f}     {slope:+8.1f}')
    print(f'  (fitted on {n} team-seasons, demeaned within season; realised wt is how far'
          f'\n   the score tracks actual goals rather than chance quality)')

    print(f"\nDefensive APM: alpha={apm_info['alpha']:g}  in-sample R^2={apm_info['r2']:.3f}  "
          f"home advantage={apm_info['home_effect']:+.3f} xG")

    accounting = team_accounting(report, season, fits, data_dir)
    teams = [team] if team else sorted(report.team.unique())
    for name in teams:
        squad = report[(report.team == name) & (report.minutes >= MIN_MINUTES)]
        if squad.empty:
            continue
        row = accounting[accounting.team == name]
        print(f'\n=== {name} ===')
        if not row.empty:
            row = row.iloc[0]
            for score in ['attack', 'defence', 'overall']:
                print(f'  published {score:<8} {row[f"fpl_{score}"]:.0f} = '
                      f'{row[f"{score}_baseline"]:.0f} replacement-level baseline + '
                      f'{row[f"{score}_attributed"]:.0f} attributed to players')
        cols = ['name', 'position', 'minutes', 'GI_90', 'xGI_90', 'attack_share_pct',
                'attack_points', 'defence_share_pct', 'defence_points', 'overall_points']
        print(squad.nlargest(top, 'overall_points')[cols].round(2).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--season', default='2024-25', help='season to analyse')
    parser.add_argument('--team', help='restrict the printed summary to one team')
    parser.add_argument('--data-dir', default='data', help='root of the data directory')
    parser.add_argument('--top', type=int, default=10, help='players listed per team')
    parser.add_argument('--out', help='write the full player table to this CSV')
    args = parser.parse_args()

    if args.season not in XG_SEASONS:
        parser.error(f'expected data is only available for {", ".join(XG_SEASONS)}')

    report, fits, apm_info = build_report(args.season, args.data_dir)
    print_summary(report, fits, apm_info, args.season, args.team, args.top, args.data_dir)

    if args.out:
        report.to_csv(args.out, index=False)
        print(f'\nWrote {len(report)} player rows to {args.out}')


if __name__ == '__main__':
    main()
