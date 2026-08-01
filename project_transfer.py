"""Project a player's goals and assists if they moved to a different club.

The projection follows what the transfer data supports (see
``transfer_analysis.py``): a player carries their *chance* rates with them, and
the destination squad supplies the rate at which chances become goals and
assists.  So:

    projected goals per 90   = shrunk xG per 90 x destination squad finishing
    projected assists per 90 = shrunk xA per 90 x destination squad conversion

Both halves are shrunk toward the league mean by factors fitted on players who
actually moved, because a single season over- or under-states a rate.

The range matters more than the point estimate here, and it is wide.  Intervals
come from the empirical spread of this projection's own errors across observed
transfers, so an 80% range means 80% of real moves landed inside it.

Usage::

    python project_transfer.py --player Salah --to Burnley
    python project_transfer.py --player B.Fernandes --to Everton --minutes 3000
    python project_transfer.py --validate
"""

import argparse

import numpy as np
import pandas as pd

from transfer_analysis import (FIRST_XG_SEASON, SEASONS, add_team_context,
                               consecutive_pairs, load_player_seasons)

# Interval widths reported, as quantiles of the projection's error ratio.
BANDS = {'50%': (0.25, 0.75), '80%': (0.10, 0.90)}

MIN_MINUTES = 900

# Ratios are taken on counts shifted by this much so they stay defined when a
# projection or an outcome is zero.  Every conversion back to a count has to undo
# the same shift, or the interval is not the one that was fitted.
SHIFT = 0.5


def ratio_band(point, ratios, low, high):
    """Turn a quantile of the fitted error ratio back into a count interval."""
    lo = max(0.0, np.quantile(ratios, low) * (point + SHIFT) - SHIFT)
    hi = max(0.0, np.quantile(ratios, high) * (point + SHIFT) - SHIFT)
    return lo, hi


def fit_shrinkage(pairs, rate):
    """How much of a player's rate survives a move, as slope and league mean.

    A season is a small sample, so a player's measured rate overstates how far
    they really sit from the league mean.  Regressing next season's rate on this
    season's across actual movers recovers how much of that gap persists.
    """
    frame = pairs[[f'{rate}_0', f'{rate}_1']].replace([np.inf, -np.inf], np.nan).dropna()
    slope, intercept = np.polyfit(frame[f'{rate}_0'], frame[f'{rate}_1'], 1)
    return float(slope), float(intercept)


def fit_error_bands(pairs, shrinkage, teams):
    """Empirical spread of the projection's error, as ratios of actual to predicted.

    Applying the projection to moves whose outcome is known gives the
    distribution of how far it lands from the truth, which is the only honest
    source for an interval.
    """
    movers = pairs.copy()
    ratios = {}
    for metric, rate, conversion in [('goals', 'expected_goals_90', 'finishing'),
                                     ('assists', 'expected_assists_90', 'assist_conversion')]:
        slope, intercept = shrinkage[rate]
        predicted_90 = (slope * movers[f'{rate}_0'] + intercept) * movers[f'{conversion}_1']
        nineties = movers[f'minutes_1'] / 90
        predicted = predicted_90 * nineties
        actual = movers[f'{metric}_1']
        ok = np.isfinite(predicted) & np.isfinite(actual)
        ratios[metric] = np.sort(((actual[ok] + SHIFT)
                                  / (predicted[ok] + SHIFT)).to_numpy())
    return ratios


def build_model(data_dir='data', min_minutes=MIN_MINUTES):
    """Load the data and fit every parameter the projection needs."""
    players = add_team_context(load_player_seasons(data_dir))
    pairs = consecutive_pairs(players, min_minutes)
    movers = pairs[pairs.moved & pairs.xg_era]

    shrinkage = {rate: fit_shrinkage(movers, rate)
                 for rate in ['expected_goals_90', 'expected_assists_90']}
    teams = players[players['index'] >= FIRST_XG_SEASON]
    bands = fit_error_bands(movers, shrinkage, teams)
    return players, movers, shrinkage, bands


def latest_season_row(frame):
    return frame.sort_values('index').iloc[-1]


def find_player(players, query, season=None):
    matches = players[players.name.str.contains(query, case=False, na=False)
                      & (players['index'] >= FIRST_XG_SEASON)
                      & (players.minutes >= 450)]
    if season:
        matches = matches[matches.season == season]
    if matches.empty:
        raise SystemExit(f'no player matching "{query}" with 450+ minutes in the xG era')
    return latest_season_row(matches)


def find_team(players, query, season=None):
    squads = players[players['index'] >= FIRST_XG_SEASON].dropna(subset=['team'])
    matches = squads[squads.team.str.contains(query, case=False, na=False)]
    if season:
        matches = matches[matches.season == season]
    if matches.empty:
        raise SystemExit(f'no team matching "{query}" in the xG era')
    season_row = latest_season_row(matches)
    return matches[(matches.team == season_row.team)
                   & (matches.season == season_row.season)].iloc[0]


def project(player, destination, shrinkage, bands, minutes=None):
    """Projected season goals and assists for a player at a destination squad."""
    minutes = minutes or player.minutes
    nineties = minutes / 90

    result = {'minutes': minutes}
    for metric, rate, conversion in [('goals', 'expected_goals', 'finishing'),
                                     ('assists', 'expected_assists', 'assist_conversion')]:
        observed_90 = player[rate] / (player.minutes / 90)
        slope, intercept = shrinkage[f'{rate}_90']
        carried_90 = slope * observed_90 + intercept
        point = carried_90 * destination[conversion] * nineties
        ranges = {label: ratio_band(point, bands[metric], low, high)
                  for label, (low, high) in BANDS.items()}
        result[metric] = {'observed_90': observed_90, 'carried_90': carried_90,
                          'point': point, 'ranges': ranges,
                          'actual': player[metric if metric == 'assists' else 'goals']}
    return result


def print_projection(player, destination, result):
    print(f'\n{player["name"]}  ({player.team}, {player.season})  ->  '
          f'{destination.team} ({destination.season})')
    print(f'  actual last season: {int(player.goals)} goals, {int(player.assists)} assists '
          f'in {int(player.minutes)} minutes')
    print(f'  {destination.team} convert {destination.finishing:.2f} goals per xG and '
          f'{destination.assist_conversion:.2f} assists per xA')
    print(f'\n  projected over {int(result["minutes"])} minutes:')
    for metric in ['goals', 'assists']:
        block = result[metric]
        low50, high50 = block['ranges']['50%']
        low80, high80 = block['ranges']['80%']
        print(f'    {metric:<8} {block["point"]:5.1f}    '
              f'50% range {low50:4.1f} - {high50:4.1f}    '
              f'80% range {low80:4.1f} - {high80:4.1f}')
    print('\n  The range is the answer here -- the point estimate alone overstates')
    print('  what one season of data can tell you.')
    print('  Trust the goals line further than the assists line: squad finishing')
    print('  is a real trait, squad assist conversion is mostly noise (the worst')
    print('  attacks in the league top that table), so it barely beat carrying the')
    print('  player\'s own xA across unadjusted.')


def validate(movers, shrinkage, bands):
    """Check the intervals cover what they claim, holding each move out in turn.

    Quantiles fitted on the same rows they are tested against would cover by
    construction, so each move is scored against bands fitted without it.
    """
    print('\nInterval coverage, leave-one-out across observed transfers')
    for metric, rate, conversion in [('goals', 'expected_goals_90', 'finishing'),
                                     ('assists', 'expected_assists_90', 'assist_conversion')]:
        slope, intercept = shrinkage[rate]
        nineties = movers['minutes_1'] / 90
        predicted = ((slope * movers[f'{rate}_0'] + intercept)
                     * movers[f'{conversion}_1'] * nineties)
        actual = movers[f'{metric}_1']
        ok = (np.isfinite(predicted) & np.isfinite(actual)).to_numpy()
        predicted, actual = predicted[ok].to_numpy(), actual[ok].to_numpy()
        ratios = (actual + SHIFT) / (predicted + SHIFT)

        counts = {label: 0 for label in BANDS}
        for i in range(len(ratios)):
            others = np.delete(ratios, i)
            for label, (low, high) in BANDS.items():
                lo, hi = ratio_band(predicted[i], others, low, high)
                if lo <= actual[i] <= hi:
                    counts[label] += 1
        summary = '   '.join(f'{label} band covers {100 * counts[label] / len(ratios):.0f}%'
                             for label in BANDS)
        print(f'  {metric:<8} n = {len(ratios)}   {summary}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--player', help='substring of the player display name')
    parser.add_argument('--to', dest='destination', help='substring of the destination club')
    parser.add_argument('--player-season', help='season to take the player\'s form from')
    parser.add_argument('--team-season', help='season to take the destination squad from')
    parser.add_argument('--minutes', type=int, help='minutes to project over')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--validate', action='store_true',
                        help='report interval coverage instead of a projection')
    args = parser.parse_args()

    players, movers, shrinkage, bands = build_model(args.data_dir)

    if args.validate:
        validate(movers, shrinkage, bands)
        return
    if not (args.player and args.destination):
        parser.error('--player and --to are required unless --validate is given')

    player = find_player(players, args.player, args.player_season)
    destination = find_team(players, args.destination, args.team_season)
    result = project(player, destination, shrinkage, bands, args.minutes)
    print_projection(player, destination, result)


if __name__ == '__main__':
    main()
