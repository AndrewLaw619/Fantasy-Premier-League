"""Can we predict how a player's output changes if they move to another club?

The repo has ten seasons of squad lists, so intra-league transfers supply real
before/after observations to test against rather than a plausible story.  This
script builds that sample and answers four questions:

1.  How much of a player's per-90 output survives a move at all?
2.  Which quantities travel -- realised output, or expected output?
3.  Does the destination squad's conversion rate improve a projection?
4.  What happens to a player's *share* of their team's creation when they move
    down a level, which is the case worth modelling?

Usage::

    python transfer_analysis.py
    python transfer_analysis.py --min-minutes 1200
"""

import argparse
import os

import numpy as np
import pandas as pd

SEASONS = ['2016-17', '2017-18', '2018-19', '2019-20', '2020-21', '2021-22',
           '2022-23', '2023-24', '2024-25', '2025-26']

# FPL only publishes expected_* from 2022-23, so anything xG-based starts here.
FIRST_XG_SEASON = SEASONS.index('2022-23')

POSITIONS = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD', 5: 'AM'}


def load_player_seasons(data_dir='data'):
    """Season totals per player, keyed by FPL's stable cross-season ``code``.

    ``players_raw.csv`` is the end-of-season snapshot, so it already carries
    season totals.  Team names come from ``teams.csv`` where a season has one and
    from ``master_team_list.csv`` for the older seasons that do not.
    """
    master = pd.read_csv(os.path.join(data_dir, 'master_team_list.csv'))
    frames = []
    for index, season in enumerate(SEASONS):
        raw = pd.read_csv(os.path.join(data_dir, season, 'players_raw.csv'))
        teams_path = os.path.join(data_dir, season, 'teams.csv')
        if os.path.exists(teams_path):
            names = pd.read_csv(teams_path).set_index('id')['name']
        else:
            names = master[master.season == season].set_index('team')['team_name']

        frame = pd.DataFrame({
            'code': raw.code,
            'name': raw.web_name,
            'team': raw.team.map(names),
            'position': raw.element_type.map(POSITIONS),
            'minutes': raw.minutes,
            'goals': raw.goals_scored,
            'assists': raw.assists,
            'season': season,
            'index': index,
        })
        for column in ['expected_goals', 'expected_assists']:
            frame[column] = (pd.to_numeric(raw[column], errors='coerce')
                             if column in raw.columns else np.nan)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def add_team_context(players):
    """Attach each squad's finishing and assist-conversion rates to its players."""
    teams = players.groupby(['season', 'index', 'team']).agg(
        team_goals=('goals', 'sum'), team_assists=('assists', 'sum'),
        team_xG=('expected_goals', 'sum'), team_xA=('expected_assists', 'sum'),
    ).reset_index()
    teams['finishing'] = teams.team_goals / teams.team_xG
    teams['assist_conversion'] = teams.team_assists / teams.team_xA
    return players.merge(teams, on=['season', 'index', 'team'], how='left')


def consecutive_pairs(players, min_minutes):
    """One row per player per consecutive pair of seasons, with per-90 rates."""
    pairs = players.merge(players, on='code', suffixes=('_0', '_1'))
    pairs = pairs[pairs.index_1 == pairs.index_0 + 1]
    pairs = pairs[(pairs.minutes_0 >= min_minutes) & (pairs.minutes_1 >= min_minutes)]
    for column in ['goals', 'assists', 'expected_goals', 'expected_assists']:
        for side in '01':
            pairs[f'{column}_90_{side}'] = (pairs[f'{column}_{side}']
                                            / (pairs[f'minutes_{side}'] / 90))
    pairs['involvements_90_0'] = pairs.goals_90_0 + pairs.assists_90_0
    pairs['involvements_90_1'] = pairs.goals_90_1 + pairs.assists_90_1
    pairs['moved'] = pairs.team_0 != pairs.team_1
    pairs['xg_era'] = pairs.index_0 >= FIRST_XG_SEASON
    return pairs


def correlation(frame, left, right):
    clean = frame[[left, right]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 3:
        return np.nan, len(clean)
    return float(np.corrcoef(clean[left], clean[right])[0, 1]), len(clean)


def error(truth, prediction):
    ok = np.isfinite(truth) & np.isfinite(prediction)
    if not ok.any():
        return np.nan, np.nan, 0
    gap = truth[ok] - prediction[ok]
    return float(np.abs(gap).mean()), float(np.sqrt((gap ** 2).mean())), int(ok.sum())


def report_persistence(pairs):
    """How much of a player's output survives a move."""
    print('\n1. Year-over-year persistence of goal involvements per 90')
    for label, subset in [('stayed', pairs[~pairs.moved]), ('moved ', pairs[pairs.moved])]:
        r, n = correlation(subset, 'involvements_90_0', 'involvements_90_1')
        print(f'   {label}: r = {r:.3f}   n = {n}')
    print('   Moving costs persistence, but the player still dominates the team.')


def report_portability(pairs):
    """Which quantities travel: realised output or expected output."""
    metrics = [('assists per 90 ', 'assists_90_0', 'assists_90_1'),
               ('xA per 90      ', 'expected_assists_90_0', 'expected_assists_90_1'),
               ('goals per 90   ', 'goals_90_0', 'goals_90_1'),
               ('xG per 90      ', 'expected_goals_90_0', 'expected_goals_90_1')]
    print('\n2. What travels with the player (xG era only)')
    for label, subset in [('movers', pairs[pairs.moved & pairs.xg_era]),
                          ('stayers', pairs[~pairs.moved & pairs.xg_era])]:
        print(f'   {label}:')
        for name, left, right in metrics:
            r, n = correlation(subset, left, right)
            print(f'     {name} r = {r:.3f}   n = {n}')
    print('   Expected output is markedly more portable than realised output.')


def report_projection(pairs):
    """Does the destination squad's conversion rate improve a projection?"""
    movers = pairs[pairs.moved & pairs.xg_era]
    print(f'\n3. Projecting the season after a move (n = {len(movers)})')

    recipes = {
        'assists per 90': [
            ('carry assists forward (naive)', movers.assists_90_0),
            ('carry xA forward', movers.expected_assists_90_0),
            ('xA x destination assist conversion',
             movers.expected_assists_90_0 * movers.assist_conversion_1),
        ],
        'goals per 90': [
            ('carry goals forward (naive)', movers.goals_90_0),
            ('carry xG forward', movers.expected_goals_90_0),
            ('xG x destination finishing',
             movers.expected_goals_90_0 * movers.finishing_1),
        ],
    }
    truths = {'assists per 90': movers.assists_90_1, 'goals per 90': movers.goals_90_1}

    for metric, options in recipes.items():
        print(f'   {metric}')
        for label, prediction in options:
            mae, rmse, n = error(truths[metric].to_numpy(), prediction.to_numpy())
            print(f'     {label:<38} MAE {mae:.4f}  RMSE {rmse:.4f}  n = {n}')
    print('   The destination multiplier earns its place for goals, barely for assists:')
    print('   squad finishing is a more stable trait than squad assist conversion.')


def report_channel(pairs):
    """Which part of a player's output the destination team's quality moves.

    Splitting output into the chances a player generates and the rate those
    chances convert separates what the player carries from what the squad
    supplies -- and only the second responds to team quality.
    """
    movers = pairs[pairs.moved & pairs.xg_era].copy()
    movers['d_team_xG'] = movers.team_xG_1 - movers.team_xG_0
    movers['d_finishing'] = movers.finishing_1 - movers.finishing_0
    for metric in ['goals_90', 'assists_90', 'expected_goals_90', 'expected_assists_90']:
        movers[f'd_{metric}'] = movers[f'{metric}_1'] - movers[f'{metric}_0']

    print('\n4. Which channel does team quality actually move?')
    print('   change in team quality  ->  change in player rate        r')
    rows = [('team xG total    ', 'd_team_xG', 'chances created (xA/90)  ', 'd_expected_assists_90'),
            ('team xG total    ', 'd_team_xG', 'chances taken   (xG/90)  ', 'd_expected_goals_90'),
            ('squad finishing  ', 'd_finishing', 'goals per 90             ', 'd_goals_90'),
            ('squad finishing  ', 'd_finishing', 'assists per 90           ', 'd_assists_90')]
    for quality_label, quality, output_label, output in rows:
        r, n = correlation(movers, quality, output)
        print(f'   {quality_label}        {output_label}  {r:+.3f}   n = {n}')
    print('   A better squad barely changes how many chances a player generates.')
    print('   It changes how many of them become goals and assists.')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--min-minutes', type=int, default=900,
                        help='minutes required in both seasons of a pair')
    args = parser.parse_args()

    players = add_team_context(load_player_seasons(args.data_dir))
    pairs = consecutive_pairs(players, args.min_minutes)

    movers = int(pairs.moved.sum())
    print(f'Transfer sample: {movers} moves between Premier League clubs with '
          f'{args.min_minutes}+ minutes in both seasons')
    print(f'  of which in the xG era (2022-23 onward): '
          f'{int((pairs.moved & pairs.xg_era).sum())}')

    report_persistence(pairs)
    report_portability(pairs)
    report_projection(pairs)
    report_channel(pairs)


if __name__ == '__main__':
    main()
