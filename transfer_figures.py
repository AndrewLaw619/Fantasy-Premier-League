"""Render the figures behind the transfer projection in project_transfer.

Five figures, each answering one question:

1. portability   -- what survives a move: chances, or finished output
2. channels      -- which half of a player's output team quality actually moves
3. model         -- whether the destination multiplier earns its place
4. accuracy      -- how the projection lands against 73 real moves
5. range         -- destination versus uncertainty, for one player

Usage::

    python transfer_figures.py --outdir figures
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

import project_transfer as pt
import transfer_analysis as ta
from contribution_figures import (GRID, INK, INK_MUTED, INK_SOFT, SERIES,
                                  SHADE_LIGHT, SURFACE, bar_height, caption,
                                  fit_line, px_to_data, rounded_barh, scatter,
                                  style)

METRICS = [('expected_goals_90', 'chances taken (xG/90)'),
           ('expected_assists_90', 'chances created (xA/90)'),
           ('goals_90', 'goals per 90'),
           ('assists_90', 'assists per 90')]


# --------------------------------------------------------------------------
# Figure 1 -- what survives a move
# --------------------------------------------------------------------------

def fig_portability(pairs, outdir):
    """Year-over-year persistence for movers against stayers."""
    movers = pairs[pairs.moved & pairs.xg_era]
    stayers = pairs[~pairs.moved & pairs.xg_era]

    rows = []
    for rate, label in METRICS:
        r_move, n_move = ta.correlation(movers, f'{rate}_0', f'{rate}_1')
        r_stay, n_stay = ta.correlation(stayers, f'{rate}_0', f'{rate}_1')
        rows.append((label, r_stay, r_move, n_stay, n_move))
    rows.sort(key=lambda row: row[2])

    fig, ax = plt.subplots(figsize=(9.4, 3.9))
    ax.grid(axis='x', alpha=0.9)
    ax.set_axisbelow(True)
    ys = np.arange(len(rows))

    for y, (_, r_stay, r_move, _, _) in zip(ys, rows):
        ax.plot([r_move, r_stay], [y, y], color=GRID, linewidth=2,
                solid_capstyle='round', zorder=2)
    ax.scatter([r[1] for r in rows], ys, s=80, color=SHADE_LIGHT,
               edgecolor=SURFACE, linewidth=2, zorder=4, label='stayed at the same club')
    ax.scatter([r[2] for r in rows], ys, s=80, color=SERIES[0],
               edgecolor=SURFACE, linewidth=2, zorder=5, label='changed club')

    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(0, 1)
    ax.set_xlabel('year-over-year correlation of the rate (r)')
    ax.set_title('Chances travel with the player. Finished output does not.',
                 loc='left', fontsize=12.5)
    ax.legend(loc='lower right')
    caption(fig, f'Consecutive seasons with 900+ minutes in both, 2022-23 onward. '
                 f'Movers n = {len(movers)}, stayers n = {len(stayers)}.')
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    path = os.path.join(outdir, '7-transfer-portability.png')
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Figure 2 -- which channel team quality moves
# --------------------------------------------------------------------------

def fig_channels(pairs, outdir):
    """The flat relationship beside the sloped one, on a shared y-scale."""
    movers = pairs[pairs.moved & pairs.xg_era].copy()
    movers['d_team_xG'] = movers.team_xG_1 - movers.team_xG_0
    movers['d_finishing'] = movers.finishing_1 - movers.finishing_0
    movers['d_xG_90'] = movers.expected_goals_90_1 - movers.expected_goals_90_0
    movers['d_goals_90'] = movers.goals_90_1 - movers.goals_90_0

    panels = [('d_team_xG', 'd_xG_90',
               'change in the squad\'s season xG total',
               'change in the player\'s own xG per 90',
               'Chances taken: barely moves'),
              ('d_finishing', 'd_goals_90',
               'change in the squad\'s finishing rate (goals per xG)',
               'change in the player\'s goals per 90',
               'Goals: moves a lot')]

    # Both y-axes are a change in the player's own per-90 rate, so they share a
    # scale -- otherwise "flat" versus "sloped" is partly an artefact of zooming.
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.0), sharey=True)
    for ax, (xcol, ycol, xlabel, ylabel, title) in zip(axes, panels):
        frame = movers[[xcol, ycol]].replace([np.inf, -np.inf], np.nan).dropna()
        r = np.corrcoef(frame[xcol], frame[ycol])[0, 1]
        ax.grid(alpha=0.9)
        ax.set_axisbelow(True)
        ax.axhline(0, color=GRID, linewidth=1, zorder=1)
        ax.axvline(0, color=GRID, linewidth=1, zorder=1)
        fit_line(ax, frame[xcol].to_numpy(), frame[ycol].to_numpy())
        scatter(ax, frame[xcol], frame[ycol])
        ax.set_title(f'{title}   r = {r:+.2f}', loc='left')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    fig.suptitle('A better squad changes how many chances become goals, '
                 'not how many chances you get',
                 x=0.5, y=0.98, fontsize=12.5, fontweight='semibold', color=INK)
    caption(fig, f'One dot per move, n = {len(movers)}. Line is least squares.')
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    path = os.path.join(outdir, '8-transfer-channels.png')
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Figure 3 -- does the destination multiplier earn its place
# --------------------------------------------------------------------------

def fig_model(pairs, outdir):
    """Out-of-sample error for three projections, per metric."""
    movers = pairs[pairs.moved & pairs.xg_era]
    recipes = {
        'goals per 90': [
            ('carry goals forward', movers.goals_90_0),
            ('carry xG forward', movers.expected_goals_90_0),
            ('xG x destination finishing',
             movers.expected_goals_90_0 * movers.finishing_1)],
        'assists per 90': [
            ('carry assists forward', movers.assists_90_0),
            ('carry xA forward', movers.expected_assists_90_0),
            ('xA x destination conversion',
             movers.expected_assists_90_0 * movers.assist_conversion_1)],
    }
    truths = {'goals per 90': movers.goals_90_1, 'assists per 90': movers.assists_90_1}

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.5), sharex=True)
    worst = 0
    scored = {}
    for metric, options in recipes.items():
        scored[metric] = [(label, ta.error(truths[metric].to_numpy(),
                                           prediction.to_numpy())[0])
                          for label, prediction in options]
        worst = max(worst, max(mae for _, mae in scored[metric]))

    for ax, (metric, results) in zip(axes, scored.items()):
        ax.grid(axis='x', alpha=0.9)
        ax.set_axisbelow(True)
        ax.set_yticks(np.arange(len(results)))
        ax.set_yticklabels([label for label, _ in results][::-1], fontsize=9)
        ax.set_ylim(-0.7, len(results) - 0.3)
        ax.set_xlim(0, worst * 1.3)
        fig.canvas.draw()

        height = bar_height(ax)
        baseline = results[0][1]
        for y, (label, mae) in enumerate(results[::-1]):
            rounded_barh(ax, y, 0, mae, height, SERIES[0])
            change = 100 * (mae / baseline - 1)
            text = f'{mae:.4f}' if label == results[0][0] else f'{mae:.4f}   {change:+.0f}%'
            ax.annotate(text, (mae, y), textcoords='offset points', xytext=(7, 0),
                        va='center', fontsize=8.5, color=INK)
        ax.set_title(metric, loc='left')
        ax.set_xlabel('mean absolute error (lower is better)')

    fig.suptitle('The destination multiplier earns its place for goals, not for assists',
                 x=0.5, y=0.98, fontsize=12.5, fontweight='semibold', color=INK)
    caption(fig, f'Projecting the season after a move, n = {len(movers)}. '
                 'Percentages are against the naive carry-forward.')
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    path = os.path.join(outdir, '9-transfer-model.png')
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Figure 4 -- how the projection lands against real moves
# --------------------------------------------------------------------------

def fig_accuracy(pairs, shrinkage, bands, outdir):
    """Predicted against actual season goals, with the fitted 80% band drawn."""
    movers = pairs[pairs.moved & pairs.xg_era]
    slope, intercept = shrinkage['expected_goals_90']
    nineties = movers.minutes_1 / 90
    predicted = ((slope * movers.expected_goals_90_0 + intercept)
                 * movers.finishing_1 * nineties)
    actual = movers.goals_1
    ok = np.isfinite(predicted) & np.isfinite(actual)
    predicted, actual = predicted[ok].to_numpy(), actual[ok].to_numpy()

    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    ax.grid(alpha=0.9)
    ax.set_axisbelow(True)

    grid = np.linspace(0, predicted.max() * 1.1, 120)
    low = [pt.ratio_band(value, bands['goals'], 0.10, 0.90)[0] for value in grid]
    high = [pt.ratio_band(value, bands['goals'], 0.10, 0.90)[1] for value in grid]
    ax.fill_between(grid, low, high, color=SERIES[0], alpha=0.10, zorder=1,
                    label='80% range the model claims')
    ax.plot(grid, grid, color=INK_MUTED, linewidth=1.5, alpha=0.7, zorder=2,
            label='perfect projection')
    scatter(ax, predicted, actual)

    inside = sum(1 for p, a in zip(predicted, actual)
                 if pt.ratio_band(p, bands['goals'], 0.10, 0.90)[0] <= a
                 <= pt.ratio_band(p, bands['goals'], 0.10, 0.90)[1])
    ax.set_xlabel('projected goals for the season after the move')
    ax.set_ylabel('goals actually scored')
    ax.set_title('The projection tracks the outcome, but the band is wide\n'
                 f'{inside} of {len(predicted)} real moves land inside the 80% range',
                 loc='left', fontsize=12.5)
    ax.legend(loc='upper left')
    caption(fig, 'One dot per transfer. The band is fitted from the spread of the '
                 'projection\'s own errors, not assumed.')
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = os.path.join(outdir, '10-transfer-accuracy.png')
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Figure 5 -- destination against uncertainty
# --------------------------------------------------------------------------

def fig_range(players, shrinkage, bands, outdir, player_query='B.Fernandes',
              season='2025-26'):
    """One player projected to every club, so the two spreads sit side by side."""
    player = pt.find_player(players, player_query, season)
    squads = (players[(players.season == season)].dropna(subset=['team'])
              .groupby('team').first().reset_index())

    projections = []
    for squad in squads.itertuples():
        if squad.team == player.team:
            continue
        destination = players[(players.season == season)
                              & (players.team == squad.team)].iloc[0]
        projections.append((squad.team,
                            pt.project(player, destination, shrinkage, bands)))
    # Both panels keep the ordering of the validated metric, so the assists panel
    # shows how little its own multiplier tracks squad quality.
    projections.sort(key=lambda row: row[1]['goals']['point'])
    teams = [team for team, _ in projections]
    ys = np.arange(len(projections))

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 8.0), sharey=True)
    for ax, metric in zip(axes, ['goals', 'assists']):
        ax.grid(axis='x', alpha=0.9)
        ax.set_axisbelow(True)
        for y, (_, result) in zip(ys, projections):
            block = result[metric]
            lo80, hi80 = block['ranges']['80%']
            lo50, hi50 = block['ranges']['50%']
            ax.plot([lo80, hi80], [y, y], color=SHADE_LIGHT, linewidth=3,
                    solid_capstyle='round', zorder=2)
            ax.plot([lo50, hi50], [y, y], color=SERIES[0], linewidth=3,
                    solid_capstyle='round', zorder=3)
        points = [result[metric]['point'] for _, result in projections]
        ax.scatter(points, ys, s=46, color=INK, edgecolor=SURFACE, linewidth=2,
                   zorder=5)
        ax.set_ylim(-0.8, len(projections) - 0.2)
        ax.set_xlim(left=0)
        ax.set_xlabel(f'projected {metric} over the same minutes')
        widest = max(result[metric]['ranges']['80%'][1]
                     - result[metric]['ranges']['80%'][0] for _, result in projections)
        ax.set_title(f'{metric.capitalize()}   destination moves it '
                     f'{max(points) - min(points):.1f}, the 80% range spans '
                     f'up to {widest:.1f}', loc='left', fontsize=10.5)

    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(teams, fontsize=8.5)

    handles = [plt.Line2D([], [], color=INK, marker='o', linestyle='none',
                          markersize=6, label='point estimate'),
               plt.Line2D([], [], color=SERIES[0], linewidth=3, label='50% range'),
               plt.Line2D([], [], color=SHADE_LIGHT, linewidth=3, label='80% range')]
    fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.062))
    fig.suptitle(f'{player["name"]} projected to all 19 other clubs, {season}',
                 x=0.5, y=0.98, fontsize=12.5, fontweight='semibold', color=INK)
    fig.text(0.5, 0.012,
             'Choosing the club shifts the projection far less than the uncertainty '
             'around any single one of them. Clubs are ordered by projected goals;\n'
             'the assists panel barely follows that order because squad assist '
             'conversion is mostly noise.',
             ha='center', fontsize=8, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.115, 1, 0.94))
    path = os.path.join(outdir, '11-transfer-range.png')
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--outdir', default='figures')
    parser.add_argument('--player', default='B.Fernandes')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    style()

    players, movers, shrinkage, bands = pt.build_model(args.data_dir)
    everyone = ta.add_team_context(ta.load_player_seasons(args.data_dir))
    pairs = ta.consecutive_pairs(everyone, pt.MIN_MINUTES)

    written = [
        fig_portability(pairs, args.outdir),
        fig_channels(pairs, args.outdir),
        fig_model(pairs, args.outdir),
        fig_accuracy(pairs, shrinkage, bands, args.outdir),
        fig_range(players, shrinkage, bands, args.outdir, args.player),
    ]
    for path in written:
        print(f'wrote {path}')


if __name__ == '__main__':
    main()
