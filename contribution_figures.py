"""Render the figures behind the strength-score attribution in player_contributions.

Six figures, each answering one question:

1. calibration       -- how much of each published score the player data rebuilds
2. blend-weight      -- why realised output belongs in the driver at all
3. finishing-premium -- which players the blend moves, and by how much
4. concentration     -- how reliant each team is on its single biggest contributor
5. by-position       -- how attacking and defensive credit divide by position
6. squad             -- one team's published score broken down player by player

Usage::

    python contribution_figures.py --season 2024-25 --outdir figures
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path
from matplotlib.patches import PathPatch

import player_contributions as pc

# Validated categorical slots (light surface #fcfcfb) -- see the dataviz palette
# reference.  Slot 3 sits below 3:1 contrast, so anything drawn in it carries a
# visible direct label rather than relying on the legend alone.
SERIES = ['#2a78d6', '#eb6834', '#1baf7a']
SHADE_LIGHT = '#9ec5f4'
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_SOFT = '#52514e'
INK_MUTED = '#8a887f'
GRID = '#e5e4e0'

BAR_RADIUS_PX = 4.0
BAR_CAP_PX = 24.0
GAP_PX = 2.0


def style():
    """Recessive axes, hairline solid grid, text in ink tokens."""
    plt.rcParams.update({
        'figure.facecolor': SURFACE,
        'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE,
        'axes.edgecolor': GRID,
        'axes.linewidth': 1.0,
        'axes.labelcolor': INK_SOFT,
        'axes.titlecolor': INK,
        'axes.titlesize': 11,
        'axes.titleweight': 'semibold',
        'axes.labelsize': 9,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.color': INK_SOFT,
        'ytick.color': INK_SOFT,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'xtick.direction': 'out',
        'grid.color': GRID,
        'grid.linewidth': 1.0,
        'grid.linestyle': '-',
        'legend.frameon': False,
        'legend.fontsize': 9,
        'font.size': 9,
        'figure.dpi': 150,
    })


def px_to_data(ax, pixels, axis='x'):
    """Convert a pixel distance into data units on one axis."""
    origin = ax.transData.transform((0, 0))
    offset = (pixels, 0) if axis == 'x' else (0, pixels)
    shifted = ax.transData.inverted().transform(origin + np.array(offset))
    base = ax.transData.inverted().transform(origin)
    return abs(shifted[0] - base[0]) if axis == 'x' else abs(shifted[1] - base[1])


def rounded_barh(ax, y, start, end, height, color, round_end=True, radius=None):
    """Horizontal bar, square at the baseline and rounded at the data end.

    Matplotlib has no per-corner bar rounding, so the outline is built as an
    explicit path with quadratic corners on the data end only.
    """
    radius = px_to_data(ax, BAR_RADIUS_PX, 'x') if radius is None else radius
    span = abs(end - start)
    if span == 0:
        return
    radius = min(radius, span / 2, height / 2)
    sign = 1.0 if end >= start else -1.0
    bottom, top = y - height / 2, y + height / 2

    if not round_end or radius <= 0:
        verts = [(start, bottom), (end, bottom), (end, top), (start, top)]
        codes = [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
        verts.append((start, bottom))
    else:
        inner = end - sign * radius
        verts = [
            (start, bottom), (inner, bottom),
            (end, bottom), (end, bottom + radius),
            (end, top - radius),
            (end, top), (inner, top),
            (start, top), (start, bottom),
        ]
        codes = [Path.MOVETO, Path.LINETO,
                 Path.CURVE3, Path.CURVE3,
                 Path.LINETO,
                 Path.CURVE3, Path.CURVE3,
                 Path.LINETO, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor='none',
                           clip_on=False, zorder=3))


def bar_height(ax, band=0.62):
    """Bar thickness in data units, capped so the band keeps its air."""
    return min(band, px_to_data(ax, BAR_CAP_PX, 'y'))


def fit_line(ax, x, y, color=INK_MUTED):
    """Draw the least-squares line behind the points."""
    slope, intercept = np.polyfit(x, y, 1)
    span = np.array([x.min(), x.max()])
    ax.plot(span, slope * span + intercept, color=color, linewidth=1.5,
            zorder=2, alpha=0.7)


def scatter(ax, x, y, color=SERIES[0], size=34):
    """Dots with a 2px surface ring so overlaps stay legible."""
    ax.scatter(x, y, s=size, color=color, edgecolor=SURFACE, linewidth=2,
               zorder=4)


def label_points(ax, frame, xcol, ycol, labels, dx=6, dy=0):
    for _, row in frame.iterrows():
        ax.annotate(row[labels], (row[xcol], row[ycol]),
                    textcoords='offset points', xytext=(dx, dy),
                    fontsize=7.5, color=INK_SOFT, va='center', zorder=5)


def label_declutter(ax, frame, xcol, ycol, labels, min_gap_px=15.0):
    """Label points, nudging collisions apart and drawing a leader to each.

    Stacking labels straight onto crowded points detaches them from their marks,
    so labels are separated in display space and any that moved keeps a thin
    connector back to its dot.
    """
    if frame.empty:
        return
    points = [(ax.transData.transform((row[xcol], row[ycol])), row[labels])
              for _, row in frame.iterrows()]
    points.sort(key=lambda item: item[0][1], reverse=True)

    # Walking top to bottom and keeping each label at least one gap below the
    # last one placed guarantees no two overlap -- checking against every placed
    # label independently does not, because one nudge can create a new collision.
    placed = []
    lowest = None
    for (px, py), text in points:
        target = py if lowest is None else min(py, lowest - min_gap_px)
        lowest = target
        placed.append(((px, py), target, text))

    for (px, py), target, text in placed:
        anchor_x, anchor_y = ax.transData.inverted().transform((px, py))
        moved = abs(target - py) > 0.5
        ax.annotate(text, xy=(anchor_x, anchor_y), xytext=(8, target - py),
                    textcoords='offset points', fontsize=7.5, color=INK_SOFT,
                    va='center', ha='left', zorder=6,
                    arrowprops=dict(arrowstyle='-', color=GRID, linewidth=1,
                                    shrinkA=0, shrinkB=3) if moved else None)


def label_extremes(ax, frame, xcol, ycol, labels):
    """Label the corners of a cloud, turning each label inward off the edge.

    A label placed outward at the extremes either overflows the axes or lands on
    its neighbour, so the side is chosen from which half the point sits in.
    """
    midpoint = (frame[xcol].max() + frame[xcol].min()) / 2
    for _, row in frame.iterrows():
        inward = row[xcol] > midpoint
        ax.annotate(row[labels], (row[xcol], row[ycol]),
                    textcoords='offset points',
                    xytext=(-8 if inward else 8, 0),
                    ha='right' if inward else 'left',
                    fontsize=7.5, color=INK_SOFT, va='center', zorder=5)


def caption(fig, text):
    fig.text(0.5, 0.015, text, ha='center', fontsize=8, color=INK_MUTED)


# --------------------------------------------------------------------------
# Figure 1 -- calibration
# --------------------------------------------------------------------------

def fig_calibration(pooled, fits, season, outdir):
    """Published score against the driver rebuilt from player data."""
    panels = [('fpl_overall', 'overall', 'blended goal difference per match'),
              ('fpl_attack', 'attack', 'blended goal involvements per match'),
              ('fpl_defence', 'defence', 'blended goals conceded per match')]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.3))

    for ax, (score, kind, xlabel) in zip(axes, panels):
        _, _, weight, within_r2, _, _ = fits[score]
        driver = pc.blended_driver(pooled, weight, kind)
        ax.grid(axis='both', alpha=0.9)
        ax.set_axisbelow(True)
        fit_line(ax, driver.to_numpy(), pooled[score].to_numpy())
        scatter(ax, driver, pooled[score])

        # Label the two extremes only -- the cloud carries the relationship, and
        # the season tag keeps a repeat team from looking like a duplicate.
        marked = pooled.assign(_d=driver,
                               _label=pooled.team + '  ' + pooled.season.str[2:])
        extremes = pd.concat([marked.nlargest(1, score), marked.nsmallest(1, score)])
        label_extremes(ax, extremes, '_d', score, '_label')
        ax.margins(x=0.08)

        ax.set_title(f'{kind.capitalize()}   R² = {within_r2:.2f}', loc='left')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('published FPL strength score' if kind == 'overall' else '')

    fig.suptitle('The player data rebuilds the overall score well, the split legs less so',
                 x=0.5, y=0.98, fontsize=13, fontweight='semibold', color=INK)
    caption(fig, f'One dot per team-season, {len(pooled)} across '
                 f'{pooled.season.nunique()} seasons. Line is least squares; '
                 f'R² is within-season.')
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    path = os.path.join(outdir, '1-calibration.png')
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Figure 2 -- the blend weight
# --------------------------------------------------------------------------

def fig_blend_weight(pooled, outdir):
    """R-squared as the driver shifts from expected output to realised output."""
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    ax.grid(axis='y', alpha=0.9)
    ax.set_axisbelow(True)

    series = [('fpl_overall', 'overall', 'Overall'),
              ('fpl_attack', 'attack', 'Attack'),
              ('fpl_defence', 'defence', 'Defence')]
    peaks = []

    for colour, (score, kind, label) in zip(SERIES, series):
        curve = np.array([pc._within_season_fit(
            pooled, pc.blended_driver(pooled, w, kind), score)[1]
            for w in pc.BLEND_GRID])
        ax.plot(pc.BLEND_GRID, curve, color=colour, linewidth=2,
                solid_capstyle='round', label=label, zorder=3)

        peak = int(np.argmax(curve))
        peaks.append((label, pc.BLEND_GRID[peak], curve[peak], colour))
        ax.scatter([pc.BLEND_GRID[peak]], [curve[peak]], s=64, color=colour,
                   edgecolor=SURFACE, linewidth=2, zorder=5)

    # Headroom first, so the topmost peak label has somewhere to sit.
    low = min(p[2] for p in peaks)
    high = max(p[2] for p in peaks)
    ax.set_ylim(low - 0.09, high + 0.035)
    ax.set_xlim(-0.02, 1.02)

    for label, weight, value, _ in peaks:
        # Every series is directly labelled: slot 3 needs it for contrast, and
        # the peak is the number the reader came for.
        ax.annotate(f'{label}  peak R² {value:.2f} at {weight:.2f}',
                    (weight, value), textcoords='offset points', xytext=(0, 11),
                    ha='center', fontsize=8.5, color=INK_SOFT, zorder=6)

    ax.set_xlabel('weight on realised output  (0 = expected output only, '
                  '1 = actual goals only)')
    ax.set_ylabel('within-season R² against the published score')
    ax.set_title('Every published score is explained best by a blend,\n'
                 'not by expected output alone', loc='left', fontsize=12.5)
    ax.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.235))
    caption(fig, 'Peaks sit in the middle: the scores respond to what teams did '
                 'and to how good their chances were.')
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    path = os.path.join(outdir, '2-blend-weight.png')
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Figure 3 -- the finishing premium
# --------------------------------------------------------------------------

def fig_finishing_premium(gw, fits, season, outdir, n=9):
    """Dumbbell: attacking points on expected output only, then blended."""
    _, slope, weight, _, _, _ = fits['fpl_attack']
    expected = pc.attack_contributions(gw, slope, 0.0).set_index(['team', 'name'])
    blended = pc.attack_contributions(gw, slope, weight).set_index(['team', 'name'])

    moved = blended[['minutes', 'goals', 'assists', 'xG', 'xA', 'attack_points']].copy()
    moved['expected_only'] = expected.attack_points
    moved['delta'] = moved.attack_points - moved.expected_only
    moved = moved[moved.minutes >= 900]
    short = pc.load_web_names(season, gw)

    fig, ax = plt.subplots(figsize=(9.4, 7.6))
    ax.grid(axis='x', alpha=0.9)
    ax.set_axisbelow(True)

    # Worst movers at the bottom of the axis, best at the top, so the move runs
    # monotonically up the chart.
    order = (list(moved.nsmallest(n, 'delta').index)
             + list(moved.nlargest(n, 'delta').index)[::-1])
    rows = moved.loc[order]
    ys = np.arange(len(rows))

    for y, (_, row) in zip(ys, rows.iterrows()):
        ax.plot([row.expected_only, row.attack_points], [y, y],
                color=GRID, linewidth=2, solid_capstyle='round', zorder=2)
    ax.scatter(rows.expected_only, ys, s=70, color=SHADE_LIGHT,
               edgecolor=SURFACE, linewidth=2, zorder=4,
               label='expected output only')
    ax.scatter(rows.attack_points, ys, s=70, color=SERIES[0],
               edgecolor=SURFACE, linewidth=2, zorder=5,
               label='blended (what the model uses)')

    ticks = [f'{short[name]}   {int(row.goals)}G {int(row.assists)}A  vs  '
             f'{row.xG:.1f} xG {row.xA:.1f} xA'
             for (team, name), row in rows.iterrows()]
    ax.set_yticks(ys)
    ax.set_yticklabels(ticks, fontsize=8)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.axvline(0, color=GRID, linewidth=1, zorder=1)

    ax.set_xlabel('attacking contribution (FPL strength points above replacement)')
    ax.set_title('Finishers gain, wasteful finishers lose\n'
                 f'largest moves either way, {season}', loc='left', fontsize=12.5)
    ax.legend(loc='lower right')
    caption(fig, 'Each row is one player: pale dot is credit from chance quality '
                 'alone, solid dot is credit once goals count.')
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    path = os.path.join(outdir, '3-finishing-premium.png')
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Figure 4 -- squad concentration
# --------------------------------------------------------------------------

def fig_concentration(report, season, outdir, floor=50.0):
    """How much of each squad's attributed credit sits with one player.

    A squad that barely cleared replacement level has a tiny denominator, so its
    share is arithmetic noise -- Southampton's best 2024-25 contributor holds a
    third of six points.  Those squads are dropped and named in the caption
    rather than shown as the most one-man teams in the league.
    """
    rows = []
    for team, squad in report.groupby('team'):
        positive = squad[squad.overall_points > 0]
        total = positive.overall_points.sum()
        if total <= 0:
            continue
        best = positive.nlargest(1, 'overall_points').iloc[0]
        rows.append({'team': team, 'total': total,
                     'share': 100 * best.overall_points / total,
                     'player': best.web_name, 'points': best.overall_points})
    everyone = pd.DataFrame(rows)
    frame = everyone[everyone.total >= floor].sort_values('share')
    dropped = everyone[everyone.total < floor].sort_values('total')

    fig, ax = plt.subplots(figsize=(9.0, 7.4))
    ax.grid(axis='x', alpha=0.9)
    ax.set_axisbelow(True)
    ax.set_xlim(0, frame.share.max() * 1.22)
    ax.set_ylim(-0.8, len(frame) - 0.2)
    ax.set_yticks(np.arange(len(frame)))
    ax.set_yticklabels([f'{r.team}   {r.player}' for r in frame.itertuples()],
                       fontsize=8.5)
    fig.canvas.draw()

    height = bar_height(ax)
    for y, row in enumerate(frame.itertuples()):
        rounded_barh(ax, y, 0, row.share, height, SERIES[0])
    # Label the ends of the range -- the axis carries the rest.
    for y in (0, len(frame) - 1):
        row = frame.iloc[y]
        ax.annotate(f'{row.share:.0f}%', (row.share, y), textcoords='offset points',
                    xytext=(7, 0), va='center', fontsize=8.5, color=INK,
                    fontweight='semibold')

    ax.set_xlabel("biggest contributor's share of the squad's attributed "
                  'overall points (%)')
    ax.set_title(f'Which teams lean on one player, {season}',
                 loc='left', fontsize=12.5)
    note = ('Share of all positive player contributions held by the single '
            'largest contributor.')
    if len(dropped):
        excluded = ', '.join(f'{r.team} ({r.total:.0f})' for r in dropped.itertuples())
        note += (f'\nExcluded for having under {floor:.0f} attributed points in '
                 f'total, which makes any share meaningless: {excluded}.')
    caption(fig, note)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    path = os.path.join(outdir, '4-concentration.png')
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Figure 5 -- attack against defence, by position
# --------------------------------------------------------------------------

def fig_by_position(report, season, outdir):
    """Small multiples, so each panel is a single series on a shared scale."""
    positions = [('FWD', 'Forwards'), ('MID', 'Midfielders'),
                 ('DEF', 'Defenders'), ('GK', 'Goalkeepers')]
    regulars = report[report.minutes >= pc.MIN_MINUTES]
    shown = {code for code, _ in positions}
    missing = regulars[~regulars.position.isin(shown)].position.value_counts()

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.4), sharex=True, sharey=True)
    xlim = (regulars.attack_points.min() - 6, regulars.attack_points.max() + 22)
    ylim = (regulars.defence_points.min() - 3, regulars.defence_points.max() + 4)

    for ax, (code, label) in zip(axes.ravel(), positions):
        group = regulars[regulars.position == code]
        ax.grid(alpha=0.9)
        ax.set_axisbelow(True)
        ax.axhline(0, color=GRID, linewidth=1, zorder=1)
        ax.axvline(0, color=GRID, linewidth=1, zorder=1)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        scatter(ax, group.attack_points, group.defence_points)
        best = group.nlargest(3, 'overall_points').copy()
        best['label'] = best.web_name
        ax.set_title(f'{label}   n = {len(group)}', loc='left')
        fig.canvas.draw()
        label_declutter(ax, best, 'attack_points', 'defence_points', 'label')

    for ax in axes[1]:
        ax.set_xlabel('attacking points')
    for ax in axes[:, 0]:
        ax.set_ylabel('defensive points')

    fig.suptitle('Attacking credit is measured directly; defensive credit is '
                 'regularised, so its range is narrower',
                 x=0.5, y=0.98, fontsize=12.5, fontweight='semibold', color=INK)
    note = (f'Players with {pc.MIN_MINUTES}+ minutes, {season}. Shared axes; '
            'three largest contributors labelled per panel.')
    if len(missing):
        note += ('  Not shown: '
                 + ', '.join(f'{code} ({count})' for code, count in missing.items()))
    caption(fig, note)
    fig.tight_layout(rect=(0, 0.035, 1, 0.945))
    path = os.path.join(outdir, '5-by-position.png')
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Figure 6 -- one squad, broken down
# --------------------------------------------------------------------------

def fig_squad(report, accounting, season, outdir, teams=('Liverpool', 'Arsenal'),
              top=11):
    """Stacked contribution per player, with the published score reconciled."""
    fig, axes = plt.subplots(1, len(teams), figsize=(12.8, 6.4), sharex=True)
    axes = np.atleast_1d(axes)

    # Both panels carry the same units side by side, so they share one scale --
    # otherwise Saka's bar would look the length of Salah's.
    squads = {team: report[(report.team == team)
                           & (report.minutes >= pc.MIN_MINUTES)]
              .nlargest(top, 'overall_points').iloc[::-1] for team in teams}
    reach = max((s.attack_points + s.defence_points.clip(lower=0)).max()
                for s in squads.values())
    floor = min(min(0, s.defence_points.min()) for s in squads.values())

    for ax, team in zip(axes, teams):
        squad = squads[team]
        row = accounting[accounting.team == team].iloc[0]

        ax.grid(axis='x', alpha=0.9)
        ax.set_axisbelow(True)
        ax.set_yticks(np.arange(len(squad)))
        ax.set_yticklabels(squad.web_name, fontsize=8.5)
        ax.set_ylim(-0.8, len(squad) - 0.2)
        ax.set_xlim(floor * 1.4 - 1, reach * 1.06)
        fig.canvas.draw()

        height = bar_height(ax)
        gap = px_to_data(ax, GAP_PX, 'x')
        for y, player in enumerate(squad.itertuples()):
            attack = player.attack_points
            defence = player.defence_points
            # Attack from the baseline; defence continues it, separated by a 2px
            # surface gap.  A negative defensive contribution runs left of zero.
            rounded_barh(ax, y, 0, attack, height, SERIES[0],
                         round_end=defence <= 0)
            if defence > 0:
                rounded_barh(ax, y, attack + gap, attack + defence, height,
                             SERIES[1])
            elif defence < 0:
                rounded_barh(ax, y, -gap, defence, height, SERIES[1])

        ax.axvline(0, color=GRID, linewidth=1, zorder=1)
        ax.set_xlabel('FPL strength points above replacement')
        ax.set_title(f'{team}   overall {row.fpl_overall:.0f}\n'
                     f'{row.overall_baseline:.0f} squad baseline + '
                     f'{row.overall_attributed:.0f} from players',
                     loc='left', fontsize=10.5)

    handles = [plt.Line2D([], [], marker='s', linestyle='none', markersize=8,
                          color=SERIES[0], label='attacking contribution'),
               plt.Line2D([], [], marker='s', linestyle='none', markersize=8,
                          color=SERIES[1], label='defensive contribution')]
    fig.legend(handles=handles, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.0))
    caption(fig, '')
    fig.suptitle(f'What each squad’s published score is made of, {season}',
                 x=0.5, y=0.98, fontsize=12.5, fontweight='semibold', color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    path = os.path.join(outdir, '6-squad-breakdown.png')
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--season', default='2024-25')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--outdir', default='figures')
    parser.add_argument('--teams', nargs='+', default=['Liverpool', 'Arsenal'])
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    style()

    pooled = pc.pooled_team_seasons(data_dir=args.data_dir)
    report, fits, _ = pc.build_report(args.season, args.data_dir)
    accounting = pc.team_accounting(report, args.season, fits, args.data_dir)
    gw = pc.load_gameweeks(args.season, args.data_dir)

    written = [
        fig_calibration(pooled, fits, args.season, args.outdir),
        fig_blend_weight(pooled, args.outdir),
        fig_finishing_premium(gw, fits, args.season, args.outdir),
        fig_concentration(report, args.season, args.outdir),
        fig_by_position(report, args.season, args.outdir),
        fig_squad(report, accounting, args.season, args.outdir, tuple(args.teams)),
    ]
    for path in written:
        print(f'wrote {path}')


if __name__ == '__main__':
    main()
