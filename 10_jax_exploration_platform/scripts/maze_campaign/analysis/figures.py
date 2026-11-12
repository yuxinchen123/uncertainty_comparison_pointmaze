"""Training-curve figures: small multiples of seed-mean curves with bands of one standard error.

Chart rules followed here: one hue per series, taken in a fixed order and never cycled; a legend whenever two
or more series are drawn; thin lines; one y axis; one panel per environment (or per setting); a recessive
grid; the baseline and "the other configurations" in neutral greys. Text is never drawn in a series colour.
"""

import math
import re

import numpy as np

# Hues in their fixed order (the validated order of the chart palette); a series beyond the last is not drawn
# in an invented hue.
SERIES_COLORS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948')
BASELINE_COLOR = '#52514e'
OTHER_COLOR = '#c3c2b7'
INK, SECONDARY, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'


def file_slug(*parts):
    """A file name from words: lower case, words joined by hyphens, parts joined by underscores."""
    # before: ('training curves', 'reward 1 on arrival', 'CleanRL PPO-RND'); after:
    # 'training-curves_reward-1-on-arrival_cleanrl-ppo-rnd'.
    return '_'.join(re.sub(r'[^a-z0-9.]+', '-', str(part).lower()).strip('-') for part in parts)


def series_color(index):
    """The hue of the series at a position in the fixed order."""
    if index >= len(SERIES_COLORS):
        raise ValueError(f'A figure has more than {len(SERIES_COLORS)} coloured series; split it into panels.')
    return SERIES_COLORS[index]


def step_unit(largest):
    """The unit of the step axis: millions from two million steps on, else thousands."""
    return (1e6, 'millions') if largest >= 2e6 else (1e3, 'thousands')


def curve_figure(panels, legend, plots, name, y_label):
    """Draw seed-mean curves in small multiples and save them as PNG and PDF; returns the file name's stem.

    panels: [{'title': words, 'series': [{'steps', 'mean', 'error', 'color', 'width', 'band', 'order'}]}]
    legend: [{'label': words, 'color': hue, 'width': line width, 'band': whether it has a band}], drawn once
            above the panels when it has two or more entries.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    # At most three panels per row, so a row fits the width of a page.
    columns = min(3, len(panels))
    rows = math.ceil(len(panels) / columns)
    long_labels = max(len(entry['label']) for entry in legend) > 44
    legend_columns = 1 if long_labels else min(len(legend), 3)
    legend_height = 0.22 * math.ceil(len(legend) / legend_columns) + 0.12 if len(legend) > 1 else 0.0
    width = {1: 3.9, 2: 5.4, 3: 6.9}[columns]
    height = 2.15 * rows + 0.35 + legend_height
    figure, axes = plt.subplots(rows, columns, figsize=(width, height), sharey=True, squeeze=False)
    scale, unit = step_unit(max(series['steps'][-1] for panel in panels for series in panel['series']))

    for axis, panel in zip(axes.flat, panels):
        for series in panel['series']:
            # The curve is a step function: each value holds over the window that ends at its step, and the
            # first window starts at step 0. before: steps [100, 200], mean [0.1, 0.4]; after: steps
            # [0, 100, 200], mean [0.1, 0.1, 0.4], drawn so that 0.1 holds on (0, 100] and 0.4 on (100, 200].
            steps = np.concatenate([[0.0], np.asarray(series['steps'], np.float64)]) / scale
            mean = np.concatenate([series['mean'][:1], series['mean']])
            error = np.concatenate([series['error'][:1], series['error']])
            axis.step(steps, mean, where='pre', color=series['color'], linewidth=series['width'],
                      zorder=series['order'])
            if series['band'] and np.all(np.isfinite(error)):
                axis.fill_between(steps, mean - error, mean + error, step='pre', color=series['color'],
                                  alpha=0.18, linewidth=0, zorder=series['order'] - 0.5)
        axis.set_title(panel['title'], fontsize=8.5, color=INK)
        axis.set_xlim(left=0)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
        axis.grid(color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
        axis.tick_params(labelsize=7.5, length=0, labelcolor=SECONDARY)
        for name_of_side in ('top', 'right'):
            axis.spines[name_of_side].set_visible(False)
        for name_of_side in ('left', 'bottom'):
            axis.spines[name_of_side].set_color(AXIS)
    for axis in axes.flat[len(panels):]:
        axis.set_visible(False)
    for axis in axes[:, 0]:
        axis.set_ylabel(y_label, fontsize=8, color=INK)
    figure.supxlabel(f'environment steps per policy ({unit})', fontsize=8, color=INK)

    # One legend for the whole figure, above the panels, so it never covers a curve.
    if len(legend) > 1:
        handles = [Line2D([], [], color=entry['color'], linewidth=max(entry['width'], 1.2), label=entry['label'])
                   for entry in legend]
        figure.legend(handles=handles, loc='upper center', ncol=legend_columns, frameon=False, fontsize=7.5,
                      labelcolor=INK, bbox_to_anchor=(0.5, 1.0), handlelength=1.8, columnspacing=1.4)
    figure.tight_layout(rect=(0, 0, 1, 1 - legend_height / height))
    figure.savefig(plots / f'{name}.png', dpi=170)
    # A fixed creation date keeps the PDF's bytes the same when the data are the same.
    figure.savefig(plots / f'{name}.pdf', metadata={'CreationDate': None})
    plt.close(figure)
    return name
