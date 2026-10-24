"""
Dense PDP matrices keep one position per point even when labels repeat, and the detailed
difference panels share linked zooming with square cells. Synthetic data only.

Run from streamlit_deploy/:  python -m pytest tests
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules import pdp_analysis as p  # noqa: E402


def window_with_external_points(n_times, external):
    df = pd.DataFrame([{'config_source': 'A', 'tst': t, 'obj': o, 'x': 1.0 * t + o, 'y': 0.5 * t - o}
                       for t in range(n_times) for o in range(2)])
    ext = p.add_external_points_to_data(df, external, list(range(n_times)))
    return p.sort_points(ext, ['tst', 'obj', 'sub_order'])


def matrix_trace(window, n_times):
    meta = p.point_metadata(window, list(range(n_times)))
    display = p.object_major_display(meta)
    matrix = p.compute_inequality_matrix(window['x'].values, window['x'].values, n_times)
    fig = make_subplots(rows=1, cols=1)
    p.add_point_matrix_heatmap(fig, matrix, display, 1, 1, 'x', panel_name='Window 0')
    return fig, fig.data[0], matrix, display


SAME_NAME = [('Pole', 10.0, 2.0), ('Pole', 40.0, -3.0)]   # two distinct points, one display name


def test_repeated_labels_stay_separate_in_dense_matrix():
    window = window_with_external_points(60, SAME_NAME)      # 2 objects + 2 external points x 60 = 240 > 128
    fig, trace, matrix, display = matrix_trace(window, 60)
    n = len(matrix)
    assert n == 240 and trace.customdata is None              # dense path
    assert len(display['hover_labels']) - len(set(display['hover_labels'])) == 60   # labels really repeat
    assert list(trace.x) == list(range(n)) and list(trace.y) == list(range(n))     # N distinct positions
    np.testing.assert_array_equal(np.array(trace.z), matrix[np.ix_(display['perm'], display['perm'])])
    # The two same-named points remain different rows (their coordinates differ, so their relations do)
    perm = list(display['perm'])
    names = window['external_name'].fillna('').to_numpy()
    objs = window['obj'].astype(str).to_numpy()
    pole_rows = [k for k, i in enumerate(perm) if names[i] == 'Pole' and window['tst'].to_numpy()[i] == 0]
    assert len(pole_rows) == 2 and {objs[perm[k]] for k in pole_rows} == {'EXT_0', 'EXT_1'}
    z = np.array(trace.z)
    assert not np.array_equal(z[pole_rows[0]], z[pole_rows[1]])
    # Identities stay available: compact index-to-point lookup, no per-cell strings
    tables = p.point_lookup_tables(fig)
    table = tables['Window 0']
    assert len(table) == n and list(table['Point']) == display['hover_labels']
    assert len(fig.to_json()) < 500_000


def test_dense_and_ordinary_paths_are_consistent():
    for n_times, dense in ((30, False), (60, True)):          # 120 points (ordinary), 240 points (dense)
        window = window_with_external_points(n_times, SAME_NAME)
        fig, trace, matrix, display = matrix_trace(window, n_times)
        n = len(matrix)
        assert (trace.customdata is None) == dense
        assert list(trace.x) == list(range(n)) and list(trace.y) == list(range(n))
        np.testing.assert_array_equal(np.array(trace.z), matrix[np.ix_(display['perm'], display['perm'])])
        ticks = fig.layout.xaxis
        assert [t.split(' ', 1)[-1] if dense else t for t in ticks.ticktext] == \
               [display['tick_labels'][k] for k in ticks.tickvals]
        assert fig.layout.yaxis.autorange == 'reversed'
        assert bool(p.point_lookup_tables(fig)) == dense


def comparison_figure(n_times=8, window_length=4, windows=(0, 2)):
    rows = []
    for config, shift in (('A', 0.0), ('B', 1.5)):
        for t in range(n_times):
            for o in range(2):
                rows.append({'config_source': config, 'tst': t, 'obj': o,
                             'x': t + o * shift, 'y': (t - o) * (1 + shift)})
    df = pd.DataFrame(rows)
    a, b = df[df.config_source == 'A'], df[df.config_source == 'B']
    result = p.compute_pairwise_inequality_differences(a, b, window_length)
    return p.create_difference_matrices_figure(result, 'A', 'B', list(windows))


def test_difference_panels_share_linked_zoom_with_square_master():
    fig = comparison_figure()
    layout = fig.layout.to_plotly_json()
    xaxes = sorted(k for k in layout if k.startswith('xaxis'))
    yaxes = sorted(k for k in layout if k.startswith('yaxis'))
    assert len(xaxes) == len(yaxes) == 12                    # 2 windows x 6 panels
    assert fig.layout.yaxis.scaleanchor == 'x' and fig.layout.yaxis.scaleratio == 1
    assert fig.layout.xaxis.matches is None and fig.layout.yaxis.matches is None
    for name in xaxes[1:]:
        assert fig.layout[name].matches == 'x'
    for name in yaxes[1:]:
        assert fig.layout[name].matches == 'y'
        assert fig.layout[name].scaleanchor is None           # never match AND anchor one axis
        assert fig.layout[name].autorange == 'reversed'       # first row stays on top
    assert fig.layout.yaxis.autorange == 'reversed'
    sizes = {len(t.z) for t in fig.data if t.type == 'heatmap'}
    assert len(sizes) == 1                                    # linked panels have the same dimensions


def test_difference_lines_display_budget_keeps_full_summary():
    rows = []
    for config, sign in (('A', 1.0), ('B', -1.0)):
        for t in range(40):
            for o in range(2):
                rows.append({'config_source': config, 'tst': t, 'obj': o, 'x': sign * (t + 3 * o), 'y': sign * (t - o)})
    df = pd.DataFrame(rows)
    fig, result, summary = p.create_difference_visualization(df, 'A', 'B', [0, 1], 0, 39, 40, show_court=False)
    assert summary['position_differences'] > p.MAX_DRAWN_DIFFERENCE_LINES       # many differing pairs
    assert summary['drawn_position_differences'] == p.MAX_DRAWN_DIFFERENCE_LINES
    a, b = df[df.config_source == 'A'], df[df.config_source == 'B']
    assert summary['total_cell_differences'] == p.compute_pdp_distance_pair(a, b, 40)   # summary counts everything
    difference_traces = [t for t in fig.data if (t.name or '').startswith('Diff #')]
    assert len(difference_traces) <= 2 * p.MAX_DRAWN_DIFFERENCE_LINES
