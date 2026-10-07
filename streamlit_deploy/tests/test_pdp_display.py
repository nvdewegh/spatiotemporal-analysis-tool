"""
Tests for the object-major display of PDP point matrices.

The display is a presentation-only permutation: every displayed cell must equal the
computed matrix at (perm[row], perm[col]), and the computation itself must not change.

Run from streamlit_deploy/:  python -m pytest tests
Checks on the real 1 Hz dataset run when PDP_1HZ_CSV points to that CSV file.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import pdp_analysis as p  # noqa: E402


def time_major_frame(n_objects, n_times, config='A', values=None):
    """Points in the current computational order: all objects at T0, then T1, ..."""
    rows = []
    for t in range(n_times):
        for o in range(n_objects):
            x, y = values(o, t) if values else (10.0 * o + t, 5.0 * t - o)
            rows.append({'config_source': config, 'tst': t, 'obj': o, 'x': x, 'y': y})
    return pd.DataFrame(rows)


def heatmaps(fig):
    return [t for t in fig.data if t.type == 'heatmap']


def assert_display_matches(trace, matrix, perm):
    z = np.array(trace.z)
    expected = np.asarray(matrix)[np.ix_(perm, perm)]
    assert z.shape == expected.shape
    for i in range(len(perm)):
        for j in range(len(perm)):
            assert z[i, j] == matrix[perm[i], perm[j]]


# ---------------------------------------------------------------- A. small known example

def small_example():
    # O0 at T1 and O1 at T2 share x = 7.0: an equality between distinct points
    xs = {(0, 0): 1.0, (1, 0): 9.0, (0, 1): 7.0, (1, 1): 8.0,
          (0, 2): 3.0, (1, 2): 7.0, (0, 3): 4.0, (1, 3): 2.0}
    return time_major_frame(2, 4, values=lambda o, t: (xs[(o, t)], 0.0))


def test_small_example_order_permutation_and_cells():
    df = small_example()
    meta = p.point_metadata(df, [0, 1, 2, 3])
    display = p.object_major_display(meta)
    assert list(display['perm']) == [0, 2, 4, 6, 1, 3, 5, 7]
    assert display['tick_labels'] == ['O0_T0', 'O0_T1', 'O0_T2', 'O0_T3', 'O1_T0', 'O1_T1', 'O1_T2', 'O1_T3']
    assert display['groups'] == [(0, 4, 'Object 0'), (4, 8, 'Object 1')]

    ineq = p.compute_inequality_matrix(df['x'].values, df['x'].values, 4)
    fig = make_subplots(rows=1, cols=1)
    p.add_point_matrix_heatmap(fig, ineq, display, 1, 1, 'x')
    assert_display_matches(heatmaps(fig)[0], ineq, display['perm'])
    # The distinct-point equality is retained (O0_T1 vs O1_T2 -> displayed (1, 6))
    assert np.array(heatmaps(fig)[0].z)[1, 6] == 1
    # Relation encoding unchanged: 0 means row < column
    assert np.array(heatmaps(fig)[0].z)[0, 4] == 0   # O0_T0 (1.0) < O1_T0 (9.0)


def test_computed_matrix_is_not_modified():
    df = small_example()
    ineq = p.compute_inequality_matrix(df['x'].values, df['x'].values, 4)
    before = ineq.copy()
    display = p.object_major_display(p.point_metadata(df, [0, 1, 2, 3]))
    p.add_point_matrix_heatmap(make_subplots(rows=1, cols=1), ineq, display, 1, 1, 'x')
    assert np.array_equal(ineq, before)


def test_timestamps_sorted_numerically_not_lexically():
    df = time_major_frame(1, 12)
    meta = p.point_metadata(df, list(range(12)))
    display = p.object_major_display(meta)
    assert [meta[i]['t_rel'] for i in display['perm']] == list(range(12))   # T10 after T9, not after T1


# ---------------------------------------------------------------- B. orientation

def test_first_row_on_top_and_diagonal_top_left_to_bottom_right():
    df = small_example()
    display = p.object_major_display(p.point_metadata(df, [0, 1, 2, 3]))
    ineq = p.compute_inequality_matrix(df['x'].values, df['x'].values, 4)
    fig = make_subplots(rows=1, cols=1)
    p.add_point_matrix_heatmap(fig, ineq, display, 1, 1, 'x')
    assert fig.layout.yaxis.autorange == 'reversed'      # y index 0 drawn at the top
    assert fig.layout.xaxis.side == 'top'                # column labels above the matrix
    z = np.array(heatmaps(fig)[0].z)
    assert all(z[k, k] == 1 for k in range(len(z)))      # self-comparison diagonal
    assert list(heatmaps(fig)[0].y) == list(range(8))    # z rows are not reversed or transposed


# ---------------------------------------------------------------- C. real 1 Hz input

CSV = os.environ.get('PDP_1HZ_CSV')
needs_csv = pytest.mark.skipif(not CSV or not os.path.exists(CSV), reason='set PDP_1HZ_CSV to the 1 Hz dataset')


def load_1hz():
    from modules import utils
    with open(CSV, 'rb') as f:
        return utils.load_data(f, update_state=False, show_success=False)


@needs_csv
@pytest.mark.parametrize('window_length, size, block', [(6, 12, 6), (3, 6, 3)])
def test_1hz_matrix_size_and_blocks(window_length, size, block):
    df = load_1hz()
    fig = p.visualize_inequality_matrices(df, ['Config_1', 'Config_3'], [0, 1], 0, 5,
                                          window_length=window_length, window_indices=[0])
    traces = heatmaps(fig)
    assert len(traces) == 4
    for trace in traces:
        assert np.array(trace.z).shape == (size, size)
    labels = list(fig.layout.xaxis.ticktext)
    assert labels == [f'O0_T{t}' for t in range(block)] + [f'O1_T{t}' for t in range(block)]
    separators = [s for s in fig.layout.shapes if s.xref == 'x' and s.yref == 'y']
    assert len(separators) == 2 and {s.x0 for s in separators} | {s.y0 for s in separators} >= {block - 0.5}


@needs_csv
def test_1hz_displayed_values_match_computation():
    df = load_1hz()
    fig = p.visualize_inequality_matrices(df, ['Config_1'], [0, 1], 0, 5, window_length=6, window_indices=[0])
    window = df[df['config_source'] == 'Config_1'].sort_values(['tst', 'obj'])
    perm = [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11]
    for trace, dim in zip(heatmaps(fig), ['x', 'y']):
        ineq = p.compute_inequality_matrix(window[dim].values, window[dim].values, 6)
        assert_display_matches(trace, ineq, perm)


# ---------------------------------------------------------------- D. comparisons

def comparison(buffer=0.0, rough=0.0, external=None, n_objects=2, n_times=6, window_length=3):
    df = pd.concat([time_major_frame(n_objects, n_times, 'A'),
                    time_major_frame(n_objects, n_times, 'B', values=lambda o, t: (3.0 * t - 4 * o, 2.0 * o + t))])
    _, diff_result, _ = p.create_difference_visualization(
        df, 'A', 'B', list(range(n_objects)), 0, n_times - 1, window_length,
        buffer, buffer, rough, rough, external, show_court=False)
    return diff_result


@pytest.mark.parametrize('kwargs', [{}, {'rough': 0.5}, {'buffer': 0.3}, {'external': [('Pole', 1.0, 2.0)]}])
def test_comparison_views_share_one_order(kwargs):
    diff_result = comparison(**kwargs)
    windows = [w['window_idx'] for w in diff_result['differences']]
    fig = p.create_difference_matrices_figure(diff_result, 'A', 'B', windows)
    traces = heatmaps(fig)
    assert len(traces) == 6 * len(windows)
    for w, window in enumerate(diff_result['differences']):
        perm = p.object_major_display(window['point_meta1'])['perm']
        sources = ['ineq_x1', 'ineq_x2', 'x_diff_matrix', 'ineq_y1', 'ineq_y2', 'y_diff_matrix']
        for trace, key in zip(traces[6 * w:6 * w + 6], sources):
            assert_display_matches(trace, window[key], perm)
        # Each displayed difference cell is the difference of the two displayed relations
        x1, x2, xd = (np.array(t.z) for t in traces[6 * w:6 * w + 3])
        assert np.array_equal(xd, np.abs(x1 - x2))


def test_comparison_distance_equals_sum_of_window_differences():
    df = pd.concat([time_major_frame(2, 6, 'A'),
                    time_major_frame(2, 6, 'B', values=lambda o, t: (3.0 * t - 4 * o, 2.0 * o + t))])
    a, b = df[df['config_source'] == 'A'], df[df['config_source'] == 'B']
    distance = p.compute_pdp_distance_pair(a, b, 3)
    result = p.compute_pairwise_inequality_differences(a, b, 3)
    assert distance == sum(w['x_diff_matrix'].sum() + w['y_diff_matrix'].sum() for w in result['differences'])


# ---------------------------------------------------------------- E. other supported cases

def test_single_object():
    display = p.object_major_display(p.point_metadata(time_major_frame(1, 4), [0, 1, 2, 3]))
    assert list(display['perm']) == [0, 1, 2, 3]
    assert display['groups'] == [(0, 4, 'Object 0')]


def test_three_objects_boundaries_from_metadata():
    df = time_major_frame(3, 3)
    display = p.object_major_display(p.point_metadata(df, [0, 1, 2]))
    assert list(display['perm']) == [0, 3, 6, 1, 4, 7, 2, 5, 8]
    assert [g[:2] for g in display['groups']] == [(0, 3), (3, 6), (6, 9)]


def test_multiple_windows_use_window_relative_labels_and_input_timestamps():
    df = time_major_frame(2, 5)
    fig = p.visualize_inequality_matrices(df, ['A'], [0, 1], 0, 4, window_length=3, window_indices=[0, 2])
    traces = heatmaps(fig)
    assert len(traces) == 4
    assert 'input TST 2' in traces[2].customdata[0][0]   # window 2 starts at input timestamp 2


def test_rough_equality_explained():
    df = time_major_frame(2, 3, values=lambda o, t: (t + 0.1 * o, 0.0))
    fig = p.visualize_inequality_matrices(df, ['A'], [0, 1], 0, 2, window_length=3, rough_x=0.2,
                                          window_indices=[0])
    x_trace = heatmaps(fig)[0]
    assert 'within tolerance' in x_trace.customdata[0][3]   # O0_T0 vs O1_T0 differ by 0.1 <= 0.2
    assert any('within rough tolerance' in (t.name or '') for t in fig.data)


def test_buffer_points_retained_and_grouped():
    df = time_major_frame(2, 3)
    fig = p.visualize_inequality_matrices(df, ['A'], [0, 1], 0, 2, window_length=3, buffer_x=0.3, buffer_y=0.3,
                                          window_indices=[0])
    trace = heatmaps(fig)[0]
    assert np.array(trace.z).shape == (30, 30)            # 2 objects x 3 times x 5 subpoints
    labels = list(fig.layout.xaxis.ticktext)
    assert labels[:5] == ['O0_T0_L', 'O0_T0_R', 'O0_T0', 'O0_T0_B', 'O0_T0_U']
    assert all(l.startswith('O0_') for l in labels[:15]) and all(l.startswith('O1_') for l in labels[15:])
    assert len(set(labels)) == 30                          # no duplicate point identities


def test_external_points_get_their_own_groups_after_objects():
    df = time_major_frame(2, 3)
    fig = p.visualize_inequality_matrices(df, ['A'], [0, 1], 0, 2, window_length=3,
                                          external_points=[('Pole', 1.0, 2.0), ('Gate', 5.0, 0.0)],
                                          window_indices=[0])
    labels = list(fig.layout.xaxis.ticktext)
    assert len(labels) == 12
    assert labels[6:] == ['E:Pole_T0', 'E:Pole_T1', 'E:Pole_T2', 'E:Gate_T0', 'E:Gate_T1', 'E:Gate_T2']
    separators = [s for s in fig.layout.shapes if s.xref == 'x' and s.yref == 'y' and s.x0 == s.x1]
    assert sorted(s.x0 for s in separators) == [2.5, 5.5, 8.5]


# ---------------------------------------------------------------- fixes for issues found

def test_distance_cache_notices_a_different_dataset():
    a = time_major_frame(2, 6, 'A')
    b = time_major_frame(2, 6, 'B', values=lambda o, t: (3.0 * t - 4 * o, 2.0 * o + t))
    data1 = pd.concat([a, b])
    data2 = pd.concat([a, time_major_frame(2, 6, 'B', values=lambda o, t: (-2.0 * t + o, 1.0 * o - t))])
    d1, _ = p.compute_pdp_distance_matrix(data1, ('A', 'B'), (0, 1), 0, 5, window_length=3)
    d2, _ = p.compute_pdp_distance_matrix(data2, ('A', 'B'), (0, 1), 0, 5, window_length=3)
    direct = p.compute_pdp_distance_pair(data2[data2['config_source'] == 'A'],
                                         data2[data2['config_source'] == 'B'], 3)
    assert d2[0, 1] == direct and d1[0, 1] != d2[0, 1]


def test_difference_positions_use_original_points_with_buffer():
    df = time_major_frame(2, 3)
    a = p.apply_buffer_to_trajectories(df, 0.3, 0.3)
    result = p.compute_pairwise_inequality_differences(a, a, 3)
    for (obj, tst), (x, y) in result['differences'][0]['positions1'].items():
        original = df[(df['obj'] == obj) & (df['tst'] == tst)].iloc[0]
        assert (x, y) == (original['x'], original['y'])


def test_more_than_ten_objects_with_external_points_sort_numerically():
    df = time_major_frame(12, 3)
    ext = p.add_external_points_to_data(df, [('Pole', 1.0, 2.0)], [0, 1, 2])   # obj becomes text
    ordered = p.sort_points(ext[ext['tst'] == 0], ['tst', 'obj', 'sub_order'])
    assert list(ordered['obj']) == [str(o) for o in range(12)] + ['EXT_0']
    # Distances do not depend on the input row order
    other = time_major_frame(12, 3, 'B', values=lambda o, t: (2.0 * t - o, o + 0.5 * t))
    ext_b = p.add_external_points_to_data(other, [('Pole', 1.0, 2.0)], [0, 1, 2])
    d = p.compute_pdp_distance_pair(ext, ext_b, 2)
    assert d == p.compute_pdp_distance_pair(ext.sample(frac=1, random_state=1), ext_b.sample(frac=1, random_state=2), 2)
