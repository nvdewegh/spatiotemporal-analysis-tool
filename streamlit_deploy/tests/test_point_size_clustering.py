"""
Tests for the Point size and clustering view (modules/point_size_clustering.py).

Small synthetic fixtures are used only here, for targeted cases. The A2 check runs on the real course
dataset when PDP_A2_CSV points to A2_1Hz_dataset.csv.
Run from streamlit_deploy/:  python -m pytest tests/test_point_size_clustering.py
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules import point_size_clustering as psc  # noqa: E402


def frame(rows):
    return pd.DataFrame(rows, columns=['config_source', 'obj', 'tst', 'x', 'y'])


def check_consistency(xy, method):
    """Every reachable cut: single linkage equals linked circles; scores match sklearn on the same D."""
    _, D, Z = psc.compute_hierarchy(xy, method)
    for d in sorted(set(np.round(np.linspace(0, D.max() * 1.01, 60), 9)) | set(Z[:, 2])):
        labels = psc.cut_at_distance(Z, d)
        n_comp, comp = psc.circle_components(D, d)
        if method == 'single':
            assert psc.same_partition(labels, comp), d
            assert n_comp == len(np.unique(labels))
        score = psc.mean_silhouette(D, labels)
        k = len(np.unique(labels))
        if 2 <= k <= len(xy) - 1:
            assert score == pytest.approx(silhouette_score(xy, labels, metric='euclidean'))
        else:
            assert score is None
    return D, Z


@pytest.mark.parametrize('method', psc.LINKAGES)
def test_random_points_consistent(method):
    rng = np.random.default_rng(3)
    xy = np.vstack([rng.normal(c, 1.0, (15, 2)) for c in ((0, 0), (8, 1), (3, 9))])
    check_consistency(xy, method)


def test_single_linkage_exact_at_touching_distance():
    # Points exactly d apart: circles touch, so they are linked and in one cluster
    xy = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [10.0, 0.0]])
    _, D, Z = psc.compute_hierarchy(xy, 'single')
    labels = psc.canonical_labels(psc.cut_at_distance(Z, 2.0))
    assert list(labels) == [0, 0, 0, 1]
    assert psc.circle_components(D, 2.0)[0] == 2
    assert list(psc.canonical_labels(psc.cut_at_distance(Z, 1.999))) == [0, 1, 2, 3]


def test_complete_linkage_may_differ_and_is_reported():
    xy = np.array([[0.0, 0], [1, 0], [2, 0], [3, 0]])     # chain with equal steps
    _, D, Z = psc.compute_hierarchy(xy, 'complete')
    labels = psc.cut_at_distance(Z, 1.0)
    n_comp, comp = psc.circle_components(D, 1.0)
    assert n_comp == 1 and len(np.unique(labels)) > 1
    assert not psc.same_partition(labels, comp)


def test_tied_merge_distances_skip_unreachable_counts():
    # Square: all four nearest-neighbour distances equal, so 3 and 2 clusters cannot come from a cut
    xy = np.array([[0.0, 0], [1, 0], [0, 1], [1, 1]])
    _, D, Z = psc.compute_hierarchy(xy, 'single')
    cuts = psc.reachable_cuts(Z)
    assert set(cuts) == {4, 1}
    assert psc.silhouette_curve(D, Z, 10) == []        # no reachable k in 2..n-1, no forced split


def test_coincident_points_kept_and_singletons():
    df = frame([('C1', 0, 0, 1.0, 1.0), ('C2', 0, 0, 1.0, 1.0), ('C3', 0, 0, 9.0, 9.0), ('C4', 0, 0, 1.5, 1.0)])
    pts, rep = psc.build_points(df, ['C1', 'C2', 'C3', 'C4'], [0], 0, 0, 'time_point', 0)
    assert len(pts) == 4 and pts['point_id'].is_unique
    _, D, Z = psc.compute_hierarchy(pts[['x', 'y']], 'single')
    assert 4 not in psc.reachable_cuts(Z)               # coincident pair: 4 clusters unreachable
    labels = psc.cut_at_distance(Z, 0.6)                # {C1,C2,C4} and singleton {C3}
    assert len(np.unique(labels)) == 2
    # Singleton convention of sklearn: silhouette 0 for the point alone
    assert psc.mean_silhouette(D, labels) == pytest.approx(silhouette_score(D, labels, metric='precomputed'))


def test_undefined_silhouette_cases():
    D = np.array([[0, 1.0, 5], [1, 0, 5], [5, 5, 0]])
    assert psc.mean_silhouette(D, np.array([1, 1, 1])) is None
    assert psc.mean_silhouette(D, np.array([1, 2, 3])) is None
    assert psc.mean_silhouette(D, np.array([1, 1, 2])) is not None


def test_missing_values_counted_not_filled():
    df = frame([('C1', 0, 0, 1.0, 1.0), ('C1', 0, 1, np.nan, 2.0), ('C2', 0, 0, 3.0, 3.0),
                ('C2', 0, 1, 3.0, 4.0), ('C3', 0, 1, 5.0, 5.0)])
    pts, rep = psc.build_points(df, ['C1', 'C2', 'C3'], [0], 0, 1, 'time_point', 1)
    assert rep['invalid_rows'] == 1 and rep['pairs_without_position'] == 1
    assert list(pts['config']) == ['C2', 'C3']
    pts, rep = psc.build_points(df, ['C1', 'C2', 'C3'], [0], 0, 1, 'interval_mean')
    assert list(pts['n_obs']) == [1, 2, 1] and pts.loc[0, 'x'] == 1.0
    assert (pts['source'] == 'aggregated (mean)').all()


def test_empty_selection():
    df = frame([('C1', 0, 0, 1.0, 1.0)])
    pts, rep = psc.build_points(df, [], [0], 0, 1, 'interval_mean')
    assert pts.empty and rep['selected_rows'] == 0
    pts, rep = psc.build_points(df, ['C1'], [0], 5, 6, 'interval_mean')
    assert pts.empty and rep['pairs_without_position'] == 1


def test_signature_ignores_diameter_but_tracks_data_and_method():
    pts = pd.DataFrame({'point_id': ['a', 'b'], 'x': [0.0, 1.0], 'y': [0.0, 0.0]})
    s = psc.data_signature(pts, 'single', 10)
    assert s == psc.data_signature(pts.copy(), 'single', 10)
    assert s != psc.data_signature(pts, 'average', 10)
    moved = pts.copy(); moved.loc[1, 'x'] = 1.5
    assert s != psc.data_signature(moved, 'single', 10)


def test_dendrogram_colours_follow_cut():
    xy = np.array([[0.0, 0], [0.5, 0], [5, 0], [5.4, 0]])
    _, D, Z = psc.compute_hierarchy(xy, 'single')
    labels = psc.canonical_labels(psc.cut_at_distance(Z, 1.0))
    segs, order = psc.dendrogram_segments(Z, labels, 1.0)
    assert sorted(s['cluster'] for s in segs if s['cluster'] is not None) == [0, 1]
    assert sum(s['cluster'] is None for s in segs) == 1


def test_degree_warning():
    assert psc.coordinate_warning([3.71, 3.72], [51.05, 51.06]) is not None
    assert psc.coordinate_warning([0, 40.5], [0, 30.2]) is None


CSV = os.environ.get('PDP_A2_CSV')


@pytest.mark.skipif(not CSV or not os.path.exists(CSV), reason='set PDP_A2_CSV to A2_1Hz_dataset.csv')
@pytest.mark.parametrize('mode', psc.MODES)
def test_a2_dataset(mode):
    from modules import utils
    with open(CSV, 'rb') as f:
        df = utils.load_data(f, update_state=False, show_success=False)
    configs, objects = list(df['config_source'].unique()), sorted(df['obj'].unique())
    t0, t1 = float(df['tst'].min()), float(df['tst'].max())
    stamp = sorted(df['tst'].unique())[len(df['tst'].unique()) // 2]
    pts, rep = psc.build_points(df, configs, objects, t0, t1, mode, stamp)
    assert pts['point_id'].is_unique and len(pts) + rep['pairs_without_position'] >= rep['pairs_requested'] - 0
    for method in psc.LINKAGES:
        check_consistency(pts[['x', 'y']].to_numpy(float), method)
