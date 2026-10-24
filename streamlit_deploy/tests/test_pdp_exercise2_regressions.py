"""Regressions from Exercise 2: accurate runs and affordable detailed views."""
import ast
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import fcluster

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules import pdp_analysis as p


def test_pdp_baseline_uses_one_average_linkage_hierarchy():
    distances = np.array([[0, 144, 0, 120], [144, 0, 144, 24],
                          [0, 144, 0, 120], [120, 24, 120, 0]])
    for k, expected in ((2, [{0, 2}, {1, 3}]), (3, [{0, 2}, {1}, {3}])):
        labels, hierarchy = p.perform_hierarchical_clustering(distances, k)
        partitions = [set(np.flatnonzero(labels == label)) for label in set(labels)]
        assert all(group in partitions for group in expected)
        np.testing.assert_allclose(hierarchy[:, 2], [0, 24, 132])
        cut_height = (hierarchy[-k, 2] + hierarchy[-(k - 1), 2]) / 2
        cut_labels = fcluster(hierarchy, cut_height, criterion='distance')
        np.testing.assert_array_equal(labels[:, None] == labels[None, :],
                                      cut_labels[:, None] == cut_labels[None, :])


def test_tied_hierarchy_retains_requested_cluster_count_and_explains_cut():
    distances = np.ones((4, 4)) - np.eye(4)
    labels, hierarchy = p.perform_hierarchical_clustering(distances, 3)
    assert len(set(labels)) == 3
    fig = p.create_interactive_dendrogram(hierarchy, list('ABCD'), distances, 3)
    assert any('Tied merge heights' in annotation.text for annotation in fig.layout.annotations)


def test_vectorized_relations_match_original_scalar_definition():
    values = np.array([-2.0, -0.5, 0.0, 0.5, 2.0, np.nan])
    for rough in (0.0, 0.5, 2.0):
        expected = np.empty((len(values), len(values)), dtype=int)
        for i, a in enumerate(values):
            for j, b in enumerate(values):
                delta = b - a
                expected[i, j] = 1 if abs(delta) <= rough else (0 if delta > rough else 2)
        np.testing.assert_array_equal(p.compute_inequality_matrix(values, values, 3, rough), expected)


def test_run_identity_changes_with_data_and_analytical_inputs():
    frame = pd.DataFrame({'config_source': ['A'], 'tst': [0], 'obj': [0], 'x': [1.0], 'y': [2.0]})
    args = [frame, ['A'], [0], 0, 5, 6, 0.5, 0.5, 0.1, 0.1, ['fundamental', 'buffer_rough'], []]
    original = p.pdp_run_signature(*args)
    assert original == p.pdp_run_signature(*args)
    for index, value in ((1, ['A', 'B']), (2, [0, 1]), (3, 1), (4, 4), (5, 2),
                         (6, 1.0), (7, 1.0), (8, 2.0), (9, 2.0),
                         (10, ['rough']), (11, [('Ref', 30.0, 10.0)])):
        changed = args.copy()
        changed[index] = value
        assert original != p.pdp_run_signature(*changed)
    # Buffer/Rough values not used by any selected variant do not change the run
    fundamental = args[:10] + [['fundamental'], []]
    for index in (6, 7, 8, 9):
        changed = fundamental.copy()
        changed[index] = 3.0
        assert p.pdp_run_signature(*fundamental) == p.pdp_run_signature(*changed)
    changed = frame.copy()
    changed.loc[0, 'x'] = 3.0
    assert original != p.pdp_run_signature(changed, *args[1:])


def test_dense_matrix_keeps_relations_without_quadratic_hover_strings():
    n = 242
    frame = pd.DataFrame({'obj': [0] * 121 + [1] * 121,
                          'tst': list(range(121)) * 2,
                          'x': np.arange(n), 'y': np.arange(n)})
    meta = p.point_metadata(frame, list(range(121)))
    display = p.object_major_display(meta)
    matrix = p.compute_inequality_matrix(frame.x.values, frame.x.values, 121)
    fig = make_subplots(rows=1, cols=1)
    p.add_point_matrix_heatmap(fig, matrix, display, 1, 1, 'x')
    trace = fig.data[0]
    np.testing.assert_array_equal(trace.z, matrix[np.ix_(display['perm'], display['perm'])])
    assert trace.customdata is None
    assert trace.xgap == trace.ygap == 0
    assert len(trace.x) == len(trace.y) == n
    assert fig.layout.yaxis.scaleanchor == 'x'
    assert fig.layout.yaxis.scaleratio == 1
    assert len(fig.to_json()) < 500_000


def test_missing_password_widget_cannot_crash_or_authenticate():
    source = Path(__file__).resolve().parents[1] / 'streamlit_visualization.py'
    tree = ast.parse(source.read_text())
    check_password = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'check_password')
    callback = next(n for n in check_password.body if isinstance(n, ast.FunctionDef) and n.name == 'password_entered')
    class FakeStreamlit:
        session_state = {}
    fake = FakeStreamlit()
    ns = {'st': fake}
    exec(compile(ast.Module(body=[callback], type_ignores=[]), str(source), 'exec'), ns)
    ns['password_entered']()
    assert fake.session_state['password_correct'] is False
    fake.session_state['password'] = 'incorrect'
    ns['password_entered']()
    assert fake.session_state['password_correct'] is False
