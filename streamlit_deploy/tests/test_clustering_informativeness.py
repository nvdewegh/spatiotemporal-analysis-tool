"""
P1: a wholly uninformative clustering input (all pairwise dissimilarities zero within a documented relative
tolerance) gives one undifferentiated group, no cluster-count suggestion and no silhouette or MDS claim.
P2: every clustering figure carries the computed run's identity (calculation, features, linkage, objects,
window, actual cluster count, run ID) inside the Plotly figure.

Uses A2_1Hz_dataset.csv (PDP_1HZ_CSV or the course location). Run from streamlit_deploy/ (PYTHONPATH=.).
"""
import io
import json
import os
import warnings

import numpy as np
import pandas as pd
import pytest
import streamlit
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import Selectbox

from modules import clustering as c

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(HERE, 'streamlit_visualization.py')
CSV_1HZ = os.environ.get('PDP_1HZ_CSV') or os.path.join(
    HERE, '..', '..', '..', '04Exercises', 'AJ2627', 'ExercisesForStudents', 'ForStudents', 'Data', 'Data',
    'CSV_PDP input', 'A2_1Hz_dataset.csv')
FEATURES, CHAMFER, DTW = 'Features (Euclidean)', 'Spatial (Chamfer)', 'Spatiotemporal (DTW)'
CONFIGS = [f'Config_{i}' for i in range(26)]
needs_data = pytest.mark.skipif(not os.path.exists(CSV_1HZ), reason='A2_1Hz_dataset.csv not available')


@pytest.fixture(scope='module')
def a2():
    r = pd.read_csv(CSV_1HZ)
    return pd.DataFrame({'config_source': 'Config_' + r.conID.astype(str), 'obj': r.poiID, 'tst': r.tstID,
                         'x': r.x, 'y': r.y})


def feature_run(df, features):
    D, ids, F, T = c.compute_feature_distance_matrix(df, CONFIGS, [0], 0, 5, features)
    return D, ids, F, T, c.build_linkage(D, FEATURES, ids)


def fake_run(method, features=(), variation=None, run_id='abc1234567'):
    return {'method': method, 'features': list(features), 'linkage': c.linkage_for(method), 'run_id': run_id,
            'objects': [0], 'start': 0.0, 'end': 5.0, 'trajectory_ids': [f'T{i}' for i in range(26)],
            'variation': variation or {'uninformative': False, 'max_distance': 1.0, 'tolerance': 1e-9,
                                       'scale': 1.0, 'constant_features': []}}


# --------------------------------------------------------------------------- P1 acceptance 1 and 2

@needs_data
def test_duration_only_is_one_undifferentiated_group(a2):
    D, ids, F, T, Z = feature_run(a2, ['duration'])
    assert len(ids) == 26 and all(len(t) == 6 for t in T) and (F['duration'] == 5).all()
    assert np.all(D == 0)
    v = c.assess_input_variation(D, FEATURES, T, F)
    assert v['uninformative'] and v['constant_features'] == ['duration']
    k, plot = c.evaluate_cluster_counts(D, Z)
    assert k is None and plot['optimal_k'] is None and plot['reason'] == 'uninformative'
    assert all(np.isnan(plot['silhouette_scores'])) and set(plot['actual_counts']) == {1}
    assert len(set(c.cut_hierarchy(Z, 1))) == 1 and len(set(c.cut_hierarchy(Z, 3))) == 1   # working hierarchy
    msg = c.uninformative_message({'method': FEATURES, 'features': ['duration'], 'variation': v})
    assert 'does not distinguish the trajectories' in msg and 'Duration' in msg


@needs_data
def test_constant_duration_plus_average_speed_equals_average_speed_alone(a2):
    D_both, ids, F_both, T, Z_both = feature_run(a2, ['duration', 'avg_speed'])
    D_speed, ids_s, _, _, Z_speed = feature_run(a2, ['avg_speed'])
    assert ids == ids_s and np.array_equal(D_both, D_speed)
    for k in (2, 3, 4, 5):
        np.testing.assert_array_equal(c.cut_hierarchy(Z_both, k), c.cut_hierarchy(Z_speed, k))
    v = c.assess_input_variation(D_both, FEATURES, T, F_both)
    assert not v['uninformative'] and v['constant_features'] == ['duration']
    note = c.constant_feature_note({'method': FEATURES, 'features': ['duration', 'avg_speed'], 'variation': v})
    assert note.startswith('Duration has the same value') and 'contributes no distinction' in note
    assert note.endswith('determined by Average Speed.')


# --------------------------------------------------------------------------- P1 acceptance 3 and 4

@needs_data
@pytest.mark.parametrize('method', [CHAMFER, DTW])
def test_trajectory_runs_are_informative_and_unchanged(a2, method):
    fn = c.compute_chamfer_distance_matrix if method == CHAMFER else c.compute_dtw_distance_matrix
    D, ids, T = fn(a2, CONFIGS, [0], 0, 5)
    assert not c.assess_input_variation(D, method, T)['uninformative']
    Z = c.build_linkage(D, method, ids)
    np.testing.assert_array_equal(c.cut_hierarchy(Z, 4), fcluster(linkage(squareform(D), 'average'), 4, 'maxclust'))


def test_ties_and_partial_zeros_are_not_rejected():
    feats = pd.DataFrame({'a': [1.0, 1.0, 1.0, 4.0, 4.0, 9.0]}, index=list('ABCDEF'))   # duplicates -> zeros
    from sklearn.preprocessing import StandardScaler
    from scipy.spatial.distance import cdist
    z = StandardScaler().fit_transform(feats)
    D = cdist(z, z)
    assert (D[np.triu_indices(6, 1)] == 0).sum() == 4
    v = c.assess_input_variation(D, FEATURES, features_df=feats)
    assert not v['uninformative'] and v['constant_features'] == []
    k, plot = c.evaluate_cluster_counts(D, c.build_linkage(D, FEATURES))
    assert 'reason' not in plot


def test_relative_tolerance_keeps_tiny_but_real_variation():
    # Micrometre-scale trajectories: Chamfer values around 1e-7, below a naive absolute 1e-6 threshold
    base = np.array([[0.0, 0.0], [1e-6, 0.0], [2e-6, 0.0]])
    T = [base, base + [0, 1e-7], base + [0, 3e-7], base + [0, 6e-7]]
    D = np.array([[np.mean([np.min(np.hypot(*(a - B).T)) for a in A]) for B in T] for A in T])
    D = (D + D.T) / 2
    assert 0 < D.max() < 1e-6
    v = c.assess_input_variation(D, CHAMFER, T)
    assert not v['uninformative'] and v['tolerance'] == pytest.approx(1e-9 * 2e-6)
    # Round-off on identical trajectories with metre-scale coordinates is read as no variation
    big = [base * 1e8] * 4
    D_noise = np.full((4, 4), 1e-12)
    np.fill_diagonal(D_noise, 0)
    assert c.assess_input_variation(D_noise, CHAMFER, big)['uninformative']
    # DTW scale includes the trajectory length (one cost per aligned pair)
    T3 = [np.c_[t, np.arange(3)] for t in big]
    assert c.input_scale(DTW, T3) == pytest.approx(2e2 * 3)


def test_exports_distinguish_requested_and_actual():
    labels = np.ones(26, dtype=int)
    run = fake_run(FEATURES, ['duration'])
    assert c.run_file_tag(run, 3, 1) == 'features-duration_ward_3requested-1actual_runabc1234567'
    assert c.run_file_tag(run, 4, 4) == c.run_file_tag(run, 4) == 'features-duration_ward_4clusters_runabc1234567'
    assert c.cluster_count_text(1, labels) == '1 cluster'
    assert c.cluster_count_text(3, labels).startswith('1 actual group (3 requested')


# --------------------------------------------------------------------------- P2 headers

def test_header_names_calculation_separately_from_display():
    labels = np.repeat([1, 2, 3], [10, 10, 6])
    feat = c.figure_header(fake_run(FEATURES, ['avg_speed']), '3D spatiotemporal view', 3, labels)
    assert 'Feature-based clustering (Euclidean) on Average Speed, Ward linkage' in feat
    assert 'DTW' not in feat and 'Object 0' in feat and 'tst 0–5' in feat
    assert '3 clusters' in feat and 'run abc1234567' in feat
    dtw = c.figure_header(fake_run(DTW, run_id='fff0001111'), '3D spatiotemporal view', 3, labels)
    assert 'Spatiotemporal clustering (DTW), average linkage' in dtw and 'run fff0001111' in dtw
    mds = c.figure_header(fake_run(DTW), '2D MDS projection', 3, labels)
    assert 'embedding dimensions' in mds and 'not the original' in mds
    assert all(len(line) <= 120 for line in mds.replace('<b>', '').replace('</b>', '').split('<br>'))


# --------------------------------------------------------------------------- app

@pytest.fixture
def at(monkeypatch):
    if not os.path.exists(CSV_1HZ):
        pytest.skip('A2_1Hz_dataset.csv not available')
    payload = open(CSV_1HZ, 'rb').read()
    downloads = []

    class Upload(io.BytesIO):
        name = 'A2_1Hz_dataset.csv'

    monkeypatch.setattr(streamlit, 'file_uploader',
                        lambda label, *a, **k: [Upload(payload)] if label.startswith('Upload CSV') else None)

    def download_button(label, data=None, file_name=None, *a, **k):
        downloads.append({'label': label, 'data': data if isinstance(data, str) else data.decode(),
                          'file_name': file_name})
        return False

    monkeypatch.setattr(streamlit, 'download_button', download_button)
    from test_feature_clustering_runs import _selectbox_state
    monkeypatch.setattr(Selectbox, '_widget_state', property(_selectbox_state))
    app = AppTest.from_file(APP, default_timeout=300)
    app.session_state['password_correct'] = True
    app.downloads = downloads
    app.run()
    app.multiselect(key='sidebar_configs').set_value(CONFIGS).run()
    app.multiselect(key='sidebar_objects').set_value([0]).run()
    app.selectbox(key='analysis_method').set_value('Clustering').run()
    app.number_input(key='clustering_start').set_value(0.0).run()
    app.number_input(key='clustering_end').set_value(5.0).run()
    return app


def ok(app):
    assert not app.exception, [e.message for e in app.exception]


def titles(app):
    """Figure titles with line breaks read as spaces (wrapping position depends on the figure width)."""
    return [json.loads(ch.proto.spec)['layout'].get('title', {}).get('text', '').replace('<br>', ' ')
            for ch in app.get('plotly_chart')]


def latest(app, prefix):
    return [d for d in app.downloads if str(d['file_name']).startswith(prefix)][-1]


def run_features(app, features):
    app.radio(key='clustering_method_radio').set_value(FEATURES).run()
    app.multiselect(key='selected_features').set_value(features).run()
    del app.downloads[:]
    app.button(key='compute_features').click().run()
    ok(app)


def test_app_duration_only(at):
    run_features(at, ['duration'])
    run = at.session_state['clustering_run']
    assert run['variation']['uninformative']
    assert any('does not distinguish the trajectories' in w.value and 'Duration' in w.value for w in at.warning)
    assert not [s for s in at.slider if s.key == 'n_clusters_slider']
    assert not [b for b in at.button if b.label == 'Suggest Number of Clusters']
    assert set(at.session_state['cluster_labels']) == {1}
    assert any('Silhouette score: **unavailable**' in i.value for i in at.info)
    assert any('No MDS projection is drawn' in i.value for i in at.info)
    conf = pd.read_csv(io.StringIO(latest(at, 'analysis_configuration')['data']), index_col=0)['Value']
    assert conf['Requested_Number_of_Clusters'] == '1' and conf['Actual_Number_of_Clusters'] == '1'
    assert conf['Input_Distinguishes_Trajectories'].startswith('no') and conf['Constant_Selected_Features'] == 'Duration'
    assignments = pd.read_csv(io.StringIO(latest(at, 'cluster_assignments')['data']))
    assert (assignments['Cluster'] == 1).all() and len(assignments) == 26
    assert latest(at, 'cluster_assignments')['file_name'].endswith(f"_1clusters_run{run['run_id']}.csv")
    assert all(f"run {run['run_id']}" in t for t in titles(at))
    three_d = [t for t in titles(at) if t.startswith('<b>3D spatiotemporal view')]
    assert three_d and '1 cluster' in three_d[0] and 'input does not distinguish' in three_d[0]


def test_app_headers_feature_then_dtw_and_stale_protection(at):
    run_features(at, ['avg_speed'])
    at.slider(key='n_clusters_slider').set_value(3).run()
    ok(at)
    feat_run = at.session_state['clustering_run']
    assert not feat_run['variation']['uninformative']
    t = titles(at)
    assert t and all(f"run {feat_run['run_id']}" in x for x in t)
    three_d = next(x for x in t if x.startswith('<b>3D spatiotemporal view'))
    assert 'Feature-based clustering (Euclidean) on Average Speed, Ward linkage' in three_d and 'DTW' not in three_d
    assert '3 clusters' in three_d

    # Change a control without recomputing: the existing stale-result protection hides every figure,
    # so nothing from the feature run can be shown under the new settings
    at.number_input(key='clustering_end').set_value(4.0).run()
    ok(at)
    assert any('Inputs changed. Recompute to update results.' in w.value for w in at.warning)
    assert not titles(at)
    at.number_input(key='clustering_end').set_value(5.0).run()
    # Switching the method clears the earlier run altogether (existing behaviour)
    at.radio(key='clustering_method_radio').set_value(DTW).run()
    ok(at)
    assert not titles(at)

    del at.downloads[:]
    at.button(key='compute_dtw').click().run()
    ok(at)
    dtw_run = at.session_state['clustering_run']
    assert dtw_run['run_id'] != feat_run['run_id'] and dtw_run['linkage'] == 'average'
    t = titles(at)
    assert t and all(f"run {dtw_run['run_id']}" in x and 'Spatiotemporal clustering (DTW), average linkage' in x
                     for x in t)
    # Labels and exports use the same actual memberships
    labels = at.session_state['cluster_labels']
    assignments = pd.read_csv(io.StringIO(latest(at, 'cluster_assignments')['data']))
    np.testing.assert_array_equal(assignments['Cluster'].to_numpy(), labels)
    conf = pd.read_csv(io.StringIO(latest(at, 'analysis_configuration')['data']), index_col=0)['Value']
    assert conf['Actual_Number_of_Clusters'] == str(len(set(labels))) and conf['Linkage_Method'] == 'Average'
