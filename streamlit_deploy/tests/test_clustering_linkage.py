"""
Linkage per clustering approach, one hierarchy per run, Minimum Speed, interface wording and
pitch backgrounds of the cluster views.

- Feature-based clustering (Euclidean): Ward; Spatial (Chamfer) and Spatiotemporal (DTW): average.
- The dendrogram, memberships, cluster-count evaluation and exports cut the same hierarchy.
- Precomputed matrices are validated and condensed (never clustered as observation vectors).
- Thursday's six-timestamp reference (A2_1Hz_dataset.csv, tst 0-5, object 0) is unchanged.

Run from streamlit_deploy/:  PYTHONPATH=. python -m pytest tests/test_clustering_linkage.py
The A2 checks run when PDP_1HZ_CSV points to A2_1Hz_dataset.csv.
"""
import io
import json
import os
import re
import tokenize
import warnings

import numpy as np
import pandas as pd
import pytest
import streamlit
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import Selectbox

from modules import clustering as c

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(HERE, 'streamlit_visualization.py')
CSV_1HZ = os.environ.get('PDP_1HZ_CSV', '')
FEATURES = 'Features (Euclidean)'
CHAMFER = 'Spatial (Chamfer)'
DTW = 'Spatiotemporal (DTW)'


def random_points(n=12, d=3, seed=0):
    return np.random.default_rng(seed).normal(size=(n, d))


# --------------------------------------------------------------------------- linkage per approach

def test_linkage_mapping_and_labels():
    assert c.linkage_for(FEATURES) == 'ward'
    assert c.linkage_for(CHAMFER) == 'average' and c.linkage_for(DTW) == 'average'
    assert c.method_label(FEATURES, with_linkage=True) == 'Feature-based clustering (Euclidean), Ward linkage'
    assert c.method_label(CHAMFER, with_linkage=True) == 'Spatial clustering (Chamfer), average linkage'
    assert c.method_label(DTW, with_linkage=True) == 'Spatiotemporal clustering (DTW), average linkage'


def test_ward_on_condensed_euclidean_equals_ward_on_standardised_vectors():
    z = StandardScaler().fit_transform(random_points())
    D = squareform(pdist(z))
    Z = c.build_linkage(D, FEATURES)
    np.testing.assert_allclose(Z, linkage(z, method='ward'), atol=1e-10)


@pytest.mark.parametrize('method', [CHAMFER, DTW])
def test_average_linkage_on_condensed_dissimilarities(method):
    D = squareform(pdist(random_points(seed=1), 'cityblock'))  # a non-Euclidean dissimilarity
    Z = c.build_linkage(D, method)
    np.testing.assert_allclose(Z, linkage(squareform(D), method='average'))
    # A square matrix passed as observations would cluster distance profiles: a different hierarchy
    assert not np.allclose(Z[:, 2], linkage(D, method='average')[:, 2])


@pytest.mark.parametrize('bad, message', [
    (lambda D: D[:, :-1], 'square'),
    (lambda D: np.where(np.eye(len(D)) == 1, 0.5, D), 'diagonal'),
    (lambda D: D + np.triu(np.ones_like(D), 1) * 0.1, 'symmetric'),
    (lambda D: np.where(np.arange(D.size).reshape(D.shape) == 1, np.nan, D), 'missing'),
    (lambda D: -D, 'negative'),
])
def test_invalid_matrices_are_rejected(bad, message):
    D = squareform(pdist(random_points(seed=2)))
    with pytest.raises(ValueError, match=message):
        c.build_linkage(bad(D), CHAMFER)


def test_trajectory_order_and_tolerance():
    D = squareform(pdist(random_points(n=4, seed=3)))
    with pytest.raises(ValueError, match='trajectory IDs'):
        c.build_linkage(D, CHAMFER, ['a', 'b', 'c'])
    noisy = D + np.triu(np.full_like(D, 1e-13), 1)   # round-off only: accepted
    c.build_linkage(noisy, CHAMFER, ['a', 'b', 'c', 'd'])


@pytest.mark.parametrize('method', [FEATURES, CHAMFER])
def test_cluster_count_evaluation_cuts_the_displayed_hierarchy(method):
    D = squareform(pdist(random_points(n=20, seed=4)))
    Z = c.build_linkage(D, method)
    k, data = c.evaluate_cluster_counts(D, Z)
    assert data['k_values'] == list(range(2, 11)) and k in data['k_values']
    for kk, sil in zip(data['k_values'], data['silhouette_scores']):
        labels = fcluster(Z, kk, 'maxclust')
        np.testing.assert_array_equal(labels, c.cut_hierarchy(Z, kk))
        assert sil == pytest.approx(silhouette_score(D, labels, metric='precomputed'))


def test_linkage_is_part_of_the_run_signature():
    sig = c.run_signature('fp', CHAMFER, ['Config_1'], [0], 0, 5)
    assert sig[-1] == 'average'
    old_ward_signature = sig[:-1] + ('Ward',)   # a Chamfer result computed before this change
    assert old_ward_signature != sig
    assert c.run_signature('fp', FEATURES, ['Config_1'], [0], 0, 5, ['sinuosity'])[-1] == 'ward'


def test_configuration_export_and_filename_name_the_linkage():
    run = {'run_id': 'abc', 'computed_at': 'now', 'filename': 'f.csv', 'fingerprint': 'fp', 'method': DTW,
           'linkage': 'average', 'features': [], 'configs': ['Config_1', 'Config_2'], 'objects': [0],
           'start': 0.0, 'end': 5.0, 'trajectory_ids': ['a', 'b'], 'trajectory_lengths': [6, 6]}
    conf = c.run_configuration(run, 2, np.array([1, 2]))
    assert conf['Linkage_Method'] == 'Average' and conf['Clustering_Method'] == 'Spatiotemporal clustering (DTW)'
    assert c.run_file_tag(run, 2) == 'dtw_average_2clusters_runabc'


# --------------------------------------------------------------------------- Minimum Speed

def traj(tst, x, y=None):
    return pd.DataFrame({'tst': tst, 'x': x, 'y': y if y is not None else [0.0] * len(x)})


def test_minimum_speed_definition():
    f = c.extract_trajectory_features(traj([2, 0, 1, 3], [4.0, 0.0, 2.0, 4.0]))   # unsorted input
    assert f['min_speed'] == 0.0            # genuine standstill between tst 2 and 3 is kept
    assert f['max_speed'] == pytest.approx(2.0)
    assert f['avg_speed'] == pytest.approx(4.0 / 3)
    assert c.extract_trajectory_features(traj([0, 2], [0.0, 3.0]))['min_speed'] == pytest.approx(1.5)


def test_minimum_speed_excludes_invalid_segments():
    # Duplicate timestamp: the zero-duration segment is excluded rather than given a tiny duration
    speeds = c.segment_speeds(traj([0, 1, 1, 2], [0.0, 1.0, 9.0, 10.0]))
    assert list(speeds) == [1.0, 1.0]
    assert c.extract_trajectory_features(traj([0, 1, 1, 2], [0.0, 1.0, 9.0, 10.0]))['max_speed'] == 1.0
    # Missing coordinate: both adjoining segments are unavailable
    assert list(c.segment_speeds(traj([0, 1, 2, 3], [0.0, np.nan, 2.0, 5.0]))) == [3.0]
    # Insufficient observations or only zero elapsed time: unavailable, not zero
    assert np.isnan(c.extract_trajectory_features(traj([0], [0.0]))['min_speed'])
    assert np.isnan(c.extract_trajectory_features(traj([1, 1], [0.0, 2.0]))['min_speed'])


def test_unavailable_minimum_speed_stops_the_computation_with_a_clear_message():
    df = pd.DataFrame({'config_source': ['A', 'A', 'B', 'B'], 'obj': 0, 'tst': [0, 1, 0, 0],
                       'x': [0.0, 1.0, 0.0, 1.0], 'y': 0.0})
    with pytest.raises(ValueError, match='Minimum Speed: B_obj0'):
        c.compute_feature_distance_matrix(df, ['A', 'B'], [0], 0, 1, ['min_speed'])


def test_minimum_speed_is_optional_and_not_in_the_default_selection():
    assert 'min_speed' in c.ALL_FEATURES and 'min_speed' not in c.DEFAULT_FEATURES
    assert len(c.DEFAULT_FEATURES) == 8
    assert c.FEATURE_COLUMN_LABELS['min_speed'] == 'Min Speed (m per tst unit)'


# --------------------------------------------------------------------------- Thursday reference

@pytest.fixture(scope='module')
def a2_six_timestamps():
    if not os.path.exists(CSV_1HZ):
        pytest.skip('A2_1Hz_dataset.csv not available (set PDP_1HZ_CSV)')
    r = pd.read_csv(CSV_1HZ)
    return pd.DataFrame({'config_source': 'Config_' + r.conID.astype(str), 'obj': r.poiID, 'tst': r.tstID,
                         'x': r.x, 'y': r.y})


REFERENCE = {'Config_5_obj0': (27.164759, 1.007871), 'Config_8_obj0': (29.600800, 1.000000),
             'Config_16_obj0': (28.669882, 1.003359)}


def thursday_run(df, feature):
    configs = [f'Config_{i}' for i in range(26)]
    D, ids, features, trajectories = c.compute_feature_distance_matrix(df, configs, [0], 0, 5, [feature])
    assert len(ids) == 26 and all(len(t) == 6 for t in trajectories)
    labels = c.cut_hierarchy(c.build_linkage(D, FEATURES, ids), 4)
    return dict(zip(ids, labels)), features, D, labels


def test_thursday_six_timestamp_reference(a2_six_timestamps):
    distance, feats_d, D_d, lab_d = thursday_run(a2_six_timestamps, 'total_distance')
    sinuosity, feats_s, D_s, lab_s = thursday_run(a2_six_timestamps, 'sinuosity')
    for tid, (dist, sin) in REFERENCE.items():
        assert feats_d.loc[tid, 'total_distance'] == pytest.approx(dist, abs=5e-7)
        assert feats_s.loc[tid, 'sinuosity'] == pytest.approx(sin, abs=5e-7)
    assert len(set(lab_d)) == 4 and len(set(lab_s)) == 4
    for a, b in (('Config_5_obj0', 'Config_16_obj0'), ('Config_8_obj0', 'Config_16_obj0')):
        assert distance[a] == distance[b]       # share a distance cluster
        assert sinuosity[a] != sinuosity[b]     # separate under sinuosity
    # Same partitions as plain scipy Ward on the condensed standardised Euclidean distances
    for D, labels in ((D_d, lab_d), (D_s, lab_s)):
        np.testing.assert_array_equal(labels, fcluster(linkage(squareform(D, checks=False), 'ward'), 4, 'maxclust'))


def test_adding_minimum_speed_option_leaves_single_feature_runs_unchanged(a2_six_timestamps):
    configs = [f'Config_{i}' for i in range(26)]
    _, _, all_feats, _ = c.compute_feature_distance_matrix(a2_six_timestamps, configs, [0], 0, 5, c.ALL_FEATURES)
    assert all_feats['min_speed'].notna().all() and (all_feats['min_speed'] <= all_feats['avg_speed'] + 1e-12).all()
    assert (all_feats['avg_speed'] <= all_feats['max_speed'] + 1e-12).all()
    for feature in ('total_distance', 'sinuosity'):
        _, feats, D, _ = thursday_run(a2_six_timestamps, feature)
        z = StandardScaler().fit_transform(all_feats[[feature]])
        np.testing.assert_allclose(D, squareform(pdist(z)), atol=1e-12)


# --------------------------------------------------------------------------- wording

DECORATIVE = re.compile('[\U0001F300-\U0001F5FF\U0001F680-\U0001FAFF✅❌⚠⚙⏸�'
                        '\U0001F600-\U0001F64F]')
FUNCTIONAL = set('\U0001F534\U0001F535\U0001F7E2\U0001F7E0')   # coloured dots that explain plot colours
US = re.compile(r'(?<![A-Za-z0-9_\-#.])([Vv]isualiz|[Mm]odeling\b|[Aa]nalyz|[Ss]ynchroniz|[Mm]inimiz|'
                r'[Cc]olor(?:s|ed|ing)?\b(?!\s*[:=;"\']))')
UI_FILES = ['streamlit_visualization.py', 'ui_components.py', 'pdp_reference_ui.py', 'modules/clustering.py']


def user_strings(path):
    src = open(os.path.join(HERE, path), encoding='utf-8').read()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.STRING and re.search(r'\s', tok.string):
            yield tok.start[0], tok.string


@pytest.mark.parametrize('path', UI_FILES)
def test_uk_spelling_in_user_facing_text(path):
    hits = [(line, m.group(0)) for line, text in user_strings(path) for m in US.finditer(text)
            if not re.search(r'(color|colorscale|colorbar)\s*=', text[max(0, m.start() - 2):m.end() + 3])]
    assert not hits, hits[:10]


@pytest.mark.parametrize('path', UI_FILES)
def test_no_decorative_icons(path):
    hits = [(line, ch) for line, text in user_strings(path) for ch in DECORATIVE.findall(text)
            if ch not in FUNCTIONAL]
    assert not hits, hits[:10]


def test_no_blanket_ward_statement():
    src = open(APP, encoding='utf-8').read()
    assert 'All clustering methods use Hierarchical Ward' not in src
    assert "method='ward'" not in src.split('elif analysis_method == "Clustering":')[1].split(
        'elif analysis_method == "Extra":')[0]


# --------------------------------------------------------------------------- cluster views: pitch and zoom

@pytest.fixture
def clustered(monkeypatch):
    # Coordinates far from any court, so a tennis-sized view could not show them
    rows = ''.join(f'{cfg},{t},0,{200 + 10 * cfg + 3 * t},{-50 + (cfg % 3) * 4 + t}\n'
                   for cfg in range(6) for t in range(6))
    data = ('conID,tstID,poiID,x,y\n' + rows).encode()

    class Upload(io.BytesIO):
        name = 'synthetic.csv'

    monkeypatch.setattr(streamlit, 'file_uploader',
                        lambda label, *a, **k: [Upload(data)] if label.startswith('Upload CSV') else None)
    # Same AppTest harness workaround as test_feature_clustering_runs ("Select reference trajectory")
    from test_feature_clustering_runs import _selectbox_state
    monkeypatch.setattr(Selectbox, '_widget_state', property(_selectbox_state))
    app = AppTest.from_file(APP, default_timeout=300)
    app.session_state['password_correct'] = True
    app.run()
    app.multiselect(key='sidebar_configs').set_value([f'Config_{i}' for i in range(6)]).run()
    app.multiselect(key='sidebar_objects').set_value([0]).run()
    app.selectbox(key='analysis_method').set_value('Clustering').run()
    app.radio(key='clustering_method_radio').set_value(CHAMFER).run()
    app.button(key='compute_chamfer').click().run()
    assert not app.exception, [e.message for e in app.exception]
    app.slider(key='n_clusters_slider').set_value(2).run()
    return app


def values(v):
    """Plotly 6 serialises numpy arrays as base64 typed arrays ({'dtype', 'bdata'})."""
    if isinstance(v, dict) and 'bdata' in v:
        import base64
        return np.frombuffer(base64.b64decode(v['bdata']), dtype=v['dtype']).tolist()
    return list(v)


def chart_specs(app):
    specs = [json.loads(ch.proto.spec) for ch in app.get('plotly_chart')]
    for spec in specs:
        for t in spec['data']:
            for axis in ('x', 'y'):
                if axis in t:
                    t[axis] = values(t[axis])
    return specs


def set_court(app, court, background):
    app.checkbox(key='show_court_background').set_value(background).run()
    next(r for r in app.radio if r.label == 'Select court type').set_value(court).run()


def test_chamfer_run_uses_average_linkage_in_app(clustered):
    run = clustered.session_state['clustering_run']
    assert run['linkage'] == 'average'
    D = clustered.session_state['distance_matrix']
    np.testing.assert_allclose(run['linkage_matrix'], linkage(squareform(D, checks=False), 'average'))
    np.testing.assert_array_equal(clustered.session_state['cluster_labels'],
                                  fcluster(run['linkage_matrix'], 2, 'maxclust'))
    assert any('Spatial clustering (Chamfer), average linkage' in m.value for m in clustered.caption)


@pytest.mark.parametrize('court, background, expect_court', [
    ('No selection', True, False), ('Tennis', False, False), ('Tennis', True, True), ('Football', True, True)])
def test_cluster_views_follow_course_type(clustered, court, background, expect_court):
    set_court(clustered, court, background)
    clustered.button(key='btn_2d_cluster').click().run()
    assert not clustered.exception
    all_specs = chart_specs(clustered)
    clustered.button(key='btn_cluster_compare').click().run()   # Overlay of the default (first) cluster
    assert not clustered.exception
    all_specs += [s for s in chart_specs(clustered) if 'Overlay' in s['layout'].get('title', {}).get('text', '')]
    with_traj = [s for s in all_specs if any(str(t.get('name', '')).startswith('TConfig') for t in s['data'])]
    specs = [s for s in with_traj if 'scene' not in s['layout']]
    assert len(specs) >= 2
    # 3D view: pitch surface only for a drawn Football/Tennis background, x/y fitted to the data
    scene = next(s for s in with_traj if 'scene' in s['layout'])
    assert any(t.get('name') in ('Football Pitch', 'Tennis Court') for t in scene['data']) == expect_court
    xs3 = [v for t in scene['data'] if str(t.get('name', '')).startswith('TConfig') for v in t['x']]
    r3 = scene['layout']['scene']['xaxis']['range']
    assert r3[0] <= min(xs3) and max(xs3) <= r3[1]
    for spec in specs:
        layout = spec['layout']
        drawn = bool(layout.get('shapes')) or any(t.get('name') in ('Net', 'Tennis Court') for t in spec['data'])
        assert drawn == expect_court, (court, background)
        assert layout['yaxis'].get('scaleanchor') == 'x' and layout['yaxis'].get('scaleratio', 1) == 1
        xs = [v for t in spec['data'] if str(t.get('name', '')).startswith('TConfig') for v in t['x']]
        x0, x1 = layout['xaxis']['range']
        assert x0 <= min(xs) and max(xs) <= x1
        if not expect_court:
            assert x0 > 150     # fitted to the data, not to a court at the origin


def test_side_by_side_individual_clusters_fit_their_own_trajectories(clustered):
    set_court(clustered, 'No selection', True)
    labels = clustered.session_state['cluster_labels']
    next(m for m in clustered.multiselect if m.label == 'Select clusters to visualise').set_value(
        sorted(int(v) for v in set(labels))).run()
    next(r for r in clustered.radio if r.label == 'View mode').set_value('Side-by-side').run()
    clustered.button(key='btn_cluster_compare').click().run()
    assert not clustered.exception
    specs = [s for s in chart_specs(clustered) if re.match(r'Cluster \d+ \(', s['layout'].get('title', {}).get('text', ''))]
    assert len(specs) == len(set(labels))
    for spec in specs:
        assert not spec['layout'].get('shapes')
        xs = [v for t in spec['data'] for v in t['x']]
        x0, x1 = spec['layout']['xaxis']['range']
        assert x0 <= min(xs) and max(xs) <= x1 and (x1 - x0) < 1.5 * (max(xs) - min(xs)) + 2
