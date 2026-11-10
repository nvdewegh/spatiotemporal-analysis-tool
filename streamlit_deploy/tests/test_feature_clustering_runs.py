"""
Feature-based clustering for the distance travelled versus sinuosity exercise (A2_detailed_dataset.csv):
six-decimal display with numeric full-precision values, results and downloads tied to the computed run
(outdated results hidden after any input change, also after replacing the dataset under the same name),
and exports that describe the same run with distinct filenames.

Download payloads are captured from st.download_button calls (not browser downloads).
Uses PDP_DETAILED_CSV or the course location. Run from streamlit_deploy/ (PYTHONPATH=.).
"""
import io
import json
import os
import warnings

import numpy as np
import pandas as pd
import pytest
import streamlit
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import Selectbox, WidgetState

from modules import clustering

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(HERE, 'streamlit_visualization.py')
_COURSE = os.path.join(HERE, '..', '..', '..', '04Exercises', 'AJ2627')
CSV = os.environ.get('PDP_DETAILED_CSV') or os.path.join(
    _COURSE, 'ExercisesForStudents', 'ForStudents', 'Data', 'Data', 'CSV_PDP input', 'A2_detailed_dataset.csv')
REFERENCE = os.path.join(_COURSE, 'Other', 'DistanceSinuosity_v01', 'calculation_reference.json')
pytestmark = pytest.mark.skipif(not os.path.exists(CSV), reason='A2_detailed_dataset.csv not available')
OUTDATED = 'Inputs changed. Recompute to update results.'
CONFIGS = [f'Config_{i}' for i in range(26)]
EXPECTED_SIZES = {'total_distance': [2, 6, 7, 11], 'sinuosity': [2, 5, 7, 12]}
FOCUS = {'Config_5_obj0': (30.944657, 1.010015), 'Config_8_obj0': (32.744352, 1.000000),
         'Config_16_obj0': (32.637392, 1.008446)}


_ORIGINAL_SELECTBOX_STATE = Selectbox._widget_state


def _selectbox_state(self):
    """
    Harness only: "Select reference trajectory" formats its integer options with a lambda that reads
    st.session_state, which AppTest calls outside a script run (the audit's AppTest limitation). For that
    widget the option string is taken by index instead; every other selectbox uses AppTest's own state.
    """
    try:
        return _ORIGINAL_SELECTBOX_STATE.fget(self)
    except Exception:
        if self.label != 'Select reference trajectory':
            raise
        ws = WidgetState()
        ws.id = self.id
        value = self.value
        if isinstance(value, (int, np.integer)) and 0 <= value < len(self.options):
            ws.string_value = self.options[value]
        return ws


@pytest.fixture
def at(monkeypatch):
    payload = {'bytes': open(CSV, 'rb').read()}
    downloads = []

    class Upload(io.BytesIO):
        name = 'A2_detailed_dataset.csv'

    def uploader(label, *a, **k):
        return [Upload(payload['bytes'])] if label.startswith('Upload CSV') else None

    def download_button(label, data=None, file_name=None, *a, **k):
        downloads.append({'label': label, 'data': data if isinstance(data, str) else data.decode(),
                          'file_name': file_name})
        return False

    monkeypatch.setattr(streamlit, 'file_uploader', uploader)
    monkeypatch.setattr(streamlit, 'download_button', download_button)
    monkeypatch.setattr(Selectbox, '_widget_state', property(_selectbox_state))
    app = AppTest.from_file(APP, default_timeout=300)
    app.session_state['password_correct'] = True
    app.payload, app.downloads = payload, downloads
    app.run()
    app.multiselect(key='sidebar_configs').set_value(CONFIGS).run()
    app.multiselect(key='sidebar_objects').set_value([0]).run()
    app.selectbox(key='analysis_method').set_value('Clustering').run()
    app.number_input(key='clustering_start').set_value(0.0).run()
    app.number_input(key='clustering_end').set_value(136.0).run()
    app.radio(key='clustering_method_radio').set_value('Features (Euclidean)').run()
    return app


def ok(app):
    assert not app.exception, [e.message for e in app.exception]


CLUSTERING_FILES = ('cluster_assignments', 'distance_matrix', 'feature_values', 'cluster_statistics',
                    'analysis_configuration')


def clustering_downloads(app):
    return [d for d in app.downloads if str(d['file_name']).startswith(CLUSTERING_FILES)]


def outdated(app):
    return any(OUTDATED in w.value for w in app.warning)


def compute(app, feature, k=4):
    app.multiselect(key='selected_features').set_value([feature]).run()
    del app.downloads[:]
    app.button(key='compute_features').click().run()
    ok(app)
    app.slider(key='n_clusters_slider').set_value(k).run()
    ok(app)
    assert not outdated(app)
    return run_outputs(app)


def run_outputs(app):
    """Latest payload per download kind from the last script run."""
    seen = {}
    for d in app.downloads:
        seen[d['file_name'].split('_features-')[0].split('_run')[0]] = d
    return seen


def partition(ids, labels):
    groups = {}
    for i, label in zip(ids, labels):
        groups.setdefault(label, set()).add(i)
    return sorted(frozenset(g) for g in groups.values())


def reference_partition(feature):
    ref = json.load(open(REFERENCE))['feature_runs'][feature]
    return partition(ref['ids'], ref['labels'])


def check_run(app, feature, outputs):
    s = app.session_state
    run = s['clustering_run']
    labels = s['cluster_labels']
    ids = s['trajectory_ids']
    assert run['features'] == [feature] and run['method'] == 'Features (Euclidean)'
    assert sorted(np.bincount(labels)[1:]) == EXPECTED_SIZES[feature]
    if os.path.exists(REFERENCE):
        assert partition(ids, labels) == reference_partition(feature)
    label_of = dict(zip(ids, labels))
    assert label_of['Config_5_obj0'] == label_of['Config_16_obj0']
    assert (label_of['Config_8_obj0'] == label_of['Config_16_obj0']) == (feature == 'total_distance')

    tag = f"features-{feature}_4clusters_run{run['run_id']}"
    assignments = outputs['cluster_assignments']
    assert assignments['file_name'] == f"cluster_assignments_{tag}.csv"
    table = pd.read_csv(io.StringIO(assignments['data']))
    assert len(table) == 26 and table['Trajectory_ID'].nunique() == 26
    assert set(table['Trajectory_ID']) == {f'Config_{i}_obj0' for i in range(26)}
    assert (table['Trajectory_Length'] == 137).all()
    assert dict(zip(table['Trajectory_ID'], table['Cluster'])) == {i: int(l) for i, l in label_of.items()}

    features = outputs['feature_values']
    assert features['file_name'] == f"feature_values_features-{feature}_run{run['run_id']}.csv"
    values = pd.read_csv(io.StringIO(features['data']), index_col='Trajectory_ID', float_precision='round_trip')
    assert list(values.columns) == [feature]
    assert np.array_equal(values[feature].to_numpy(), s['features_df'][feature].to_numpy())   # exact
    for tid, (dist, sin) in FOCUS.items():
        assert values.loc[tid, feature] == pytest.approx(dist if feature == 'total_distance' else sin, abs=5e-7)

    distances = outputs['distance_matrix']
    assert distances['file_name'] == f"distance_matrix_{tag}.csv"
    matrix = pd.read_csv(io.StringIO(distances['data']), index_col=0, float_precision='round_trip')
    assert list(matrix.index) == list(matrix.columns) == list(ids)
    assert np.array_equal(matrix.to_numpy(), s['distance_matrix'])                           # full precision

    configuration = outputs['analysis_configuration']
    assert configuration['file_name'] == f"analysis_configuration_{tag}.csv"
    conf = pd.read_csv(io.StringIO(configuration['data']), index_col=0)['Value']
    assert conf['Clustering_Method'] == 'Features (Euclidean)' and conf['Distance_Metric'] == 'Euclidean'
    assert conf['Selected_Features'] == feature and conf['Linkage_Method'] == 'Ward'
    assert conf['Standardisation'].startswith('StandardScaler')
    assert conf['Requested_Number_of_Clusters'] == '4' and conf['Actual_Number_of_Clusters'] == '4'
    assert conf['Dataset_Content_SHA256'] == run['fingerprint'] and conf['Run_ID'] == run['run_id']
    assert conf['Configuration_IDs'] == ', '.join(CONFIGS) and conf['Object_IDs'] == '0'
    assert (conf['Start_Observation_tst'], conf['End_Observation_tst']) == ('0', '136')
    assert 'not seconds' in conf['Time_Unit']
    assert conf['Number_of_Trajectories'] == '26' and conf['Positions_per_Trajectory'] == '137'
    assert outputs['cluster_statistics']['file_name'] == f"cluster_statistics_{tag}.csv"
    return run, partition(ids, labels), s['distance_matrix'].copy()


def test_six_decimal_display_keeps_numeric_values():
    df = pd.DataFrame({'total_distance': [30.94465715190379], 'sinuosity': [1.0100148]}, index=['Config_5_obj0'])
    shown = clustering.format_features_dataframe(df)
    assert shown.dtypes.apply(lambda t: np.issubdtype(t, np.floating)).all()                 # not strings
    assert shown['Sinuosity (ratio)'].iloc[0] == 1.0100148                                    # not rounded
    config = clustering.feature_column_config(df)
    assert config['Sinuosity (ratio)']['type_config']['format'] == '%.6f'
    assert '%.6f' % 1.0 == '1.000000'
    assert float(clustering.features_csv(df).splitlines()[1].split(',')[1]) == 30.94465715190379


def test_distance_sinuosity_a_b_a_with_consistent_exports(at):
    run_a, part_a, matrix_a = check_run(at, 'total_distance', compute(at, 'total_distance'))
    run_b, part_b, _ = check_run(at, 'sinuosity', compute(at, 'sinuosity'))
    assert run_a['run_id'] != run_b['run_id']
    run_a2, part_a2, matrix_a2 = check_run(at, 'total_distance', compute(at, 'total_distance'))
    assert part_a2 == part_a and np.array_equal(matrix_a2, matrix_a) and run_a2['run_id'] == run_a['run_id']
    # Six-decimal display of the sinuosity table (numeric column, display format)
    compute(at, 'sinuosity')
    table = [d for d in at.dataframe if 'Sinuosity (ratio)' in d.value.columns][0]
    assert np.issubdtype(table.value['Sinuosity (ratio)'].dtype, np.floating)
    for tid, (_, sin) in FOCUS.items():
        assert '%.6f' % table.value.loc[tid, 'Sinuosity (ratio)'] == '%.6f' % sin


@pytest.mark.parametrize('change', ['feature', 'empty_features', 'interval', 'objects', 'configs'])
def test_changed_inputs_hide_outdated_results_and_downloads(at, change):
    compute(at, 'total_distance')
    del at.downloads[:]
    if change == 'feature':
        at.multiselect(key='selected_features').set_value(['sinuosity']).run()
    elif change == 'empty_features':
        at.multiselect(key='selected_features').set_value([]).run()
    elif change == 'interval':
        at.number_input(key='clustering_end').set_value(100.0).run()
    elif change == 'objects':
        at.multiselect(key='sidebar_objects').set_value([0, 1]).run()
    else:
        at.multiselect(key='sidebar_configs').set_value(CONFIGS[:20]).run()
    ok(at)
    assert outdated(at)
    assert not clustering_downloads(at)                         # no download of the earlier computation
    assert not any(e.label == 'Extracted Features' for e in at.expander)
    assert not any('Cluster Assignment' in m.value for m in at.markdown)
    assert not at.get('plotly_chart')                            # no matrix, dendrogram or dependent plots
    # Restoring the computed inputs shows the same results again without recomputing
    if change != 'empty_features':
        if change == 'feature':
            at.multiselect(key='selected_features').set_value(['total_distance']).run()
        elif change == 'interval':
            at.number_input(key='clustering_end').set_value(136.0).run()
        elif change == 'objects':
            at.multiselect(key='sidebar_objects').set_value([0]).run()
        else:
            at.multiselect(key='sidebar_configs').set_value(CONFIGS).run()
        ok(at)
        assert not outdated(at) and clustering_downloads(at)


def test_cluster_count_only_recuts_the_current_hierarchy(at):
    compute(at, 'sinuosity')
    run = at.session_state['clustering_run']
    at.slider(key='n_clusters_slider').set_value(3).run()
    ok(at)
    assert not outdated(at) and at.session_state['clustering_run']['run_id'] == run['run_id']
    outputs = run_outputs(at)
    assert outputs['cluster_assignments']['file_name'].endswith(f"_3clusters_run{run['run_id']}.csv")
    conf = pd.read_csv(io.StringIO(outputs['analysis_configuration']['data']), index_col=0)['Value']
    assert conf['Requested_Number_of_Clusters'] == '3' and conf['Actual_Number_of_Clusters'] == '3'
    assigned = pd.read_csv(io.StringIO(outputs['cluster_assignments']['data']))
    assert sorted(assigned['Cluster'].unique()) == [1, 2, 3]


def test_replaced_dataset_with_same_name_clears_results(at):
    compute(at, 'total_distance')
    old = at.session_state['clustering_run']
    data = pd.read_csv(io.BytesIO(at.payload['bytes']))
    data['x'] = data['x'] * 1.1                                  # same file name, different content
    at.payload['bytes'] = data.to_csv(index=False).encode()
    at.run()
    ok(at)
    # Back to the identical clustering inputs: nothing from the earlier dataset is shown or downloadable
    at.multiselect(key='sidebar_configs').set_value(CONFIGS).run()
    at.multiselect(key='sidebar_objects').set_value([0]).run()
    at.selectbox(key='analysis_method').set_value('Clustering').run()
    at.number_input(key='clustering_start').set_value(0.0).run()
    at.number_input(key='clustering_end').set_value(136.0).run()
    at.radio(key='clustering_method_radio').set_value('Features (Euclidean)').run()
    del at.downloads[:]
    at.multiselect(key='selected_features').set_value(['total_distance']).run()
    ok(at)
    assert 'clustering_run' not in at.session_state or at.session_state['clustering_run'] is None
    assert not clustering_downloads(at)
    assert not any('Cluster Assignment' in m.value for m in at.markdown)
    outputs = compute(at, 'total_distance')
    new = at.session_state['clustering_run']
    assert new['fingerprint'] != old['fingerprint'] and new['run_id'] != old['run_id']
    assert new['filename'] == old['filename'] == 'A2_detailed_dataset.csv'
    values = pd.read_csv(io.StringIO(outputs['feature_values']['data']), index_col='Trajectory_ID')
    assert values.loc['Config_5_obj0', 'total_distance'] != pytest.approx(30.944657, abs=1e-3)


def test_visual_inspection_route_then_all_configurations_again(at):
    """Visual Exploration of Config_5/8/16 (object 0, tst 0–136, no court background), then clustering again."""
    at.multiselect(key='sidebar_configs').set_value(['Config_5', 'Config_8', 'Config_16']).run()
    at.selectbox(key='analysis_method').set_value('Visual Exploration').run()
    if 'show_court_background' in [c.key for c in at.checkbox]:
        at.checkbox(key='show_court_background').uncheck().run()
    at.number_input(key='visual_start').set_value(0.0).run()
    at.number_input(key='visual_end').set_value(136.0).run()
    ok(at)
    assert at.session_state['court_type'] == 'No selection'
    charts = [json.loads(c.proto.spec) for c in at.get('plotly_chart')]
    assert charts
    positional = [c for c in charts if c.get('layout', {}).get('yaxis', {}).get('scaleanchor')]
    assert positional and all(c['layout']['yaxis'].get('scaleratio', 1) == 1 for c in positional)
    traces = {str(t.get('name')) for c in charts for t in c.get('data', [])}
    assert not any(name.startswith(('Config_0', 'Config_1 ', 'Config_2')) for name in traces)
    # Restore all 26 configurations before clustering: the run covers all of them again
    at.multiselect(key='sidebar_configs').set_value(CONFIGS).run()
    at.selectbox(key='analysis_method').set_value('Clustering').run()
    check_run(at, 'sinuosity', compute(at, 'sinuosity'))
