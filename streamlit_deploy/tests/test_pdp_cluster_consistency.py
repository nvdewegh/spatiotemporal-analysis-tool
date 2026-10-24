"""
The user's chosen cluster count drives the membership table, dendrogram, MDS colours,
cluster-based selections and the cluster export, per variant and per analytical run.
Partitions are compared as sets of configurations (cluster numbers and colours are arbitrary).

Uses the real app in Streamlit's test runner with B1 (A2_1Hz_dataset.csv, configurations
1, 2, 4, 11, objects 0 and 1, indices 0-5, window 6, Fundamental).
Run from streamlit_deploy/ with PDP_1HZ_CSV pointing to A2_1Hz_dataset.csv.
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

warnings.filterwarnings('ignore')
APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'streamlit_visualization.py')
CSV = os.environ.get('PDP_1HZ_CSV')
pytestmark = pytest.mark.skipif(not CSV or not os.path.exists(CSV), reason='set PDP_1HZ_CSV to A2_1Hz_dataset.csv')

B1_CONFIGS = ['Config_1', 'Config_2', 'Config_4', 'Config_11']
TWO = [{'1', '4'}, {'2', '11'}]
THREE = [{'1', '4'}, {'2'}, {'11'}]


def as_partition(configs, labels):
    groups = {}
    for cfg, lab in zip(configs, labels):
        groups.setdefault(lab, set()).add(str(cfg).replace('Config_', ''))
    return sorted(groups.values(), key=lambda g: sorted(g))


def same(p1, p2):
    return sorted(map(sorted, p1)) == sorted(map(sorted, p2))


@pytest.fixture
def app(monkeypatch):
    data = open(CSV, 'rb').read()

    class Upload(io.BytesIO):
        name = 'A2_1Hz_dataset.csv'

    captured = {}
    original_download = streamlit.download_button

    def capture_download(label, data, *a, **k):
        captured[label] = data
        return original_download(label, data, *a, **k)

    monkeypatch.setattr(streamlit, 'file_uploader',
                        lambda label, *a, **k: [Upload(data)] if label.startswith('Upload CSV') else None)
    monkeypatch.setattr(streamlit, 'download_button', capture_download)
    at = AppTest.from_file(APP, default_timeout=300)
    at.session_state['password_correct'] = True
    at.run()
    at.multiselect(key='sidebar_configs').set_value(B1_CONFIGS).run()
    at.multiselect(key='sidebar_objects').set_value([0, 1]).run()
    at.selectbox(key='analysis_method').set_value('PDP Analysis').run()
    at.slider(key='pdp_window').set_value(6).run()
    at.captured = captured
    return at


def compute(at, variants=None):
    if variants:
        [m for m in at.multiselect if m.label == 'Select PDP Variants to Compute'][0].set_value(variants).run()
    at.button(key='compute_pdp').click().run()
    assert not at.exception, [e.message for e in at.exception]


def displayed(at):
    for frame in at.dataframe:
        v = frame.value
        if list(v.columns) == ['Configuration', 'Cluster']:
            return as_partition(v['Configuration'], v['Cluster'])
    raise AssertionError('membership table not found')


def exported(at):
    csv = pd.read_csv(io.StringIO(at.captured['Download Cluster Labels']))
    return as_partition(csv.iloc[:, 0], csv.iloc[:, 1])


def mds_partition(at):
    for chart in at.get('plotly_chart'):
        spec = json.loads(chart.proto.spec)
        title = spec.get('layout', {}).get('title', {})
        if isinstance(title, dict) and 'MDS' in title.get('text', ''):
            groups = []
            for trace in spec['data']:
                names = trace.get('text') or trace.get('customdata')
                if names and trace.get('mode', '').startswith('markers'):
                    groups.append({str(n).replace('Config_', '') for n in names})
            return [g for g in groups if g]
    raise AssertionError('MDS chart not found')


def set_clusters(at, k):
    at.slider(key='pdp_n_clusters').set_value(k).run()
    assert not at.exception


def test_b1_raw_distances_unchanged(app):
    compute(app)
    dm = app.session_state['pdp_results_all']['fundamental']
    ids = list(app.session_state['pdp_config_ids'])
    pairs = [(1, 2), (1, 4), (1, 11), (2, 4), (2, 11), (4, 11)]
    got = [int(dm[ids.index(f'Config_{a}'), ids.index(f'Config_{b}')]) for a, b in pairs]
    assert got == [144, 0, 120, 144, 24, 120]


@pytest.mark.parametrize('k, expected', [(2, TWO), (3, THREE)])
def test_displayed_exported_and_mds_follow_chosen_count(app, k, expected):
    compute(app)
    set_clusters(app, k)
    assert same(displayed(app), expected)
    assert same(exported(app), expected)                      # download reproduces the displayed partition
    assert same(mds_partition(app), expected)                 # MDS colours follow the chosen partition
    assert same(as_partition(app.session_state['pdp_config_ids'], app.session_state['pdp_cluster_labels']), expected)


def test_changing_count_updates_views_and_export(app):
    compute(app)
    set_clusters(app, 2)
    assert same(exported(app), TWO)
    set_clusters(app, 3)
    assert same(displayed(app), THREE) and same(exported(app), THREE) and same(mds_partition(app), THREE)


def test_recompute_with_window_1_keeps_chosen_count(app):
    compute(app)
    set_clusters(app, 2)
    app.slider(key='pdp_window').set_value(1).run()             # results now outdated (page stops before slider)
    compute(app)
    assert app.slider(key='pdp_n_clusters').value == 2         # deliberate choice kept, not the recommendation
    labels = app.session_state['pdp_cluster_labels']
    assert len(set(labels)) == 2 and same(exported(app), displayed(app))


def test_switching_variant_uses_that_variants_labels(app):
    app.number_input(key='pdp_rough_x').set_value(2.0).run()
    app.number_input(key='pdp_rough_y').set_value(2.0).run()
    compute(app, ['Fundamental', 'Rough'])
    set_clusters(app, 2)
    ids = list(app.session_state['pdp_config_ids'])
    from modules import pdp_analysis as p
    for name, key in (('Fundamental', 'fundamental'), ('Rough', 'rough')):
        app.selectbox(key='pdp_clust_variant').set_value(name).run()
        app.selectbox(key='pdp_exp_variant').set_value(name).run()
        expected_labels, _ = p.perform_hierarchical_clustering(app.session_state['pdp_results_all'][key], 2)
        expected = as_partition(ids, expected_labels)
        assert same(displayed(app), expected), name
        assert same(exported(app), expected), name
        assert same(mds_partition(app), expected), name


def test_reduced_selection_adjusts_invalid_count_query_and_k(app):
    compute(app)
    set_clusters(app, 3)
    app.selectbox(key='pdp_target_config').set_value('Config_11').run()
    app.slider(key='pdp_k_similar').set_value(3).run()
    app.multiselect(key='sidebar_configs').set_value(['Config_1', 'Config_2']).run()
    compute(app)
    infos = ' '.join(i.value for i in app.info)
    assert 'Number of clusters adjusted from 3 to 2' in infos
    assert 'Config_11 is no longer selected' in infos
    assert app.session_state['pdp_target_config'] in ('Config_1', 'Config_2')
    assert not app.exception
