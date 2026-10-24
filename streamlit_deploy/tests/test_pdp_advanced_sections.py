"""
The three advanced PDP sections (Parameter Impact Analysis, Configuration Similarity Explorer, Cluster
Quality Metrics) are reachable with both datasets, explained when results are missing or outdated,
run only on request, use the intended run and variant, and never show results of other settings.

B1: A2_1Hz_dataset.csv, configurations 1, 2, 4, 11, objects 0 and 1, indices 0-5, window 6, Fundamental.
D1: A2_detailed_dataset.csv, same configurations and objects, indices 2-122, window 121, Fundamental.
Run from streamlit_deploy/ with PDP_1HZ_CSV pointing to A2_1Hz_dataset.csv (detailed file alongside).
"""
import io
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
DETAILED = os.path.join(os.path.dirname(CSV), 'A2_detailed_dataset.csv') if CSV else None
pytestmark = pytest.mark.skipif(not CSV or not os.path.exists(CSV) or not os.path.exists(DETAILED),
                                reason='set PDP_1HZ_CSV to A2_1Hz_dataset.csv (with A2_detailed_dataset.csv)')

B1_CONFIGS = ['Config_1', 'Config_2', 'Config_4', 'Config_11']
SECTIONS = ('Parameter Impact Analysis', 'Configuration Similarity Explorer', 'Cluster Quality Metrics')


@pytest.fixture
def app(monkeypatch):
    current = {'path': CSV}

    class Upload(io.BytesIO):
        pass

    def uploader(label, *a, **k):
        if not label.startswith('Upload CSV'):
            return None
        upload = Upload(open(current['path'], 'rb').read())
        upload.name = os.path.basename(current['path'])
        return [upload]

    captured = {}
    original_download = streamlit.download_button

    def capture_download(label, data, *a, **k):
        captured[label] = data
        return original_download(label, data, *a, **k)

    monkeypatch.setattr(streamlit, 'file_uploader', uploader)
    monkeypatch.setattr(streamlit, 'download_button', capture_download)
    at = AppTest.from_file(APP, default_timeout=600)
    at.session_state['password_correct'] = True
    at.run()
    at.current, at.captured = current, captured
    select(at, B1_CONFIGS)
    at.selectbox(key='analysis_method').set_value('PDP Analysis').run()
    at.slider(key='pdp_window').set_value(6).run()
    return at


def select(at, configs):
    at.multiselect(key='sidebar_configs').set_value(configs).run()
    at.multiselect(key='sidebar_objects').set_value([0, 1]).run()


def use_dataset(at, path, first, last, window):
    at.current['path'] = path
    at.run()
    select(at, B1_CONFIGS)
    data = at.session_state['data']
    times = sorted(data[data.config_source.isin(B1_CONFIGS) & data.obj.isin([0, 1])].tst.unique())
    at.number_input(key='pdp_start').set_value(float(times[first])).run()   # start first: it resets End
    at.number_input(key='pdp_end').set_value(float(times[last])).run()
    at.slider(key='pdp_window').set_value(window).run()


def compute(at, variants=None):
    if variants:
        at.multiselect(key='pdp_variants').set_value(variants).run()
    at.button(key='compute_pdp').click().run()
    assert not at.exception, [e.message for e in at.exception]


def sections(at):
    return [s for s in SECTIONS if any(e.label.startswith(s) for e in at.expander)]


def explained(at):
    return any('appear below the results once' in str(c.value) for c in at.caption)


def texts(elements):
    return [str(e.value) for e in elements]


def ranking(at, variant, a, b):
    at.selectbox(key='pdp_exp_variant').set_value(variant).run()
    df = pd.read_csv(io.StringIO(at.captured['Download Rankings']))
    row = df[(df.Target_Config == f'Config_{a}') & (df.Similar_Config == f'Config_{b}')].iloc[0]
    return row['Raw_PDP_Distance'], round(row['Normalised_Distance_%'], 2)


def table(at, first_column, column):
    for f in at.dataframe:
        if list(f.value.columns)[:2] == ['Rank', 'Config'] and column in f.value.columns:
            if first_column is None or f.value.iloc[0]['Config'] == first_column:
                return f.value
    raise AssertionError('table not found')


def compare(at, bx, by, rx, ry):
    for key, value in (('comp_buffer_x', bx), ('comp_buffer_y', by), ('comp_rough_x', rx), ('comp_rough_y', ry)):
        at.number_input(key=key).set_value(value).run()
    at.button(key='compare_variants').click().run()
    assert not at.exception, [e.message for e in at.exception]


def test_sections_explained_until_results_are_current(app):
    assert sections(app) == [] and explained(app)               # before any computation
    compute(app)
    assert sections(app) == list(SECTIONS)
    assert 'variant_comparison_results' not in app.session_state  # nothing computed on opening
    assert 'cluster_quality_metrics' not in app.session_state
    app.slider(key='pdp_window').set_value(5).run()                # outdated
    assert sections(app) == [] and explained(app)


def test_parameter_impact_uses_main_run_and_trial_values(app):
    app.checkbox(key='pdp_use_external').check().run()
    app.text_input(key='pdp_custom_name').input('Net post').run()
    app.number_input(key='pdp_custom_x').set_value(5.0).run()
    app.number_input(key='pdp_custom_y').set_value(10.0).run()
    app.button(key='pdp_add_custom').click().run()
    compute(app)
    main = app.session_state['pdp_results_all']['fundamental'].copy()
    chosen_k = app.slider(key='pdp_n_clusters').value

    compare(app, 0.5, 0.5, 2.0, 2.0)
    stored = app.session_state['variant_comparison_results']
    results = stored['results']
    np.testing.assert_array_equal(results['fundamental']['matrix'], main)   # same inputs, external point kept
    caption = [c for c in texts(app.caption) if c.startswith('Compared on the main run')][0]
    assert 'Buffer X/Y 0.5/0.5 m' in caption and 'Rough X/Y 2/2 m' in caption and '1 external point(s)' in caption
    # The main analysis is untouched
    assert set(app.session_state['pdp_results_all']) == {'fundamental'}
    np.testing.assert_array_equal(app.session_state['pdp_results_all']['fundamental'], main)
    assert app.slider(key='pdp_n_clusters').value == chosen_k
    assert app.session_state['pdp_run_settings']['rough'] is None

    app.number_input(key='comp_rough_x').set_value(1.0).run()           # other trial value
    assert any('previous comparison used another main run or other trial values' in t for t in texts(app.info))
    assert not any(t.startswith('Compared on the main run') for t in texts(app.caption))


def test_parameter_impact_rough_values_without_external_points(app):
    compute(app)
    compare(app, 0.5, 0.5, 2.0, 2.0)
    results = app.session_state['variant_comparison_results']['results']
    ids = list(results['rough']['config_ids'])
    i, j = ids.index('Config_1'), ids.index('Config_4')
    assert results['fundamental']['matrix'][i, j] == 0 and results['rough']['matrix'][i, j] == 14


def test_similarity_explorer_query_k_and_variant(app):
    app.number_input(key='pdp_rough_x').set_value(2.0).run()
    app.number_input(key='pdp_rough_y').set_value(2.0).run()
    compute(app, ['Fundamental', 'Rough'])
    main_query = app.session_state['pdp_target_config'], app.session_state['pdp_k_similar']
    app.selectbox(key='similarity_target_config').set_value('Config_1').run()
    app.number_input(key='similarity_k').set_value(2).run()
    for variant, raw in (('Fundamental', '0'), ('Rough', '14')):
        app.selectbox(key='pdp_sim_variant').set_value(variant).run()
        similar = table(app, 'Config_4', 'Raw PDP distance')
        assert 'Config_1' not in list(similar['Config']) and len(similar) == 2
        assert similar.iloc[0]['Raw PDP distance'] == raw, variant
    assert similar.iloc[0]['Normalised distance (%)'] == '2.65%'
    assert (app.session_state['pdp_target_config'], app.session_state['pdp_k_similar']) == main_query

    app.number_input(key='similarity_k').set_value(3).run()
    select(app, ['Config_2', 'Config_4', 'Config_11'])                  # Config_1 no longer selected
    compute(app, ['Fundamental'])
    assert not app.exception
    assert app.selectbox(key='similarity_target_config').value == 'Config_2'
    assert app.number_input(key='similarity_k').value == 2
    assert any('Similarity Explorer target reset to Config_2' in t for t in texts(app.info))


def test_cluster_quality_small_range_and_insufficient_configurations(app):
    compute(app)
    chosen_k = app.slider(key='pdp_n_clusters').value
    app.number_input(key='cluster_quality_min_k').set_value(2).run()
    app.number_input(key='cluster_quality_max_k').set_value(3).run()
    app.button(key='compute_cluster_quality').click().run()
    assert not app.exception
    stored = app.session_state['cluster_quality_metrics']
    assert stored['results']['k_values'] == [2, 3] and stored['key'][1:] == ('fundamental', 2, 3)
    assert any('2D MDS embedding' in t for t in texts(app.caption))
    assert app.slider(key='pdp_n_clusters').value == chosen_k

    app.number_input(key='cluster_quality_max_k').set_value(2).run()   # single k is a valid request
    assert any('previous evaluation used another run, variant or k range' in t for t in texts(app.info))
    app.button(key='compute_cluster_quality').click().run()
    assert app.session_state['cluster_quality_metrics']['results']['k_values'] == [2]

    select(app, ['Config_1', 'Config_2'])
    compute(app)
    assert any('Cluster quality needs at least 3 configurations' in t for t in texts(app.warning))


def test_coarse_detailed_coarse(app):
    compute(app)
    assert ranking(app, 'Fundamental', 1, 2) == (144, 27.27)
    compare(app, 0.5, 0.5, 2.0, 2.0)
    app.button(key='compute_cluster_quality').click().run()

    use_dataset(app, DETAILED, 2, 122, 121)
    assert sections(app) == [] and explained(app)                     # outdated until recomputed
    compute(app)
    assert sections(app) == list(SECTIONS)
    assert ranking(app, 'Fundamental', 1, 4) == (3208, 1.38)
    # Earlier coarse results are not presented for the detailed run
    assert any('previous comparison used another main run' in t for t in texts(app.info))
    assert any('previous evaluation used another run' in t for t in texts(app.info))
    # Minimal detailed requests in each section
    compare(app, 0.0, 0.0, 0.5, 0.5)                                    # Rough only: no extra points
    results = app.session_state['variant_comparison_results']['results']
    ids = list(results['fundamental']['config_ids'])
    assert results['fundamental']['matrix'][ids.index('Config_1'), ids.index('Config_4')] == 3208
    app.button(key='compute_cluster_quality').click().run()
    assert not app.exception and app.session_state['cluster_quality_metrics']['results']['k_values']
    similar = table(app, None, 'Raw PDP distance')
    assert len(similar) >= 1

    use_dataset(app, CSV, 0, 5, 6)
    compute(app)
    assert sections(app) == list(SECTIONS)
    assert ranking(app, 'Fundamental', 1, 2) == (144, 27.27)
    assert any('previous comparison used another main run' in t for t in texts(app.info))
