"""
PDP progress reports the configuration pairs actually calculated; a repeated identical request says
its stored results were reused; recoverable failures name the stage, settings and next action, and a
failed view keeps the computed distances.

B1: A2_1Hz_dataset.csv, configurations 1, 2, 4, 11, objects 0 and 1, indices 0-5, window 6, Fundamental.
Run from streamlit_deploy/ with PDP_1HZ_CSV pointing to A2_1Hz_dataset.csv.
"""
import io
import os
import warnings

import pandas as pd
import pytest
import streamlit
from streamlit.testing.v1 import AppTest

from modules import pdp_analysis as p
from modules.utils import load_data

warnings.filterwarnings('ignore')
APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'streamlit_visualization.py')
CSV = os.environ.get('PDP_1HZ_CSV')
pytestmark = pytest.mark.skipif(not CSV or not os.path.exists(CSV), reason='set PDP_1HZ_CSV to A2_1Hz_dataset.csv')

B1_CONFIGS = ['Config_1', 'Config_2', 'Config_4', 'Config_11']


class Upload(io.BytesIO):
    name = 'A2_1Hz_dataset.csv'


def coarse():
    return load_data(Upload(open(CSV, 'rb').read()), update_state=False, show_success=False)


def all_configs():
    return sorted(coarse()['config_source'].unique(), key=lambda c: int(c.split('_')[1]))


@pytest.fixture(autouse=True)
def fresh_cache():
    p.clear_pdp_distance_cache()
    yield
    p.clear_pdp_distance_cache()


def test_callback_counts_actual_pairs_and_stays_silent_on_reuse():
    df = coarse()
    calls = []
    args = (df, tuple(B1_CONFIGS), (0, 1), df['tst'].min(), df['tst'].max())
    dm, ids = p.compute_pdp_distance_matrix(*args, window_length=6, progress=lambda d, t: calls.append((d, t)))
    assert calls == [(i, 6) for i in range(1, 7)]
    assert dm[0, 1] == 144
    reused = []
    dm2, _ = p.compute_pdp_distance_matrix(*args, window_length=6, progress=lambda d, t: reused.append((d, t)))
    assert reused == [] and (dm2 == dm).all()


def test_26_configurations_report_325_pairs_per_variant():
    df = coarse()
    configs = all_configs()
    assert len(configs) == 26
    calls = []
    p.compute_pdp_distance_matrix(df, tuple(configs), (0, 1), df['tst'].min(), df['tst'].max(),
                                  window_length=6, progress=lambda d, t: calls.append((d, t)))
    assert len(calls) == 325 and calls[-1] == (325, 325) and {t for _, t in calls} == {325}


# ---------------------------------------------------------------- real app

@pytest.fixture
def app(monkeypatch):
    captured = {}
    original_download = streamlit.download_button

    def capture_download(label, data, *a, **k):
        captured[label] = data
        return original_download(label, data, *a, **k)

    data = open(CSV, 'rb').read()
    monkeypatch.setattr(streamlit, 'file_uploader',
                        lambda label, *a, **k: [Upload(data)] if label.startswith('Upload CSV') else None)
    monkeypatch.setattr(streamlit, 'download_button', capture_download)
    at = AppTest.from_file(APP, default_timeout=300)
    at.session_state['password_correct'] = True
    at.run()
    at.captured = captured
    return at


def prepare(at, configs):
    at.multiselect(key='sidebar_configs').set_value(configs).run()
    at.multiselect(key='sidebar_objects').set_value([0, 1]).run()
    at.selectbox(key='analysis_method').set_value('PDP Analysis').run()
    at.slider(key='pdp_window').set_value(6).run()


def compute(at, variants=None):
    if variants:
        [m for m in at.multiselect if m.label == 'Select PDP Variants to Compute'][0].set_value(variants).run()
    at.button(key='compute_pdp').click().run()
    assert not at.exception, [e.message for e in at.exception]


def texts(elements):
    return [str(e.value) for e in elements]


def b1_pair_1_2(at):
    for f in at.dataframe:
        v = f.value
        if 'Raw PDP distance' in v.columns and 'Config A' in v.columns:
            r = v[(v['Config A'] == 'Config_1') & (v['Config B'] == 'Config_2')]
            if len(r):
                return r.iloc[0]['Raw PDP distance'], r.iloc[0]['Normalised distance (%)']
    raise AssertionError('pair table not found')


def test_fresh_then_repeated_identical_request(app):
    prepare(app, B1_CONFIGS)
    compute(app)
    ready = [t for t in texts(app.success) if 'PDP analysis ready' in t]
    assert ready and 'Fundamental: 6 configuration pairs calculated in' in ready[0]
    assert b1_pair_1_2(app) == ('144', '27.27%')

    compute(app)                                            # same settings again
    ready = [t for t in texts(app.success) if 'PDP analysis ready' in t]
    assert ready and 'Fundamental: stored results for identical settings and data reused' in ready[0]
    assert 'calculated in' not in ready[0]
    assert b1_pair_1_2(app) == ('144', '27.27%')


def test_all_26_configurations(app):
    prepare(app, all_configs())
    compute(app)
    ready = [t for t in texts(app.success) if 'PDP analysis ready' in t]
    assert ready and 'Fundamental: 325 configuration pairs calculated in' in ready[0]


def test_calculation_failure_names_stage_and_hides_older_results(app, monkeypatch):
    prepare(app, B1_CONFIGS)
    compute(app)                                            # valid Fundamental results
    real = p.compute_pdp_distance_matrix

    def fail_on_buffer(*a, **k):
        if k.get('pdp_variant') == 'buffer':
            raise MemoryError('controlled test failure')
        return real(*a, **k)

    monkeypatch.setattr(p, 'compute_pdp_distance_matrix', fail_on_buffer)
    compute(app, ['Fundamental', 'Buffer'])
    errors = [t for t in texts(app.error) if 'PDP calculation failed' in t]
    assert errors, texts(app.error)
    msg = errors[0]
    assert 'distance calculation (Buffer variant)' in msg
    assert 'more memory than is available' in msg
    assert '4 configurations, objects 0, 1' in msg and 'window 6' in msg and 'Fundamental, Buffer' in msg
    assert 'Completed before the failure: Fundamental' in msg
    assert 'Compute PDP Analysis' in msg
    assert any('MemoryError: controlled test failure' in c.value for c in app.code)
    # The older Fundamental-only results are kept but not presented for the failed request
    assert set(app.session_state['pdp_results_all']) == {'fundamental'}
    assert any('Inputs or analysis options changed' in t for t in texts(app.warning))
    assert not any('Raw PDP distance' in f.value.columns for f in app.dataframe)
    assert not any('PDP analysis ready' in t for t in texts(app.success))

    # Retrying after the cause is gone succeeds; the failure message disappears
    monkeypatch.setattr(p, 'compute_pdp_distance_matrix', real)
    compute(app)
    assert not any('PDP calculation failed' in t for t in texts(app.error))
    assert set(app.session_state['pdp_results_all']) == {'fundamental', 'buffer'}


def test_failed_view_keeps_distance_results(app, monkeypatch):
    def broken_mds(*a, **k):
        raise ValueError('controlled MDS failure')

    monkeypatch.setattr(p, 'create_mds_visualization', broken_mds)
    prepare(app, B1_CONFIGS)
    compute(app)
    warnings_ = [t for t in texts(app.warning) if 'Could not build the 2D MDS projection' in t]
    assert warnings_ and 'distance results are not affected' in warnings_[0]
    assert '4 configurations' in warnings_[0] and 'window 6' in warnings_[0]
    assert any('ValueError: controlled MDS failure' in c.value for c in app.code)
    assert not any('PDP calculation failed' in t for t in texts(app.error))
    assert b1_pair_1_2(app) == ('144', '27.27%')
    rankings = pd.read_csv(io.StringIO(app.captured['Download Rankings']))
    row = rankings[(rankings.Target_Config == 'Config_1') & (rankings.Similar_Config == 'Config_2')].iloc[0]
    assert row['Raw_PDP_Distance'] == 144


def test_on_demand_view_opens(app):
    prepare(app, B1_CONFIGS)
    compute(app)
    app.toggle(key='pdp_show_ineq_matrices').set_value(True).run()
    assert not app.exception
    assert not any('Could not build' in t for t in texts(app.warning))
