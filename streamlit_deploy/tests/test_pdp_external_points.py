"""
External reference points: unrestricted coordinates, coordinates copied unrounded from a measured
observation, points with a shared name kept separate, and the PDP inputs always equal to the enabled list.

B1: A2_1Hz_dataset.csv, configurations 1, 2, 4, 11, objects 0 and 1, indices 0-5, window 6, Fundamental.
Run from streamlit_deploy/ with PDP_1HZ_CSV pointing to A2_1Hz_dataset.csv.
"""
import io
import json
import base64
import os
import warnings

import numpy as np
import pandas as pd
import pytest
import streamlit
from streamlit.testing.v1 import AppTest

from modules import pdp_analysis as p

warnings.filterwarnings('ignore')
APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'streamlit_visualization.py')
CSV = os.environ.get('PDP_1HZ_CSV')
pytestmark = pytest.mark.skipif(not CSV or not os.path.exists(CSV), reason='set PDP_1HZ_CSV to A2_1Hz_dataset.csv')

B1_CONFIGS = ['Config_1', 'Config_2', 'Config_4', 'Config_11']


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

    captured, used = {}, []
    original_download = streamlit.download_button
    real_compute = p.compute_pdp_distance_matrix

    def capture_download(label, data, *a, **k):
        captured[label] = data
        return original_download(label, data, *a, **k)

    def spy(*a, **k):
        used.append(k.get('external_points'))
        return real_compute(*a, **k)

    monkeypatch.setattr(streamlit, 'file_uploader', uploader)
    monkeypatch.setattr(streamlit, 'download_button', capture_download)
    monkeypatch.setattr(p, 'compute_pdp_distance_matrix', spy)
    at = AppTest.from_file(APP, default_timeout=300)
    at.session_state['password_correct'] = True
    at.run()
    at.multiselect(key='sidebar_configs').set_value(B1_CONFIGS).run()
    at.multiselect(key='sidebar_objects').set_value([0, 1]).run()
    at.selectbox(key='analysis_method').set_value('PDP Analysis').run()
    at.slider(key='pdp_window').set_value(6).run()
    at.current, at.captured, at.used = current, captured, used
    return at


def compute(at):
    at.used.clear()
    at.button(key='compute_pdp').click().run()
    assert not at.exception, [e.message for e in at.exception]
    return at.used


def add_point(at, name, x, y):
    at.text_input(key='pdp_custom_name').input(name).run()
    at.number_input(key='pdp_custom_x').set_value(x).run()
    at.number_input(key='pdp_custom_y').set_value(y).run()
    at.button(key='pdp_add_custom').click().run()
    assert not at.exception


def preview(at):
    def decode(v):
        return np.frombuffer(base64.b64decode(v['bdata']), dtype=v['dtype']).tolist() if isinstance(v, dict) else v
    for chart in at.get('plotly_chart'):
        spec = json.loads(chart.proto.spec)
        if 'External Reference Points' in str(spec['layout'].get('title')):
            return {t.get('name'): (decode(t['x']), decode(t['y'])) for t in spec['data']}, spec['layout']
    raise AssertionError('preview not found')


def outdated(at):
    return any('Inputs or analysis options changed' in str(w.value) for w in at.warning)


def test_manual_points_beyond_old_limits_reach_pdp(app):
    app.checkbox(key='pdp_use_external').check().run()
    add_point(app, 'A', 35.0, 10.0)
    add_point(app, 'Far', 150.0, -40.0)
    expected = [('A', 35.0, 10.0), ('Far', 150.0, -40.0)]
    assert app.session_state['pdp_external_points'] == expected
    traces, layout = preview(app)
    assert traces['External points (in the list)'] == ([35.0, 150.0], [10.0, -40.0])
    assert traces['Candidate (not added)'] == ([150.0], [-40.0])
    x0, x1 = layout['xaxis']['range']
    y0, y1 = layout['yaxis']['range']
    assert x0 < 35 and x1 > 150 and y0 < -40 and y1 > 10                # reset view shows both points
    used = compute(app)
    assert used and all(list(u) == expected for u in used)
    assert app.session_state['pdp_run_settings']['external'] == expected


def test_copy_observation_unrounded_then_edit(app):
    data = app.session_state['data']
    sidebar = list(app.multiselect(key='sidebar_configs').value)
    app.checkbox(key='pdp_use_external').check().run()
    app.selectbox(key='pdp_src_config').set_value('Config_2').run()
    app.selectbox(key='pdp_src_obj').set_value(1).run()
    app.selectbox(key='pdp_src_tst').set_value(3).run()
    row = data[(data.config_source == 'Config_2') & (data.obj == 1) & (data.tst == 3)].iloc[0]
    app.button(key='pdp_src_use').click().run()
    assert app.number_input(key='pdp_custom_x').value == float(row.x)          # full precision
    assert app.number_input(key='pdp_custom_y').value == float(row.y)
    assert app.session_state['pdp_external_points'] == []                    # nothing added yet
    assert list(app.multiselect(key='sidebar_configs').value) == sidebar     # analysis selection unchanged
    traces, _ = preview(app)
    assert traces['Candidate (not added)'] == ([float(row.x)], [float(row.y)])
    app.number_input(key='pdp_custom_y').set_value(12.5).run()               # edit
    app.button(key='pdp_add_custom').click().run()
    name = app.text_input(key='pdp_custom_name').value
    assert app.session_state['pdp_external_points'] == [(name, float(row.x), 12.5)]
    used = compute(app)
    # Static: one fixed position, replicated for every timestamp
    assert all(list(u) == [(name, float(row.x), 12.5)] for u in used)


def test_same_name_points_stay_separate(app):
    app.checkbox(key='pdp_use_external').check().run()
    add_point(app, 'Ref', 35.0, 10.0)
    add_point(app, 'Ref', 150.0, -40.0)
    points = app.session_state['pdp_external_points']
    assert points == [('Ref', 35.0, 10.0), ('Ref', 150.0, -40.0)]
    data = app.session_state['data']
    sample = data[(data.config_source == 'Config_1') & data.obj.isin([0, 1]) & (data.tst <= 5)]
    expanded = p.add_external_points_to_data(sample, points, sorted(sample.tst.unique()))
    per_time = expanded[expanded.tst == 0]
    assert len(per_time) == 4 and set(per_time[per_time.is_external == True].obj) == {'EXT_0', 'EXT_1'}
    compute(app)
    matrix = app.session_state['pdp_results_all']['fundamental']
    assert matrix.shape == (4, 4)


def test_remove_disable_then_b1_unchanged(app):
    app.checkbox(key='pdp_use_external').check().run()
    add_point(app, 'A', 35.0, 10.0)
    add_point(app, 'Far', 150.0, -40.0)
    compute(app)
    app.button(key='pdp_del_manual_0').click().run()                         # remove the first point
    assert app.session_state['pdp_external_points'] == [('Far', 150.0, -40.0)]
    assert outdated(app)
    used = compute(app)
    assert all(list(u) == [('Far', 150.0, -40.0)] for u in used)
    app.checkbox(key='pdp_use_external').uncheck().run()
    assert outdated(app)
    used = compute(app)
    assert used and all(u is None for u in used)                             # excluded from the calculation
    assert app.session_state['pdp_run_settings']['external'] == []
    app.selectbox(key='pdp_exp_variant').set_value('Fundamental').run()
    rankings = pd.read_csv(io.StringIO(app.captured['Download Rankings']))
    row = rankings[(rankings.Target_Config == 'Config_1') & (rankings.Similar_Config == 'Config_2')].iloc[0]
    assert row.Raw_PDP_Distance == 144 and round(row['Normalised_Distance_%'], 2) == 27.27


def test_source_selection_valid_after_dataset_switch(app, tmp_path):
    app.checkbox(key='pdp_use_external').check().run()
    app.selectbox(key='pdp_src_config').set_value('Config_11').run()
    raw = pd.read_csv(CSV)
    small = tmp_path / 'A2_small.csv'
    raw[raw.iloc[:, 0].isin(sorted(raw.iloc[:, 0].unique())[:3])].to_csv(small, index=False)
    app.current['path'] = str(small)
    app.run()                                       # opens with the baseline: Visual Exploration, points off
    assert not app.exception, [e.message for e in app.exception]
    assert app.session_state['pdp_use_external'] is False
    app.selectbox(key='analysis_method').set_value('PDP Analysis').run()
    app.checkbox(key='pdp_use_external').check().run()
    assert not app.exception, [e.message for e in app.exception]
    options = app.selectbox(key='pdp_src_config').options
    assert 'Config_11' not in options and app.selectbox(key='pdp_src_config').value in options
