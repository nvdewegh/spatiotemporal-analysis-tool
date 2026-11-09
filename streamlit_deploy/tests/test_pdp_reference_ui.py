"""
Reference systems in the running app: default G2, unavailable combinations, outdated results after system,
estimator or reference changes, linked cell explanation, dataset level and settings round trip.

Uses A2_detailed_dataset.csv (PDP_DETAILED_CSV, or the course location) with encounters 0-3, tst 0-30.
Run from streamlit_deploy/ (PYTHONPATH=.).
"""
import io
import json
import os
import warnings

import pytest
import streamlit
from streamlit.testing.v1 import AppTest

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(HERE, 'streamlit_visualization.py')
_COURSE_CSV = os.path.join(HERE, '..', '..', '..', '04Exercises', 'AJ2627', 'ExercisesForStudents', 'ForStudents',
                           'Data', 'Data', 'CSV_PDP input', 'A2_detailed_dataset.csv')
CSV = os.environ.get('PDP_DETAILED_CSV') or _COURSE_CSV
pytestmark = pytest.mark.skipif(not os.path.exists(CSV), reason='A2_detailed_dataset.csv not available')
OUTDATED = 'Inputs or analysis options changed'


@pytest.fixture
def at(monkeypatch):
    data = open(CSV, 'rb').read()

    class Upload(io.BytesIO):
        name = 'A2_detailed_dataset.csv'

    monkeypatch.setattr(streamlit, 'file_uploader',
                        lambda label, *a, **k: [Upload(data)] if label.startswith('Upload CSV') else None)
    app = AppTest.from_file(APP, default_timeout=300)
    app.session_state['password_correct'] = True
    app.run()
    app.multiselect(key='sidebar_configs').set_value(['Config_0', 'Config_1', 'Config_2', 'Config_3']).run()
    app.multiselect(key='sidebar_objects').set_value([0, 1]).run()
    app.selectbox(key='analysis_method').set_value('PDP Analysis').run()
    app.number_input(key='pdp_end').set_value(30.0).run()
    return app


def ok(app):
    assert not app.exception, [e.message for e in app.exception]


def outdated(app):
    return any(OUTDATED in w.value for w in app.warning)


def compute(app):
    app.button(key='compute_pdp').click().run()
    ok(app)
    assert not outdated(app)


def test_default_is_g2_mean_and_g2_results_unchanged(at):
    assert at.session_state['pdp_reference_system'] == 'G2'
    assert at.session_state['pdp_c4_estimator'] == 'mean'
    compute(at)
    assert at.session_state['pdp_run_settings']['reference']['system'] == 'G2'
    assert at.session_state['pdp_norm_n_descriptors'] == 2
    assert len(at.session_state['pdp_run_signature']) == 12          # legacy signature, no reference element


def test_buffer_rough_per_descriptor_and_external_points(at):
    at.selectbox(key='pdp_reference_system').set_value('R4').run()
    at.multiselect(key='pdp_variants').set_value(['Fundamental', 'Buffer + Rough']).run()
    ok(at)
    assert not at.button(key='compute_pdp').disabled
    at.number_input(key='pdp_ref_rough_cyclist_road_left').set_value(0.8).run()
    compute(at)
    run = at.session_state['pdp_run_settings']
    assert run['rough']['cyclist_road_left'] == 0.8 and run['rough']['vehicle_road_along'] == 0.5
    assert run['buffer']['cyclist_road_along'] == 1.0
    results = at.session_state['pdp_results_all']
    assert (results['buffer_rough'] != results['fundamental']).any()
    assert at.session_state['pdp_norm_buffers'] == (1.0, 1.0, 1.0, 1.0)
    at.number_input(key='pdp_ref_buffer_vehicle_road_left').set_value(0.0).run()
    assert outdated(at)                                                 # a per-descriptor value is part of the run
    at.number_input(key='pdp_ref_buffer_vehicle_road_left').set_value(1.0).run()
    assert not outdated(at)
    # External points: projected with the same C4 paths; switching them on makes results outdated
    at.selectbox(key='pdp_reference_system').set_value('C4').run()
    at.multiselect(key='pdp_variants').set_value(['Fundamental']).run()
    compute(at)
    without = at.session_state['pdp_results_all']['fundamental'].copy()
    at.session_state['pdp_manual_points'] = [('P', 40.0, 10.0)]
    at.checkbox(key='pdp_use_external').check().run()
    ok(at)
    assert outdated(at)
    compute(at)
    assert at.session_state['pdp_run_settings']['external'] == [('P', 40.0, 10.0)]
    assert at.session_state['pdp_norm_n_external_points'] == 1
    assert (at.session_state['pdp_results_all']['fundamental'] != without).any()


def test_system_estimator_changes_make_results_outdated(at):
    at.selectbox(key='pdp_reference_system').set_value('C4').run()
    compute(at)
    first = at.session_state['pdp_results_all']['fundamental'].copy()
    assert at.session_state['pdp_norm_n_descriptors'] == 4
    at.radio(key='pdp_c4_estimator').set_value('median').run()
    ok(at)
    assert outdated(at)                                                 # no stale matrices or clusters shown
    assert not any(e.label.startswith('Reference systems:') for e in at.expander)
    compute(at)
    assert (at.session_state['pdp_results_all']['fundamental'] != first).any()
    assert at.session_state['pdp_run_settings']['reference']['estimator'] == 'median'
    at.selectbox(key='pdp_reference_system').set_value('R2').run()
    assert outdated(at)


def test_xy_views_guarded_and_cell_explanation_linked(at):
    at.selectbox(key='pdp_reference_system').set_value('C4').run()
    compute(at)
    assert any('available for G2 results only' in i.value for i in at.info)
    at.selectbox(key='pdp_xs_chan_b_C4').set_value('vehicle_flow_along').run()
    ok(at)
    cell = at.selectbox(key='pdp_xs_cell')
    assert len(cell.options) > 0
    cell.set_value(0).run()
    ok(at)
    table = [d.value for d in at.dataframe if 'Role in cell' in d.value.columns][0]
    assert list(table['Role in cell']) == ['row point i', 'column point j']
    assert any('closest projection on the vehicle mean flow path' in v for v in table['B projection'])


def test_dataset_level_tied_to_run(at):
    at.selectbox(key='pdp_reference_system').set_value('R4').run()
    compute(at)
    at.multiselect(key='pdp_ds_specs').set_value(['G2', 'R4']).run()
    at.button(key='pdp_ds_compute').click().run()
    ok(at)
    assert 'pdp_ds_compute' not in [b.key for b in at.button]           # results shown
    at.number_input(key='pdp_end').set_value(25.0).run()
    assert outdated(at)
    compute(at)
    assert 'pdp_ds_compute' in [b.key for b in at.button]               # new run: compute again, nothing stale


def test_settings_round_trip(at):
    at.selectbox(key='pdp_reference_system').set_value('C4').run()
    at.radio(key='pdp_c4_estimator').set_value('median').run()
    box = [c for c in at.sidebar.code if '"samtool-settings"' in c.value]
    payload = json.loads(box[0].value)
    assert payload['settings']['pdp_reference_system'] == 'C4'
    assert payload['settings']['pdp_c4_estimator'] == 'median'
    at.selectbox(key='pdp_reference_system').set_value('G2').run()
    at.text_area(key='settings_import_text').set_value(box[0].value)
    at.button(key='btn_import_settings').click().run()
    ok(at)
    assert at.session_state['pdp_reference_system'] == 'C4'
    assert at.session_state['pdp_c4_estimator'] == 'median'
    bad = dict(payload, settings=dict(payload['settings'], pdp_reference_system='R9'))
    at.text_area(key='settings_import_text').set_value(json.dumps(bad))
    at.button(key='btn_import_settings').click().run()
    assert at.session_state['pdp_reference_system'] == 'C4'
