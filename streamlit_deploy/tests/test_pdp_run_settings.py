"""
Variant activation stays explicit, results are tied to the settings that produced them, and only
changes that affect the calculation make results outdated.

B1: A2_1Hz_dataset.csv, configurations 1, 2, 4, 11, objects 0 and 1, indices 0-5, window 6,
Fundamental, external points off. Run from streamlit_deploy/ with PDP_1HZ_CSV pointing to it.
"""
import io
import os
import warnings

import pandas as pd
import pytest
import streamlit
from streamlit.testing.v1 import AppTest

warnings.filterwarnings('ignore')
APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'streamlit_visualization.py')
CSV = os.environ.get('PDP_1HZ_CSV')
pytestmark = pytest.mark.skipif(not CSV or not os.path.exists(CSV), reason='set PDP_1HZ_CSV to A2_1Hz_dataset.csv')

OUTDATED = 'Inputs or analysis options changed'


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
    at.multiselect(key='sidebar_configs').set_value(['Config_1', 'Config_2', 'Config_4', 'Config_11']).run()
    at.multiselect(key='sidebar_objects').set_value([0, 1]).run()
    at.selectbox(key='analysis_method').set_value('PDP Analysis').run()
    at.slider(key='pdp_window').set_value(6).run()
    at.captured = captured
    return at


def compute(at, variants=None):
    if variants:
        at.multiselect(key='pdp_variants').set_value(variants).run()
    at.button(key='compute_pdp').click().run()
    assert not at.exception, [e.message for e in at.exception]


def outdated(at):
    return any(OUTDATED in str(w.value) for w in at.warning)


def ranking(at, variant, a, b):
    at.selectbox(key='pdp_exp_variant').set_value(variant).run()
    df = pd.read_csv(io.StringIO(at.captured['Download Rankings']))
    row = df[(df.Target_Config == f'Config_{a}') & (df.Similar_Config == f'Config_{b}')].iloc[0]
    return row['Raw_PDP_Distance'], round(row['Normalised_Distance_%'], 4)


def pending(at):
    return next(str(m.value) for m in at.markdown if '**Variants:**' in str(m.value))


def captions(at):
    return [str(c.value) for c in at.caption]


def test_unused_rough_fields_do_not_activate_rough(app):
    compute(app)
    assert ranking(app, 'Fundamental', 1, 2) == (144, 27.2727)
    assert ranking(app, 'Fundamental', 1, 4) == (0, 0)

    app.number_input(key='pdp_rough_x').set_value(2.0).run()
    app.number_input(key='pdp_rough_y').set_value(2.0).run()
    assert app.multiselect(key='pdp_variants').value == ['Fundamental']          # not switched on
    assert 'not used: entered X/Y 2/2 m' in pending(app)
    assert not outdated(app)                        # the Fundamental result did not use Rough
    compute(app)
    assert set(app.session_state['pdp_results_all']) == {'fundamental'}
    assert app.session_state['pdp_run_settings']['rough'] is None
    assert ranking(app, 'Fundamental', 1, 4) == (0, 0)


def test_explicit_rough_and_switching_views(app):
    app.number_input(key='pdp_rough_x').set_value(2.0).run()
    app.number_input(key='pdp_rough_y').set_value(2.0).run()
    compute(app, ['Fundamental', 'Rough'])
    run = app.session_state['pdp_run_settings']
    assert run['rough'] == (2.0, 2.0) and run['buffer'] is None and run['external'] == []
    assert ranking(app, 'Rough', 1, 4) == (14, 2.6515)
    assert ranking(app, 'Fundamental', 1, 4) == (0, 0)

    for name, expected in (('Fundamental', 'no Rough'), ('Rough', 'Rough X/Y 2/2 m')):
        app.selectbox(key='pdp_clust_variant').set_value(name).run()
        line = [c for c in captions(app) if f'**{name} result**' in c and 'window 6' in c]
        assert line and expected in line[0] and 'external points off' in line[0], name
    # Overlays follow the displayed variant's completed settings
    app.selectbox(key='pdp_clust_variant').set_value('Fundamental').run()
    assert app.checkbox(key='pdp_show_rough').disabled
    app.selectbox(key='pdp_clust_variant').set_value('Rough').run()
    assert not app.checkbox(key='pdp_show_rough').disabled

    # Changing a value that the computed Rough result used makes results outdated
    app.number_input(key='pdp_rough_x').set_value(1.0).run()
    assert outdated(app)


def test_external_point_on_then_off(app):
    compute(app)
    app.checkbox(key='pdp_use_external').check().run()
    assert '**External points:** ON, but no points chosen' in pending(app)
    assert not outdated(app)                              # no point added yet: same calculation
    app.text_input(key='pdp_custom_name').input('Net post').run()
    app.number_input(key='pdp_custom_x').set_value(5.0).run()
    app.number_input(key='pdp_custom_y').set_value(10.0).run()
    app.button(key='pdp_add_custom').click().run()
    assert 'Net post (5, 10)' in pending(app)
    assert outdated(app)                                  # results did not include the point

    app.checkbox(key='pdp_use_external').uncheck().run()
    assert '**External points:** OFF' in pending(app)
    assert not outdated(app)                              # back to the computed settings
    assert app.session_state['pdp_run_settings']['external'] == []

    # Computed with the point, then switched off: the earlier result is outdated
    app.checkbox(key='pdp_use_external').check().run()
    compute(app)
    assert app.session_state['pdp_run_settings']['external'] == [('Net post', 5.0, 10.0)]
    app.checkbox(key='pdp_use_external').uncheck().run()
    assert '**External points:** OFF' in pending(app)
    assert outdated(app)


def test_display_only_interactions_keep_results_current(app):
    compute(app)
    app.checkbox(key='pdp_show_normalized').check().run()
    app.slider(key='pdp_n_clusters').set_value(3).run()
    app.selectbox(key='pdp_clust_variant').set_value('Fundamental').run()
    app.toggle(key='pdp_show_ineq_matrices').set_value(True).run()
    app.number_input(key='pdp_buffer_x').set_value(0.4).run()      # Buffer not selected: unused
    assert not app.exception
    assert not outdated(app)
    assert ranking(app, 'Fundamental', 1, 2) == (144, 27.2727)


def test_scale_choice_survives_outdated_results_and_recompute(app):
    compute(app)
    app.checkbox(key='pdp_show_normalized').check().run()
    app.slider(key='pdp_window').set_value(5).run()       # outdated: the matrix (and its checkbox) is not drawn
    assert outdated(app)
    compute(app)
    assert app.checkbox(key='pdp_show_normalized').value
