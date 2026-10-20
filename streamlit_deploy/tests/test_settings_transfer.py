"""
Tests for "Copy / import settings": the copied JSON is valid and complete, importing it
restores the settings, and invalid or incompatible input leaves the current settings unchanged.

Run from streamlit_deploy/:  python -m pytest tests
"""
import io
import json
import os
import warnings

import pytest
import streamlit
from streamlit.testing.v1 import AppTest

warnings.filterwarnings('ignore')
APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'streamlit_visualization.py')


def csv_bytes(n_configs=12, n_objects=3, n_times=6):
    rows = ''.join(f'{c},{t},{o},{c + t + o},{t * o}\n'
                   for c in range(n_configs) for t in range(n_times) for o in range(n_objects))
    return ('conID,tstID,poiID,x,y\n' + rows).encode()


@pytest.fixture
def at(monkeypatch):
    data = csv_bytes()

    class Upload(io.BytesIO):
        name = 'test.csv'

    monkeypatch.setattr(streamlit, 'file_uploader',
                        lambda label, *a, **k: [Upload(data)] if label.startswith('Upload CSV') else None)
    app = AppTest.from_file(APP, default_timeout=180)
    app.session_state['password_correct'] = True
    app.run()
    app.run()   # second run: the selection widgets exist and the copy box reflects them
    return app


def copied_json(app):
    box = [c for c in app.sidebar.code if '"samtool-settings"' in c.value]
    assert box, 'copy box not found'
    return box[0].value


def import_text(app, text):
    app.text_area(key='settings_import_text').set_value(text)
    app.button(key='btn_import_settings').click().run()
    assert not app.exception


def configure(app):
    app.multiselect(key='sidebar_configs').set_value(['Config_2', 'Config_3', 'Config_10']).run()
    app.multiselect(key='sidebar_objects').set_value([0, 2]).run()
    app.selectbox(key='visual_agg_type').set_value('mean').run()
    app.number_input(key='visual_temp_res').set_value(2.5).run()
    app.number_input(key='visual_start').set_value(1.0).run()
    app.slider(key='anim_speed').set_value(700).run()


def current(app):
    keys = ['sidebar_configs', 'sidebar_objects', 'visual_agg_type', 'visual_temp_res', 'visual_start', 'anim_speed']
    return {k: app.session_state[k] for k in keys if k in app.session_state}


def test_copy_change_import_restores_settings(at):
    configure(at)
    original = current(at)
    payload = json.loads(copied_json(at))            # 1. copied content is valid JSON
    assert payload['format'] == 'samtool-settings' and payload['version'] == 1
    assert payload['settings']['sidebar_configs'] == ['Config_2', 'Config_3', 'Config_10']
    assert payload['settings']['anim_speed'] == 700
    text = copied_json(at)

    at.multiselect(key='sidebar_configs').set_value(['Config_0']).run()      # 2. change settings
    at.selectbox(key='visual_agg_type').set_value('median').run()
    at.slider(key='anim_speed').set_value(200).run()
    assert current(at) != original

    import_text(at, text)                                                     # 3. import the original
    assert current(at) == original
    assert at.multiselect(key='sidebar_configs').value == ['Config_2', 'Config_3', 'Config_10']
    assert at.session_state['shared_selected_configs'] == ['Config_2', 'Config_3', 'Config_10']   # views follow
    assert any('Imported' in s.value for s in at.success)


@pytest.mark.parametrize('text, message', [
    ('not json at all', 'not valid JSON'),
    ('{"settings": {"anim_speed": 300}}', 'not a SAMTool settings text'),
    ('{"format": "samtool-settings", "version": 99, "settings": {"anim_speed": 300}}', 'Unsupported settings version'),
    ('{"format": "samtool-settings", "version": 1}', '"settings" is missing'),
    ('{"format": "samtool-settings", "version": 1, "settings": {"anim_speed": 333}}', 'Unsupported value for "anim_speed"'),
    ('{"format": "samtool-settings", "version": 1, "settings": {"colour_scheme": "x"}}', 'Unknown settings'),
    ('{"format": "samtool-settings", "version": 1, "settings": {"sidebar_configs": ["Config_99"]}}',
     'Configurations not in the loaded dataset: Config_99'),
    ('{"format": "samtool-settings", "version": 1, "settings": {"visual_start": 50}}', 'allowed: 0 to 5'),
])
def test_invalid_or_incompatible_import_keeps_settings(at, text, message):
    configure(at)
    before = current(at)
    import_text(at, text)
    assert current(at) == before
    errors = ' '.join(e.value for e in at.error)
    assert 'current settings are unchanged' in errors and message in errors


def test_partly_invalid_import_changes_nothing(at):
    configure(at)
    before = current(at)
    import_text(at, json.dumps({'format': 'samtool-settings', 'version': 1,
                                'settings': {'visual_agg_type': 'median', 'anim_speed': 12345}}))
    assert current(at) == before          # the valid part is not applied either
