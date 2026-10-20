"""
Regression tests for the sidebar "Select Range" buttons (configurations and objects).

The range must replace the current selection (not be added to it), include both ends,
work for single-item and reversed ranges, and configurations must follow numeric order.
The app runs in Streamlit's test runner; the file upload is simulated and the password
check is marked as passed.

Run from streamlit_deploy/:  python -m pytest tests
"""
import io
import os
import random
import warnings

import pytest
import streamlit
from streamlit.testing.v1 import AppTest

warnings.filterwarnings('ignore')
APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'streamlit_visualization.py')


def csv_bytes(n_configs=26, n_objects=3, shuffle=False):
    configs = list(range(n_configs))
    if shuffle:
        random.Random(3).shuffle(configs)   # configurations in scrambled file order
    rows = ''.join(f'{c},{t},{o},{c + t + o},{t * o}\n'
                   for c in configs for t in range(6) for o in range(n_objects))
    return ('conID,tstID,poiID,x,y\n' + rows).encode()


@pytest.fixture
def app(monkeypatch):
    def start(**kwargs):
        data = csv_bytes(**kwargs)

        class Upload(io.BytesIO):
            name = 'test.csv'

        monkeypatch.setattr(streamlit, 'file_uploader', lambda *a, **k: [Upload(data)])
        at = AppTest.from_file(APP, default_timeout=180)
        at.session_state['password_correct'] = True
        at.run()
        return at
    return start


def select_range(at, kind, start, end):
    at.selectbox(key=f'range_start_{kind}').set_value(start)
    at.selectbox(key=f'range_end_{kind}').set_value(end)
    button = {'obj': 'btn_select_range_objects', 'config': 'btn_select_range_configs'}[kind]
    at.button(key=button).click().run()
    assert not at.exception


@pytest.mark.parametrize('start, end, expected', [(0, 0, [0]), (1, 2, [1, 2]), (2, 0, [0, 1, 2])])
def test_object_range_replaces_selection(app, start, end, expected):
    at = app()
    at.multiselect(key='sidebar_objects').set_value([1, 2]).run()
    select_range(at, 'obj', start, end)
    assert at.multiselect(key='sidebar_objects').value == expected
    assert list(at.session_state['shared_selected_objects']) == expected


@pytest.mark.parametrize('shuffle', [False, True])
@pytest.mark.parametrize('start, end, expected', [
    ('Config_1', 'Config_10', [f'Config_{i}' for i in range(1, 11)]),
    ('Config_15', 'Config_8', [f'Config_{i}' for i in range(8, 16)]),
    ('Config_4', 'Config_4', ['Config_4']),
])
def test_config_range_replaces_selection_in_numeric_order(app, shuffle, start, end, expected):
    at = app(shuffle=shuffle)
    at.multiselect(key='sidebar_configs').set_value(['Config_0', 'Config_20']).run()
    select_range(at, 'config', start, end)
    assert at.multiselect(key='sidebar_configs').value == expected
    assert at.session_state['shared_selected_configs'] == expected
