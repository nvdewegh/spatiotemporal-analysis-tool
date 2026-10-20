"""
The Animated Trajectories view renders its chart, or shows a clear message instead of a
blank plot (empty or invalid time range).

Run from streamlit_deploy/:  python -m pytest tests
"""
import io
import os
import warnings

import pytest
import streamlit
from streamlit.testing.v1 import AppTest

warnings.filterwarnings('ignore')
APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'streamlit_visualization.py')


@pytest.fixture
def at(monkeypatch):
    rows = ''.join(f'{c},{t},{o},{c + t + o},{t * o}\n' for c in range(4) for t in range(6) for o in range(2))
    data = ('conID,tstID,poiID,x,y\n' + rows).encode()

    class Upload(io.BytesIO):
        name = 'test.csv'

    monkeypatch.setattr(streamlit, 'file_uploader',
                        lambda label, *a, **k: [Upload(data)] if label.startswith('Upload CSV') else None)
    app = AppTest.from_file(APP, default_timeout=180)
    app.session_state['password_correct'] = True
    app.run()
    app.multiselect(key='sidebar_configs').set_value(['Config_0', 'Config_1']).run()
    app.multiselect(key='sidebar_objects').set_value([0, 1]).run()
    return app


def charts(app):
    return app.get('plotly_chart')


def messages(app):
    return ' '.join(m.value for m in list(app.warning) + list(app.error))


def test_animation_chart_renders(at):
    assert not at.exception
    assert len(charts(at)) >= 4          # static, animated, time point and average views
    assert 'animation' not in messages(at).lower()


def test_changed_settings_still_render(at):
    before = len(charts(at))
    at.checkbox(key='anim_smooth').check().run()
    at.slider(key='anim_speed').set_value(1000).run()
    assert not at.exception and len(charts(at)) == before


def test_inverted_time_range_shows_message(at):
    at.number_input(key='visual_start').set_value(4.0).run()
    at.number_input(key='visual_end').set_value(1.0).run()
    assert not at.exception
    assert 'Start time is after end time' in messages(at)
