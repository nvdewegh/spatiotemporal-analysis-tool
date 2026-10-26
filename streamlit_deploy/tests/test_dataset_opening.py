"""
Time ranges keep both ends independently, and every opened dataset starts from one baseline (also after
Reset settings to defaults). Datasets are identified by content, not by file name.

Uses small synthetic CSVs (conID, tstID, poiID, x, y) and, where noted, A2_detailed_dataset.csv (next to
PDP_1HZ_CSV). Run from streamlit_deploy/.
"""
import io
import json
import os
import warnings

import pandas as pd
import pytest
import streamlit
from streamlit.testing.v1 import AppTest

warnings.filterwarnings('ignore')
APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'streamlit_visualization.py')
CSV = os.environ.get('PDP_1HZ_CSV')
DETAILED = os.path.join(os.path.dirname(CSV), 'A2_detailed_dataset.csv') if CSV else None


def write_csv(path, times, configs=3, objects=7):
    rows = [(c, t, o, 10.0 * o + c + 0.1 * i, 2.0 * o - i)
            for c in range(configs) for i, t in enumerate(times) for o in range(objects)]
    pd.DataFrame(rows, columns=['conID', 'tstID', 'poiID', 'x', 'y']).to_csv(path, index=False)
    return str(path)


@pytest.fixture
def app(monkeypatch):
    current = {'path': None}

    class Upload(io.BytesIO):
        pass

    def uploader(label, *a, **k):
        if not label.startswith('Upload CSV') or current['path'] is None:
            return None
        upload = Upload(open(current['path'], 'rb').read())
        upload.name = current.get('name') or os.path.basename(current['path'])
        return [upload]

    monkeypatch.setattr(streamlit, 'file_uploader', uploader)
    at = AppTest.from_file(APP, default_timeout=300)
    at.session_state['password_correct'] = True
    at.current = current
    at.run()
    return at


def open_file(at, path, name=None):
    at.current.update(path=path, name=name)
    at.run()
    assert not at.exception, [e.message for e in at.exception]


def baseline(at, t_min, t_max):
    s = at.session_state
    data = s['data']
    objects = sorted(data.obj.unique())
    configs = sorted(data.config_source.unique(), key=lambda c: int(c.split('_')[1]))
    return (s['analysis_method'] == 'Visual Exploration' and s['visual_agg_type'] == 'none'
            and s['show_court_background'] is False and s['court_type'] == 'No selection'
            and list(s['sidebar_configs']) == configs and list(s['sidebar_objects']) == objects[:5]
            and all(s[k] == t_min for k in ('visual_start', 'pdp_start', 'clustering_start', 'extra_start'))
            and all(s[k] == t_max for k in ('visual_end', 'pdp_end', 'clustering_end', 'extra_end'))
            and s['pdp_variants'] == ['Fundamental'] and s['pdp_use_external'] is False
            and s['pdp_window'] == min(3, data.tst.nunique()))


def to_pdp(at):
    at.selectbox(key='analysis_method').set_value('PDP Analysis').run()
    assert not at.exception, [e.message for e in at.exception]


@pytest.mark.skipif(not DETAILED or not os.path.exists(DETAILED), reason='A2_detailed_dataset.csv not found')
def test_end_kept_when_start_changes_detailed(app):
    open_file(app, DETAILED)
    assert baseline(app, 0, 136)
    to_pdp(app)
    app.number_input(key='pdp_start').set_value(0.0).run()
    app.number_input(key='pdp_end').set_value(122.0).run()
    app.number_input(key='pdp_start').set_value(2.0).run()               # the reported reproduction
    assert (app.number_input(key='pdp_start').value, app.number_input(key='pdp_end').value) == (2.0, 122.0)
    app.number_input(key='pdp_end').set_value(100.0).run()               # End alone
    assert (app.number_input(key='pdp_start').value, app.number_input(key='pdp_end').value) == (2.0, 100.0)
    app.checkbox(key='pdp_use_external').check().run()                   # ordinary reruns
    app.selectbox(key='analysis_method').set_value('Visual Exploration').run()
    to_pdp(app)
    assert (app.number_input(key='pdp_start').value, app.number_input(key='pdp_end').value) == (2.0, 100.0)


def test_start_after_end_blocks_computation(app, tmp_path):
    open_file(app, write_csv(tmp_path / 'a.csv', range(10)))
    to_pdp(app)
    app.number_input(key='pdp_end').set_value(4.0).run()
    app.number_input(key='pdp_start').set_value(6.0).run()
    assert (app.number_input(key='pdp_start').value, app.number_input(key='pdp_end').value) == (6.0, 4.0)
    assert any('Start time (6) is after End time (4)' in str(e.value) for e in app.error)
    assert app.button(key='compute_pdp').disabled
    app.number_input(key='pdp_start').set_value(1.0).run()
    assert not app.button(key='compute_pdp').disabled


@pytest.mark.parametrize('times, t_min, t_max', [
    (list(range(-5, 5)), -5, 4),                              # negative start
    ([100.5, 101.0, 103.25, 110.0, 111.5], 100.5, 111.5),     # decimals with gaps, non-zero start
    ([42.0], 42.0, 42.0),                                     # one distinct timestamp
])
def test_baseline_uses_actual_timestamps(app, tmp_path, times, t_min, t_max):
    open_file(app, write_csv(tmp_path / 'edge.csv', times))
    assert baseline(app, t_min, t_max)
    to_pdp(app)
    assert app.number_input(key='pdp_start').value == t_min and app.number_input(key='pdp_end').value == t_max
    assert not app.button(key='compute_pdp').disabled


def test_other_dataset_and_same_name_with_changed_contents(app, tmp_path):
    first = write_csv(tmp_path / 'first.csv', range(20))
    open_file(app, first, name='data.csv')
    to_pdp(app)
    app.number_input(key='pdp_end').set_value(12.0).run()
    app.slider(key='pdp_window').set_value(8).run()
    app.multiselect(key='sidebar_configs').set_value(['Config_1']).run()
    app.run()                                                            # unchanged data keeps choices
    assert app.session_state['pdp_end'] == 12.0 and app.session_state['pdp_window'] == 8
    changed = write_csv(tmp_path / 'second.csv', [t + 50 for t in range(4)])
    open_file(app, changed, name='data.csv')                             # same name, other contents
    assert baseline(app, 50, 53)
    assert 'pdp_results_all' not in app.session_state


def test_reset_to_defaults_and_import_after_loading(app, tmp_path):
    path = write_csv(tmp_path / 'r.csv', range(8))
    open_file(app, path)
    app.selectbox(key='analysis_method').set_value('PDP Analysis').run()
    app.number_input(key='pdp_start').set_value(2.0).run()
    app.multiselect(key='sidebar_objects').set_value([0]).run()
    app.button(key='btn_reset_settings').click().run()
    assert baseline(app, 0, 7)

    settings = {'format': 'samtool-settings', 'version': 1,
                'settings': {'analysis_method': 'PDP Analysis', 'sidebar_objects': [1, 2],
                             'visual_start': 3.0, 'visual_end': 6.0}}
    app.text_area(key='settings_import_text').input(json.dumps(settings)).run()
    app.button(key='btn_import_settings').click().run()
    s = app.session_state
    assert s['analysis_method'] == 'PDP Analysis' and list(s['sidebar_objects']) == [1, 2]
    assert (s['visual_start'], s['visual_end']) == (3.0, 6.0)          # import wins over the baseline
    app.run()
    assert s['analysis_method'] == 'PDP Analysis'                       # not re-baselined on rerun


def test_window_kept_or_adjusted_when_interval_shrinks(app, tmp_path):
    open_file(app, write_csv(tmp_path / 'w.csv', range(12)))
    to_pdp(app)
    app.slider(key='pdp_window').set_value(5).run()
    app.number_input(key='pdp_end').set_value(8.0).run()               # 9 steps: 5 still valid
    assert app.slider(key='pdp_window').value == 5
    app.number_input(key='pdp_end').set_value(2.0).run()               # 3 steps: adjusted visibly
    assert app.slider(key='pdp_window').value == 3
    assert any('Window length adjusted from 5 to 3' in str(i.value) for i in app.info)


def test_restored_saved_settings_do_not_override_baseline(app, tmp_path):
    from modules import browser_storage
    path = write_csv(tmp_path / 's.csv', range(6))
    browser_storage._state()._bs_pending_settings = {'analysis_method': 'PDP Analysis', 'show_court_background': True}
    app.session_state['_bs_pending_settings'] = {'analysis_method': 'PDP Analysis', 'show_court_background': True}
    open_file(app, path)
    assert baseline(app, 0, 5)
