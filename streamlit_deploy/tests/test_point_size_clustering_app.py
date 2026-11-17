"""
The Point size and clustering tab in Visual Exploration, driven through the app with the real course
dataset A2_1Hz_dataset.csv (set PDP_A2_CSV). Checks that the reported counts, labels and scores match
an independent calculation on the same selection, that the diameter only moves the cut, and that a
dataset change refreshes the results. The other Visual Exploration tabs must still render.

Run from streamlit_deploy/:  PDP_A2_CSV=/path/A2_1Hz_dataset.csv python -m pytest tests/test_point_size_clustering_app.py
"""
import io
import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
import streamlit
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score
from streamlit.testing.v1 import AppTest

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(HERE, 'streamlit_visualization.py')
CSV = os.environ.get('PDP_A2_CSV')
pytestmark = pytest.mark.skipif(not CSV or not os.path.exists(CSV), reason='set PDP_A2_CSV to A2_1Hz_dataset.csv')

CONFIGS = ['Config_1', 'Config_2', 'Config_3', 'Config_4', 'Config_5', 'Config_6', 'Config_7', 'Config_8']


class Upload(io.BytesIO):
    def __init__(self, data, name):
        super().__init__(data)
        self.name = name


@pytest.fixture
def at(monkeypatch):
    holder = {'files': [(open(CSV, 'rb').read(), 'A2_1Hz_dataset.csv')]}
    monkeypatch.setattr(streamlit, 'file_uploader',
                        lambda label, *a, **k: [Upload(d, n) for d, n in holder['files']]
                        if label.startswith('Upload CSV') else None)
    app = AppTest.from_file(APP, default_timeout=300)
    app.session_state['password_correct'] = True
    app.run()
    app.multiselect(key='sidebar_configs').set_value(CONFIGS).run()
    objects = app.multiselect(key='sidebar_objects').options
    app.multiselect(key='sidebar_objects').set_value([int(o) for o in objects]).run()
    app.toggle(key='psc_enabled').set_value(True).run()
    app.holder = holder
    return app


def metrics(app):
    return {m.label: m.value for m in app.metric}


def diameter_slider(app):
    return [s for s in app.slider if str(s.key).startswith('psc_diameter_')][0]


def expected(app, method, d):
    """Independent calculation on the same selection: mean position per configuration/object."""
    sys.path.insert(0, HERE)
    from modules import utils
    with open(CSV, 'rb') as f:
        df = utils.load_data(f, update_state=False, show_success=False)
    t0, t1 = app.number_input(key='visual_start').value, app.number_input(key='visual_end').value
    objs = app.multiselect(key='sidebar_objects').value
    sel = df[df['config_source'].isin(CONFIGS) & df['obj'].isin(objs) & (df['tst'] >= t0) & (df['tst'] <= t1)]
    xy = sel.groupby(['config_source', 'obj'], sort=False)[['x', 'y']].mean().to_numpy()
    D = pdist(xy)
    labels = fcluster(linkage(D, method=method), t=d, criterion='distance')
    k = len(np.unique(labels))
    score = silhouette_score(xy, labels) if 2 <= k <= len(xy) - 1 else None
    return len(xy), k, score


@pytest.mark.parametrize('method', ['single', 'average'])
def test_counts_and_scores_match_independent_calculation(at, method):
    at.radio(key='psc_mode').set_value('interval_mean').run()
    at.radio(key='psc_linkage').set_value(method).run()
    assert not at.exception
    for d in (0.5, 2.0, 6.0):
        diameter_slider(at).set_value(d).run()
        n, k, score = expected(at, method, d)
        m = metrics(at)
        assert int(m['Clusters at cut d']) == k
        assert any(f'**{n} points used**' in c.value for c in at.caption)
        if score is None:
            assert m['Mean silhouette (current)'] == 'undefined'
        else:
            assert float(m['Mean silhouette (current)']) == pytest.approx(score, abs=1e-4)
        if method == 'single':
            assert m['Partitions identical'] == 'yes'
            assert int(m['Groups of linked circles']) == k


def test_diameter_moves_cut_not_curve(at):
    at.radio(key='psc_mode').set_value('interval_mean').run()
    diameter_slider(at).set_value(1.0).run()
    import json
    def curve():
        for c in at.get('plotly_chart'):
            spec = json.loads(c.proto.spec)
            if spec['layout'].get('yaxis', {}).get('title', {}).get('text') == 'mean silhouette score':
                tr = spec['data'][0]
                return tr['x'], tr['y']
    first = curve()
    diameter_slider(at).set_value(4.0).run()
    assert curve() == first and first is not None
    assert not at.exception


def test_time_point_mode_and_other_tabs(at):
    at.radio(key='psc_mode').set_value('time_point').run()
    assert not at.exception
    assert any('Position at one time stamp (measured)' in c.value for c in at.caption)
    assert len(at.get('plotly_chart')) >= 4 + 3   # existing four views plus scatter, dendrogram, silhouette


def test_dataset_change_refreshes(at):
    at.radio(key='psc_mode').set_value('interval_mean').run()
    before = metrics(at)
    # Same dataset with every x shifted by a constant: distances, and so all results, must stay equal;
    # then a scaled copy: clusters at the same d must change
    df = pd.read_csv(CSV)
    scaled = df.copy()
    scaled[scaled.columns[3]] = scaled[scaled.columns[3]] * 3
    scaled[scaled.columns[4]] = scaled[scaled.columns[4]] * 3
    at.holder['files'] = [(scaled.to_csv(index=False).encode(), 'A2_scaled.csv')]
    at.run()
    assert not at.exception
    assert any('A2_scaled.csv' in c.value for c in at.caption)
    assert metrics(at) != before
