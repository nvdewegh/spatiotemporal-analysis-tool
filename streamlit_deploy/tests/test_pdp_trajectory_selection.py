"""
Trajectory Comparison (Clustering & Projection): quick-selection buttons and the configuration field share
one selection. A quick choice updates the field; a later manual choice drives the chart and the pairwise
table; ordinary reruns (e.g. another cluster count) keep it.

B1: A2_1Hz_dataset.csv, configurations 1, 2, 4, 11, objects 0 and 1, indices 0-5, window 6, Fundamental.
Run from streamlit_deploy/ with PDP_1HZ_CSV pointing to A2_1Hz_dataset.csv.
"""
import json
import os

import pytest

from test_pdp_advanced_sections import app, compute, select  # noqa: F401  (fixture)

CSV = os.environ.get('PDP_1HZ_CSV')
pytestmark = pytest.mark.skipif(not CSV or not os.path.exists(CSV), reason='set PDP_1HZ_CSV to A2_1Hz_dataset.csv')

FIELD = 'Select configurations to visualize'


def field(at):
    # The distance-matrix inspector uses the same label; the trajectory field comes last on the page
    matches = [m for m in at.multiselect if m.label == FIELD]
    return next((m for m in matches if m.key == 'pdp_viz_configs'), matches[-1])


def button(at, label):
    return [b for b in at.button if b.label == label][0]


def chart_configs(at):
    """Configurations drawn in the trajectory-comparison chart (legend groups)."""
    for chart in at.get('plotly_chart'):
        spec = json.loads(chart.proto.spec)
        title = spec['layout'].get('title', {})
        if 'Trajectory Comparison' in str(title.get('text', '') if isinstance(title, dict) else title):
            groups = {t.get('legendgroup') or t.get('name') for t in spec['data'] if t.get('legendgroup')}
            return {g.split(' ')[0] if g else g for g in groups}, spec
    raise AssertionError('trajectory comparison chart not found')


def pair_table(at):
    for f in at.dataframe:
        v = f.value
        if 'Raw PDP distance' in v.columns and 'Config A' in v.columns and len(v) >= 1:
            last = v
    return {frozenset((r['Config A'], r['Config B'])): r['Raw PDP distance'] for _, r in last.iterrows()}


def shows(at, configs):
    names, spec = chart_configs(at)
    title = spec['layout']['title']['text']
    return (sorted(field(at).value) == sorted(configs)
            and all(any(c in n for n in names) for c in configs)
            and all(f'{c} vs' in title or f'vs {c}' in title for c in configs) if len(configs) == 2 else True)


def test_quick_then_manual_then_rerun(app):
    compute(app)
    assert sorted(field(app).value) == ['Config_1', 'Config_2']               # default: first two
    button(app, 'Top Similar').click().run()
    assert not app.exception
    assert sorted(field(app).value) == ['Config_1', 'Config_4'] and shows(app, ['Config_1', 'Config_4'])
    assert '0' in pair_table(app)[frozenset(('Config_1', 'Config_4'))]

    field(app).set_value(['Config_1', 'Config_11']).run()                   # manual choice wins
    assert shows(app, ['Config_1', 'Config_11'])
    assert pair_table(app)[frozenset(('Config_1', 'Config_11'))] == '120'

    app.slider(key='pdp_n_clusters').set_value(3).run()                     # ordinary rerun keeps it
    assert sorted(field(app).value) == ['Config_1', 'Config_11'] and shows(app, ['Config_1', 'Config_11'])


def test_other_quick_buttons_then_manual(app):
    compute(app)
    for label in ('Top Dissimilar', 'Cluster Centroids', 'Random Sample'):
        button(app, label).click().run()
        assert not app.exception, label
        chosen = sorted(field(app).value)
        assert chosen and set(chosen) <= {'Config_1', 'Config_2', 'Config_4', 'Config_11'}, label
        names, _ = chart_configs(app)
        assert all(any(c in n for n in names) for c in chosen), label
        field(app).set_value(['Config_2', 'Config_4']).run()
        assert shows(app, ['Config_2', 'Config_4']), label


def test_selection_survives_outdated_results_and_drops_unavailable_configs(app):
    compute(app)
    field(app).set_value(['Config_4', 'Config_11']).run()
    app.slider(key='pdp_window').set_value(5).run()                         # outdated: section not drawn
    app.slider(key='pdp_window').set_value(6).run()
    compute(app)
    assert sorted(field(app).value) == ['Config_11', 'Config_4']
    select(app, ['Config_1', 'Config_2', 'Config_4'])                       # Config_11 no longer available
    compute(app)
    assert not app.exception
    assert field(app).value == ['Config_4']
