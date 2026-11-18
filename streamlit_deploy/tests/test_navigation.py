"""
Moving between Sequence Analysis, Visual Exploration and PDP on A2_1Hz_dataset.csv (Object 1, all 26 configurations):
kept settings and results, Inspect matching trajectories, Compare these configurations in PDP, custom encoding.
Run from streamlit_deploy/ (PYTHONPATH=.).
"""
import io
import json

import pandas as pd
import pytest

from test_exercise5 import at, a2, CONFIGS, ok, latest   # noqa: F401  (fixtures)
from modules import sequence_analysis as sa

SETTINGS = {'seq_start': 1.0, 'seq_end': 5.0, 'seq_step_threshold': 1.5, 'seq_step_compress': True,
            'seq_step_pattern': 'H L', 'seq_step_mode': 'Token co-occurrence (order ignored)',
            'seq_step_compare': ['Config_3_obj1', 'Config_7_obj1']}


def lh_page(app):
    app.multiselect(key='sidebar_configs').set_value(CONFIGS).run()
    app.selectbox(key='analysis_method').set_value('Sequence Analysis').run()
    app.radio(key='seq_encoding').set_value('Movement steps (L/H)').run()
    ok(app)


def set_non_default(app):
    app.number_input(key='seq_start').set_value(SETTINGS['seq_start']).run()
    app.number_input(key='seq_end').set_value(SETTINGS['seq_end']).run()
    app.number_input(key='seq_step_threshold').set_value(SETTINGS['seq_step_threshold']).run()
    app.checkbox(key='seq_step_compress').set_value(True).run()
    app.text_input(key='seq_step_pattern').set_value(SETTINGS['seq_step_pattern']).run()
    app.radio(key='seq_step_mode').set_value(SETTINGS['seq_step_mode']).run()
    app.multiselect(key='seq_step_compare').set_value(SETTINGS['seq_step_compare']).run()
    ok(app)


def snapshot(app):
    """Widget values, the sequence and pattern exports and the frequency metric."""
    widgets = {
        'seq_start': app.number_input(key='seq_start').value, 'seq_end': app.number_input(key='seq_end').value,
        'seq_step_threshold': app.number_input(key='seq_step_threshold').value,
        'seq_step_compress': app.checkbox(key='seq_step_compress').value,
        'seq_step_pattern': app.text_input(key='seq_step_pattern').value,
        'seq_step_mode': app.radio(key='seq_step_mode').value,
        'seq_step_compare': list(app.multiselect(key='seq_step_compare').value),
        'seq_encoding': app.radio(key='seq_encoding').value,
    }
    return widgets, latest(app, 'lh_sequences')['data'], latest(app, 'lh_pattern_')['data'], \
        {m.label: m.value for m in app.metric}['Relative frequency']


def expected_support(a2):
    _, seqs = sa.step_sequences(sa.step_table(a2, CONFIGS, [1], 1.0, 5.0), threshold=1.5, compress=True)
    return sa.pattern_support(dict(zip(seqs['Trajectory'], seqs['Sequence'])), ['H', 'L'], ordered=False)


def test_settings_and_result_survive_visual_and_pdp(at, a2):
    lh_page(at)
    set_non_default(at)
    before = snapshot(at)
    assert before[0] == {**SETTINGS, 'seq_encoding': 'Movement steps (L/H)'}
    supporting, k, n, _ = expected_support(a2)
    assert before[3] == f"{k}/{n} = {k / n:.3f}"

    for method in ('Visual Exploration', 'PDP Analysis', 'Sequence Analysis'):
        at.selectbox(key='analysis_method').set_value(method).run()
        ok(at)
    assert at.multiselect(key='sidebar_configs').value == CONFIGS and at.multiselect(key='sidebar_objects').value == [1]
    del at.downloads[:]
    at.run()
    assert snapshot(at) == before                       # settings, sequences, pattern result and metric restored
    assert not any('changed' in w.value and 'sequence result' in w.value for w in at.warning)


def test_inspect_highlights_exactly_the_supporting_trajectories(at, a2):
    lh_page(at)
    set_non_default(at)
    before = snapshot(at)
    supporting = expected_support(a2)[0]
    assert 0 < len(supporting) < 26
    at.button(key='lh_inspect').click().run()
    ok(at)
    assert at.selectbox(key='analysis_method').value == 'Visual Exploration'
    assert (at.number_input(key='visual_start').value, at.number_input(key='visual_end').value) == (1.0, 5.0)
    specs = [json.loads(c.proto.spec) for c in at.get('plotly_chart')]
    fig = next(s for s in specs if 'match' in str(s['layout'].get('title', {}).get('text', '')))
    roles = [t.get('meta', {}) for t in fig['data'] if t.get('meta')]
    assert sorted(r['trajectory'] for r in roles if r['role'] == 'supporting') == sorted(supporting)
    others = [r['trajectory'] for r in roles if r['role'] == 'other']
    assert len(others) == 26 - len(supporting) and not set(others) & set(supporting)   # original population kept
    for t in fig['data']:                               # supporting paths cover the window tst 1..5 only
        if t.get('meta', {}).get('role') == 'supporting':
            assert t['meta']['tst'] == [1.0, 2.0, 3.0, 4.0, 5.0]

    # Back to the same sequence result
    del at.downloads[:]
    at.button(key='inspect_back').click().run()
    ok(at)
    assert at.selectbox(key='analysis_method').value == 'Sequence Analysis'
    assert snapshot(at) == before


def test_compare_in_pdp_carries_configurations_and_window(at, a2):
    lh_page(at)
    set_non_default(at)
    supporting = expected_support(a2)[0]
    configs = sorted({t.rsplit('_obj', 1)[0] for t in supporting}, key=lambda c: int(c.split('_')[1]))
    at.button(key='lh_to_pdp').click().run()
    ok(at)
    assert at.selectbox(key='analysis_method').value == 'PDP Analysis'
    assert (at.number_input(key='pdp_start').value, at.number_input(key='pdp_end').value) == (1.0, 5.0)
    info = next(i.value for i in at.info if 'Comparing configurations from Sequence Analysis' in i.value)
    assert ', '.join(configs) in info and 'PDP will use objects 1' in info
    assert at.multiselect(key='sidebar_configs').value == CONFIGS          # shared selection untouched
    at.button(key='pdp_handoff_back').click().run()
    assert at.selectbox(key='analysis_method').value == 'Sequence Analysis'
    assert at.number_input(key='seq_step_threshold').value == 1.5


def test_shared_selection_change_is_flagged(at):
    lh_page(at)
    at.selectbox(key='analysis_method').set_value('Visual Exploration').run()
    at.multiselect(key='sidebar_configs').set_value(CONFIGS[:10]).run()
    at.selectbox(key='analysis_method').set_value('Sequence Analysis').run()
    ok(at)
    assert any('configurations changed' in w.value for w in at.warning)


def test_custom_encoding(at, a2):
    lh_page(at)
    # The 2 m L/H preset of the custom panel reproduces the L/H sequences exactly
    lh = pd.read_csv(io.StringIO(latest(at, 'lh_sequences')['data']))
    custom = pd.read_csv(io.StringIO(latest(at, 'custom_sequences')['data']))
    assert list(custom['Sequence']) == list(lh['Sequence'])
    at.text_input(key='seq_custom_cuts').set_value('1, 3').run()
    at.text_input(key='seq_custom_labels').set_value('S M F').run()
    at.text_input(key='seq_custom_pattern').set_value('S F').run()
    ok(at)
    custom = pd.read_csv(io.StringIO(latest(at, 'custom_sequences_dist_cuts1-3_SMF')['data']))
    steps = sa.step_table(a2, CONFIGS, [1], 0, 5)
    for _, row in custom.iterrows():
        d = steps[steps.Configuration == row.Configuration]['Distance_m']
        assert row.Sequence == ''.join('S' if v < 1 else 'M' if v < 3 else 'F' for v in d)
    m = dict(zip(custom['Trajectory'], custom['Sequence']))
    k, n = sa.pattern_support(m, ['S', 'F'])[1:3]
    assert {x.label: x.value for x in at.metric}['Custom pattern frequency'] == f"{k}/{n} = {k / n:.3f}"
    # Invalid input is explained, the L/H workflow is unaffected
    at.text_input(key='seq_custom_cuts').set_value('3, 1').run()
    assert any('strictly increasing' in e.value for e in at.error)
    assert pd.read_csv(io.StringIO(latest(at, 'lh_sequences')['data']))['Sequence'].tolist() == lh['Sequence'].tolist()


def test_cutoff_parsing():
    assert sa.parse_cutoffs('1, 3.5') == [1.0, 3.5]
    with pytest.raises(ValueError):
        sa.parse_labels('L H', 3)
    assert sa.custom_token(2.0, [2.0], ['L', 'H']) == 'H' and sa.custom_token(1.999, [2.0], ['L', 'H']) == 'L'


def test_zone_settings_survive_switching(at):
    at.multiselect(key='sidebar_configs').set_value(CONFIGS[:8]).run()
    at.selectbox(key='analysis_method').set_value('Sequence Analysis').run()
    at.slider(key='seq_grid_rows').set_value(4).run()
    at.slider(key='seq_grid_cols').set_value(6).run()
    at.checkbox(key='seq_compress').set_value(False).run()
    at.radio(key='seq_bounds_mode').set_value('Explicit bounds').run()
    at.number_input(key='seq_bound_1').set_value(70.0).run()
    ok(at)
    seqs = latest(at, 'sequences_4x6')['data']
    for method in ('Visual Exploration', 'PDP Analysis', 'Sequence Analysis'):
        at.selectbox(key='analysis_method').set_value(method).run()
        ok(at)
        assert not any('default value' in str(getattr(e, 'value', '')) for e in list(at.warning) + list(at.exception))
    assert (at.slider(key='seq_grid_rows').value, at.slider(key='seq_grid_cols').value) == (4, 6)
    assert at.checkbox(key='seq_compress').value is False and at.radio(key='seq_encoding').value == 'Spatial zones'
    assert at.radio(key='seq_bounds_mode').value == 'Explicit bounds' and at.number_input(key='seq_bound_1').value == 70.0
    assert latest(at, 'sequences_4x6')['data'] == seqs
