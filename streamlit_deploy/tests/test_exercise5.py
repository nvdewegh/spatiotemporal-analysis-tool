"""
Exercise 5 (associations, L/H sequences, outliers) on A2_1Hz_dataset.csv, Object 1 only.

Reference values from the instructor guide (Exercise5Preparation_v02/Verification):
- Feature bins, support 0.10, confidence 0.60: 111 rules with three bins, 101 with two bins.
- IQR on total_distance: factor 1.5 flags Config_2, 11, 16, 17, 18; factor 3.0 flags none.
- L/H sequences (threshold 2 m, no compression) of eight configurations and their pattern frequencies.

Run from streamlit_deploy/ (PYTHONPATH=.). Uses PDP_1HZ_CSV or the course location.
"""
import io
import json
import os
import warnings

import numpy as np
import pandas as pd
import pytest
import streamlit
from streamlit.testing.v1 import AppTest

from modules import association_rules as ar, outlier_detection as od, sequence_analysis as sa

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(HERE, 'streamlit_visualization.py')
CSV_1HZ = os.environ.get('PDP_1HZ_CSV') or os.path.join(
    HERE, '..', '..', '..', '04Exercises', 'AJ2627', 'ExercisesForStudents', 'ForStudents', 'Data', 'Data',
    'CSV_PDP input', 'A2_1Hz_dataset.csv')
CONFIGS = [f'Config_{i}' for i in range(26)]
EIGHT = ['Config_0', 'Config_1', 'Config_2', 'Config_5', 'Config_15', 'Config_17', 'Config_18', 'Config_22']
EXPECTED_LH = {'Config_0': 'LLLLH', 'Config_1': 'LLHHH', 'Config_2': 'HHHHH', 'Config_5': 'LLLHH',
               'Config_15': 'HHLHH', 'Config_17': 'LLLLL', 'Config_18': 'LLLLL', 'Config_22': 'HHLLL'}
OUTDATED = 'Inputs changed. Recompute to update results.'
needs_data = pytest.mark.skipif(not os.path.exists(CSV_1HZ), reason='A2_1Hz_dataset.csv not available')


@pytest.fixture(scope='module')
def a2():
    r = pd.read_csv(CSV_1HZ)
    return pd.DataFrame({'config_source': 'Config_' + r.conID.astype(str), 'obj': r.poiID, 'tst': r.tstID,
                         'x': r.x, 'y': r.y})


# --------------------------------------------------------------------------- units

def test_two_metre_step_is_h():
    below = np.nextafter(2.0, 0)                                          # largest double below 2
    df = pd.DataFrame({'config_source': ['C'] * 4 + ['D'] * 2, 'obj': 1, 'tst': [0, 1, 2, 3, 0, 1],
                       'x': [0.0, 2.0, 2.0, 2.0, 0.0, below], 'y': [0.0, 0.0, 2.0, 3.5, 0.0, 0.0]})
    steps, seqs = sa.step_sequences(sa.step_table(df, ['C', 'D'], [1], 0, 3))
    assert list(steps['Distance_m']) == [2.0, 2.0, 1.5, below]         # exact binary values
    assert list(seqs['Sequence']) == ['HHL', 'L']                      # exactly 2 m is H, just below is L
    assert sa.step_sequences(sa.step_table(df, ['C'], [1], 0, 3), compress=True)[1]['Sequence'].iloc[0] == 'HL'


def test_unsorted_input_is_sorted_by_time():
    df = pd.DataFrame({'config_source': 'C', 'obj': 1, 'tst': [2, 0, 1], 'x': [5.0, 0.0, 1.0], 'y': 0.0})
    steps, seqs = sa.step_sequences(sa.step_table(df, ['C'], [1], 0, 2))
    assert list(steps['Distance_m']) == [1.0, 4.0] and seqs['Sequence'].iloc[0] == 'LH'


def test_ordered_patterns_with_gaps_count_each_trajectory_once():
    seqs = {'a': 'LHLH', 'b': 'HHL', 'c': 'LLL'}
    assert sa.pattern_support(seqs, ['L', 'H'], True)[1:] == (1, 3, 1 / 3)
    assert sa.pattern_support(seqs, ['H', 'L'], True)[0] == ['a', 'b']
    assert sa.pattern_support(seqs, ['L', 'H'], False)[1] == 2
    assert sa.parse_pattern('L → h, x') == ['L', 'H', 'X']


@needs_data
def test_reference_sequences_and_patterns(a2):
    _, seqs = sa.step_sequences(sa.step_table(a2, EIGHT, [1], 0, 5))
    assert dict(zip(seqs['Configuration'], seqs['Sequence'])) == EXPECTED_LH
    assert (seqs['Positions'] == 6).all() and (seqs['Steps'] == 5).all()
    m = dict(zip(seqs['Trajectory'], seqs['Sequence']))
    assert sa.pattern_support(m, ['L', 'H'])[0] == [f'{c}_obj1' for c in ('Config_0', 'Config_1', 'Config_5', 'Config_15')]
    assert sa.pattern_support(m, ['H', 'L'])[0] == ['Config_15_obj1', 'Config_22_obj1']
    assert sa.pattern_support(m, ['L', 'H'], ordered=False)[1:3] == (5, 8)


def features(a2):
    rows = []
    for c in CONFIGS:
        g = a2[(a2.config_source == c) & (a2.obj == 1)].sort_values('tst')
        d = np.hypot(np.diff(g.x), np.diff(g.y))
        rows.append({'total_distance': d.sum(), 'duration': g.tst.max() - g.tst.min(),
                     'avg_speed': np.mean(d / np.diff(g.tst)), 'x_range': g.x.max() - g.x.min(),
                     'y_range': g.y.max() - g.y.min()})
    return pd.DataFrame(rows)


@needs_data
@pytest.mark.parametrize('bins, expected', [(3, 111), (2, 101)])
def test_rule_counts_match_support_and_confidence(a2, bins, expected):
    tx, _, _ = ar.prepare_feature_transactions(features(a2), bins)
    rules = ar.add_rule_counts(ar.compute_association_rules(tx, 0.1, 0.6)[0], tx)
    assert len(rules) == expected and (rules['n_transactions'] == 26).all()
    np.testing.assert_allclose(rules['n_joint'] / 26, rules['support'], rtol=0, atol=1e-15)
    np.testing.assert_allclose(rules['n_joint'] / rules['n_antecedent'], rules['confidence'], rtol=0, atol=1e-15)
    np.testing.assert_allclose(rules['consequent_prevalence'], rules['n_consequent'] / 26)
    for col in ('n_antecedent', 'n_consequent', 'n_joint'):
        assert rules[col].dtype.kind == 'i'


def test_supermarket_counts():
    tx = [['Milk', 'Diaper', 'Beer'], ['Milk', 'Diaper', 'Cola'], ['Milk', 'Beer'], ['Diaper', 'Beer', 'Cola'],
          ['Milk', 'Diaper', 'Beer', 'Cola']]
    rules = ar.add_rule_counts(ar.compute_association_rules(tx, 0.2, 0.1)[0], tx)
    r = rules[rules['rule'] == 'Diaper, Milk → Beer'].iloc[0]
    assert (r['n_joint'], r['n_antecedent'], r['n_consequent']) == (2, 3, 4)
    assert r['support'] == pytest.approx(2 / 5) and r['confidence'] == pytest.approx(2 / 3)
    src = open(os.path.join(HERE, 'modules', 'association_rules.py'), encoding='utf-8').read()
    assert '2/5 = 0.4' in src and '2/3' in src and '3/5 = 0.6' not in src


@needs_data
def test_iqr_reference(a2):
    f = pd.DataFrame([od.extract_trajectory_features(a2, c, 1) for c in CONFIGS])
    for factor, flagged in ((1.5, ['Config_2', 'Config_11', 'Config_16', 'Config_17', 'Config_18']), (3.0, [])):
        b = od.iqr_bounds(f, ['total_distance'], factor).loc['total_distance']
        assert (b['Q1'], b['Q3']) == (10.9407518363118, 14.463867020058307)
        assert b['Lower_Bound'] == b['Q1'] - factor * b['IQR'] and b['Upper_Bound'] == b['Q3'] + factor * b['IQR']
        out = od.detect_outliers_iqr(f, ['total_distance'], factor)
        assert list(out[out.is_outlier_iqr].config) == flagged


# --------------------------------------------------------------------------- app route

@pytest.fixture
def at(monkeypatch):
    if not os.path.exists(CSV_1HZ):
        pytest.skip('A2_1Hz_dataset.csv not available')
    payload = open(CSV_1HZ, 'rb').read()
    downloads = []

    class Upload(io.BytesIO):
        name = 'A2_1Hz_dataset.csv'

    monkeypatch.setattr(streamlit, 'file_uploader',
                        lambda label, *a, **k: [Upload(payload)] if label.startswith('Upload CSV') else None)

    def download_button(label, data=None, file_name=None, *a, **k):
        downloads.append({'label': label, 'data': data if isinstance(data, str) else data.decode(),
                          'file_name': file_name})
        return False

    monkeypatch.setattr(streamlit, 'download_button', download_button)
    # Module code calls st.download_button through its own "st" reference: patch it there as well
    from streamlit.delta_generator import DeltaGenerator
    monkeypatch.setattr(DeltaGenerator, 'download_button',
                        lambda self, label, data=None, file_name=None, *a, **k: download_button(label, data, file_name))
    app = AppTest.from_file(APP, default_timeout=300)
    app.session_state['password_correct'] = True
    app.downloads = downloads
    app.run()
    app.multiselect(key='sidebar_objects').set_value([1]).run()
    return app


def ok(app):
    assert not app.exception, [e.message for e in app.exception]


def latest(app, prefix):
    hits = [d for d in app.downloads if str(d['file_name']).startswith(prefix)]
    return hits[-1] if hits else None


def mine(app):
    del app.downloads[:]
    next(b for b in app.button if b.label == 'Mine Association Rules').click().run()
    ok(app)


def test_associations_route(at):
    at.multiselect(key='sidebar_configs').set_value(CONFIGS).run()
    at.selectbox(key='analysis_method').set_value('Association Rules').run()
    next(r for r in at.radio if r.label == 'Analysis Type:').set_value('Feature Bins').run()
    at.select_slider(key='ar_n_bins').set_value(3).run()
    at.slider(key='ar_min_support').set_value(0.1).run()
    at.slider(key='ar_min_confidence').set_value(0.6).run()
    ok(at)
    mine(at)
    assert any('111 rules' in c.value for c in at.caption)
    full = pd.read_csv(io.StringIO(latest(at, 'association_rules_all')['data']))
    assert len(full) == 111 and {'n_antecedent', 'n_consequent', 'n_joint', 'consequent_prevalence'} <= set(full.columns)
    np.testing.assert_allclose(full['support'], full['n_joint'] / 26, rtol=0, atol=1e-15)   # full precision
    shown = pd.read_csv(io.StringIO(latest(at, 'association_rules_displayed')['data']))
    assert len(shown) == 50

    # Find specific rules: the displayed export follows the search, the full export does not
    at.text_input(key='ar_find_text').set_value('total_distance_high').run()
    ok(at)
    shown = pd.read_csv(io.StringIO(latest(at, 'association_rules_displayed')['data']))
    assert 0 < len(shown) < 111 and shown['rule'].str.contains('total_distance_high').all()
    assert len(pd.read_csv(io.StringIO(latest(at, 'association_rules_all')['data']))) == 111

    # A changed input hides the earlier result and its downloads until recomputed
    del at.downloads[:]
    at.select_slider(key='ar_n_bins').set_value(2).run()
    ok(at)
    assert any(OUTDATED in w.value and 'number of bins' in w.value for w in at.warning)
    assert latest(at, 'association_rules') is None
    mine(at)
    assert any('101 rules' in c.value for c in at.caption)
    assert len(pd.read_csv(io.StringIO(latest(at, 'association_rules_all')['data']))) == 101
    at.slider(key='ar_min_confidence').set_value(0.7).run()
    assert any(OUTDATED in w.value and 'minimum confidence' in w.value for w in at.warning)
    at.multiselect(key='sidebar_configs').set_value(CONFIGS[:20]).run()
    assert any(OUTDATED in w.value and 'configurations' in w.value for w in at.warning)


def test_sequence_page_opens_and_lh_route(at):
    at.multiselect(key='sidebar_configs').set_value(EIGHT).run()
    at.selectbox(key='analysis_method').set_value('Sequence Analysis').run()
    ok(at)                                      # zone-based page including the alignment rows
    assert any('Pairwise Sequence Alignment' in s.value for s in at.subheader)
    assert any('Match | ' in c.value for c in at.caption)
    at.radio(key='seq_encoding').set_value('Movement steps (L/H)').run()
    at.number_input(key='seq_start').set_value(0.0).run()
    at.number_input(key='seq_end').set_value(5.0).run()
    ok(at)
    seqs = pd.read_csv(io.StringIO(latest(at, 'lh_sequences')['data']))
    assert dict(zip(seqs['Configuration'], seqs['Sequence'])) == EXPECTED_LH
    assert not seqs['Compressed'].any() and (seqs['Threshold_m'] == 2).all()
    steps = pd.read_csv(io.StringIO(latest(at, 'lh_step_distances')['data']), float_precision='round_trip')
    assert len(steps) == 40 and set(steps['Token']) == {'L', 'H'}
    assert steps.loc[steps.Configuration == 'Config_1', 'Distance_m'].iloc[0] == 0.7960343745198054
    standard = pd.read_csv(io.StringIO(latest(at, 'lh_standard_patterns')['data']))
    assert list(standard['Fraction']) == ['4/8', '2/8', '5/8']
    metrics = {m.label: m.value for m in at.metric}
    assert metrics['Relative frequency'] == '4/8 = 0.500'
    at.text_input(key='seq_step_pattern').set_value('H L').run()
    assert {m.label: m.value for m in at.metric}['Relative frequency'] == '2/8 = 0.250'
    pattern = pd.read_csv(io.StringIO(latest(at, 'lh_pattern_HL_ordered')['data']))
    assert sorted(pattern.loc[pattern.Supports_Pattern, 'Configuration']) == ['Config_15', 'Config_22']
    # Changed settings are reflected at once (nothing stale): threshold 1 m and compression
    at.number_input(key='seq_step_threshold').set_value(1.0).run()
    at.checkbox(key='seq_step_compress').set_value(True).run()
    ok(at)
    seqs = pd.read_csv(io.StringIO(latest(at, 'lh_sequences')['data']))
    assert seqs['Compressed'].all() and (seqs['Threshold_m'] == 1).all()
    assert seqs.loc[seqs.Configuration == 'Config_18', 'Sequence'].iloc[0] == 'L'
    assert latest(at, 'lh_sequences')['file_name'].startswith('lh_sequences_thr1m_compressed')


def test_outlier_route(at):
    at.multiselect(key='sidebar_configs').set_value(CONFIGS).run()
    at.selectbox(key='analysis_method').set_value('Outlier Detection').run()
    ok(at)
    next(m for m in at.multiselect if m.label.startswith('Select features')).set_value(['total_distance']).run()
    method = next(s for s in at.selectbox if s.label.startswith('Choose outlier detection method'))
    method.set_value('Graphical: Box Plots').run()
    ok(at)
    box = [json.loads(ch.proto.spec) for ch in at.get('plotly_chart')]
    box = next(s for s in box if s['data'] and s['data'][0].get('type') == 'box')
    assert 'Config_2 · Object 1' in box['data'][0]['text'] and box['data'][0]['boxpoints'] == 'all'
    assert any('does not end exactly at 1.5×IQR' in m.value or 'always uses factor 1.5' in m.value
               for m in at.markdown)
    method.set_value('Statistical: IQR (Interquartile Range)').run()
    for factor, n_out in ((1.5, 5), (3.0, 0)):
        next(s for s in at.slider if s.label.startswith('IQR Multiplier')).set_value(factor).run()
        ok(at)
        b = pd.read_csv(io.StringIO(latest(at, 'iqr_bounds')['data']), float_precision='round_trip').iloc[0]
        assert (b['Q1'], b['Q3'], b['Factor']) == (10.9407518363118, 14.463867020058307, factor)
        assert b['Flagged_Count'] == n_out
        assert {m.label: m.value for m in at.metric}['Outliers Detected'] == str(n_out)
    assert any('not a measure of traffic danger' in i.value for i in at.info)


# --------------------------------------------------------------------------- Exercise 5 v05

def test_shared_feature_names_match_clustering(a2):
    if a2 is None:
        pytest.skip()
    from modules import clustering
    for c in CONFIGS[:5]:
        g = a2[(a2.config_source == c) & (a2.obj == 1)].sort_values('tst')
        ours, theirs = ar.association_features(g), clustering.extract_trajectory_features(g)
        for name in clustering.ALL_FEATURES:
            assert ours[name] == theirs[name] or (np.isnan(ours[name]) and np.isnan(theirs[name]))
            assert ar.AR_FEATURE_DEFINITIONS[name][1] in clustering.FEATURE_COLUMN_LABELS[name]   # same unit


def test_grid_geometry():
    tennis = sa.create_spatial_grid('Tennis', 3, 5)
    assert tennis['buffer'] == 5 and tennis['grid_cols'] == 7
    with pytest.raises(ValueError):
        sa.create_spatial_grid('No selection', 3, 5)
    g = sa.create_spatial_grid(None, 2, 2, bounds=(0, 10, 0, 4), geometry='data extent')
    assert g['buffer'] == 0 and len(g['zone_labels']) == 4
    assert g['get_zone'](10, 4) == 'B2' and g['get_zone'](0, 0) == 'A1' and g['get_zone'](11, 1) is None


def test_feature_bin_filter():
    tx = [['total_distance_high', 'x_range_low'], ['total_distance_high', 'x_range_low'], ['x_range_high']]
    rules = ar.add_rule_counts(ar.compute_association_rules(tx, 0.1, 0.1)[0], tx)
    assert len(ar.filter_feature_bins(rules, ['x_range'], ['high'])) == 0
    assert len(ar.filter_feature_bins(rules, ['total_distance'], [])) == len(rules)


def setup_ar(at, analysis='Feature Bins'):
    at.multiselect(key='sidebar_configs').set_value(CONFIGS).run()
    at.selectbox(key='analysis_method').set_value('Association Rules').run()
    next(r for r in at.radio if r.label == 'Analysis Type:').set_value(analysis).run()
    ok(at)


def test_v05_associations_banner_features_and_exports(at):
    setup_ar(at)
    assert any('26 trajectories · 156 sampled positions' in s.value and 'tst 0 to 5' in s.value for s in at.success)
    assert any('Duration (tst units) = 5' in w.value for w in at.warning)             # constant feature reported
    at.select_slider(key='ar_n_bins').set_value(3).run()
    at.slider(key='ar_min_support').set_value(0.1).run()
    at.slider(key='ar_min_confidence').set_value(0.6).run()
    mine(at)
    assert any('111 rules' in c.value for c in at.caption)
    assert any('Denominator N = 26 transactions' in m.value for m in at.markdown)
    settings = pd.read_csv(io.StringIO(latest(at, 'association_settings')['data'])).set_index('Setting')['Value']
    assert settings['mining features'] == 'total_distance, duration, avg_speed, x_range, y_range'
    assert settings['transactions N'] == '26' and settings['time window'] == '0.0, 5.0'
    bins = pd.read_csv(io.StringIO(latest(at, 'association_bins')['data']), float_precision='round_trip')
    assert set(bins['Feature']) == {'total_distance', 'avg_speed', 'x_range', 'y_range'}       # duration constant

    # Feature/bin result filter changes only the displayed rules
    at.multiselect(key='ar_filter_features').set_value(['x_range']).run()
    at.multiselect(key='ar_filter_bins').set_value(['high']).run()
    ok(at)
    shown = pd.read_csv(io.StringIO(latest(at, 'association_rules_displayed')['data']))
    assert len(shown) > 0 and shown['rule'].str.contains('x_range_high').all()
    assert len(pd.read_csv(io.StringIO(latest(at, 'association_rules_all')['data']))) == 111

    # A second setting: drop a feature -> stale, then new results and new exports
    del at.downloads[:]
    at.multiselect(key='ar_features').set_value(['total_distance', 'avg_speed']).run()
    assert any(OUTDATED in w.value and 'mining features' in w.value for w in at.warning)
    assert latest(at, 'association_rules') is None
    mine(at)
    at.multiselect(key='ar_filter_features').set_value([]).run()       # clear the result filter of the first run
    full = pd.read_csv(io.StringIO(latest(at, 'association_rules_all')['data']))
    assert 0 < len(full) < 111 and not full['rule'].str.contains('x_range').any()
    settings = pd.read_csv(io.StringIO(latest(at, 'association_settings')['data'])).set_index('Setting')['Value']
    assert settings['mining features'] == 'total_distance, avg_speed' and settings['rules'] == str(len(full))
    assert any(f'{len(full)} rules' in c.value for c in at.caption)

    # Time window is part of the signature
    at.number_input(key='ar_end').set_value(4.0).run()
    assert any(OUTDATED in w.value and 'time window' in w.value for w in at.warning)


@pytest.mark.parametrize('analysis', ['Spatial Zones', 'Combined (Spatial + Features)'])
def test_v05_spatial_zones_use_data_extent(at, analysis):
    setup_ar(at, analysis)
    assert any('Zone grid: data extent' in c.value for c in at.caption)
    mine(at)
    tx = at.session_state['ar_transactions']
    assert len(tx) == 26
    zones = {i for t in tx for i in t if not i.startswith(('total', 'duration', 'avg', 'x_', 'y_'))}
    assert len(zones) > 3                       # spread over the grid, not clamped into one edge column
    assert any('Denominator N = 26 transactions' in m.value for m in at.markdown)


def test_v05_zone_sequences_data_extent(at):
    at.multiselect(key='sidebar_configs').set_value(EIGHT).run()
    at.selectbox(key='analysis_method').set_value('Sequence Analysis').run()
    ok(at)
    assert any('Zone grid: data extent' in c.value for c in at.caption)
    assert any('8 trajectories · 48 sampled positions' in c.value for c in at.caption)
    seqs = pd.read_csv(io.StringIO(latest(at, 'sequences_')['data']))
    tokens = {t for s in seqs['Sequence'] for t in s.split('-')}
    assert len(tokens) > 3


def test_v05_lh_chart_and_comparison(at, a2):
    at.multiselect(key='sidebar_configs').set_value(CONFIGS).run()
    at.selectbox(key='analysis_method').set_value('Sequence Analysis').run()
    at.radio(key='seq_encoding').set_value('Movement steps (L/H)').run()
    ok(at)
    assert at.radio(key='seq_step_mode').options == ['Ordered, gaps allowed', 'Token co-occurrence (order ignored)']
    assert at.radio(key='seq_step_mode').value == 'Ordered, gaps allowed'
    chart = pd.read_csv(io.StringIO(latest(at, 'lh_pattern_frequency')['data'])).set_index('Pattern')
    assert chart.loc['L followed by H (gaps allowed)', 'Fraction'] == '20/26'
    assert chart.loc['H followed by L (gaps allowed)', 'Fraction'] == '6/26'
    cmp_ = pd.read_csv(io.StringIO(latest(at, 'lh_step_comparison')['data']))
    assert sorted(cmp_['Configuration'].unique()) == ['Config_0', 'Config_1'] and len(cmp_) == 10
    at.multiselect(key='seq_step_compare').set_value(['Config_2_obj1', 'Config_22_obj1']).run()
    cmp_ = pd.read_csv(io.StringIO(latest(at, 'lh_step_comparison')['data']))
    assert ''.join(cmp_[cmp_.Configuration == 'Config_22']['Token']) == 'HHLLL'
    # Second setting: threshold 1 m updates the chart export
    at.number_input(key='seq_step_threshold').set_value(1.0).run()
    d = latest(at, 'lh_pattern_frequency')
    assert 'thr1m' in d['file_name']
    _, s1 = sa.step_sequences(sa.step_table(a2, CONFIGS, [1], 0, 5), threshold=1.0)
    m1 = dict(zip(s1['Trajectory'], s1['Sequence']))
    chart = pd.read_csv(io.StringIO(d['data'])).set_index('Pattern')
    for label, pat in (('L followed by H (gaps allowed)', ['L', 'H']), ('H followed by L (gaps allowed)', ['H', 'L'])):
        assert chart.loc[label, 'Fraction'] == f"{sa.pattern_support(m1, pat)[1]}/26"
    assert chart.loc['L followed by H (gaps allowed)', 'Fraction'] != '20/26'      # the setting really changed it


def test_v05_outlier_summary_and_fences(at):
    at.multiselect(key='sidebar_configs').set_value(CONFIGS).run()
    at.selectbox(key='analysis_method').set_value('Outlier Detection').run()
    next(m for m in at.multiselect if m.label.startswith('Select features')).set_value(['total_distance']).run()
    next(s for s in at.selectbox if s.label.startswith('Choose outlier detection method')).set_value(
        'Statistical: IQR (Interquartile Range)').run()
    assert any('26 trajectories · 156 sampled positions' in c.value for c in at.caption)
    for factor, ids in ((1.5, 'Config_2_obj1, Config_11_obj1, Config_16_obj1, Config_17_obj1, Config_18_obj1'),
                        (3.0, 'none (zero flags)')):
        next(s for s in at.slider if s.label.startswith('IQR Multiplier')).set_value(factor).run()
        ok(at)
        summary = at.dataframe[[i for i, d in enumerate(at.dataframe) if 'Flagged IDs' in d.value.columns][0]].value
        assert summary['Multiplier'].iloc[0] == f'{factor:g} × IQR'
        assert sorted(summary['Flagged IDs'].iloc[0].split(', ')) == sorted(ids.split(', '))
        specs = [json.loads(ch.proto.spec) for ch in at.get('plotly_chart')]
        fence = next(s for s in specs if 'active fences' in s['layout']['title']['text'])
        assert fence['data'][0]['name'] == 'Standard box plot (whiskers: 1.5 × IQR rule)'
        texts = [a['text'] for a in fence['layout']['annotations']]
        assert any(f'({factor:g} × IQR)' in t for t in texts)
        b = pd.read_csv(io.StringIO(latest(at, 'iqr_bounds')['data']), float_precision='round_trip').iloc[0]
        assert b['Factor'] == factor
    assert any('Result: no trajectory flagged' in s.value for s in at.success)


def test_outlier_avg_speed_matches_clustering():
    from modules import clustering
    df = pd.DataFrame({'config_source': 'C', 'obj': 1, 'tst': [0, 1, 3], 'x': [0.0, 3.0, 3.0], 'y': [0.0, 0.0, 4.0]})
    f = od.extract_trajectory_features(df, 'C', 1)
    assert f['avg_speed'] == clustering.extract_trajectory_features(df)['avg_speed'] == 7 / 3   # not mean(3, 2)
