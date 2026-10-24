"""
Raw PDP distances are never presented as percentages. Percentages are raw distance divided by the
analytical maximum of the completed run and variant (objects, window, windows, Buffer expansion,
external points); the similarity complement is 100 minus that percentage.

B1: A2_1Hz_dataset.csv, configurations 1, 2, 4, 11, objects 0 and 1, indices 0-5, window 6, Fundamental.
D1: A2_detailed_dataset.csv, same configurations and objects, indices 2-122, window 121, Fundamental.
Run from streamlit_deploy/ with PDP_1HZ_CSV pointing to A2_1Hz_dataset.csv
(A2_detailed_dataset.csv is expected in the same folder).
"""
import io
import json
import os
import warnings

import pandas as pd
import pytest
import streamlit
from streamlit.testing.v1 import AppTest

from modules import pdp_analysis as p
from modules.utils import load_data

warnings.filterwarnings('ignore')
APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'streamlit_visualization.py')
CSV = os.environ.get('PDP_1HZ_CSV')
DETAILED = os.path.join(os.path.dirname(CSV), 'A2_detailed_dataset.csv') if CSV else None
pytestmark = pytest.mark.skipif(not CSV or not os.path.exists(CSV), reason='set PDP_1HZ_CSV to A2_1Hz_dataset.csv')

CONFIGS = ('Config_1', 'Config_2', 'Config_4', 'Config_11')
OBJECTS = (0, 1)


class Upload(io.BytesIO):
    def __init__(self, path):
        super().__init__(open(path, 'rb').read())
        self.name = os.path.basename(path)


def load(path):
    return load_data(Upload(path), update_state=False, show_success=False)


def run(df, first, last, window, variant='fundamental', buffer_x=0, buffer_y=0, external=None):
    """Distances and analytical maximum for indices first..last, as the app computes them."""
    times = sorted(df[df['config_source'].isin(CONFIGS) & df['obj'].isin(OBJECTS)]['tst'].unique())
    start, end = times[first], times[last]
    dm, ids = p.compute_pdp_distance_matrix(df, CONFIGS, OBJECTS, start, end, window_length=window,
                                            buffer_x=buffer_x, buffer_y=buffer_y, pdp_variant=variant,
                                            external_points=tuple(external) if external else None)
    n_windows = p.pdp_count_windows(df, CONFIGS, OBJECTS, start, end, window)
    max_possible = p.pdp_max_possible_distance(len(OBJECTS), window, n_windows,
                                               p.pdp_buffer_factor(variant, buffer_x, buffer_y),
                                               len(external or ()))
    return dm, list(ids), max_possible


def pair(dm, ids, a, b):
    return dm[ids.index(f'Config_{a}'), ids.index(f'Config_{b}')]


def ranking(csv, target, similar):
    df = pd.read_csv(io.StringIO(csv))
    row = df[(df['Target_Config'] == f'Config_{target}') & (df['Similar_Config'] == f'Config_{similar}')]
    assert len(row) == 1
    return row.iloc[0]


def test_b1_reference_values_and_ranking_csv():
    dm, ids, max_possible = run(load(CSV), 0, 5, 6)
    assert pair(dm, ids, 1, 2) == 144 and pair(dm, ids, 1, 4) == 0
    assert max_possible == 528
    assert p.normalised_pdp_percent(144, max_possible) == pytest.approx(27.2727, abs=1e-4)

    csv = p.export_similarity_rankings_to_csv(ids, dm, top_k=3, max_possible_distance=max_possible)
    r12 = ranking(csv, 1, 2)
    assert r12['Raw_PDP_Distance'] == 144
    assert r12['Normalised_Distance_%'] == pytest.approx(27.272727, abs=1e-6)
    assert r12['Similarity_Complement_%'] == pytest.approx(72.727273, abs=1e-6)
    r14 = ranking(csv, 1, 4)
    assert (r14['Raw_PDP_Distance'], r14['Normalised_Distance_%'], r14['Similarity_Complement_%']) == (0, 0, 100)
    # Ordering is by raw distance, unchanged by the units
    df = pd.read_csv(io.StringIO(csv))
    assert list(df[df['Target_Config'] == 'Config_1']['Similar_Config']) == ['Config_4', 'Config_11', 'Config_2']


def test_ranking_csv_omits_percentages_without_run_context():
    dm, ids, _ = run(load(CSV), 0, 5, 6)
    df = pd.read_csv(io.StringIO(p.export_similarity_rankings_to_csv(ids, dm, top_k=3)))
    assert 'Normalised_Distance_%' not in df.columns and 'Similarity_Complement_%' not in df.columns
    assert (df['Raw_PDP_Distance'] >= 0).all()


def test_trajectory_title_uses_normalised_distance():
    df = load(CSV)
    dm, ids, max_possible = run(df, 0, 5, 6)
    fig = p.plot_trajectory_comparison(df, ids, ['Config_1', 'Config_2'], df['tst'].min(), df['tst'].max(),
                                       selected_objects=list(OBJECTS), distance_matrix=dm,
                                       max_possible_distance=max_possible, show_court=False)
    title = fig.layout.title.text
    assert 'Raw PDP distance: 144' in title and '27.27%' in title and '72.73%' in title
    assert '-44' not in title


@pytest.mark.skipif(not DETAILED or not os.path.exists(DETAILED), reason='A2_detailed_dataset.csv not found')
def test_d1_reference_values():
    dm, ids, max_possible = run(load(DETAILED), 2, 122, 121)
    assert pair(dm, ids, 1, 4) == 3208
    assert max_possible == 233288
    csv = p.export_similarity_rankings_to_csv(ids, dm, top_k=3, max_possible_distance=max_possible)
    r14 = ranking(csv, 1, 4)
    assert r14['Normalised_Distance_%'] == pytest.approx(1.3751243, abs=1e-6)
    assert r14['Similarity_Complement_%'] == pytest.approx(98.6248757, abs=1e-6)


def test_buffer_with_external_points_uses_its_own_context():
    df = load(CSV)
    external = [('E1', 0.0, 0.0), ('E2', 5.0, 10.0)]
    fund, ids, fund_max = run(df, 0, 5, 6)
    dm, ids, max_possible = run(df, 0, 5, 6, variant='buffer', buffer_x=0.5, buffer_y=0.5, external=external)
    # 5 points per original (left, right, orig, bottom, top), external points buffered as well
    assert max_possible == p.pdp_max_possible_distance(2, 6, 1, buffer_factor=5, n_external_points=2)
    assert max_possible != fund_max
    assert dm.max() > 100                       # raw values above 100 ...
    assert dm.max() <= max_possible             # ... stay within the analytical maximum
    out = pd.read_csv(io.StringIO(p.export_similarity_rankings_to_csv(ids, dm, top_k=3,
                                                                      max_possible_distance=max_possible)))
    assert (out['Normalised_Distance_%'].between(0, 100)).all()
    assert (out['Similarity_Complement_%'] >= 0).all()


# ---------------------------------------------------------------- real app, B1

@pytest.fixture
def app(monkeypatch):
    captured = {}
    original_download = streamlit.download_button

    def capture_download(label, data, *a, **k):
        captured[label] = data
        return original_download(label, data, *a, **k)

    monkeypatch.setattr(streamlit, 'file_uploader',
                        lambda label, *a, **k: [Upload(CSV)] if label.startswith('Upload CSV') else None)
    monkeypatch.setattr(streamlit, 'download_button', capture_download)
    at = AppTest.from_file(APP, default_timeout=300)
    at.session_state['password_correct'] = True
    at.run()
    at.multiselect(key='sidebar_configs').set_value(list(CONFIGS)).run()
    at.multiselect(key='sidebar_objects').set_value(list(OBJECTS)).run()
    at.selectbox(key='analysis_method').set_value('PDP Analysis').run()
    at.slider(key='pdp_window').set_value(6).run()
    at.button(key='compute_pdp').click().run()
    assert not at.exception, [e.message for e in at.exception]
    at.captured = captured
    return at


def pair_tables(at):
    return [f.value for f in at.dataframe if 'Raw PDP distance' in f.value.columns and 'Config A' in f.value.columns]


def titles(at):
    out = []
    for chart in at.get('plotly_chart'):
        title = json.loads(chart.proto.spec).get('layout', {}).get('title', {})
        out.append(title.get('text', '') if isinstance(title, dict) else str(title))
    return out


def test_app_views_and_exports_agree(app):
    # Ranking CSV
    r12 = ranking(app.captured['Download Rankings'], 1, 2)
    assert (r12['Raw_PDP_Distance'], round(r12['Normalised_Distance_%'], 4),
            round(r12['Similarity_Complement_%'], 4)) == (144, 27.2727, 72.7273)

    # Clustering-section trajectory comparison (defaults to Config_1 vs Config_2) and its table
    assert any('Raw PDP distance: 144' in t and '27.27%' in t and '72.73%' in t for t in titles(app))
    rows = [r for t in pair_tables(app) for _, r in t.iterrows()
            if {r['Config A'], r['Config B']} == {'Config_1', 'Config_2'}]
    assert rows and all(r['Raw PDP distance'] == '144' and r['Normalised distance (%)'] == '27.27%'
                        and r['Similarity complement (100 − %)'] == '72.73%' for r in rows)

    # Distance matrix shown normalised, inspect 1 vs 4, normalised-distances CSV
    app.checkbox(key='pdp_show_normalized').check().run()
    app.multiselect(key='pdp_matrix_inspect_multiselect').set_value(['Config_1', 'Config_4']).run()
    assert not app.exception
    rows = [r for t in pair_tables(app) for _, r in t.iterrows()
            if {r['Config A'], r['Config B']} == {'Config_1', 'Config_4'}]
    assert rows and rows[0]['Normalised distance (%)'] == '0.00%' and rows[0]['Similarity complement (100 − %)'] == '100.00%'
    heatmap = next(json.loads(c.proto.spec) for c in app.get('plotly_chart')
                   if 'Distance Matrix' in str(json.loads(c.proto.spec)['layout'].get('title')))
    hover = [h for row in heatmap['data'][0]['hovertext'] for h in row]
    assert any('From: Config_1<br>To: Config_2<br>Normalised distance: 27.27%<br>Raw PDP distance: 144' == h
               for h in hover)
    norm_csv = pd.read_csv(io.StringIO(app.captured['📥 Download Normalized Distances (CSV)']))
    row = norm_csv[(norm_csv['Config_A'] == 'Config_1') & (norm_csv['Config_B'] == 'Config_2')].iloc[0]
    assert row['Raw_Distance'] == 144 and row['Normalized_Distance'] == pytest.approx(27.2727, abs=1e-4)
