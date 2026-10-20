"""
Regression tests for the Animated Trajectories view (visualize_animated).

Plotly only tweens a line between animation frames when its point count stays the same.
If the trail grows, the longer line is drawn instantly while the head marker is still
transitioning, so the line appears ahead of the head. These tests check that the trail has
a constant length per object, ends exactly at the head and never contains future points.

Run from streamlit_deploy/:  python -m pytest tests
Checks on the A2 dataset run when PDP_A2_CSV points to A2_1Hz_dataset.csv.
"""
import ast
import os
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
import streamlit as st

warnings.filterwarnings('ignore')
APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'streamlit_visualization.py')
FUNCTIONS = {'format_tst', 'create_pitch_figure', 'get_court_dimensions', 'aggregate_points',
             'interpolate_points', 'visualize_animated'}


def load_app_functions():
    """Load only the needed functions from the app file (importing it would run the app)."""
    tree = ast.parse(open(APP).read())
    namespace = {'np': np, 'pd': pd, 'go': go, 'st': st}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS:
            exec(compile(ast.Module([node], []), APP, 'exec'), namespace)
    return namespace


app = load_app_functions()


def synthetic_data():
    rows = []
    for config in ('C1', 'C2'):
        for obj in (0, 1):
            for t in range(8):
                rows.append({'config_source': config, 'obj': obj, 'tst': t,
                             'x': 3.0 * t + obj + (config == 'C2'), 'y': np.sin(t / 2) * 5 + obj})
    return pd.DataFrame(rows)


def check_frames(fig, df, configs, objects):
    n_pairs = len(configs) * len(objects)
    trail_lengths = {}
    for frame in fig.frames:
        traces = frame.data
        assert len(traces) == 2 * n_pairs
        for k in range(n_pairs):
            trail, head = traces[2 * k], traces[2 * k + 1]
            if len(head.x) == 0:
                continue
            # 1. Constant trail length per object, so Plotly can tween it with the head
            trail_lengths.setdefault(k, len(trail.x))
            assert len(trail.x) == trail_lengths[k]
            # 2. The line ends exactly at the head
            assert trail.x[-1] == head.x[0] and trail.y[-1] == head.y[0]
    return trail_lengths


def check_no_future_points(fig):
    """The synthetic paths move in +x, so no trail point may lie beyond the head's x."""
    for frame in fig.frames:
        for k in range(0, len(frame.data), 2):
            trail, head = frame.data[k], frame.data[k + 1]
            if len(head.x) > 0:
                assert max(trail.x) <= head.x[0] + 1e-9


@pytest.mark.parametrize('interpolation, steps, speed', [(True, 2, 1000), (True, 5, 200), (False, 5, 1000)])
def test_trail_tweenable_and_behind_head(interpolation, steps, speed):
    df = synthetic_data()
    fig = app['visualize_animated'](df, ['C1', 'C2'], [0, 1], 0, 7, 'none', 1, 'No selection',
                                    speed, interpolation, steps)
    assert len(fig.frames) > 1
    check_frames(fig, df, ['C1', 'C2'], [0, 1])
    check_no_future_points(fig)


CSV = os.environ.get('PDP_A2_CSV')


@pytest.mark.skipif(not CSV or not os.path.exists(CSV), reason='set PDP_A2_CSV to A2_1Hz_dataset.csv')
def test_a2_dataset_all_configs():
    import sys
    sys.path.insert(0, os.path.dirname(APP))
    from modules import utils
    with open(CSV, 'rb') as f:
        df = utils.load_data(f, update_state=False, show_success=False)
    objects = sorted(df['obj'].unique())
    for config in df['config_source'].unique():
        fig = app['visualize_animated'](df, [config], objects, df['tst'].min(), df['tst'].max(),
                                        'none', 1, 'No selection', 1000, True, 2)
        check_frames(fig, df, [config], objects)
