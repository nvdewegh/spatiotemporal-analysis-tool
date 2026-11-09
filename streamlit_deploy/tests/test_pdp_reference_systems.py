"""
G2/R2/R4/C4 reference systems: geometry, estimators, engine parity, caching, normalisation and linking.

Data-dependent checks use A2_detailed_dataset.csv (PDP_DETAILED_CSV, or the course location if present).
Run from streamlit_deploy/: python -m pytest tests/test_pdp_reference_systems.py
"""
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from modules import pdp_analysis as pa                      # noqa: E402
from modules import pdp_reference_systems as rs             # noqa: E402

_COURSE_CSV = os.path.join(HERE, '..', '..', '..', '04Exercises', 'AJ2627', 'ExercisesForStudents', 'ForStudents',
                           'Data', 'Data', 'CSV_PDP input', 'A2_detailed_dataset.csv')
DETAILED = os.environ.get('PDP_DETAILED_CSV') or _COURSE_CSV
needs_data = pytest.mark.skipif(not os.path.exists(DETAILED), reason='A2_detailed_dataset.csv not available')


def load_detailed():
    raw = pd.read_csv(DETAILED)
    df = raw.rename(columns={'conID': 'config', 'tstID': 'tst', 'poiID': 'obj'})
    df['config_source'] = 'Config_' + df['config'].astype(str)
    return df


@pytest.fixture(scope='module')
def ref():
    return rs.load_reference()


# ---------------------------------------------------------------- geometry

def test_packaged_road_directions_are_the_corrected_ones(ref):
    roads = ref['road_estimates']
    assert np.allclose(roads['cyclist_road']['origin_xy_m'], [37.06378722096845, 8.814612134203648])
    assert np.allclose(roads['cyclist_road']['unit_direction'], [0.9858695442405665, -0.16751490005040584])
    u_vehicle = roads['vehicle_approach_road']['unit_direction']
    assert np.allclose(u_vehicle, [-0.012124247997652368, -0.999926498604018])
    assert u_vehicle[0] < 0 and u_vehicle[1] < 0            # downward approach has a small negative x
    assert ref['geometry_source']['pixel_to_metre_scale'] == pytest.approx(0.04629183315554431)
    assert ref['packaging']['source_content_sha256']


def test_independent_non_perpendicular_projections_and_positive_left():
    origin = np.array([1.0, 2.0])
    u_a = np.array([1.0, 0.0])
    u_b = np.array([1.0, 1.0]) / np.sqrt(2)               # 45°, not perpendicular to u_a
    p = np.array([[3.0, 5.0]])
    a = rs.project_straight(p, origin, u_a)
    b = rs.project_straight(p, origin, u_b)
    assert a['along'][0] == pytest.approx(2.0)             # dot((2, 3), (1, 0))
    assert a['left'][0] == pytest.approx(3.0)              # point is left of +x
    assert b['along'][0] == pytest.approx(5 / np.sqrt(2))  # independent dot product, not an oblique inverse
    assert b['left'][0] == pytest.approx(1 / np.sqrt(2))   # cross(u_b, (2, 3)) = (3 - 2)/sqrt2
    right = rs.project_straight([[3.0, 0.0]], origin, u_a)
    assert right['left'][0] == pytest.approx(-2.0)         # right of the direction is negative


def test_vehicle_road_left_sign_with_corrected_direction(ref):
    road = ref['road_estimates']['vehicle_approach_road']
    o = np.array(road['origin_xy_m'])
    # travelling down (negative y), positive left is towards +x
    res = rs.project_straight([o + [1.0, 0.0], o + [0.0, -1.0]], o, road['unit_direction'])
    assert res['left'][0] > 0.99
    assert res['along'][1] == pytest.approx(0.999926498604018)


def test_polyline_bend_extension_ties_and_flags():
    poly = [[0, 0], [10, 0], [10, 10]]                     # right-angle left bend
    pts = [[5, 1], [11, 5], [-3, -2], [10, 14], [7, 3]]
    r = rs.project_polyline(pts, poly)
    assert r['along'][0] == pytest.approx(5) and r['left'][0] == pytest.approx(1)
    assert r['along'][1] == pytest.approx(15) and r['left'][1] == pytest.approx(-1)   # right of upward segment
    assert r['along'][2] == pytest.approx(-3) and r['extrapolated'][2] == -1          # before start: negative
    assert r['left'][2] == pytest.approx(-2)
    assert r['along'][3] == pytest.approx(24) and r['extrapolated'][3] == 1           # beyond end
    assert list(r['extrapolated'][[0, 1, 4]]) == [0, 0, 0]
    # (7, 3) is equally far (3 m) from both segments, at along 7 and 13: tie -> smallest along, flagged ambiguous
    assert r['along'][4] == pytest.approx(7) and r['segment'][4] == 0 and r['ambiguous'][4]
    # outer corner: closest point is the shared vertex (along 10); smallest segment index, not ambiguous
    v = rs.project_polyline([[12, -2]], poly)
    assert v['along'][0] == pytest.approx(10) and v['segment'][0] == 0 and not v['ambiguous'][0]
    assert v['left'][0] == pytest.approx(-2)                                              # tangent of segment 0


def test_single_segment_extends_both_ways_and_degenerate_rejected():
    r = rs.project_polyline([[-2, 1], [5, -1]], [[0, 0], [2, 0]])
    assert list(r['along']) == pytest.approx([-2, 5]) and list(r['extrapolated']) == [-1, 1]
    with pytest.raises(rs.ReferenceError):
        rs.project_polyline([[0, 0]], [[0, 0], [0, 0], [1, 0]])


def test_invalid_reference_is_rejected_not_replaced_by_g2(ref):
    bad = json.loads(json.dumps({k: v for k, v in ref.items() if k != '_identity'}))
    bad['flows']['cyclist']['directed_polyline_xy_m'][1] = bad['flows']['cyclist']['directed_polyline_xy_m'][0]
    with pytest.raises(rs.ReferenceError, match='degenerate'):
        rs.load_reference(text=json.dumps(bad))
    bad = json.loads(json.dumps({k: v for k, v in ref.items() if k != '_identity'}))
    bad['road_estimates']['cyclist_road']['unit_direction'] = [1.0, float('nan')]
    with pytest.raises(rs.ReferenceError):
        rs.load_reference(text=json.dumps(bad))
    df = pd.DataFrame({'x': [1.0], 'y': [2.0], 'tst': [0], 'obj': [0], 'config_source': ['c']})
    with pytest.raises(rs.ReferenceError):
        rs.add_descriptor_columns(df, ('R2', None), None)


# ---------------------------------------------------------------- estimators

def test_mean_median_equal_weighting_and_changing_support():
    d = np.array([1.0, 0.0])
    xs = np.linspace(0, 10, 41)
    paths = {
        'a': np.column_stack([xs, np.zeros_like(xs)]),
        'b': np.column_stack([xs, np.full_like(xs, 1.0)]),
        'c': np.column_stack([xs, np.full_like(xs, 5.0)]),     # outlier lane
        # dense sampling must not add weight: same lane, 10x more points, covers only x in [4, 6]
        'd': np.column_stack([np.linspace(4, 6, 400), np.full(400, 6.0)]),
    }
    mean = rs.build_flow_reference(paths, d, 'mean')
    median = rs.build_flow_reference(paths, d, 'median')
    st = np.array(mean['alignment_stations_m'])
    counts = np.array(mean['support_counts'])
    unsmoothed = np.array(mean['unsmoothed_reference_xy_m'])
    inside = (st >= 4) & (st <= 6)
    assert set(counts[inside]) == {4} and set(counts[~inside]) == {3}
    assert np.allclose(unsmoothed[inside, 1], (0 + 1 + 5 + 6) / 4)            # equal weight per encounter
    assert np.allclose(unsmoothed[~inside, 1], 2.0)                             # (0 + 1 + 5) / 3
    med = np.array(median['unsmoothed_reference_xy_m'])
    assert np.allclose(med[~inside, 1], 1.0) and np.allclose(med[inside, 1], 3.0)
    assert mean['label'] == 'smoothed spatial mean reference path'
    assert median['label'] == 'smoothed spatial median reference path'
    # minimum support: two paths only -> no supported stations
    with pytest.raises(rs.ReferenceError):
        rs.build_flow_reference({'a': paths['a'], 'b': paths['b']}, d)


def test_non_monotone_alignment_is_rejected():
    back = np.array([[0, 0], [2, 0], [1, 0.1]])
    with pytest.raises(rs.ReferenceError, match='Non-monotone'):
        rs.build_flow_reference({'a': back, 'b': back, 'c': back}, [1, 0])


@needs_data
def test_rebuild_from_full_dataset_reproduces_packaged_flows(ref):
    df = load_detailed()
    assert rs.data_compatibility(ref, df)[0] == 'reference_dataset'
    mean, median = rs.rebuild_flows_from_data(df, ref)
    for role in ('cyclist', 'vehicle'):
        assert np.allclose(mean[role]['directed_polyline_xy_m'], ref['flows'][role]['directed_polyline_xy_m'])
        assert mean[role]['support_counts'] == ref['flows'][role]['support_counts']
        packaged_median = ref['median_flows_sensitivity_comparison'][role]['directed_polyline_xy_m']
        assert np.allclose(median[role]['directed_polyline_xy_m'], packaged_median)


@needs_data
def test_subset_is_not_silently_treated_as_reference_dataset(ref):
    df = load_detailed()
    sub = df[df['config'].isin([1, 2, 3])]
    assert rs.data_compatibility(ref, sub)[0] == 'other_in_extent'
    moved = df.assign(x=df['x'] + 500)
    assert rs.data_compatibility(ref, moved)[0] == 'incompatible'


# ---------------------------------------------------------------- engine

def _legacy_module():
    """pdp_analysis.py as last committed (before the reference-system extension)."""
    try:
        src = subprocess.run(['git', 'show', 'HEAD:streamlit_deploy/modules/pdp_analysis.py'], cwd=HERE,
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip('git history not available')
    if 'reference_key' in src:
        pytest.skip('HEAD already contains the extension; parity was checked before committing')
    spec = importlib.util.spec_from_loader('modules._pdp_legacy', loader=None)
    legacy = importlib.util.module_from_spec(spec)
    legacy.__package__ = 'modules'
    exec(compile(src, 'pdp_legacy', 'exec'), legacy.__dict__)
    return legacy


def _toy(seed=0, configs=4, steps=6):
    rng = np.random.default_rng(seed)
    rows = [{'config_source': f'Config_{c}', 'config': c, 'tst': t, 'obj': o,
             'x': float(np.round(rng.uniform(20, 60), 1)), 'y': float(np.round(rng.uniform(3, 23), 1))}
            for c in range(configs) for t in range(steps) for o in (0, 1)]
    return pd.DataFrame(rows)


@pytest.mark.parametrize('variant,ext', [('fundamental', None), ('rough', None), ('buffer_rough', (('P', 30.0, 10.0),))])
def test_g2_parity_with_committed_engine(variant, ext):
    legacy = _legacy_module()
    df = _toy()
    configs = tuple(sorted(df['config_source'].unique()))
    kw = dict(window_length=3, buffer_x=0.5, buffer_y=0.5, rough_x=0.3, rough_y=0.3, pdp_variant=variant,
              external_points=ext)
    kw.update(pa.pdp_variant_parameters(variant, 0.5, 0.5, 0.3, 0.3))
    old, _ = legacy._compute_pdp_distance_matrix_uncached(df, configs, (0, 1), 0, 5, **kw)
    pa.clear_pdp_distance_cache()
    new, _ = pa.compute_pdp_distance_matrix(df, configs, (0, 1), 0, 5, **kw)
    assert np.array_equal(old, new)
    assert pa.pdp_run_signature(df, configs, (0, 1), 0, 5, 3, 0, 0, 0, 0, ['fundamental'], None) == \
        legacy.pdp_run_signature(df, configs, (0, 1), 0, 5, 3, 0, 0, 0, 0, ['fundamental'], None)
    assert pa.pdp_max_possible_distance(2, 3, 4) == legacy.pdp_max_possible_distance(2, 3, 4)


def test_descriptor_contributions_and_self_mutual_sum_to_total():
    df = _toy(1)
    a = df[df['config_source'] == 'Config_0']
    b = df[df['config_source'] == 'Config_1']
    total, info = pa.compute_pdp_distance_pair(a, b, 3, details=True)
    assert total == pa.compute_pdp_distance_pair(a, b, 3)
    assert sum(info['per_descriptor'].values()) == total
    for col in ('x', 'y'):
        assert info['self'][col] + info['mutual'][col] == info['per_descriptor'][col]
    # self includes different times of one object, not only the diagonal (which is always 0)
    assert info['self']['x'] > 0
    # the same calculation through explicit descriptor columns gives identical channels
    a2, b2 = a.assign(**{'d:u': a['x'], 'd:v': a['y']}), b.assign(**{'d:u': b['x'], 'd:v': b['y']})
    t2, info2 = pa.compute_pdp_distance_pair(a2, b2, 3, descriptors=('d:u', 'd:v'), details=True)
    assert t2 == total and info2['per_descriptor']['d:u'] == info['per_descriptor']['x']


@needs_data
def test_systems_cache_keys_and_components(ref):
    df = load_detailed()
    configs = tuple(f'Config_{c}' for c in range(4))
    results = {}
    for spec in [('G2', None), ('R2', None), ('R4', None), ('C4', 'mean'), ('C4', 'median')]:
        d = rs.add_descriptor_columns(df, spec, ref)
        assert np.array_equal(d['x'], df['x']) and np.array_equal(d['y'], df['y'])   # x/y never overwritten
        cols = rs.descriptor_columns(spec[0])
        m, ids, comp = pa.compute_pdp_distance_matrix(d, configs, (0, 1), 0, 20, window_length=3,
                                                      descriptors=cols, reference_key=rs.spec_identity(ref, spec),
                                                      return_components=True)
        assert np.allclose(sum(comp.values()), m)
        assert list(comp) == list(cols)
        results[spec] = m
    assert not np.array_equal(results[('C4', 'mean')], results[('C4', 'median')])   # estimator changes results
    assert not np.array_equal(results[('R2', None)], results[('R4', None)])
    # R4's first two channels are R2's
    d = rs.add_descriptor_columns(df, ('R4', None), ref)
    _, _, comp4 = pa.compute_pdp_distance_matrix(d, configs, (0, 1), 0, 20, window_length=3,
                                                 descriptors=rs.descriptor_columns('R4'),
                                                 reference_key=rs.spec_identity(ref, ('R4', None)),
                                                 return_components=True)
    assert np.allclose(comp4['d:cyclist_road_along'] + comp4['d:vehicle_road_along'], results[('R2', None)])
    with pytest.raises(ValueError):            # external points need their descriptor values, never x/y instead
        pa.compute_pdp_distance_matrix(d, configs, (0, 1), 0, 20, descriptors=rs.descriptor_columns('R4'),
                                       external_points=(('P', 30.0, 10.0),))
    with pytest.raises(ValueError):            # one Rough value per descriptor
        pa.compute_pdp_distance_matrix(d, configs, (0, 1), 0, 20, descriptors=rs.descriptor_columns('R4'),
                                       pdp_variant='rough', descriptor_roughs=(0.3, 0.3))


# ---------------------------------------------------------------- Buffer, Rough, external points per descriptor

def _random_configs(n_configs=4, n_times=6, seed=1):
    rng = np.random.default_rng(seed)
    rows = [dict(config_source=f'Config_{c}', tst=t, obj=o, x=rng.normal(), y=rng.normal())
            for c in range(n_configs) for t in range(n_times) for o in (0, 1)]
    return pd.DataFrame(rows)


@pytest.mark.parametrize('variant', ['fundamental', 'buffer', 'rough', 'buffer_rough'])
@pytest.mark.parametrize('ext', [None, (('P', 0.1, 0.2), ('Q', -0.3, 0.5))])
@pytest.mark.parametrize('bx,by', [(0.4, 0.3), (0.4, 0), (0, 0.3)])
def test_descriptor_buffer_rough_external_reproduce_g2_when_descriptors_are_x_y(variant, ext, bx, by):
    """The per-descriptor path with descriptors copied from x/y gives exactly the G2 results."""
    df = _random_configs()
    df['d:a'], df['d:b'] = df['x'], df['y']
    configs = tuple(df['config_source'].unique())
    pa.clear_pdp_distance_cache()
    g2, _ = pa.compute_pdp_distance_matrix(df, configs, (0, 1), 0, 5, 3, bx, by, 0.2, 0.1, variant, ext)
    gen, _ = pa.compute_pdp_distance_matrix(
        df, configs, (0, 1), 0, 5, 3, pdp_variant=variant, external_points=ext, descriptors=('d:a', 'd:b'),
        reference_key='copy', descriptor_buffers=(bx, by), descriptor_roughs=(0.2, 0.1),
        external_descriptors=tuple((x, y) for _, x, y in ext) if ext else None)
    assert np.array_equal(g2, gen)


def test_descriptor_rough_and_buffer_act_per_descriptor():
    # One object, three times; descriptor a differs by 0.3 between t0 and t1, b is identical everywhere
    base = pd.DataFrame({'config_source': ['c1'] * 3 + ['c2'] * 3, 'tst': [0, 1, 2] * 2, 'obj': [0] * 6,
                         'x': 0.0, 'y': 0.0, 'd:a': [0, 0.3, 1, 0, -0.3, 1], 'd:b': 0.0})
    run = lambda **kw: pa.compute_pdp_distance_matrix(base, ('c1', 'c2'), (0,), 0, 2, 3, descriptors=('d:a', 'd:b'),
                                                      reference_key='toy', return_components=True, **kw)[2]
    fundamental = run()
    assert fundamental['d:a'][0, 1] == 4 and fundamental['d:b'][0, 1] == 0   # cells (0,1) and (1,0) flip: 2 + 2
    rough_a = run(pdp_variant='rough', descriptor_roughs=(0.5, 0.0))
    assert rough_a['d:a'][0, 1] == 0                                         # within Rough a: equal in both
    rough_b = run(pdp_variant='rough', descriptor_roughs=(0.0, 0.5))
    assert rough_b['d:a'][0, 1] == fundamental['d:a'][0, 1]                  # Rough b leaves descriptor a alone
    buffered = pa.apply_descriptor_buffer(base, ('d:a', 'd:b'), (0.25, 0.0))
    assert len(buffered) == 3 * len(base)
    first = buffered.iloc[:3]
    assert list(first['sub_type']) == ['minus:d:a', 'plus:d:a', 'orig']
    assert list(first['d:a']) == [-0.25, 0.25, 0.0] and list(first['d:b']) == [0.0, 0.0, 0.0]
    assert (buffered['x'] == 0).all()                                        # positions are never moved
    assert pa.pdp_buffer_factor('buffer', 0.25, 0.0, 1.0, 0.0) == 5
    assert pa.pdp_buffer_factor('rough', 0.25, 1.0) == 1


def test_external_points_projected_like_trajectories(ref):
    import pdp_reference_ui as ui
    ox, oy = ref['road_estimates'][rs.ROAD_KEYS['cyclist']]['origin_xy_m']
    points = (('A', ox, oy), ('B', 40.0, 12.0))
    frame = ui.external_frame(points, ('R4', None), ref)
    traj = rs.add_descriptor_columns(pd.DataFrame({'config_source': 'c', 'tst': [0.0, 1.0], 'obj': 0,
                                                   'x': [ox, 40.0], 'y': [oy, 12.0]}), ('R4', None), ref)
    for column in rs.descriptor_columns('R4'):
        assert np.allclose(frame[column], traj[column])
    assert frame['d:cyclist_road_along'].iloc[0] == pytest.approx(0, abs=1e-9)   # the road origin


def test_identity_and_signatures_change_with_reference_and_estimator(ref):
    ids = {rs.spec_identity(ref, s) for s in [('G2', None), ('R2', None), ('R4', None), ('C4', 'mean'), ('C4', 'median')]}
    assert len(ids) == 5
    other = json.loads(json.dumps({k: v for k, v in ref.items() if k != '_identity'}))
    other['flows']['cyclist']['directed_polyline_xy_m'][0][0] += 0.01
    other = rs.load_reference(text=json.dumps(other))
    assert rs.spec_identity(other, ('C4', 'mean')) != rs.spec_identity(ref, ('C4', 'mean'))
    df = _toy()
    sig = lambda key: pa.pdp_run_signature(df, ['a'], [0], 0, 5, 3, 0, 0, 0, 0, ['fundamental'], None, key)
    assert sig(rs.spec_identity(ref, ('C4', 'mean'))) != sig(rs.spec_identity(ref, ('C4', 'median')))
    assert sig(None) == pa.pdp_run_signature(df, ['a'], [0], 0, 5, 3, 0, 0, 0, 0, ['fundamental'], None)


def test_normalisation_scales_with_descriptor_count():
    two = pa.pdp_max_possible_distance(2, 3, 10)
    four = pa.pdp_max_possible_distance(2, 3, 10, n_descriptors=4)
    assert four == 2 * two == 2 * 4 * (36 - 6) * 10
    info = pa.compute_distance_normalization_info(np.array([[0, 480.], [480., 0]]), ['a', 'b'], 2, 3, 1, 0, 10, 4)
    assert info['normalized_matrix'][0, 1] == pytest.approx(20.0)


# ---------------------------------------------------------------- linking

@needs_data
def test_cell_linking_identifies_points_and_relations(ref):
    df = load_detailed()
    a = rs.add_descriptor_columns(df, ('G2', None), ref)
    b = rs.add_descriptor_columns(df, ('C4', 'mean'), ref)
    times = [40, 41, 42]
    pa_ = rs.window_points(a, 'Config_3', [0, 1], times)
    pb_ = rs.window_points(b, 'Config_3', [0, 1], times)
    assert list(pa_['obj']) == [0, 1, 0, 1, 0, 1] and list(pa_['tst']) == [40, 40, 41, 41, 42, 42]
    ma = rs.channel_matrix(pa_, 'x')
    mb = rs.channel_matrix(pb_, 'd:cyclist_flow_along')
    for i, j, ra, rb in rs.changed_cells(ma, mb):
        rows = rs.explain_cell(pa_, pb_, i, j, 'x', 'd:cyclist_flow_along')
        assert ra == rs.relation_code(rows[0]['A scalar'], rows[1]['A scalar'])
        assert rb == rs.relation_code(rows[0]['B scalar'], rows[1]['B scalar'])
        assert rows[0]['x (m)'] == df[(df.config_source == 'Config_3') & (df.obj == pa_.obj[i]) &
                                      (df.tst == pa_.tst[i])]['x'].item()
    with pytest.raises(ValueError):
        rs.changed_cells(ma, ma[:-1, :-1])


@needs_data
def test_scalar_export_keeps_identity(ref):
    df = load_detailed()
    d = rs.add_descriptor_columns(df, ('C4', 'median'), ref)
    table = rs.scalar_table(d, ('C4', 'median'), ['Config_0'], [0, 1], 0, 5)
    assert list(table.columns[:6]) == ['config_source', 'obj', 'role', 'tst', 'x', 'y']
    assert list(table.columns[6:10]) == list(rs.DESCRIPTOR_ORDER['C4'])
    meta = rs.reference_metadata(ref, ('C4', 'median'))
    assert meta['flows']['cyclist']['estimator'] == 'median' and meta['descriptor_order'][0] == 'cyclist_flow_along'
    json.dumps(meta)
