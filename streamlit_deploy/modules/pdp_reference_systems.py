"""
PDP reference systems: G2, R2, R4 and C4
=========================================

A reference system defines the ordered list of scalar descriptors on which PDP inequality
matrices are built. The original x/y coordinates are never changed: descriptor values are
stored in separate columns ('d:<descriptor id>'), with projection metadata in 'p:<...>' columns.

G2  global_x, global_y                                (the existing Cartesian baseline)
R2  cyclist_road_along, vehicle_road_along             (independent projections on two road directions)
R4  R2 + cyclist_road_left, vehicle_road_left          (signed perpendicular offsets, positive left)
C4  cyclist_flow_along, vehicle_flow_along,
    cyclist_flow_left, vehicle_flow_left               (projections on fixed directed flow reference paths)

Every point, cyclist or vehicle, is evaluated under the same complete descriptor set.
R/C reference systems are separate from the Buffer/Rough variants: "R" does not mean Rough.

Road geometry is diagram-derived with approximate metric registration (not surveyed). Flow paths
are smoothed spatial mean (default) or median (comparison) reference paths fitted once on the full
26-encounter A2 detailed dataset; they are frozen and never refitted per selection, encounter or cluster.
"""

import hashlib
import json
import math
import os

import numpy as np
import pandas as pd

REFERENCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reference_data')
PACKAGED_REFERENCE_PATH = os.path.join(REFERENCE_DIR, 'A2_pdp_references_v02.json')
PACKAGED_OVERLAY_PATH = os.path.join(REFERENCE_DIR, 'A2_pdp_references_v02_overlay.png')

SYSTEMS = ('G2', 'R2', 'R4', 'C4')
ESTIMATORS = ('mean', 'median')
DEFAULT_SYSTEM, DEFAULT_ESTIMATOR = 'G2', 'mean'

DESCRIPTOR_ORDER = {
    'G2': ('global_x', 'global_y'),
    'R2': ('cyclist_road_along', 'vehicle_road_along'),
    'R4': ('cyclist_road_along', 'vehicle_road_along', 'cyclist_road_left', 'vehicle_road_left'),
    'C4': ('cyclist_flow_along', 'vehicle_flow_along', 'cyclist_flow_left', 'vehicle_flow_left'),
}
SYSTEM_LABELS = {
    'G2': 'G2: global x and y',
    'R2': 'R2: along the two road directions',
    'R4': 'R4: along and positive-left of the two roads',
    'C4': 'C4: along and positive-left of the cyclist and vehicle flow paths',
}
DESCRIPTOR_LABELS = {
    'global_x': 'global x (m)', 'global_y': 'global y (m)',
    'cyclist_road_along': 'along lower (cyclist) road (m)',
    'vehicle_road_along': 'along vehicle approach road (m)',
    'cyclist_road_left': 'left of lower (cyclist) road (m)',
    'vehicle_road_left': 'left of vehicle approach road (m)',
    'cyclist_flow_along': 'along cyclist flow path (m)',
    'vehicle_flow_along': 'along vehicle flow path (m)',
    'cyclist_flow_left': 'left of cyclist flow path (m)',
    'vehicle_flow_left': 'left of vehicle flow path (m)',
}
# G2 descriptors are the existing coordinate columns (exact legacy parity); others are derived columns
G2_COLUMNS = {'global_x': 'x', 'global_y': 'y'}
OBJECT_ROLES = {0: 'cyclist', 1: 'vehicle'}

ROAD_KEYS = {'cyclist': 'cyclist_road', 'vehicle': 'vehicle_approach_road'}
TIE_TOLERANCE = 1e-9          # numerical tie tolerance for projections (independent of Rough tolerances)
DISCONTINUITY_RATIO, DISCONTINUITY_SLACK = 3.0, 0.5   # inspection heuristic, see projection_flags


class ReferenceError(ValueError):
    """Invalid or incompatible reference geometry. Never silently replaced by G2."""


# =============================================================================
# SYSTEM SPECIFICATIONS
# =============================================================================

def system_spec(system, estimator=DEFAULT_ESTIMATOR):
    """Normalised (system, estimator); the estimator only matters for C4."""
    if system not in SYSTEMS:
        raise ReferenceError(f"Unknown reference system {system!r}; choose one of {', '.join(SYSTEMS)}.")
    if estimator not in ESTIMATORS:
        raise ReferenceError(f"Unknown flow estimator {estimator!r}; choose mean or median.")
    return (system, estimator if system == 'C4' else None)


def spec_label(spec):
    system, estimator = spec
    return f"C4 ({estimator})" if system == 'C4' else system


def descriptor_column(descriptor_id):
    return G2_COLUMNS.get(descriptor_id, f'd:{descriptor_id}')


def descriptor_columns(system):
    return tuple(descriptor_column(d) for d in DESCRIPTOR_ORDER[system])


# =============================================================================
# REFERENCE LOADING, VALIDATION AND IDENTITY
# =============================================================================

def _finite_array(values, shape_tail, what):
    a = np.asarray(values, dtype=float)
    if a.ndim != len(shape_tail) + 1 or a.shape[1:] != shape_tail or not np.all(np.isfinite(a)):
        raise ReferenceError(f"{what} must be finite with shape (n, {', '.join(map(str, shape_tail))}).")
    return a


def validate_reference(ref):
    """Check every geometry the descriptors use; raise ReferenceError with a precise reason."""
    if not isinstance(ref, dict):
        raise ReferenceError("Reference must be a JSON object.")
    roads = ref.get('road_estimates') or {}
    for role, key in ROAD_KEYS.items():
        road = roads.get(key)
        if not isinstance(road, dict):
            raise ReferenceError(f"road_estimates.{key} is missing.")
        origin = np.asarray(road.get('origin_xy_m'), dtype=float)
        u = np.asarray(road.get('unit_direction'), dtype=float)
        if origin.shape != (2,) or u.shape != (2,) or not np.all(np.isfinite(origin)) or not np.all(np.isfinite(u)):
            raise ReferenceError(f"road_estimates.{key} needs finite origin_xy_m and unit_direction.")
        if abs(np.linalg.norm(u) - 1) > 1e-6:
            raise ReferenceError(f"road_estimates.{key}.unit_direction is not a unit vector.")
    for group in ('flows', 'median_flows_sensitivity_comparison'):
        flows = ref.get(group) or {}
        for role in ('cyclist', 'vehicle'):
            flow = flows.get(role)
            if not isinstance(flow, dict):
                raise ReferenceError(f"{group}.{role} is missing.")
            poly = _finite_array(flow.get('directed_polyline_xy_m'), (2,), f"{group}.{role}.directed_polyline_xy_m")
            if len(poly) < 2:
                raise ReferenceError(f"{group}.{role} needs at least two vertices.")
            if np.any(np.linalg.norm(np.diff(poly, axis=0), axis=1) <= TIE_TOLERANCE):
                raise ReferenceError(f"{group}.{role} has a degenerate (zero-length) segment; remove it "
                                     "explicitly when preparing the reference.")
            counts = flow.get('support_counts')
            if counts is not None and len(counts) != len(poly):
                raise ReferenceError(f"{group}.{role}.support_counts does not match the vertices.")
    if group_estimator(ref, 'flows') != 'mean' or group_estimator(ref, 'median_flows_sensitivity_comparison') != 'median':
        raise ReferenceError("flows must hold the mean paths and median_flows_sensitivity_comparison the median paths.")
    return ref


def group_estimator(ref, group):
    estimators = {ref[group][r].get('estimator') for r in ('cyclist', 'vehicle')}
    return estimators.pop() if len(estimators) == 1 else None


def load_reference(path=None, text=None):
    """Load and validate a reference JSON (the packaged A2 preset by default, or explicitly supplied text)."""
    if text is None:
        with open(path or PACKAGED_REFERENCE_PATH, encoding='utf-8') as f:
            text = f.read()
    try:
        ref = json.loads(text)
    except ValueError as exc:
        raise ReferenceError(f"Reference is not valid JSON ({exc}).") from None
    validate_reference(ref)
    ref['_identity'] = reference_identity(ref)
    return ref


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'))


def reference_identity(ref):
    """Content hash of the geometry and provenance that determine descriptor values."""
    core = {k: ref.get(k) for k in ('road_estimates', 'flows', 'median_flows_sensitivity_comparison',
                                    'source_sha256', 'geometry_source', 'coordinate_frame')}
    return hashlib.sha256(_canonical(core).encode()).hexdigest()


def spec_identity(ref, spec):
    """Identity of one descriptor definition: reference hash (if used), system, estimator, descriptor order."""
    system, estimator = spec
    used = 'none' if system == 'G2' else (ref or {}).get('_identity') or reference_identity(ref)
    return f"{system}|{estimator or '-'}|{','.join(DESCRIPTOR_ORDER[system])}|{used[:16]}"


def flows_for(ref, estimator):
    return ref['flows'] if estimator == 'mean' else ref['median_flows_sensitivity_comparison']


def data_content_sha256(df):
    """Same definition as packaging.source_content_sha256: rows (config, tst, obj, x, y) sorted by config, obj, tst."""
    a = df.sort_values(['config', 'obj', 'tst'], kind='mergesort')[['config', 'tst', 'obj', 'x', 'y']]
    return hashlib.sha256(np.ascontiguousarray(a.to_numpy(dtype=float)).tobytes()).hexdigest()


def data_compatibility(ref, df):
    """
    ('reference_dataset', msg) when the loaded data is the dataset the references were fitted on;
    ('other_in_extent', msg) when it lies inside the reference extent but is a different dataset (needs
    explicit acknowledgement); ('incompatible', msg) otherwise.
    """
    packaging = ref.get('packaging') or {}
    required = {'config', 'tst', 'obj', 'x', 'y'}
    if not required.issubset(df.columns):
        return 'incompatible', "The loaded data lacks config/tst/obj/x/y columns."
    if packaging.get('source_content_sha256') and data_content_sha256(df) == packaging['source_content_sha256']:
        return 'reference_dataset', "Loaded data is the dataset these references were fitted on."
    roles = packaging.get('object_roles') or {'0': 'cyclist', '1': 'vehicle'}
    if not set(df['obj'].unique()).issubset({int(k) for k in roles}):
        return 'incompatible', "Objects other than 0 (cyclist) and 1 (vehicle) are present; roles are undefined."
    extent = packaging.get('source_extent_xy_m')
    if extent:
        margin = 5.0
        inside = (df['x'].between(extent[0] - margin, extent[1] + margin).all() and
                  df['y'].between(extent[2] - margin, extent[3] + margin).all())
        if not inside:
            return 'incompatible', ("The loaded coordinates lie outside the reference frame "
                                    f"(x {extent[0]:.1f}–{extent[1]:.1f} m, y {extent[2]:.1f}–{extent[3]:.1f} m).")
    return 'other_in_extent', ("The loaded data is not the dataset the references were fitted on, although it lies "
                               "in the same local metre extent. The frozen references are not refitted.")


# =============================================================================
# PROJECTIONS
# =============================================================================

def cross2(u, v):
    """cross(u, v) = u_x v_y - u_y v_x (broadcast over rows)."""
    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def project_straight(points, origin, unit_direction):
    """Along s = dot(p - origin, u) and positive-left l = cross(u, p - origin); foot = origin + s u."""
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    o, u = np.asarray(origin, dtype=float), np.asarray(unit_direction, dtype=float)
    rel = p - o
    along = rel @ u
    left = cross2(u, rel)
    return {'along': along, 'left': left, 'proj_xy': o + along[:, None] * u}


def project_polyline(points, polyline, tie_tolerance=TIE_TOLERANCE):
    """
    Closest-point projection onto a directed polyline with tangent-ray extensions at both ends.

    Interior segment parameters are clamped to [0, 1]; the first segment allows t < 0 and the last t > 1
    (a single segment allows both). Ties within tie_tolerance are resolved by the smallest longitudinal value,
    then the smallest segment index. Longitudinal = arc length from the first vertex (negative before it);
    lateral = cross(unit tangent of the selected segment, p - projection), positive left.

    Returns along, left, segment, t, proj_xy, extrapolated (-1 before start, +1 after end, 0 inside)
    and ambiguous (equally close candidates at genuinely different longitudinal positions).
    """
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    v = np.asarray(polyline, dtype=float)
    a, b = v[:-1], v[1:]
    seg = b - a
    length = np.linalg.norm(seg, axis=1)
    if np.any(length <= tie_tolerance):
        raise ReferenceError("Polyline has a degenerate segment.")
    tangent = seg / length[:, None]
    cum = np.r_[0.0, np.cumsum(length)]
    m = len(seg)
    t = ((p[:, None, :] - a[None]) * seg[None]).sum(-1) / (length ** 2)[None]
    lo = np.full(m, 0.0); hi = np.full(m, 1.0)
    lo[0] = -np.inf; hi[-1] = np.inf
    t = np.clip(t, lo[None], hi[None])
    foot = a[None] + t[..., None] * seg[None]
    dist = np.linalg.norm(p[:, None, :] - foot, axis=-1)
    along_all = cum[:-1][None] + t * length[None]
    dmin = dist.min(axis=1, keepdims=True)
    candidate = dist <= dmin + tie_tolerance
    # smallest longitudinal value among the tied candidates, then the smallest segment index
    masked_along = np.where(candidate, along_all, np.inf)
    best_along = masked_along.min(axis=1, keepdims=True)
    chosen = candidate & (masked_along <= best_along + tie_tolerance)
    idx = np.argmax(chosen, axis=1)
    rows = np.arange(len(p))
    ambiguous = (np.where(candidate, along_all, -np.inf).max(axis=1) - best_along[:, 0]) > max(tie_tolerance, 1e-6)
    ts = t[rows, idx]
    proj = foot[rows, idx]
    left = cross2(tangent[idx], p - proj)
    extrapolated = np.where((idx == 0) & (ts < 0), -1, np.where((idx == m - 1) & (ts > 1), 1, 0))
    return {'along': along_all[rows, idx], 'left': left, 'segment': idx, 't': ts, 'proj_xy': proj,
            'extrapolated': extrapolated, 'ambiguous': ambiguous}


def projection_flags(along, xy, ratio=DISCONTINUITY_RATIO, slack=DISCONTINUITY_SLACK):
    """
    Inspection flag for one object's time-ordered points: the projected longitudinal value jumps by more than
    ratio × the observed movement + slack metres (projection switched branch, e.g. inside a bend).
    Results are never smoothed; this only marks points to inspect.
    """
    along, xy = np.asarray(along, dtype=float), np.asarray(xy, dtype=float)
    flags = np.zeros(len(along), dtype=bool)
    if len(along) > 1:
        jump = np.abs(np.diff(along)) > ratio * np.linalg.norm(np.diff(xy, axis=0), axis=1) + slack
        flags[1:] |= jump
    return flags


# =============================================================================
# DESCRIPTOR CHANNELS
# =============================================================================

def add_descriptor_columns(df, spec, ref):
    """
    Return a copy of df with the descriptor columns of spec added ('d:<id>') plus projection metadata
    ('p:<role>_<kind>:<field>'). x and y are never modified. Each point is transformed once per definition.
    """
    system, estimator = spec
    out = df.copy()
    if system == 'G2':
        return out
    if ref is None:
        raise ReferenceError(f"{system} needs reference geometry; none is loaded.")
    validate_reference(ref)
    xy = out[['x', 'y']].to_numpy(dtype=float)
    if not np.all(np.isfinite(xy)):
        raise ReferenceError("Some x/y values are missing or not finite; descriptors cannot be computed.")
    order = DESCRIPTOR_ORDER[system]
    if system in ('R2', 'R4'):
        for role, key in ROAD_KEYS.items():
            road = ref['road_estimates'][key]
            res = project_straight(xy, road['origin_xy_m'], road['unit_direction'])
            out[f'd:{role}_road_along'] = res['along']
            if f'{role}_road_left' in order:
                out[f'd:{role}_road_left'] = res['left']
            out[f'p:{role}_road:proj_x'], out[f'p:{role}_road:proj_y'] = res['proj_xy'][:, 0], res['proj_xy'][:, 1]
    else:
        flows = flows_for(ref, estimator)
        for role in ('cyclist', 'vehicle'):
            res = project_polyline(xy, flows[role]['directed_polyline_xy_m'])
            out[f'd:{role}_flow_along'] = res['along']
            out[f'd:{role}_flow_left'] = res['left']
            prefix = f'p:{role}_flow:'
            out[prefix + 'proj_x'], out[prefix + 'proj_y'] = res['proj_xy'][:, 0], res['proj_xy'][:, 1]
            out[prefix + 'segment'] = res['segment']
            out[prefix + 'extrapolated'] = res['extrapolated']
            out[prefix + 'ambiguous'] = res['ambiguous']
            disc = np.zeros(len(out), dtype=bool)
            for _, part in out.groupby(['config_source', 'obj'] if 'config_source' in out else ['obj'], sort=False):
                ordered = part.sort_values('tst', kind='mergesort')
                disc[out.index.get_indexer(ordered.index)] = projection_flags(
                    ordered[f'd:{role}_flow_along'], ordered[['x', 'y']])
            out[prefix + 'discontinuity'] = disc
    return out


def projection_summary(df_channels, spec):
    """Counts of extrapolated, ambiguous and discontinuous projections (C4 only)."""
    if spec[0] != 'C4':
        return []
    rows = []
    for role in ('cyclist', 'vehicle'):
        p = f'p:{role}_flow:'
        rows.append({'Flow path': role, 'Points': len(df_channels),
                     'Before start (tangent ray)': int((df_channels[p + 'extrapolated'] == -1).sum()),
                     'After end (tangent ray)': int((df_channels[p + 'extrapolated'] == 1).sum()),
                     'Ambiguous projection': int(df_channels[p + 'ambiguous'].sum()),
                     'Possible discontinuity': int(df_channels[p + 'discontinuity'].sum())})
    return rows


# =============================================================================
# FLOW REFERENCE CONSTRUCTION (same explicit choices as build_references_v02.py)
# =============================================================================

def spatial_samples(path, step=0.25):
    """Drop consecutive zero-distance points (geometric copy only) and resample every ~step metres."""
    a = np.asarray(path, dtype=float)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(a, axis=0), axis=1))]
    keep = np.r_[True, np.diff(s) > 1e-9]
    s, a = s[keep], a[keep]
    if len(s) < 2:
        raise ReferenceError('Stationary path has no direction')
    grid = np.linspace(0, s[-1], max(2, int(np.ceil(s[-1] / step)) + 1))
    return np.column_stack([np.interp(grid, s, a[:, j]) for j in range(2)])


def build_flow_reference(paths, direction, estimator='mean', station_step=0.5, minimum_support=3):
    """
    Smoothed spatial mean/median reference path from {encounter: (n, 2) prepared path}.

    Fixed cross-sections q = a·p (a = direction); each trajectory contributes one interpolated position per
    station inside its own observed coverage (no extrapolation), equally weighted per encounter. Stations with
    fewer than minimum_support contributors are dropped; the remainder must be contiguous. Local linear
    smoothing over at most five stations keeps the endpoints.
    """
    if estimator not in ESTIMATORS:
        raise ReferenceError(f"Unknown estimator {estimator!r}.")
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    prepared = []
    for cid, a in paths.items():
        a = np.asarray(a, dtype=float)
        q = a @ direction
        if np.min(np.diff(q)) < -1e-8:
            raise ReferenceError(f'Non-monotone alignment for encounter {cid}; inspect before fitting')
        keep = np.r_[True, np.diff(q) > 1e-9]
        prepared.append((q[keep], a[keep]))
    lo = min(q[0] for q, _ in prepared)
    hi = max(q[-1] for q, _ in prepared)
    stations = np.linspace(lo, hi, int(np.ceil((hi - lo) / station_step)) + 1)
    reducer = np.mean if estimator == 'mean' else np.median
    centres, counts = [], []
    for q0 in stations:
        values = [np.array([np.interp(q0, q, a[:, j]) for j in range(2)])
                  for q, a in prepared if q[0] <= q0 <= q[-1]]
        counts.append(len(values))
        centres.append(reducer(values, axis=0) if values else np.full(2, np.nan))
    centres, counts = np.array(centres), np.array(counts)
    supported = counts >= minimum_support
    indices = np.flatnonzero(supported)
    if len(indices) < 3 or np.any(np.diff(indices) != 1):
        raise ReferenceError('Reference support is insufficient or disconnected')
    stations, centres, counts = stations[supported], centres[supported], counts[supported]
    smooth = centres.copy()
    for i in range(1, len(centres) - 1):
        ix = np.arange(max(0, i - 2), min(len(centres), i + 3))
        design = np.column_stack([np.ones(len(ix)), stations[ix] - stations[i]])
        smooth[i] = np.linalg.lstsq(design, centres[ix], rcond=None)[0][0]
    if np.any(np.linalg.norm(np.diff(smooth, axis=0), axis=1) < 1e-9):
        raise ReferenceError('Degenerate reference segment')
    return {'directed_polyline_xy_m': smooth.tolist(), 'unsmoothed_reference_xy_m': centres.tolist(),
            'estimator': estimator, 'label': f'smoothed spatial {estimator} reference path',
            'support_counts': counts.tolist(), 'alignment_stations_m': stations.tolist(),
            'alignment_unit_direction': direction.tolist(), 'minimum_support': minimum_support,
            'smoothing': 'local linear fit over at most 5 spatial stations; endpoints retained',
            'low_support_warning': 'Inspect tails and transitions in support; counts are exported'}


def rebuild_flows_from_data(df, ref):
    """Refit mean and median flows on df with the reference's road directions (explicit rebuild only)."""
    u0 = np.asarray(ref['road_estimates']['cyclist_road']['unit_direction'], dtype=float)
    u1 = np.asarray(ref['road_estimates']['vehicle_approach_road']['unit_direction'], dtype=float)
    bisector = (u0 + u1) / np.linalg.norm(u0 + u1)
    paths = {o: {c: spatial_samples(g.sort_values('tst')[['x', 'y']].to_numpy())
                 for c, g in df[df['obj'] == o].groupby('config')} for o in (0, 1)}
    return ({'cyclist': build_flow_reference(paths[0], u0, 'mean'),
             'vehicle': build_flow_reference(paths[1], bisector, 'mean')},
            {'cyclist': build_flow_reference(paths[0], u0, 'median'),
             'vehicle': build_flow_reference(paths[1], bisector, 'median')})


# =============================================================================
# ONE ENCOUNTER: POINTS, MATRICES AND CELL EXPLANATION
# =============================================================================

def window_points(df_channels, config, objects, window_times):
    """Points of one window in the engine's order (time-major: tst, then obj), with index 0..n-1."""
    from .pdp_analysis import sort_points
    part = df_channels[(df_channels['config_source'] == config) & df_channels['obj'].isin(objects) &
                       df_channels['tst'].isin(window_times)]
    part = sort_points(part, ['tst', 'obj']).reset_index(drop=True)
    part.insert(0, 'point', np.arange(len(part)))
    return part


def point_label(row):
    role = OBJECT_ROLES.get(int(row['obj']), f"object {row['obj']}") if str(row['obj']).lstrip('-').isdigit() else str(row['obj'])
    return f"#{int(row['point'])} {role} (obj {row['obj']}) tst {row['tst']:g}"


def channel_matrix(points, column, rough=0):
    from .pdp_analysis import compute_inequality_matrix
    values = points[column].to_numpy(dtype=float)
    return compute_inequality_matrix(values, values, None, rough)


def changed_cells(matrix_a, matrix_b):
    """Off-diagonal cells whose relation differs; (i, j) and the two relation codes."""
    a, b = np.asarray(matrix_a), np.asarray(matrix_b)
    if a.shape != b.shape:
        raise ValueError("Matrices of different point sets cannot be compared cell by cell.")
    ii, jj = np.nonzero(a != b)
    return [(int(i), int(j), int(a[i, j]), int(b[i, j])) for i, j in zip(ii, jj) if i != j]


RELATION_WORDS = {0: 'row < column', 1: 'equal', 2: 'row > column'}


def relation_code(value_row, value_column, rough=0):
    d = value_column - value_row
    return 1 if abs(d) <= rough else (0 if d > rough else 2)


def projection_of(points_row, descriptor_id, ref, spec):
    """Foot point (x, y) of one point's projection for one descriptor, and a short description."""
    x, y = float(points_row['x']), float(points_row['y'])
    if descriptor_id == 'global_x':
        return None, "x coordinate (no reference geometry)"
    if descriptor_id == 'global_y':
        return None, "y coordinate (no reference geometry)"
    role, kind, _ = descriptor_id.split('_')
    if kind == 'road':
        return (float(points_row[f'p:{role}_road:proj_x']), float(points_row[f'p:{role}_road:proj_y'])), \
            f"straight projection on the {role} road centre line"
    flags = []
    extr = int(points_row[f'p:{role}_flow:extrapolated'])
    if extr:
        flags.append('tangent-ray extension ' + ('before the start' if extr < 0 else 'after the end'))
    if bool(points_row[f'p:{role}_flow:ambiguous']):
        flags.append('ambiguous projection')
    if bool(points_row[f'p:{role}_flow:discontinuity']):
        flags.append('possible discontinuity')
    text = f"closest projection on the {role} {spec[1]} flow path, segment {int(points_row[f'p:{role}_flow:segment'])}"
    return (float(points_row[f'p:{role}_flow:proj_x']), float(points_row[f'p:{role}_flow:proj_y'])), \
        text + (f" ({'; '.join(flags)})" if flags else "")


def explain_cell(points_a, points_b, i, j, column_a, column_b):
    """Rows describing points i and j and their relations under two channels (same point identities)."""
    for pts in (points_a, points_b):
        if len(pts) <= max(i, j):
            raise IndexError("Cell outside the matrix.")
    if not (points_a[['obj', 'tst']].astype(str).values == points_b[['obj', 'tst']].astype(str).values).all():
        raise ValueError("The two systems index different points; matrices cannot be linked.")
    rows = []
    for k in (i, j):
        rows.append({'Point': point_label(points_a.iloc[k]), 'x (m)': float(points_a.iloc[k]['x']),
                     'y (m)': float(points_a.iloc[k]['y']),
                     'A scalar': float(points_a.iloc[k][column_a]), 'B scalar': float(points_b.iloc[k][column_b])})
    return rows


# =============================================================================
# DISTANCE ACROSS SYSTEMS
# =============================================================================

def unique_pairs(n):
    return [(a, b) for a in range(n) for b in range(a + 1, n)]


def adjusted_rand(labels_a, labels_b):
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(labels_a, labels_b))


def spearman(a, b):
    from scipy.stats import spearmanr
    r = spearmanr(a, b).correlation
    return float(r) if r is not None and not math.isnan(r) else float('nan')


# =============================================================================
# EXPORT
# =============================================================================

def reference_metadata(ref, spec):
    """Reference geometry and metadata for exports (no absolute local paths beyond those recorded at build)."""
    system, estimator = spec
    meta = {'system': system, 'estimator': estimator, 'descriptor_order': list(DESCRIPTOR_ORDER[system]),
            'descriptor_columns': list(descriptor_columns(system)), 'identity': spec_identity(ref, spec)}
    if system == 'G2' or ref is None:
        return meta
    meta.update({k: ref.get(k) for k in ('reference_id', 'status', 'source_sha256', 'coordinate_frame',
                                         'construction', 'geometry_source', 'estimator_status', 'scope',
                                         'proposed_projection_convention', 'packaging')})
    meta['road_estimates'] = ref['road_estimates']
    if system == 'C4':
        meta['flows'] = flows_for(ref, estimator)
    return meta


def scalar_table(df_channels, spec, configs, objects, start, end):
    """Original identity, x/y, descriptor scalars and projection flags for export."""
    system = spec[0]
    part = df_channels[df_channels['config_source'].isin(configs) & df_channels['obj'].isin(objects) &
                       (df_channels['tst'] >= start) & (df_channels['tst'] <= end)]
    cols = ['config_source', 'obj', 'tst', 'x', 'y']
    out = part[cols].copy()
    out.insert(2, 'role', out['obj'].map(OBJECT_ROLES))
    for d in DESCRIPTOR_ORDER[system]:
        out[d] = part[descriptor_column(d)].to_numpy()
    out = out.join(part[[c for c in part.columns if c.startswith('p:')]])
    return out.sort_values(['config_source', 'obj', 'tst'], kind='mergesort')
