"""Build inspectable, dataset-derived candidate PDP references. Does not edit the app.

Run with the existing tennis_infer_rf/.venv Python. NumPy, Matplotlib and Pillow are needed.
Roads come from the existing diagram. Mean and median flow estimators are our explicit
choices, not a verified reproduction of CB's code.
"""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import os

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/pdp_reference_matplotlib')
import numpy as np

HERE = Path(__file__).resolve().parent
COURSE = next(p for p in HERE.parents if (p / '04Exercises').is_dir())
SOURCE = COURSE / '04Exercises/AJ2627/ExercisesForStudents/ForStudents/Data/Data/CSV_PDP input/A2_detailed_dataset.csv'


def spatial_samples(a, step=0.25):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(a, axis=0), axis=1))]
    keep = np.r_[True, np.diff(s) > 1e-9]
    s, a = s[keep], a[keep]
    if len(s) < 2:
        raise ValueError('Stationary path has no direction')
    grid = np.linspace(0, s[-1], max(2, int(np.ceil(s[-1] / step)) + 1))
    return np.column_stack([np.interp(grid, s, a[:, j]) for j in range(2)])


def road_geometry():
    """Read road centres from the existing basemap, not observed vehicle headings."""
    from zipfile import ZipFile
    from io import BytesIO
    from PIL import Image
    deck = COURSE / '04Exercises/AJ2627/ExercisesForStudents/UFO_data/A2/PDP curve/curve descriptor.pptx'
    with ZipFile(deck) as z:
        background = Image.open(BytesIO(z.read('ppt/media/image1.png'))).convert('RGB')
    a = np.array(background).astype(float)
    road = ((a[:, :, 1] > a[:, :, 0] + 4) & (a[:, :, 1] > a[:, :, 2] + 2)
            & (a[:, :, 0] > 70) & (a[:, :, 0] < 180))
    h, w = a.shape[:2]
    # Rounded metre labels in slide 6: top y=62.45 m, right edge x=72.12 m.
    scales = np.array([62.45 / h, 72.12 / w])
    scale = float(scales.mean())
    approach, main = [], []
    for y in range(20, 851, 10):
        x = np.flatnonzero(road[y]); x = x[(x > 550) & (x < 1050)]
        approach.append([y, (x.min() + x.max()) / 2])
    for x in list(range(20, 551, 10)) + list(range(1000, 1151, 10)):
        y = np.flatnonzero(road[:, x]); y = y[y > 850]
        if y.min() <= 851 or y.max() >= h - 10:
            raise ValueError('Clipped road edges cannot define the centre')
        main.append([x, (y.min() + y.max()) / 2])
    approach, main = np.array(approach), np.array(main)
    ma, ba = np.polyfit(approach[:, 0], approach[:, 1], 1)  # x = ma*y + ba
    mm, bm = np.polyfit(main[:, 0], main[:, 1], 1)  # y = mm*x + bm
    px = (ma * bm + ba) / (1 - ma * mm)
    py = mm * px + bm
    origin = np.array([px * scale, (h - py) * scale])
    def record(vector):
        vector = np.array(vector); vector /= np.linalg.norm(vector)
        return {'origin_xy_m': origin.tolist(), 'unit_direction': vector.tolist(),
                'meaning': 'road-centre direction recovered from existing basemap edges',
                'status': 'diagram-derived, approximate metric registration; not surveyed geometry'}
    roads = (record([1., -mm]), record([ma, -1.]))
    metadata = {'source_pptx': str(deck), 'source_part': 'ppt/media/image1.png',
                'road_diagram_slide': 4, 'metric_label_slide': 6,
                'pixel_to_metre_scale': scale, 'independent_scale_estimates': scales.tolist(),
                'pixel_to_metre_transform': 'x=scale*pixel_x; y=scale*(image_height-pixel_y)',
                'calibration_note': 'Scale inferred from rounded labels, consistent with map extent; approximate.',
                'approach_fit_x_from_y': [float(ma), float(ba)],
                'main_fit_y_from_x': [float(mm), float(bm)]}
    return roads, metadata, background, [0, w * scale, 0, h * scale]


def mean_flow(paths, direction, statistic="mean"):
    """Align by a fixed spatial projection, never by timestamp or path fraction."""
    prepared = []
    for cid, a in paths.items():
        q = a @ direction
        if np.min(np.diff(q)) < -1e-8:
            raise ValueError(f'Non-monotone alignment for encounter {cid}; inspect before fitting')
        keep = np.r_[True, np.diff(q) > 1e-9]
        prepared.append((q[keep], a[keep]))
    lo = min(q[0] for q, _ in prepared)
    hi = max(q[-1] for q, _ in prepared)
    stations = np.linspace(lo, hi, int(np.ceil((hi - lo) / 0.5)) + 1)
    means, counts = [], []
    for q0 in stations:
        # Interpolate within each path's observed coverage, without extrapolation.
        values = [np.array([np.interp(q0, q, a[:, j]) for j in range(2)])
                  for q, a in prepared if q[0] <= q0 <= q[-1]]
        counts.append(len(values))
        reducer = np.mean if statistic == "mean" else np.median
        means.append(reducer(values, axis=0) if values else np.full(2, np.nan))
    means, counts = np.array(means), np.array(counts)
    supported = counts >= 3
    indices = np.flatnonzero(supported)
    if len(indices) < 3 or np.any(np.diff(indices) != 1):
        raise ValueError('Reference support is insufficient or disconnected')
    stations, means, counts = stations[supported], means[supported], counts[supported]
    # Local linear smoothing of the spatial mean reduces changing-coverage kinks.
    # Preserve endpoints; no smoothing of the original trajectory data.
    smooth = means.copy()
    for i in range(1, len(means) - 1):
        ix = np.arange(max(0, i - 2), min(len(means), i + 3))
        design = np.column_stack([np.ones(len(ix)), stations[ix] - stations[i]])
        smooth[i] = np.linalg.lstsq(design, means[ix], rcond=None)[0][0]
    if np.any(np.linalg.norm(np.diff(smooth, axis=0), axis=1) < 1e-9):
        raise ValueError('Degenerate reference segment')
    return {'directed_polyline_xy_m': smooth.tolist(), 'unsmoothed_reference_xy_m': means.tolist(),
            'estimator': statistic, 'label': f'smoothed spatial {statistic} reference path',
            'support_counts': counts.tolist(), 'alignment_stations_m': stations.tolist(),
            'alignment_unit_direction': direction.tolist(), 'minimum_support': 3,
            'smoothing': 'local linear fit over at most 5 spatial stations; endpoints retained',
            'low_support_warning': 'Inspect tails and transitions in support; counts are exported'}


def main():
    rows = defaultdict(list)
    with SOURCE.open() as f:
        for r in csv.DictReader(f):
            rows[int(r['poiID']), int(r['conID'])].append(
                (int(r['tstID']), float(r['x']), float(r['y'])))
    assert len(rows) == 52
    raw = {o: {} for o in (0, 1)}
    for (o, cid), values in rows.items():
        assert len(values) == 137 and sorted(v[0] for v in values) == list(range(137))
        raw[o][cid] = np.array(sorted(values))[:, 1:]
    paths = {o: {c: spatial_samples(a) for c, a in raw[o].items()} for o in (0, 1)}
    (cycle_road, vehicle_road), geometry_source, background, extent = road_geometry()
    u0, u1 = np.array(cycle_road['unit_direction']), np.array(vehicle_road['unit_direction'])
    # A fixed bisector gives a spatial station increasing on both parts of the turn.
    turn_station = u0 + u1
    turn_station /= np.linalg.norm(turn_station)
    flows = {'cyclist': mean_flow(paths[0], u0), 'vehicle': mean_flow(paths[1], turn_station)}
    median_flows = {'cyclist': mean_flow(paths[0], u0, 'median'),
                    'vehicle': mean_flow(paths[1], turn_station, 'median')}
    references = {
        'status': 'v02: road geometry corrected from existing diagram; flow estimator explicitly defined; app integration pending',
        'source_csv': str(SOURCE), 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'coordinate_frame': 'original local x/y, metres; geographical CRS not inferred',
        'construction': 'diagram-derived road centres; 0.25 m path resampling; equal encounter weights; fixed spatial alignment',
        'geometry_source': geometry_source,
        'road_estimates': {'cyclist_road': cycle_road, 'vehicle_approach_road': vehicle_road},
        'flows': flows, 'median_flows_sensitivity_comparison': median_flows,
        'estimator_status': 'mean default and median comparison are explicit choices for this implementation; no estimator verified from CB',
        'descriptor_order': {
            'G2': ['global_x', 'global_y'],
            'R2': ['cyclist_road_along', 'vehicle_road_along'],
            'R4': ['cyclist_road_along', 'vehicle_road_along', 'cyclist_road_left', 'vehicle_road_left'],
            'C4': ['cyclist_flow_along', 'vehicle_flow_along', 'cyclist_flow_left', 'vehicle_flow_left']},
        'proposed_projection_convention': {
            'straight': 'independent dot products against unit directions; normals positive left',
            'curved': 'closest projection onto polyline, with first and last tangent rays for out-of-range coverage',
            'longitudinal': 'signed arc length from first stored vertex; tangent-ray extensions may be negative',
            'lateral': 'signed cross(unit segment tangent, point minus projected point), positive left',
            'ties': 'smallest longitudinal value, then smallest segment index at a shared vertex',
            'all_points': 'evaluate both objects against the same complete descriptor set',
            'extension_warning': 'flag use of extrapolated tangent rays; do not present them as observed flow'},
        'scope': 'descriptive exploration on these 26 encounters; not held-out validation or CB reproduction'}
    (HERE / 'candidate_references.json').write_text(json.dumps(references, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharex=True, sharey=True)
    for ax in axes:
        for o, colour in [(0, '#3675a9'), (1, '#d47726')]:
            for a in raw[o].values():
                ax.plot(*a.T, color=colour, alpha=.20, lw=.8)
        ax.imshow(background, extent=extent, origin='upper', alpha=.45, zorder=-2)
        ax.set_aspect('equal'); ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    for line, colour, label in [(cycle_road, '#174f85', 'Road centre, lower road'),
                                (vehicle_road, '#a64c00', 'Road centre, vehicle approach')]:
        origin, u = np.array(line['origin_xy_m']), np.array(line['unit_direction'])
        a = origin + np.array([-60., 60.])[:, None] * u
        axes[0].plot(*a.T, color=colour, lw=2.5, label=label)
    for role, colour in [('cyclist', '#174f85'), ('vehicle', '#a64c00')]:
        a = np.array(flows[role]['directed_polyline_xy_m'])
        axes[1].plot(*a.T, color=colour, lw=3, label=f'{role.capitalize()} smoothed mean')
        median = np.array(median_flows[role]['directed_polyline_xy_m'])
        axes[1].plot(*median.T, color=colour, lw=1.8, ls='--', label=f'{role.capitalize()} smoothed median')
        low = np.array(flows[role]['support_counts']) < 10
        axes[1].scatter(*a[low].T, color=colour, s=18, marker='x')
    # Historical overlay only; no dependency on the superseded reference file.
    line = {'origin_xy_m': [35.91095416382577, 17.152202920117332], 'unit_direction': [0.0912947209636204, -0.9958239171280105]}
    a = np.array(line['origin_xy_m']) + np.array([-60., 60.])[:, None] * np.array(line['unit_direction'])
    axes[0].plot(*a.T, color='#b61c2b', lw=1.5, ls='--', label='Previous vehicle-road estimate, rejected')
    axes[0].set_title('R2: road geometry from the existing basemap')
    axes[1].set_title('C4: smoothed mean and median reference paths')
    for ax in axes:
        ax.set_xlim(18, 67); ax.set_ylim(1, 25); ax.legend(loc='upper right', fontsize=8)
    fig.suptitle('Road geometry and observed flow paths are different reference choices', fontsize=13)
    fig.tight_layout(); fig.savefig(HERE / 'candidate_references.png', dpi=160)
    print(json.dumps({'road_directions': [u0.tolist(), u1.tolist()],
        'acute_road_angle_degrees': float(np.degrees(np.arccos(abs(u0 @ u1)))),
        'mean_median_separation_m': {k: {'median': float(np.median(np.linalg.norm(np.array(flows[k]['directed_polyline_xy_m']) - np.array(median_flows[k]['directed_polyline_xy_m']), axis=1))), 'max': float(np.max(np.linalg.norm(np.array(flows[k]['directed_polyline_xy_m']) - np.array(median_flows[k]['directed_polyline_xy_m']), axis=1)))} for k in flows},
        'flow_vertices': {k: len(v['support_counts']) for k, v in flows.items()}}, indent=2))


if __name__ == '__main__':
    main()
