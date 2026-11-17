"""
Point size and clustering (Visual Exploration).

Links the visible group structure of drawn points to hierarchical clustering in the displayed
two-dimensional data space. One point is either the measured position of one configuration/object
at one time stamp, or the mean position of one configuration/object over a time interval.

All calculations use the same points and the same Euclidean distance matrix:
- the hierarchy (single, complete or average linkage),
- the cut of that hierarchy at distance d (the point diameter),
- the connected components of circles with diameter d (points linked when their distance <= d),
- the mean silhouette scores of the candidate partitions.

No interpolation, sampling or synthetic data is used here.
"""
import hashlib

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage, leaves_list
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score

MAX_POINTS = 1500          # explicit safe limit for the n x n distance matrix and the drawing
DEFAULT_MAX_CLUSTERS = 10  # initial upper end of the silhouette search range
LINKAGES = ('single', 'complete', 'average')
LINKAGE_LABELS = {'single': 'Single linkage', 'complete': 'Complete linkage', 'average': 'Average linkage'}
MODES = ('time_point', 'interval_mean')
MODE_LABELS = {'time_point': 'Position at one time stamp (measured)',
               'interval_mean': 'Mean position over the selected interval (aggregated)'}

HELP = (
    "This view connects visualisation and clustering. Reducing the point diameter can reveal groups that "
    "were hidden by overlapping symbols. The point coordinates themselves do not change.\n\n"
    "With single linkage, groups connected through overlapping or touching circles match the clusters "
    "obtained by cutting the dendrogram at the same distance. Complete and average linkage can produce "
    "different groups at that distance.\n\n"
    "Silhouette scores compare cohesion within clusters with separation between clusters. They are "
    "calculated in the displayed two-dimensional data space. A higher score favours an evaluated partition "
    "under these settings; it does not establish one universally correct number of clusters."
)


def build_points(df, configs, objects, start_time, end_time, mode, time_stamp=None):
    """
    Return (points, report).

    points: DataFrame with point_id, config, obj, x, y, tst_info, n_obs, source; one row per point,
            in a stable order (configuration order as given, then object order as given).
    report: dict with counts of selected rows, rows with missing/invalid coordinates and
            configuration/object pairs without a usable position.
    Coincident points are kept as separate points.
    """
    sel = df[df['config_source'].isin(configs) & df['obj'].isin(objects)
             & (df['tst'] >= start_time) & (df['tst'] <= end_time)].copy()
    sel['x'] = pd.to_numeric(sel['x'], errors='coerce')
    sel['y'] = pd.to_numeric(sel['y'], errors='coerce')
    finite = np.isfinite(sel['x'].to_numpy(float)) & np.isfinite(sel['y'].to_numpy(float))
    report = {'selected_rows': int(len(sel)), 'invalid_rows': int((~finite).sum()),
              'pairs_requested': len(configs) * len(objects), 'pairs_without_position': 0}
    valid = sel[finite]
    rows = []
    for config in configs:
        for obj in objects:
            pair = valid[(valid['config_source'] == config) & (valid['obj'] == obj)]
            if mode == 'time_point':
                pair = pair[pair['tst'] == time_stamp]
                if pair.empty:
                    report['pairs_without_position'] += 1
                    continue
                # Several measurements at the same time stamp stay separate points
                for i, (_, r) in enumerate(pair.iterrows()):
                    suffix = f' #{i + 1}' if len(pair) > 1 else ''
                    rows.append({'point_id': f'{config} | obj {obj} | tst {time_stamp:g}{suffix}',
                                 'config': config, 'obj': obj, 'x': float(r['x']), 'y': float(r['y']),
                                 'tst_info': f'tst {time_stamp:g}', 'n_obs': 1, 'source': 'measured'})
            else:
                if pair.empty:
                    report['pairs_without_position'] += 1
                    continue
                rows.append({'point_id': f'{config} | obj {obj} | mean tst {start_time:g}-{end_time:g}',
                             'config': config, 'obj': obj,
                             'x': float(pair['x'].mean()), 'y': float(pair['y'].mean()),
                             'tst_info': f'mean of {len(pair)} measured positions, tst {pair["tst"].min():g}-'
                                         f'{pair["tst"].max():g}',
                             'n_obs': int(len(pair)), 'source': 'aggregated (mean)'})
    points = pd.DataFrame(rows, columns=['point_id', 'config', 'obj', 'x', 'y', 'tst_info', 'n_obs', 'source'])
    return points, report


def coordinate_warning(x, y):
    """Return a warning when the coordinates look like geographic degrees rather than metres."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) == 0:
        return None
    if (np.all(np.abs(x) <= 180) and np.all(np.abs(y) <= 90)
            and np.ptp(x) < 1 and np.ptp(y) < 1 and (np.any(x % 1 != 0) or np.any(y % 1 != 0))):
        return ("These coordinates look like geographic degrees. Euclidean distances in degrees are not "
                "distances in metres; project the data to a metric coordinate system first.")
    return None


def data_signature(points, method, max_clusters):
    """Identity of everything the hierarchy and silhouette curve depend on (not the diameter)."""
    h = hashlib.sha256()
    h.update('\n'.join(points['point_id'].astype(str)).encode())
    h.update(np.ascontiguousarray(points[['x', 'y']].to_numpy(float)).tobytes())
    h.update(f'{method}|{max_clusters}'.encode())
    return h.hexdigest()


def compute_hierarchy(xy, method):
    """Euclidean condensed distances, square matrix and linkage matrix for the same points."""
    xy = np.asarray(xy, float)
    if method not in LINKAGES:
        raise ValueError(f'Unknown linkage: {method}')
    condensed = pdist(xy, metric='euclidean')
    return condensed, squareform(condensed), linkage(condensed, method=method)


def cut_at_distance(Z, d):
    """Cluster labels after cutting the hierarchy at distance d (merges with height <= d are kept)."""
    return fcluster(Z, t=d, criterion='distance')


def circle_components(D, d):
    """Connected components of circles with diameter d: points are linked when their distance <= d."""
    n_comp, labels = connected_components(D <= d, directed=False)
    return n_comp, labels


def canonical_labels(labels):
    """Relabel clusters 0, 1, 2, ... in order of their first point, so colours and ids are stable."""
    mapping, out = {}, np.empty(len(labels), int)
    for i, lab in enumerate(labels):
        out[i] = mapping.setdefault(lab, len(mapping))
    return out


def same_partition(a, b):
    return bool(np.array_equal(canonical_labels(a), canonical_labels(b)))


def mean_silhouette(D, labels):
    """Mean silhouette score, or None when undefined (1 cluster, or as many clusters as points)."""
    k = len(np.unique(labels))
    if k < 2 or k > len(labels) - 1:
        return None
    # sklearn convention: a point alone in its cluster scores 0
    return float(silhouette_score(D, labels, metric='precomputed'))


def reachable_cuts(Z):
    """
    Cluster counts reachable by a distance cut, each with the cut distance that produces it.
    With tied merge heights some counts cannot be reached by any cut; those are absent.
    """
    n = Z.shape[0] + 1
    heights = np.unique(Z[:, 2])
    cuts = {n: 0.0 if heights[0] > 0 else None}
    for h in heights:
        k = n - int(np.sum(Z[:, 2] <= h))
        cuts[k] = float(h)
    return {k: t for k, t in cuts.items() if t is not None}


def silhouette_curve(D, Z, max_clusters, extra_k=None):
    """
    Mean silhouette for reachable partitions with 2..max_clusters clusters (and extra_k, the current
    count, when it is valid). Returns a list of dicts: k, cut, score (None when undefined).
    """
    n = D.shape[0]
    cuts = reachable_cuts(Z)
    wanted = {k for k in cuts if 2 <= k <= min(max_clusters, n - 1)}
    if extra_k is not None and extra_k in cuts and 2 <= extra_k <= n - 1:
        wanted.add(extra_k)
    curve = []
    for k in sorted(wanted):
        labels = cut_at_distance(Z, cuts[k])
        curve.append({'k': k, 'cut': cuts[k], 'score': mean_silhouette(D, labels)})
    return curve


def dendrogram_segments(Z, labels, d):
    """
    Plotly-ready dendrogram segments. Leaves are placed in scipy's leaf order. A merge below or at the
    cut distance d gets the canonical cluster id of its leaves; merges above d get None (grey).
    Returns (segments, leaf_order) with segments as dicts: xs, ys, cluster.
    """
    n = Z.shape[0] + 1
    order = leaves_list(Z)
    x_pos = {int(leaf): 10.0 * i + 5 for i, leaf in enumerate(order)}
    height = {i: 0.0 for i in range(n)}
    first_leaf = {i: i for i in range(n)}
    segments = []
    for j, (a, b, h, _) in enumerate(Z):
        a, b, node = int(a), int(b), n + j
        x_pos[node] = (x_pos[a] + x_pos[b]) / 2
        height[node] = float(h)
        first_leaf[node] = first_leaf[a]
        cluster = int(labels[first_leaf[node]]) if h <= d else None
        segments.append({'xs': [x_pos[a], x_pos[a], x_pos[b], x_pos[b]],
                         'ys': [height[a], h, h, height[b]], 'cluster': cluster})
    return segments, [int(i) for i in order]
