"""
Clustering analysis module for spatiotemporal trajectories.

This module provides three distance-based clustering methods:
1. Feature-based clustering (trajectory statistics)
2. Chamfer distance clustering (spatial similarity)
3. DTW clustering (spatiotemporal similarity)
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist, euclidean
from scipy.cluster.hierarchy import dendrogram, linkage

from .common import render_interactive_chart


# =============================================================================
# TRAJECTORY FEATURE EXTRACTION
# =============================================================================

FEATURE_COLUMN_LABELS = {
    'total_distance': 'Total Distance (m)',
    'duration': 'Duration (tst units)',
    'avg_speed': 'Avg Speed (m per tst unit)',
    'net_displacement': 'Net Displacement (m)',
    'sinuosity': 'Sinuosity (ratio)',
    'bbox_area': 'Bbox Area (m²)',
    'avg_direction': 'Avg Direction (rad)',
    'max_speed': 'Max Speed (m per tst unit)',
    'min_speed': 'Min Speed (m per tst unit)'
}
# Feature order in the selector; the default selection stays the original eight (min_speed is optional)
ALL_FEATURES = ['total_distance', 'duration', 'avg_speed', 'net_displacement', 'sinuosity', 'bbox_area',
                'avg_direction', 'max_speed', 'min_speed']
DEFAULT_FEATURES = ALL_FEATURES[:8]
FEATURE_SELECTOR_LABELS = {
    'total_distance': 'Total Distance', 'duration': 'Duration', 'avg_speed': 'Average Speed',
    'net_displacement': 'Net Displacement', 'sinuosity': 'Sinuosity (Path Efficiency)',
    'bbox_area': 'Bounding Box Area', 'avg_direction': 'Average Direction', 'max_speed': 'Maximum Speed',
    'min_speed': 'Minimum Speed',
}
# Displayed decimals per feature (display only: values stay numeric at full precision)
FEATURE_DISPLAY_DECIMALS = {'sinuosity': 6}
DEFAULT_DISPLAY_DECIMALS = 2


def format_features_dataframe(features_df):
    """
    Features with units in the column names. Values stay numeric at full precision; rounding for display
    is done by feature_column_config (sinuosity six decimals, other features two).
    """
    return features_df.rename(columns=FEATURE_COLUMN_LABELS)


def feature_display_format(feature):
    return f"%.{FEATURE_DISPLAY_DECIMALS.get(feature, DEFAULT_DISPLAY_DECIMALS)}f"


def feature_column_config(features_df):
    """st.dataframe column_config for the formatted features table (display formatting only)."""
    return {FEATURE_COLUMN_LABELS.get(f, f): st.column_config.NumberColumn(
                FEATURE_COLUMN_LABELS.get(f, f), format=feature_display_format(f))
            for f in features_df.columns}


def features_csv(features_df):
    """Raw numeric feature values (full precision), one row per trajectory ID."""
    return features_df.rename_axis('Trajectory_ID').to_csv(float_format='%.17g')


# =============================================================================
# COMPUTED-RUN IDENTITY (results and exports belong to the inputs they were computed from)
# =============================================================================

STANDARDISATION = {'Features (Euclidean)': 'StandardScaler (z-score per feature)'}
DISTANCE_METRICS = {'Features (Euclidean)': 'Euclidean', 'Spatial (Chamfer)': 'Chamfer',
                    'Spatiotemporal (DTW)': 'DTW'}

# Stable internal identifiers (saved settings, tests) -> canonical user-facing labels
METHOD_IDS = ['Features (Euclidean)', 'Spatial (Chamfer)', 'Spatiotemporal (DTW)']
METHOD_LABELS = {
    'Features (Euclidean)': 'Feature-based clustering (Euclidean)',
    'Spatial (Chamfer)': 'Spatial clustering (Chamfer)',
    'Spatiotemporal (DTW)': 'Spatiotemporal clustering (DTW)',
}

# The single place that decides which linkage each approach uses (scipy method names).
# Ward needs Euclidean geometry, so it is only used with Euclidean distances between standardised
# feature vectors. Chamfer and DTW dissimilarities are not Euclidean in general (using Euclidean
# point-to-point costs does not make the trajectory dissimilarity Euclidean), so they use average
# linkage, which works directly with any supplied dissimilarity.
LINKAGE_BY_METHOD = {
    'Features (Euclidean)': 'ward',
    'Spatial (Chamfer)': 'average',
    'Spatiotemporal (DTW)': 'average',
}
LINKAGE_LABELS = {'ward': 'Ward linkage', 'average': 'average linkage'}


def linkage_for(method):
    return LINKAGE_BY_METHOD[method]


def method_label(method, with_linkage=False):
    label = METHOD_LABELS.get(method, method)
    if with_linkage and method in LINKAGE_BY_METHOD:
        label += ', ' + LINKAGE_LABELS[linkage_for(method)]
    return label


# Concise explanations, reused wherever the terms recur (shown behind question-mark help icons)
HELP = {
    'hierarchical': ("Starts with each trajectory in its own group and repeatedly merges groups, producing a "
                     "hierarchy that can be displayed as a dendrogram."),
    'linkage': "The rule used to decide which groups to merge at each step.",
    'ward': ("Merges the groups that produce the smallest increase in variation within groups. Used here with "
             "Euclidean distances between standardised trajectory features."),
    'average': ("Compares two groups using the mean dissimilarity between all pairs of trajectories across those "
                "groups. Used here with Chamfer and DTW dissimilarities."),
    'Features (Euclidean)': ("Compares selected summary properties, such as travelled distance, speed or "
                             "sinuosity, using Euclidean distances between their standardised values "
                             "(z-scores: each feature has mean 0 and standard deviation 1 across the selected "
                             "trajectories)."),
    'Spatial (Chamfer)': ("Compares the spatial locations occupied by trajectories using nearest-point distances "
                          "in both directions: for every position, the distance to the nearest position of the "
                          "other trajectory, averaged per direction and then over both directions. It does not "
                          "use observation order or timing."),
    'Spatiotemporal (DTW)': ("Dynamic Time Warping compares ordered trajectories by allowing observations to be "
                             "aligned at different positions in their sequences. Each aligned pair costs the "
                             "Euclidean distance between (x, y, tst): the timestamp enters directly as a third, "
                             "unscaled coordinate, so one tst step weighs as much as one coordinate unit (one "
                             "metre for metric coordinates). The DTW value is the sum of the costs along the "
                             "cheapest monotone alignment, so it also grows with trajectory length."),
}


def linkage_help(method):
    return HELP[linkage_for(method)]


def validate_distance_matrix(distance_matrix, trajectory_ids=None, atol=1e-9):
    """
    Check that a square dissimilarity matrix can be condensed for scipy.cluster.hierarchy.linkage:
    matching dimensions and trajectory order, finite, non-negative, symmetric, zero diagonal.
    Tolerance (relative to the largest entry) covers floating-point round-off only.
    Returns a float copy with exact symmetry and an exact zero diagonal.
    """
    D = np.asarray(distance_matrix, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"Distance matrix must be square, got shape {D.shape}.")
    n = D.shape[0]
    if trajectory_ids is not None and len(trajectory_ids) != n:
        raise ValueError(f"Distance matrix has {n} rows but {len(trajectory_ids)} trajectory IDs.")
    if trajectory_ids is not None and len(set(map(str, trajectory_ids))) != n:
        raise ValueError("Trajectory IDs are not unique.")
    if n < 2:
        raise ValueError("At least two trajectories are needed for hierarchical clustering.")
    if not np.all(np.isfinite(D)):
        raise ValueError("Distance matrix contains missing or infinite values.")
    tol = atol * max(1.0, float(np.abs(D).max()))
    if (D < -tol).any():
        raise ValueError("Distance matrix contains negative values.")
    if np.abs(D - D.T).max() > tol:
        raise ValueError("Distance matrix is not symmetric.")
    if np.abs(np.diag(D)).max() > tol:
        raise ValueError("Distance matrix diagonal is not zero.")
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    return np.clip(D, 0.0, None)


def build_linkage(distance_matrix, method, trajectory_ids=None):
    """
    The hierarchy of a run: scipy linkage on the condensed form of the validated pairwise matrix,
    with the linkage that LINKAGE_BY_METHOD assigns to the approach. For Ward the matrix holds Euclidean
    distances between standardised feature vectors, for which Ward on the condensed distances equals
    Ward on the vectors. Never passes the square matrix itself (that would cluster distance profiles).
    """
    from scipy.spatial.distance import squareform
    D = validate_distance_matrix(distance_matrix, trajectory_ids)
    return linkage(squareform(D, checks=True), method=linkage_for(method))


def cut_hierarchy(linkage_matrix, n_clusters):
    """Partition into (at most) n_clusters from an existing hierarchy; labels 1..k."""
    from scipy.cluster.hierarchy import fcluster
    return fcluster(linkage_matrix, int(n_clusters), criterion='maxclust')


def evaluate_cluster_counts(distance_matrix, linkage_matrix, max_clusters=10):
    """
    Score cluster counts k = 2..max on cuts of the SAME hierarchy shown to the user.
    Within-cluster dispersion = sum over clusters of (sum of within-cluster pairwise dissimilarities) / (2 n_c);
    silhouette uses the original pairwise dissimilarities (precomputed).
    Returns (recommended_k, plot_data), or (None, plot_data) when there are too few trajectories.
    """
    D = np.asarray(distance_matrix, dtype=float)
    n = len(D)
    max_k = min(max_clusters, n - 1)
    k_values, inertias, sils = [], [], []
    for k in range(2, max_k + 1):
        labels = cut_hierarchy(linkage_matrix, k)
        inertia = 0.0
        for c in np.unique(labels):
            m = labels == c
            inertia += D[m][:, m].sum() / (2 * m.sum())
        n_found = len(np.unique(labels))
        sil = (silhouette_score(D, labels, metric='precomputed') if 2 <= n_found <= n - 1 else np.nan)
        k_values.append(k)
        inertias.append(float(inertia))
        sils.append(float(sil))
    plot_data = {'k_values': k_values, 'inertias': inertias, 'silhouette_scores': sils}
    if len(k_values) < 3:
        plot_data['optimal_k'] = None
        return None, plot_data
    norm = np.array(inertias)
    norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-12)
    angles = []
    for i in range(1, len(norm) - 1):
        v1 = np.array([-1.0, norm[i - 1] - norm[i]])
        v2 = np.array([1.0, norm[i + 1] - norm[i]])
        angles.append(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)))
    # Sharpest bend = smallest angle between the two neighbouring segments
    optimal_k = k_values[int(np.argmin(angles)) + 1]
    sil_arr = np.array(sils)
    if np.isfinite(sil_arr).any():
        idx = k_values.index(optimal_k)
        if not (sil_arr[idx] >= 0.25):
            best = int(np.nanargmax(sil_arr))
            if sil_arr[best] > 0.25:
                optimal_k = k_values[best]
    plot_data['optimal_k'] = optimal_k
    return optimal_k, plot_data


def data_fingerprint(df):
    """Content identity of the columns clustering uses (not the filename): SHA-256 over their values."""
    import hashlib
    cols = [c for c in ('config_source', 'obj', 'tst', 'x', 'y', 'X', 'Y') if c in df.columns]
    hashes = pd.util.hash_pandas_object(df[cols], index=False).to_numpy()
    return hashlib.sha256(hashes.tobytes() + ','.join(cols).encode()).hexdigest()


def run_signature(fingerprint, method, configs, objects, start, end, features=None):
    """Everything that determines a clustering result (cluster count excluded: it only cuts the hierarchy)."""
    return (fingerprint, method,
            tuple(features or ()) if method == 'Features (Euclidean)' else (),
            tuple(sorted(map(str, configs))), tuple(sorted(map(str, objects))),
            float(start), float(end),
            STANDARDISATION.get(method, 'none'), linkage_for(method))


def run_id(signature):
    import hashlib
    return hashlib.sha256(repr(signature).encode()).hexdigest()[:10]


def run_file_tag(run, n_clusters):
    """Distinct, informative filename part: method/feature, linkage, cluster count and run identifier."""
    if run['method'] == 'Features (Euclidean)':
        what = 'features-' + '+'.join(run['features'])
    else:
        what = DISTANCE_METRICS[run['method']].lower()
    return f"{what}_{run['linkage']}_{n_clusters}clusters_run{run['run_id']}"


def run_configuration(run, n_requested, labels):
    """Configuration export of a computed run and its current cluster cut (not live widget values)."""
    lengths = run['trajectory_lengths']
    return {key: str(value) for key, value in {
        'Run_ID': run['run_id'],
        'Computed_At': run['computed_at'],
        'Dataset_File': run['filename'],
        'Dataset_Content_SHA256': run['fingerprint'],
        'Clustering_Method': method_label(run['method']),
        'Distance_Metric': DISTANCE_METRICS[run['method']],
        'Selected_Features': ', '.join(run['features']) if run['method'] == 'Features (Euclidean)' else 'not applicable',
        'Standardisation': STANDARDISATION.get(run['method'], 'none'),
        'Linkage_Method': LINKAGE_LABELS[run['linkage']].replace(' linkage', '').capitalize(),
        'Requested_Number_of_Clusters': int(n_requested),
        'Actual_Number_of_Clusters': int(len(np.unique(labels))),
        'Configuration_IDs': ', '.join(map(str, run['configs'])),
        'Object_IDs': ', '.join(map(str, run['objects'])),
        'Start_Observation_tst': f"{run['start']:g}",
        'End_Observation_tst': f"{run['end']:g}",
        'Time_Unit': ('tst as given in the dataset (observation index, not necessarily seconds); '
                      'speeds and durations use tst units'),
        'Number_of_Trajectories': len(run['trajectory_ids']),
        'Positions_per_Trajectory': (f"{min(lengths)}" if min(lengths) == max(lengths)
                                     else f"{min(lengths)}–{max(lengths)}"),
        'Total_Positions': int(sum(lengths)),
    }.items()}


def segment_speeds(traj_df, x_col='x', y_col='y'):
    """
    Speeds of valid segments: distance between consecutive positions (sorted by tst) divided by their
    elapsed tst. Segments with missing values or non-positive elapsed time (duplicate timestamps) are
    excluded, never replaced by an invented small time step. Genuine zero-speed segments are kept.
    """
    t = traj_df[['tst', x_col, y_col]].sort_values('tst', kind='mergesort').to_numpy(dtype=float)
    if len(t) < 2:
        return np.array([])
    dt = np.diff(t[:, 0])
    dist = np.hypot(np.diff(t[:, 1]), np.diff(t[:, 2]))
    valid = np.isfinite(dt) & np.isfinite(dist) & (dt > 0)
    return dist[valid] / dt[valid]


def extract_trajectory_features(traj_df):
    """
    Extract statistical features from a single trajectory (observations sorted by tst).
    Speeds and durations are in coordinate units per tst unit (m/s only when tst is in seconds).
    min_speed / max_speed: lowest / highest valid segment speed (see segment_speeds); NaN when the
    interval has no valid segment.
    """
    if 'X' in traj_df.columns:
        x_col, y_col = 'X', 'Y'
    else:
        x_col, y_col = 'x', 'y'

    speeds = segment_speeds(traj_df, x_col, y_col)
    min_speed = float(np.min(speeds)) if len(speeds) else float('nan')

    if len(traj_df) < 2:
        return {
            'total_distance': 0.0, 'duration': 0.0, 'avg_speed': 0.0,
            'net_displacement': 0.0, 'sinuosity': 1.0, 'bbox_area': 0.0,
            'avg_direction': 0.0, 'max_speed': 0.0, 'min_speed': min_speed
        }

    traj_df = traj_df.sort_values('tst').reset_index(drop=True)

    distances = np.sqrt(np.diff(traj_df[x_col])**2 + np.diff(traj_df[y_col])**2)
    total_distance = np.sum(distances)

    duration = traj_df['tst'].iloc[-1] - traj_df['tst'].iloc[0]
    if duration == 0:
        duration = 1.0

    avg_speed = total_distance / duration if duration > 0 else 0.0

    net_displacement = np.sqrt(
        (traj_df[x_col].iloc[-1] - traj_df[x_col].iloc[0])**2 +
        (traj_df[y_col].iloc[-1] - traj_df[y_col].iloc[0])**2
    )

    sinuosity = total_distance / net_displacement if net_displacement > 0 else 1.0
    bbox_area = (traj_df[x_col].max() - traj_df[x_col].min()) * \
                (traj_df[y_col].max() - traj_df[y_col].min())

    dx = np.diff(traj_df[x_col])
    dy = np.diff(traj_df[y_col])
    angles = np.arctan2(dy, dx)
    avg_direction = np.mean(angles) if len(angles) > 0 else 0.0

    # Same valid-segment definition as min_speed (previously zero elapsed time was replaced by 0.01,
    # which inflated the maximum for duplicate timestamps)
    max_speed = float(np.max(speeds)) if len(speeds) else float('nan')

    return {
        'total_distance': float(total_distance),
        'duration': float(duration),
        'avg_speed': float(avg_speed),
        'net_displacement': float(net_displacement),
        'sinuosity': float(sinuosity),
        'bbox_area': float(bbox_area),
        'avg_direction': float(avg_direction),
        'max_speed': max_speed,
        'min_speed': min_speed
    }


# =============================================================================
# DISTANCE MATRIX COMPUTATION
# =============================================================================

@st.cache_data
def compute_feature_distance_matrix(df, selected_configs, selected_objects, start_time, end_time, selected_features=None):
    """Compute distance matrix based on trajectory features (Method 1)."""
    filtered_df = df[
        (df['config_source'].isin(selected_configs)) &
        (df['obj'].isin(selected_objects)) &
        (df['tst'] >= start_time) &
        (df['tst'] <= end_time)
    ].copy()
    
    trajectory_groups = filtered_df.groupby(['config_source', 'obj'])
    
    features_list = []
    trajectory_ids = []
    trajectories = []
    
    for (config, obj), traj_df in trajectory_groups:
        features = extract_trajectory_features(traj_df)
        features_list.append(features)
        trajectory_ids.append(f"{config}_obj{obj}")
        
        if 'X' in traj_df.columns:
            traj_coords = traj_df[['X', 'Y', 'tst']].values
        else:
            traj_coords = traj_df[['x', 'y', 'tst']].values
        trajectories.append(traj_coords)
    
    features_df = pd.DataFrame(features_list, index=trajectory_ids)
    
    if selected_features is not None and len(selected_features) > 0:
        features_df = features_df[selected_features]

    unavailable = features_df.isna()
    if unavailable.any().any():
        details = '; '.join(f"{FEATURE_SELECTOR_LABELS.get(f, f)}: " + ', '.join(features_df.index[unavailable[f]])
                            for f in features_df.columns if unavailable[f].any())
        raise ValueError("Some selected features are unavailable (no segment with positive elapsed time in the "
                         f"interval) for: {details}. Widen the interval or deselect these features.")
    
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features_df)
    distance_matrix = cdist(features_normalized, features_normalized, metric='euclidean')
    
    return distance_matrix, trajectory_ids, features_df, trajectories


@st.cache_data
def compute_chamfer_distance_matrix(df, selected_configs, selected_objects, start_time, end_time):
    """Compute distance matrix based on Chamfer distance (Method 2)."""
    filtered_df = df[
        (df['config_source'].isin(selected_configs)) &
        (df['obj'].isin(selected_objects)) &
        (df['tst'] >= start_time) &
        (df['tst'] <= end_time)
    ].copy()
    
    trajectory_groups = filtered_df.groupby(['config_source', 'obj'])
    
    trajectories = []
    trajectory_ids = []
    
    for (config, obj), traj_df in trajectory_groups:
        traj_df = traj_df.sort_values('tst').reset_index(drop=True)
        coords = traj_df[['x', 'y']].values
        trajectories.append(coords)
        trajectory_ids.append(f"{config}_obj{obj}")
    
    n = len(trajectories)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            traj_A = trajectories[i]
            traj_B = trajectories[j]
            
            dist_A_to_B = np.mean([np.min(cdist([a], traj_B, metric='euclidean')) for a in traj_A])
            dist_B_to_A = np.mean([np.min(cdist([b], traj_A, metric='euclidean')) for b in traj_B])
            chamfer_dist = (dist_A_to_B + dist_B_to_A) / 2.0
            
            distance_matrix[i, j] = chamfer_dist
            distance_matrix[j, i] = chamfer_dist
    
    return distance_matrix, trajectory_ids, trajectories


def dtw_distance(traj_A, traj_B):
    """Compute Dynamic Time Warping distance between two trajectories."""
    n, m = len(traj_A), len(traj_B)
    
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = euclidean(traj_A[i-1], traj_B[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],
                dtw_matrix[i, j-1],
                dtw_matrix[i-1, j-1]
            )
    
    return dtw_matrix[n, m]


@st.cache_data
def compute_dtw_distance_matrix(df, selected_configs, selected_objects, start_time, end_time):
    """Compute distance matrix based on Dynamic Time Warping (Method 3)."""
    filtered_df = df[
        (df['config_source'].isin(selected_configs)) &
        (df['obj'].isin(selected_objects)) &
        (df['tst'] >= start_time) &
        (df['tst'] <= end_time)
    ].copy()
    
    trajectory_groups = filtered_df.groupby(['config_source', 'obj'])
    
    trajectories = []
    trajectory_ids = []
    
    for (config, obj), traj_df in trajectory_groups:
        traj_df = traj_df.sort_values('tst').reset_index(drop=True)
        coords = traj_df[['x', 'y', 'tst']].values
        trajectories.append(coords)
        trajectory_ids.append(f"{config}_obj{obj}")
    
    n = len(trajectories)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            dtw_dist = dtw_distance(trajectories[i], trajectories[j])
            distance_matrix[i, j] = dtw_dist
            distance_matrix[j, i] = dtw_dist
    
    return distance_matrix, trajectory_ids, trajectories


# =============================================================================
# OPTIMAL CLUSTER DETECTION
# =============================================================================

def detect_optimal_clusters(distance_matrix, max_clusters=10, return_plot_data=False):
    """Auto-detect optimal number of clusters using elbow method with silhouette validation."""
    n_samples = len(distance_matrix)
    
    if n_samples < 3:
        if return_plot_data:
            return 2, {'k_values': [2], 'inertias': [0], 'silhouette_scores': [0], 'optimal_k': 2}
        return 2
    if n_samples < 10:
        optimal = min(3, n_samples - 1)
        if return_plot_data:
            return optimal, {'k_values': [2, optimal], 'inertias': [0, 0], 'silhouette_scores': [0, 0], 'optimal_k': optimal}
        return optimal
    
    max_k = min(max_clusters, n_samples - 1)
    
    inertias = []
    silhouette_scores_list = []
    
    for k in range(2, max_k + 1):
        clustering = AgglomerativeClustering(
            n_clusters=k,
            metric='precomputed',
            linkage='average'
        )
        labels = clustering.fit_predict(distance_matrix)
        
        inertia = 0
        for cluster_id in range(k):
            cluster_mask = labels == cluster_id
            if np.sum(cluster_mask) > 0:
                cluster_distances = distance_matrix[cluster_mask][:, cluster_mask]
                inertia += np.sum(cluster_distances) / (2 * np.sum(cluster_mask))
        inertias.append(inertia)
        
        try:
            sil_score = silhouette_score(distance_matrix, labels, metric='precomputed')
            silhouette_scores_list.append(sil_score)
        except:
            silhouette_scores_list.append(0)
    
    if len(inertias) < 2:
        if return_plot_data:
            k_values = list(range(2, max_k + 1)) if max_k > 2 else [2, 3]
            return 3, {'k_values': k_values, 'inertias': inertias, 'silhouette_scores': silhouette_scores_list, 'optimal_k': 3}
        return 3
    
    inertias_norm = np.array(inertias)
    inertias_norm = (inertias_norm - inertias_norm.min()) / (inertias_norm.max() - inertias_norm.min() + 1e-10)
    
    angles = []
    for i in range(1, len(inertias_norm) - 1):
        p1 = np.array([i-1, inertias_norm[i-1]])
        p2 = np.array([i, inertias_norm[i]])
        p3 = np.array([i+1, inertias_norm[i+1]])
        
        v1 = p1 - p2
        v2 = p3 - p2
        
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))
        angles.append(angle)
    
    if len(angles) > 0:
        elbow_idx = np.argmax(angles) + 1
        optimal_k = elbow_idx + 2
    else:
        optimal_k = 3
    
    if len(silhouette_scores_list) > 0:
        if silhouette_scores_list[optimal_k - 2] < 0.25:
            best_sil_idx = np.argmax(silhouette_scores_list)
            if silhouette_scores_list[best_sil_idx] > 0.25:
                optimal_k = best_sil_idx + 2
    
    optimal_k = max(2, min(optimal_k, max_k))
    
    if return_plot_data:
        k_values = list(range(2, max_k + 1))
        return optimal_k, {
            'k_values': k_values,
            'inertias': inertias,
            'silhouette_scores': silhouette_scores_list,
            'optimal_k': optimal_k
        }
    
    return optimal_k


def initialize_clustering_session_state():
    """Initialise session state variables for clustering."""
    if 'clustering_method' not in st.session_state:
        st.session_state.clustering_method = None
    if 'clustering_distance_matrix' not in st.session_state:
        st.session_state.clustering_distance_matrix = None
    if 'clustering_linkage_matrix' not in st.session_state:
        st.session_state.clustering_linkage_matrix = None
    if 'clustering_trajectory_ids' not in st.session_state:
        st.session_state.clustering_trajectory_ids = None
    if 'clustering_optimal_n' not in st.session_state:
        st.session_state.clustering_optimal_n = None
    if 'clustering_current_n' not in st.session_state:
        st.session_state.clustering_current_n = None
    if 'clustering_labels' not in st.session_state:
        st.session_state.clustering_labels = None
    if 'clustering_features_df' not in st.session_state:
        st.session_state.clustering_features_df = None
    if 'clustering_trajectories' not in st.session_state:
        st.session_state.clustering_trajectories = None
