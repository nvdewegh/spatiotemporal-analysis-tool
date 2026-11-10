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
    'duration': 'Duration (s)',
    'avg_speed': 'Avg Speed (m/s)',
    'net_displacement': 'Net Displacement (m)',
    'sinuosity': 'Sinuosity (ratio)',
    'bbox_area': 'Bbox Area (m²)',
    'avg_direction': 'Avg Direction (rad)',
    'max_speed': 'Max Speed (m/s)'
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
LINKAGE_METHOD = 'Ward'


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
            STANDARDISATION.get(method, 'none'), LINKAGE_METHOD)


def run_id(signature):
    import hashlib
    return hashlib.sha256(repr(signature).encode()).hexdigest()[:10]


def run_file_tag(run, n_clusters):
    """Distinct, informative filename part: method/feature, cluster count and run identifier."""
    if run['method'] == 'Features (Euclidean)':
        what = 'features-' + '+'.join(run['features'])
    else:
        what = DISTANCE_METRICS[run['method']].lower()
    return f"{what}_{n_clusters}clusters_run{run['run_id']}"


def run_configuration(run, n_requested, labels):
    """Configuration export of a computed run and its current cluster cut (not live widget values)."""
    lengths = run['trajectory_lengths']
    return {key: str(value) for key, value in {
        'Run_ID': run['run_id'],
        'Computed_At': run['computed_at'],
        'Dataset_File': run['filename'],
        'Dataset_Content_SHA256': run['fingerprint'],
        'Clustering_Method': run['method'],
        'Distance_Metric': DISTANCE_METRICS[run['method']],
        'Selected_Features': ', '.join(run['features']) if run['method'] == 'Features (Euclidean)' else 'not applicable',
        'Standardisation': STANDARDISATION.get(run['method'], 'none'),
        'Linkage_Method': LINKAGE_METHOD,
        'Requested_Number_of_Clusters': int(n_requested),
        'Actual_Number_of_Clusters': int(len(np.unique(labels))),
        'Configuration_IDs': ', '.join(map(str, run['configs'])),
        'Object_IDs': ', '.join(map(str, run['objects'])),
        'Start_Observation_tst': f"{run['start']:g}",
        'End_Observation_tst': f"{run['end']:g}",
        'Time_Unit': 'tst observation index of the dataset (not seconds)',
        'Number_of_Trajectories': len(run['trajectory_ids']),
        'Positions_per_Trajectory': (f"{min(lengths)}" if min(lengths) == max(lengths)
                                     else f"{min(lengths)}–{max(lengths)}"),
        'Total_Positions': int(sum(lengths)),
    }.items()}


def extract_trajectory_features(traj_df):
    """Extract 8 statistical features from a single trajectory."""
    if 'X' in traj_df.columns:
        x_col, y_col = 'X', 'Y'
    else:
        x_col, y_col = 'x', 'y'
    
    if len(traj_df) < 2:
        return {
            'total_distance': 0.0, 'duration': 0.0, 'avg_speed': 0.0,
            'net_displacement': 0.0, 'sinuosity': 1.0, 'bbox_area': 0.0,
            'avg_direction': 0.0, 'max_speed': 0.0
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
    
    time_diffs = np.diff(traj_df['tst'])
    time_diffs[time_diffs == 0] = 0.01
    speeds = distances / time_diffs
    max_speed = np.max(speeds) if len(speeds) > 0 else 0.0
    
    return {
        'total_distance': float(total_distance),
        'duration': float(duration),
        'avg_speed': float(avg_speed),
        'net_displacement': float(net_displacement),
        'sinuosity': float(sinuosity),
        'bbox_area': float(bbox_area),
        'avg_direction': float(avg_direction),
        'max_speed': float(max_speed)
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


def perform_hierarchical_clustering(distance_matrix, n_clusters):
    """Perform hierarchical clustering with average linkage."""
    linkage_matrix = linkage(distance_matrix, method='average')
    
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average'
    )
    cluster_labels = clustering.fit_predict(distance_matrix)
    
    return cluster_labels, linkage_matrix


def initialize_clustering_session_state():
    """Initialize session state variables for clustering."""
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
