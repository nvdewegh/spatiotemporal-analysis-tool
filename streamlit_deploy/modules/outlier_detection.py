"""
Outlier Detection Module

This module implements various outlier detection methods for spatiotemporal trajectory data:
- Graphical methods (box plots, scatter plots)
- Statistical methods (Z-score, modified Z-score, IQR)
- Depth-based methods (convex hull layers)
- Distance-based methods (k-nearest neighbors)
- Density-based methods (Local Outlier Factor, Isolation Forest)

Author: Spatiotemporal Analysis Tool
Date: November 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from scipy.spatial import ConvexHull, distance
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
import warnings
warnings.filterwarnings('ignore')

from .common import PLOTLY_CONFIG, render_interactive_chart


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_trajectory_features(df, config, obj_id):
    """
    Extract statistical features from a single trajectory for outlier detection.
    
    Parameters:
        df: DataFrame with trajectory data
        config: Configuration name
        obj_id: Object ID
        
    Returns:
        Dictionary of trajectory features
    """
    traj = df[(df['config_source'] == config) & (df['obj'] == obj_id)].sort_values('tst')
    
    if len(traj) < 2:
        return None
    
    # Spatial features
    x_coords = traj['x'].values
    y_coords = traj['y'].values
    
    # Calculate displacements
    dx = np.diff(x_coords)
    dy = np.diff(y_coords)
    distances = np.sqrt(dx**2 + dy**2)
    
    # Temporal features
    times = traj['tst'].values
    dt = np.diff(times)
    dt = dt[dt > 0]  # Remove zero time differences
    
    # Velocities
    velocities = distances / dt if len(dt) > 0 else np.array([0])
    velocities = velocities[np.isfinite(velocities)]
    
    # Accelerations
    if len(velocities) > 1:
        dv = np.diff(velocities)
        accelerations = dv / dt[:-1] if len(dt) > 1 else np.array([0])
        accelerations = accelerations[np.isfinite(accelerations)]
    else:
        accelerations = np.array([0])
    
    features = {
        'config': config,
        'object': obj_id,
        'trajectory_id': f"{config}_obj{obj_id}",
        
        # Basic statistics
        'total_distance': np.sum(distances),
        'duration': times[-1] - times[0],
        'n_points': len(traj),
        
        # Spatial extent
        'x_range': x_coords.max() - x_coords.min(),
        'y_range': y_coords.max() - y_coords.min(),
        'x_mean': x_coords.mean(),
        'y_mean': y_coords.mean(),
        'x_std': x_coords.std(),
        'y_std': y_coords.std(),
        
        # Velocity statistics
        'avg_speed': np.mean(velocities) if len(velocities) > 0 else 0,
        'max_speed': np.max(velocities) if len(velocities) > 0 else 0,
        'std_speed': np.std(velocities) if len(velocities) > 0 else 0,
        
        # Acceleration statistics
        'avg_acceleration': np.mean(np.abs(accelerations)) if len(accelerations) > 0 else 0,
        'max_acceleration': np.max(np.abs(accelerations)) if len(accelerations) > 0 else 0,
        
        # Path complexity
        'sinuosity': np.sum(distances) / np.sqrt((x_coords[-1] - x_coords[0])**2 + 
                                                   (y_coords[-1] - y_coords[0])**2) if np.sqrt((x_coords[-1] - x_coords[0])**2 + (y_coords[-1] - y_coords[0])**2) > 0 else 1,
        
        # Directional change
        'total_turning_angle': calculate_total_turning_angle(x_coords, y_coords),
    }
    
    return features


def calculate_total_turning_angle(x_coords, y_coords):
    """Calculate total turning angle along trajectory."""
    if len(x_coords) < 3:
        return 0
    
    angles = []
    for i in range(1, len(x_coords) - 1):
        v1 = np.array([x_coords[i] - x_coords[i-1], y_coords[i] - y_coords[i-1]])
        v2 = np.array([x_coords[i+1] - x_coords[i], y_coords[i+1] - y_coords[i]])
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 > 0 and norm2 > 0:
            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)
            angles.append(angle)
    
    return np.sum(angles) if angles else 0


# =============================================================================
# STATISTICAL METHODS
# =============================================================================

def detect_outliers_zscore(features_df, feature_cols, threshold=3.0):
    """
    Detect outliers using Z-score method.
    
    Parameters:
        features_df: DataFrame with trajectory features
        feature_cols: List of feature column names to use
        threshold: Z-score threshold (default: 3.0)
        
    Returns:
        DataFrame with outlier scores and labels
    """
    result_df = features_df.copy()
    
    # Calculate Z-scores for each feature
    z_scores = np.abs(stats.zscore(features_df[feature_cols], nan_policy='omit'))
    
    # A point is an outlier if ANY feature has |z-score| > threshold
    result_df['max_zscore'] = np.max(z_scores, axis=1)
    result_df['is_outlier_zscore'] = result_df['max_zscore'] > threshold
    result_df['outlier_score_zscore'] = result_df['max_zscore'] / threshold
    
    return result_df


def detect_outliers_modified_zscore(features_df, feature_cols, threshold=3.5):
    """
    Detect outliers using Modified Z-score (based on median absolute deviation).
    More robust to outliers than standard Z-score.
    
    Parameters:
        features_df: DataFrame with trajectory features
        feature_cols: List of feature column names to use
        threshold: Modified Z-score threshold (default: 3.5)
        
    Returns:
        DataFrame with outlier scores and labels
    """
    result_df = features_df.copy()
    
    modified_z_scores = []
    for col in feature_cols:
        median = np.median(features_df[col])
        mad = np.median(np.abs(features_df[col] - median))
        
        if mad == 0:
            modified_z = np.zeros(len(features_df))
        else:
            modified_z = 0.6745 * (features_df[col] - median) / mad
        
        modified_z_scores.append(np.abs(modified_z))
    
    modified_z_scores = np.array(modified_z_scores).T
    
    result_df['max_modified_zscore'] = np.max(modified_z_scores, axis=1)
    result_df['is_outlier_modified_zscore'] = result_df['max_modified_zscore'] > threshold
    result_df['outlier_score_modified_zscore'] = result_df['max_modified_zscore'] / threshold
    
    return result_df


def detect_outliers_iqr(features_df, feature_cols, multiplier=1.5):
    """
    Detect outliers using Interquartile Range (IQR) method.
    
    Parameters:
        features_df: DataFrame with trajectory features
        feature_cols: List of feature column names to use
        multiplier: IQR multiplier (default: 1.5, use 3.0 for extreme outliers)
        
    Returns:
        DataFrame with outlier labels
    """
    result_df = features_df.copy()
    
    outlier_counts = np.zeros(len(features_df))
    
    for col in feature_cols:
        Q1 = features_df[col].quantile(0.25)
        Q3 = features_df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outliers = (features_df[col] < lower_bound) | (features_df[col] > upper_bound)
        outlier_counts += outliers.astype(int)
    
    result_df['outlier_feature_count_iqr'] = outlier_counts
    result_df['is_outlier_iqr'] = outlier_counts > 0
    result_df['outlier_score_iqr'] = outlier_counts / len(feature_cols)
    
    return result_df


def detect_outliers_mahalanobis(features_df, feature_cols, threshold=3.0):
    """
    Detect outliers using Mahalanobis distance (accounts for feature correlations).
    
    Parameters:
        features_df: DataFrame with trajectory features
        feature_cols: List of feature column names to use
        threshold: Chi-square threshold
        
    Returns:
        DataFrame with outlier scores and labels
    """
    result_df = features_df.copy()
    
    X = features_df[feature_cols].values
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Calculate covariance matrix
    cov_matrix = np.cov(X_scaled.T)
    
    try:
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        
        # Calculate Mahalanobis distance for each point
        mahal_distances = []
        mean = np.mean(X_scaled, axis=0)
        
        for i in range(len(X_scaled)):
            diff = X_scaled[i] - mean
            mahal_dist = np.sqrt(diff @ inv_cov_matrix @ diff.T)
            mahal_distances.append(mahal_dist)
        
        result_df['mahalanobis_distance'] = mahal_distances
        result_df['is_outlier_mahalanobis'] = np.array(mahal_distances) > threshold
        result_df['outlier_score_mahalanobis'] = np.array(mahal_distances) / threshold
        
    except np.linalg.LinAlgError:
        # Singular matrix - use alternative
        result_df['mahalanobis_distance'] = 0
        result_df['is_outlier_mahalanobis'] = False
        result_df['outlier_score_mahalanobis'] = 0
    
    return result_df


# =============================================================================
# DEPTH-BASED METHODS
# =============================================================================

def detect_outliers_convex_hull(features_df, feature_cols, depth_threshold=2):
    """
    Detect outliers using convex hull depth method.
    Points in outer layers are considered outliers.
    
    Parameters:
        features_df: DataFrame with trajectory features
        feature_cols: List of 2 feature column names for 2D convex hull
        depth_threshold: Maximum depth to be considered outlier
        
    Returns:
        DataFrame with depth values and outlier labels
    """
    result_df = features_df.copy()
    
    # Use only first 2 features for 2D convex hull
    if len(feature_cols) > 2:
        feature_cols = feature_cols[:2]
    
    points = features_df[feature_cols].values
    n_points = len(points)
    
    depths = np.zeros(n_points)
    remaining_indices = np.arange(n_points)
    current_depth = 1
    
    while len(remaining_indices) > 2:
        try:
            current_points = points[remaining_indices]
            
            if len(current_points) < 3:
                depths[remaining_indices] = current_depth
                break
            
            hull = ConvexHull(current_points)
            hull_point_indices = remaining_indices[hull.vertices]
            
            depths[hull_point_indices] = current_depth
            
            # Remove hull points for next iteration
            mask = np.ones(len(remaining_indices), dtype=bool)
            mask[hull.vertices] = False
            remaining_indices = remaining_indices[mask]
            
            current_depth += 1
            
        except Exception:
            # If convex hull fails, assign remaining points to current depth
            depths[remaining_indices] = current_depth
            break
    
    # Handle remaining points
    if len(remaining_indices) > 0:
        depths[remaining_indices] = current_depth
    
    result_df['convex_hull_depth'] = depths
    result_df['is_outlier_convex_hull'] = depths <= depth_threshold
    result_df['outlier_score_convex_hull'] = 1.0 / depths  # Inverse depth as score
    
    return result_df


# =============================================================================
# DISTANCE-BASED METHODS
# =============================================================================

def detect_outliers_knn_distance(features_df, feature_cols, k=5, threshold_percentile=95):
    """
    Detect outliers based on distance to k-nearest neighbors.
    
    Parameters:
        features_df: DataFrame with trajectory features
        feature_cols: List of feature column names to use
        k: Number of nearest neighbors
        threshold_percentile: Percentile threshold for outlier detection
        
    Returns:
        DataFrame with distance scores and outlier labels
    """
    result_df = features_df.copy()
    
    X = features_df[feature_cols].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k+1)  # +1 because point itself is included
    nbrs.fit(X_scaled)
    
    distances, indices = nbrs.kneighbors(X_scaled)
    
    # Average distance to k nearest neighbors (excluding self)
    avg_knn_distances = np.mean(distances[:, 1:], axis=1)
    
    # Determine threshold
    threshold = np.percentile(avg_knn_distances, threshold_percentile)
    
    result_df['knn_distance'] = avg_knn_distances
    result_df['is_outlier_knn'] = avg_knn_distances > threshold
    result_df['outlier_score_knn'] = avg_knn_distances / threshold
    
    return result_df


# =============================================================================
# DENSITY-BASED METHODS
# =============================================================================

def detect_outliers_lof(features_df, feature_cols, n_neighbors=20, contamination=0.1):
    """
    Detect outliers using Local Outlier Factor (LOF).
    
    Parameters:
        features_df: DataFrame with trajectory features
        feature_cols: List of feature column names to use
        n_neighbors: Number of neighbors for density estimation
        contamination: Expected proportion of outliers
        
    Returns:
        DataFrame with LOF scores and outlier labels
    """
    result_df = features_df.copy()
    
    X = features_df[feature_cols].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply LOF
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    outlier_labels = lof.fit_predict(X_scaled)
    lof_scores = -lof.negative_outlier_factor_  # Convert to positive scores
    
    result_df['lof_score'] = lof_scores
    result_df['is_outlier_lof'] = outlier_labels == -1
    result_df['outlier_score_lof'] = lof_scores
    
    return result_df


def detect_outliers_isolation_forest(features_df, feature_cols, contamination=0.1, random_state=42):
    """
    Detect outliers using Isolation Forest.
    
    Parameters:
        features_df: DataFrame with trajectory features
        feature_cols: List of feature column names to use
        contamination: Expected proportion of outliers
        random_state: Random seed
        
    Returns:
        DataFrame with anomaly scores and outlier labels
    """
    result_df = features_df.copy()
    
    X = features_df[feature_cols].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    outlier_labels = iso_forest.fit_predict(X_scaled)
    anomaly_scores = -iso_forest.score_samples(X_scaled)  # Higher = more anomalous
    
    result_df['isolation_forest_score'] = anomaly_scores
    result_df['is_outlier_isolation_forest'] = outlier_labels == -1
    result_df['outlier_score_isolation_forest'] = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
    
    return result_df


def detect_outliers_elliptic_envelope(features_df, feature_cols, contamination=0.1):
    """
    Detect outliers using Elliptic Envelope (assumes Gaussian distribution).
    
    Parameters:
        features_df: DataFrame with trajectory features
        feature_cols: List of feature column names to use
        contamination: Expected proportion of outliers
        
    Returns:
        DataFrame with outlier scores and labels
    """
    result_df = features_df.copy()
    
    X = features_df[feature_cols].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    try:
        # Apply Elliptic Envelope
        ee = EllipticEnvelope(contamination=contamination, random_state=42)
        outlier_labels = ee.fit_predict(X_scaled)
        
        # Get Mahalanobis distances
        mahal_distances = ee.mahalanobis(X_scaled)
        
        result_df['elliptic_envelope_distance'] = mahal_distances
        result_df['is_outlier_elliptic_envelope'] = outlier_labels == -1
        result_df['outlier_score_elliptic_envelope'] = mahal_distances / mahal_distances.max()
        
    except Exception:
        result_df['elliptic_envelope_distance'] = 0
        result_df['is_outlier_elliptic_envelope'] = False
        result_df['outlier_score_elliptic_envelope'] = 0
    
    return result_df


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_outliers_scatter(features_df, x_feature, y_feature, outlier_col, title):
    """Create scatter plot highlighting outliers."""
    fig = go.Figure()
    
    # Normal points
    normal = features_df[~features_df[outlier_col]]
    fig.add_trace(go.Scatter(
        x=normal[x_feature],
        y=normal[y_feature],
        mode='markers',
        name='Normal',
        marker=dict(size=8, color='blue', opacity=0.6),
        text=normal['trajectory_id'],
        hovertemplate='<b>%{text}</b><br>' + f'{x_feature}: %{{x:.2f}}<br>{y_feature}: %{{y:.2f}}<extra></extra>'
    ))
    
    # Outliers
    outliers = features_df[features_df[outlier_col]]
    if len(outliers) > 0:
        fig.add_trace(go.Scatter(
            x=outliers[x_feature],
            y=outliers[y_feature],
            mode='markers',
            name='Outlier',
            marker=dict(size=12, color='red', symbol='x', line=dict(width=2)),
            text=outliers['trajectory_id'],
            hovertemplate='<b>%{text}</b><br>' + f'{x_feature}: %{{x:.2f}}<br>{y_feature}: %{{y:.2f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_feature.replace('_', ' ').title(),
        yaxis_title=y_feature.replace('_', ' ').title(),
        hovermode='closest',
        height=500
    )
    
    return fig


def plot_outlier_scores(features_df, score_col, title):
    """Create bar plot of outlier scores."""
    sorted_df = features_df.sort_values(score_col, ascending=False)
    
    # Convert score column name to is_outlier column name
    # e.g., 'outlier_score_zscore' -> 'is_outlier_zscore'
    outlier_col = score_col.replace('outlier_score_', 'is_outlier_')
    
    colors = ['red' if is_outlier else 'blue' 
              for is_outlier in sorted_df[outlier_col]]
    
    fig = go.Figure(data=[
        go.Bar(
            x=sorted_df['trajectory_id'],
            y=sorted_df[score_col],
            marker_color=colors,
            hovertemplate='<b>%{x}</b><br>Score: %{y:.2f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title='Trajectory',
        yaxis_title='Outlier Score',
        height=400,
        xaxis_tickangle=-45
    )
    
    return fig

def plot_boxplots(features_df, feature_cols):
    """Create box plots for multiple features to visualize outliers."""
    fig = go.Figure()
    
    for col in feature_cols:
        fig.add_trace(go.Box(
            y=features_df[col],
            name=col.replace('_', ' ').title(),
            boxmean='sd'
        ))
    
    fig.update_layout(
        title='Feature Distributions (Box Plots)',
        yaxis_title='Value',
        height=500,
        showlegend=True
    )
    
    return fig


def plot_convex_hull_layers(features_df, x_feature, y_feature):
    """Visualize convex hull layers."""
    fig = go.Figure()
    
    max_depth = int(features_df['convex_hull_depth'].max())
    colors = px.colors.sequential.Viridis
    
    for depth in range(1, max_depth + 1):
        layer_points = features_df[features_df['convex_hull_depth'] == depth]
        
        if len(layer_points) > 0:
            color_idx = min(int((depth - 1) / max_depth * (len(colors) - 1)), len(colors) - 1)
            
            fig.add_trace(go.Scatter(
                x=layer_points[x_feature],
                y=layer_points[y_feature],
                mode='markers',
                name=f'Depth {depth}',
                marker=dict(
                    size=10,
                    color=colors[color_idx],
                    line=dict(width=1, color='white')
                ),
                text=layer_points['trajectory_id'],
                hovertemplate='<b>%{text}</b><br>Depth: ' + str(depth) + '<br>' +
                              f'{x_feature}: %{{x:.2f}}<br>{y_feature}: %{{y:.2f}}<extra></extra>'
            ))
    
    fig.update_layout(
        title='Convex Hull Depth Layers',
        xaxis_title=x_feature.replace('_', ' ').title(),
        yaxis_title=y_feature.replace('_', ' ').title(),
        hovermode='closest',
        height=600
    )
    
    return fig


# =============================================================================
# MAIN RENDERING FUNCTION
# =============================================================================

def render_outlier_detection_section(data, selected_configs, selected_objects):
    """
    Main function to render the outlier detection section in Streamlit.
    
    Parameters:
        data: Full trajectory dataset
        selected_configs: List of selected configurations
        selected_objects: List of selected objects
    """
    st.header("🔍 Outlier Detection")
    
    st.markdown("""
    **Outlier detection** identifies trajectories that deviate significantly from normal behavior patterns.
    This module implements multiple detection methods based on different theoretical approaches.
    """)
    
    # Filter data
    filtered_df = data[
        (data['config_source'].isin(selected_configs)) &
        (data['obj'].isin(selected_objects))
    ]
    
    if filtered_df.empty:
        st.warning("No data available for the selected configurations and objects.")
        return
    
    # Extract features for all trajectories
    st.subheader("1️⃣ Feature Extraction")
    
    with st.spinner("Extracting trajectory features..."):
        features_list = []
        
        for config in selected_configs:
            for obj in selected_objects:
                features = extract_trajectory_features(filtered_df, config, obj)
                if features is not None:
                    features_list.append(features)
        
        if not features_list:
            st.error("No valid trajectories found.")
            return
        
        features_df = pd.DataFrame(features_list)
    
    st.success(f"Extracted features from {len(features_df)} trajectories")
    
    # Display feature statistics
    with st.expander("📊 View Feature Statistics"):
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        st.dataframe(features_df[numeric_cols].describe())
    
    # Select features for outlier detection
    st.subheader("2️⃣ Select Features for Analysis")
    
    available_features = [
        'total_distance', 'duration', 'avg_speed', 'max_speed', 'std_speed',
        'avg_acceleration', 'max_acceleration', 'x_range', 'y_range',
        'sinuosity', 'total_turning_angle', 'x_std', 'y_std'
    ]
    
    selected_features = st.multiselect(
        "Select features to use for outlier detection:",
        available_features,
        default=['total_distance', 'avg_speed', 'duration', 'sinuosity']
    )
    
    if len(selected_features) < 2:
        st.warning("Please select at least 2 features.")
        return
    
    # Select detection method
    st.subheader("3️⃣ Select Detection Method")
    
    method = st.selectbox(
        "Choose outlier detection method:",
        [
            "Statistical: Z-Score",
            "Statistical: Modified Z-Score (Robust)",
            "Statistical: IQR (Interquartile Range)",
            "Statistical: Mahalanobis Distance",
            "Depth-Based: Convex Hull Layers",
            "Distance-Based: k-Nearest Neighbors",
            "Density-Based: Local Outlier Factor (LOF)",
            "Density-Based: Isolation Forest",
            "Density-Based: Elliptic Envelope",
            "Graphical: Box Plots"
        ]
    )
    
    # Method-specific parameters
    st.subheader("4️⃣ Configure Parameters")
    
    col1, col2 = st.columns(2)
    
    if method == "Statistical: Z-Score":
        with col1:
            threshold = st.slider("Z-Score Threshold:", 1.0, 5.0, 3.0, 0.1)
        
        with st.spinner("Detecting outliers..."):
            result_df = detect_outliers_zscore(features_df, selected_features, threshold)
            outlier_col = 'is_outlier_zscore'
            score_col = 'outlier_score_zscore'
        
        st.markdown(f"**Method**: Points with |z-score| > {threshold} in any feature are marked as outliers.")
    
    elif method == "Statistical: Modified Z-Score (Robust)":
        with col1:
            threshold = st.slider("Modified Z-Score Threshold:", 2.0, 5.0, 3.5, 0.1)
        
        with st.spinner("Detecting outliers..."):
            result_df = detect_outliers_modified_zscore(features_df, selected_features, threshold)
            outlier_col = 'is_outlier_modified_zscore'
            score_col = 'outlier_score_modified_zscore'
        
        st.markdown(f"**Method**: Uses median absolute deviation (MAD) - more robust to outliers than standard z-score.")
    
    elif method == "Statistical: IQR (Interquartile Range)":
        with col1:
            multiplier = st.slider("IQR Multiplier:", 1.0, 3.0, 1.5, 0.1)
        
        with st.spinner("Detecting outliers..."):
            result_df = detect_outliers_iqr(features_df, selected_features, multiplier)
            outlier_col = 'is_outlier_iqr'
            score_col = 'outlier_score_iqr'
        
        st.markdown(f"**Method**: Points outside [Q1 - {multiplier}×IQR, Q3 + {multiplier}×IQR] are outliers.")
    
    elif method == "Statistical: Mahalanobis Distance":
        with col1:
            threshold = st.slider("Mahalanobis Threshold:", 2.0, 5.0, 3.0, 0.1)
        
        with st.spinner("Detecting outliers..."):
            result_df = detect_outliers_mahalanobis(features_df, selected_features, threshold)
            outlier_col = 'is_outlier_mahalanobis'
            score_col = 'outlier_score_mahalanobis'
        
        st.markdown("**Method**: Accounts for correlations between features using covariance matrix.")
    
    elif method == "Depth-Based: Convex Hull Layers":
        with col1:
            depth_threshold = st.slider("Depth Threshold:", 1, 5, 2, 1)
        
        # Use only first 2 features for 2D convex hull
        hull_features = selected_features[:2]
        
        with st.spinner("Computing convex hull layers..."):
            result_df = detect_outliers_convex_hull(features_df, hull_features, depth_threshold)
            outlier_col = 'is_outlier_convex_hull'
            score_col = 'outlier_score_convex_hull'
        
        st.markdown(f"**Method**: Points in outer {depth_threshold} convex hull layer(s) are outliers. Using features: {', '.join(hull_features)}")
        
        # Visualize convex hull layers
        st.subheader("5️⃣ Convex Hull Visualization")
        fig_hull = plot_convex_hull_layers(result_df, hull_features[0], hull_features[1])
        render_interactive_chart(st, fig_hull, "Convex Hull Depth Layers")
    
    elif method == "Distance-Based: k-Nearest Neighbors":
        with col1:
            k = st.slider("Number of Neighbors (k):", 3, 20, 5, 1)
        with col2:
            percentile = st.slider("Distance Percentile Threshold:", 80, 99, 95, 1)
        
        with st.spinner("Computing k-NN distances..."):
            result_df = detect_outliers_knn_distance(features_df, selected_features, k, percentile)
            outlier_col = 'is_outlier_knn'
            score_col = 'outlier_score_knn'
        
        st.markdown(f"**Method**: Points with avg distance to {k} nearest neighbors above {percentile}th percentile are outliers.")
    
    elif method == "Density-Based: Local Outlier Factor (LOF)":
        with col1:
            n_neighbors = st.slider("Number of Neighbors:", 5, 50, 20, 5)
        with col2:
            contamination = st.slider("Expected Outlier Proportion:", 0.01, 0.5, 0.1, 0.01)
        
        with st.spinner("Computing LOF scores..."):
            result_df = detect_outliers_lof(features_df, selected_features, n_neighbors, contamination)
            outlier_col = 'is_outlier_lof'
            score_col = 'outlier_score_lof'
        
        st.markdown("**Method**: Compares local density of each point to densities of its neighbors.")
    
    elif method == "Density-Based: Isolation Forest":
        with col1:
            contamination = st.slider("Expected Outlier Proportion:", 0.01, 0.5, 0.1, 0.01)
        
        with st.spinner("Training Isolation Forest..."):
            result_df = detect_outliers_isolation_forest(features_df, selected_features, contamination)
            outlier_col = 'is_outlier_isolation_forest'
            score_col = 'outlier_score_isolation_forest'
        
        st.markdown("**Method**: Isolates anomalies using random tree partitioning - outliers are easier to isolate.")
    
    elif method == "Density-Based: Elliptic Envelope":
        with col1:
            contamination = st.slider("Expected Outlier Proportion:", 0.01, 0.5, 0.1, 0.01)
        
        with st.spinner("Fitting Elliptic Envelope..."):
            result_df = detect_outliers_elliptic_envelope(features_df, selected_features, contamination)
            outlier_col = 'is_outlier_elliptic_envelope'
            score_col = 'outlier_score_elliptic_envelope'
        
        st.markdown("**Method**: Assumes data follows Gaussian distribution and fits robust covariance estimate.")
    
    elif method == "Graphical: Box Plots":
        st.subheader("5️⃣ Box Plot Visualization")
        fig_box = plot_boxplots(features_df, selected_features)
        render_interactive_chart(st, fig_box, "Feature distributions showing outliers")
        
        st.markdown("""
        **Interpretation**: 
        - Box represents interquartile range (IQR: Q1 to Q3)
        - Line in box is the median
        - Whiskers extend to 1.5×IQR
        - Points beyond whiskers are potential outliers
        """)
        return
    
    # Display results
    st.subheader("5️⃣ Detection Results")
    
    n_outliers = result_df[outlier_col].sum()
    n_total = len(result_df)
    outlier_pct = (n_outliers / n_total * 100) if n_total > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Trajectories", n_total)
    col2.metric("Outliers Detected", n_outliers)
    col3.metric("Outlier Percentage", f"{outlier_pct:.1f}%")
    
    # Outlier list
    if n_outliers > 0:
        st.subheader("📋 Detected Outliers")
        outliers_df = result_df[result_df[outlier_col]][['trajectory_id', score_col] + selected_features]
        outliers_df = outliers_df.sort_values(score_col, ascending=False)
        st.dataframe(outliers_df.style.format({score_col: '{:.3f}'}))
    
    # Visualizations
    st.subheader("6️⃣ Visualizations")
    
    # Scatter plot
    if len(selected_features) >= 2:
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            x_feat = st.selectbox("X-axis feature:", selected_features, index=0)
        with viz_col2:
            y_feat = st.selectbox("Y-axis feature:", selected_features, index=min(1, len(selected_features)-1))
        
        fig_scatter = plot_outliers_scatter(result_df, x_feat, y_feat, outlier_col, 
                                           f"Outlier Detection: {method}")
        render_interactive_chart(st, fig_scatter, "Outlier scatter plot")
    
    # Score distribution
    if score_col in result_df.columns:
        fig_scores = plot_outlier_scores(result_df, score_col, f"Outlier Scores: {method}")
        render_interactive_chart(st, fig_scores, "Outlier scores")
    
    # Download results
    st.subheader("7️⃣ Export Results")
    
    csv = result_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name=f"outlier_detection_{method.replace(':', '').replace(' ', '_').lower()}.csv",
        mime="text/csv"
    )
