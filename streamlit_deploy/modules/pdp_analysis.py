"""
PDP (Qualitative Trajectory Calculus) Analysis Module
=====================================================

This module implements PDP distance computation for spatiotemporal trajectories.
PDP compares trajectories using inequality matrices that capture relative positions.

Four PDP variants are supported:
1. Fundamental - Basic inequality comparison
2. Buffer - Add buffer zones around points
3. Rough - Allow tolerance in comparisons  
4. Buffer + Rough - Combined approach

The module provides:
- PDP distance matrix computation
- Interactive dendrogram visualization
- MDS embedding visualization
- Top-K similar configuration analysis
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from .common import render_interactive_chart


# =============================================================================
# CORE PDP COMPUTATION
# =============================================================================

def compute_inequality_matrix(coords, dim_values, window_length, rough=0):
    """
    Compute inequality matrix for a dimension.
    
    Args:
        coords: Array of coordinate values for this dimension
        dim_values: Values to compare (x or y coordinates)
        window_length: Number of time steps in window
        rough: Roughness tolerance (0 = no tolerance)
    
    Returns:
        Inequality matrix where:
        - 0 = second value is greater than first
        - 1 = values are equal (within rough tolerance)
        - 2 = second value is less than first
    """
    n_points = len(dim_values)
    inequality_matrix = np.zeros((n_points, n_points), dtype=int)
    
    for i in range(n_points):
        for j in range(n_points):
            diff = dim_values[j] - dim_values[i]
            
            if abs(diff) <= rough:
                inequality_matrix[i, j] = 1  # Equal (within rough tolerance)
            elif diff > rough:
                inequality_matrix[i, j] = 0  # j > i
            else:
                inequality_matrix[i, j] = 2  # j < i
    
    return inequality_matrix


def compute_pdp_distance_pair(config1_data, config2_data, window_length, rough_x=0, rough_y=0):
    """
    Compute PDP distance between two configurations.
    
    Args:
        config1_data: DataFrame with columns ['tst', 'obj', 'x', 'y'] for first config
        config2_data: DataFrame with columns ['tst', 'obj', 'x', 'y'] for second config
        window_length: Window length for temporal analysis
        rough_x: Roughness tolerance for x dimension
        rough_y: Roughness tolerance for y dimension
    
    Returns:
        Normalized PDP distance (0-100 scale)
    """
    # Get unique timestamps and objects
    timestamps1 = sorted(config1_data['tst'].unique())
    timestamps2 = sorted(config2_data['tst'].unique())
    
    # Determine time range based on window length
    max_tst = min(len(timestamps1), len(timestamps2)) - window_length + 1
    if max_tst <= 0:
        return 0  # Not enough data
    
    n_objects = len(config1_data['obj'].unique())
    points_per_window = n_objects * window_length
    
    abs_distance_x = 0
    abs_distance_y = 0
    
    # Loop over time windows
    for t_idx in range(max_tst):
        # Get data for this time window
        window_times1 = timestamps1[t_idx:t_idx + window_length]
        window_times2 = timestamps2[t_idx:t_idx + window_length]
        
        window_data1 = config1_data[config1_data['tst'].isin(window_times1)].sort_values(['tst', 'obj'])
        window_data2 = config2_data[config2_data['tst'].isin(window_times2)].sort_values(['tst', 'obj'])
        
        if len(window_data1) != points_per_window or len(window_data2) != points_per_window:
            continue
        
        # Compute inequality matrices for x dimension
        x_vals1 = window_data1['x'].values
        x_vals2 = window_data2['x'].values
        
        ineq_x1 = compute_inequality_matrix(x_vals1, x_vals1, window_length, rough_x)
        ineq_x2 = compute_inequality_matrix(x_vals2, x_vals2, window_length, rough_x)
        
        # Compute inequality matrices for y dimension
        y_vals1 = window_data1['y'].values
        y_vals2 = window_data2['y'].values
        
        ineq_y1 = compute_inequality_matrix(y_vals1, y_vals1, window_length, rough_y)
        ineq_y2 = compute_inequality_matrix(y_vals2, y_vals2, window_length, rough_y)
        
        # Accumulate absolute differences
        abs_distance_x += np.sum(np.abs(ineq_x1 - ineq_x2))
        abs_distance_y += np.sum(np.abs(ineq_y1 - ineq_y2))
    
    # Normalize distance to 0-100 scale
    # Maximum possible difference per comparison: 2 (0 vs 2 or 2 vs 0)
    # Number of comparisons per window: points^2 - points (exclude diagonal)
    max_diff_per_window = 2 * (points_per_window * points_per_window - points_per_window)
    max_total_diff = max_diff_per_window * max_tst
    
    if max_total_diff == 0:
        return 0
    
    # Combine x and y distances
    total_distance = abs_distance_x + abs_distance_y
    
    # Normalize to 0-100
    normalized_distance = int(round((total_distance / (2 * max_total_diff)) * 100))
    
    return normalized_distance


@st.cache_data
def compute_pdp_distance_matrix(df, selected_configs, selected_objects, start_time, end_time,
                                window_length=3, buffer_x=0, buffer_y=0, rough_x=0, rough_y=0,
                                pdp_variant="fundamental"):
    """
    Compute PDP distance matrix for all selected configurations.
    
    Args:
        df: DataFrame with trajectory data
        selected_configs: List of configuration IDs to analyze
        selected_objects: List of object IDs to analyze
        start_time: Start timestamp
        end_time: End timestamp
        window_length: Temporal window length for PDP analysis
        buffer_x: Buffer distance for x dimension (for buffer variant)
        buffer_y: Buffer distance for y dimension (for buffer variant)
        rough_x: Roughness tolerance for x dimension (for rough variant)
        rough_y: Roughness tolerance for y dimension (for rough variant)
        pdp_variant: One of "fundamental", "buffer", "rough", "buffer_rough"
    
    Returns:
        (distance_matrix, config_ids)
    """
    # Filter data
    filtered_df = df[
        (df['config_source'].isin(selected_configs)) &
        (df['obj'].isin(selected_objects)) &
        (df['tst'] >= start_time) &
        (df['tst'] <= end_time)
    ].copy()
    
    config_ids = selected_configs
    n_configs = len(config_ids)
    
    # Apply buffer if needed
    if pdp_variant in ["buffer", "buffer_rough"]:
        filtered_df = apply_buffer_to_trajectories(filtered_df, buffer_x, buffer_y)
    
    # Set roughness based on variant
    rough_x_val = rough_x if pdp_variant in ["rough", "buffer_rough"] else 0
    rough_y_val = rough_y if pdp_variant in ["rough", "buffer_rough"] else 0
    
    # Initialize distance matrix
    distance_matrix = np.zeros((n_configs, n_configs), dtype=int)
    
    # Compute pairwise distances
    with st.spinner(f'Computing PDP distances ({pdp_variant})...'):
        progress_bar = st.progress(0)
        total_comparisons = n_configs * (n_configs - 1) // 2
        completed = 0
        
        for i in range(n_configs):
            config1_data = filtered_df[filtered_df['config_source'] == config_ids[i]]
            
            for j in range(i + 1, n_configs):
                config2_data = filtered_df[filtered_df['config_source'] == config_ids[j]]
                
                dist = compute_pdp_distance_pair(
                    config1_data, config2_data, 
                    window_length, rough_x_val, rough_y_val
                )
                
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
                
                completed += 1
                progress_bar.progress(completed / total_comparisons)
        
        progress_bar.empty()
    
    return distance_matrix, config_ids


def apply_buffer_to_trajectories(df, buffer_x, buffer_y):
    """
    Apply buffer by adding points around each original point.
    
    For each point (x, y), adds:
    - (x - buffer_x, y)
    - (x + buffer_x, y)
    - (x, y - buffer_y)
    - (x, y + buffer_y)
    
    Args:
        df: DataFrame with trajectory data
        buffer_x: Buffer distance for x dimension
        buffer_y: Buffer distance for y dimension
    
    Returns:
        Expanded DataFrame with buffer points
    """
    if buffer_x == 0 and buffer_y == 0:
        return df
    
    buffer_points = []
    
    for _, row in df.iterrows():
        # Original point
        buffer_points.append(row.to_dict())
        
        # Add buffer points
        if buffer_x > 0:
            # Left buffer point
            left_point = row.to_dict()
            left_point['x'] = row['x'] - buffer_x
            buffer_points.append(left_point)
            
            # Right buffer point
            right_point = row.to_dict()
            right_point['x'] = row['x'] + buffer_x
            buffer_points.append(right_point)
        
        if buffer_y > 0:
            # Bottom buffer point
            bottom_point = row.to_dict()
            bottom_point['y'] = row['y'] - buffer_y
            buffer_points.append(bottom_point)
            
            # Top buffer point
            top_point = row.to_dict()
            top_point['y'] = row['y'] + buffer_y
            buffer_points.append(top_point)
    
    return pd.DataFrame(buffer_points)


def visualize_inequality_matrices(df, config_ids, selected_objects, start_time, end_time,
                                   window_length=3, buffer_x=0, buffer_y=0, rough_x=0, rough_y=0):
    """
    Visualize inequality matrices for multiple configurations.
    
    Shows the fundamental PDP representation: how configurations encode
    spatial relationships as inequality matrices (0=smaller, 1=equal, 2=bigger).
    
    Args:
        df: DataFrame with trajectory data
        config_ids: List of configuration IDs to visualize
        selected_objects: List of object IDs
        start_time, end_time: Time range
        window_length: Window size
        buffer_x, buffer_y: Buffer parameters
        rough_x, rough_y: Rough parameters
    
    Returns:
        Plotly figure with inequality matrix heatmaps
    """
    from plotly.subplots import make_subplots
    
    # Filter data
    filtered_df = df[
        (df['tst'] >= start_time) &
        (df['tst'] <= end_time) &
        (df['obj'].isin(selected_objects))
    ].copy()
    
    n_configs = len(config_ids)
    
    # Create subplots: 2 columns (X and Y) × n_configs rows
    subplot_titles = []
    for config_id in config_ids:
        subplot_titles.extend([f"Config {config_id} - X dimension", f"Config {config_id} - Y dimension"])
    
    # Use row_heights to give each row equal absolute height (not proportional!)
    # Each row gets equal weight, and we'll set total height large enough
    row_heights = [1] * n_configs  # Equal weight for all rows
    
    # Vertical spacing: use inverse scaling to keep absolute spacing constant
    # As n_configs increases, we want smaller fraction but same absolute pixels
    # Target: ~50px spacing regardless of n_configs
    # Formula: spacing_fraction ≈ 50px / (total_height / n_configs) = 50 / 700 ≈ 0.07
    vertical_spacing = max(0.02, 0.07 / n_configs)  # Smaller fraction for more configs
    
    fig = make_subplots(
        rows=n_configs,
        cols=2,
        subplot_titles=subplot_titles,
        row_heights=row_heights,
        horizontal_spacing=0.12,
        vertical_spacing=vertical_spacing
    )
    
    # Discrete colorscale for inequality values (0, 1, 2)
    # 0 = Green (smaller/left/below), 1 = Yellow (equal), 2 = Red (bigger/right/above)
    # Use sharp transitions to make it truly discrete
    colorscale = [
        [0, '#2ecc71'],      # 0 = green (smaller)
        [0.333, '#2ecc71'],  # hold green
        [0.333, '#ffeb3b'],  # 1 = bright yellow (equal)
        [0.666, '#ffeb3b'],  # hold yellow
        [0.666, '#e74c3c'],  # 2 = red (bigger)
        [1, '#e74c3c']       # hold red
    ]
    
    for row_idx, config_id in enumerate(config_ids, start=1):
        config_data = filtered_df[filtered_df['config_source'] == config_id].copy()
        
        if len(config_data) == 0:
            continue
        
        # Apply buffer if needed
        if buffer_x > 0 or buffer_y > 0:
            buffer_data = apply_buffer_to_trajectories(config_data, buffer_x, buffer_y)
            config_data = pd.concat([config_data, buffer_data], ignore_index=True)
            config_data = config_data.sort_values(['tst', 'obj'])
        
        # Get timestamps
        timestamps = sorted(config_data['tst'].unique())
        if len(timestamps) < window_length:
            continue
        
        # Use first time window for visualization
        window_times = timestamps[:window_length]
        window_data = config_data[config_data['tst'].isin(window_times)].sort_values(['tst', 'obj'])
        
        x_vals = window_data['x'].values
        y_vals = window_data['y'].values
        
        # Compute inequality matrices
        ineq_x = compute_inequality_matrix(x_vals, x_vals, window_length, rough_x)
        ineq_y = compute_inequality_matrix(y_vals, y_vals, window_length, rough_y)
        
        # Create labels for axes (object-timestamp pairs)
        labels = []
        for t_idx, t in enumerate(window_times):
            for obj in sorted(config_data[config_data['tst'] == t]['obj'].unique()):
                labels.append(f"O{obj}_T{t_idx}")
        
        # X dimension heatmap (without text annotations and without colorbar)
        fig.add_trace(
            go.Heatmap(
                z=ineq_x,
                x=labels,
                y=labels,
                colorscale=colorscale,
                zmin=0,
                zmax=2,
                showscale=False,  # No colorbar - legend is in text above
                hovertemplate='Row: %{y}<br>Col: %{x}<br>Value: %{z}<extra></extra>'
            ),
            row=row_idx,
            col=1
        )
        
        # Y dimension heatmap (without text annotations and without colorbar)
        fig.add_trace(
            go.Heatmap(
                z=ineq_y,
                x=labels,
                y=labels,
                colorscale=colorscale,
                zmin=0,
                zmax=2,
                showscale=False,  # No colorbar - legend is in text above
                hovertemplate='Row: %{y}<br>Col: %{x}<br>Value: %{z}<extra></extra>'
            ),
            row=row_idx,
            col=2
        )
    
    # Update layout - each matrix needs LARGE fixed height to maintain size
    # With subplots, Plotly divides space proportionally, so we need generous height
    # to ensure matrices don't shrink when adding more configs
    height_per_config = 700  # Large fixed height per configuration row
    total_height = height_per_config * n_configs
    
    # Fixed width for consistent display
    width = 1400
    
    fig.update_layout(
        title=f"Inequality Matrices - First Time Window (window_length={window_length})",
        height=total_height,
        width=width,
        showlegend=False
    )
    
    # Update axes - simple settings, no constraints
    for i in range(1, n_configs + 1):
        fig.update_xaxes(
            tickangle=-45,
            row=i, 
            col=1
        )
        fig.update_xaxes(
            tickangle=-45,
            row=i, 
            col=2
        )
    
    return fig


# =============================================================================
# CLUSTERING & DENDROGRAM
# =============================================================================

def perform_hierarchical_clustering(distance_matrix, n_clusters):
    """Perform hierarchical clustering with Ward linkage on PDP distances."""
    linkage_matrix = linkage(squareform(distance_matrix), method='ward')
    
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average'
    )
    cluster_labels = clustering.fit_predict(distance_matrix)
    
    return cluster_labels, linkage_matrix


def detect_optimal_clusters(distance_matrix, max_clusters=10):
    """Auto-detect optimal number of clusters using elbow method."""
    n_samples = len(distance_matrix)
    
    if n_samples < 3:
        return 2
    if n_samples < 10:
        return min(3, n_samples - 1)
    
    max_k = min(max_clusters, n_samples - 1)
    inertias = []
    silhouette_scores = []
    
    for k in range(2, max_k + 1):
        clustering = AgglomerativeClustering(
            n_clusters=k,
            metric='precomputed',
            linkage='average'
        )
        labels = clustering.fit_predict(distance_matrix)
        
        # Compute inertia (within-cluster sum of distances)
        inertia = 0
        for cluster_id in range(k):
            cluster_mask = labels == cluster_id
            if np.sum(cluster_mask) > 0:
                cluster_distances = distance_matrix[cluster_mask][:, cluster_mask]
                inertia += np.sum(cluster_distances) / (2 * np.sum(cluster_mask))
        inertias.append(inertia)
        
        # Compute silhouette score
        try:
            sil_score = silhouette_score(distance_matrix, labels, metric='precomputed')
            silhouette_scores.append(sil_score)
        except:
            silhouette_scores.append(0)
    
    # Find elbow point
    if len(inertias) < 2:
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
    
    optimal_k = np.argmax(angles) + 2 if len(angles) > 0 else 3
    optimal_k = max(2, min(optimal_k, max_k))
    
    return optimal_k


def create_interactive_dendrogram(linkage_matrix, labels, distance_matrix, n_clusters=None):
    """
    Create interactive Plotly dendrogram from linkage matrix.
    
    Args:
        linkage_matrix: Scipy linkage matrix
        labels: List of configuration labels
        distance_matrix: Original distance matrix for hover info
        n_clusters: Number of clusters to highlight (optional)
    
    Returns:
        Plotly figure
    """
    # Create dendrogram using scipy
    dend_data = dendrogram(linkage_matrix, labels=labels, no_plot=True)
    
    # Extract coordinates
    icoord = np.array(dend_data['icoord'])
    dcoord = np.array(dend_data['dcoord'])
    colors = dend_data['color_list']
    
    # Convert matplotlib colors to Plotly-compatible colors
    color_map = {
        'C0': '#1f77b4',  # blue
        'C1': '#ff7f0e',  # orange
        'C2': '#2ca02c',  # green
        'C3': '#d62728',  # red
        'C4': '#9467bd',  # purple
        'C5': '#8c564b',  # brown
        'C6': '#e377c2',  # pink
        'C7': '#7f7f7f',  # gray
        'C8': '#bcbd22',  # olive
        'C9': '#17becf',  # cyan
        'b': '#0000ff',   # blue
        'g': '#008000',   # green
        'r': '#ff0000',   # red
        'c': '#00ffff',   # cyan
        'm': '#ff00ff',   # magenta
        'y': '#ffff00',   # yellow
        'k': '#000000',   # black
    }
    
    # Create figure
    fig = go.Figure()
    
    # Add dendrogram lines
    for i in range(len(icoord)):
        x = icoord[i]
        y = dcoord[i]
        
        # Convert color to Plotly format - use default blue if not in map
        line_color = color_map.get(colors[i], '#1f77b4')
        
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            line=dict(color=line_color, width=2),
            hoverinfo='skip',
            showlegend=False
        ))
    
    # Extract leaf positions properly
    # The dendrogram ivl contains labels in the order they appear left-to-right
    # Leaves are positioned at x = 5, 15, 25, ... (i.e., 5 + 10*i for i in range(n_leaves))
    n_leaves = len(dend_data['ivl'])
    tick_vals = [5 + 10*i for i in range(n_leaves)]
    tick_text = dend_data['ivl']
    
    # Add labels at bottom
    fig.update_layout(
        title="PDP Distance Dendrogram (Hierarchical Clustering)",
        xaxis=dict(
            title="Configuration",
            tickmode='array',
            tickvals=tick_vals,
            ticktext=tick_text,
            tickangle=45
        ),
        yaxis=dict(
            title="Distance"
        ),
        width=1000,
        height=600,
        hovermode='closest',
        showlegend=False
    )
    
    return fig


# =============================================================================
# MDS VISUALIZATION
# =============================================================================

def create_mds_visualization(distance_matrix, labels, cluster_labels=None):
    """
    Create 2D MDS (Multidimensional Scaling) visualization of configurations.
    
    Args:
        distance_matrix: Pairwise distance matrix
        labels: Configuration labels
        cluster_labels: Optional cluster assignments for coloring
    
    Returns:
        Tuple of (Plotly figure, stress value)
    """
    # Perform MDS
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords_2d = mds.fit_transform(distance_matrix)
    
    # Calculate stress (quality metric)
    stress = mds.stress_
    
    # Create figure
    fig = go.Figure()
    
    if cluster_labels is not None:
        # Color by cluster
        unique_clusters = sorted(set(cluster_labels))
        colors = px.colors.qualitative.Plotly
        
        for cluster_id in unique_clusters:
            mask = np.array(cluster_labels) == cluster_id
            cluster_coords = coords_2d[mask]
            cluster_labels_list = [labels[i] for i, m in enumerate(mask) if m]
            
            fig.add_trace(go.Scatter(
                x=cluster_coords[:, 0],
                y=cluster_coords[:, 1],
                mode='markers+text',
                marker=dict(
                    size=12,
                    color=colors[cluster_id % len(colors)],
                    line=dict(color='white', width=1)
                ),
                text=cluster_labels_list,
                textposition='top center',
                name=f'Cluster {cluster_id}',
                hovertemplate='<b>%{text}</b><br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>'
            ))
    else:
        # Single color
        fig.add_trace(go.Scatter(
            x=coords_2d[:, 0],
            y=coords_2d[:, 1],
            mode='markers+text',
            marker=dict(
                size=12,
                color='#1f77b4',
                line=dict(color='white', width=1)
            ),
            text=labels,
            textposition='top center',
            hovertemplate='<b>%{text}</b><br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=f"2D MDS Projection: Configuration Similarity Map<br><sub>Stress: {stress:.2f} (lower is better)</sub>",
        xaxis=dict(title="MDS Dimension 1", zeroline=True),
        yaxis=dict(title="MDS Dimension 2", zeroline=True, scaleanchor="x", scaleratio=1),
        width=900,
        height=700,
        hovermode='closest',
        showlegend=True if cluster_labels is not None else False
    )
    
    return fig, stress


def create_mds_visualization_3d(distance_matrix, labels, cluster_labels=None):
    """
    Create 3D MDS (Multidimensional Scaling) visualization of configurations.
    
    Args:
        distance_matrix: Pairwise distance matrix
        labels: Configuration labels
        cluster_labels: Optional cluster assignments for coloring
    
    Returns:
        Plotly figure with 3D scatter plot
    """
    # Perform MDS with 3 components
    mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42)
    coords_3d = mds.fit_transform(distance_matrix)
    
    # Calculate stress (quality metric)
    stress = mds.stress_
    
    # Create figure
    fig = go.Figure()
    
    if cluster_labels is not None:
        # Color by cluster
        unique_clusters = sorted(set(cluster_labels))
        colors = px.colors.qualitative.Plotly
        
        for cluster_id in unique_clusters:
            mask = np.array(cluster_labels) == cluster_id
            cluster_coords = coords_3d[mask]
            cluster_labels_list = [labels[i] for i, m in enumerate(mask) if m]
            
            fig.add_trace(go.Scatter3d(
                x=cluster_coords[:, 0],
                y=cluster_coords[:, 1],
                z=cluster_coords[:, 2],
                mode='markers+text',
                marker=dict(
                    size=8,
                    color=colors[cluster_id % len(colors)],
                    line=dict(color='white', width=1)
                ),
                text=cluster_labels_list,
                textposition='top center',
                name=f'Cluster {cluster_id}',
                hovertemplate='<b>%{text}</b><br>x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>'
            ))
    else:
        # Single color
        fig.add_trace(go.Scatter3d(
            x=coords_3d[:, 0],
            y=coords_3d[:, 1],
            z=coords_3d[:, 2],
            mode='markers+text',
            marker=dict(
                size=8,
                color='#1f77b4',
                line=dict(color='white', width=1)
            ),
            text=labels,
            textposition='top center',
            hovertemplate='<b>%{text}</b><br>x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=f"3D MDS Projection: Configuration Similarity Map<br><sub>Stress: {stress:.2f} (lower is better)</sub>",
        scene=dict(
            xaxis=dict(title="MDS Dimension 1", zeroline=True),
            yaxis=dict(title="MDS Dimension 2", zeroline=True),
            zaxis=dict(title="MDS Dimension 3", zeroline=True),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1000,
        height=800,
        hovermode='closest',
        showlegend=True if cluster_labels is not None else False
    )
    
    return fig, stress


# =============================================================================
# TOP-K SIMILAR CONFIGURATIONS
# =============================================================================

def find_top_k_similar(distance_matrix, config_ids, target_config, k=5):
    """
    Find the top-k most similar configurations to a target configuration.
    
    Args:
        distance_matrix: Pairwise distance matrix
        config_ids: List of configuration IDs
        target_config: Configuration to compare against
        k: Number of similar configurations to return
    
    Returns:
        List of (config_id, distance) tuples sorted by similarity
    """
    # Find index of target configuration
    try:
        target_idx = config_ids.index(target_config)
    except ValueError:
        return []
    
    # Get distances to all other configurations
    distances = distance_matrix[target_idx]
    
    # Create list of (config, distance) pairs, excluding target itself
    config_distances = [
        (config_ids[i], distances[i]) 
        for i in range(len(config_ids)) 
        if i != target_idx
    ]
    
    # Sort by distance (ascending = most similar first)
    config_distances.sort(key=lambda x: x[1])
    
    # Return top k
    return config_distances[:k]


def create_tennis_court_base():
    """
    Create base tennis court figure with all court lines.
    
    Returns:
        plotly.graph_objects.Figure: Tennis court figure
    """
    fig = go.Figure()
    
    # Court dimensions (in meters)
    court_width = 8.23  # Singles court width
    court_length = 23.77  # Total length
    doubles_width = 10.97
    doubles_alley_width = (doubles_width - court_width) / 2
    service_line_distance = 6.40
    net_position = court_length / 2
    
    # Outer boundary (doubles court)
    fig.add_shape(
        type="rect", 
        x0=-doubles_alley_width, y0=0, 
        x1=court_width + doubles_alley_width, y1=court_length,
        line=dict(color="white", width=3)
    )
    
    # Singles sidelines
    fig.add_shape(type="line", x0=0, y0=0, x1=0, y1=court_length,
                  line=dict(color="white", width=2))
    fig.add_shape(type="line", x0=court_width, y0=0, x1=court_width, y1=court_length,
                  line=dict(color="white", width=2))
    
    # Baselines
    fig.add_shape(type="line", 
                  x0=-doubles_alley_width, y0=0, 
                  x1=court_width + doubles_alley_width, y1=0,
                  line=dict(color="white", width=3))
    fig.add_shape(type="line", 
                  x0=-doubles_alley_width, y0=court_length, 
                  x1=court_width + doubles_alley_width, y1=court_length,
                  line=dict(color="white", width=3))
    
    # Net
    fig.add_shape(type="line", 
                  x0=-doubles_alley_width, y0=net_position, 
                  x1=court_width + doubles_alley_width, y1=net_position,
                  line=dict(color="white", width=2))
    
    # Service lines
    service_line_bottom = net_position - service_line_distance
    service_line_top = net_position + service_line_distance
    fig.add_shape(type="line", x0=0, y0=service_line_bottom, 
                  x1=court_width, y1=service_line_bottom,
                  line=dict(color="white", width=2))
    fig.add_shape(type="line", x0=0, y0=service_line_top, 
                  x1=court_width, y1=service_line_top,
                  line=dict(color="white", width=2))
    
    # Center service line
    center_x = court_width / 2
    fig.add_shape(type="line", x0=center_x, y0=service_line_bottom, 
                  x1=center_x, y1=service_line_top,
                  line=dict(color="white", width=2))
    
    # Layout with proper aspect ratio
    x_margin = 2.0
    y_margin = 3.0
    
    fig.update_layout(
        width=500,
        height=900,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            range=[-doubles_alley_width - x_margin, court_width + doubles_alley_width + x_margin],
            showgrid=False,
            zeroline=False,
            title="Court Width (m)",
            constrain='domain',
            fixedrange=False
        ),
        yaxis=dict(
            range=[-y_margin, court_length + y_margin],
            showgrid=False,
            zeroline=False,
            title="Court Length (m)",
            scaleanchor="x",
            scaleratio=1,
            constrain='domain',
            fixedrange=False
        ),
        plot_bgcolor='#25D366',  # Grass green
        showlegend=True,
        hovermode='closest',
        dragmode='pan'
    )
    
    return fig


def plot_trajectory_comparison(df, config_ids, selected_configs, start_time, end_time, 
                               selected_objects=None, cluster_labels=None, distance_matrix=None,
                               show_buffers=False, buffer_size=0.5, show_rough=False, rough_tolerance=0.3):
    """
    Compare trajectories of selected configurations on tennis court.
    
    Args:
        df: DataFrame with trajectory data
        config_ids: List of all configuration IDs
        selected_configs: List of configs to visualize
        start_time: Start time for trajectory window
        end_time: End time for trajectory window
        selected_objects: List of object IDs to show (None = all)
        cluster_labels: Optional cluster labels for color coding
        distance_matrix: Optional distance matrix for showing similarities
        show_buffers: Whether to show buffer zones around points
        buffer_size: Radius of buffer zones (in meters)
        show_rough: Whether to show rough tolerance zones
        rough_tolerance: Radius of rough tolerance zones (in meters)
    
    Returns:
        plotly.graph_objects.Figure: Tennis court with trajectories
    """
    fig = create_tennis_court_base()
    
    # Color palette for configurations
    colors = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel
    
    # If cluster labels provided, use cluster colors
    if cluster_labels is not None:
        config_to_cluster = {config_ids[i]: cluster_labels[i] for i in range(len(config_ids))}
        cluster_colors = px.colors.qualitative.Bold
    
    # Handle object selection - if None or empty list, use empty list (show nothing)
    if selected_objects is None:
        selected_objects = []
    elif len(selected_objects) == 0:
        selected_objects = []
    
    # Plot each configuration's trajectories
    for config_idx, config in enumerate(selected_configs):
        config_data = df[(df['config_source'] == config) & 
                        (df['tst'] >= start_time) & 
                        (df['tst'] <= end_time)]
        
        if len(config_data) == 0:
            continue
        
        # Determine color
        if cluster_labels is not None and config in config_to_cluster:
            cluster_id = config_to_cluster[config]
            color = cluster_colors[cluster_id % len(cluster_colors)]
            config_label = f"{config} (Cluster {cluster_id})"
        else:
            color = colors[config_idx % len(colors)]
            config_label = str(config)
        
        # Plot trajectories ONLY for selected objects
        for obj_id in selected_objects:
            obj_data = config_data[config_data['obj'] == obj_id].sort_values('tst')
            
            if len(obj_data) < 2:
                continue
            
            # Add buffer POINTS if requested (visualize buffer parameter)
            # Buffer adds actual extra points to the data at cardinal directions
            if show_buffers and buffer_size > 0:
                buffer_x_coords = []
                buffer_y_coords = []
                
                for idx, row in obj_data.iterrows():
                    # Add 4 buffer points: left, right, bottom, top
                    buffer_x_coords.extend([
                        row['x'] - buffer_size,  # left
                        row['x'] + buffer_size,  # right
                        row['x'],                # bottom (same x)
                        row['x']                 # top (same x)
                    ])
                    buffer_y_coords.extend([
                        row['y'],                # left (same y)
                        row['y'],                # right (same y)
                        row['y'] - buffer_size,  # bottom
                        row['y'] + buffer_size   # top
                    ])
                
                # Plot buffer points as small markers
                fig.add_trace(go.Scatter(
                    x=buffer_x_coords,
                    y=buffer_y_coords,
                    mode='markers',
                    marker=dict(size=3, color=color, symbol='x', opacity=0.4),
                    showlegend=False,
                    hovertemplate="<b>Buffer Point</b><br>X: %{x:.2f}m<br>Y: %{y:.2f}m<extra></extra>"
                ))
            
            # Add rough tolerance ZONES if requested (visualize rough parameter)
            # Rough adds tolerance zones where comparisons are considered "equal"
            if show_rough and rough_tolerance > 0:
                for idx, row in obj_data.iterrows():
                    fig.add_shape(
                        type="circle",
                        xref="x", yref="y",
                        x0=row['x'] - rough_tolerance,
                        y0=row['y'] - rough_tolerance,
                        x1=row['x'] + rough_tolerance,
                        y1=row['y'] + rough_tolerance,
                        line=dict(color=color, width=2, dash="dash"),
                        fillcolor=color,
                        opacity=0.08,
                        layer="below"
                    )
            
            # Add trajectory line
            fig.add_trace(go.Scatter(
                x=obj_data['x'].values,
                y=obj_data['y'].values,
                mode='lines+markers',
                name=f"{config_label} - Obj {obj_id}",
                line=dict(color=color, width=2),
                marker=dict(size=6, color=color),
                opacity=0.8,
                hovertemplate=f"<b>{config} - Object {obj_id}</b><br>" +
                             "X: %{x:.2f}m<br>" +
                             "Y: %{y:.2f}m<br>" +
                             "<extra></extra>"
            ))
            
            # Mark start and end points
            fig.add_trace(go.Scatter(
                x=[obj_data.iloc[0]['x']],
                y=[obj_data.iloc[0]['y']],
                mode='markers',
                marker=dict(size=12, color=color, symbol='circle', 
                           line=dict(color='white', width=2)),
                showlegend=False,
                hovertemplate=f"<b>START: {config} - Obj {obj_id}</b><extra></extra>"
            ))
            
            fig.add_trace(go.Scatter(
                x=[obj_data.iloc[-1]['x']],
                y=[obj_data.iloc[-1]['y']],
                mode='markers',
                marker=dict(size=12, color=color, symbol='square',
                           line=dict(color='white', width=2)),
                showlegend=False,
                hovertemplate=f"<b>END: {config} - Obj {obj_id}</b><extra></extra>"
            ))
    
    # Add similarity info if distance matrix provided
    if distance_matrix is not None and len(selected_configs) == 2:
        idx1 = config_ids.index(selected_configs[0])
        idx2 = config_ids.index(selected_configs[1])
        similarity = 100 - distance_matrix[idx1, idx2]
        
        fig.update_layout(
            title=f"Trajectory Comparison<br>" +
                  f"<sub>Configs: {selected_configs[0]} vs {selected_configs[1]} | " +
                  f"Similarity: {similarity:.1f}%</sub>"
        )
    else:
        fig.update_layout(
            title=f"Trajectory Comparison<br>" +
                  f"<sub>{len(selected_configs)} configurations, " +
                  f"Time: {start_time:.1f}s - {end_time:.1f}s</sub>"
        )
    
    return fig


def export_pdp_results_to_csv(distance_matrix, config_ids, cluster_labels=None):
    """
    Export PDP analysis results to CSV format.
    
    Args:
        distance_matrix: PDP distance matrix
        config_ids: List of configuration IDs
        cluster_labels: Optional cluster assignments
    
    Returns:
        tuple: (distance_matrix_csv, cluster_assignments_csv)
    """
    # Export distance matrix
    dist_df = pd.DataFrame(
        distance_matrix,
        index=config_ids,
        columns=config_ids
    )
    dist_csv = dist_df.to_csv(index=True)
    
    # Export cluster assignments if available
    cluster_csv = None
    if cluster_labels is not None:
        cluster_df = pd.DataFrame({
            'Configuration': config_ids,
            'Cluster': cluster_labels
        })
        cluster_csv = cluster_df.to_csv(index=False)
    
    return dist_csv, cluster_csv


def export_similarity_rankings_to_csv(config_ids, distance_matrix, top_k=10):
    """
    Export similarity rankings for all configurations to CSV.
    
    Args:
        config_ids: List of configuration IDs
        distance_matrix: PDP distance matrix
        top_k: Number of similar configs to export per configuration
    
    Returns:
        str: CSV string with similarity rankings
    """
    rankings_data = []
    
    for i, target_config in enumerate(config_ids):
        # Get distances to all other configs
        distances = distance_matrix[i, :]
        config_distances = [
            (config_ids[j], distances[j]) 
            for j in range(len(config_ids)) 
            if j != i
        ]
        
        # Sort by distance (ascending)
        config_distances.sort(key=lambda x: x[1])
        
        # Add top k to results
        for rank, (similar_config, pdp_dist) in enumerate(config_distances[:top_k], 1):
            similarity_pct = 100 - pdp_dist
            rankings_data.append({
                'Target_Config': target_config,
                'Rank': rank,
                'Similar_Config': similar_config,
                'PDP_Distance': f"{pdp_dist:.4f}",
                'Similarity_%': f"{similarity_pct:.2f}"
            })
    
    rankings_df = pd.DataFrame(rankings_data)
    return rankings_df.to_csv(index=False)



def compute_distance_normalization_info(distance_matrix, config_ids):
    """
    Compute normalized distances and statistics for educational purposes.
    
    Args:
        distance_matrix: Raw PDP distance matrix (n x n)
        config_ids: List of configuration IDs
    
    Returns:
        Dictionary containing:
        - normalized_matrix: Distances scaled 0-100
        - max_possible_distance: Theoretical maximum
        - stats: Dictionary with mean, median, std, quartiles
        - example_calculation: Step-by-step example for one pair
        - histogram_data: Data for distribution plotting
    """
    n = len(config_ids)
    
    # Calculate maximum possible distance
    # This depends on the inequality matrix size and structure
    # For PDP: each cell can differ (0 vs 1, 0 vs 2, 1 vs 2), so max diff per cell = 2
    # Total cells in both X and Y matrices determine max possible
    max_possible = distance_matrix.max()  # Use empirical max as proxy
    
    # Normalize to 0-100 scale
    if max_possible > 0:
        normalized_matrix = (distance_matrix / max_possible) * 100
    else:
        normalized_matrix = distance_matrix.copy()
    
    # Get upper triangle (exclude diagonal) for statistics
    triu_indices = np.triu_indices_from(distance_matrix, k=1)
    raw_distances = distance_matrix[triu_indices]
    normalized_distances = normalized_matrix[triu_indices]
    
    # Compute statistics
    stats = {
        'raw': {
            'mean': np.mean(raw_distances),
            'median': np.median(raw_distances),
            'std': np.std(raw_distances),
            'min': np.min(raw_distances),
            'max': np.max(raw_distances),
            'q25': np.percentile(raw_distances, 25),
            'q75': np.percentile(raw_distances, 75)
        },
        'normalized': {
            'mean': np.mean(normalized_distances),
            'median': np.median(normalized_distances),
            'std': np.std(normalized_distances),
            'min': np.min(normalized_distances),
            'max': np.max(normalized_distances),
            'q25': np.percentile(normalized_distances, 25),
            'q75': np.percentile(normalized_distances, 75)
        }
    }
    
    # Create example calculation (first non-zero distance)
    example_i, example_j = 0, 1
    example_raw = distance_matrix[example_i, example_j]
    example_normalized = normalized_matrix[example_i, example_j]
    
    example_calculation = {
        'config_a': config_ids[example_i],
        'config_b': config_ids[example_j],
        'raw_distance': example_raw,
        'max_possible': max_possible,
        'normalized_distance': example_normalized,
        'formula': f"({example_raw:.1f} / {max_possible:.1f}) × 100 = {example_normalized:.2f}"
    }
    
    # Prepare histogram data
    histogram_data = {
        'raw': raw_distances,
        'normalized': normalized_distances
    }
    
    return {
        'normalized_matrix': normalized_matrix,
        'max_possible_distance': max_possible,
        'stats': stats,
        'example_calculation': example_calculation,
        'histogram_data': histogram_data
    }


def create_distance_distribution_plot(histogram_data, show_normalized=True):
    """
    Create histogram showing distribution of distances.
    
    Args:
        histogram_data: Dict with 'raw' and 'normalized' arrays
        show_normalized: If True, show normalized; else show raw
    
    Returns:
        Plotly figure
    """
    if show_normalized:
        data = histogram_data['normalized']
        title = "Distribution of Normalized PDP Distances (0-100 scale)"
        xaxis_title = "Normalized Distance"
    else:
        data = histogram_data['raw']
        title = "Distribution of Raw PDP Distances"
        xaxis_title = "Raw Distance"
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=data,
        nbinsx=30,
        marker_color='steelblue',
        opacity=0.7,
        name='Distance Distribution'
    ))
    
    # Add mean line
    mean_val = np.mean(data)
    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_val:.2f}",
        annotation_position="top right"
    )
    
    # Add median line
    median_val = np.median(data)
    fig.add_vline(
        x=median_val,
        line_dash="dot",
        line_color="green",
        annotation_text=f"Median: {median_val:.2f}",
        annotation_position="top left"
    )
    
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title="Frequency (Number of Config Pairs)",
        showlegend=False,
        height=400
    )
    
    return fig


def initialize_pdp_session_state():
    """Initialize session state variables for PDP analysis."""
    if 'pdp_distance_matrix' not in st.session_state:
        st.session_state.pdp_distance_matrix = None
    if 'pdp_linkage_matrix' not in st.session_state:
        st.session_state.pdp_linkage_matrix = None
    if 'pdp_config_ids' not in st.session_state:
        st.session_state.pdp_config_ids = None
    if 'pdp_optimal_n' not in st.session_state:
        st.session_state.pdp_optimal_n = None
    if 'pdp_current_n' not in st.session_state:
        st.session_state.pdp_current_n = None
    if 'pdp_cluster_labels' not in st.session_state:
        st.session_state.pdp_cluster_labels = None
    if 'pdp_variant' not in st.session_state:
        st.session_state.pdp_variant = "fundamental"
    if 'pdp_window_length' not in st.session_state:
        st.session_state.pdp_window_length = 3


# =============================================================================
# PARAMETER IMPACT VISUALIZATION (FEATURE #2)
# =============================================================================

def compare_pdp_variants(df, selected_configs, selected_objects, start_time, end_time,
                        window_length=3, buffer_x=0.5, buffer_y=0.5, rough_x=0.3, rough_y=0.3):
    """
    Compare distances across all four PDP variants.
    
    Args:
        df: DataFrame with trajectory data
        selected_configs: List of configurations to analyze
        selected_objects: List of objects to include
        start_time: Start time for analysis
        end_time: End time for analysis
        window_length: Window length for PDP
        buffer_x, buffer_y: Buffer parameters
        rough_x, rough_y: Rough parameters
    
    Returns:
        dict: Distance matrices for each variant with statistics
    """
    variants = {
        'fundamental': {'buffer_x': 0, 'buffer_y': 0, 'rough_x': 0, 'rough_y': 0},
        'buffer': {'buffer_x': buffer_x, 'buffer_y': buffer_y, 'rough_x': 0, 'rough_y': 0},
        'rough': {'buffer_x': 0, 'buffer_y': 0, 'rough_x': rough_x, 'rough_y': rough_y},
        'buffer_rough': {'buffer_x': buffer_x, 'buffer_y': buffer_y, 'rough_x': rough_x, 'rough_y': rough_y}
    }
    
    results = {}
    
    for variant_name, params in variants.items():
        # Compute distance matrix for this variant
        distance_matrix, config_ids = compute_pdp_distance_matrix(
            df, selected_configs, selected_objects,
            start_time, end_time,
            window_length=window_length,
            buffer_x=params['buffer_x'],
            buffer_y=params['buffer_y'],
            rough_x=params['rough_x'],
            rough_y=params['rough_y'],
            pdp_variant=variant_name
        )
        
        # Get upper triangle (exclude diagonal) for statistics
        triu_indices = np.triu_indices_from(distance_matrix, k=1)
        distances = distance_matrix[triu_indices]
        
        results[variant_name] = {
            'matrix': distance_matrix,
            'config_ids': config_ids,
            'mean': np.mean(distances),
            'median': np.median(distances),
            'std': np.std(distances),
            'min': np.min(distances),
            'max': np.max(distances),
            'distances': distances
        }
    
    return results


def create_parameter_comparison_plot(variant_results):
    """
    Create visualization comparing distances across PDP variants.
    
    Args:
        variant_results: Dict from compare_pdp_variants()
    
    Returns:
        plotly figure
    """
    fig = go.Figure()
    
    # Box plots for each variant
    colors = {
        'fundamental': '#3498db',  # Blue
        'buffer': '#e74c3c',       # Red
        'rough': '#2ecc71',        # Green
        'buffer_rough': '#f39c12' # Orange
    }
    
    labels = {
        'fundamental': '🔹 Fundamental',
        'buffer': '🔹 Buffer',
        'rough': '🔹 Rough',
        'buffer_rough': '🔹 Buffer + Rough'
    }
    
    for variant_name, data in variant_results.items():
        fig.add_trace(go.Box(
            y=data['distances'],
            name=labels[variant_name],
            marker_color=colors[variant_name],
            boxmean='sd',  # Show mean and standard deviation
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'Distance: %{y:.2f}<br>' +
                         '<extra></extra>'
        ))
    
    fig.update_layout(
        title="PDP Distance Distribution Across Variants",
        yaxis_title="PDP Distance",
        xaxis_title="Variant",
        showlegend=False,
        height=500,
        width=900,
        hovermode='closest'
    )
    
    return fig


def create_parameter_sensitivity_scatter(variant_results):
    """
    Create scatter plot showing how parameters affect distances.
    
    Args:
        variant_results: Dict from compare_pdp_variants()
    
    Returns:
        plotly figure with subplots
    """
    from plotly.subplots import make_subplots
    
    # Get pairwise config comparisons
    config_ids = variant_results['fundamental']['config_ids']
    n_configs = len(config_ids)
    
    # Collect all pairwise comparisons across variants
    comparison_data = []
    
    for i in range(n_configs):
        for j in range(i+1, n_configs):
            comparison_data.append({
                'pair': f"{config_ids[i]} vs {config_ids[j]}",
                'fundamental': variant_results['fundamental']['matrix'][i, j],
                'buffer': variant_results['buffer']['matrix'][i, j],
                'rough': variant_results['rough']['matrix'][i, j],
                'buffer_rough': variant_results['buffer_rough']['matrix'][i, j]
            })
    
    comp_df = pd.DataFrame(comparison_data)
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            'Fundamental vs Buffer',
            'Fundamental vs Rough',
            'Fundamental vs Buffer+Rough'
        ),
        horizontal_spacing=0.1
    )
    
    # Plot 1: Fundamental vs Buffer
    fig.add_trace(
        go.Scatter(
            x=comp_df['fundamental'],
            y=comp_df['buffer'],
            mode='markers',
            marker=dict(color='#e74c3c', size=8, opacity=0.6),
            text=comp_df['pair'],
            hovertemplate='<b>%{text}</b><br>' +
                         'Fundamental: %{x:.2f}<br>' +
                         'Buffer: %{y:.2f}<br>' +
                         '<extra></extra>',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Add diagonal reference line
    max_val = max(comp_df['fundamental'].max(), comp_df['buffer'].max())
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode='lines',
            line=dict(color='gray', dash='dash', width=1),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=1, col=1
    )
    
    # Plot 2: Fundamental vs Rough
    fig.add_trace(
        go.Scatter(
            x=comp_df['fundamental'],
            y=comp_df['rough'],
            mode='markers',
            marker=dict(color='#2ecc71', size=8, opacity=0.6),
            text=comp_df['pair'],
            hovertemplate='<b>%{text}</b><br>' +
                         'Fundamental: %{x:.2f}<br>' +
                         'Rough: %{y:.2f}<br>' +
                         '<extra></extra>',
            showlegend=False
        ),
        row=1, col=2
    )
    
    max_val2 = max(comp_df['fundamental'].max(), comp_df['rough'].max())
    fig.add_trace(
        go.Scatter(
            x=[0, max_val2],
            y=[0, max_val2],
            mode='lines',
            line=dict(color='gray', dash='dash', width=1),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=1, col=2
    )
    
    # Plot 3: Fundamental vs Buffer+Rough
    fig.add_trace(
        go.Scatter(
            x=comp_df['fundamental'],
            y=comp_df['buffer_rough'],
            mode='markers',
            marker=dict(color='#f39c12', size=8, opacity=0.6),
            text=comp_df['pair'],
            hovertemplate='<b>%{text}</b><br>' +
                         'Fundamental: %{x:.2f}<br>' +
                         'Buffer+Rough: %{y:.2f}<br>' +
                         '<extra></extra>',
            showlegend=False
        ),
        row=1, col=3
    )
    
    max_val3 = max(comp_df['fundamental'].max(), comp_df['buffer_rough'].max())
    fig.add_trace(
        go.Scatter(
            x=[0, max_val3],
            y=[0, max_val3],
            mode='lines',
            line=dict(color='gray', dash='dash', width=1),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=1, col=3
    )
    
    # Update axes
    fig.update_xaxes(title_text="Fundamental Distance", row=1, col=1)
    fig.update_xaxes(title_text="Fundamental Distance", row=1, col=2)
    fig.update_xaxes(title_text="Fundamental Distance", row=1, col=3)
    
    fig.update_yaxes(title_text="Buffer Distance", row=1, col=1)
    fig.update_yaxes(title_text="Rough Distance", row=1, col=2)
    fig.update_yaxes(title_text="Buffer+Rough Distance", row=1, col=3)
    
    fig.update_layout(
        title="Parameter Impact on Pairwise Distances<br><sub>Points above diagonal: parameter increases distance | Below: parameter decreases distance</sub>",
        height=500,
        width=1400,
        showlegend=False
    )
    
    return fig


def create_correlation_heatmap(variant_results):
    """
    Create correlation heatmap between variant distance matrices.
    
    Args:
        variant_results: Dict from compare_pdp_variants()
    
    Returns:
        plotly figure
    """
    # Calculate correlation between variants
    variants = ['fundamental', 'buffer', 'rough', 'buffer_rough']
    n_variants = len(variants)
    
    correlation_matrix = np.zeros((n_variants, n_variants))
    
    for i, var1 in enumerate(variants):
        for j, var2 in enumerate(variants):
            corr = np.corrcoef(
                variant_results[var1]['distances'],
                variant_results[var2]['distances']
            )[0, 1]
            correlation_matrix[i, j] = corr
    
    labels = ['🔹 Fundamental', '🔹 Buffer', '🔹 Rough', '🔹 Buffer + Rough']
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=labels,
        y=labels,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix,
        texttemplate='%{text:.3f}',
        textfont={"size": 14},
        colorbar=dict(title="Correlation"),
        hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Correlation Between PDP Variants<br><sub>Higher correlation = variants produce similar distance rankings</sub>",
        width=700,
        height=700,
        xaxis=dict(side='bottom'),
        yaxis=dict(autorange='reversed')
    )
    
    return fig


# =============================================================================
# CONFIGURATION SIMILARITY EXPLORER (FEATURE #5)
# =============================================================================

def find_similar_and_dissimilar_configs(distance_matrix, config_ids, target_config, k=5):
    """
    Find both most similar and most dissimilar configurations to a target.
    
    Args:
        distance_matrix: Pairwise distance matrix
        config_ids: List of configuration IDs
        target_config: Target configuration to compare against
        k: Number of similar/dissimilar configs to return
    
    Returns:
        dict with 'similar' and 'dissimilar' lists
    """
    target_idx = config_ids.index(target_config)
    distances = distance_matrix[target_idx, :]
    
    # Get indices of all other configs (exclude target itself)
    other_indices = [i for i in range(len(config_ids)) if i != target_idx]
    other_distances = [(config_ids[i], distances[i]) for i in other_indices]
    
    # Sort by distance
    other_distances.sort(key=lambda x: x[1])
    
    # Get k most similar (smallest distances)
    similar = other_distances[:k]
    
    # Get k most dissimilar (largest distances)
    dissimilar = other_distances[-k:][::-1]  # Reverse to show largest first
    
    return {
        'similar': similar,
        'dissimilar': dissimilar
    }


def create_neighborhood_visualization(distance_matrix, config_ids, target_config, 
                                      cluster_labels=None, k=10):
    """
    Create interactive visualization showing neighborhood of a configuration.
    
    Args:
        distance_matrix: Pairwise distance matrix
        config_ids: List of configuration IDs
        target_config: Target configuration
        cluster_labels: Optional cluster assignments
        k: Number of neighbors to highlight
    
    Returns:
        plotly figure
    """
    target_idx = config_ids.index(target_config)
    distances = distance_matrix[target_idx, :]
    
    # Get k nearest neighbors (excluding self)
    neighbor_indices = np.argsort(distances)[1:k+1]
    
    # Create network-style visualization
    # Use MDS to position configurations
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords_2d = mds.fit_transform(distance_matrix)
    
    fig = go.Figure()
    
    # Plot all configurations in background
    if cluster_labels is not None:
        unique_clusters = sorted(set(cluster_labels))
        colors = px.colors.qualitative.Plotly
        
        for cluster_id in unique_clusters:
            mask = np.array(cluster_labels) == cluster_id
            cluster_coords = coords_2d[mask]
            cluster_labels_list = [config_ids[i] for i, m in enumerate(mask) if m]
            
            fig.add_trace(go.Scatter(
                x=cluster_coords[:, 0],
                y=cluster_coords[:, 1],
                mode='markers',
                marker=dict(
                    size=8,
                    color=colors[cluster_id % len(colors)],
                    opacity=0.3,
                    line=dict(color='white', width=1)
                ),
                text=cluster_labels_list,
                name=f'Cluster {cluster_id}',
                hovertemplate='<b>%{text}</b><br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>',
                showlegend=True
            ))
    else:
        fig.add_trace(go.Scatter(
            x=coords_2d[:, 0],
            y=coords_2d[:, 1],
            mode='markers',
            marker=dict(size=8, color='lightgray', opacity=0.3),
            text=config_ids,
            hovertemplate='<b>%{text}</b><br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>',
            showlegend=False
        ))
    
    # Draw lines from target to neighbors
    for neighbor_idx in neighbor_indices:
        fig.add_trace(go.Scatter(
            x=[coords_2d[target_idx, 0], coords_2d[neighbor_idx, 0]],
            y=[coords_2d[target_idx, 1], coords_2d[neighbor_idx, 1]],
            mode='lines',
            line=dict(color='rgba(255, 165, 0, 0.4)', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Highlight neighbors
    neighbor_coords = coords_2d[neighbor_indices]
    neighbor_labels = [config_ids[i] for i in neighbor_indices]
    neighbor_distances = [distances[i] for i in neighbor_indices]
    
    fig.add_trace(go.Scatter(
        x=neighbor_coords[:, 0],
        y=neighbor_coords[:, 1],
        mode='markers+text',
        marker=dict(
            size=12,
            color='orange',
            line=dict(color='white', width=2)
        ),
        text=neighbor_labels,
        textposition='top center',
        name=f'Top {k} Neighbors',
        hovertemplate='<b>%{text}</b><br>Distance: %{customdata:.2f}<extra></extra>',
        customdata=neighbor_distances
    ))
    
    # Highlight target configuration
    fig.add_trace(go.Scatter(
        x=[coords_2d[target_idx, 0]],
        y=[coords_2d[target_idx, 1]],
        mode='markers+text',
        marker=dict(
            size=20,
            color='red',
            symbol='star',
            line=dict(color='white', width=2)
        ),
        text=[target_config],
        textposition='top center',
        textfont=dict(size=14, color='red'),
        name='Selected Config',
        hovertemplate=f'<b>TARGET: {target_config}</b><extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Configuration Neighborhood Explorer<br><sub>Target: {target_config} | Showing {k} nearest neighbors</sub>",
        xaxis=dict(title="MDS Dimension 1", zeroline=True, showgrid=False),
        yaxis=dict(title="MDS Dimension 2", zeroline=True, showgrid=False, scaleanchor="x", scaleratio=1),
        width=900,
        height=700,
        hovermode='closest',
        showlegend=True,
        plot_bgcolor='rgba(250,250,250,0.95)'
    )
    
    return fig


def create_distance_radial_chart(distance_matrix, config_ids, target_config, top_k=10):
    """
    Create radial/polar chart showing distances from target to other configs.
    
    Args:
        distance_matrix: Pairwise distance matrix
        config_ids: List of configuration IDs
        target_config: Target configuration
        top_k: Number of closest configs to show
    
    Returns:
        plotly figure
    """
    target_idx = config_ids.index(target_config)
    distances = distance_matrix[target_idx, :]
    
    # Get top k similar configs (excluding self)
    other_indices = [i for i in range(len(config_ids)) if i != target_idx]
    sorted_indices = sorted(other_indices, key=lambda i: distances[i])[:top_k]
    
    selected_configs = [config_ids[i] for i in sorted_indices]
    selected_distances = [distances[i] for i in sorted_indices]
    
    # Create polar bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Barpolar(
        r=selected_distances,
        theta=selected_configs,
        marker=dict(
            color=selected_distances,
            colorscale='Viridis_r',  # Reverse so darker = closer
            showscale=True,
            colorbar=dict(title="Distance")
        ),
        text=[f"{d:.2f}" for d in selected_distances],
        hovertemplate='<b>%{theta}</b><br>Distance: %{r:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Distance Radial View<br><sub>From {target_config} to {top_k} nearest configurations</sub>",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(selected_distances) * 1.1]
            ),
            angularaxis=dict(
                direction="clockwise"
            )
        ),
        width=700,
        height=700
    )
    
    return fig


# =============================================================================
# CLUSTER QUALITY METRICS (FEATURE #8)
# =============================================================================

def compute_cluster_quality_metrics(distance_matrix, min_k=2, max_k=10):
    """
    Compute multiple clustering quality metrics for different k values.
    
    Args:
        distance_matrix: Pairwise distance matrix
        min_k: Minimum number of clusters to test
        max_k: Maximum number of clusters to test
    
    Returns:
        dict with metrics for each k value
    """
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    
    n_samples = distance_matrix.shape[0]
    max_k = min(max_k, n_samples - 1)
    
    if min_k >= max_k:
        min_k = 2
        max_k = min(10, n_samples - 1)
    
    results = {
        'k_values': [],
        'silhouette': [],
        'davies_bouldin': [],
        'calinski_harabasz': [],
        'inertia': []
    }
    
    # Convert distance matrix to condensed form for hierarchical clustering
    condensed_dist = squareform(distance_matrix, checks=False)
    
    for k in range(min_k, max_k + 1):
        # Perform hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=k,
            metric='precomputed',
            linkage='average'
        )
        labels = clustering.fit_predict(distance_matrix)
        
        # Compute metrics
        results['k_values'].append(k)
        
        # Silhouette Score (higher is better, range [-1, 1])
        sil_score = silhouette_score(distance_matrix, labels, metric='precomputed')
        results['silhouette'].append(sil_score)
        
        # Davies-Bouldin Index (lower is better, minimum 0)
        # Need actual feature matrix, so we use MDS to get coordinates
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        coords = mds.fit_transform(distance_matrix)
        db_score = davies_bouldin_score(coords, labels)
        results['davies_bouldin'].append(db_score)
        
        # Calinski-Harabasz Index (higher is better)
        ch_score = calinski_harabasz_score(coords, labels)
        results['calinski_harabasz'].append(ch_score)
        
        # Inertia (within-cluster sum of squares)
        inertia = 0
        for cluster_id in range(k):
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            if len(cluster_indices) > 1:
                cluster_distances = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
                inertia += np.sum(cluster_distances ** 2) / (2 * len(cluster_indices))
        results['inertia'].append(inertia)
    
    return results


def create_quality_metrics_plot(metrics_results):
    """
    Create multi-panel plot showing all clustering quality metrics.
    
    Args:
        metrics_results: Results from compute_cluster_quality_metrics
    
    Returns:
        plotly figure with subplots
    """
    from plotly.subplots import make_subplots
    
    k_values = metrics_results['k_values']
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '📈 Silhouette Score (Higher = Better)',
            '📉 Davies-Bouldin Index (Lower = Better)',
            '📈 Calinski-Harabasz Score (Higher = Better)',
            '📉 Inertia / WCSS (Lower = Better, Elbow Method)'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.12
    )
    
    # Silhouette Score
    fig.add_trace(
        go.Scatter(
            x=k_values,
            y=metrics_results['silhouette'],
            mode='lines+markers',
            name='Silhouette',
            line=dict(color='#2E86AB', width=3),
            marker=dict(size=10, symbol='circle'),
            hovertemplate='<b>k=%{x}</b><br>Silhouette: %{y:.3f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Highlight best silhouette
    best_sil_idx = np.argmax(metrics_results['silhouette'])
    fig.add_trace(
        go.Scatter(
            x=[k_values[best_sil_idx]],
            y=[metrics_results['silhouette'][best_sil_idx]],
            mode='markers',
            marker=dict(size=15, color='red', symbol='star'),
            name='Best',
            showlegend=False,
            hovertemplate=f'<b>Optimal k={k_values[best_sil_idx]}</b><br>Score: {metrics_results["silhouette"][best_sil_idx]:.3f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Davies-Bouldin Index
    fig.add_trace(
        go.Scatter(
            x=k_values,
            y=metrics_results['davies_bouldin'],
            mode='lines+markers',
            name='Davies-Bouldin',
            line=dict(color='#A23B72', width=3),
            marker=dict(size=10, symbol='square'),
            hovertemplate='<b>k=%{x}</b><br>DB Index: %{y:.3f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Highlight best DB
    best_db_idx = np.argmin(metrics_results['davies_bouldin'])
    fig.add_trace(
        go.Scatter(
            x=[k_values[best_db_idx]],
            y=[metrics_results['davies_bouldin'][best_db_idx]],
            mode='markers',
            marker=dict(size=15, color='red', symbol='star'),
            name='Best',
            showlegend=False,
            hovertemplate=f'<b>Optimal k={k_values[best_db_idx]}</b><br>Score: {metrics_results["davies_bouldin"][best_db_idx]:.3f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Calinski-Harabasz Score
    fig.add_trace(
        go.Scatter(
            x=k_values,
            y=metrics_results['calinski_harabasz'],
            mode='lines+markers',
            name='Calinski-Harabasz',
            line=dict(color='#F18F01', width=3),
            marker=dict(size=10, symbol='diamond'),
            hovertemplate='<b>k=%{x}</b><br>CH Score: %{y:.1f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Highlight best CH
    best_ch_idx = np.argmax(metrics_results['calinski_harabasz'])
    fig.add_trace(
        go.Scatter(
            x=[k_values[best_ch_idx]],
            y=[metrics_results['calinski_harabasz'][best_ch_idx]],
            mode='markers',
            marker=dict(size=15, color='red', symbol='star'),
            name='Best',
            showlegend=False,
            hovertemplate=f'<b>Optimal k={k_values[best_ch_idx]}</b><br>Score: {metrics_results["calinski_harabasz"][best_ch_idx]:.1f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Inertia (Elbow Method)
    fig.add_trace(
        go.Scatter(
            x=k_values,
            y=metrics_results['inertia'],
            mode='lines+markers',
            name='Inertia',
            line=dict(color='#06A77D', width=3),
            marker=dict(size=10, symbol='triangle-up'),
            hovertemplate='<b>k=%{x}</b><br>Inertia: %{y:.1f}<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Update axes
    fig.update_xaxes(title_text="Number of Clusters (k)", row=1, col=1)
    fig.update_xaxes(title_text="Number of Clusters (k)", row=1, col=2)
    fig.update_xaxes(title_text="Number of Clusters (k)", row=2, col=1)
    fig.update_xaxes(title_text="Number of Clusters (k)", row=2, col=2)
    
    fig.update_yaxes(title_text="Silhouette Score", row=1, col=1)
    fig.update_yaxes(title_text="DB Index", row=1, col=2)
    fig.update_yaxes(title_text="CH Score", row=2, col=1)
    fig.update_yaxes(title_text="Inertia", row=2, col=2)
    
    fig.update_layout(
        title="Clustering Quality Metrics Comparison<br><sub>Red stars indicate optimal k for each metric</sub>",
        height=800,
        width=1000,
        showlegend=False,
        hovermode='closest'
    )
    
    return fig


def get_optimal_k_consensus(metrics_results):
    """
    Determine optimal k based on consensus across metrics.
    
    Args:
        metrics_results: Results from compute_cluster_quality_metrics
    
    Returns:
        dict with optimal k for each metric and consensus recommendation
    """
    k_values = metrics_results['k_values']
    
    # Find optimal k for each metric
    optimal_k = {
        'silhouette': k_values[np.argmax(metrics_results['silhouette'])],
        'davies_bouldin': k_values[np.argmin(metrics_results['davies_bouldin'])],
        'calinski_harabasz': k_values[np.argmax(metrics_results['calinski_harabasz'])]
    }
    
    # Elbow method - find "elbow" in inertia curve
    inertia = np.array(metrics_results['inertia'])
    if len(inertia) > 2:
        # Calculate second derivative to find inflection point
        first_diff = np.diff(inertia)
        second_diff = np.diff(first_diff)
        if len(second_diff) > 0:
            elbow_idx = np.argmax(second_diff) + 1  # +1 because of diff
            optimal_k['elbow'] = k_values[min(elbow_idx, len(k_values) - 1)]
        else:
            optimal_k['elbow'] = k_values[len(k_values) // 2]
    else:
        optimal_k['elbow'] = k_values[0]
    
    # Consensus: most common optimal k
    k_counts = {}
    for k in optimal_k.values():
        k_counts[k] = k_counts.get(k, 0) + 1
    
    consensus_k = max(k_counts.items(), key=lambda x: x[1])[0]
    consensus_count = k_counts[consensus_k]
    
    return {
        'optimal_k_per_metric': optimal_k,
        'consensus_k': consensus_k,
        'consensus_strength': f"{consensus_count}/{len(optimal_k)} metrics agree"
    }


def create_silhouette_per_cluster_plot(distance_matrix, cluster_labels, config_ids):
    """
    Create detailed silhouette plot showing per-sample scores.
    
    Args:
        distance_matrix: Pairwise distance matrix
        cluster_labels: Cluster assignments
        config_ids: Configuration IDs
    
    Returns:
        plotly figure
    """
    from sklearn.metrics import silhouette_samples
    
    # Compute per-sample silhouette scores
    silhouette_vals = silhouette_samples(distance_matrix, cluster_labels, metric='precomputed')
    
    # Sort by cluster and silhouette value
    n_clusters = len(set(cluster_labels))
    
    fig = go.Figure()
    
    y_lower = 10
    colors = px.colors.qualitative.Plotly
    
    for i in range(n_clusters):
        # Get silhouette values for cluster i
        cluster_silhouette_vals = silhouette_vals[cluster_labels == i]
        cluster_silhouette_vals.sort()
        
        cluster_size = cluster_silhouette_vals.shape[0]
        y_upper = y_lower + cluster_size
        
        # Plot silhouette values for this cluster
        fig.add_trace(go.Bar(
            x=cluster_silhouette_vals,
            y=np.arange(y_lower, y_upper),
            orientation='h',
            marker=dict(
                color=colors[i % len(colors)],
                line=dict(width=0)
            ),
            name=f'Cluster {i}',
            hovertemplate='<b>Cluster %{fullData.name}</b><br>Silhouette: %{x:.3f}<extra></extra>'
        ))
        
        y_lower = y_upper + 10
    
    # Add average silhouette score line
    avg_score = np.mean(silhouette_vals)
    fig.add_vline(
        x=avg_score,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text=f"Average: {avg_score:.3f}",
        annotation_position="top"
    )
    
    fig.update_layout(
        title=f"Silhouette Analysis<br><sub>Individual configuration scores grouped by cluster (n={n_clusters})</sub>",
        xaxis_title="Silhouette Coefficient",
        yaxis_title="Cluster",
        yaxis=dict(showticklabels=False),
        height=400 + n_clusters * 50,
        width=800,
        showlegend=True,
        hovermode='closest'
    )
    
    return fig
