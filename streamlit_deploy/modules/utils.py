"""
Shared utility functions for spatiotemporal analysis.

This module contains common utilities used across different analysis methods,
including color management, distance functions, and data processing helpers.
"""

import numpy as np
import pandas as pd
import streamlit as st


# =============================================================================
# COLOR MANAGEMENT
# =============================================================================

def get_color(obj_id):
    """Return a consistent color for a given object ID."""
    colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
        '#bcbd22',  # yellow-green
        '#17becf'   # cyan
    ]
    return colors[int(obj_id) % len(colors)]


# =============================================================================
# DOUGLAS-PEUCKER SIMPLIFICATION
# =============================================================================

def perpendicular_distance(point, start, end):
    """Calculate perpendicular distance from point to line segment."""
    if np.array_equal(start, end):
        return np.linalg.norm(point - start)
    
    n = np.abs((end[1] - start[1]) * point[0] - (end[0] - start[0]) * point[1] +
               end[0] * start[1] - end[1] * start[0])
    d = np.linalg.norm(end - start)
    return n / d


def douglas_peucker(points, tolerance):
    """
    Simplify a trajectory using the Douglas-Peucker algorithm (spatial only).
    
    Args:
        points: Array of [x, y] coordinates
        tolerance: Maximum perpendicular distance threshold
        
    Returns:
        Simplified array of points
    """
    if len(points) < 3:
        return points
    
    # Find point with maximum distance
    dmax = 0.0
    index = 0
    for i in range(1, len(points) - 1):
        d = perpendicular_distance(points[i], points[0], points[-1])
        if d > dmax:
            dmax = d
            index = i
    
    # If max distance is greater than tolerance, recursively simplify
    if dmax > tolerance:
        # Recursive call
        rec_results1 = douglas_peucker(points[:index+1], tolerance)
        rec_results2 = douglas_peucker(points[index:], tolerance)
        
        # Build result list
        return np.vstack((rec_results1[:-1], rec_results2))
    else:
        return np.array([points[0], points[-1]])


def spatiotemporal_distance(point, start, end):
    """Calculate spatiotemporal distance from point to line segment."""
    # point, start, end are [x, y, t]
    if np.array_equal(start, end):
        return np.linalg.norm(point - start)
    
    # Project point onto line segment
    t = np.dot(point - start, end - start) / np.dot(end - start, end - start)
    t = np.clip(t, 0, 1)
    projection = start + t * (end - start)
    
    return np.linalg.norm(point - projection)


def douglas_peucker_spatiotemporal(points, tolerance):
    """
    Simplify a trajectory using Douglas-Peucker with spatiotemporal distance.
    
    Args:
        points: Array of [x, y, t] coordinates
        tolerance: Maximum spatiotemporal distance threshold
        
    Returns:
        Simplified array of points
    """
    if len(points) < 3:
        return points
    
    # Find point with maximum distance
    dmax = 0.0
    index = 0
    for i in range(1, len(points) - 1):
        d = spatiotemporal_distance(points[i], points[0], points[-1])
        if d > dmax:
            dmax = d
            index = i
    
    # If max distance is greater than tolerance, recursively simplify
    if dmax > tolerance:
        rec_results1 = douglas_peucker_spatiotemporal(points[:index+1], tolerance)
        rec_results2 = douglas_peucker_spatiotemporal(points[index:], tolerance)
        return np.vstack((rec_results1[:-1], rec_results2))
    else:
        return np.array([points[0], points[-1]])


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(uploaded_file, update_state=True, show_success=True):
    """
    Load trajectory data from uploaded CSV file.
    
    Supports multiple formats:
    - With/without header
    - 5 or 6 columns
    - Single or multiple configurations
    
    Returns:
        DataFrame with columns: config, obj, rally_id, x, y, t, config_source
    """
    try:
        # Try reading with header first
        df_test = pd.read_csv(uploaded_file, nrows=5)
        uploaded_file.seek(0)
        
        has_header = not all(df_test.columns.str.match(r'^\d+$|^Unnamed:'))
        
        if has_header:
            df = pd.read_csv(uploaded_file)
            
            if len(df.columns) == 5:
                df.columns = ['config', 'tst', 'obj', 'x', 'y']
                df['config_name'] = None
            elif len(df.columns) == 6:
                df.columns = ['config', 'tst', 'obj', 'x', 'y', 'config_name']
            else:
                raise ValueError(f"Expected 5 or 6 columns, got {len(df.columns)}")
        else:
            df = pd.read_csv(uploaded_file, header=None)
            
            if len(df.columns) == 5:
                df.columns = ['config', 'tst', 'obj', 'x', 'y']
                df['config_name'] = None
            elif len(df.columns) == 6:
                df.columns = ['config', 'tst', 'obj', 'x', 'y', 'config_name']
            else:
                raise ValueError(f"Expected 5 or 6 columns, got {len(df.columns)}")
        
        # Convert types
        df['config'] = df['config'].astype(int)
        df['obj'] = df['obj'].astype(int)
        df['tst'] = pd.to_numeric(df['tst'], errors='coerce')
        df['x'] = pd.to_numeric(df['x'], errors='coerce')
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        
        # Handle config names
        if df['config_name'].isna().all():
            unique_configs = df['config'].unique()
            config_map = {c: f"Config_{i+1}" for i, c in enumerate(sorted(unique_configs))}
            df['config_name'] = df['config'].map(config_map)
        
        # Create rally_id and config_source
        df['rally_id'] = df.groupby(['config', 'obj']).ngroup()
        df['config_source'] = df['config_name']
        
        # Reorder columns
        df = df[['config', 'obj', 'rally_id', 'x', 'y', 'tst', 'config_source']]
        
        if update_state:
            st.session_state.data = df
            st.session_state.uploaded = True
        
        if show_success:
            st.success(f"✅ Loaded {len(df)} points from {df['rally_id'].nunique()} trajectories "
                      f"across {df['config'].nunique()} configurations.")
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        return None


# =============================================================================
# DATA FILTERING
# =============================================================================

def aggregate_points(points, aggregation_type, temporal_resolution):
    """
    Aggregate trajectory points based on temporal resolution.
    
    Args:
        points: DataFrame with columns [x, y, tst]
        aggregation_type: 'none', 'mean', or 'median'
        temporal_resolution: Time window in seconds
        
    Returns:
        Aggregated DataFrame
    """
    if aggregation_type == "none":
        return points
    
    points = points.copy()
    points['time_bin'] = (points['tst'] / temporal_resolution).astype(int)
    
    if aggregation_type == "mean":
        aggregated = points.groupby('time_bin').agg({'x': 'mean', 'y': 'mean', 'tst': 'mean'}).reset_index(drop=True)
    else:  # median
        aggregated = points.groupby('time_bin').agg({'x': 'median', 'y': 'median', 'tst': 'median'}).reset_index(drop=True)
    
    return aggregated


def get_trajectory_coords(df, obj_id, config, start_time, end_time):
    """Extract trajectory coordinates for a specific object and configuration."""
    mask = (
        (df['obj'] == obj_id) &
        (df['config'] == config) &
        (df['tst'] >= start_time) &
        (df['tst'] <= end_time)
    )
    traj = df[mask][['x', 'y']].values
    return traj
