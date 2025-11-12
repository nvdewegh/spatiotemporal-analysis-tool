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
    
    Format 1: Standard long format (5-6 columns)
        - config, tst, obj, x, y [, config_name]
        - Each row is one point in space-time
        - Multiple rows per configuration/object
    
    Format 2: Wide format (many columns)
        - First column: Configuration ID (unique per row)
        - Remaining columns: (x, y) pairs for each time step
        - Example: config_id, x1, y1, x2, y2, x3, y3, ...
        - Each row represents a complete trajectory for one configuration
    
    Args:
        uploaded_file: File object from st.file_uploader
        update_state: Whether to update st.session_state
        show_success: Whether to show success message
    
    Returns:
        DataFrame with columns: config, obj, rally_id, x, y, tst, config_source
    """
    try:
        # Try reading with header first
        df_test = pd.read_csv(uploaded_file, nrows=5)
        uploaded_file.seek(0)
        
        has_header = not all(df_test.columns.str.match(r'^\d+$|^Unnamed:'))
        
        if has_header:
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file, header=None)
        
        # Detect format based on columns
        if len(df.columns) == 5:
            # Standard format: config, tst, obj, x, y
            if has_header:
                df.columns = ['config', 'tst', 'obj', 'x', 'y']
            else:
                df.columns = ['config', 'tst', 'obj', 'x', 'y']
            df['config_name'] = None
            
        elif len(df.columns) == 6:
            # Extended format could be: 
            # 1) config, tst, obj, x, y, config_name
            # 2) config, tst, obj, x, y, events (event-based data)
            
            # Check if the last column looks like event data (many NaN, short strings)
            last_col = df.iloc[:, 5]
            nan_ratio = last_col.isna().sum() / len(last_col)
            
            # If >50% NaN and contains string values, likely events column
            if nan_ratio > 0.5 and last_col.dtype == 'object':
                # This is event-based data - drop or store the events column
                if has_header:
                    df.columns = ['config', 'tst', 'obj', 'x', 'y', 'events']
                    # Keep events for later use but don't use as config_name
                    events_col = df['events'].copy()
                    df = df[['config', 'tst', 'obj', 'x', 'y']]
                    df['events'] = events_col
                else:
                    df.columns = ['config', 'tst', 'obj', 'x', 'y', 'events']
                    events_col = df['events'].copy()
                    df = df[['config', 'tst', 'obj', 'x', 'y']]
                    df['events'] = events_col
                df['config_name'] = None
            else:
                # Normal config_name column
                if has_header:
                    df.columns = ['config', 'tst', 'obj', 'x', 'y', 'config_name']
                else:
                    df.columns = ['config', 'tst', 'obj', 'x', 'y', 'config_name']
                
        else:
            # Check if first column contains configuration IDs (each row = different config)
            first_col_values = df.iloc[:, 0].values
            
            # Check if first column looks like sequential or unique configuration IDs
            is_config_format = False
            
            # Try numeric check first
            try:
                first_col_numeric = pd.to_numeric(df.iloc[:, 0], errors='coerce')
                # If most values are numeric and unique/sequential, treat as config IDs
                if first_col_numeric.notna().sum() > len(df) * 0.8:
                    unique_ratio = len(first_col_numeric.dropna().unique()) / len(first_col_numeric.dropna())
                    if unique_ratio > 0.8:  # Most values are unique
                        is_config_format = True
            except:
                pass
            
            if is_config_format:
                # Format: Each row is a configuration, first column is config ID
                # Remaining columns are trajectory data (likely alternating x,y pairs)
                st.info(f"🔍 Detected multi-configuration format: {len(df)} configurations with {len(df.columns)-1} data columns")
                
                # Extract config IDs
                config_ids = pd.to_numeric(df.iloc[:, 0], errors='coerce').fillna(0).astype(int)
                
                # Parse remaining columns as trajectory data
                data_cols = df.iloc[:, 1:]
                
                # Try to determine the pattern
                num_data_cols = len(data_cols.columns)
                
                # Assume pattern: repeating (x, y) pairs for multiple time steps/objects
                # Each configuration has the same temporal sequence
                if num_data_cols % 2 == 0:
                    # Even number of columns - assume (x, y) pairs
                    num_time_steps = num_data_cols // 2
                    
                    rows_list = []
                    for idx, config_id in enumerate(config_ids):
                        for t in range(num_time_steps):
                            x_col = t * 2
                            y_col = t * 2 + 1
                            x_val = data_cols.iloc[idx, x_col]
                            y_val = data_cols.iloc[idx, y_col]
                            
                            # Skip if both x and y are NaN
                            if pd.notna(x_val) and pd.notna(y_val):
                                rows_list.append({
                                    'config': config_id,
                                    'tst': t,
                                    'obj': 1,  # Assume single object per config
                                    'x': x_val,
                                    'y': y_val,
                                    'config_name': f"Config_{config_id}"
                                })
                    
                    if len(rows_list) == 0:
                        raise ValueError("No valid data points found in the file")
                    
                    df = pd.DataFrame(rows_list)
                    st.success(f"✅ Parsed {len(config_ids)} configurations with {num_time_steps} time steps each")
                else:
                    raise ValueError(f"Cannot parse {num_data_cols} data columns (expected even number for x,y pairs)")
            else:
                raise ValueError(f"Expected 5 or 6 columns, got {len(df.columns)}. "
                               f"For multi-configuration format, first column should contain unique configuration IDs.")
        
        # Convert types
        df['config'] = df['config'].astype(int)
        df['obj'] = df['obj'].astype(int)
        df['tst'] = pd.to_numeric(df['tst'], errors='coerce')
        df['x'] = pd.to_numeric(df['x'], errors='coerce')
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        
        # Handle config names
        if 'config_name' not in df.columns or df['config_name'].isna().all():
            unique_configs = sorted(df['config'].unique())
            config_map = {c: f"Config_{c}" for c in unique_configs}  # Use actual config ID
            df['config_name'] = df['config'].map(config_map)
        
        # Create rally_id and config_source
        df['rally_id'] = df.groupby(['config', 'obj']).ngroup()
        df['config_source'] = df['config_name']
        
        # Reorder columns - preserve events column if present
        base_columns = ['config', 'obj', 'rally_id', 'x', 'y', 'tst', 'config_source']
        if 'events' in df.columns:
            df = df[base_columns + ['events']]
        else:
            df = df[base_columns]
        
        if update_state:
            st.session_state.data = df
            st.session_state.uploaded = True
        
        if show_success:
            num_configs = df['config'].nunique()
            num_objects = df['obj'].nunique()
            st.success(f"✅ Loaded {len(df)} points from {df['rally_id'].nunique()} trajectories "
                      f"across {num_configs} configurations with {num_objects} objects.")
        
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
