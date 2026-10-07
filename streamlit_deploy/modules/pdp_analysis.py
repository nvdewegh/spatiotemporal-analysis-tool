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
# TENNIS COURT REFERENCE POINTS
# =============================================================================

# Tennis court dimensions (in meters) - matching streamlit_visualization.py
COURT_WIDTH = 8.23  # Singles court width
COURT_LENGTH = 23.77
DOUBLES_WIDTH = 10.97
DOUBLES_ALLEY_WIDTH = (DOUBLES_WIDTH - COURT_WIDTH) / 2  # 1.37m on each side
NET_POSITION = COURT_LENGTH / 2  # 11.885m
SERVICE_LINE_DISTANCE = 6.40  # Distance from net to service line
CENTER_X = COURT_WIDTH / 2  # 4.115m

# Service line positions
SERVICE_LINE_BOTTOM = NET_POSITION - SERVICE_LINE_DISTANCE  # ~5.485m
SERVICE_LINE_TOP = NET_POSITION + SERVICE_LINE_DISTANCE  # ~18.285m

# Predefined tennis court reference points
# Format: {name: (x, y, description)}
TENNIS_COURT_REFERENCE_POINTS = {
    # Court corners (singles)
    "Bottom-Left Corner": (0, 0, "Bottom-left corner of singles court"),
    "Bottom-Right Corner": (COURT_WIDTH, 0, "Bottom-right corner of singles court"),
    "Top-Left Corner": (0, COURT_LENGTH, "Top-left corner of singles court"),
    "Top-Right Corner": (COURT_WIDTH, COURT_LENGTH, "Top-right corner of singles court"),
    
    # Court corners (doubles)
    "Bottom-Left Doubles": (-DOUBLES_ALLEY_WIDTH, 0, "Bottom-left corner of doubles court"),
    "Bottom-Right Doubles": (COURT_WIDTH + DOUBLES_ALLEY_WIDTH, 0, "Bottom-right corner of doubles court"),
    "Top-Left Doubles": (-DOUBLES_ALLEY_WIDTH, COURT_LENGTH, "Top-left corner of doubles court"),
    "Top-Right Doubles": (COURT_WIDTH + DOUBLES_ALLEY_WIDTH, COURT_LENGTH, "Top-right corner of doubles court"),
    
    # Net positions
    "Net Center": (CENTER_X, NET_POSITION, "Center of the net"),
    "Net Left (Singles)": (0, NET_POSITION, "Left side of net at singles line"),
    "Net Right (Singles)": (COURT_WIDTH, NET_POSITION, "Right side of net at singles line"),
    "Net Left (Doubles)": (-DOUBLES_ALLEY_WIDTH, NET_POSITION, "Left side of net at doubles line"),
    "Net Right (Doubles)": (COURT_WIDTH + DOUBLES_ALLEY_WIDTH, NET_POSITION, "Right side of net at doubles line"),
    
    # Service box corners (bottom court - near baseline y=0)
    "Service Box BL-Bottom": (0, SERVICE_LINE_BOTTOM, "Service box bottom-left corner (bottom court)"),
    "Service Box BR-Bottom": (COURT_WIDTH, SERVICE_LINE_BOTTOM, "Service box bottom-right corner (bottom court)"),
    "Service Box Center-Bottom": (CENTER_X, SERVICE_LINE_BOTTOM, "Service box center line (bottom court)"),
    
    # Service box corners (top court - near baseline y=court_length)
    "Service Box TL-Top": (0, SERVICE_LINE_TOP, "Service box top-left corner (top court)"),
    "Service Box TR-Top": (COURT_WIDTH, SERVICE_LINE_TOP, "Service box top-right corner (top court)"),
    "Service Box Center-Top": (CENTER_X, SERVICE_LINE_TOP, "Service box center line (top court)"),
    
    # Baseline center marks
    "Center Mark Bottom": (CENTER_X, 0, "Center mark on bottom baseline"),
    "Center Mark Top": (CENTER_X, COURT_LENGTH, "Center mark on top baseline"),
    
    # Court center
    "Court Center": (CENTER_X, NET_POSITION, "Center of the court (same as net center)"),
}


def get_reference_point_names():
    """Get list of all reference point names for UI selection."""
    return list(TENNIS_COURT_REFERENCE_POINTS.keys())


def get_reference_point_coordinates(point_name):
    """Get coordinates for a named reference point.
    
    Args:
        point_name: Name of the reference point
        
    Returns:
        Tuple (x, y) or None if not found
    """
    if point_name in TENNIS_COURT_REFERENCE_POINTS:
        x, y, _ = TENNIS_COURT_REFERENCE_POINTS[point_name]
        return (x, y)
    return None


def get_reference_points_dict():
    """Get the full reference points dictionary for UI display."""
    return TENNIS_COURT_REFERENCE_POINTS


# =============================================================================
# EXTERNAL POINTS HANDLING
# =============================================================================

def add_external_points_to_data(df, external_points, timestamps):
    """
    Add external (reference) points to trajectory data.
    
    External points are static points that participate in PDP calculations.
    They are replicated for each timestamp to compare against moving objects.
    
    Args:
        df: DataFrame with trajectory data (must have columns: tst, obj, x, y, config_source)
        external_points: List of tuples [(name, x, y), ...] for external points
        timestamps: List of timestamps to replicate external points for
        
    Returns:
        DataFrame with external points added
    """
    if not external_points:
        return df
    
    # Create rows for external points
    external_rows = []
    
    for tst in timestamps:
        for i, (name, x, y) in enumerate(external_points):
            # External points get special object IDs to distinguish from regular objects
            # They also get a special marker to indicate they are external
            external_rows.append({
                'tst': tst,
                'obj': f'EXT_{i}',  # String ID for external points
                'x': x,
                'y': y,
                'config_source': df['config_source'].iloc[0] if len(df) > 0 else 'unknown',
                'is_external': True,
                'external_name': name,
                'sub_type': 'external',
                'sub_order': 100 + i  # High sub_order to sort after regular points
            })
    
    if not external_rows:
        return df
    
    # Add columns to original data if not present and fill NaN values
    df_copy = df.copy()
    if 'is_external' not in df_copy.columns:
        df_copy['is_external'] = False
    else:
        df_copy['is_external'] = df_copy['is_external'].fillna(False)
    if 'external_name' not in df_copy.columns:
        df_copy['external_name'] = ''
    else:
        df_copy['external_name'] = df_copy['external_name'].fillna('')
    if 'sub_type' not in df_copy.columns:
        df_copy['sub_type'] = 'orig'
    else:
        df_copy['sub_type'] = df_copy['sub_type'].fillna('orig')
    if 'sub_order' not in df_copy.columns:
        df_copy['sub_order'] = 0  # Regular points get sub_order 0
    else:
        df_copy['sub_order'] = df_copy['sub_order'].fillna(0)
    
    # Convert obj to string for consistent sorting with external points
    df_copy['obj'] = df_copy['obj'].astype(str)
        
    # Combine original data with external points
    external_df = pd.DataFrame(external_rows)
    
    # Get all unique columns from both dataframes
    df_columns = set(df_copy.columns)
    ext_columns = set(external_df.columns)
    
    # Add missing columns to df_copy (columns in external_df but not in df_copy)
    for col in ext_columns - df_columns:
        df_copy[col] = None
    
    # Add missing columns to external_df (columns in df_copy but not in external_df)
    for col in df_columns - ext_columns:
        external_df[col] = None
    
    # Now concatenate - both have the same columns
    combined_df = pd.concat([df_copy, external_df], ignore_index=True)
    
    # Fill any remaining NaN values in key columns
    combined_df['is_external'] = combined_df['is_external'].fillna(False)
    combined_df['external_name'] = combined_df['external_name'].fillna('')
    combined_df['sub_type'] = combined_df['sub_type'].fillna('orig')
    combined_df['sub_order'] = combined_df['sub_order'].fillna(0)
    
    return combined_df


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


def _point_sort_key(column):
    """
    Sort key for point ordering: object IDs sort numerically even when stored as text
    (external points turn 'obj' into strings), with external points 'EXT_i' after real objects.
    """
    if column.name != 'obj':
        return column
    text_ids = sorted({str(v) for v in column})
    def key(value):
        text = str(value)
        if text.startswith('EXT_') and text[4:].isdigit():
            return 2e15 + int(text[4:])
        try:
            return float(value)
        except (TypeError, ValueError):
            return 1e15 + text_ids.index(text)
    return column.map(key)


def sort_points(df, cols):
    """Sort points by the given columns using _point_sort_key (shared by computation and display)."""
    return df.sort_values(cols, key=_point_sort_key, kind='mergesort')


def compute_pdp_distance_pair(config1_data, config2_data, window_length, rough_x=0, rough_y=0):
    """
    Compute PDP distance between two configurations.
    
    This follows the original N_PDP.py algorithm:
    1. For each time window, build inequality matrices for X and Y dimensions
    2. Sum absolute differences across all windows for each dimension
    3. Return raw sum of X + Y distances (normalization happens separately)
    
    Args:
        config1_data: DataFrame with columns ['tst', 'obj', 'x', 'y'] for first config
        config2_data: DataFrame with columns ['tst', 'obj', 'x', 'y'] for second config
        window_length: Window length for temporal analysis (av.window_length_tst in original)
        rough_x: Roughness tolerance for x dimension
        rough_y: Roughness tolerance for y dimension
    
    Returns:
        Raw PDP distance (sum of X and Y absolute differences in inequality matrices)
    """
    # Get unique timestamps and objects
    timestamps1 = sorted(config1_data['tst'].unique())
    timestamps2 = sorted(config2_data['tst'].unique())
    
    # Determine number of time windows (same as original: av.tst-(av.window_length_tst-1))
    n_windows = min(len(timestamps1), len(timestamps2)) - window_length + 1
    if n_windows <= 0:
        return 0  # Not enough data
    
    # Calculate points per timestamp based on actual data structure
    first_timestamp = timestamps1[0]
    points_per_timestamp1 = len(config1_data[config1_data['tst'] == first_timestamp])
    first_timestamp2 = timestamps2[0]
    points_per_timestamp2 = len(config2_data[config2_data['tst'] == first_timestamp2])
    
    # Both configs must have the same structure for meaningful comparison
    if points_per_timestamp1 != points_per_timestamp2:
        return 0
    
    # Points per window = poi * window_length_tst (as in original)
    points_per_window = points_per_timestamp1 * window_length
    
    abs_distance_x = 0
    abs_distance_y = 0
    
    # Loop over time windows (same as original: for tst_id in range(av.tst-(av.window_length_tst-1)))
    for t_idx in range(n_windows):
        # Get data for this time window
        window_times1 = timestamps1[t_idx:t_idx + window_length]
        window_times2 = timestamps2[t_idx:t_idx + window_length]
        
        # Sort by timestamp, then obj, then sub_order (if present) for consistent ordering
        sort_cols1 = ['tst', 'obj']
        sort_cols2 = ['tst', 'obj']
        if 'sub_order' in config1_data.columns:
            sort_cols1.append('sub_order')
        elif 'sub_type' in config1_data.columns:
            sort_cols1.append('sub_type')
        if 'sub_order' in config2_data.columns:
            sort_cols2.append('sub_order')
        elif 'sub_type' in config2_data.columns:
            sort_cols2.append('sub_type')
            
        window_data1 = sort_points(config1_data[config1_data['tst'].isin(window_times1)], sort_cols1)
        window_data2 = sort_points(config2_data[config2_data['tst'].isin(window_times2)], sort_cols2)

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
        
        # Accumulate absolute differences (same as original nested loops)
        abs_distance_x += np.sum(np.abs(ineq_x1 - ineq_x2))
        abs_distance_y += np.sum(np.abs(ineq_y1 - ineq_y2))
    
    # Return raw sum of X and Y distances
    # The original algorithm returns X + Y (then normalizes separately or averages normalized values)
    total_distance = abs_distance_x + abs_distance_y
    
    return total_distance


def _data_fingerprint(df):
    """Cheap content hash of the columns PDP uses, so the cache notices a different dataset."""
    cols = [c for c in ('config_source', 'tst', 'obj', 'x', 'y') if c in df.columns]
    return int(pd.util.hash_pandas_object(df[cols], index=False).sum())


def compute_pdp_distance_matrix(_df, selected_configs, selected_objects, start_time, end_time,
                                window_length=3, buffer_x=0, buffer_y=0, rough_x=0, rough_y=0,
                                pdp_variant="fundamental", external_points=None):
    """
    Compute PDP distance matrix for all selected configurations (cached).

    The DataFrame itself is excluded from the cache key, but a fingerprint of its
    content is included, so loading a different dataset never returns stale distances.
    Arguments and result are as for _compute_pdp_distance_matrix_cached.
    """
    return _compute_pdp_distance_matrix_cached(
        _df, _data_fingerprint(_df), selected_configs, selected_objects, start_time, end_time,
        window_length, buffer_x, buffer_y, rough_x, rough_y, pdp_variant, external_points)


@st.cache_data
def _compute_pdp_distance_matrix_cached(_df, data_fingerprint, selected_configs, selected_objects, start_time, end_time,
                                        window_length=3, buffer_x=0, buffer_y=0, rough_x=0, rough_y=0,
                                        pdp_variant="fundamental", external_points=None):
    """
    Compute PDP distance matrix for all selected configurations.

    Note: _df has underscore prefix to exclude it from cache key (large DataFrames
    can cause hashing issues); data_fingerprint identifies its content instead.

    Args:
        _df: DataFrame with trajectory data (excluded from cache key)
        selected_configs: List of configuration IDs to analyze (tuple for reliable hashing)
        selected_objects: List of object IDs to analyze (tuple for reliable hashing)
        start_time: Start timestamp
        end_time: End timestamp
        window_length: Temporal window length for PDP analysis
        buffer_x: Buffer distance for x dimension (for buffer variant)
        buffer_y: Buffer distance for y dimension (for buffer variant)
        rough_x: Roughness tolerance for x dimension (for rough variant)
        rough_y: Roughness tolerance for y dimension (for rough variant)
        pdp_variant: One of "fundamental", "buffer", "rough", "buffer_rough"
        external_points: Tuple of tuples ((name, x, y), ...) for static reference points (must be tuple for caching)
    
    Returns:
        (distance_matrix, config_ids)
    """
    # Convert external_points tuple back to list for processing
    external_points_list = list(external_points) if external_points else None
    
    # Filter data (use _df which is excluded from cache key)
    filtered_df = _df[
        (_df['config_source'].isin(selected_configs)) &
        (_df['obj'].isin(selected_objects)) &
        (_df['tst'] >= start_time) &
        (_df['tst'] <= end_time)
    ].copy()
    
    config_ids = selected_configs
    n_configs = len(config_ids)
    
    # Get unique timestamps for external points
    all_timestamps = sorted(filtered_df['tst'].unique())
    
    # Add external points to data (before buffer transformation)
    if external_points_list:
        # For each config, add external points to its data
        config_dfs = []
        for config_id in config_ids:
            config_data = filtered_df[filtered_df['config_source'] == config_id].copy()
            config_timestamps = sorted(config_data['tst'].unique())
            config_data_with_ext = add_external_points_to_data(config_data, external_points_list, config_timestamps)
            config_dfs.append(config_data_with_ext)
        filtered_df = pd.concat(config_dfs, ignore_index=True)
    
    # Apply buffer if needed
    if pdp_variant in ["buffer", "buffer_rough"]:
        filtered_df = apply_buffer_to_trajectories(filtered_df, buffer_x, buffer_y)
    
    # Set roughness based on variant
    rough_x_val = rough_x if pdp_variant in ["rough", "buffer_rough"] else 0
    rough_y_val = rough_y if pdp_variant in ["rough", "buffer_rough"] else 0
    
    # Initialize distance matrix (use float for raw distances)
    distance_matrix = np.zeros((n_configs, n_configs), dtype=float)
    
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
    - (x - buffer_x, y)  [left]
    - (x + buffer_x, y)  [right]
    - (x, y - buffer_y)  [bottom]
    - (x, y + buffer_y)  [top]
    
    The points are ordered to match the original N_T_OB.py implementation:
    - Index 0: left (x - buffer_x)
    - Index 1: right (x + buffer_x)
    - Index 2: orig (original point)
    - Index 3: bottom (y - buffer_y)
    - Index 4: top (y + buffer_y)
    
    Args:
        df: DataFrame with trajectory data
        buffer_x: Buffer distance for x dimension
        buffer_y: Buffer distance for y dimension
    
    Returns:
        Expanded DataFrame with buffer points
    """
    if buffer_x == 0 and buffer_y == 0:
        # Add sub_type and sub_order columns for consistency
        df_copy = df.copy()
        df_copy['sub_type'] = 'orig'
        df_copy['sub_order'] = 0
        return df_copy
    
    buffer_points = []
    
    for _, row in df.iterrows():
        # Following the order from N_T_OB.py:
        # 0: left (x - buffer_x), 1: right (x + buffer_x), 2: orig, 3: bottom (y - buffer_y), 4: top (y + buffer_y)
        
        # Add buffer points for x dimension first (if active)
        if buffer_x > 0:
            # Index 0: Left buffer point
            left_point = row.to_dict()
            left_point['x'] = row['x'] - buffer_x
            left_point['sub_type'] = 'left'
            left_point['sub_order'] = 0
            buffer_points.append(left_point)
            
            # Index 1: Right buffer point
            right_point = row.to_dict()
            right_point['x'] = row['x'] + buffer_x
            right_point['sub_type'] = 'right'
            right_point['sub_order'] = 1
            buffer_points.append(right_point)
        
        # Index 2: Original point
        p = row.to_dict()
        p['sub_type'] = 'orig'
        p['sub_order'] = 2
        buffer_points.append(p)
        
        # Add buffer points for y dimension (if active)
        if buffer_y > 0:
            # Index 3: Bottom buffer point
            bottom_point = row.to_dict()
            bottom_point['y'] = row['y'] - buffer_y
            bottom_point['sub_type'] = 'bottom'
            bottom_point['sub_order'] = 3
            buffer_points.append(bottom_point)
            
            # Index 4: Top buffer point
            top_point = row.to_dict()
            top_point['y'] = row['y'] + buffer_y
            top_point['sub_type'] = 'top'
            top_point['sub_order'] = 4
            buffer_points.append(top_point)
    
    return pd.DataFrame(buffer_points)


# =============================================================================
# POINT-MATRIX DISPLAY (object-major ordering)
# =============================================================================
# Inequality matrices are computed in time-major point order (all points at the
# first timestamp, then the next, ...). For display only, rows and columns are
# regrouped object by object ("object-major"), chronologically within each object,
# so self-precedence (diagonal blocks) and mutual precedence (off-diagonal blocks)
# appear as contiguous blocks. The computed matrices themselves are never changed:
# the same permutation is applied to both dimensions of a copy, and to the labels.

INEQ_COLORS = ['#2ecc71', '#ffeb3b', '#e74c3c']   # 0: row < column, 1: equal, 2: row > column
DIFF_COLORS = ['#e3f2fd', '#ffcc80', '#e65100']   # |difference| 0: none, 1: adjacent, 2: opposite
INEQ_SYMBOLS = {0: '<', 1: '=', 2: '>'}
DIFF_MEANINGS = {0: 'no relation difference', 1: 'adjacent relation difference', 2: 'opposite relation difference'}
SUBPOINT_SUFFIX = {'left': '_L', 'right': '_R', 'bottom': '_B', 'top': '_U'}


def _discrete_colorscale(colors):
    """Three-step colorscale for integer values 0, 1, 2 (use with zmin=-0.5, zmax=2.5)."""
    return [[0, colors[0]], [1 / 3, colors[0]], [1 / 3, colors[1]],
            [2 / 3, colors[1]], [2 / 3, colors[2]], [1, colors[2]]]


def point_metadata(window_data, window_times):
    """
    Describe each point (row) of a window in the order used for computation.

    Args:
        window_data: DataFrame of the window's points, in computational order
        window_times: Sorted timestamps of the window

    Returns:
        List of dicts with obj, tst, t_rel, sub_type, sub_order, is_external, name
    """
    t_index = {t: i for i, t in enumerate(window_times)}
    meta = []
    for _, row in window_data.iterrows():
        obj = row['obj']
        is_ext = row.get('is_external', None)
        is_ext = str(obj).startswith('EXT_') if is_ext is None or pd.isna(is_ext) else bool(is_ext)
        sub_type = row.get('sub_type', None)
        sub_type = 'orig' if sub_type is None or pd.isna(sub_type) else str(sub_type)
        sub_order = row.get('sub_order', None)
        sub_order = 0.0 if sub_order is None or pd.isna(sub_order) else float(sub_order)
        name = row.get('external_name', None) if is_ext else None
        name = str(obj) if name is None or pd.isna(name) or str(name) == '' else str(name)
        meta.append({'obj': obj, 'tst': row['tst'], 't_rel': t_index[row['tst']], 'sub_type': sub_type,
                     'sub_order': sub_order, 'is_external': is_ext, 'name': name})
    return meta


def _object_sort_key(obj, is_external):
    """Real objects first (numeric IDs in numeric order), then external points by their index."""
    text = str(obj)
    if is_external and text.startswith('EXT_') and text[4:].isdigit():
        return (1, 0, float(text[4:]), text)
    try:
        return (int(is_external), 0, float(obj), text)
    except (TypeError, ValueError):
        return (int(is_external), 1, 0.0, text)


def object_major_display(meta, rough=0):
    """
    Build the object-major display order, labels and object-block boundaries.

    Points are grouped by object (real objects first, then each external point as its
    own group), then ordered by actual timestamp and by the existing subpoint order.
    Spatial coordinates are never used for ordering.

    Args:
        meta: Point metadata in computational order (from point_metadata)
        rough: Rough tolerance of the descriptor (only used for explanatory text)

    Returns:
        dict with 'perm' (computational index of each displayed position),
        'tick_labels', 'hover_labels' and 'groups' [(start, end, heading), ...]
    """
    perm = sorted(range(len(meta)), key=lambda i: (_object_sort_key(meta[i]['obj'], meta[i]['is_external']),
                                                   meta[i]['tst'], meta[i]['sub_order'], i))
    tick_labels, hover_labels, groups = [], [], []
    for pos, i in enumerate(perm):
        m = meta[i]
        suffix = SUBPOINT_SUFFIX.get(m['sub_type'], '')
        if m['is_external']:
            short_name = m['name'] if len(m['name']) <= 10 else m['name'][:8] + '..'
            tick_labels.append(f"E:{short_name}_T{m['t_rel']}{suffix}")
            text = f"External point '{m['name']}'"
            heading = f"Ext. {short_name}"
        else:
            tick_labels.append(f"O{m['obj']}_T{m['t_rel']}{suffix}")
            text = f"Object {m['obj']}"
            heading = f"Object {m['obj']}"
        text += f", T{m['t_rel']} (window position; input TST {m['tst']:g})"
        if m['sub_type'] not in ('orig', 'external'):
            text += f", {m['sub_type']} buffer point"
        hover_labels.append(text)
        key = (m['is_external'], str(m['obj']))
        if not groups or groups[-1][3] != key:
            groups.append([pos, pos + 1, heading, key])
        else:
            groups[-1][1] = pos + 1
    return {'perm': np.array(perm, dtype=int), 'tick_labels': tick_labels, 'hover_labels': hover_labels,
            'groups': [(g[0], g[1], g[2]) for g in groups], 'rough': rough}


def _relation_text(value, rough):
    if value == 1 and rough > 0:
        return f"= (equal within tolerance ±{rough:g})"
    return {0: '< (row smaller than column)', 1: '= (equal)', 2: '> (row greater than column)'}[int(value)]


def add_point_matrix_heatmap(fig, matrix, display, row, col, descriptor, kind='ineq', compared=None):
    """
    Add one PDP point matrix to a subplot in object-major order, first row at the top.

    Args:
        fig: Plotly figure with subplots
        matrix: Matrix in computational order (not modified)
        display: Result of object_major_display
        row, col: Subplot position
        descriptor: 'x' or 'y' (for hover text)
        kind: 'ineq' (relation 0/1/2) or 'diff' (absolute difference 0/1/2)
        compared: For kind='diff': ((label1, ineq1), (label2, ineq2)) in computational order
    """
    perm = display['perm']
    n = len(perm)
    z = np.asarray(matrix)[np.ix_(perm, perm)]
    rows_txt, cols_txt = display['hover_labels'], display['hover_labels']
    if kind == 'diff' and compared is not None:
        (name1, ineq1), (name2, ineq2) = compared
        ineq1 = np.asarray(ineq1)[np.ix_(perm, perm)]
        ineq2 = np.asarray(ineq2)[np.ix_(perm, perm)]
    hover = []
    for i in range(n):
        hover_row = []
        for j in range(n):
            text = f"Row: {rows_txt[i]}<br>Column: {cols_txt[j]}<br>Descriptor: {descriptor}<br>"
            if kind == 'ineq':
                text += f"Relation (row vs column): {_relation_text(z[i, j], display['rough'])}"
            else:
                text += f"|Difference| = {int(z[i, j])}: {DIFF_MEANINGS[int(z[i, j])]}"
                if compared is not None:
                    text += (f"<br>{name1}: {INEQ_SYMBOLS[int(ineq1[i, j])]}"
                             f" · {name2}: {INEQ_SYMBOLS[int(ineq2[i, j])]}")
            hover_row.append(text)
        hover.append(hover_row)

    fig.add_trace(go.Heatmap(
        z=z, x=list(range(n)), y=list(range(n)),
        colorscale=_discrete_colorscale(INEQ_COLORS if kind == 'ineq' else DIFF_COLORS),
        zmin=-0.5, zmax=2.5, showscale=False,
        customdata=hover, hovertemplate='%{customdata}<extra></extra>',
        xgap=1, ygap=1
    ), row=row, col=col)

    # Axis labels: columns above the matrix, rows on the left; sample labels for large matrices
    step = max(1, int(np.ceil(n / 30)))
    tickvals = list(range(0, n, step))
    ticktext = [display['tick_labels'][k] for k in tickvals]
    fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, side='top', tickangle=-45,
                     range=[-0.5, n - 0.5], showgrid=False, zeroline=False, tickfont=dict(size=9),
                     row=row, col=col)
    fig.update_yaxes(tickvals=tickvals, ticktext=ticktext, autorange='reversed',
                     showgrid=False, zeroline=False, tickfont=dict(size=9), row=row, col=col)

    # Object-block separators and block headings
    subplot = fig.get_subplot(row, col)
    xref = subplot.xaxis.plotly_name.replace('axis', '')
    yref = subplot.yaxis.plotly_name.replace('axis', '')
    for start, end, heading in display['groups']:
        if start > 0:
            for coords in ({'x0': start - 0.5, 'x1': start - 0.5, 'y0': -0.5, 'y1': n - 0.5},
                           {'x0': -0.5, 'x1': n - 0.5, 'y0': start - 0.5, 'y1': start - 0.5}):
                fig.add_shape(type='line', xref=xref, yref=yref, line=dict(color='black', width=2), **coords)
        centre = (start + end - 1) / 2
        fig.add_annotation(x=centre, y=n - 0.5, xref=xref, yref=yref, text=heading, showarrow=False,
                           yanchor='top', yshift=-4, font=dict(size=10))
        fig.add_annotation(x=n - 0.5, y=centre, xref=xref, yref=yref, text=heading, showarrow=False,
                           xanchor='left', xshift=4, textangle=90, font=dict(size=10))


def add_matrix_legend(fig, rough=0, include_diff=False):
    """Add a discrete legend for the relation colours (and difference colours if requested)."""
    entries = [(INEQ_COLORS[0], '< row smaller than column'),
               (INEQ_COLORS[1], '= equal' + (' (within rough tolerance)' if rough else '')),
               (INEQ_COLORS[2], '> row greater than column')]
    if include_diff:
        entries += [(DIFF_COLORS[k], f'|diff| = {k}: {DIFF_MEANINGS[k]}') for k in range(3)]
    for color, name in entries:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name=name, showlegend=True,
                                 marker=dict(symbol='square', size=14, color=color,
                                             line=dict(color='#666', width=1)),
                                 hoverinfo='skip'), row=1, col=1)


def visualize_inequality_matrices(df, config_ids, selected_objects, start_time, end_time,
                                   window_length=3, buffer_x=0, buffer_y=0, rough_x=0, rough_y=0,
                                   window_indices=None, external_points=None):
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
        window_indices: List of window indices to display (None = first window only)
        external_points: List of tuples [(name, x, y), ...] for static reference points
    
    Returns:
        Plotly figure with inequality matrix heatmaps, or dict with metadata if window_indices is None
    """
    from plotly.subplots import make_subplots
    
    # Filter data
    filtered_df = df[
        (df['tst'] >= start_time) &
        (df['tst'] <= end_time) &
        (df['obj'].isin(selected_objects))
    ].copy()
    
    # Ensure required columns exist with default values for original data
    if 'sub_type' not in filtered_df.columns:
        filtered_df['sub_type'] = 'orig'
    if 'sub_order' not in filtered_df.columns:
        filtered_df['sub_order'] = 0
    else:
        filtered_df['sub_order'] = filtered_df['sub_order'].fillna(0)
    if 'is_external' not in filtered_df.columns:
        filtered_df['is_external'] = False
    else:
        filtered_df['is_external'] = filtered_df['is_external'].fillna(False)
    if 'external_name' not in filtered_df.columns:
        filtered_df['external_name'] = ''
    else:
        filtered_df['external_name'] = filtered_df['external_name'].fillna('')
    
    # Fill any NaN in sub_type
    if 'sub_type' in filtered_df.columns:
        filtered_df['sub_type'] = filtered_df['sub_type'].fillna('orig')
    
    # Add external points to the filtered data (before buffer transformation)
    if external_points:
        # Process each config separately and collect results
        new_dfs = []
        for config_id in config_ids:
            config_data = filtered_df[filtered_df['config_source'] == config_id].copy()
            if len(config_data) > 0:
                config_timestamps = sorted(config_data['tst'].unique())
                ext_data = add_external_points_to_data(
                    config_data,
                    external_points,
                    config_timestamps
                )
                new_dfs.append(ext_data)
        
        # Also keep data for configs not in config_ids (if any)
        other_data = filtered_df[~filtered_df['config_source'].isin(config_ids)]
        if len(other_data) > 0:
            new_dfs.append(other_data)
        
        if new_dfs:
            filtered_df = pd.concat(new_dfs, ignore_index=True)
            # Remove any duplicate rows that might have been created
            # Keep the first occurrence based on key columns
            filtered_df = filtered_df.drop_duplicates(
                subset=['config_source', 'tst', 'obj', 'x', 'y', 'sub_order'],
                keep='first'
            ).reset_index(drop=True)
            
            # Ensure no NaN values in key columns after concat
            filtered_df['sub_type'] = filtered_df['sub_type'].fillna('orig')
            filtered_df['sub_order'] = filtered_df['sub_order'].fillna(0)
            filtered_df['is_external'] = filtered_df['is_external'].fillna(False)
            filtered_df['external_name'] = filtered_df['external_name'].fillna('')
    
    # If window_indices is None, return metadata about available windows
    if window_indices is None:
        # Return info about available windows per configuration
        window_info = {}
        for config_id in config_ids:
            config_data = filtered_df[filtered_df['config_source'] == config_id].copy()
            if len(config_data) > 0:
                timestamps = sorted(config_data['tst'].unique())
                max_windows = len(timestamps) - window_length + 1
                window_info[config_id] = {
                    'n_timestamps': len(timestamps),
                    'max_windows': max(0, max_windows),
                    'timestamps': timestamps
                }
        return window_info
    
    n_configs = len(config_ids)
    n_windows = len(window_indices)
    
    # Create subplots: 2 columns (X and Y) × (n_configs * n_windows) rows
    # Each config gets n_windows rows, one per selected time window
    subplot_titles = []
    for config_id in config_ids:
        for window_idx in window_indices:
            subplot_titles.extend([
                f"Config {config_id} - Window {window_idx} - X dimension",
                f"Config {config_id} - Window {window_idx} - Y dimension"
            ])
    
    total_rows = n_configs * n_windows
    row_heights = [1] * total_rows  # Equal weight for all rows
    
    # Vertical spacing: about 200 px between rows, for block headings below each matrix,
    # column labels above the next matrix and its subplot title
    vertical_spacing = min(200 / (760 * total_rows + 120), 0.9 / max(1, total_rows - 1))
    
    fig = make_subplots(
        rows=total_rows,
        cols=2,
        subplot_titles=subplot_titles,
        row_heights=row_heights,
        horizontal_spacing=0.12,
        vertical_spacing=vertical_spacing
    )
    
    current_row = 1
    
    for config_id in config_ids:
        config_data = filtered_df[filtered_df['config_source'] == config_id].copy()
        
        if len(config_data) == 0:
            current_row += n_windows
            continue
        
        # Ensure sub_order column exists for consistent sorting
        if 'sub_order' not in config_data.columns:
            config_data['sub_order'] = 0
        
        # Apply buffer if needed
        if buffer_x > 0 or buffer_y > 0:
            config_data = apply_buffer_to_trajectories(config_data, buffer_x, buffer_y)
        
        # Sort by tst, then obj, then sub_order: the same point order as the distance computation
        config_data = sort_points(config_data, ['tst', 'obj', 'sub_order'])
        
        # Get timestamps
        timestamps = sorted(config_data['tst'].unique())
        if len(timestamps) < window_length:
            current_row += n_windows
            continue
        
        # Process each selected window
        for window_idx in window_indices:
            # Check if window_idx is valid
            max_window_idx = len(timestamps) - window_length
            if window_idx > max_window_idx:
                current_row += 1
                continue
            
            # Get data for this time window
            window_times = timestamps[window_idx:window_idx + window_length]
            # Sort consistently with the distance computation: tst, then obj, then sub_order
            window_data = sort_points(config_data[config_data['tst'].isin(window_times)], ['tst', 'obj', 'sub_order'])
            
            x_vals = window_data['x'].values
            y_vals = window_data['y'].values
            
            # Compute inequality matrices
            ineq_x = compute_inequality_matrix(x_vals, x_vals, window_length, rough_x)
            ineq_y = compute_inequality_matrix(y_vals, y_vals, window_length, rough_y)
            
            # Display in object-major order (presentation only; ineq_x/ineq_y stay as computed)
            meta = point_metadata(window_data, list(window_times))
            display_x = object_major_display(meta, rough=rough_x)
            display_y = object_major_display(meta, rough=rough_y)
            add_point_matrix_heatmap(fig, ineq_x, display_x, current_row, 1, 'x')
            add_point_matrix_heatmap(fig, ineq_y, display_y, current_row, 2, 'y')

            current_row += 1

    # Each matrix gets a large, roughly square area so cells stay square and readable
    height_per_row = 760
    total_height = height_per_row * total_rows + 120
    width = 1400

    add_matrix_legend(fig, rough=max(rough_x, rough_y))
    fig.update_layout(
        title=dict(text=f"Inequality Matrices (window_length={window_length}) – each cell compares its row point with its column point",
                   yref='container', y=1, yanchor='top', pad=dict(t=12)),
        height=total_height,
        width=width,
        showlegend=True,
        legend=dict(orientation='h', yref='container', y=1 - 45 / total_height, yanchor='top',
                    xanchor='center', x=0.5, font=dict(size=11)),
        margin=dict(t=210)
    )
    # Lift subplot titles above the column labels, which sit on top of each matrix
    fig.update_annotations(selector=dict(xref='paper', yref='paper'), yshift=60)

    # Synchronise zooming across all matrices (all share the same point order and size)
    fig.update_xaxes(matches='x')
    fig.update_yaxes(matches='y', autorange='reversed')

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
    # Calculate color threshold if n_clusters is provided
    color_threshold = 0
    threshold_line_val = None
    
    if n_clusters is not None and n_clusters > 1:
        n_samples = len(linkage_matrix) + 1
        if n_clusters < n_samples:
            # Z[-(k-1), 2] is the merge distance that reduces k to k-1 clusters
            # Z[-k, 2] is the merge distance that reduces k+1 to k clusters
            dist_above = linkage_matrix[-(n_clusters-1), 2]
            dist_below = linkage_matrix[-n_clusters, 2]
            # Set threshold slightly below the merge that creates k-1 clusters
            # But typically dendrogram uses threshold < merge_height to keep clusters separate
            # If we set threshold = (dist_above + dist_below) / 2, it will cut the tree correctly
            threshold_line_val = (dist_above + dist_below) / 2
            color_threshold = threshold_line_val

    # Create dendrogram using scipy
    dend_data = dendrogram(linkage_matrix, labels=labels, no_plot=True, color_threshold=color_threshold)
    
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
    
    # Add horizontal cut-off line if applicable
    if threshold_line_val is not None:
        # Determine x range
        x_min = np.min(icoord)
        x_max = np.max(icoord)
        
        fig.add_shape(
            type="line",
            x0=x_min - 5, # Extend slightly
            y0=threshold_line_val,
            x1=x_max + 5, # Extend slightly
            y1=threshold_line_val,
            line=dict(
                color="Red",
                width=2,
                dash="dashdot",
            ),
        )
        
        fig.add_annotation(
            x=x_max,
            y=threshold_line_val,
            text=f"Cut-off: {threshold_line_val:.2f}",
            showarrow=False,
            yshift=10,
            xanchor="right",
            font=dict(color="Red")
        )

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
    
    # Calculate normalized stress (Kruskal's Stress-1)
    # sklearn's mds.stress_ is raw stress (sum of squared errors), not normalized
    # Normalized stress = sqrt(sum((d_orig - d_mds)^2) / sum(d_orig^2))
    from scipy.spatial.distance import pdist, squareform
    mds_distances = squareform(pdist(coords_2d))
    # Get upper triangle of distance matrices (excluding diagonal)
    triu_indices = np.triu_indices_from(distance_matrix, k=1)
    orig_dists = distance_matrix[triu_indices]
    mds_dists = mds_distances[triu_indices]
    # Kruskal's Stress-1 formula
    stress = np.sqrt(np.sum((orig_dists - mds_dists) ** 2) / np.sum(orig_dists ** 2))
    
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
    
    # Calculate normalized stress (Kruskal's Stress-1)
    # sklearn's mds.stress_ is raw stress (sum of squared errors), not normalized
    # Normalized stress = sqrt(sum((d_orig - d_mds)^2) / sum(d_orig^2))
    from scipy.spatial.distance import pdist, squareform
    mds_distances = squareform(pdist(coords_3d))
    # Get upper triangle of distance matrices (excluding diagonal)
    triu_indices = np.triu_indices_from(distance_matrix, k=1)
    orig_dists = distance_matrix[triu_indices]
    mds_dists = mds_distances[triu_indices]
    # Kruskal's Stress-1 formula
    stress = np.sqrt(np.sum((orig_dists - mds_dists) ** 2) / np.sum(orig_dists ** 2))
    
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
                               show_buffers=False, buffer_size=0.5, show_rough=False, 
                               rough_x=0.3, rough_y=0.3, show_court=True):
    """
    Compare trajectories of selected configurations (on a tennis court if show_court).
    
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
        show_rough: Whether to show rough tolerance zones (rectangles)
        rough_x: Rough tolerance for X dimension (half-width of rectangle in meters)
        rough_y: Rough tolerance for Y dimension (half-height of rectangle in meters)
        show_court: Whether to draw the tennis court (False = plain x/y axes)

    Returns:
        plotly.graph_objects.Figure: Tennis court with trajectories
    """
    # Handle object selection - if None or empty list, use empty list (show nothing)
    # Make a COPY of the list to avoid reference issues
    if selected_objects is None:
        selected_objects = []
    else:
        selected_objects = list(selected_objects)  # Create a copy
    
    # Create a BRAND NEW figure from scratch - don't reuse anything
    fig = go.Figure()
    
    # Add tennis court shapes directly (not using cached base)
    court_width = 8.23
    court_length = 23.77
    doubles_width = 10.97
    doubles_alley_width = (doubles_width - court_width) / 2
    service_line_distance = 6.40
    net_position = court_length / 2
    x_margin = 2.0
    y_margin = 3.0
    
    # Court boundary
    fig.add_shape(type="rect", x0=-doubles_alley_width, y0=0, 
                  x1=court_width + doubles_alley_width, y1=court_length,
                  line=dict(color="white", width=3))
    # Singles sidelines
    fig.add_shape(type="line", x0=0, y0=0, x1=0, y1=court_length,
                  line=dict(color="white", width=2))
    fig.add_shape(type="line", x0=court_width, y0=0, x1=court_width, y1=court_length,
                  line=dict(color="white", width=2))
    # Baselines
    fig.add_shape(type="line", x0=-doubles_alley_width, y0=0, 
                  x1=court_width + doubles_alley_width, y1=0,
                  line=dict(color="white", width=3))
    fig.add_shape(type="line", x0=-doubles_alley_width, y0=court_length, 
                  x1=court_width + doubles_alley_width, y1=court_length,
                  line=dict(color="white", width=3))
    # Net
    fig.add_shape(type="line", x0=-doubles_alley_width, y0=net_position, 
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
    
    # Set layout
    fig.update_layout(
        width=500, height=900,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(range=[-doubles_alley_width - x_margin, court_width + doubles_alley_width + x_margin],
                   showgrid=False, zeroline=False, title="Court Width (m)",
                   constrain='domain', fixedrange=False),
        yaxis=dict(range=[-y_margin, court_length + y_margin],
                   showgrid=False, zeroline=False, title="Court Length (m)",
                   scaleanchor="x", scaleratio=1, constrain='domain', fixedrange=False),
        plot_bgcolor='#25D366',
        showlegend=True,
        hovermode='closest',
        dragmode='pan'
    )

    # No court selected: remove the court lines and use plain axes fitted to the data
    if not show_court:
        fig.layout.shapes = ()
        fig.layout.width = None
        fig.update_layout(height=600, plot_bgcolor='white')
        fig.update_xaxes(range=None, autorange=True, showgrid=True, title="x (m)")
        fig.update_yaxes(range=None, autorange=True, showgrid=True, title="y (m)")

    # Color palette for configurations
    colors = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel
    
    # If cluster labels provided, use cluster colors
    if cluster_labels is not None:
        config_to_cluster = {config_ids[i]: cluster_labels[i] for i in range(len(config_ids))}
        cluster_colors = px.colors.qualitative.Bold
    
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
            
            # Define legend group for this object to ensure all parts hide/show together
            legend_group = f"{config}_{obj_id}"
            
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
                    legendgroup=legend_group,
                    hovertemplate="<b>Buffer Point</b><br>X: %{x:.2f}m<br>Y: %{y:.2f}m<extra></extra>"
                ))
            
            # Add rough tolerance ZONES if requested (visualize rough parameter)
            # Rough adds tolerance zones where comparisons are considered "equal"
            # Now uses RECTANGLES (axis-aligned) instead of circles
            if show_rough and (rough_x > 0 or rough_y > 0):
                for idx, row in obj_data.iterrows():
                    fig.add_shape(
                        type="rect",
                        xref="x", yref="y",
                        x0=row['x'] - rough_x,
                        y0=row['y'] - rough_y,
                        x1=row['x'] + rough_x,
                        y1=row['y'] + rough_y,
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
                legendgroup=legend_group,
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
                legendgroup=legend_group,
                hovertemplate=f"<b>START: {config} - Obj {obj_id}</b><extra></extra>"
            ))
            
            fig.add_trace(go.Scatter(
                x=[obj_data.iloc[-1]['x']],
                y=[obj_data.iloc[-1]['y']],
                mode='markers',
                marker=dict(size=12, color=color, symbol='square',
                           line=dict(color='white', width=2)),
                showlegend=False,
                legendgroup=legend_group,
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
                  f"Similarity: {similarity:.1f}%</sub>",
            uirevision=f"traj-{'-'.join(map(str, selected_configs))}-{'-'.join(map(str, selected_objects))}"
        )
    else:
        fig.update_layout(
            title=f"Trajectory Comparison<br>" +
                  f"<sub>{len(selected_configs)} configurations, " +
                  f"Time: TST {start_time:g} - TST {end_time:g}</sub>",
            uirevision=f"traj-{'-'.join(map(str, selected_configs))}-{'-'.join(map(str, selected_objects))}"
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



def compute_distance_normalization_info(distance_matrix, config_ids, n_objects=None, window_length=None,
                                        buffer_factor=1, n_external_points=0, n_windows=None):
    """
    Compute normalized distances and statistics for educational purposes.
    
    Args:
        distance_matrix: Raw PDP distance matrix (n x n)
        config_ids: List of configuration IDs
        n_objects: Number of original objects (before buffer/external points)
        window_length: Temporal window length used in PDP computation
        buffer_factor: Multiplier for buffer points (1=no buffer, 5=full x+y buffer)
        n_external_points: Number of external reference points added
        n_windows: Number of time windows used in computation
    
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
    # For PDP: each cell in inequality matrix can differ by at most 2 (0 vs 2 or 2 vs 0)
    # The inequality matrix is points_per_window x points_per_window
    # We have both X and Y dimensions
    
    if n_objects is not None and window_length is not None and n_windows is not None:
        # Calculate theoretical maximum based on actual parameters
        # Points per timestamp:
        # - Moving objects: n_objects * buffer_factor (with buffer points)
        # - External (static) points: n_external_points
        n_moving_points = n_objects * buffer_factor
        n_static_points = n_external_points
        
        # Points in one window
        moving_per_window = n_moving_points * window_length
        static_per_window = n_static_points * window_length
        total_per_window = moving_per_window + static_per_window
        
        # IMPORTANT: External points have SAME coordinates across all configurations!
        # Therefore, comparisons between external points will always be 0 (no difference).
        # Only these comparisons can actually differ between configurations:
        # 1. Moving vs Moving (both off-diagonal): moving^2 - moving (exclude diagonal)
        # 2. Moving vs Static (both directions): 2 * moving * static
        # 3. Static vs Static: ALWAYS 0 (same in all configs) - DO NOT COUNT
        
        # Comparisons that CAN differ:
        moving_vs_moving = moving_per_window * moving_per_window - moving_per_window  # exclude diagonal
        moving_vs_static = 2 * moving_per_window * static_per_window  # both directions (i,j) and (j,i)
        
        variable_comparisons = moving_vs_moving + moving_vs_static
        
        # Max diff per comparison: 2 (for 0 vs 2)
        # We have both X and Y dimensions
        max_diff_per_window = 2 * 2 * variable_comparisons  # factor of 2 for X and Y
        
        # Total max across all windows
        max_possible = max_diff_per_window * n_windows
    else:
        # Fallback to empirical max when parameters not provided
        max_possible = distance_matrix.max()
    
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
        'fundamental': 'Fundamental',
        'buffer': 'Buffer',
        'rough': 'Rough',
        'buffer_rough': 'Buffer + Rough'
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
    
    labels = ['Fundamental', 'Buffer', 'Rough', 'Buffer + Rough']
    
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
            'Silhouette Score (Higher = Better)',
            'Davies-Bouldin Index (Lower = Better)',
            'Calinski-Harabasz Score (Higher = Better)',
            'Inertia / WCSS (Lower = Better, Elbow Method)'
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


# =============================================================================
# FINE-GRAINED PINPOINT DIFFERENCES
# =============================================================================

def compute_pairwise_inequality_differences(config1_data, config2_data, window_length, rough_x=0, rough_y=0):
    """
    Compute detailed inequality matrix differences between two configurations.
    
    Returns the raw differences per time window and per object pair, so we can
    visualize exactly where the differences occur.
    
    Args:
        config1_data: DataFrame with columns ['tst', 'obj', 'x', 'y'] for first config
        config2_data: DataFrame with columns ['tst', 'obj', 'x', 'y'] for second config
        window_length: Window length for temporal analysis
        rough_x: Roughness tolerance for x dimension
        rough_y: Roughness tolerance for y dimension
    
    Returns:
        dict with:
        - 'windows': list of window indices
        - 'differences': list of dicts per window, each containing:
            - 'x_diff_matrix': absolute difference matrix for X
            - 'y_diff_matrix': absolute difference matrix for Y
            - 'total_diff_matrix': combined difference matrix
            - 'timestamps': list of timestamps in this window
            - 'objects': list of object IDs in order
            - 'positions1': dict mapping (obj, tst) -> (x, y) for config1
            - 'positions2': dict mapping (obj, tst) -> (x, y) for config2
    """
    # Get unique timestamps and objects
    timestamps1 = sorted(config1_data['tst'].unique())
    timestamps2 = sorted(config2_data['tst'].unique())
    
    # Determine number of time windows
    n_windows = min(len(timestamps1), len(timestamps2)) - window_length + 1
    if n_windows <= 0:
        return {'windows': [], 'differences': []}
    
    # Get objects consistently
    objects1 = sorted(config1_data['obj'].unique())
    objects2 = sorted(config2_data['obj'].unique())
    
    # Objects must match for meaningful comparison
    if set(objects1) != set(objects2):
        return {'windows': [], 'differences': [], 'error': 'Objects do not match between configurations'}
    
    objects = objects1  # They're the same
    
    result = {
        'windows': list(range(n_windows)),
        'differences': [],
        'objects': objects,
        'rough_x': rough_x,
        'rough_y': rough_y
    }
    
    # Process each time window
    for t_idx in range(n_windows):
        window_times1 = timestamps1[t_idx:t_idx + window_length]
        window_times2 = timestamps2[t_idx:t_idx + window_length]
        
        # Get data for this window, sorted consistently
        sort_cols = ['tst', 'obj']
        if 'sub_order' in config1_data.columns:
            sort_cols.append('sub_order')
            
        window_data1 = sort_points(config1_data[config1_data['tst'].isin(window_times1)], sort_cols)
        window_data2 = sort_points(config2_data[config2_data['tst'].isin(window_times2)], sort_cols)
        
        if len(window_data1) == 0 or len(window_data2) == 0:
            continue
        
        # Build position mappings (obj, tst) -> (x, y) from the original points only,
        # so buffer subpoints of the same point do not overwrite its true position
        positions1 = {}
        positions2 = {}

        for positions, window_data in ((positions1, window_data1), (positions2, window_data2)):
            for _, row in window_data.iterrows():
                key = (row['obj'], row['tst'])
                sub_type = row.get('sub_type', 'orig')
                if key not in positions or sub_type in ('orig', 'external') or pd.isna(sub_type):
                    positions[key] = (row['x'], row['y'])
        
        # Compute inequality matrices
        x_vals1 = window_data1['x'].values
        x_vals2 = window_data2['x'].values
        y_vals1 = window_data1['y'].values
        y_vals2 = window_data2['y'].values
        
        if len(x_vals1) != len(x_vals2):
            continue
        
        ineq_x1 = compute_inequality_matrix(x_vals1, x_vals1, window_length, rough_x)
        ineq_x2 = compute_inequality_matrix(x_vals2, x_vals2, window_length, rough_x)
        ineq_y1 = compute_inequality_matrix(y_vals1, y_vals1, window_length, rough_y)
        ineq_y2 = compute_inequality_matrix(y_vals2, y_vals2, window_length, rough_y)
        
        # Compute differences
        x_diff = np.abs(ineq_x1 - ineq_x2)
        y_diff = np.abs(ineq_y1 - ineq_y2)
        total_diff = x_diff + y_diff
        
        # Store row/column labels (object-timestamp pairs)
        labels = []
        for _, row in window_data1.iterrows():
            t_relative = list(window_times1).index(row['tst'])
            labels.append((row['obj'], row['tst'], t_relative))
        
        result['differences'].append({
            'window_idx': t_idx,
            'x_diff_matrix': x_diff,
            'y_diff_matrix': y_diff,
            'total_diff_matrix': total_diff,
            'ineq_x1': ineq_x1,
            'ineq_x2': ineq_x2,
            'ineq_y1': ineq_y1,
            'ineq_y2': ineq_y2,
            'timestamps': list(window_times1),
            'labels': labels,
            # Full point metadata (incl. buffer subpoints / external points) for display
            'point_meta1': point_metadata(window_data1, list(window_times1)),
            'point_meta2': point_metadata(window_data2, list(window_times2)),
            'positions1': positions1,
            'positions2': positions2
        })
    
    return result


def create_difference_visualization(df, config1_id, config2_id, selected_objects, 
                                     start_time, end_time, window_length,
                                     buffer_x=0, buffer_y=0, rough_x=0, rough_y=0,
                                     external_points=None, show_court=True):
    """
    Create side-by-side visualization (on tennis courts if show_court) showing where two configurations differ.
    
    Draws difference lines between objects at time points where the inequality matrices differ.
    Line thickness indicates the magnitude of difference.
    
    Args:
        df: DataFrame with trajectory data
        config1_id: First configuration ID
        config2_id: Second configuration ID
        selected_objects: List of object IDs to analyze
        start_time, end_time: Time range
        window_length: PDP window length
        buffer_x, buffer_y: Buffer parameters
        rough_x, rough_y: Rough parameters
        external_points: List of external reference points
    
    Returns:
        tuple: (fig1, fig2, diff_summary) - Two tennis court figures and summary stats
    """
    from plotly.subplots import make_subplots
    
    # Filter data for each configuration
    config1_data = df[
        (df['config_source'] == config1_id) &
        (df['obj'].isin(selected_objects)) &
        (df['tst'] >= start_time) &
        (df['tst'] <= end_time)
    ].copy()
    
    config2_data = df[
        (df['config_source'] == config2_id) &
        (df['obj'].isin(selected_objects)) &
        (df['tst'] >= start_time) &
        (df['tst'] <= end_time)
    ].copy()
    
    if len(config1_data) == 0 or len(config2_data) == 0:
        return None, None, {'error': 'No data for selected configurations'}
    
    # Add external points if specified
    if external_points:
        timestamps1 = sorted(config1_data['tst'].unique())
        timestamps2 = sorted(config2_data['tst'].unique())
        config1_data = add_external_points_to_data(config1_data, external_points, timestamps1)
        config2_data = add_external_points_to_data(config2_data, external_points, timestamps2)
    
    # Apply buffer if needed
    if buffer_x > 0 or buffer_y > 0:
        config1_data = apply_buffer_to_trajectories(config1_data, buffer_x, buffer_y)
        config2_data = apply_buffer_to_trajectories(config2_data, buffer_x, buffer_y)
    
    # Compute detailed differences
    diff_result = compute_pairwise_inequality_differences(
        config1_data, config2_data, window_length, rough_x, rough_y
    )
    
    if not diff_result['differences']:
        return None, None, {'error': 'Could not compute differences'}
    
    # Create two tennis court figures side by side
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f"Configuration: {config1_id}", f"Configuration: {config2_id}"],
        horizontal_spacing=0.08
    )
    
    # Court dimensions
    court_width = 8.23
    court_length = 23.77
    doubles_width = 10.97
    doubles_alley_width = (doubles_width - court_width) / 2
    service_line_distance = 6.40
    net_position = court_length / 2
    
    # Add court shapes to both subplots
    for col in ([1, 2] if show_court else []):
        # Outer boundary
        fig.add_shape(type="rect", 
                     x0=-doubles_alley_width, y0=0,
                     x1=court_width + doubles_alley_width, y1=court_length,
                     line=dict(color="white", width=2),
                     row=1, col=col)
        # Singles sidelines
        fig.add_shape(type="line", x0=0, y0=0, x1=0, y1=court_length,
                     line=dict(color="white", width=1), row=1, col=col)
        fig.add_shape(type="line", x0=court_width, y0=0, x1=court_width, y1=court_length,
                     line=dict(color="white", width=1), row=1, col=col)
        # Net
        fig.add_shape(type="line", 
                     x0=-doubles_alley_width, y0=net_position,
                     x1=court_width + doubles_alley_width, y1=net_position,
                     line=dict(color="white", width=2), row=1, col=col)
        # Service lines
        fig.add_shape(type="line", x0=0, y0=net_position - service_line_distance,
                     x1=court_width, y1=net_position - service_line_distance,
                     line=dict(color="white", width=1), row=1, col=col)
        fig.add_shape(type="line", x0=0, y0=net_position + service_line_distance,
                     x1=court_width, y1=net_position + service_line_distance,
                     line=dict(color="white", width=1), row=1, col=col)
        # Center line
        fig.add_shape(type="line", x0=court_width/2, y0=net_position - service_line_distance,
                     x1=court_width/2, y1=net_position + service_line_distance,
                     line=dict(color="white", width=1), row=1, col=col)
    
    # Color palette for objects
    object_colors = px.colors.qualitative.Set1
    obj_list = sorted(selected_objects)
    obj_color_map = {obj: object_colors[i % len(object_colors)] for i, obj in enumerate(obj_list)}
    
    # Track differences for summary
    diff_summary = {
        'total_cell_differences': 0,
        'windows_with_differences': 0,
        'max_difference_window': None,
        'max_difference_value': 0,
        'difference_by_window': []
    }
    
    # Process each window and accumulate differences per object-pair-timestamp
    # We'll aggregate differences to show on the court
    position_differences = {}  # key: (obj_i, tst_i, obj_j, tst_j) -> total_diff
    
    for window_data in diff_result['differences']:
        total_diff_matrix = window_data['total_diff_matrix']
        labels = window_data['labels']
        positions1 = window_data['positions1']
        positions2 = window_data['positions2']
        
        window_diff_sum = np.sum(total_diff_matrix)
        diff_summary['total_cell_differences'] += window_diff_sum
        diff_summary['difference_by_window'].append({
            'window': window_data['window_idx'],
            'total_diff': window_diff_sum
        })
        
        if window_diff_sum > 0:
            diff_summary['windows_with_differences'] += 1
            if window_diff_sum > diff_summary['max_difference_value']:
                diff_summary['max_difference_value'] = window_diff_sum
                diff_summary['max_difference_window'] = window_data['window_idx']
        
        # For each cell with a difference, record it
        n = len(labels)
        for i in range(n):
            for j in range(i + 1, n):  # Upper triangle only to avoid duplicates
                diff_val = total_diff_matrix[i, j]
                if diff_val > 0:
                    obj_i, tst_i, _ = labels[i]
                    obj_j, tst_j, _ = labels[j]
                    
                    # Store the difference keyed by the objects and timestamps involved
                    key = (obj_i, tst_i, obj_j, tst_j)
                    if key not in position_differences:
                        position_differences[key] = {
                            'diff': 0,
                            'pos1_i': positions1.get((obj_i, tst_i)),
                            'pos1_j': positions1.get((obj_j, tst_j)),
                            'pos2_i': positions2.get((obj_i, tst_i)),
                            'pos2_j': positions2.get((obj_j, tst_j))
                        }
                    position_differences[key]['diff'] += diff_val
    
    # Plot trajectories for both configurations
    # Get unique original data (before buffer) for trajectory plotting
    config1_orig = df[
        (df['config_source'] == config1_id) &
        (df['obj'].isin(selected_objects)) &
        (df['tst'] >= start_time) &
        (df['tst'] <= end_time)
    ].copy()
    
    config2_orig = df[
        (df['config_source'] == config2_id) &
        (df['obj'].isin(selected_objects)) &
        (df['tst'] >= start_time) &
        (df['tst'] <= end_time)
    ].copy()
    
    # Plot trajectories for config 1
    for obj in obj_list:
        obj_data = config1_orig[config1_orig['obj'] == obj].sort_values('tst')
        if len(obj_data) > 0:
            # Create custom hover text with timestamp
            hover_texts = [f'Object {obj}<br>x: {x:.2f}<br>y: {y:.2f}<br>t: {t:.2f}s' 
                          for x, y, t in zip(obj_data['x'], obj_data['y'], obj_data['tst'])]
            fig.add_trace(go.Scatter(
                x=obj_data['x'],
                y=obj_data['y'],
                mode='lines+markers',
                name=f'Object {obj}',
                line=dict(color=obj_color_map[obj], width=2),
                marker=dict(size=6),
                legendgroup=f'obj_{obj}',
                showlegend=True,
                hoverinfo='text',
                hovertext=hover_texts
            ), row=1, col=1)
    
    # Plot trajectories for config 2
    for obj in obj_list:
        obj_data = config2_orig[config2_orig['obj'] == obj].sort_values('tst')
        if len(obj_data) > 0:
            # Create custom hover text with timestamp
            hover_texts = [f'Object {obj}<br>x: {x:.2f}<br>y: {y:.2f}<br>t: {t:.2f}s' 
                          for x, y, t in zip(obj_data['x'], obj_data['y'], obj_data['tst'])]
            fig.add_trace(go.Scatter(
                x=obj_data['x'],
                y=obj_data['y'],
                mode='lines+markers',
                name=f'Object {obj}',
                line=dict(color=obj_color_map[obj], width=2),
                marker=dict(size=6),
                legendgroup=f'obj_{obj}',
                showlegend=False,
                hoverinfo='text',
                hovertext=hover_texts
            ), row=1, col=2)
    
    # Draw difference lines
    # For each difference, draw a line connecting the two points involved
    # Line thickness proportional to difference magnitude
    # Use unique colors for each difference to help visually match lines between subplots
    max_diff = max([v['diff'] for v in position_differences.values()]) if position_differences else 1
    
    # Color palette for difference lines - distinct colors to help match between subplots
    diff_colors = [
        '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
        '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
        '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
        '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080'
    ]
    
    # Sort differences by magnitude (largest first) for consistent numbering
    sorted_diffs = sorted(position_differences.items(), key=lambda x: x[1]['diff'], reverse=True)
    
    for diff_idx, (key, diff_data) in enumerate(sorted_diffs):
        obj_i, tst_i, obj_j, tst_j = key
        diff_val = diff_data['diff']
        
        # Assign unique color based on index
        diff_color = diff_colors[diff_idx % len(diff_colors)]
        diff_id = diff_idx + 1  # 1-based ID for display
        
        # Normalize line width (2-10 range)
        line_width = 2 + (diff_val / max_diff) * 8
        
        # Get positions from both configurations
        pos1_i = diff_data['pos1_i']
        pos1_j = diff_data['pos1_j']
        pos2_i = diff_data['pos2_i']
        pos2_j = diff_data['pos2_j']
        
        # Build comprehensive hover text showing BOTH configurations + ID for matching
        hover_lines = [
            f"<b>🔗 Difference #{diff_id}</b>",
            f"<b>Value: {diff_val}</b>",
            f"Objects: O{obj_i} (t={tst_i:.2f}s) ↔ O{obj_j} (t={tst_j:.2f}s)",
            "",
            f"<b>{config1_id}:</b>",
        ]
        if pos1_i:
            hover_lines.append(f"  O{obj_i}: ({pos1_i[0]:.2f}, {pos1_i[1]:.2f})")
        if pos1_j:
            hover_lines.append(f"  O{obj_j}: ({pos1_j[0]:.2f}, {pos1_j[1]:.2f})")
        
        hover_lines.append("")
        hover_lines.append(f"<b>{config2_id}:</b>")
        if pos2_i:
            hover_lines.append(f"  O{obj_i}: ({pos2_i[0]:.2f}, {pos2_i[1]:.2f})")
        if pos2_j:
            hover_lines.append(f"  O{obj_j}: ({pos2_j[0]:.2f}, {pos2_j[1]:.2f})")
        
        hover_text = "<br>".join(hover_lines)
        
        # Draw on config 1 - include endpoints and midpoint for better hover detection
        if pos1_i and pos1_j:
            mid_x = (pos1_i[0] + pos1_j[0]) / 2
            mid_y = (pos1_i[1] + pos1_j[1]) / 2
            fig.add_trace(go.Scatter(
                x=[pos1_i[0], mid_x, pos1_j[0]],
                y=[pos1_i[1], mid_y, pos1_j[1]],
                mode='lines+markers+text',
                line=dict(color=diff_color, width=line_width),
                marker=dict(size=10, color=diff_color, symbol='circle', 
                           line=dict(color='white', width=1)),
                text=['', str(diff_id), ''],  # Show ID number at midpoint
                textposition='top center',
                textfont=dict(size=10, color='white' if show_court else 'black', family='Arial Black'),
                showlegend=False,
                hoverinfo='text',
                hovertext=[hover_text, hover_text, hover_text],
                name=f'Diff #{diff_id}'
            ), row=1, col=1)
        
        # Draw on config 2 - include endpoints and midpoint for better hover detection
        if pos2_i and pos2_j:
            mid_x = (pos2_i[0] + pos2_j[0]) / 2
            mid_y = (pos2_i[1] + pos2_j[1]) / 2
            fig.add_trace(go.Scatter(
                x=[pos2_i[0], mid_x, pos2_j[0]],
                y=[pos2_i[1], mid_y, pos2_j[1]],
                mode='lines+markers+text',
                line=dict(color=diff_color, width=line_width),
                marker=dict(size=10, color=diff_color, symbol='circle',
                           line=dict(color='white', width=1)),
                text=['', str(diff_id), ''],  # Show ID number at midpoint
                textposition='top center',
                textfont=dict(size=10, color='white' if show_court else 'black', family='Arial Black'),
                showlegend=False,
                hoverinfo='text',
                hovertext=[hover_text, hover_text, hover_text],
                name=f'Diff #{diff_id}'
            ), row=1, col=2)
    
    # Update layout
    x_margin = 2.0
    y_margin = 3.0
    
    fig.update_layout(
        title=f"Fine-Grained Difference Analysis: {config1_id} vs {config2_id}",
        height=700,
        width=1200,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        plot_bgcolor='#25D366' if show_court else 'white'
    )

    # Update axes for both subplots (court extent, or fitted to the data without a court)
    for col in [1, 2]:
        fig.update_xaxes(
            range=[-doubles_alley_width - x_margin, court_width + doubles_alley_width + x_margin] if show_court else None,
            autorange=not show_court,
            showgrid=not show_court,
            zeroline=False,
            title="Court Width (m)" if show_court else "x (m)",
            row=1, col=col
        )
        fig.update_yaxes(
            range=[-y_margin, court_length + y_margin] if show_court else None,
            autorange=not show_court,
            showgrid=not show_court,
            zeroline=False,
            title="Court Length (m)" if show_court else "y (m)",
            scaleanchor="x" if col == 1 else "x2",
            scaleratio=1,
            row=1, col=col
        )
    
    diff_summary['n_windows'] = len(diff_result['differences'])
    diff_summary['position_differences'] = len(position_differences)
    
    return fig, diff_result, diff_summary


def create_difference_matrices_figure(diff_result, config1_id, config2_id, window_indices=None):
    """
    Create a figure showing the inequality matrix differences for selected windows.
    
    Shows side-by-side: Config1 X matrix, Config2 X matrix, Difference matrix (for X and Y).
    
    Args:
        diff_result: Result from compute_pairwise_inequality_differences
        config1_id, config2_id: Configuration IDs for labels
        window_indices: List of window indices to show (None = all)
    
    Returns:
        Plotly figure
    """
    from plotly.subplots import make_subplots
    
    if not diff_result['differences']:
        return None
    
    if window_indices is None:
        window_indices = [0]  # Default to first window
    
    # Filter to requested windows
    windows_to_show = [w for w in diff_result['differences'] if w['window_idx'] in window_indices]
    
    if not windows_to_show:
        return None
    
    n_windows = len(windows_to_show)
    
    # Create subplots: 6 columns (X1, X2, X_diff, Y1, Y2, Y_diff) x n_windows rows
    subplot_titles = []
    for w in windows_to_show:
        win_idx = w['window_idx']
        subplot_titles.extend([
            f"W{win_idx}: {config1_id} X",
            f"W{win_idx}: {config2_id} X",
            f"W{win_idx}: X Diff",
            f"W{win_idx}: {config1_id} Y",
            f"W{win_idx}: {config2_id} Y",
            f"W{win_idx}: Y Diff"
        ])
    
    height_per_row = 420
    fig = make_subplots(
        rows=n_windows, cols=6,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.045,
        vertical_spacing=min(190 / (height_per_row * n_windows + 150), 0.9 / max(1, n_windows - 1))
    )

    rough_x = diff_result.get('rough_x', 0)
    rough_y = diff_result.get('rough_y', 0)
    misaligned = False

    for row_idx, window_data in enumerate(windows_to_show, 1):
        # Both configurations are compared position by position; check that each position
        # refers to the same object, window time and subpoint in both, then use one order
        meta1 = window_data['point_meta1']
        meta2 = window_data['point_meta2']
        identity = lambda m: (str(m['obj']), m['t_rel'], m['sub_type'], m['is_external'])
        if [identity(m) for m in meta1] != [identity(m) for m in meta2]:
            misaligned = True
        display_x = object_major_display(meta1, rough=rough_x)
        display_y = object_major_display(meta1, rough=rough_y)

        # X matrices
        add_point_matrix_heatmap(fig, window_data['ineq_x1'], display_x, row_idx, 1, 'x')
        add_point_matrix_heatmap(fig, window_data['ineq_x2'], display_x, row_idx, 2, 'x')
        add_point_matrix_heatmap(fig, window_data['x_diff_matrix'], display_x, row_idx, 3, 'x', kind='diff',
                                 compared=((str(config1_id), window_data['ineq_x1']),
                                           (str(config2_id), window_data['ineq_x2'])))

        # Y matrices
        add_point_matrix_heatmap(fig, window_data['ineq_y1'], display_y, row_idx, 4, 'y')
        add_point_matrix_heatmap(fig, window_data['ineq_y2'], display_y, row_idx, 5, 'y')
        add_point_matrix_heatmap(fig, window_data['y_diff_matrix'], display_y, row_idx, 6, 'y', kind='diff',
                                 compared=((str(config1_id), window_data['ineq_y1']),
                                           (str(config2_id), window_data['ineq_y2'])))

    # Square cells: each matrix's y axis is scaled to its own x axis
    for row_idx in range(1, n_windows + 1):
        for col_idx in range(1, 7):
            subplot = fig.get_subplot(row_idx, col_idx)
            fig.update_yaxes(scaleanchor=subplot.xaxis.plotly_name.replace('axis', ''), scaleratio=1,
                             constrain='domain', row=row_idx, col=col_idx)
            fig.update_xaxes(constrain='domain', row=row_idx, col=col_idx)

    title = f"Inequality Matrix Comparison: {config1_id} vs {config2_id} – row point compared with column point"
    if misaligned:
        title += "<br><sup>Warning: point identities differ between the two configurations</sup>"
    add_matrix_legend(fig, rough=max(rough_x, rough_y), include_diff=True)
    total_height = height_per_row * n_windows + 190
    fig.update_layout(
        title=dict(text=title, yref='container', y=1, yanchor='top', pad=dict(t=12)),
        height=total_height,
        width=1600,
        showlegend=True,
        legend=dict(orientation='h', yref='container', y=1 - 50 / total_height, yanchor='top',
                    xanchor='center', x=0.5, font=dict(size=11)),
        margin=dict(t=240)
    )
    fig.update_annotations(selector=dict(xref='paper', yref='paper'), yshift=55)

    return fig
