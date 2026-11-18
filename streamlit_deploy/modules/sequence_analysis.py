"""
Sequence analysis module for spatiotemporal trajectories.

This module converts continuous trajectories into symbolic sequences and provides
tools for sequence comparison, alignment, and pattern mining.
"""

import numpy as np
import pandas as pd
from collections import Counter
from itertools import groupby


# =============================================================================
# SPATIAL GRID CREATION
# =============================================================================

def get_court_dimensions(court_type='Football'):
    """Return court dimensions based on type."""
    if court_type == 'Tennis':
        return {
            'width': 8.23,
            'height': 23.77,
            'aspect_width': 400,
            'aspect_height': 1100
        }
    else:
        return {
            'width': 110,
            'height': 72,
            'aspect_width': 900,
            'aspect_height': 600
        }


SPORTS_COURTS = ('Tennis', 'Football')


def data_bounds(df):
    """(x_min, x_max, y_min, y_max) of the given positions: the data extent used for zone grids."""
    return (float(df['x'].min()), float(df['x'].max()), float(df['y'].min()), float(df['y'].max()))


def create_spatial_grid(court_type='Tennis', grid_rows=3, grid_cols=5, bounds=None, geometry=None):
    """
    Create a spatial grid and return its zone mapping function.

    - bounds=None and court_type 'Tennis' or 'Football': the court or pitch geometry, with a 5 m buffer ring of
      zones around it; positions beyond the buffer are assigned to the nearest edge zone.
    - bounds=(x_min, x_max, y_min, y_max): grid_rows x grid_cols zones covering exactly these bounds, no buffer
      and no court geometry. Positions outside the bounds get no zone (None) and are left out.
    geometry: label for the bounds ('data extent' or 'explicit bounds'), shown in captions and exports.
    """
    if bounds is not None:
        x_min, x_max, y_min, y_max = map(float, bounds)
        if not (x_max > x_min and y_max > y_min):
            raise ValueError(f"Grid bounds need x_max > x_min and y_max > y_min (got {bounds}).")
        return _make_grid(x_min, x_max, y_min, y_max, grid_rows, grid_cols, buffer=0.0,
                          width=x_max - x_min, height=y_max - y_min, clamp=False,
                          geometry=geometry or 'explicit bounds')
    if court_type not in SPORTS_COURTS:
        raise ValueError("Without a selected court or pitch the grid needs bounds (data extent or explicit).")
    dims = get_court_dimensions(court_type)
    width, height = dims['width'], dims['height']
    buffer = 5.0
    return _make_grid(-buffer, width + buffer, -buffer, height + buffer, grid_rows + 2, grid_cols + 2,
                      buffer=buffer, width=width, height=height, clamp=True, geometry=court_type)


def _make_grid(x_min, x_max, y_min, y_max, actual_rows, actual_cols, buffer, width, height, clamp, geometry):
    """Equal-width zones between the bounds; labels A1, B1, ... from the lowest row, left to right."""
    x_bins = np.linspace(x_min, x_max, actual_cols + 1)
    y_bins = np.linspace(y_min, y_max, actual_rows + 1)
    
    def get_column_label(col_idx):
        """Generate column label (A, B, C, ..., Z, AA, AB, ...)"""
        if col_idx < 26:
            return chr(65 + col_idx)
        else:
            return chr(65 + (col_idx // 26) - 1) + chr(65 + (col_idx % 26))
    
    zone_labels = []
    for row in range(actual_rows):
        for col in range(actual_cols):
            col_label = get_column_label(col)
            row_label = str(row + 1)
            zone_labels.append(f"{col_label}{row_label}")
    
    def get_zone(x, y):
        """Map (x, y) coordinate to zone label (zones are [left, right), the last one also includes its right edge)."""
        if pd.isna(x) or pd.isna(y):
            return None
        if not clamp and not (x_min <= x <= x_max and y_min <= y <= y_max):
            return None
        col_idx = np.digitize(x, x_bins) - 1
        row_idx = np.digitize(y, y_bins) - 1
        col_idx = max(0, min(actual_cols - 1, col_idx))
        row_idx = max(0, min(actual_rows - 1, row_idx))
        return zone_labels[row_idx * actual_cols + col_idx]

    return {
        'zone_labels': zone_labels,
        'x_bins': x_bins,
        'y_bins': y_bins,
        'get_zone': get_zone,
        'grid_rows': actual_rows,
        'grid_cols': actual_cols,
        'court_width': width,
        'court_height': height,
        'buffer': buffer,
        'geometry': geometry,
        'bounds': (float(x_min), float(x_max), float(y_min), float(y_max)),
    }


def grid_description(grid_info):
    """One line: geometry, zone count and the exact bounds of the grid."""
    x0, x1, y0, y1 = grid_info['bounds']
    n = grid_info['grid_rows'] * grid_info['grid_cols']
    if grid_info['buffer'] > 0:
        kind = (f"{grid_info['geometry']} geometry with a {grid_info['buffer']:g} m buffer ring "
                f"({grid_info['grid_rows'] - 2} × {grid_info['grid_cols'] - 2} court zones plus buffer)")
    else:
        kind = f"{grid_info['geometry']} ({grid_info['grid_rows']} × {grid_info['grid_cols']} zones, no court geometry)"
    return f"Zone grid: {kind}, {n} zones, x {x0:.6g} to {x1:.6g}, y {y0:.6g} to {y1:.6g}"


def outside_grid_count(df, grid_info):
    """Number of positions that receive no zone (outside explicit bounds, or missing coordinates)."""
    get_zone = grid_info['get_zone']
    return int(sum(get_zone(x, y) is None for x, y in zip(df['x'], df['y'])))


def draw_zone_grid(fig, grid_info, antecedents=(), consequents=()):
    """
    Grid lines and zone labels on fig. Antecedent zones are filled red, consequent zones blue.
    A buffer background is drawn only for a court or pitch geometry. Axes are fitted to the grid.
    """
    x_bins, y_bins = grid_info['x_bins'], grid_info['y_bins']
    buffer = grid_info['buffer']
    if buffer > 0:
        fig.add_shape(type="rect", x0=-buffer, y0=-buffer, x1=grid_info['court_width'] + buffer,
                      y1=grid_info['court_height'] + buffer, fillcolor='rgba(200, 200, 200, 0.3)',
                      line=dict(color='rgba(150, 150, 150, 0.5)', width=2), layer="below")
    for x in x_bins:
        fig.add_shape(type="line", x0=x, y0=y_bins[0], x1=x, y1=y_bins[-1],
                      line=dict(color='rgba(255, 0, 0, 0.6)', width=2, dash='dash'), layer="above")
    for y in y_bins:
        fig.add_shape(type="line", x0=x_bins[0], y0=y, x1=x_bins[-1], y1=y,
                      line=dict(color='rgba(255, 0, 0, 0.6)', width=2, dash='dash'), layer="above")
    cols = grid_info['grid_cols']
    for row in range(grid_info['grid_rows']):
        for col in range(cols):
            label = grid_info['zone_labels'][row * cols + col]
            ante, cons = label in antecedents, label in consequents
            if ante or cons:
                fig.add_shape(type="rect", x0=x_bins[col], y0=y_bins[row], x1=x_bins[col + 1], y1=y_bins[row + 1],
                              fillcolor='rgba(255, 0, 0, 0.3)' if ante else 'rgba(0, 0, 255, 0.3)',
                              line=dict(color='red' if ante else 'blue', width=3), layer="above")
            fig.add_annotation(x=(x_bins[col] + x_bins[col + 1]) / 2, y=(y_bins[row] + y_bins[row + 1]) / 2,
                               text=f"<b>{label}</b>", showarrow=False,
                               font=dict(size=14, color='white'),
                               bgcolor=('rgba(200, 0, 0, 0.9)' if ante else 'rgba(0, 0, 200, 0.9)' if cons
                                        else 'rgba(0, 0, 0, 0.6)'), borderpad=4)
    pad_x = max(0.5, 0.02 * (x_bins[-1] - x_bins[0]))
    pad_y = max(0.5, 0.02 * (y_bins[-1] - y_bins[0]))
    fig.update_xaxes(range=[x_bins[0] - pad_x, x_bins[-1] + pad_x], autorange=False)
    fig.update_yaxes(range=[y_bins[0] - pad_y, y_bins[-1] + pad_y], autorange=False)
    return fig


def zone_grid_controls(st, court_type, positions, key_prefix, grid_rows, grid_cols):
    """
    Zone geometry for Spatial Zones, Combined and zone-based sequences. The court or pitch geometry is used only
    when that court type is selected in the sidebar; otherwise the grid covers the data extent of `positions`
    (the selected trajectories in the time window) or explicit bounds entered here.
    Returns (grid_info, settings) where settings is a hashable tuple for stale-result checks and exports.
    """
    if court_type in SPORTS_COURTS:
        st.caption(f"Zone geometry: {court_type} (court type selected in the sidebar).")
        grid = create_spatial_grid(court_type, grid_rows, grid_cols)
        return grid, (court_type, grid_rows, grid_cols)
    mode = st.radio("Grid bounds", ["Data extent", "Explicit bounds"], horizontal=True, key=f"{key_prefix}_bounds_mode",
                    help="No court or pitch is selected, so zones cover the coordinates themselves. Data extent: "
                         "the minimum and maximum x and y of the selected positions. Explicit bounds: enter them.")
    ext = data_bounds(positions)
    if mode == "Data extent":
        bounds, geometry = ext, 'data extent'
    else:
        b = st.columns(4)
        names = ('x min', 'x max', 'y min', 'y max')
        for i in range(4):                       # first use: start from the data extent; afterwards keep the entry
            if not isinstance(st.session_state.get(f"{key_prefix}_bound_{i}"), (int, float)):
                st.session_state[f"{key_prefix}_bound_{i}"] = float(ext[i])
        bounds = tuple(float(b[i].number_input(names[i], format="%.3f", key=f"{key_prefix}_bound_{i}"))
                       for i in range(4))
        geometry = 'explicit bounds'
    try:
        grid = create_spatial_grid(None, grid_rows, grid_cols, bounds=bounds, geometry=geometry)
    except ValueError as e:
        st.error(str(e))
        return None, None
    return grid, (geometry, grid_rows, grid_cols) + tuple(round(v, 9) for v in bounds)


# =============================================================================
# TIME WINDOW (shared by Association Rules, Sequence Analysis and Outlier Detection)
# =============================================================================

def time_window_inputs(st, df, start_key, end_key):
    """
    Start/End time inputs over the data's tst range (defaults: full range). Both ends are included.
    Returns (start, end) or None when Start is after End (an error is shown).
    """
    lo, hi = float(df['tst'].min()), float(df['tst'].max())
    s = st.session_state
    for key, default in ((start_key, lo), (end_key, hi)):
        v = s.get(key)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not lo <= v <= hi:
            s[key] = default
    c1, c2 = st.columns(2)
    start = c1.number_input("Start time", min_value=lo, max_value=hi, step=0.01, format="%.2f", key=start_key)
    end = c2.number_input("End time", min_value=lo, max_value=hi, step=0.01, format="%.2f", key=end_key)
    if start > end:
        st.error(f"Start time ({start:g}) is after End time ({end:g}). Correct the time range before computing.")
        return None
    return float(start), float(end)


def select_window(df, configs, objects, start, end):
    """Positions of the selected configurations and objects with start <= tst <= end."""
    return df[df['config_source'].isin(configs) & df['obj'].isin(objects)
              & (df['tst'] >= start) & (df['tst'] <= end)]


def time_window_text(start, end, positions):
    """Same wording on every page: window, trajectories (configuration-object pairs) and sampled positions."""
    n_traj = positions.groupby(['config_source', 'obj']).ngroups if len(positions) else 0
    return (f"Time window: tst {start:g} to {end:g} (both ends included) · {n_traj} trajectories · "
            f"{len(positions)} sampled positions")


# =============================================================================
# SEQUENCE BUILDING
# =============================================================================

def build_event_based_sequence(df, config, obj_id, start_time, end_time, grid_info, compress=True):
    """Build event-based sequence (one token per hit/bounce)."""
    obj_data = df[(df['config_source'] == config) &
                  (df['obj'] == obj_id) &
                  (df['tst'] >= start_time) &
                  (df['tst'] <= end_time)].sort_values('tst')
    
    if len(obj_data) == 0:
        return ""
    
    get_zone = grid_info['get_zone']
    tokens = [get_zone(row['x'], row['y']) for _, row in obj_data.iterrows()]
    tokens = [t for t in tokens if t is not None]
    
    if compress:
        tokens = [k for k, _ in groupby(tokens)]
    
    return tokens


def build_interval_based_sequence(df, config, obj_id, start_time, end_time, 
                                  grid_info, delta_t=0.2, compress=True):
    """Build equal-interval sequence (fixed time steps)."""
    obj_data = df[(df['config_source'] == config) &
                  (df['obj'] == obj_id) &
                  (df['tst'] >= start_time) &
                  (df['tst'] <= end_time)].sort_values('tst')
    
    if len(obj_data) == 0:
        return ""
    
    get_zone = grid_info['get_zone']
    
    time_points = np.arange(start_time, end_time + delta_t, delta_t)
    tokens = []
    
    for t in time_points:
        closest_idx = (obj_data['tst'] - t).abs().idxmin()
        row = obj_data.loc[closest_idx]
        zone = get_zone(row['x'], row['y'])
        if zone is not None:
            tokens.append(zone)
    
    if compress:
        tokens = [k for k, _ in groupby(tokens)]
    
    return tokens


def build_multi_entity_sequence(df, config, entity_ids, start_time, end_time,
                                grid_info, mode='event', delta_t=0.2, compress=True):
    """Build joint sequence combining multiple entities."""
    sequences = {}
    for entity_id in entity_ids:
        if mode == 'event':
            seq = build_event_based_sequence(df, config, entity_id, start_time, 
                                            end_time, grid_info, compress=False)
        else:
            seq = build_interval_based_sequence(df, config, entity_id, start_time, 
                                               end_time, grid_info, delta_t, compress=False)
        sequences[entity_id] = seq
    
    max_len = max(len(s) for s in sequences.values()) if sequences else 0
    
    for eid in sequences:
        while len(sequences[eid]) < max_len:
            sequences[eid].append(sequences[eid][-1] if sequences[eid] else 'X')
    
    joint_tokens = []
    for i in range(max_len):
        token_parts = [f"{eid}:{sequences[eid][i]}" for eid in entity_ids]
        joint_tokens.append('|'.join(token_parts))
    
    if compress:
        joint_tokens = [k for k, _ in groupby(joint_tokens)]
    
    return joint_tokens


# =============================================================================
# SEQUENCE COMPARISON AND ALIGNMENT
# =============================================================================

def levenshtein_distance(seq1, seq2):
    """Compute Levenshtein (edit) distance between two sequences."""
    len1, len2 = len(seq1), len(seq2)
    
    dp = np.zeros((len1 + 1, len2 + 1), dtype=int)
    
    for i in range(len1 + 1):
        dp[i, 0] = i
    for j in range(len2 + 1):
        dp[0, j] = j
    
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if seq1[i-1] == seq2[j-1]:
                cost = 0
            else:
                cost = 1
            
            dp[i, j] = min(
                dp[i-1, j] + 1,
                dp[i, j-1] + 1,
                dp[i-1, j-1] + cost
            )
    
    return int(dp[len1, len2])


def needleman_wunsch(seq1, seq2, match=2, mismatch=-1, gap=-1):
    """Global alignment using Needleman-Wunsch algorithm."""
    len1, len2 = len(seq1), len(seq2)
    
    score_matrix = np.zeros((len1 + 1, len2 + 1))
    
    for i in range(len1 + 1):
        score_matrix[i, 0] = gap * i
    for j in range(len2 + 1):
        score_matrix[0, j] = gap * j
    
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if seq1[i-1] == seq2[j-1]:
                diagonal = score_matrix[i-1, j-1] + match
            else:
                diagonal = score_matrix[i-1, j-1] + mismatch
            
            score_matrix[i, j] = max(
                diagonal,
                score_matrix[i-1, j] + gap,
                score_matrix[i, j-1] + gap
            )
    
    aligned1, aligned2 = [], []
    i, j = len1, len2
    
    while i > 0 or j > 0:
        current_score = score_matrix[i, j]
        
        if i > 0 and j > 0:
            diag_score = match if seq1[i-1] == seq2[j-1] else mismatch
            if current_score == score_matrix[i-1, j-1] + diag_score:
                aligned1.append(seq1[i-1])
                aligned2.append(seq2[j-1])
                i -= 1
                j -= 1
                continue
        
        if i > 0 and current_score == score_matrix[i-1, j] + gap:
            aligned1.append(seq1[i-1])
            aligned2.append('-')
            i -= 1
        elif j > 0 and current_score == score_matrix[i, j-1] + gap:
            aligned1.append('-')
            aligned2.append(seq2[j-1])
            j -= 1
        else:
            break
    
    return {
        'score': score_matrix[len1, len2],
        'aligned_seq1': list(reversed(aligned1)),
        'aligned_seq2': list(reversed(aligned2))
    }


def longest_common_subsequence(seq1, seq2):
    """
    Find the Longest Common Substring (continuous) between two sequences.
    
    The Longest Common Substring is the longest continuous sequence of symbols 
    that appears in both sequences without any gaps or interruptions.
    
    Example:
        S1: A-B-A-C-B-T-E
        S2: D-B-A-C-T-E
        Result: B-A-C (continuous in both, length = 3)
        
    Note: This is different from longest common subsequence which allows gaps.
    """
    len1, len2 = len(seq1), len(seq2)
    
    # Create DP table
    # dp[i][j] = length of common substring ending at seq1[i-1] and seq2[j-1]
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    
    max_length = 0
    end_pos_seq1 = 0
    end_pos_seq2 = 0
    
    # Fill the table
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    end_pos_seq1 = i
                    end_pos_seq2 = j
            else:
                dp[i][j] = 0  # Reset to 0 for substring (must be continuous)
    
    # Extract the longest common substring
    start1 = end_pos_seq1 - max_length
    start2 = end_pos_seq2 - max_length
    lcs = seq1[start1:end_pos_seq1] if max_length > 0 else []
    
    # Create aligned sequences with gaps showing the context
    aligned1 = []
    aligned2 = []
    
    if max_length > 0:
        # Add symbols before the match with gaps to align
        prefix_len = max(start1, start2)
        for i in range(prefix_len):
            if i < start1:
                aligned1.append('-')
            else:
                aligned1.append(seq1[i])
            
            if i < start2:
                aligned2.append('-')
            else:
                aligned2.append(seq2[i])
        
        # Add the matching substring
        for i in range(max_length):
            aligned1.append(seq1[start1 + i])
            aligned2.append(seq2[start2 + i])
        
        # Add symbols after the match with gaps to align
        remaining1 = len1 - end_pos_seq1
        remaining2 = len2 - end_pos_seq2
        suffix_len = max(remaining1, remaining2)
        for i in range(suffix_len):
            if i < remaining1:
                aligned1.append(seq1[end_pos_seq1 + i])
            else:
                aligned1.append('-')
            
            if i < remaining2:
                aligned2.append(seq2[end_pos_seq2 + i])
            else:
                aligned2.append('-')
    
    return {
        'score': max_length,
        'lcs': lcs,
        'lcs_length': max_length,
        'aligned_seq1': aligned1,
        'aligned_seq2': aligned2,
        'start1': start1,
        'start2': start2,
        'positions_seq1': list(range(start1, end_pos_seq1)) if max_length > 0 else [],
        'positions_seq2': list(range(start2, end_pos_seq2)) if max_length > 0 else []
    }


# Keep the old function name for backwards compatibility
def smith_waterman(seq1, seq2, match=2, mismatch=-1, gap=-1):
    """
    Wrapper that calls longest_common_subsequence for backwards compatibility.
    Note: Parameters match, mismatch, gap are ignored as LCS only finds exact matches.
    """
    return longest_common_subsequence(seq1, seq2)


# =============================================================================
# PATTERN MINING
# =============================================================================

def extract_ngrams(sequence, n=2):
    """Extract n-grams from sequence."""
    if len(sequence) < n:
        return Counter()
    
    ngrams = [tuple(sequence[i:i+n]) for i in range(len(sequence) - n + 1)]
    return Counter(ngrams)


def compute_sequence_distance_matrix(sequences, method='levenshtein'):
    """Compute pairwise distance matrix for sequences."""
    n = len(sequences)
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = levenshtein_distance(sequences[i], sequences[j])
            
            if method == 'normalized_levenshtein':
                max_len = max(len(sequences[i]), len(sequences[j]))
                if max_len > 0:
                    dist = dist / max_len
            
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    
    return dist_matrix


# =============================================================================
# MOVEMENT-STEP (L/H) SEQUENCES
# =============================================================================

STEP_THRESHOLD_M = 2.0


def step_table(df, configs, objects, start_time, end_time):
    """
    One row per step between consecutive positions of each (configuration, object), sorted by tst (stable).
    Distance = Euclidean displacement in the x/y units of the data. n positions give n - 1 steps.
    """
    rows = []
    sel = df[df['config_source'].isin(configs) & df['obj'].isin(objects)
             & (df['tst'] >= start_time) & (df['tst'] <= end_time)]
    order = {c: i for i, c in enumerate(configs)}
    for (config, obj), g in sorted(sel.groupby(['config_source', 'obj']),
                                   key=lambda kv: (order.get(kv[0][0], 0), kv[0][1])):
        g = g.sort_values('tst', kind='mergesort')
        t, x, y = g['tst'].to_numpy(float), g['x'].to_numpy(float), g['y'].to_numpy(float)
        d = np.hypot(np.diff(x), np.diff(y))
        for i in range(len(d)):
            rows.append({'Configuration': config, 'Object': obj, 'Step': i + 1,
                         'tst_from': t[i], 'tst_to': t[i + 1], 'Distance_m': float(d[i])})
    return pd.DataFrame(rows, columns=['Configuration', 'Object', 'Step', 'tst_from', 'tst_to', 'Distance_m'])


def step_token(distance, threshold=STEP_THRESHOLD_M):
    """'L' when the step is shorter than the threshold, 'H' when it is at least the threshold."""
    return 'L' if distance < threshold else 'H'


def run_length_compress(tokens):
    return [k for k, _ in groupby(tokens)]


def step_sequences(steps, threshold=STEP_THRESHOLD_M, compress=False):
    """Per trajectory: positions, step distances, tokens (optionally run-length compressed)."""
    steps = steps.copy()
    steps['Token'] = [step_token(d, threshold) for d in steps['Distance_m']]
    rows = []
    for (config, obj), g in steps.groupby(['Configuration', 'Object'], sort=False):
        tokens = list(g['Token'])
        seq = run_length_compress(tokens) if compress else tokens
        rows.append({'Configuration': config, 'Object': obj, 'Trajectory': f"{config}_obj{obj}",
                     'Positions': len(g) + 1, 'Steps': len(g), 'Threshold_m': float(threshold),
                     'Compressed': bool(compress),
                     'Distances_m': ';'.join(repr(float(v)) for v in g['Distance_m']),
                     'Sequence': ''.join(seq)})
    return steps, pd.DataFrame(rows)


# ---------------------------------------------------------------------------- custom step encoding
CUSTOM_FEATURES = {
    'Step distance (m)': 'Distance_m',
    'Step speed (m per tst unit)': 'Speed_m_per_tst',
}


def parse_cutoffs(text):
    """'1, 3.5' -> [1.0, 3.5]. Raises ValueError unless the cut-offs are numbers in strictly increasing order."""
    parts = [p for p in str(text).replace(';', ',').replace(' ', ',').split(',') if p.strip()]
    if not parts:
        raise ValueError("Enter at least one cut-off.")
    try:
        cuts = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"Cut-offs must be numbers (got '{text}').")
    if any(b <= a for a, b in zip(cuts, cuts[1:])):
        raise ValueError("Cut-offs must be in strictly increasing order.")
    return cuts


def parse_labels(text, n_needed):
    """Token labels: single letters or digits, all different, exactly one more than the number of cut-offs."""
    labels = [c for c in str(text).upper() if c.isalnum()]
    if len(labels) != n_needed:
        raise ValueError(f"Enter {n_needed} token labels (one more than the number of cut-offs); got {len(labels)}.")
    if len(set(labels)) != len(labels):
        raise ValueError("Token labels must all be different.")
    return labels


def custom_token(value, cuts, labels):
    """labels[i] for cuts[i-1] <= value < cuts[i]: a value equal to a cut-off gets the higher token (as 2 m is H)."""
    if value != value:                          # NaN (e.g. speed with zero elapsed tst)
        return None
    return labels[int(np.searchsorted(cuts, value, side='right'))]


def custom_step_sequences(steps, feature, cuts, labels, compress=False):
    """Same layout as step_sequences, with tokens from the chosen feature, cut-offs and labels."""
    steps = steps.copy()
    dt = steps['tst_to'] - steps['tst_from']
    steps['Speed_m_per_tst'] = np.where(dt > 0, steps['Distance_m'] / dt.where(dt > 0), np.nan)
    col = CUSTOM_FEATURES.get(feature, feature)
    steps['Token'] = [custom_token(v, cuts, labels) for v in steps[col]]
    rows = []
    for (config, obj), g in steps.groupby(['Configuration', 'Object'], sort=False):
        tokens = [t for t in g['Token'] if t is not None]
        seq = run_length_compress(tokens) if compress else tokens
        rows.append({'Configuration': config, 'Object': obj, 'Trajectory': f"{config}_obj{obj}",
                     'Positions': len(g) + 1, 'Steps': len(g), 'Feature': feature,
                     'Cutoffs': ';'.join(repr(c) for c in cuts), 'Labels': ''.join(labels),
                     'Compressed': bool(compress), 'Sequence': ''.join(seq)})
    return steps, pd.DataFrame(rows)


def parse_pattern(text):
    """'L H', 'L,H', 'L→H' or 'LH' -> ['L', 'H'] (tokens are single characters)."""
    return [c for c in str(text).upper() if c.isalnum()]


def contains_ordered(sequence, pattern):
    """True when pattern occurs in order in sequence, gaps allowed (a subsequence, not a substring)."""
    it = iter(sequence)
    return all(any(s == p for s in it) for p in pattern)


def contains_all(sequence, pattern):
    """True when every token of pattern occurs somewhere in sequence (order ignored)."""
    return set(pattern) <= set(sequence)


def pattern_support(sequences, pattern, ordered=True):
    """
    sequences: {trajectory: sequence string}. Each trajectory counts at most once.
    Returns (supporting trajectories, count, denominator, relative frequency).
    """
    test = contains_ordered if ordered else contains_all
    supporting = [tid for tid, seq in sequences.items() if pattern and test(seq, pattern)]
    n = len(sequences)
    return supporting, len(supporting), n, (len(supporting) / n if n else float('nan'))
