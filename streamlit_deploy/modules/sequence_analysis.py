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


def create_spatial_grid(court_type='Tennis', grid_rows=3, grid_cols=5):
    """
    Create a spatial grid for the court and return zone mapping function.
    Includes buffer zones around the court to capture out-of-bounds positions.
    """
    dims = get_court_dimensions(court_type)
    width, height = dims['width'], dims['height']
    
    buffer = 5.0
    
    x_min, x_max = -buffer, width + buffer
    y_min, y_max = -buffer, height + buffer
    
    actual_cols = grid_cols + 2
    actual_rows = grid_rows + 2
    
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
        """Map (x, y) coordinate to zone label."""
        if pd.isna(x) or pd.isna(y):
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
        'buffer': buffer
    }


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


def smith_waterman(seq1, seq2, match=2, mismatch=-1, gap=-1):
    """Local alignment using Smith-Waterman algorithm."""
    len1, len2 = len(seq1), len(seq2)
    
    score_matrix = np.zeros((len1 + 1, len2 + 1))
    
    max_score = 0
    max_pos = (0, 0)
    
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if seq1[i-1] == seq2[j-1]:
                diagonal = score_matrix[i-1, j-1] + match
            else:
                diagonal = score_matrix[i-1, j-1] + mismatch
            
            score_matrix[i, j] = max(
                0,
                diagonal,
                score_matrix[i-1, j] + gap,
                score_matrix[i, j-1] + gap
            )
            
            if score_matrix[i, j] > max_score:
                max_score = score_matrix[i, j]
                max_pos = (i, j)
    
    aligned1, aligned2 = [], []
    i, j = max_pos
    
    while i > 0 and j > 0 and score_matrix[i, j] > 0:
        current_score = score_matrix[i, j]
        
        diag_score = match if seq1[i-1] == seq2[j-1] else mismatch
        if current_score == score_matrix[i-1, j-1] + diag_score:
            aligned1.append(seq1[i-1])
            aligned2.append(seq2[j-1])
            i -= 1
            j -= 1
        elif current_score == score_matrix[i-1, j] + gap:
            aligned1.append(seq1[i-1])
            aligned2.append('-')
            i -= 1
        elif current_score == score_matrix[i, j-1] + gap:
            aligned1.append('-')
            aligned2.append(seq2[j-1])
            j -= 1
        else:
            break
    
    return {
        'score': max_score,
        'aligned_seq1': list(reversed(aligned1)),
        'aligned_seq2': list(reversed(aligned2)),
        'start1': i,
        'start2': j
    }


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
