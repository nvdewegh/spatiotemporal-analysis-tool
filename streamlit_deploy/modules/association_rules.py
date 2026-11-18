"""
Association Rules Analysis Module

This module implements association rule learning (market basket analysis) 
for spatiotemporal trajectory data.

Key Features:
- Transaction preparation (spatial zones, feature bins, combined)
- Apriori algorithm for frequent itemset mining
- Association rule generation with multiple metrics
- Interactive visualisations (network, scatter, heatmap, MDS)
"""

import hashlib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import networkx as nx
from sklearn.manifold import MDS
from itertools import combinations
from .common import render_interactive_chart, format_number
from . import sequence_analysis


# ============================================================================
# TRANSACTION PREPARATION FUNCTIONS
# ============================================================================

def prepare_spatial_transactions(trajectories_dict, get_zone_func):
    """
    Convert trajectories into transactions based on spatial zones visited.
    
    Args:
        trajectories_dict: Dictionary of {traj_id: DataFrame with x, y columns}
        get_zone_func: Function that maps (x, y) -> zone_label
        
    Returns:
        transactions: List of lists (each inner list is a transaction)
        item_names: Sorted list of unique items
        trajectory_mapping: List mapping transaction index to trajectory ID
    """
    transactions = []
    trajectory_mapping = []
    
    for traj_id, traj_data in trajectories_dict.items():
        x_coords = traj_data['x'].values
        y_coords = traj_data['y'].values
        
        zones = []
        for x, y in zip(x_coords, y_coords):
            zone = get_zone_func(x, y)
            if zone:
                zones.append(zone)
        
        if zones:
            transaction = list(set(zones))  # Unique zones visited
            transactions.append(transaction)
            trajectory_mapping.append(traj_id)
    
    all_items = set()
    for transaction in transactions:
        all_items.update(transaction)
    item_names = sorted(list(all_items))
    
    return transactions, item_names, trajectory_mapping


# ---------------------------------------------------------------------------- features for Feature Bins
# Names, formulas and units shared with Clustering (clustering.extract_trajectory_features); x_range and y_range
# are kept from the original Association Rules feature set.
AR_FEATURE_DEFINITIONS = {
    'total_distance': ('Total Distance', 'm', 'Sum of the Euclidean distances between consecutive positions'),
    'duration': ('Duration', 'tst units', 'Last tst minus first tst'),
    'avg_speed': ('Average Speed', 'm per tst unit', 'Total distance / duration'),
    'net_displacement': ('Net Displacement', 'm', 'Straight-line distance from first to last position'),
    'sinuosity': ('Sinuosity (Path Efficiency)', 'ratio', 'Total distance / net displacement (1 when net '
                                                           'displacement is 0)'),
    'bbox_area': ('Bounding Box Area', 'm²', 'x range × y range'),
    'avg_direction': ('Average Direction', 'rad', 'Mean of the step headings atan2(dy, dx)'),
    'max_speed': ('Maximum Speed', 'm per tst unit', 'Highest segment speed (distance / elapsed tst)'),
    'min_speed': ('Minimum Speed', 'm per tst unit', 'Lowest segment speed (distance / elapsed tst)'),
    'x_range': ('x Range', 'm', 'Maximum x minus minimum x'),
    'y_range': ('y Range', 'm', 'Maximum y minus minimum y'),
}
AR_FEATURES = list(AR_FEATURE_DEFINITIONS)
AR_DEFAULT_FEATURES = ['total_distance', 'duration', 'avg_speed', 'x_range', 'y_range']


def feature_label(name):
    label, unit, _ = AR_FEATURE_DEFINITIONS.get(name, (name, '', ''))
    return f"{label} ({unit})" if unit else label


def feature_definitions_table(features=AR_FEATURES):
    return pd.DataFrame([{'Feature': f, 'Name': AR_FEATURE_DEFINITIONS[f][0], 'Unit': AR_FEATURE_DEFINITIONS[f][1],
                          'Formula': AR_FEATURE_DEFINITIONS[f][2]} for f in features])


def association_features(traj_df):
    """Clustering features of one trajectory (same function, same values) plus x_range and y_range."""
    from .clustering import extract_trajectory_features
    feats = extract_trajectory_features(traj_df)
    feats['x_range'] = float(traj_df['x'].max() - traj_df['x'].min())
    feats['y_range'] = float(traj_df['y'].max() - traj_df['y'].min())
    return feats


def bin_labels_for(n_bins):
    return ['low', 'medium', 'high'] if n_bins == 3 else [f'bin{i+1}' for i in range(n_bins)]


def feature_bin_edges(values, n_bins):
    """n_bins equal-width bins between minimum and maximum: n_bins + 1 edges, or None for a constant feature."""
    v = pd.Series(values).dropna()
    if v.empty or not v.max() > v.min():
        return None
    lo, hi = float(v.min()), float(v.max())
    width = (hi - lo) / n_bins
    return [lo + i * width for i in range(n_bins)] + [hi]


def constant_features(features_df):
    """Numeric feature columns with a single value (or none): they cannot be binned and produce no items."""
    cols = [c for c in features_df.select_dtypes(include=[np.number]).columns if c not in ('config', 'object', 'rally_id')]
    return [c for c in cols if feature_bin_edges(features_df[c], 2) is None]


def feature_items(features_df, n_bins=3):
    """
    Items per row (aligned with features_df rows, possibly empty) and the bin table.
    A value v is in bin i when edge_i <= v < edge_(i+1); the last bin also contains the maximum.
    """
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in ('config', 'object', 'rally_id')]
    labels = bin_labels_for(n_bins)
    bin_info, edges = {}, {}
    for col in feature_cols:
        e = feature_bin_edges(features_df[col], n_bins)
        if e is not None:
            edges[col] = e
            bin_info[col] = {labels[i]: (e[i], e[i + 1]) for i in range(n_bins)}
    rows = []
    for _, row in features_df.iterrows():
        items = []
        for col, e in edges.items():
            value = row[col]
            if pd.notna(value):
                idx = min(int(np.searchsorted(e[1:-1], value, side='right')), n_bins - 1)
                items.append(f"{col}_{labels[idx]}")
        rows.append(items)
    return rows, bin_info


def prepare_feature_transactions(features_df, n_bins=3):
    """
    Convert feature data into transactions based on binned feature values.
    Returns transactions (rows without any item are left out), sorted item names and the bin ranges per feature.
    """
    rows, bin_info = feature_items(features_df, n_bins)
    transactions = [r for r in rows if r]
    item_names = sorted({i for t in transactions for i in t})
    return transactions, item_names, bin_info


def bin_table(bin_info, n_bins):
    """One row per feature bin with the exact boundaries and the interval convention."""
    rows = []
    for feature, bins in bin_info.items():
        for i, (label, (lo, hi)) in enumerate(bins.items()):
            last = i == len(bins) - 1
            rows.append({'Feature': feature, 'Unit': AR_FEATURE_DEFINITIONS.get(feature, ('', '', ''))[1],
                         'Bin': label, 'Item': f"{feature}_{label}", 'Lower': lo, 'Upper': hi,
                         'Interval': f"[{lo!r}, {hi!r}" + (']' if last else ')')})
    return pd.DataFrame(rows, columns=['Feature', 'Unit', 'Bin', 'Item', 'Lower', 'Upper', 'Interval'])


def combine_transactions(spatial_by_id, feature_by_id):
    """Combined transactions matched by trajectory ID (never by list position). Zone items get a 'zone_' prefix."""
    ids = [t for t in spatial_by_id if t in feature_by_id]
    transactions, mapping = [], []
    for t in ids:
        items = [f"zone_{z}" for z in spatial_by_id[t]] + list(feature_by_id[t])
        if items:
            transactions.append(items)
            mapping.append(t)
    return transactions, sorted({i for tr in transactions for i in tr}), mapping


def prepare_combined_transactions(spatial_trans, spatial_items, feature_trans, feature_items_list):
    """Combine spatial and feature transactions given in the same trajectory order (kept for compatibility)."""
    if len(spatial_trans) != len(feature_trans):
        raise ValueError("Spatial and feature transactions must describe the same trajectories in the same order.")
    tx, items, _ = combine_transactions(dict(enumerate(spatial_trans)), dict(enumerate(feature_trans)))
    return tx, items


# ============================================================================
# ASSOCIATION RULE MINING
# ============================================================================

def compute_association_rules(transactions, min_support=0.1, min_confidence=0.5):
    """
    Compute association rules using Apriori algorithm.
    
    Args:
        transactions: List of transaction lists
        min_support: Minimum support threshold
        min_confidence: Minimum confidence threshold
        
    Returns:
        rules_df: DataFrame with rules and metrics
        frequent_itemsets: DataFrame with frequent itemsets
    """
    if not transactions or len(transactions) == 0:
        return pd.DataFrame(), pd.DataFrame()
    
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    
    if len(frequent_itemsets) == 0:
        return pd.DataFrame(), pd.DataFrame()
    
    try:
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(sorted(list(x))))
        rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(sorted(list(x))))
        # Sort by confidence (primary metric) by default
        rules = rules.sort_values('confidence', ascending=False).reset_index(drop=True)
        return rules, frequent_itemsets
    except ValueError:
        return pd.DataFrame(), frequent_itemsets

def add_rule_counts(rules, transactions):
    """
    Whole-number counts behind support and confidence, counted directly in the transactions:
    N (all transactions), antecedent count, consequent count and joint count (antecedent and consequent).
    support = joint / N, confidence = joint / antecedent, consequent prevalence = consequent / N.
    """
    rules = rules.copy()
    sets = [set(t) for t in transactions]
    n = len(sets)

    def count(items):
        return sum(1 for s in sets if items <= s)

    rules['n_transactions'] = n
    rules['n_antecedent'] = [count(set(a)) for a in rules['antecedents']]
    rules['n_consequent'] = [count(set(c)) for c in rules['consequents']]
    rules['n_joint'] = [count(set(a) | set(c)) for a, c in zip(rules['antecedents'], rules['consequents'])]
    rules['consequent_prevalence'] = rules['n_consequent'] / n if n else np.nan
    rules['rule'] = rules['antecedents_str'] + ' → ' + rules['consequents_str']
    return rules


RULE_EXPORT_COLUMNS = ['rule', 'antecedents_str', 'consequents_str', 'n_transactions', 'n_antecedent',
                       'n_consequent', 'n_joint', 'support', 'confidence', 'consequent_prevalence', 'lift']


def rules_export(rules):
    """Rules with counts and full-precision metrics as CSV text."""
    cols = [c for c in RULE_EXPORT_COLUMNS if c in rules.columns]
    out = rules[cols].rename(columns={'antecedents_str': 'antecedent', 'consequents_str': 'consequent'})
    return out.to_csv(index=False, float_format='%.17g')


def filter_rules(rules, text='', antecedent_items=(), consequent_items=()):
    """Rules whose text contains `text` (case-insensitive) and whose sides contain all chosen items."""
    keep = pd.Series(True, index=rules.index)
    if text.strip():
        t = text.strip().lower()
        keep &= rules['rule'].str.lower().str.contains(t, regex=False)
    for item in antecedent_items:
        keep &= rules['antecedents'].apply(lambda s, i=item: i in s)
    for item in consequent_items:
        keep &= rules['consequents'].apply(lambda s, i=item: i in s)
    return rules[keep]


def split_feature_item(item, features):
    """'total_distance_high' -> ('total_distance', 'high'); zone items and unknown items -> None."""
    for f in sorted(features, key=len, reverse=True):
        if item.startswith(f + '_'):
            return f, item[len(f) + 1:]
    return None


def filter_feature_bins(rules, features=(), bins=()):
    """
    Rules containing at least one feature/bin item that matches: feature among `features` (any when empty) and
    bin among `bins` (any when empty). With both empty, all rules are kept.
    """
    if not features and not bins:
        return rules
    known = list(AR_FEATURE_DEFINITIONS)

    def hit(row):
        for item in set(row['antecedents']) | set(row['consequents']):
            fb = split_feature_item(item[5:] if item.startswith('zone_') else item, known)
            if fb and (not features or fb[0] in features) and (not bins or fb[1] in bins):
                return True
        return False
    return rules[rules.apply(hit, axis=1)] if len(rules) else rules


SIGNATURE_NAMES = ['dataset content', 'configurations', 'objects', 'time window', 'analysis type', 'zone grid',
                   'number of bins', 'mining features', 'minimum support', 'minimum confidence']


def mining_signature(data, configs, objects, transaction_type, grid, n_bins, min_support, min_confidence,
                     window=None, features=None):
    """Everything that determines a mining result; results are shown only while this is unchanged."""
    from .clustering import data_fingerprint
    return (data_fingerprint(data), tuple(sorted(map(str, configs))), tuple(sorted(map(str, objects))),
            tuple(window) if window else None, transaction_type, grid, n_bins,
            tuple(features) if features is not None else None, float(min_support), float(min_confidence))


def settings_table(signature, extra=()):
    """Mining selections as Setting/Value rows (for the settings export)."""
    rows = [{'Setting': n, 'Value': '' if v is None else (', '.join(map(str, v)) if isinstance(v, tuple) else str(v))}
            for n, v in zip(SIGNATURE_NAMES, signature)]
    rows += [{'Setting': k, 'Value': str(v)} for k, v in extra]
    return pd.DataFrame(rows)


AR_RESULT_KEYS = ['ar_rules', 'ar_frequent_itemsets', 'ar_transactions', 'ar_item_names', 'ar_trajectory_mapping',
                  'ar_trajectories_dict', 'ar_bin_info', 'ar_grid_info', 'ar_cooccurrence_matrix',
                  'ar_distance_matrix', 'ar_signature', 'ar_constant_features', 'ar_features_df',
                  'ar_n_trajectories']


# ============================================================================
# MATRIX COMPUTATIONS
# ============================================================================

def create_cooccurrence_matrix(transactions, item_names):
    """Create co-occurrence matrix from transactions."""
    n_items = len(item_names)
    item_to_idx = {item: idx for idx, item in enumerate(item_names)}
    cooccurrence = np.zeros((n_items, n_items))
    
    for transaction in transactions:
        for item1, item2 in combinations(transaction, 2):
            if item1 in item_to_idx and item2 in item_to_idx:
                idx1 = item_to_idx[item1]
                idx2 = item_to_idx[item2]
                cooccurrence[idx1, idx2] += 1
                cooccurrence[idx2, idx1] += 1
    
    return cooccurrence


def create_distance_matrix(cooccurrence_matrix):
    """Convert co-occurrence to distance matrix for MDS."""
    max_cooccurrence = np.max(cooccurrence_matrix)
    if max_cooccurrence == 0:
        return np.ones_like(cooccurrence_matrix)
    
    normalized = cooccurrence_matrix / max_cooccurrence
    distance_matrix = 1.0 / (1.0 + normalized)
    np.fill_diagonal(distance_matrix, 0)
    
    return distance_matrix


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_rules_network(rules_df, top_k=20):
    """Create network graph of association rules."""
    if len(rules_df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No rules to display", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False, font=dict(size=20))
        return fig
    
    top_rules = rules_df.head(top_k)
    G = nx.DiGraph()
    
    for _, rule in top_rules.iterrows():
        antecedents = list(rule['antecedents'])
        consequents = list(rule['consequents'])
        
        for item in antecedents + consequents:
            G.add_node(item)
        
        for ant in antecedents:
            for cons in consequents:
                # Use confidence for edge weight (rule strength)
                G.add_edge(ant, cons, weight=rule['confidence'], 
                          support=rule['support'], confidence=rule['confidence'])
    
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    edge_trace = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_data = G.edges[edge]
        edge_trace.append(
            go.Scatter(x=[x0, x1, None], y=[y0, y1, None], mode='lines',
                      line=dict(width=edge_data['weight']*3, color='rgba(125,125,125,0.5)'),
                      hoverinfo='text',
                      text=f"Support: {edge_data['support']:.3f}<br>Confidence: {edge_data['confidence']:.3f}",
                      showlegend=False)
        )
    
    node_x, node_y, node_text = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text', text=node_text, textposition="top center",
        marker=dict(size=20, color='lightblue', line=dict(width=2, color='darkblue')),
        hoverinfo='text', hovertext=node_text
    )
    
    fig = go.Figure(data=edge_trace + [node_trace])
    fig.update_layout(
        title=f"Association Rules Network (Top {len(top_rules)} Rules)",
        showlegend=False, hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600
    )
    return fig


def plot_support_confidence_scatter(rules_df, color_by='confidence'):
    """Create scatter plot of support vs confidence."""
    if len(rules_df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No rules to display", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False, font=dict(size=20))
        return fig
    
    hover_text = [
        f"Rule: {row['antecedents_str']} → {row['consequents_str']}<br>"
        f"Support: {row['support']:.3f}<br>Confidence: {row['confidence']:.3f}"
        for _, row in rules_df.iterrows()
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rules_df['support'], y=rules_df['confidence'], mode='markers',
        marker=dict(size=10, color=rules_df[color_by], colorscale='Viridis',
                   showscale=True, colorbar=dict(title=color_by.capitalize()),
                   line=dict(width=1, color='white')),
        text=hover_text, hoverinfo='text'
    ))
    
    fig.update_layout(
        title="Association Rules: Support vs Confidence",
        xaxis_title="Support", yaxis_title="Confidence",
        hovermode='closest', height=600
    )
    return fig


def plot_cooccurrence_heatmap(cooccurrence_matrix, item_names):
    """Create heatmap of item co-occurrences."""
    fig = go.Figure(data=go.Heatmap(
        z=cooccurrence_matrix, x=item_names, y=item_names,
        colorscale='Blues',
        hovertemplate='%{y} & %{x}<br>Co-occurrences: %{z}<extra></extra>'
    ))
    fig.update_layout(
        title="Item Co-occurrence Matrix",
        xaxis_title="Items", yaxis_title="Items",
        height=600, xaxis={'side': 'bottom'}
    )
    return fig


def plot_mds_projection(distance_matrix, item_names, n_components=2):
    """Create MDS projection of items."""
    if len(item_names) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Need at least 2 items for MDS", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False, font=dict(size=20))
        return fig
    
    mds = MDS(n_components=n_components, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(distance_matrix)
    
    if n_components == 2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=coords[:, 0], y=coords[:, 1], mode='markers+text', text=item_names,
            textposition='top center',
            marker=dict(size=12, color='lightcoral', line=dict(width=1, color='darkred')),
            hovertext=item_names, hoverinfo='text'
        ))
        fig.update_layout(
            title="MDS Projection of Items (Based on Co-occurrence)",
            xaxis_title="MDS Dimension 1", yaxis_title="MDS Dimension 2",
            height=600, hovermode='closest'
        )
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode='markers+text', text=item_names, textposition='top center',
            marker=dict(size=8, color='lightcoral', line=dict(width=1, color='darkred')),
            hovertext=item_names, hoverinfo='text'
        ))
        fig.update_layout(
            title="3D MDS Projection of Items (Based on Co-occurrence)",
            scene=dict(xaxis_title="MDS Dimension 1", yaxis_title="MDS Dimension 2", 
                      zaxis_title="MDS Dimension 3"),
            height=700, hovermode='closest'
        )
    return fig


def plot_top_rules_bars(rules_df, metric='confidence', top_k=15):
    """Create bar chart of top rules by metric."""
    if len(rules_df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No rules to display", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False, font=dict(size=20))
        return fig
    
    top_rules = rules_df.nlargest(min(top_k, len(rules_df)), metric)
    rule_labels = [f"{row['antecedents_str']} → {row['consequents_str']}" 
                   for _, row in top_rules.iterrows()]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=rule_labels, x=top_rules[metric], orientation='h',
        marker=dict(color=top_rules[metric], colorscale='Viridis'),
        text=top_rules[metric].round(3), textposition='auto',
        hovertemplate='<b>%{y}</b><br>' + f'{metric.capitalize()}: %{{x:.3f}}<extra></extra>'
    ))
    fig.update_layout(
        title=f"Top {len(top_rules)} Rules by {metric.capitalize()}",
        xaxis_title=metric.capitalize(), yaxis_title="Rules",
        height=max(400, len(top_rules) * 30),
        yaxis={'categoryorder': 'total ascending'}
    )
    return fig


def find_supporting_trajectories(antecedents, consequents, transactions, trajectory_mapping):
    """
    Find which trajectories support a given association rule.
    
    Args:
        antecedents: Set of antecedent items
        consequents: Set of consequent items
        transactions: List of transactions (list of lists)
        trajectory_mapping: List mapping transaction index to trajectory ID
        
    Returns:
        List of trajectory IDs that support the rule
    """
    supporting_trajectories = []
    
    for i, transaction in enumerate(transactions):
        transaction_set = set(transaction)
        
        # Check if transaction contains both antecedents and consequents
        if antecedents.issubset(transaction_set) and consequents.issubset(transaction_set):
            supporting_trajectories.append(trajectory_mapping[i])
    
    return supporting_trajectories


def plot_court_with_grid_and_trajectories(trajectories_dict, supporting_traj_ids, 
                                          antecedents, consequents, grid_info, court_type='Tennis'):
    """
    Visualise the court with grid overlay, highlighted rule zones, and supporting trajectories.
    Uses the same tennis court visualisation as Sequence Analysis.
    
    Args:
        trajectories_dict: Dictionary of {traj_id: DataFrame}
        supporting_traj_ids: List of trajectory IDs to visualise
        antecedents: Set of antecedent zone labels
        consequents: Set of consequent zone labels
        grid_info: Grid information dictionary from create_spatial_grid()
        court_type: Type of court ('Tennis' or 'Football')
        
    Returns:
        Plotly figure
    """
    # Import the court creation functions from parent module
    import sys
    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from streamlit_visualization import create_pitch_figure

    # Court or pitch only when that court type is selected; zone items in Combined rules carry a 'zone_' prefix
    strip = lambda items: {i[5:] if i.startswith('zone_') else i for i in items}
    fig = sequence_analysis.draw_zone_grid(create_pitch_figure(court_type), grid_info,
                                           strip(antecedents), strip(consequents))

    # Plot supporting trajectories
    if supporting_traj_ids:
        colors = px.colors.qualitative.Set1
        for idx, traj_id in enumerate(supporting_traj_ids[:10]):  # Limit to 10 for clarity
            if traj_id in trajectories_dict:
                traj_data = trajectories_dict[traj_id]
                color = colors[idx % len(colors)]
                
                # Plot trajectory path
                fig.add_trace(go.Scatter(
                    x=traj_data['x'],
                    y=traj_data['y'],
                    mode='lines+markers',
                    name=f'Traj {traj_id}',
                    line=dict(color=color, width=3),
                    marker=dict(size=4, color=color),
                    hovertemplate=f'<b>Trajectory {traj_id}</b><br>x: %{{x:.2f}}m<br>y: %{{y:.2f}}m<extra></extra>'
                ))
                
                # Add start marker (green circle)
                fig.add_trace(go.Scatter(
                    x=[traj_data['x'].iloc[0]],
                    y=[traj_data['y'].iloc[0]],
                    mode='markers',
                    marker=dict(size=15, color='green', symbol='circle', 
                               line=dict(color='white', width=2)),
                    showlegend=False,
                    hovertext=f'Traj {traj_id} - Start',
                    hovertemplate='<b>%{hovertext}</b><extra></extra>'
                ))
                
                # Add end marker (red square)
                fig.add_trace(go.Scatter(
                    x=[traj_data['x'].iloc[-1]],
                    y=[traj_data['y'].iloc[-1]],
                    mode='markers',
                    marker=dict(size=15, color='red', symbol='square', 
                               line=dict(color='white', width=2)),
                    showlegend=False,
                    hovertext=f'Traj {traj_id} - End',
                    hovertemplate='<b>%{hovertext}</b><extra></extra>'
                ))
    
    fig.update_layout(height=700)
    return fig


# ============================================================================
# MAIN UI FUNCTION
# ============================================================================

def render_association_rules_section(data, selected_configs, selected_objects, create_spatial_grid_func):
    """
    Main function to render the Association Rules analysis section.
    
    Args:
        data: Full DataFrame with trajectory data
        selected_configs: List of selected configurations
        selected_objects: List of selected objects
        create_spatial_grid_func: Function to create spatial grid
    """
    st.header("Association Rule Learning")
    
    st.info("""
    **Discover relationships between trajectory patterns using Association Rules:**

    *Example: "If an object visits zones A1 and B2, it will likely also visit zone C3."*

    Association rules find patterns like **{A} → {B}** where:
    - **Antecedent {A}**: Items that appear together (left-hand side of the rule)
    - **Consequent {B}**: Items that co-occur with the antecedent (right-hand side of the rule)
    
    **Two Primary Metrics (Theoretical Focus):**
    - **Support** ⭐: How common is this pattern? (frequency in all trajectories)
    - **Confidence** ⭐: How reliable is this rule? (probability of finding consequent when antecedent is present)
    
    **Goal**: Find patterns that are both **common enough** (support) and **strong enough** (confidence) to be meaningful
    
    **Applications:** Movement pattern discovery, spatial behaviour analysis, trajectory prediction
    """)
    
    with st.expander("ℹ️ Market Basket Analysis Example", expanded=False):
        st.markdown("""
        **Classic Example: Supermarket Transactions**
        
        Consider 5 transactions:
        1. {Milk, Diaper, Beer}
        2. {Milk, Diaper, Cola}
        3. {Milk, Beer}
        4. {Diaper, Beer, Cola}
        5. {Milk, Diaper, Beer, Cola}
        
        **Rule:** {Milk, Diaper} → {Beer}
        - {Milk, Diaper} occurs in transactions 1, 2 and 5: **3** of 5
        - {Milk, Diaper, Beer} occurs in transactions 1 and 5: **2** of 5
        - **Support**: 2/5 = 0.4 (the whole pattern occurs in 2 of the 5 transactions)
        - **Confidence**: 2/3 ≈ 0.67 (of the 3 transactions with Milk and Diaper, 2 also contain Beer)

        **Interpretation**: Beer is in 4 of the 5 transactions (consequent prevalence 4/5 = 0.8). A confidence
        of 2/3 is therefore *lower* than the share of beer buyers overall: knowing that a basket contains milk
        and diapers does not make beer more likely here. Always compare confidence with the prevalence of the
        consequent.
        """)
    
    st.markdown('---')
    
    if not selected_configs or not selected_objects:
        st.warning("Please select at least one configuration and one object from the sidebar.")
        return
    
    window = sequence_analysis.time_window_inputs(st, data, "ar_start", "ar_end")
    if window is None:
        return
    filtered_df = sequence_analysis.select_window(data, selected_configs, selected_objects, *window)

    if len(filtered_df) == 0:
        st.error("No data available for the selected configurations and objects in this time window.")
        return

    # Trajectories are configuration-object pairs; rows are sampled positions
    st.success(sequence_analysis.time_window_text(*window, filtered_df))
    
    # Configuration Panel
    st.subheader("Transaction Configuration")
    
    st.markdown("""
    **What patterns do you want to discover?**
    
    Choose how to represent trajectories as "transactions" (like a shopping basket):
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        transaction_type = st.radio(
            "Analysis Type:",
            ["Spatial Zones", "Feature Bins", "Combined (Spatial + Features)"],
            help="Select what type of patterns to mine"
        )
        
        # Add explanations for each type
        if transaction_type == "Spatial Zones":
            st.info("""
            **Spatial Zones (WHERE)**
            
            Find patterns based on **which zones** are visited.
            
            *Example rule:* "If player visits zones B3 and C4, they will likely visit E4"
            """)
        elif transaction_type == "Feature Bins":
            st.info("""
            **Feature Bins (HOW)**
            
            Find patterns based on **trajectory characteristics** (speed, distance, duration).
            
            *Example rule:* "If trajectory has high speed and long distance, it will have medium duration"
            
            """)
        else:  # Combined
            st.info("""
            **Combined (WHERE + HOW)**
            
            Find patterns combining **both zones AND features**.
            
            *Example rule:* "If object visits B3 with high speed, it will visit E4 with medium duration"
            
            **Best for:** Comprehensive analysis
            """)
    
    spatial = transaction_type in ["Spatial Zones", "Combined (Spatial + Features)"]
    binned = transaction_type in ["Feature Bins", "Combined (Spatial + Features)"]
    court_type = st.session_state.get('court_type', 'No selection')
    grid_info = grid_settings = n_bins = mining_features = None

    with col2:
        if spatial:
            st.markdown("**Spatial Grid Settings:**")
            grid_rows = st.slider("Grid Rows", 2, 6, 3, key="ar_grid_rows",
                                 help="Number of horizontal divisions")
            grid_cols = st.slider("Grid Columns", 2, 8, 5, key="ar_grid_cols",
                                 help="Number of vertical divisions")
            grid_info, grid_settings = sequence_analysis.zone_grid_controls(st, court_type, filtered_df, "ar",
                                                                            grid_rows, grid_cols)
            if grid_info is None:
                return
            st.caption(sequence_analysis.grid_description(grid_info))
            n_outside = sequence_analysis.outside_grid_count(filtered_df, grid_info)
            if n_outside:
                st.warning(f"{n_outside} of {len(filtered_df)} positions lie outside the explicit bounds and get no zone.")

        if binned:
            st.markdown("**Feature Binning:**")
            n_bins = st.select_slider("Number of Bins", [2, 3, 4, 5], value=3, key="ar_n_bins",
                                      help="How many categories (e.g., low/medium/high)")
            st.caption(f"Features grouped into {n_bins} equal-width bins between their minimum and maximum")

    if binned:
        st.markdown("**Features for mining**")
        if 'ar_features' not in st.session_state:
            st.session_state.ar_features = list(AR_DEFAULT_FEATURES)
        mining_features = st.multiselect(
            "Features (type to search)", AR_FEATURES, key="ar_features", format_func=feature_label,
            help="Same names, formulas and units as in Clustering, plus x Range and y Range. Each selected feature "
                 "becomes one item per trajectory, e.g. total_distance_high.")
        with st.expander("Feature definitions", expanded=False):
            st.dataframe(feature_definitions_table(), hide_index=True, use_container_width=True)
            st.caption("Speeds and durations are in m per tst unit and tst units (m/s only when tst is in seconds).")
        if not mining_features:
            st.warning("Select at least one feature.")
            return

        # Bin boundaries from the current selection, before mining
        trajectories_now = {f"{c}_obj{o}": g.sort_values('tst', kind='mergesort')
                            for (c, o), g in filtered_df.groupby(['config_source', 'obj'])}
        preview = pd.DataFrame([association_features(g) for g in trajectories_now.values()],
                               index=list(trajectories_now))[mining_features]
        constant = constant_features(preview)
        if constant:
            st.warning("Constant in this selection (one value for every trajectory, so no bins and no items): "
                       + ", ".join(f"{feature_label(f)} = {preview[f].dropna().iloc[0]:.6g}" if preview[f].notna().any()
                                   else f"{feature_label(f)} (no values)" for f in constant) + ".")
        with st.expander("Bin boundaries", expanded=False):
            bt = bin_table(feature_items(preview, n_bins)[1], n_bins)
            st.dataframe(bt, hide_index=True, use_container_width=True,
                         column_config={c: st.column_config.NumberColumn(c, format="%.6f") for c in ('Lower', 'Upper')})
            st.caption("A value v belongs to a bin when Lower ≤ v < Upper; the highest bin also includes the maximum. "
                       "The Interval column shows the boundaries at full precision.")

    st.markdown("---")
    
    # Threshold Settings
    st.subheader("Rule Mining Thresholds")
    
    st.info("""
    **Primary Metrics (Theoretical Focus):**
    - **Support ⭐**: How frequently the pattern appears (common vs. rare patterns)
    - **Confidence ⭐**: Probability that consequent occurs given antecedent (rule strength)
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        min_support = st.slider("**Min Support** ⭐", 0.05, 0.5, 0.1, 0.05,
                               help="Support = P(A ∪ B): Frequency of the pattern in all trajectories. "
                                    "Higher support = more common pattern", 
                               key="ar_min_support")
        st.caption(f"Pattern must appear in ≥{min_support*100:.0f}% of trajectories")
    
    with col2:
        min_confidence = st.slider("**Min Confidence** ⭐", 0.1, 1.0, 0.5, 0.05,
                                   help="Confidence = P(B|A): Probability that consequent occurs given antecedent. "
                                        "Higher confidence = stronger rule", 
                                   key="ar_min_confidence")
        st.caption(f"Rule must hold in ≥{min_confidence*100:.0f}% of cases")
    
    st.markdown("---")

    signature = mining_signature(data, selected_configs, selected_objects, transaction_type,
                                 grid_settings if spatial else None, n_bins if binned else None,
                                 min_support, min_confidence, window=window,
                                 features=mining_features if binned else None)

    # Mine Rules Button
    if st.button("Mine Association Rules", type="primary"):
        # Nothing from an earlier computation may survive a new one (also not after an error)
        for key in AR_RESULT_KEYS:
            st.session_state.pop(key, None)
        with st.spinner("Preparing transactions and mining rules..."):
            try:
                trajectories_dict = {f"{config}_obj{obj}": group.sort_values('tst', kind='mergesort')
                                     for (config, obj), group in filtered_df.groupby(['config_source', 'obj'])}
                st.session_state.ar_trajectories_dict = trajectories_dict

                zones_by_id, items_by_id = {}, {}
                if spatial:
                    st.session_state.ar_grid_info = grid_info
                    get_zone = grid_info['get_zone']
                    for tid, g in trajectories_dict.items():
                        zones_by_id[tid] = sorted({z for z in map(get_zone, g['x'], g['y']) if z})
                if binned:
                    features_df = pd.DataFrame([association_features(g) for g in trajectories_dict.values()],
                                               index=list(trajectories_dict))[mining_features]
                    rows, bin_info = feature_items(features_df, n_bins)
                    items_by_id = dict(zip(trajectories_dict, rows))
                    st.session_state.ar_bin_info = bin_info
                    st.session_state.ar_constant_features = constant_features(features_df)
                    st.session_state.ar_features_df = features_df

                if transaction_type == "Spatial Zones":
                    by_id = zones_by_id
                elif transaction_type == "Feature Bins":
                    by_id = items_by_id
                if transaction_type == "Combined (Spatial + Features)":
                    transactions, item_names, trajectory_mapping = combine_transactions(zones_by_id, items_by_id)
                else:
                    trajectory_mapping = [t for t, items in by_id.items() if items]
                    transactions = [list(by_id[t]) for t in trajectory_mapping]
                    item_names = sorted({i for t in transactions for i in t})
                st.session_state.ar_trajectory_mapping = trajectory_mapping
                st.session_state.ar_n_trajectories = len(trajectories_dict)
                st.session_state.ar_transactions = transactions
                st.session_state.ar_item_names = item_names

                # Show transaction preview
                st.subheader("Transaction Preview")
                st.write(f"Total transactions: {len(transactions)} · Unique items: {len(item_names)}")
                st.dataframe(pd.DataFrame({'Trajectory': trajectory_mapping[:10],
                                           'Items': [', '.join(t) for t in transactions[:10]]}),
                             use_container_width=True, hide_index=True)

                # Compute association rules
                st.subheader("Mining Association Rules...")
                rules_df, frequent_itemsets = compute_association_rules(
                    transactions, min_support, min_confidence
                )
                if len(rules_df) > 0:
                    rules_df = add_rule_counts(rules_df, transactions)

                st.session_state.ar_rules = rules_df
                st.session_state.ar_signature = signature
                st.session_state.ar_frequent_itemsets = frequent_itemsets
                
                # Compute matrices
                cooccurrence_matrix = create_cooccurrence_matrix(transactions, item_names)
                distance_matrix = create_distance_matrix(cooccurrence_matrix)
                
                st.session_state.ar_cooccurrence_matrix = cooccurrence_matrix
                st.session_state.ar_distance_matrix = distance_matrix
                
                if len(rules_df) == 0:
                    st.warning("No rules found. Try lowering the thresholds.")
                else:
                    st.success(f"Found {len(rules_df)} association rules!")
            
            except Exception as e:
                st.error(f"Error during rule mining: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Results belong to the inputs they were computed from: hide them as soon as any input changes
    if st.session_state.get('ar_rules') is not None and st.session_state.get('ar_signature') != signature:
        old = st.session_state.get('ar_signature') or ()
        changed = [n for i, n in enumerate(SIGNATURE_NAMES) if i >= len(old) or old[i] != signature[i]]
        st.warning("**Inputs changed. Recompute to update results.** Changed since the last mining run: "
                   + ", ".join(changed) + ". Rules, plots and downloads of the earlier run are hidden until you "
                   "click Mine Association Rules again.")
        return

    # Display results if available
    if st.session_state.get('ar_rules') is not None and len(st.session_state.ar_rules) > 0:
        st.markdown("---")
        st.header("Association Rules Results")

        rules_df = st.session_state.ar_rules
        run_tag = hashlib.sha256(repr(signature).encode()).hexdigest()[:10]
        n_tx = len(st.session_state.ar_transactions)
        n_traj = st.session_state.get('ar_n_trajectories', n_tx)
        st.caption(f"Mining run {run_tag}: {n_tx} transactions, "
                   f"{len(rules_df)} rules (minimum support {min_support:g}, minimum confidence "
                   f"{min_confidence:g}).")
        st.markdown(f"**Denominator N = {n_tx} transactions** (of {n_traj} trajectories in the selection"
                    + (f"; {n_traj - n_tx} without any item left out" if n_traj != n_tx else "") + "). "
                    + sequence_analysis.time_window_text(*window, filtered_df) + ".")
        if binned:
            st.caption("Mining features: " + ", ".join(map(feature_label, mining_features)) + f" · {n_bins} bins"
                       + (" · constant, no items: " + ", ".join(st.session_state.ar_constant_features)
                          if st.session_state.get('ar_constant_features') else ""))
        if spatial:
            st.caption(sequence_analysis.grid_description(st.session_state.ar_grid_info))
        settings_df = settings_table(signature, [('run', run_tag), ('transactions N', n_tx),
                                                 ('trajectories', n_traj), ('rules', len(rules_df))])
        e1, e2 = st.columns(2)
        e1.download_button("Download mining settings (CSV)", settings_df.to_csv(index=False),
                           f"association_settings_run{run_tag}.csv", "text/csv", key="ar_download_settings")
        if st.session_state.get('ar_bin_info'):
            e2.download_button("Download bin boundaries (CSV, full precision)",
                               bin_table(st.session_state.ar_bin_info, n_bins).to_csv(index=False, float_format='%.17g'),
                               f"association_bins_run{run_tag}.csv", "text/csv", key="ar_download_bins")

        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "Rules Table", "Network", 
            "Co-occurrence Heatmap", "MDS Projection", 
            "Distance Matrix", "Top Rules", "Supporting Trajectories"
        ])
        
        with tab1:
            st.subheader("Rules Table")
            st.markdown("""
            **All discovered association rules in a sortable table.**
            
            Each row shows a rule (Antecedents → Consequents) with its Support (how common) and Confidence (how reliable).
            Sort by either metric to find the most interesting patterns.
            """)
            
            all_items = sorted({i for s in rules_df['antecedents'] for i in s} |
                               {i for s in rules_df['consequents'] for i in s})
            f1, f2, f3 = st.columns(3)
            with f1:
                find_text = st.text_input("Find rules containing", key="ar_find_text",
                                          help="Case-insensitive text search in the rule, e.g. avg_speed_high")
            with f2:
                find_ante = st.multiselect("Antecedent contains", all_items, key="ar_find_antecedent")
            with f3:
                find_cons = st.multiselect("Consequent contains", all_items, key="ar_find_consequent")
            matching = filter_rules(rules_df, find_text, find_ante, find_cons)
            if binned:
                st.markdown("**Filter by feature/bin items** (applies to the rules found; it does not change mining)")
                rule_items = [split_feature_item(i[5:] if i.startswith('zone_') else i, AR_FEATURES)
                              for i in all_items]
                fb1, fb2 = st.columns(2)
                with fb1:
                    filt_features = st.multiselect("Rules with an item of feature", sorted({x[0] for x in rule_items if x}),
                                                   key="ar_filter_features", format_func=feature_label)
                with fb2:
                    filt_bins = st.multiselect("in bin", bin_labels_for(n_bins), key="ar_filter_bins",
                                               help="Leave empty for any bin.")
                matching = filter_feature_bins(matching, filt_features, filt_bins)

            col1, col2 = st.columns(2)
            with col1:
                sort_by = st.selectbox("Sort by:", ["confidence", "support"], 
                                      help="Default sorted by Confidence (rule strength)")
            with col2:
                top_k_table = st.number_input("Show top N rules:", min_value=1, max_value=len(rules_df),
                                              value=min(50, len(rules_df)), key="ar_top_n")
            
            # Stable ordering: metric, then the rule text, so ties are listed reproducibly
            display_rules = matching.sort_values([sort_by, 'rule'], ascending=[False, True],
                                                 kind='mergesort').head(int(top_k_table))
            st.markdown(f"Showing **{len(display_rules)}** of {len(matching)} matching rules "
                        f"({len(rules_df)} rules in total).")
            
            n_total = int(rules_df['n_transactions'].iloc[0])
            display_df = pd.DataFrame({
                'Rule': display_rules['rule'],
                'Antecedent count': display_rules['n_antecedent'],
                'Consequent count': display_rules['n_consequent'],
                'Joint count': display_rules['n_joint'],
                'Support': display_rules['support'],
                'Support (counts)': display_rules['n_joint'].astype(str) + f'/{n_total}',
                'Confidence': display_rules['confidence'],
                'Confidence (counts)': display_rules['n_joint'].astype(str) + '/' + display_rules['n_antecedent'].astype(str),
                'Consequent prevalence': display_rules['consequent_prevalence'],
                'Prevalence (counts)': display_rules['n_consequent'].astype(str) + f'/{n_total}',
            })
            
            st.dataframe(
                display_df, use_container_width=True, height=400, hide_index=True,
                column_config={c: st.column_config.NumberColumn(c, format="%.3f")
                               for c in ('Support', 'Confidence', 'Consequent prevalence')}
            )
            
            st.caption(f"N = {n_total} transactions. Support = joint count / N; confidence = joint count / antecedent "
                       "count; consequent prevalence = consequent count / N. A confidence close to the consequent "
                       "prevalence means the antecedent adds little information. Display rounded to three "
                       "decimals; downloads keep full precision.")
            
            d1, d2 = st.columns(2)
            d1.download_button(f"Download all rules (CSV, {len(rules_df)} rules)", rules_export(rules_df),
                               f"association_rules_all_run{run_tag}.csv", "text/csv", key="ar_download_all")
            d2.download_button(f"Download displayed rules only (CSV, {len(display_rules)} rules)",
                               rules_export(display_rules), f"association_rules_displayed_run{run_tag}.csv",
                               "text/csv", key="ar_download_displayed")
        
        with tab2:
            st.subheader("Network Graph")
            st.markdown("""
            **Visual network showing how items connect through association rules.**
            
            - **Nodes** = Individual items (zones or features)
            - **Edges** = Association rules connecting items
            - **Edge thickness** = Confidence strength (thicker = stronger rule)
            
            Use this to identify central items and complex relationships at a glance.
            """)
            top_k_network = st.slider("Number of top rules:", 5, 50, 20, key="top_k_network")
            fig = plot_rules_network(rules_df, top_k=top_k_network)
            render_interactive_chart(st, fig)
        
        with tab3:
            st.subheader("Co-occurrence Heatmap")
            st.markdown("""
            **Shows how often pairs of items appear together in transactions.**
            
            - **Darker colours** = Items frequently co-occur (appear together in many trajectories)
            - **Lighter colours** = Items rarely appear together
            
            This helps identify which items have strong associations even before looking at specific rules.
            """)
            if st.session_state.get('ar_cooccurrence_matrix') is not None:
                fig = plot_cooccurrence_heatmap(
                    st.session_state.ar_cooccurrence_matrix,
                    st.session_state.ar_item_names
                )
                render_interactive_chart(st, fig)
            else:
                st.info("Co-occurrence matrix not available.")
        
        with tab4:
            st.subheader("MDS Projection")
            st.markdown("""
            **Multidimensional Scaling (MDS): Items positioned by similarity.**
            
            - **Close together** = Items that frequently co-occur (similar patterns)
            - **Far apart** = Items that rarely appear together (different patterns)
            
            Think of it as a "map" where similar items cluster together. Useful for discovering natural groupings in your data.
            """)
            if st.session_state.get('ar_distance_matrix') is not None:
                mds_dims = st.radio("Dimensions:", [2, 3], horizontal=True, key="ar_mds_dimensions")
                fig = plot_mds_projection(
                    st.session_state.ar_distance_matrix,
                    st.session_state.ar_item_names,
                    n_components=mds_dims
                )
                render_interactive_chart(st, fig)
            else:
                st.info("Distance matrix not available.")
        
        with tab5:
            st.subheader("Distance Matrix")
            st.markdown("""
            **Numerical matrix showing dissimilarity between all item pairs.**
            
            - **Low values (green)** = Items are similar (often appear together)
            - **High values (red)** = Items are dissimilar (rarely appear together)
            
            This is the raw data behind the MDS projection above.
            """)
            if st.session_state.get('ar_distance_matrix') is not None:
                fig = go.Figure(data=go.Heatmap(
                    z=st.session_state.ar_distance_matrix,
                    x=st.session_state.ar_item_names,
                    y=st.session_state.ar_item_names,
                    colorscale='RdYlGn_r',
                    hovertemplate='%{y} - %{x}<br>Distance: %{z:.3f}<extra></extra>'
                ))
                fig.update_layout(
                    title="Item Distance Matrix",
                    xaxis_title="Items", yaxis_title="Items", height=600
                )
                render_interactive_chart(st, fig)
            else:
                st.info("Distance matrix not available.")
        
        with tab6:
            st.subheader("Top Rules")
            st.markdown("""
            **Bar chart ranking rules by Support or Confidence.**
            
            - **Support ranking** = Shows the most frequent patterns
            - **Confidence ranking** = Shows the most reliable rules
            
            Quick way to identify your strongest patterns at a glance.
            """)
            col1, col2 = st.columns(2)
            with col1:
                rank_metric = st.selectbox("Rank by:", ["support", "confidence"])
            with col2:
                top_k_bars = st.slider("Number of rules:", 5, 30, 15, key="top_k_bars")
            
            fig = plot_top_rules_bars(rules_df, metric=rank_metric, top_k=top_k_bars)
            render_interactive_chart(st, fig)
        
        with tab7:
            st.subheader("Supporting Trajectories")
            st.markdown("""
            **See the actual trajectories that create each association rule.**
            
            - Select a rule to see which specific trajectory IDs satisfy it
            - View the trajectories visualised on a tennis court with grid overlay
            - Understand the spatial patterns behind the statistical rules
            
            This connects abstract patterns back to real movement data.
            """)
            
            # Check if required data is available
            if ('ar_trajectory_mapping' not in st.session_state or 
                'ar_trajectories_dict' not in st.session_state or
                'ar_transactions' not in st.session_state):
                st.warning("Trajectory data not available. Please re-run the association rule mining.")
            else:
                # Create rule display strings for selection
                rule_strings = []
                for idx, row in rules_df.iterrows():
                    antecedents_str = ', '.join(list(row['antecedents']))
                    consequents_str = ', '.join(list(row['consequents']))
                    rule_str = (f"{antecedents_str} → {consequents_str} (support {row['n_joint']}/{row['n_transactions']}, "
                                f"confidence {row['n_joint']}/{row['n_antecedent']})")
                    rule_strings.append(rule_str)
                
                # Rule selector
                selected_rule_idx = st.selectbox(
                    "Select a rule to explore:",
                    range(len(rule_strings)),
                    format_func=lambda i: rule_strings[i]
                )
                
                # Get selected rule details
                selected_rule = rules_df.iloc[selected_rule_idx]
                antecedents = selected_rule['antecedents']
                consequents = selected_rule['consequents']
                
                # Find supporting trajectories
                supporting_traj_ids = find_supporting_trajectories(
                    antecedents, 
                    consequents, 
                    st.session_state.ar_transactions,
                    st.session_state.ar_trajectory_mapping
                )
                
                # Display metrics
                total_transactions = len(st.session_state.ar_transactions)
                num_supporting = len(supporting_traj_ids)
                percentage = (num_supporting / total_transactions * 100) if total_transactions > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Supporting Trajectories", num_supporting)
                with col2:
                    st.metric("Total Transactions", total_transactions)
                with col3:
                    st.metric("Percentage", f"{format_number(percentage, 1)}%")
                
                # Display trajectory IDs
                with st.expander("Show Trajectory IDs", expanded=False):
                    if supporting_traj_ids:
                        st.write(f"Trajectory IDs: {', '.join(map(str, supporting_traj_ids))}")
                    else:
                        st.info("No trajectories found supporting this rule.")
                
                # Plot supporting trajectories on court
                if supporting_traj_ids:
                    st.subheader("Court Visualisation")
                    
                    # Check if grid_info is available (only for Spatial Zones and Combined types)
                    if 'ar_grid_info' in st.session_state:
                        # Determine court type from session state
                        court_type = st.session_state.get('court_type', 'Tennis')
                        
                        # Create and display the plot with grid overlay
                        fig = plot_court_with_grid_and_trajectories(
                            st.session_state.ar_trajectories_dict,
                            supporting_traj_ids[:20],  # Limit to 20 trajectories for performance
                            antecedents,
                            consequents,
                            st.session_state.ar_grid_info,
                            court_type
                        )
                        
                        if fig is not None:
                            render_interactive_chart(st, fig)
                            
                            st.info(f"""
                            **Visualisation Details:**
                            - 🔴 **Red zones** = Antecedent ({', '.join(list(antecedents))})
                            - 🔵 **Blue zones** = Consequent ({', '.join(list(consequents))})
                            - Each trajectory is shown as a coloured path
                            - 🟢 Green circles indicate start points
                            - 🔴 Red squares indicate end points
                            - Showing up to 20 trajectories for clarity
                            - Hover over points to see trajectory IDs and coordinates
                            - All {num_supporting} trajectories contain both antecedent and consequent zones
                            """)
                        else:
                            st.warning("Could not generate trajectory plot.")
                    else:
                        st.warning("Grid visualisation is only available for 'Spatial Zones' and 'Combined' transaction types.")
                else:
                    st.info("No trajectories to visualise for this rule.")
        
        with st.expander("How to Interpret Results", expanded=False):
            st.markdown("""
            **Primary Metrics (Theoretical Focus):**
            
            ### 1. **Support** = P(A ∪ B) ⭐
            - **Definition**: Frequency of the pattern in all trajectories
            - **Interpretation**: 
              - High support (e.g., 0.3) = pattern occurs in 30% of trajectories (common)
              - Low support (e.g., 0.05) = pattern occurs in 5% of trajectories (rare)
            - **Use**: Identifies how prevalent a pattern is in your data
            
            ### 2. **Confidence** = P(B|A) = P(A ∪ B) / P(A) ⭐
            - **Definition**: Probability that consequent occurs given antecedent
            - **Interpretation**: 
              - Confidence of 0.8 = 80% of the time A appears, B also appears
              - Confidence of 0.5 = 50% of the time (moderate rule)
              - Confidence of 1.0 = 100% of the time (perfect rule)
            - **Use**: Measures the strength/reliability of the rule
            
            ---
            
            **Key Insights:**
            - **Support** tells you if the pattern is common enough to matter
            - **Confidence** tells you if the rule is reliable/predictive
            - Together, they balance frequency vs. strength
            - **Strong patterns** have BOTH high support AND high confidence
            - Association rules show correlation, NOT causation
            - Focus on rules with both reasonable support (not too rare) AND high confidence (reliable)
            """)

