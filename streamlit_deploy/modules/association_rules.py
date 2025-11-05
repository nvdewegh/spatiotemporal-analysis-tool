"""
Association Rules Analysis Module

This module implements association rule learning (market basket analysis) 
for spatiotemporal trajectory data.

Key Features:
- Transaction preparation (spatial zones, feature bins, combined)
- Apriori algorithm for frequent itemset mining
- Association rule generation with multiple metrics
- Interactive visualizations (network, scatter, heatmap, MDS)
"""

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
from .common import render_interactive_chart
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


def prepare_feature_transactions(features_df, n_bins=3):
    """
    Convert feature data into transactions based on binned feature values.
    
    Args:
        features_df: DataFrame with trajectory features
        n_bins: Number of bins for each feature
        
    Returns:
        transactions: List of lists
        item_names: Sorted list of unique items
    """
    transactions = []
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['config', 'object', 'rally_id']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    for idx, row in features_df.iterrows():
        transaction = []
        for col in feature_cols:
            value = row[col]
            if pd.notna(value):
                col_min = features_df[col].min()
                col_max = features_df[col].max()
                
                if col_max > col_min:
                    bin_width = (col_max - col_min) / n_bins
                    bin_idx = min(int((value - col_min) / bin_width), n_bins - 1)
                    bin_labels = ['low', 'medium', 'high'] if n_bins == 3 else [f'bin{i+1}' for i in range(n_bins)]
                    item = f"{col}_{bin_labels[bin_idx]}"
                    transaction.append(item)
        
        if transaction:
            transactions.append(transaction)
    
    all_items = set()
    for transaction in transactions:
        all_items.update(transaction)
    item_names = sorted(list(all_items))
    
    return transactions, item_names


def prepare_combined_transactions(spatial_trans, spatial_items, feature_trans, feature_items):
    """Combine spatial and feature transactions."""
    min_len = min(len(spatial_trans), len(feature_trans))
    transactions = []
    
    for i in range(min_len):
        spatial_prefixed = [f"zone_{item}" for item in spatial_trans[i]]
        combined = spatial_prefixed + feature_trans[i]
        transactions.append(combined)
    
    spatial_items_prefixed = [f"zone_{item}" for item in spatial_items]
    item_names = sorted(list(set(spatial_items_prefixed + feature_items)))
    
    return transactions, item_names


# ============================================================================
# ASSOCIATION RULE MINING
# ============================================================================

def compute_association_rules(transactions, min_support=0.1, min_confidence=0.5, min_lift=1.0):
    """
    Compute association rules using Apriori algorithm.
    
    Args:
        transactions: List of transaction lists
        min_support: Minimum support threshold
        min_confidence: Minimum confidence threshold
        min_lift: Minimum lift threshold
        
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
        # Filter by lift if specified (keeping this for internal compatibility)
        if min_lift > 1.0:
            rules = rules[rules['lift'] >= min_lift]
        rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(sorted(list(x))))
        rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(sorted(list(x))))
        # Sort by confidence (primary metric) by default
        rules = rules.sort_values('confidence', ascending=False).reset_index(drop=True)
        return rules, frequent_itemsets
    except ValueError:
        return pd.DataFrame(), frequent_itemsets

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


def plot_support_confidence_scatter(rules_df, color_by='lift'):
    """Create scatter plot of support vs confidence."""
    if len(rules_df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No rules to display", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False, font=dict(size=20))
        return fig
    
    hover_text = [
        f"Rule: {row['antecedents_str']} → {row['consequents_str']}<br>"
        f"Support: {row['support']:.3f}<br>Confidence: {row['confidence']:.3f}<br>Lift: {row['lift']:.3f}"
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


def plot_top_rules_bars(rules_df, metric='lift', top_k=15):
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
    Visualize the court with grid overlay, highlighted rule zones, and supporting trajectories.
    Uses the same tennis court visualization as Sequence Analysis.
    
    Args:
        trajectories_dict: Dictionary of {traj_id: DataFrame}
        supporting_traj_ids: List of trajectory IDs to visualize
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
    
    # Import the court creation function
    from streamlit_visualization import create_pitch_figure
    
    # Create base tennis court figure (same as Sequence Analysis)
    fig = create_pitch_figure(court_type)
    
    # Get grid information
    x_bins = grid_info['x_bins']
    y_bins = grid_info['y_bins']
    court_width = grid_info['court_width']
    court_height = grid_info['court_height']
    buffer = grid_info['buffer']
    actual_rows = grid_info['grid_rows']
    actual_cols = grid_info['grid_cols']
    zone_labels = grid_info['zone_labels']
    
    # Add buffer zone background (light gray)
    fig.add_shape(
        type="rect",
        x0=-buffer, y0=-buffer,
        x1=court_width + buffer, y1=court_height + buffer,
        fillcolor='rgba(200, 200, 200, 0.3)',
        line=dict(color='rgba(150, 150, 150, 0.5)', width=2),
        layer="below"
    )
    
    # Add grid lines (vertical)
    for x in x_bins:
        fig.add_shape(
            type="line",
            x0=x, y0=y_bins[0],
            x1=x, y1=y_bins[-1],
            line=dict(color='rgba(255, 0, 0, 0.6)', width=3, dash='dash'),
            layer="above"
        )
    
    # Add grid lines (horizontal)
    for y in y_bins:
        fig.add_shape(
            type="line",
            x0=x_bins[0], y0=y,
            x1=x_bins[-1], y1=y,
            line=dict(color='rgba(255, 0, 0, 0.6)', width=3, dash='dash'),
            layer="above"
        )
    
    # Add zone labels and highlight zones involved in the rule
    for row in range(actual_rows):
        for col in range(actual_cols):
            zone_idx = row * actual_cols + col
            zone_label = zone_labels[zone_idx]
            
            x_center = (x_bins[col] + x_bins[col + 1]) / 2
            y_center = (y_bins[row] + y_bins[row + 1]) / 2
            
            # Determine if this is a buffer zone or court zone
            is_buffer = (col == 0 or col == actual_cols - 1 or 
                       row == 0 or row == actual_rows - 1)
            
            # Check if zone is part of the rule
            is_antecedent = zone_label in antecedents
            is_consequent = zone_label in consequents
            
            # Highlight zones involved in the rule
            if is_antecedent or is_consequent:
                # Add colored rectangle for rule zones
                highlight_color = 'rgba(255, 0, 0, 0.4)' if is_antecedent else 'rgba(0, 0, 255, 0.4)'
                fig.add_shape(
                    type="rect",
                    x0=x_bins[col], y0=y_bins[row],
                    x1=x_bins[col + 1], y1=y_bins[row + 1],
                    fillcolor=highlight_color,
                    line=dict(color='red' if is_antecedent else 'blue', width=4),
                    layer="above"
                )
            
            # Add zone label with appropriate styling
            if is_antecedent:
                # Antecedent zones - red background
                bgcolor = 'rgba(200, 0, 0, 0.9)'
                bordercolor = 'darkred'
                font_size = 22
            elif is_consequent:
                # Consequent zones - blue background
                bgcolor = 'rgba(0, 0, 200, 0.9)'
                bordercolor = 'darkblue'
                font_size = 22
            elif is_buffer:
                # Buffer zones - gray
                bgcolor = 'rgba(150, 150, 150, 0.7)'
                bordercolor = 'gray'
                font_size = 16
            else:
                # Regular court zones - black
                bgcolor = 'rgba(0, 0, 0, 0.7)'
                bordercolor = 'red'
                font_size = 20
            
            fig.add_annotation(
                x=x_center,
                y=y_center,
                text=f"<b>{zone_label}</b>",
                showarrow=False,
                font=dict(size=font_size, color='white', family='Arial Black'),
                bgcolor=bgcolor,
                bordercolor=bordercolor,
                borderwidth=2,
                borderpad=6
            )
    
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
    
    # Update layout (same as Sequence Analysis)
    fig.update_layout(
        height=700,
        xaxis=dict(range=[x_bins[0] - 0.5, x_bins[-1] + 0.5]),
        yaxis=dict(range=[y_bins[0] - 0.5, y_bins[-1] + 0.5])
    )
    
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
    st.header("🛒 Association Rule Learning")
    
    st.info("""
    **Discover relationships between trajectory patterns using Association Rules:**
    
    *Example: "If a player visits zones A1 and B2, they will likely also visit zone C3."*
    
    Association rules find patterns like **{A, B} → {C}** where:
    - **Antecedent {A, B}**: Starting pattern (zones/features that occur together)
    - **Consequent {C}**: What typically follows
    
    **Two Primary Metrics (Theoretical Focus):**
    - **Support** ⭐: How common is this pattern? (frequency in all trajectories)
    - **Confidence** ⭐: How reliable is this rule? (probability that consequent follows antecedent)
    
    **Goal**: Find patterns that are both **common enough** (support) and **strong enough** (confidence) to be meaningful
    
    **Applications:** Movement pattern discovery, spatial behavior analysis, trajectory prediction
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
        
        **Discovered Rule:** {Milk, Diaper} → {Beer}
        - **Support ⭐**: 3/5 = 0.6 (appears in 3 out of 5 transactions - 60% frequency)
        - **Confidence ⭐**: 3/3 = 1.0 (100% of times when {Milk, Diaper} appear, Beer also appears - perfect reliability!)
        
        **Interpretation**: This is a strong rule with high support (common pattern) AND high confidence (very reliable). Customers buying milk and diapers are very likely to also buy beer!
        """)
    
    st.markdown('---')
    
    if not selected_configs or not selected_objects:
        st.warning("⚠️ Please select at least one configuration and one object from the sidebar.")
        return
    
    filtered_df = data[
        (data['config_source'].isin(selected_configs)) &
        (data['obj'].isin(selected_objects))
    ]
    
    if len(filtered_df) == 0:
        st.error("No data available for selected configurations and objects.")
        return
    
    st.success(f"✅ Loaded {len(filtered_df)} trajectories")
    
    # Configuration Panel
    st.subheader("⚙️ Transaction Configuration")
    
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
            **📍 Spatial Zones (WHERE)**
            
            Find patterns based on **which zones** are visited.
            
            *Example rule:* "If player visits zones B3 and C4, they will likely visit E4"
            
            **Best for:** Spatial movement patterns
            """)
        elif transaction_type == "Feature Bins":
            st.info("""
            **📊 Feature Bins (HOW)**
            
            Find patterns based on **trajectory characteristics** (speed, distance, duration).
            
            *Example rule:* "If trajectory has high speed and long distance, it will have medium duration"
            
            **Best for:** Movement behavior patterns
            """)
        else:  # Combined
            st.info("""
            **🔄 Combined (WHERE + HOW)**
            
            Find patterns combining **both zones AND features**.
            
            *Example rule:* "If player visits B3 with high speed, they will visit E4 with medium duration"
            
            **Best for:** Comprehensive analysis
            """)
    
    with col2:
        if transaction_type in ["Spatial Zones", "Combined (Spatial + Features)"]:
            st.markdown("**🗺️ Spatial Grid Settings:**")
            st.caption("Divide the court into zones")
            grid_rows = st.slider("Grid Rows", 2, 6, 3, key="ar_grid_rows",
                                 help="Number of horizontal divisions")
            grid_cols = st.slider("Grid Columns", 2, 8, 5, key="ar_grid_cols",
                                 help="Number of vertical divisions")
            st.caption(f"💡 Court divided into {grid_rows} × {grid_cols} = {grid_rows * grid_cols} zones")
        
        if transaction_type in ["Feature Bins", "Combined (Spatial + Features)"]:
            st.markdown("**📏 Feature Binning:**")
            st.caption("Categorize trajectory characteristics")
            n_bins = st.select_slider("Number of Bins", [2, 3, 4, 5], value=3, key="ar_n_bins",
                                      help="How many categories (e.g., low/medium/high)")
            st.caption(f"💡 Features grouped into {n_bins} categories")
    
    st.markdown("---")
    
    # Threshold Settings
    st.subheader("📊 Rule Mining Thresholds")
    
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
        st.caption(f"💡 Patterns must appear in ≥{min_support*100:.0f}% of trajectories")
    
    with col2:
        min_confidence = st.slider("**Min Confidence** ⭐", 0.1, 1.0, 0.5, 0.05,
                                   help="Confidence = P(B|A): Probability that consequent occurs given antecedent. "
                                        "Higher confidence = stronger rule", 
                                   key="ar_min_confidence")
        st.caption(f"💡 Rules must be correct ≥{min_confidence*100:.0f}% of the time")
    
    min_lift = 1.0  # Keep default value for compatibility with mining function
    
    st.markdown("---")
    
    # Mine Rules Button
    if st.button("🔍 Mine Association Rules", type="primary"):
        with st.spinner("Preparing transactions and mining rules..."):
            try:
                # Prepare trajectories dictionary
                trajectories_dict = {}
                for (config, obj), group in filtered_df.groupby(['config_source', 'obj']):
                    traj_id = f"{config}_obj{obj}"
                    trajectories_dict[traj_id] = group.sort_values('tst')
                
                # Store trajectories dict in session state for visualization
                st.session_state.ar_trajectories_dict = trajectories_dict
                
                # Prepare transactions based on type
                if transaction_type == "Spatial Zones":
                    grid_info = create_spatial_grid_func('Tennis', grid_rows, grid_cols)
                    st.session_state.ar_grid_info = grid_info  # Store grid info
                    get_zone = grid_info['get_zone']
                    transactions, item_names, trajectory_mapping = prepare_spatial_transactions(trajectories_dict, get_zone)
                    st.session_state.ar_trajectory_mapping = trajectory_mapping
                
                elif transaction_type == "Feature Bins":
                    # Extract features
                    features_list = []
                    trajectory_mapping = []
                    for traj_id, traj_data in trajectories_dict.items():
                        features = {
                            'total_distance': np.sum(np.sqrt(np.diff(traj_data['x'])**2 + np.diff(traj_data['y'])**2)),
                            'duration': traj_data['tst'].max() - traj_data['tst'].min(),
                            'avg_speed': np.mean(np.sqrt(np.diff(traj_data['x'])**2 + np.diff(traj_data['y'])**2) / np.diff(traj_data['tst'])) if len(traj_data) > 1 else 0,
                            'x_range': traj_data['x'].max() - traj_data['x'].min(),
                            'y_range': traj_data['y'].max() - traj_data['y'].min(),
                        }
                        features_list.append(features)
                        trajectory_mapping.append(traj_id)
                    features_df = pd.DataFrame(features_list)
                    transactions, item_names = prepare_feature_transactions(features_df, n_bins)
                    st.session_state.ar_trajectory_mapping = trajectory_mapping
                
                else:  # Combined
                    grid_info = create_spatial_grid_func('Tennis', grid_rows, grid_cols)
                    st.session_state.ar_grid_info = grid_info  # Store grid info
                    get_zone = grid_info['get_zone']
                    spatial_trans, spatial_items, trajectory_mapping = prepare_spatial_transactions(trajectories_dict, get_zone)
                    st.session_state.ar_trajectory_mapping = trajectory_mapping
                    
                    features_list = []
                    for traj_id, traj_data in trajectories_dict.items():
                        features = {
                            'total_distance': np.sum(np.sqrt(np.diff(traj_data['x'])**2 + np.diff(traj_data['y'])**2)),
                            'duration': traj_data['tst'].max() - traj_data['tst'].min(),
                            'avg_speed': np.mean(np.sqrt(np.diff(traj_data['x'])**2 + np.diff(traj_data['y'])**2) / np.diff(traj_data['tst'])) if len(traj_data) > 1 else 0,
                            'x_range': traj_data['x'].max() - traj_data['x'].min(),
                            'y_range': traj_data['y'].max() - traj_data['y'].min(),
                        }
                        features_list.append(features)
                    features_df = pd.DataFrame(features_list)
                    feature_trans, feature_items = prepare_feature_transactions(features_df, n_bins)
                    transactions, item_names = prepare_combined_transactions(
                        spatial_trans, spatial_items, feature_trans, feature_items
                    )
                
                st.session_state.ar_transactions = transactions
                st.session_state.ar_item_names = item_names
                
                # Show transaction preview
                st.subheader("📝 Transaction Preview")
                st.write(f"Total transactions: {len(transactions)}")
                st.write(f"Unique items: {len(item_names)}")
                
                trans_df = pd.DataFrame({
                    'Transaction': [f"T{i+1}" for i in range(min(10, len(transactions)))],
                    'Items': [', '.join(trans) for trans in transactions[:10]]
                })
                st.dataframe(trans_df, use_container_width=True)
                
                # Compute association rules
                st.subheader("⚙️ Mining Association Rules...")
                rules_df, frequent_itemsets = compute_association_rules(
                    transactions, min_support, min_confidence, min_lift
                )
                
                st.session_state.ar_rules = rules_df
                st.session_state.ar_frequent_itemsets = frequent_itemsets
                
                # Compute matrices
                cooccurrence_matrix = create_cooccurrence_matrix(transactions, item_names)
                distance_matrix = create_distance_matrix(cooccurrence_matrix)
                
                st.session_state.ar_cooccurrence_matrix = cooccurrence_matrix
                st.session_state.ar_distance_matrix = distance_matrix
                
                if len(rules_df) == 0:
                    st.warning("⚠️ No rules found. Try lowering the thresholds.")
                else:
                    st.success(f"✅ Found {len(rules_df)} association rules!")
            
            except Exception as e:
                st.error(f"Error during rule mining: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Display results if available
    if st.session_state.get('ar_rules') is not None and len(st.session_state.ar_rules) > 0:
        st.markdown("---")
        st.header("📊 Association Rules Results")
        
        rules_df = st.session_state.ar_rules
        
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📋 Rules Table", "🕸️ Network", 
            "🔥 Co-occurrence Heatmap", "🗺️ MDS Projection", 
            "📊 Distance Matrix", "🏆 Top Rules", "🎯 Supporting Trajectories"
        ])
        
        with tab1:
            st.subheader("📋 Rules Table")
            st.markdown("""
            **All discovered association rules in a sortable table.**
            
            Each row shows a rule (Antecedents → Consequents) with its Support (how common) and Confidence (how reliable).
            Sort by either metric to find the most interesting patterns.
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                sort_by = st.selectbox("Sort by:", ["confidence", "support"], 
                                      help="Default sorted by Confidence (rule strength)")
            with col2:
                top_k_table = st.number_input("Show top N rules:", min_value=1, max_value=len(rules_df), value=min(50, len(rules_df)))
            
            display_rules = rules_df.nlargest(top_k_table, sort_by)
            
            # Only show Support and Confidence (primary metrics)
            display_df = display_rules[['antecedents_str', 'consequents_str', 'support', 'confidence']]
            display_df.columns = ['Antecedents →', 'Consequents', 'Support ⭐', 'Confidence ⭐']
            
            st.dataframe(
                display_df.style.format({
                    'Support ⭐': '{:.3f}', 'Confidence ⭐': '{:.3f}'
                }).background_gradient(subset=['Support ⭐', 'Confidence ⭐'], cmap='YlOrRd'),
                use_container_width=True, height=400
            )
            
            st.caption("⭐ = Primary metrics based on theoretical framework: Support (frequency) and Confidence (strength)")
            
            csv = display_df.to_csv(index=False)
            st.download_button("📥 Download Rules as CSV", csv, "association_rules.csv", "text/csv")
        
        with tab2:
            st.subheader("🕸️ Network Graph")
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
            st.subheader("🔥 Co-occurrence Heatmap")
            st.markdown("""
            **Shows how often pairs of items appear together in transactions.**
            
            - **Darker colors** = Items frequently co-occur (appear together in many trajectories)
            - **Lighter colors** = Items rarely appear together
            
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
            st.subheader("🗺️ MDS Projection")
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
            st.subheader("📊 Distance Matrix")
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
            st.subheader("🏆 Top Rules")
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
            st.subheader("🎯 Supporting Trajectories")
            st.markdown("""
            **See the actual trajectories that create each association rule.**
            
            - Select a rule to see which specific trajectory IDs satisfy it
            - View the trajectories visualized on a tennis court with grid overlay
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
                    rule_str = f"{antecedents_str} → {consequents_str} (Supp: {row['support']:.2f}, Conf: {row['confidence']:.2f})"
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
                    st.metric("Percentage", f"{percentage:.1f}%")
                
                # Display trajectory IDs
                with st.expander("📋 Show Trajectory IDs", expanded=False):
                    if supporting_traj_ids:
                        st.write(f"Trajectory IDs: {', '.join(map(str, supporting_traj_ids))}")
                    else:
                        st.info("No trajectories found supporting this rule.")
                
                # Plot supporting trajectories on court
                if supporting_traj_ids:
                    st.subheader("Court Visualization")
                    
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
                            **Visualization Details:**
                            - 🔴 **Red zones** = Antecedent ({', '.join(list(antecedents))})
                            - 🔵 **Blue zones** = Consequent ({', '.join(list(consequents))})
                            - Each trajectory is shown as a colored path
                            - 🟢 Green circles indicate start points
                            - 🔴 Red squares indicate end points
                            - Showing up to 20 trajectories for clarity
                            - Hover over points to see trajectory IDs and coordinates
                            - All {num_supporting} trajectories contain both antecedent and consequent zones
                            """)
                        else:
                            st.warning("Could not generate trajectory plot.")
                    else:
                        st.warning("Grid visualization is only available for 'Spatial Zones' and 'Combined' transaction types.")
                else:
                    st.info("No trajectories to visualize for this rule.")
        
        with st.expander("📖 How to Interpret Results", expanded=False):
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

