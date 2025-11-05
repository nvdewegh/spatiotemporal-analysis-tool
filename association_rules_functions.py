# ============================================================================
# ASSOCIATION RULES INFRASTRUCTURE FUNCTIONS
# ============================================================================

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import networkx as nx
from sklearn.manifold import MDS
from itertools import combinations


def prepare_spatial_transactions(trajectories_dict, get_zone_func, grid_rows=3, grid_cols=5):
    """
    Convert trajectories into transactions based on spatial zones visited.
    Each trajectory becomes a transaction of zone labels.
    
    Args:
        trajectories_dict: Dictionary of trajectories
        get_zone_func: Function to get zone from coordinates
        grid_rows: Number of rows in spatial grid
        grid_cols: Number of columns in spatial grid
        
    Returns:
        transactions: List of lists (transactions with items)
        item_names: List of unique items
    """
    transactions = []
    for traj_id, traj_data in trajectories_dict.items():
        # Extract x, y coordinates
        x_coords = traj_data['x'].values
        y_coords = traj_data['y'].values
        
        # Get zones for each point
        zones = []
        for x, y in zip(x_coords, y_coords):
            zone = get_zone_func(x, y)
            if zone:  # Only add valid zones
                zones.append(zone)
        
        # Create transaction from unique zones (order doesn't matter for association rules)
        if zones:
            transaction = list(set(zones))  # Unique zones visited
            transactions.append(transaction)
    
    # Get all unique items
    all_items = set()
    for transaction in transactions:
        all_items.update(transaction)
    item_names = sorted(list(all_items))
    
    return transactions, item_names


def prepare_feature_transactions(features_df, n_bins=3):
    """
    Convert feature data into transactions based on binned feature values.
    Each trajectory becomes a transaction of feature range labels.
    
    Args:
        features_df: DataFrame with trajectory features
        n_bins: Number of bins for each feature
        
    Returns:
        transactions: List of lists (transactions with items)
        item_names: List of unique items
    """
    transactions = []
    
    # Select numeric features for binning
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude identifier columns
    exclude_cols = ['config', 'object', 'rally_id']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    for idx, row in features_df.iterrows():
        transaction = []
        
        for col in feature_cols:
            value = row[col]
            if pd.notna(value):
                # Bin the value
                col_min = features_df[col].min()
                col_max = features_df[col].max()
                
                if col_max > col_min:
                    bin_width = (col_max - col_min) / n_bins
                    bin_idx = min(int((value - col_min) / bin_width), n_bins - 1)
                    
                    # Create item label: feature_bin (e.g., "speed_high", "distance_low")
                    bin_labels = ['low', 'medium', 'high'] if n_bins == 3 else [f'bin{i+1}' for i in range(n_bins)]
                    item = f"{col}_{bin_labels[bin_idx]}"
                    transaction.append(item)
        
        if transaction:
            transactions.append(transaction)
    
    # Get all unique items
    all_items = set()
    for transaction in transactions:
        all_items.update(transaction)
    item_names = sorted(list(all_items))
    
    return transactions, item_names


def prepare_combined_transactions(spatial_trans, spatial_items, feature_trans, feature_items):
    """
    Combine spatial and feature transactions.
    
    Returns:
        transactions: List of lists combining spatial zones and feature bins
        item_names: List of unique items
    """
    # Combine transactions (must have same length)
    min_len = min(len(spatial_trans), len(feature_trans))
    transactions = []
    
    for i in range(min_len):
        # Prefix spatial items with "zone_" and feature items are already prefixed
        spatial_prefixed = [f"zone_{item}" for item in spatial_trans[i]]
        combined = spatial_prefixed + feature_trans[i]
        transactions.append(combined)
    
    # Combine item names
    spatial_items_prefixed = [f"zone_{item}" for item in spatial_items]
    item_names = sorted(list(set(spatial_items_prefixed + feature_items)))
    
    return transactions, item_names


def compute_association_rules_from_transactions(transactions, min_support=0.1, 
                                                 min_confidence=0.5, min_lift=1.0):
    """
    Compute association rules from transactions using Apriori algorithm.
    
    Args:
        transactions: List of lists (each inner list is a transaction)
        min_support: Minimum support threshold
        min_confidence: Minimum confidence threshold
        min_lift: Minimum lift threshold
        
    Returns:
        rules_df: DataFrame with association rules and metrics
        frequent_itemsets_df: DataFrame with frequent itemsets
    """
    if not transactions or len(transactions) == 0:
        return pd.DataFrame(), pd.DataFrame()
    
    # Encode transactions to one-hot encoding
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Apply Apriori algorithm
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    
    if len(frequent_itemsets) == 0:
        return pd.DataFrame(), pd.DataFrame()
    
    # Generate association rules
    try:
        rules = association_rules(frequent_itemsets, metric="confidence", 
                                   min_threshold=min_confidence)
        
        # Filter by lift
        rules = rules[rules['lift'] >= min_lift]
        
        # Convert frozensets to readable strings
        rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(sorted(list(x))))
        rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(sorted(list(x))))
        
        # Sort by lift (most interesting rules first)
        rules = rules.sort_values('lift', ascending=False).reset_index(drop=True)
        
        return rules, frequent_itemsets
    
    except ValueError:
        # No rules found
        return pd.DataFrame(), frequent_itemsets


def create_item_cooccurrence_matrix(transactions, item_names):
    """
    Create a co-occurrence matrix showing how often items appear together.
    
    Args:
        transactions: List of transaction lists
        item_names: List of all unique items
        
    Returns:
        cooccurrence_matrix: 2D numpy array
    """
    n_items = len(item_names)
    item_to_idx = {item: idx for idx, item in enumerate(item_names)}
    
    # Initialize matrix
    cooccurrence = np.zeros((n_items, n_items))
    
    # Count co-occurrences
    for transaction in transactions:
        for item1, item2 in combinations(transaction, 2):
            if item1 in item_to_idx and item2 in item_to_idx:
                idx1 = item_to_idx[item1]
                idx2 = item_to_idx[item2]
                cooccurrence[idx1, idx2] += 1
                cooccurrence[idx2, idx1] += 1  # Symmetric
    
    return cooccurrence


def create_item_distance_matrix(cooccurrence_matrix):
    """
    Convert co-occurrence matrix to distance matrix for MDS visualization.
    Higher co-occurrence = smaller distance.
    
    Args:
        cooccurrence_matrix: 2D numpy array of co-occurrences
        
    Returns:
        distance_matrix: 2D numpy array of distances
    """
    # Avoid division by zero
    max_cooccurrence = np.max(cooccurrence_matrix)
    if max_cooccurrence == 0:
        return np.ones_like(cooccurrence_matrix)
    
    # Convert to distance: distance = 1 / (1 + normalized_cooccurrence)
    normalized = cooccurrence_matrix / max_cooccurrence
    distance_matrix = 1.0 / (1.0 + normalized)
    
    # Set diagonal to 0
    np.fill_diagonal(distance_matrix, 0)
    
    return distance_matrix


def plot_association_rules_network(rules_df, top_k=20):
    """
    Create a network graph visualization of association rules.
    
    Args:
        rules_df: DataFrame with association rules
        top_k: Number of top rules to display
        
    Returns:
        fig: Plotly figure object
    """
    if len(rules_df) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No rules to display",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Take top K rules
    top_rules = rules_df.head(top_k)
    
    # Create network graph
    G = nx.DiGraph()
    
    # Add nodes and edges
    for _, rule in top_rules.iterrows():
        antecedents = list(rule['antecedents'])
        consequents = list(rule['consequents'])
        
        # Add nodes
        for item in antecedents + consequents:
            G.add_node(item)
        
        # Add edges with weight = lift
        for ant in antecedents:
            for cons in consequents:
                G.add_edge(ant, cons, weight=rule['lift'], 
                          support=rule['support'], confidence=rule['confidence'])
    
    # Calculate layout
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Create edge traces
    edge_trace = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_data = G.edges[edge]
        
        # Create arrow
        edge_trace.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=edge_data['weight'], color='rgba(125,125,125,0.5)'),
                hoverinfo='text',
                text=f"Support: {edge_data['support']:.3f}<br>Confidence: {edge_data['confidence']:.3f}<br>Lift: {edge_data['weight']:.3f}",
                showlegend=False
            )
        )
    
    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        marker=dict(
            size=20,
            color='lightblue',
            line=dict(width=2, color='darkblue')
        ),
        hoverinfo='text',
        hovertext=node_text
    )
    
    # Create figure
    fig = go.Figure(data=edge_trace + [node_trace])
    fig.update_layout(
        title=f"Association Rules Network (Top {len(top_rules)} Rules)",
        showlegend=False,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600
    )
    
    return fig


def plot_support_confidence_scatter(rules_df, color_by='lift'):
    """
    Create interactive scatter plot of support vs confidence colored by lift.
    
    Args:
        rules_df: DataFrame with association rules
        color_by: Metric to use for coloring ('lift', 'leverage', 'conviction')
        
    Returns:
        fig: Plotly figure object
    """
    if len(rules_df) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No rules to display",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Create hover text
    hover_text = []
    for _, row in rules_df.iterrows():
        text = (f"Rule: {row['antecedents_str']} → {row['consequents_str']}<br>"
                f"Support: {row['support']:.3f}<br>"
                f"Confidence: {row['confidence']:.3f}<br>"
                f"Lift: {row['lift']:.3f}")
        hover_text.append(text)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=rules_df['support'],
        y=rules_df['confidence'],
        mode='markers',
        marker=dict(
            size=10,
            color=rules_df[color_by],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title=color_by.capitalize()),
            line=dict(width=1, color='white')
        ),
        text=hover_text,
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title="Association Rules: Support vs Confidence",
        xaxis_title="Support",
        yaxis_title="Confidence",
        hovermode='closest',
        height=600
    )
    
    return fig


def plot_item_cooccurrence_heatmap(cooccurrence_matrix, item_names):
    """
    Create heatmap of item co-occurrences.
    
    Args:
        cooccurrence_matrix: 2D numpy array
        item_names: List of item names
        
    Returns:
        fig: Plotly figure object
    """
    fig = go.Figure(data=go.Heatmap(
        z=cooccurrence_matrix,
        x=item_names,
        y=item_names,
        colorscale='Blues',
        hovertemplate='%{y} & %{x}<br>Co-occurrences: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Item Co-occurrence Matrix",
        xaxis_title="Items",
        yaxis_title="Items",
        height=600,
        xaxis={'side': 'bottom'},
    )
    
    return fig


def plot_items_mds(distance_matrix, item_names, n_components=2):
    """
    Create MDS projection of items based on co-occurrence distances.
    
    Args:
        distance_matrix: 2D numpy array of distances
        item_names: List of item names
        n_components: Number of MDS dimensions (2 or 3)
        
    Returns:
        fig: Plotly figure object
    """
    if len(item_names) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Need at least 2 items for MDS",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Perform MDS
    mds = MDS(n_components=n_components, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(distance_matrix)
    
    if n_components == 2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            mode='markers+text',
            text=item_names,
            textposition='top center',
            marker=dict(size=12, color='lightcoral', line=dict(width=1, color='darkred')),
            hovertext=item_names,
            hoverinfo='text'
        ))
        
        fig.update_layout(
            title="MDS Projection of Items (Based on Co-occurrence)",
            xaxis_title="MDS Dimension 1",
            yaxis_title="MDS Dimension 2",
            height=600,
            hovermode='closest'
        )
    else:  # 3D
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode='markers+text',
            text=item_names,
            textposition='top center',
            marker=dict(size=8, color='lightcoral', line=dict(width=1, color='darkred')),
            hovertext=item_names,
            hoverinfo='text'
        ))
        
        fig.update_layout(
            title="3D MDS Projection of Items (Based on Co-occurrence)",
            scene=dict(
                xaxis_title="MDS Dimension 1",
                yaxis_title="MDS Dimension 2",
                zaxis_title="MDS Dimension 3"
            ),
            height=700,
            hovermode='closest'
        )
    
    return fig


def plot_top_rules_bars(rules_df, metric='lift', top_k=15):
    """
    Create bar chart of top rules by a specific metric.
    
    Args:
        rules_df: DataFrame with association rules
        metric: Metric to rank by ('lift', 'support', 'confidence')
        top_k: Number of top rules to display
        
    Returns:
        fig: Plotly figure object
    """
    if len(rules_df) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No rules to display",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Sort and take top K
    top_rules = rules_df.nlargest(min(top_k, len(rules_df)), metric)
    
    # Create rule labels
    rule_labels = [f"{row['antecedents_str']} → {row['consequents_str']}" 
                   for _, row in top_rules.iterrows()]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=rule_labels,
        x=top_rules[metric],
        orientation='h',
        marker=dict(color=top_rules[metric], colorscale='Viridis'),
        text=top_rules[metric].round(3),
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>' + f'{metric.capitalize()}: %{{x:.3f}}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Top {len(top_rules)} Rules by {metric.capitalize()}",
        xaxis_title=metric.capitalize(),
        yaxis_title="Rules",
        height=max(400, len(top_rules) * 30),
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig
