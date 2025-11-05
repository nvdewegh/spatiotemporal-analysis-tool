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
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import networkx as nx
from sklearn.manifold import MDS
from itertools import combinations
from .common import render_interactive_chart


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
    """
    transactions = []
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
    
    all_items = set()
    for transaction in transactions:
        all_items.update(transaction)
    item_names = sorted(list(all_items))
    
    return transactions, item_names


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
        rules = rules[rules['lift'] >= min_lift]
        rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(sorted(list(x))))
        rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(sorted(list(x))))
        rules = rules.sort_values('lift', ascending=False).reset_index(drop=True)
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
                G.add_edge(ant, cons, weight=rule['lift'], 
                          support=rule['support'], confidence=rule['confidence'])
    
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    edge_trace = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_data = G.edges[edge]
        edge_trace.append(
            go.Scatter(x=[x0, x1, None], y=[y0, y1, None], mode='lines',
                      line=dict(width=edge_data['weight'], color='rgba(125,125,125,0.5)'),
                      hoverinfo='text',
                      text=f"Support: {edge_data['support']:.3f}<br>Confidence: {edge_data['confidence']:.3f}<br>Lift: {edge_data['weight']:.3f}",
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
    **Discover interesting relationships between trajectory attributes using Association Rules:**
    
    *"If a customer buys diapers and milk, then he is very likely to buy beer."*
    
    Association rule discovery finds patterns like **{A, B} → {C}** where:
    - **Antecedent {A, B}**: Items that occur together
    - **Consequent {C}**: Item that often follows
    
    **Key Metrics:**
    - **Support**: How frequently the itemset appears (relative frequency)
    - **Confidence**: Probability that consequent occurs given antecedent
    - **Lift**: How much more likely consequent is when antecedent is present
    
    **Applications:** Market basket analysis, pattern discovery, recommendation systems
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
        - **Support**: 3/5 = 0.6 (appears in 3 out of 5 transactions)
        - **Confidence**: 3/3 = 1.0 (100% of times when {Milk, Diaper} appear, Beer also appears)
        - **Lift**: 1.67 (Beer is 67% more likely when {Milk, Diaper} are purchased)
        
        **Interpretation**: Customers buying milk and diapers are very likely to also buy beer!
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
    
    col1, col2 = st.columns(2)
    
    with col1:
        transaction_type = st.radio(
            "Transaction Type:",
            ["Spatial Zones", "Feature Bins", "Combined (Spatial + Features)"],
            help="Choose what items to use in transactions"
        )
    
    with col2:
        if transaction_type in ["Spatial Zones", "Combined (Spatial + Features)"]:
            st.markdown("**Spatial Grid Settings:**")
            grid_rows = st.slider("Grid Rows", 2, 6, 3, key="ar_grid_rows")
            grid_cols = st.slider("Grid Columns", 2, 8, 5, key="ar_grid_cols")
        
        if transaction_type in ["Feature Bins", "Combined (Spatial + Features)"]:
            st.markdown("**Feature Binning:**")
            n_bins = st.select_slider("Number of Bins", [2, 3, 4, 5], value=3, key="ar_n_bins")
    
    st.markdown("---")
    
    # Threshold Settings
    st.subheader("📊 Rule Mining Thresholds")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_support = st.slider("Min Support", 0.05, 0.5, 0.1, 0.05,
                               help="Minimum frequency of itemset", key="ar_min_support")
    
    with col2:
        min_confidence = st.slider("Min Confidence", 0.1, 1.0, 0.5, 0.05,
                                   help="Minimum probability that consequent occurs given antecedent", 
                                   key="ar_min_confidence")
    
    with col3:
        min_lift = st.slider("Min Lift", 1.0, 3.0, 1.0, 0.1,
                            help="Minimum lift ratio (>1 means positive correlation)", 
                            key="ar_min_lift")
    
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
                
                # Prepare transactions based on type
                if transaction_type == "Spatial Zones":
                    grid_info = create_spatial_grid_func('Tennis', grid_rows, grid_cols)
                    get_zone = grid_info['get_zone']
                    transactions, item_names = prepare_spatial_transactions(trajectories_dict, get_zone)
                
                elif transaction_type == "Feature Bins":
                    # Extract features
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
                    transactions, item_names = prepare_feature_transactions(features_df, n_bins)
                
                else:  # Combined
                    grid_info = create_spatial_grid_func('Tennis', grid_rows, grid_cols)
                    get_zone = grid_info['get_zone']
                    spatial_trans, spatial_items = prepare_spatial_transactions(trajectories_dict, get_zone)
                    
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
            "📋 Rules Table", "🕸️ Network", "📈 Support-Confidence", 
            "🔥 Co-occurrence Heatmap", "🗺️ MDS Projection", 
            "📊 Distance Matrix", "🏆 Top Rules"
        ])
        
        with tab1:
            st.subheader("Association Rules Table")
            
            col1, col2 = st.columns(2)
            with col1:
                sort_by = st.selectbox("Sort by:", ["lift", "support", "confidence", "leverage"])
            with col2:
                top_k_table = st.number_input("Show top N rules:", min_value=1, max_value=len(rules_df), value=min(50, len(rules_df)))
            
            display_rules = rules_df.nlargest(top_k_table, sort_by)
            display_df = display_rules[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift', 'leverage', 'conviction']]
            display_df.columns = ['Antecedents', 'Consequents', 'Support', 'Confidence', 'Lift', 'Leverage', 'Conviction']
            
            st.dataframe(
                display_df.style.format({
                    'Support': '{:.3f}', 'Confidence': '{:.3f}', 'Lift': '{:.3f}',
                    'Leverage': '{:.3f}', 'Conviction': '{:.3f}'
                }),
                use_container_width=True, height=400
            )
            
            csv = display_df.to_csv(index=False)
            st.download_button("📥 Download Rules as CSV", csv, "association_rules.csv", "text/csv")
        
        with tab2:
            st.subheader("Association Rules Network Graph")
            top_k_network = st.slider("Number of top rules:", 5, 50, 20, key="top_k_network")
            fig = plot_rules_network(rules_df, top_k=top_k_network)
            render_interactive_chart(st, fig)
        
        with tab3:
            st.subheader("Support vs Confidence Scatter Plot")
            color_metric = st.radio("Color by:", ["lift", "leverage", "conviction"], horizontal=True)
            fig = plot_support_confidence_scatter(rules_df, color_by=color_metric)
            render_interactive_chart(st, fig)
        
        with tab4:
            st.subheader("Item Co-occurrence Heatmap")
            if st.session_state.get('ar_cooccurrence_matrix') is not None:
                fig = plot_cooccurrence_heatmap(
                    st.session_state.ar_cooccurrence_matrix,
                    st.session_state.ar_item_names
                )
                render_interactive_chart(st, fig)
            else:
                st.info("Co-occurrence matrix not available.")
        
        with tab5:
            st.subheader("MDS Projection of Items")
            if st.session_state.get('ar_distance_matrix') is not None:
                mds_dims = st.radio("Dimensions:", [2, 3], horizontal=True)
                fig = plot_mds_projection(
                    st.session_state.ar_distance_matrix,
                    st.session_state.ar_item_names,
                    n_components=mds_dims
                )
                render_interactive_chart(st, fig)
            else:
                st.info("Distance matrix not available.")
        
        with tab6:
            st.subheader("Distance Matrix Heatmap")
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
        
        with tab7:
            st.subheader("Top Rules Analysis")
            col1, col2 = st.columns(2)
            with col1:
                rank_metric = st.selectbox("Rank by:", ["lift", "support", "confidence"])
            with col2:
                top_k_bars = st.slider("Number of rules:", 5, 30, 15, key="top_k_bars")
            
            fig = plot_top_rules_bars(rules_df, metric=rank_metric, top_k=top_k_bars)
            render_interactive_chart(st, fig)
        
        with st.expander("📖 How to Interpret Results", expanded=False):
            st.markdown("""
            **Understanding Association Rule Metrics:**
            
            1. **Support** = P(A ∪ B)
               - Frequency of the itemset in all transactions
               - High support = common pattern
            
            2. **Confidence** = P(B|A) = P(A ∪ B) / P(A)
               - Probability of consequent given antecedent
               - Confidence of 0.8 means 80% of times A appears, B also appears
            
            3. **Lift** = P(B|A) / P(B)
               - Ratio of observed to expected support
               - Lift > 1: positive correlation (items occur together more than expected)
               - Lift = 1: independent (no relationship)
               - Lift < 1: negative correlation (items rarely occur together)
            
            4. **Leverage** = P(A ∪ B) - P(A) × P(B)
               - Difference between observed and expected co-occurrence
               - Positive leverage indicates positive association
            
            5. **Conviction** = (1 - P(B)) / (1 - Confidence)
               - How much more often A occurs without B than expected
               - Higher conviction = stronger implication
            
            **Important Notes:**
            - Association rules show correlation, NOT causation
            - Lower thresholds yield more rules but may include spurious patterns
            - Focus on rules with high lift and sufficient support
            """)
