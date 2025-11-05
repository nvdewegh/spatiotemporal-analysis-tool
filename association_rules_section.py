# INSERT THIS SECTION BEFORE "elif analysis_method == 'Sequence Analysis':" (around line 2837)

    elif analysis_method == "Association Rules":
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
        
        # Show market basket example
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
            - **Confidence**: 3/3 = 1.0 (100% of time when {Milk, Diaper} appear together, Beer also appears)
            - **Lift**: 1.67 (Beer is 67% more likely when {Milk, Diaper} are purchased)
            
            **Interpretation**: Customers buying milk and diapers are very likely to also buy beer!
            """)
        
        # Import helper functions
        from association_rules_functions import (
            prepare_spatial_transactions, prepare_feature_transactions, prepare_combined_transactions,
            compute_association_rules_from_transactions, create_item_cooccurrence_matrix,
            create_item_distance_matrix, plot_association_rules_network,
            plot_support_confidence_scatter, plot_item_cooccurrence_heatmap,
            plot_items_mds, plot_top_rules_bars
        )
        
        st.markdown('---')
        
        # Use selections from sidebar
        selected_configs = st.session_state.shared_selected_configs
        selected_objects = st.session_state.shared_selected_objects
        
        if not selected_configs or not selected_objects:
            st.warning("⚠️ Please select at least one configuration and one object from the sidebar.")
        else:
            # Filter data
            filtered_df = st.session_state.data[
                (st.session_state.data['config'].isin(selected_configs)) &
                (st.session_state.data['object'].isin(selected_objects))
            ]
            
            if len(filtered_df) == 0:
                st.error("No data available for selected configurations and objects.")
            else:
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
                    min_support = st.slider(
                        "Min Support",
                        0.05, 0.5, 0.1, 0.05,
                        help="Minimum frequency of itemset in transactions",
                        key="ar_min_support"
                    )
                
                with col2:
                    min_confidence = st.slider(
                        "Min Confidence",
                        0.1, 1.0, 0.5, 0.05,
                        help="Minimum probability that consequent occurs given antecedent",
                        key="ar_min_confidence"
                    )
                
                with col3:
                    min_lift = st.slider(
                        "Min Lift",
                        1.0, 3.0, 1.0, 0.1,
                        help="Minimum lift ratio (>1 means positive correlation)",
                        key="ar_min_lift"
                    )
                
                st.markdown("---")
                
                # Mine Rules Button
                if st.button("🔍 Mine Association Rules", type="primary"):
                    with st.spinner("Preparing transactions and mining rules..."):
                        try:
                            # Prepare trajectories dictionary
                            trajectories_dict = {}
                            for (config, obj, rally_id), group in filtered_df.groupby(['config', 'object', 'rally_id']):
                                traj_id = f"{config}_{obj}_{rally_id}"
                                trajectories_dict[traj_id] = group.sort_values('t')
                            
                            # Get feature data if needed
                            if transaction_type in ["Feature Bins", "Combined (Spatial + Features)"]:
                                # Extract features from trajectories
                                features_list = []
                                for traj_id, traj_data in trajectories_dict.items():
                                    features = {
                                        'total_distance': np.sum(np.sqrt(np.diff(traj_data['x'])**2 + np.diff(traj_data['y'])**2)),
                                        'duration': traj_data['t'].max() - traj_data['t'].min(),
                                        'avg_speed': np.mean(np.sqrt(np.diff(traj_data['x'])**2 + np.diff(traj_data['y'])**2) / np.diff(traj_data['t'])) if len(traj_data) > 1 else 0,
                                        'x_range': traj_data['x'].max() - traj_data['x'].min(),
                                        'y_range': traj_data['y'].max() - traj_data['y'].min(),
                                    }
                                    features_list.append(features)
                                features_df = pd.DataFrame(features_list)
                            
                            # Prepare transactions based on type
                            if transaction_type == "Spatial Zones":
                                get_zone = create_spatial_grid('Tennis', grid_rows, grid_cols)
                                transactions, item_names = prepare_spatial_transactions(
                                    trajectories_dict, get_zone, grid_rows, grid_cols
                                )
                            elif transaction_type == "Feature Bins":
                                transactions, item_names = prepare_feature_transactions(features_df, n_bins)
                            else:  # Combined
                                get_zone = create_spatial_grid('Tennis', grid_rows, grid_cols)
                                spatial_trans, spatial_items = prepare_spatial_transactions(
                                    trajectories_dict, get_zone, grid_rows, grid_cols
                                )
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
                            
                            # Display first few transactions
                            trans_df = pd.DataFrame({
                                'Transaction': [f"T{i+1}" for i in range(min(10, len(transactions)))],
                                'Items': [', '.join(trans) for trans in transactions[:10]]
                            })
                            st.dataframe(trans_df, use_container_width=True)
                            
                            # Compute association rules
                            st.subheader("⚙️ Mining Association Rules...")
                            rules_df, frequent_itemsets = compute_association_rules_from_transactions(
                                transactions, min_support, min_confidence, min_lift
                            )
                            
                            st.session_state.ar_rules = rules_df
                            st.session_state.ar_frequent_itemsets = frequent_itemsets
                            
                            # Compute co-occurrence and distance matrices
                            cooccurrence_matrix = create_item_cooccurrence_matrix(transactions, item_names)
                            distance_matrix = create_item_distance_matrix(cooccurrence_matrix)
                            
                            st.session_state.ar_cooccurrence_matrix = cooccurrence_matrix
                            st.session_state.ar_distance_matrix = distance_matrix
                            
                            if len(rules_df) == 0:
                                st.warning(f"⚠️ No rules found with current thresholds. Try lowering the thresholds.")
                            else:
                                st.success(f"✅ Found {len(rules_df)} association rules!")
                        
                        except Exception as e:
                            st.error(f"Error during rule mining: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                
                # Display results if available
                if st.session_state.ar_rules is not None and len(st.session_state.ar_rules) > 0:
                    st.markdown("---")
                    st.header("📊 Association Rules Results")
                    
                    rules_df = st.session_state.ar_rules
                    
                    # Tabs for different visualizations
                    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                        "📋 Rules Table", "🕸️ Network", "📈 Support-Confidence", 
                        "🔥 Co-occurrence Heatmap", "🗺️ MDS Projection", 
                        "📊 Distance Matrix", "🏆 Top Rules"
                    ])
                    
                    with tab1:
                        st.subheader("Association Rules Table")
                        
                        # Filter and sort options
                        col1, col2 = st.columns(2)
                        with col1:
                            sort_by = st.selectbox("Sort by:", ["lift", "support", "confidence", "leverage"])
                        with col2:
                            top_k_table = st.number_input("Show top N rules:", 10, len(rules_df), min(50, len(rules_df)))
                        
                        # Display table
                        display_rules = rules_df.nlargest(top_k_table, sort_by)
                        display_df = display_rules[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift', 'leverage', 'conviction']]
                        display_df.columns = ['Antecedents', 'Consequents', 'Support', 'Confidence', 'Lift', 'Leverage', 'Conviction']
                        
                        st.dataframe(
                            display_df.style.format({
                                'Support': '{:.3f}',
                                'Confidence': '{:.3f}',
                                'Lift': '{:.3f}',
                                'Leverage': '{:.3f}',
                                'Conviction': '{:.3f}'
                            }),
                            use_container_width=True,
                            height=400
                        )
                        
                        # Download button
                        csv = display_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Rules as CSV",
                            csv,
                            "association_rules.csv",
                            "text/csv"
                        )
                    
                    with tab2:
                        st.subheader("Association Rules Network Graph")
                        top_k_network = st.slider("Number of top rules to display:", 5, 50, 20, key="top_k_network")
                        fig = plot_association_rules_network(rules_df, top_k=top_k_network)
                        render_interactive_chart(fig)
                        st.caption("Network showing items as nodes and rules as directed edges. Edge thickness represents lift.")
                    
                    with tab3:
                        st.subheader("Support vs Confidence Scatter Plot")
                        color_metric = st.radio("Color by:", ["lift", "leverage", "conviction"], horizontal=True, key="color_metric")
                        fig = plot_support_confidence_scatter(rules_df, color_by=color_metric)
                        render_interactive_chart(fig)
                        st.caption("Each point is a rule. Hover for details.")
                    
                    with tab4:
                        st.subheader("Item Co-occurrence Heatmap")
                        if st.session_state.ar_cooccurrence_matrix is not None:
                            fig = plot_item_cooccurrence_heatmap(
                                st.session_state.ar_cooccurrence_matrix,
                                st.session_state.ar_item_names
                            )
                            render_interactive_chart(fig)
                            st.caption("Shows how often pairs of items appear together in transactions.")
                        else:
                            st.info("Co-occurrence matrix not available.")
                    
                    with tab5:
                        st.subheader("MDS Projection of Items")
                        if st.session_state.ar_distance_matrix is not None:
                            mds_dims = st.radio("Dimensions:", [2, 3], horizontal=True, key="mds_dims")
                            fig = plot_items_mds(
                                st.session_state.ar_distance_matrix,
                                st.session_state.ar_item_names,
                                n_components=mds_dims
                            )
                            render_interactive_chart(fig)
                            st.caption("Items that co-occur frequently are positioned closer together.")
                        else:
                            st.info("Distance matrix not available.")
                    
                    with tab6:
                        st.subheader("Distance Matrix Heatmap")
                        if st.session_state.ar_distance_matrix is not None:
                            fig = go.Figure(data=go.Heatmap(
                                z=st.session_state.ar_distance_matrix,
                                x=st.session_state.ar_item_names,
                                y=st.session_state.ar_item_names,
                                colorscale='RdYlGn_r',
                                hovertemplate='%{y} - %{x}<br>Distance: %{z:.3f}<extra></extra>'
                            ))
                            fig.update_layout(
                                title="Item Distance Matrix",
                                xaxis_title="Items",
                                yaxis_title="Items",
                                height=600
                            )
                            render_interactive_chart(fig)
                            st.caption("Lower distance = higher co-occurrence")
                        else:
                            st.info("Distance matrix not available.")
                    
                    with tab7:
                        st.subheader("Top Rules Analysis")
                        col1, col2 = st.columns(2)
                        with col1:
                            rank_metric = st.selectbox("Rank by:", ["lift", "support", "confidence"], key="rank_metric")
                        with col2:
                            top_k_bars = st.slider("Number of rules:", 5, 30, 15, key="top_k_bars")
                        
                        fig = plot_top_rules_bars(rules_df, metric=rank_metric, top_k=top_k_bars)
                        render_interactive_chart(fig)
                    
                    # Interpretation Guide
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
