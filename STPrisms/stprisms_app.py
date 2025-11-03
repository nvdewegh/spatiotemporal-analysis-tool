"""
Space-Time Prisms: Interactive Vibe Coding Environment
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json

# Import our modules
from stprisms_core import (
    Anchor, TrajectorySample, SpaceTimePrism, PrismChain,
    AlibiQuery, VisitProbability, create_trajectory_from_df
)
from vibe_coding import (
    VibeCategory, VibeAnnotation, VibeAnalyzer,
    VibeColorMapper, VibeNarrative
)

# Page configuration
st.set_page_config(
    page_title="Space-Time Prisms Explorer",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'trajectories' not in st.session_state:
    st.session_state.trajectories = []
if 'annotations' not in st.session_state:
    st.session_state.annotations = []
if 'narrative' not in st.session_state:
    st.session_state.narrative = VibeNarrative()
if 'current_query' not in st.session_state:
    st.session_state.current_query = None


def main():
    """Main application"""
    
    st.title("🌐 Space-Time Prisms: Interactive Vibe Coding Environment")
    
    st.markdown("""
    **Uncertainty-based trajectory analysis with qualitative interpretation**
    
    Explore movement scenarios where uncertainty plays a role, combining formal geometric
    models with interpretive "vibe coding" to capture epistemic and phenomenological aspects
    of spatial reasoning.
    """)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Module",
        ["📥 Scenario Manager", "🔬 Uncertainty Analysis", "👁️ Visualization", 
         "💭 Vibe Coding", "📝 Narrative Workspace", "📤 Export"]
    )
    
    if page == "📥 Scenario Manager":
        scenario_manager()
    elif page == "🔬 Uncertainty Analysis":
        uncertainty_analysis()
    elif page == "👁️ Visualization":
        visualization_module()
    elif page == "💭 Vibe Coding":
        vibe_coding_module()
    elif page == "📝 Narrative Workspace":
        narrative_workspace()
    elif page == "📤 Export":
        export_module()


def scenario_manager():
    """Scenario creation and upload interface"""
    
    st.header("📥 Scenario Manager")
    
    st.markdown("""
    Create or upload movement scenarios (trajectory samples).
    Each scenario represents observed anchor points with optional uncertainty.
    """)
    
    tab1, tab2, tab3 = st.tabs(["Upload Data", "Manual Creation", "Example Scenarios"])
    
    with tab1:
        upload_data_interface()
    
    with tab2:
        manual_creation_interface()
    
    with tab3:
        example_scenarios_interface()
    
    # Display current trajectories
    st.subheader("Current Scenarios")
    if st.session_state.trajectories:
        for i, traj in enumerate(st.session_state.trajectories):
            with st.expander(f"**{traj.object_id}** ({len(traj.anchors)} anchors, vmax={traj.vmax:.2f} m/s)"):
                df = pd.DataFrame([
                    {'time': a.t, 'x': a.x, 'y': a.y, 'error': a.error_radius}
                    for a in traj.anchors
                ])
                st.dataframe(df, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Duration", f"{traj.duration():.1f} s")
                with col2:
                    st.metric("Path Length", f"{traj.length():.1f} m")
                
                if st.button(f"Remove {traj.object_id}", key=f"remove_{i}"):
                    st.session_state.trajectories.pop(i)
                    st.rerun()
    else:
        st.info("No trajectories loaded. Upload data or create a scenario manually.")


def upload_data_interface():
    """Upload trajectory data from file"""
    
    st.subheader("Upload Trajectory Data")
    
    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=['csv'],
        help="Expected columns: timestamp, x, y, [object_id], [error_radius]"
    )
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        st.write("Preview:")
        st.dataframe(df.head(), use_container_width=True)
        
        # Configuration
        col1, col2, col3 = st.columns(3)
        
        with col1:
            time_col = st.selectbox("Timestamp column", df.columns, index=0)
        with col2:
            x_col = st.selectbox("X coordinate column", df.columns, index=1 if len(df.columns) > 1 else 0)
        with col3:
            y_col = st.selectbox("Y coordinate column", df.columns, index=2 if len(df.columns) > 2 else 0)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            has_object_id = st.checkbox("Has object ID column", value='object_id' in df.columns or 'ID' in df.columns)
            if has_object_id:
                id_col = st.selectbox("Object ID column", df.columns)
        
        with col2:
            has_error = st.checkbox("Has error radius column", value='error_radius' in df.columns or 'error' in df.columns)
            if has_error:
                error_col = st.selectbox("Error radius column", df.columns)
        
        with col3:
            vmax = st.number_input("Maximum velocity (m/s)", min_value=0.1, value=1.5, step=0.1)
        
        if st.button("Load Trajectory", type="primary"):
            # Prepare dataframe
            traj_df = pd.DataFrame({
                'timestamp': df[time_col],
                'x': df[x_col],
                'y': df[y_col]
            })
            
            if has_error:
                traj_df['error_radius'] = df[error_col]
            
            # Group by object ID if exists
            if has_object_id:
                for obj_id in df[id_col].unique():
                    obj_df = traj_df[df[id_col] == obj_id].copy()
                    traj = create_trajectory_from_df(obj_df, str(obj_id), vmax)
                    st.session_state.trajectories.append(traj)
            else:
                traj = create_trajectory_from_df(traj_df, f"object_{len(st.session_state.trajectories)}", vmax)
                st.session_state.trajectories.append(traj)
            
            st.success(f"✅ Loaded {len(st.session_state.trajectories)} trajectory/trajectories")
            st.rerun()


def manual_creation_interface():
    """Manual trajectory creation"""
    
    st.subheader("Create Trajectory Manually")
    
    object_id = st.text_input("Object ID", value=f"manual_{len(st.session_state.trajectories)}")
    vmax = st.number_input("Maximum velocity (m/s)", min_value=0.1, value=1.5, step=0.1, key="manual_vmax")
    
    st.write("**Add anchor points:**")
    
    num_anchors = st.number_input("Number of anchors", min_value=2, max_value=20, value=3, step=1)
    
    anchors = []
    for i in range(num_anchors):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            t = st.number_input(f"Time {i+1}", value=float(i*10), step=1.0, key=f"t_{i}")
        with col2:
            x = st.number_input(f"X {i+1}", value=float(i*5), step=0.1, key=f"x_{i}")
        with col3:
            y = st.number_input(f"Y {i+1}", value=float(i*3), step=0.1, key=f"y_{i}")
        with col4:
            error = st.number_input(f"Error {i+1}", value=0.0, step=0.1, key=f"error_{i}")
        
        anchors.append(Anchor(x, y, t, error))
    
    if st.button("Create Trajectory", type="primary"):
        traj = TrajectorySample(object_id, anchors, vmax)
        st.session_state.trajectories.append(traj)
        st.success(f"✅ Created trajectory '{object_id}'")
        st.rerun()


def example_scenarios_interface():
    """Load predefined example scenarios"""
    
    st.subheader("Example Scenarios")
    
    scenario = st.selectbox(
        "Select example",
        ["Urban Encounter", "Rural Tracking", "Impossible Meeting"]
    )
    
    st.write(f"**{scenario}:**")
    
    if scenario == "Urban Encounter":
        st.markdown("""
        Two pedestrians in an urban setting. Can they have met at the park?
        - **Person A**: Leaves home at t=0, arrives at work at t=600
        - **Person B**: Leaves café at t=100, arrives at gym at t=500
        - **Speed limit**: 1.5 m/s (walking speed)
        """)
    
    elif scenario == "Rural Tracking":
        st.markdown("""
        Wildlife tracking scenario with GPS uncertainty.
        - **Animal**: Multiple GPS fixes with positioning error
        - **Speed limit**: 2.0 m/s (slow movement)
        - **Uncertainty**: ±5m error radius
        """)
    
    elif scenario == "Impossible Meeting":
        st.markdown("""
        Two objects with contradictory movement constraints.
        - **Object 1**: Requires high speed to reach destination
        - **Object 2**: Similar constraints
        - **Result**: Intersection impossible due to time-space constraints
        """)
    
    if st.button("Load Example", type="primary"):
        load_example_scenario(scenario)
        st.success(f"✅ Loaded '{scenario}'")
        st.rerun()


def load_example_scenario(scenario: str):
    """Load predefined scenario into session state"""
    
    st.session_state.trajectories = []
    
    if scenario == "Urban Encounter":
        # Person A: home to work
        person_a = TrajectorySample(
            "Person A",
            [
                Anchor(0, 0, 0, 2.0),
                Anchor(100, 50, 300, 2.0),
                Anchor(200, 100, 600, 2.0)
            ],
            vmax=1.5
        )
        
        # Person B: café to gym
        person_b = TrajectorySample(
            "Person B",
            [
                Anchor(50, 120, 100, 2.0),
                Anchor(120, 80, 300, 2.0),
                Anchor(180, 40, 500, 2.0)
            ],
            vmax=1.5
        )
        
        st.session_state.trajectories = [person_a, person_b]
    
    elif scenario == "Rural Tracking":
        animal = TrajectorySample(
            "Tracked Animal",
            [
                Anchor(0, 0, 0, 5.0),
                Anchor(15, 12, 120, 5.0),
                Anchor(28, 25, 240, 5.0),
                Anchor(40, 35, 360, 5.0),
                Anchor(50, 50, 480, 5.0)
            ],
            vmax=2.0
        )
        st.session_state.trajectories = [animal]
    
    elif scenario == "Impossible Meeting":
        # Object with tight constraints
        obj1 = TrajectorySample(
            "Object 1",
            [
                Anchor(0, 0, 0, 1.0),
                Anchor(100, 0, 50, 1.0)  # Requires 2 m/s
            ],
            vmax=1.0  # But max speed is only 1 m/s!
        )
        
        obj2 = TrajectorySample(
            "Object 2",
            [
                Anchor(0, 100, 0, 1.0),
                Anchor(100, 100, 50, 1.0)
            ],
            vmax=1.0
        )
        
        st.session_state.trajectories = [obj1, obj2]


def uncertainty_analysis():
    """Uncertainty modeling and computation"""
    
    st.header("🔬 Uncertainty Analysis")
    
    if not st.session_state.trajectories:
        st.warning("⚠️ No trajectories loaded. Please add scenarios first.")
        return
    
    tab1, tab2, tab3 = st.tabs(["Space-Time Prisms", "Alibi Queries", "Visit Probability"])
    
    with tab1:
        prism_analysis()
    
    with tab2:
        alibi_query_analysis()
    
    with tab3:
        visit_probability_analysis()


def prism_analysis():
    """Space-time prism computation and analysis"""
    
    st.subheader("Space-Time Prisms")
    
    st.markdown("""
    Compute space-time prisms for each trajectory. A prism represents the reachable
    space-time region between two anchor points given maximum velocity constraints.
    """)
    
    # Select trajectory
    traj_names = [t.object_id for t in st.session_state.trajectories]
    selected_traj_name = st.selectbox("Select trajectory", traj_names)
    
    traj = next(t for t in st.session_state.trajectories if t.object_id == selected_traj_name)
    
    # Build prism chain
    prisms = []
    for i in range(len(traj.anchors) - 1):
        prism = SpaceTimePrism(traj.anchors[i], traj.anchors[i+1], traj.vmax)
        prisms.append(prism)
    
    chain = PrismChain(prisms, traj)
    
    st.write(f"**{len(prisms)} prisms** in chain")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Freedom", f"{chain.total_freedom():.1f} m²")
    with col2:
        st.metric("Constraint Ratio", f"{chain.constraint_ratio():.2%}")
    with col3:
        feasible_count = sum(1 for p in prisms if p.is_feasible())
        st.metric("Feasible Prisms", f"{feasible_count}/{len(prisms)}")
    
    # Analyze each prism
    st.write("**Individual Prisms:**")
    
    for i, prism in enumerate(prisms):
        with st.expander(f"Prism {i+1}: t={prism.anchor_start.t:.1f} → {prism.anchor_end.t:.1f}"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Time Budget", f"{prism.time_budget():.1f} s")
            with col2:
                st.metric("Distance", f"{prism.spatial_distance():.1f} m")
            with col3:
                st.metric("Feasible", "✅" if prism.is_feasible() else "❌")
            with col4:
                constraint = prism.spatial_distance() / (prism.vmax * prism.time_budget())
                st.metric("Constraint", f"{constraint:.2%}")
            
            # Auto-generate vibe
            vibe = VibeAnalyzer.analyze_prism(prism)
            
            st.write("**Auto-generated Vibe:**")
            for tag in vibe.tags:
                color = VibeColorMapper.get_color(tag.category)
                st.markdown(
                    f"<span style='background-color:{color}; color:white; padding:2px 8px; border-radius:4px; margin:2px;'>"
                    f"{tag.category.value}: {tag.intensity:.2f}</span> - {tag.description}",
                    unsafe_allow_html=True
                )
            
            # Store annotation
            st.session_state.annotations.append(vibe)


def alibi_query_analysis():
    """Alibi query evaluation"""
    
    st.subheader("Alibi Queries")
    
    st.markdown("""
    Evaluate whether multiple objects could have been at the same location simultaneously.
    Computes the intersection of space-time prisms across all selected trajectories.
    """)
    
    if len(st.session_state.trajectories) < 2:
        st.warning("⚠️ Need at least 2 trajectories for alibi query.")
        return
    
    # Select trajectories
    traj_names = [t.object_id for t in st.session_state.trajectories]
    selected_names = st.multiselect(
        "Select trajectories to analyze",
        traj_names,
        default=traj_names[:2]
    )
    
    if len(selected_names) < 2:
        st.info("Select at least 2 trajectories.")
        return
    
    selected_trajs = [t for t in st.session_state.trajectories if t.object_id in selected_names]
    
    # Time range
    all_times = [a.t for traj in selected_trajs for a in traj.anchors]
    t_min, t_max = min(all_times), max(all_times)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        t_start = st.number_input("Start time", value=t_min, step=10.0)
    with col2:
        t_end = st.number_input("End time", value=t_max, step=10.0)
    with col3:
        resolution = st.slider("Time resolution", 10, 100, 50)
    
    if st.button("Evaluate Alibi Query", type="primary"):
        # Compute query
        query = AlibiQuery(selected_trajs)
        result = query.evaluate(t_start, t_end, resolution)
        
        st.session_state.current_query = result
        
        # Display results
        st.write("### Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Intersection Exists",
                "✅ Yes" if result['intersection_exists'] else "❌ No"
            )
        with col2:
            st.metric(
                "Max Overlap Area",
                f"{result['intersection_area']:.2f} m²"
            )
        with col3:
            st.metric(
                "Temporal Overlap",
                f"{result['intersection_probability']:.1%}"
            )
        
        # Auto-generate vibe
        vibe = VibeAnalyzer.analyze_alibi_query(result)
        
        st.write("### Vibe Analysis")
        st.info(vibe.narrative)
        
        cols = st.columns(len(vibe.tags))
        for col, tag in zip(cols, vibe.tags):
            color = VibeColorMapper.get_color(tag.category)
            col.markdown(
                f"<div style='background-color:{color}; color:white; padding:10px; border-radius:8px; text-align:center;'>"
                f"<strong>{tag.category.value}</strong><br/>{tag.intensity:.2f}<br/><small>{tag.description}</small></div>",
                unsafe_allow_html=True
            )
        
        # Store annotation
        st.session_state.annotations.append(vibe)
        
        # Plot temporal overlap
        fig = plot_temporal_overlap(result)
        st.plotly_chart(fig, use_container_width=True)


def plot_temporal_overlap(result: dict) -> go.Figure:
    """Plot intersection area over time"""
    
    times = [ts[0] for ts in result['time_slices']]
    areas = [ts[1] for ts in result['time_slices']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=times,
        y=areas,
        mode='lines+markers',
        name='Intersection Area',
        line=dict(color='#1976D2', width=3),
        fill='tozeroy',
        fillcolor='rgba(25, 118, 210, 0.2)'
    ))
    
    fig.update_layout(
        title="Temporal Overlap Analysis",
        xaxis_title="Time (s)",
        yaxis_title="Intersection Area (m²)",
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def visit_probability_analysis():
    """Visit probability computation"""
    
    st.subheader("Visit Probability")
    
    st.markdown("""
    Compute the probability of visiting different locations given a space-time prism.
    Assumes uniform distribution over all feasible trajectories.
    """)
    
    # Select trajectory
    traj_names = [t.object_id for t in st.session_state.trajectories]
    selected_traj_name = st.selectbox("Select trajectory", traj_names, key="visit_prob_traj")
    
    traj = next(t for t in st.session_state.trajectories if t.object_id == selected_traj_name)
    
    # Select prism
    if len(traj.anchors) < 2:
        st.warning("Trajectory needs at least 2 anchors.")
        return
    
    prism_idx = st.slider("Select prism segment", 0, len(traj.anchors)-2, 0)
    
    prism = SpaceTimePrism(traj.anchors[prism_idx], traj.anchors[prism_idx+1], traj.vmax)
    
    resolution = st.slider("Spatial resolution", 20, 100, 50, key="visit_res")
    
    if st.button("Compute Visit Probability", type="primary"):
        with st.spinner("Computing probability field..."):
            vp = VisitProbability(prism, resolution)
            prob_field = vp.compute_probability_field()
        
        st.success("✅ Computation complete")
        
        # Visualize
        fig = px.density_heatmap(
            prob_field,
            x='x',
            y='y',
            z='probability',
            title="Visit Probability Field",
            color_continuous_scale='YlOrRd',
            labels={'probability': 'Visit Probability'}
        )
        
        # Add anchor points
        fig.add_trace(go.Scatter(
            x=[prism.anchor_start.x, prism.anchor_end.x],
            y=[prism.anchor_start.y, prism.anchor_end.y],
            mode='markers+text',
            marker=dict(size=15, color='blue', symbol='star'),
            text=['Start', 'End'],
            textposition='top center',
            name='Anchors'
        ))
        
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
        
        # Vibe analysis
        vibe = VibeAnalyzer.analyze_visit_probability(prob_field)
        
        st.write("### Vibe Analysis")
        st.info(vibe.narrative)
        
        for tag in vibe.tags:
            color = VibeColorMapper.get_color(tag.category)
            st.markdown(
                f"<span style='background-color:{color}; color:white; padding:4px 12px; border-radius:4px; margin:4px; display:inline-block;'>"
                f"{tag.category.value}: {tag.intensity:.2f} - {tag.description}</span>",
                unsafe_allow_html=True
            )


def visualization_module():
    """Visualization of prisms and trajectories"""
    
    st.header("👁️ Visualization")
    
    if not st.session_state.trajectories:
        st.warning("⚠️ No trajectories loaded.")
        return
    
    tab1, tab2, tab3 = st.tabs(["2D View", "3D Space-Time", "Potential Path Areas"])
    
    with tab1:
        plot_2d_trajectories()
    
    with tab2:
        plot_3d_spacetime()
    
    with tab3:
        plot_ppas()


def plot_2d_trajectories():
    """2D plot of trajectory anchors"""
    
    st.subheader("2D Spatial View")
    
    fig = go.Figure()
    
    for traj in st.session_state.trajectories:
        x_coords = [a.x for a in traj.anchors]
        y_coords = [a.y for a in traj.anchors]
        
        # Trajectory line
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='lines+markers',
            name=traj.object_id,
            line=dict(width=2),
            marker=dict(size=10)
        ))
        
        # Error circles
        for anchor in traj.anchors:
            if anchor.error_radius > 0:
                theta = np.linspace(0, 2*np.pi, 50)
                x_circle = anchor.x + anchor.error_radius * np.cos(theta)
                y_circle = anchor.y + anchor.error_radius * np.sin(theta)
                
                fig.add_trace(go.Scatter(
                    x=x_circle,
                    y=y_circle,
                    mode='lines',
                    line=dict(dash='dash', width=1, color='gray'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    fig.update_layout(
        title="Trajectory Anchors (2D)",
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        template='plotly_white',
        hovermode='closest',
        height=600
    )
    
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    
    st.plotly_chart(fig, use_container_width=True)


def plot_3d_spacetime():
    """3D space-time visualization"""
    
    st.subheader("3D Space-Time View")
    
    st.markdown("Visualize trajectories in 3D space-time (x, y, time).")
    
    fig = go.Figure()
    
    for traj in st.session_state.trajectories:
        x_coords = [a.x for a in traj.anchors]
        y_coords = [a.y for a in traj.anchors]
        t_coords = [a.t for a in traj.anchors]
        
        # 3D trajectory
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=t_coords,
            mode='lines+markers',
            name=traj.object_id,
            line=dict(width=4),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title="Space-Time Trajectories",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Time (s)",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        template='plotly_white',
        height=700
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_ppas():
    """Plot Potential Path Areas at specific time"""
    
    st.subheader("Potential Path Areas (PPAs)")
    
    # Select trajectory and time
    traj_names = [t.object_id for t in st.session_state.trajectories]
    selected_traj_name = st.selectbox("Select trajectory", traj_names, key="ppa_traj")
    
    traj = next(t for t in st.session_state.trajectories if t.object_id == selected_traj_name)
    
    if len(traj.anchors) < 2:
        st.warning("Need at least 2 anchors.")
        return
    
    # Time range
    t_min = traj.anchors[0].t
    t_max = traj.anchors[-1].t
    
    t_query = st.slider("Query time", t_min, t_max, (t_min + t_max) / 2, step=1.0)
    
    # Find relevant prism
    fig = go.Figure()
    
    for i in range(len(traj.anchors) - 1):
        prism = SpaceTimePrism(traj.anchors[i], traj.anchors[i+1], traj.vmax)
        
        if prism.anchor_start.t <= t_query <= prism.anchor_end.t:
            ppa = prism.ppa_at_time(t_query)
            
            if ppa is not None and not ppa.is_empty:
                # Plot PPA
                x_coords, y_coords = ppa.exterior.xy
                
                fig.add_trace(go.Scatter(
                    x=list(x_coords),
                    y=list(y_coords),
                    fill='toself',
                    fillcolor='rgba(25, 118, 210, 0.3)',
                    line=dict(color='#1976D2', width=2),
                    name=f'PPA at t={t_query:.1f}'
                ))
                
                # Add anchors
                fig.add_trace(go.Scatter(
                    x=[prism.anchor_start.x, prism.anchor_end.x],
                    y=[prism.anchor_start.y, prism.anchor_end.y],
                    mode='markers+text',
                    marker=dict(size=15, color='red', symbol='star'),
                    text=['Start', 'End'],
                    textposition='top center',
                    name='Anchors'
                ))
    
    # Add all anchors
    x_all = [a.x for a in traj.anchors]
    y_all = [a.y for a in traj.anchors]
    
    fig.add_trace(go.Scatter(
        x=x_all,
        y=y_all,
        mode='markers',
        marker=dict(size=8, color='black'),
        name='All Anchors'
    ))
    
    fig.update_layout(
        title=f"Potential Path Area at t={t_query:.1f}",
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        template='plotly_white',
        height=600
    )
    
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    
    st.plotly_chart(fig, use_container_width=True)


def vibe_coding_module():
    """Vibe coding interface"""
    
    st.header("💭 Vibe Coding")
    
    st.markdown("""
    View and edit qualitative annotations on analytical results.
    Vibe coding captures epistemic, phenomenological, and interpretive aspects
    of spatial uncertainty that go beyond numerical metrics.
    """)
    
    if not st.session_state.annotations:
        st.info("No annotations yet. Run analyses to generate automatic vibe tags.")
        return
    
    st.write(f"**{len(st.session_state.annotations)} annotations**")
    
    for i, annotation in enumerate(st.session_state.annotations):
        with st.expander(f"{annotation.target_type} - {annotation.author}"):
            st.write(f"**Target:** {annotation.target_id}")
            
            if annotation.narrative:
                st.info(annotation.narrative)
            
            st.write("**Tags:**")
            
            for tag in annotation.tags:
                col1, col2, col3 = st.columns([2, 1, 3])
                
                with col1:
                    color = VibeColorMapper.get_color(tag.category)
                    st.markdown(
                        f"<span style='background-color:{color}; color:white; padding:4px 12px; border-radius:4px;'>"
                        f"{tag.category.value}</span>",
                        unsafe_allow_html=True
                    )
                
                with col2:
                    st.metric("Intensity", f"{tag.intensity:.2f}")
                
                with col3:
                    st.write(tag.description)
            
            # Edit narrative
            new_narrative = st.text_area(
                "Add/edit interpretation:",
                value=annotation.narrative,
                key=f"narrative_{i}"
            )
            
            if st.button("Update", key=f"update_{i}"):
                annotation.narrative = new_narrative
                annotation.author = "user"
                st.success("✅ Updated")


def narrative_workspace():
    """Narrative workspace for researcher notes"""
    
    st.header("📝 Narrative Workspace")
    
    st.markdown("""
    A notebook-like interface for writing interpretations, hypotheses, and reflections.
    Link analytical results with qualitative insights.
    """)
    
    # Add new entry
    with st.expander("➕ Add New Entry", expanded=False):
        title = st.text_input("Title")
        content = st.text_area("Content", height=200)
        
        if st.button("Add Entry", type="primary"):
            if title and content:
                st.session_state.narrative.add_entry(title, content)
                st.success("✅ Entry added")
                st.rerun()
    
    # Display entries
    entries = st.session_state.narrative.get_entries()
    
    if entries:
        st.write(f"**{len(entries)} entries**")
        
        for i, entry in enumerate(reversed(entries)):
            with st.expander(f"**{entry['title']}** - {entry['author']}"):
                st.write(entry['content'])
                st.caption(f"Created: {entry['timestamp']}")
    else:
        st.info("No entries yet. Add your first reflection above.")


def export_module():
    """Export results and annotations"""
    
    st.header("📤 Export")
    
    st.markdown("""
    Export your analysis, visualizations, and vibe annotations.
    """)
    
    tab1, tab2, tab3 = st.tabs(["JSON Data", "Narrative", "Full Report"])
    
    with tab1:
        export_json()
    
    with tab2:
        export_narrative()
    
    with tab3:
        export_report()


def export_json():
    """Export data as JSON"""
    
    st.subheader("Export JSON Data")
    
    data = {
        'trajectories': [
            {
                'object_id': t.object_id,
                'vmax': t.vmax,
                'anchors': [
                    {'x': a.x, 'y': a.y, 't': a.t, 'error': a.error_radius}
                    for a in t.anchors
                ]
            }
            for t in st.session_state.trajectories
        ],
        'annotations': [ann.to_dict() for ann in st.session_state.annotations]
    }
    
    json_str = json.dumps(data, indent=2)
    
    st.download_button(
        label="Download JSON",
        data=json_str,
        file_name="stprisms_export.json",
        mime="application/json"
    )
    
    with st.expander("Preview"):
        st.json(data)


def export_narrative():
    """Export narrative workspace"""
    
    st.subheader("Export Narrative")
    
    entries = st.session_state.narrative.get_entries()
    
    # Format as markdown
    md_content = "# Space-Time Prisms Analysis Narrative\n\n"
    
    for entry in entries:
        md_content += f"## {entry['title']}\n\n"
        md_content += f"*By {entry['author']} on {entry['timestamp']}*\n\n"
        md_content += f"{entry['content']}\n\n"
        md_content += "---\n\n"
    
    st.download_button(
        label="Download Narrative (Markdown)",
        data=md_content,
        file_name="narrative.md",
        mime="text/markdown"
    )
    
    with st.expander("Preview"):
        st.markdown(md_content)


def export_report():
    """Export comprehensive HTML report"""
    
    st.subheader("Full Report")
    
    st.info("Generate a comprehensive HTML report with all visualizations and annotations.")
    
    if st.button("Generate Report", type="primary"):
        st.success("✅ Report generation (HTML export would be implemented here)")
        st.info("This would create an interactive HTML page with all analyses, plots, and vibe annotations.")


if __name__ == "__main__":
    main()
