import streamlit as st
import locale
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except:
    try:
        locale.setlocale(locale.LC_ALL, 'C')
    except:
        pass

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from itertools import combinations, groupby
from collections import Counter
from scipy.spatial.distance import cdist, euclidean, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Import analysis modules
from modules import association_rules, clustering, sequence_analysis, outlier_detection, utils, pdp_analysis
from ui_components import configuration_selector

# Common Plotly configuration for interactive charts
PLOTLY_CONFIG = {
    "displaylogo": False,
    "scrollZoom": True,
    "doubleClick": "reset",
    "modeBarButtonsToAdd": [
        "zoom2d",
        "pan2d",
        "autoScale2d",
        "resetScale2d"
    ],
    "modeBarButtonsToRemove": [
        "lasso2d",
        "select2d"
    ]
}


def render_interactive_chart(fig, caption="Use the toolbar to zoom, pan, or reset (double-click).", key=None):
    """Render a Plotly figure with consistent interactive controls."""
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key=key)
    if caption:
        st.caption(caption)

# Page configuration
st.set_page_config(
    page_title="Spatiotemporal Analysis and Modeling",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS to prevent scroll jumping
st.markdown("""
    <style>
    /* Prevent auto-scroll during updates */
    .main {
        scroll-behavior: auto;
    }
    </style>
    <script>
    // Save and restore scroll position
    window.addEventListener('beforeunload', function() {
        sessionStorage.setItem('scrollPos', window.scrollY);
    });
    window.addEventListener('load', function() {
        const scrollPos = sessionStorage.getItem('scrollPos');
        if (scrollPos) {
            window.scrollTo(0, parseInt(scrollPos));
        }
    });
    </script>
""", unsafe_allow_html=True)

# Password protection
def check_password():
    """Returns `True` if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == "GIST":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.title("Spatiotemporal Analysis and Modeling")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.info("Please enter the password provided by your instructor.")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.title("Spatiotemporal Analysis and Modeling")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("Incorrect password. Please try again.")
        return False
    else:
        # Password correct.
        return True

if not check_password():
    st.stop()  # Do not continue if check_password is not True.

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'current_time' not in st.session_state:
    st.session_state.current_time = 0
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False
if 'max_time' not in st.session_state:
    st.session_state.max_time = 0
if 'filename' not in st.session_state:
    st.session_state.filename = None
if 'court_type' not in st.session_state:
    st.session_state.court_type = 'Tennis'
if 'uploaded_filenames' not in st.session_state:
    st.session_state.uploaded_filenames = []
if 'config_sources' not in st.session_state:
    st.session_state.config_sources = []

# ============================================================================
# NOTE: Utility functions (get_color, douglas_peucker, load_data, etc.)
# are now imported from modules/utils.py
# ============================================================================

# ============================================================================
# NOTE: Clustering functions (format_features_dataframe, extract_trajectory_features, etc.)
# are now imported from modules/clustering.py
# ============================================================================

# ============================================================================
# NOTE: Sequence analysis functions (create_spatial_grid, build_*_sequence, etc.)
# are now imported from modules/sequence_analysis.py
# ============================================================================

# Draw soccer pitch
def create_football_pitch():
    """Create a Plotly figure with soccer pitch markings"""
    fig = go.Figure()
    
    # Field dimensions
    pitch_width = 110
    pitch_height = 72
    
    # Field boundary
    fig.add_shape(type="rect", x0=0, y0=0, x1=pitch_width, y1=pitch_height,
                  line=dict(color="green", width=2))
    
    # Center line
    fig.add_shape(type="line", x0=pitch_width/2, y0=0, 
                  x1=pitch_width/2, y1=pitch_height,
                  line=dict(color="green", width=2))
    
    # Center circle
    fig.add_shape(type="circle", 
                  xref="x", yref="y",
                  x0=pitch_width/2 - 9.15, y0=pitch_height/2 - 9.15,
                  x1=pitch_width/2 + 9.15, y1=pitch_height/2 + 9.15,
                  line=dict(color="green", width=2))
    
    # Penalty areas
    # Left penalty area
    fig.add_shape(type="rect", x0=0, y0=pitch_height/2 - 20.15,
                  x1=16.5, y1=pitch_height/2 + 20.15,
                  line=dict(color="green", width=2))
    # Right penalty area
    fig.add_shape(type="rect", x0=pitch_width - 16.5, y0=pitch_height/2 - 20.15,
                  x1=pitch_width, y1=pitch_height/2 + 20.15,
                  line=dict(color="green", width=2))
    
    # Goal areas
    # Left goal area
    fig.add_shape(type="rect", x0=0, y0=pitch_height/2 - 9,
                  x1=5.5, y1=pitch_height/2 + 9,
                  line=dict(color="green", width=2))
    # Right goal area
    fig.add_shape(type="rect", x0=pitch_width - 5.5, y0=pitch_height/2 - 9,
                  x1=pitch_width, y1=pitch_height/2 + 9,
                  line=dict(color="green", width=2))
    
    # Penalty spots
    fig.add_trace(go.Scatter(x=[11, pitch_width - 11], 
                             y=[pitch_height/2, pitch_height/2],
                             mode='markers', marker=dict(size=5, color='green'),
                             showlegend=False, hoverinfo='skip'))
    
    # Center spot
    fig.add_trace(go.Scatter(x=[pitch_width/2], y=[pitch_height/2],
                             mode='markers', marker=dict(size=5, color='green'),
                             showlegend=False, hoverinfo='skip'))
    
    fig.update_layout(
        width=900,
        height=600,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            range=[0, pitch_width],
            showgrid=False,
            zeroline=False,
            constrain='domain',
            fixedrange=False
        ),
        yaxis=dict(
            range=[0, pitch_height],
            showgrid=False,
            zeroline=False,
            scaleanchor='x',
            scaleratio=1,
            fixedrange=False
        ),
        plot_bgcolor='lightgreen',
        showlegend=True,
        hovermode='closest',
        dragmode='pan',
        uirevision='football-pitch'
    )
    
    return fig

# Draw tennis court
def create_tennis_court():
    """Create a Plotly figure with tennis court markings"""
    fig = go.Figure()
    
    # Court dimensions (in meters)
    court_width = 8.23  # Singles court width
    court_length = 23.77
    
    # Doubles court dimensions
    doubles_width = 10.97
    doubles_alley_width = (doubles_width - court_width) / 2  # 1.37m on each side
    
    # Service box and other measurements
    service_line_distance = 6.40  # Distance from net to service line
    center_service_line_start = 11.88  # Distance from baseline
    net_position = court_length / 2  # 11.885m
    
    # Origin is at bottom-left of SINGLES court
    # Doubles alleys extend into negative x (left) and beyond court_width (right)
    
    # Outer boundary (doubles court) - extends from -1.37 to 10.97-1.37=9.60
    fig.add_shape(type="rect", 
                  x0=-doubles_alley_width, y0=0, 
                  x1=court_width + doubles_alley_width, y1=court_length,
                  line=dict(color="white", width=3))
    
    # Singles sidelines (at x=0 and x=8.23)
    fig.add_shape(type="line", x0=0, y0=0, x1=0, y1=court_length,
                  line=dict(color="white", width=2))
    fig.add_shape(type="line", x0=court_width, y0=0, x1=court_width, y1=court_length,
                  line=dict(color="white", width=2))
    
    # Baselines (full width including doubles alleys)
    fig.add_shape(type="line", 
                  x0=-doubles_alley_width, y0=0, 
                  x1=court_width + doubles_alley_width, y1=0,
                  line=dict(color="white", width=3))
    fig.add_shape(type="line", 
                  x0=-doubles_alley_width, y0=court_length, 
                  x1=court_width + doubles_alley_width, y1=court_length,
                  line=dict(color="white", width=3))
    
    # Net (center line) - full width including doubles alleys
    fig.add_shape(type="line", 
                  x0=-doubles_alley_width, y0=net_position, 
                  x1=court_width + doubles_alley_width, y1=net_position,
                  line=dict(color="white", width=2))
    
    # Service lines (6.40m from net on each side) - only within singles court
    service_line_bottom = net_position - service_line_distance
    service_line_top = net_position + service_line_distance
    
    fig.add_shape(type="line", x0=0, y0=service_line_bottom, 
                  x1=court_width, y1=service_line_bottom,
                  line=dict(color="white", width=2))
    fig.add_shape(type="line", x0=0, y0=service_line_top, 
                  x1=court_width, y1=service_line_top,
                  line=dict(color="white", width=2))
    
    # Center service line (divides service boxes) - center of singles court
    center_x = court_width / 2  # 4.115m
    fig.add_shape(type="line", x0=center_x, y0=service_line_bottom, 
                  x1=center_x, y1=service_line_top,
                  line=dict(color="white", width=2))
    
    # Center mark on baselines (small marks)
    center_mark_length = 0.10  # 10cm
    fig.add_shape(type="line", x0=center_x, y0=0, 
                  x1=center_x, y1=center_mark_length,
                  line=dict(color="white", width=2))
    fig.add_shape(type="line", x0=center_x, y0=court_length - center_mark_length, 
                  x1=center_x, y1=court_length,
                  line=dict(color="white", width=2))
    
    # Net posts (singles) - at edges of singles court
    post_diameter = 0.15
    fig.add_trace(go.Scatter(x=[0, court_width], 
                             y=[net_position, net_position],
                             mode='markers', 
                             marker=dict(size=8, color='white', symbol='square'),
                             showlegend=False, hoverinfo='skip'))
    
    # Add margin around court for player movement
    x_margin = 3.0  # 3 meters on each side
    y_margin = 4.0  # 4 meters behind each baseline 
    
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
        plot_bgcolor='#25D366',  # WhatsApp green for grass court
        showlegend=True,
        hovermode='closest',
        dragmode='pan',
        uirevision='tennis-court'
    )
    
    return fig

# Unified function to create pitch based on court type
def create_pitch_figure(court_type='Football'):
    """Create a Plotly figure with pitch markings based on court type"""
    if court_type == 'Tennis':
        return create_tennis_court()
    else:
        return create_football_pitch()

# Get court dimensions based on type
def get_court_dimensions(court_type='Football'):
    """Return court dimensions based on type"""
    if court_type == 'Tennis':
        return {
            'width': 8.23,  # Singles court width (origin at singles court)
            'height': 23.77,  # Court length
            'aspect_width': 400,
            'aspect_height': 1100
        }
    else:  # Football
        return {
            'width': 110,
            'height': 72,
            'aspect_width': 900,
            'aspect_height': 600
        }

# Aggregate data based on method
def aggregate_points(points, aggregation_type, temporal_resolution):
    """Aggregate points based on selected method"""
    if aggregation_type == 'Skip frames':
        return [points[i] for i in range(0, len(points), temporal_resolution)]
    
    elif aggregation_type == 'Average locations':
        aggregated = []
        for i in range(0, len(points), temporal_resolution):
            subset = points[i:i + temporal_resolution]
            if subset:
                avg_point = {
                    'x': np.mean([p['x'] for p in subset]),
                    'y': np.mean([p['y'] for p in subset]),
                    'timestamp': subset[0]['timestamp']
                }
                aggregated.append(avg_point)
        return aggregated
    
    elif aggregation_type == 'Spatially generalise':
        return utils.douglas_peucker(points, temporal_resolution)
    
    elif aggregation_type == 'Spatiotemporal generalise':
        return utils.douglas_peucker_spatiotemporal(points, temporal_resolution)
    
    elif aggregation_type == 'Smoothing average':
        aggregated = []
        for i in range(len(points) - temporal_resolution + 1):
            subset = points[i:i + temporal_resolution]
            if subset:
                avg_point = {
                    'x': np.mean([p['x'] for p in subset]),
                    'y': np.mean([p['y'] for p in subset]),
                    'timestamp': subset[i]['timestamp']
                }
                aggregated.append(avg_point)
        return aggregated
    
    return points

def interpolate_points(points, interpolation_steps=5):
    """
    Interpolate points to create smooth continuous movement between timestamps.
    
    Args:
        points: List of dicts with 'x', 'y', 'timestamp' keys
        interpolation_steps: Number of intermediate points to create between each pair
    
    Returns:
        List of interpolated points with smooth transitions
    """
    if len(points) < 2 or interpolation_steps < 1:
        return points
    
    interpolated = []
    
    for i in range(len(points) - 1):
        current = points[i]
        next_point = points[i + 1]
        
        # Add the current point
        interpolated.append(current)
        
        # Create intermediate points
        for step in range(1, interpolation_steps):
            alpha = step / interpolation_steps
            interp_point = {
                'x': current['x'] + alpha * (next_point['x'] - current['x']),
                'y': current['y'] + alpha * (next_point['y'] - current['y']),
                'timestamp': current['timestamp'] + alpha * (next_point['timestamp'] - current['timestamp'])
            }
            interpolated.append(interp_point)
    
    # Add the last point
    interpolated.append(points[-1])
    
    return interpolated

# Visualize static trajectories
def visualize_static(df, selected_configs, selected_objects, start_time, end_time, 
                     aggregation_type, temporal_resolution, translate_to_center=False, court_type='Football'):
    """Create static trajectory visualization"""
    fig = create_pitch_figure(court_type)
    court_dims = get_court_dimensions(court_type)
    
    center_x = court_dims['width'] / 2
    center_y = court_dims['height'] / 2
    
    # Build a color map per (config, object) so same object in different configs is distinguishable
    try:
        palette = px.colors.qualitative.Plotly
    except Exception:
        palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    color_map = {}
    ci = 0
    for config in selected_configs:
        for obj_id in selected_objects:
            color_map[(config, obj_id)] = palette[ci % len(palette)]
            ci += 1

    for config in selected_configs:
        config_data = df[df['config_source'] == config]
        
        for obj_id in selected_objects:
            obj_data = config_data[config_data['obj'] == obj_id]
            obj_data = obj_data[(obj_data['tst'] >= start_time) & (obj_data['tst'] <= end_time)]
            obj_data = obj_data.sort_values('tst')
            
            if len(obj_data) == 0:
                continue
            
            # Convert to list of dicts
            points = obj_data[['x', 'y', 'tst']].rename(columns={'tst': 'timestamp'}).to_dict('records')
            
            # Translate to center if in 2SA mode
            if translate_to_center and points:
                start_point = points[0]
                delta_x = center_x - start_point['x']
                delta_y = center_y - start_point['y']
                points = [{'x': p['x'] + delta_x, 'y': p['y'] + delta_y, 
                          'timestamp': p['timestamp']} for p in points]
            
            # Apply aggregation
            points = aggregate_points(points, aggregation_type, temporal_resolution)
            
            if len(points) < 2:
                continue
            
            x_coords = [p['x'] for p in points]
            y_coords = [p['y'] for p in points]
            
            color = color_map.get((config, obj_id), utils.get_color(obj_id))
            
            # Create legend group name
            legend_group = f'{config} | Obj {obj_id}'
            
            # Draw trajectory line and markers, but exclude the last marker
            fig.add_trace(go.Scatter(
                x=x_coords, y=y_coords,
                mode='lines+markers',
                name=f'{config} - Obj {obj_id}',
                legendgroup=legend_group,
                line=dict(color=color, width=2),
                marker=dict(
                    size=[4] * (len(x_coords) - 1) + [0],  # Hide the last marker
                    color=color
                ),
                hovertemplate=f'Object {obj_id}<br>Config {config}<br>x: %{{x:.2f}}m<br>y: %{{y:.2f}}m<extra></extra>'
            ))

            # Add arrow at the end as a separate trace with a correctly oriented symbol
            if len(x_coords) >= 2:
                dx = x_coords[-1] - x_coords[-2]
                dy = y_coords[-1] - y_coords[-2]
                angle = np.degrees(np.arctan2(dy, dx))

                fig.add_trace(go.Scatter(
                    x=[x_coords[-1]],
                    y=[y_coords[-1]],
                    mode='markers',
                    marker=dict(
                        symbol='arrow',
                        color=color,
                        size=15,
                        angle=angle
                    ),
                    showlegend=False,
                    legendgroup=legend_group,
                    hoverinfo='skip'
                ))
    
    return fig

# Create animated visualization with Plotly frames
def visualize_animated(df, selected_configs, selected_objects, start_time, end_time, 
                       aggregation_type, temporal_resolution, court_type='Football', 
                       animation_speed=200, use_interpolation=False, interpolation_steps=5):
    """Create smooth animation using Plotly's built-in animation"""
    
    # Get unique time steps from the data
    original_time_steps = sorted(df[(df['tst'] >= start_time) & (df['tst'] <= end_time)]['tst'].unique())

    # If there are too many unique time steps, sample them for performance (keep first and last)
    max_frames = 80
    if len(original_time_steps) == 0:
        # Fallback to linspace if no data
        original_time_steps = list(np.linspace(start_time, end_time, 50))
    elif len(original_time_steps) > max_frames and not use_interpolation:
        # Only sample if not using interpolation
        indices = np.linspace(0, len(original_time_steps) - 1, max_frames).astype(int)
        original_time_steps = [original_time_steps[i] for i in indices]
    
    # If interpolation is enabled, create interpolated time steps for smoother animation
    if use_interpolation and len(original_time_steps) > 1:
        time_steps = []
        for i in range(len(original_time_steps) - 1):
            t_start = original_time_steps[i]
            t_end = original_time_steps[i + 1]
            # Add intermediate time steps
            for step in range(interpolation_steps):
                alpha = step / interpolation_steps
                time_steps.append(t_start + alpha * (t_end - t_start))
        time_steps.append(original_time_steps[-1])  # Add the last timestamp
        
        # Limit total frames to avoid performance issues
        if len(time_steps) > max_frames * 2:
            indices = np.linspace(0, len(time_steps) - 1, max_frames * 2).astype(int)
            time_steps = [time_steps[i] for i in indices]
    else:
        time_steps = original_time_steps
    
    # Initialize frames list
    frames = []
    
    # Create initial figure with court background and fixed dimensions
    fig = create_pitch_figure(court_type)
    
    # Build color map for consistency
    try:
        palette = px.colors.qualitative.Plotly
    except Exception:
        palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    color_map = {}
    ci = 0
    for config in selected_configs:
        for obj_id in selected_objects:
            color_map[(config, obj_id)] = palette[ci % len(palette)]
            ci += 1
    
    # Prepare data for all objects at all times
    for frame_idx, current_time in enumerate(time_steps):
        frame_data = []
        
        for config in selected_configs:
            config_data = df[df['config_source'] == config]
            
            for obj_id in selected_objects:
                obj_data = config_data[config_data['obj'] == obj_id]
                # Get all data from start to beyond current_time for interpolation
                obj_data_extended = obj_data[(obj_data['tst'] >= start_time)]
                obj_data_extended = obj_data_extended.sort_values('tst')
                
                if len(obj_data_extended) == 0:
                    continue
                
                # Get all available points for interpolation
                all_points = obj_data_extended[['x', 'y', 'tst']].rename(columns={'tst': 'timestamp'}).to_dict('records')
                all_points = aggregate_points(all_points, aggregation_type, temporal_resolution)
                
                if len(all_points) == 0:
                    continue
                
                # Apply interpolation if enabled
                if use_interpolation and len(all_points) > 1:
                    all_points = interpolate_points(all_points, interpolation_steps)
                
                # Now filter to show only up to current_time
                points = [p for p in all_points if p['timestamp'] <= current_time]
                
                # If current_time is between two points, interpolate to exact current_time
                if use_interpolation and len(points) > 0 and points[-1]['timestamp'] < current_time:
                    # Find the next point after current_time
                    next_points = [p for p in all_points if p['timestamp'] > current_time]
                    if next_points:
                        prev_point = points[-1]
                        next_point = next_points[0]
                        # Interpolate to exact current_time
                        time_diff = next_point['timestamp'] - prev_point['timestamp']
                        if time_diff > 0:
                            alpha = (current_time - prev_point['timestamp']) / time_diff
                            interpolated_point = {
                                'x': prev_point['x'] + alpha * (next_point['x'] - prev_point['x']),
                                'y': prev_point['y'] + alpha * (next_point['y'] - prev_point['y']),
                                'timestamp': current_time
                            }
                            points.append(interpolated_point)
                
                if len(points) == 0:
                    continue
                
                color = color_map.get((config, obj_id), utils.get_color(obj_id))
                legend_group = f'{config} | Obj {obj_id}'
                
                # For animated trajectories, only show the current position marker
                # No trajectory lines - this keeps the animation clean and smooth
                current_point = points[-1]
                
                frame_data.append(go.Scatter(
                    x=[current_point['x']], y=[current_point['y']],
                    mode='markers',
                    marker=dict(size=10, color=color),
                    name=f'{config} - Obj {obj_id}',
                    legendgroup=legend_group,
                    showlegend=(frame_idx == 0),
                    hovertemplate=f'Object {obj_id}<br>Config: {config}<br>Time: {current_time:.2f}<br>x: {current_point["x"]:.2f}m<br>y: {current_point["y"]:.2f}m<extra></extra>'
                ))
        
        # Create frame with layout that matches initial figure to prevent jumping
        frames.append(go.Frame(
            data=frame_data,
            name=str(frame_idx)
        ))
    
    # Add initial frame data to figure
    if frames:
        fig.add_traces(frames[0].data)
    
    # Add frames to figure
    fig.frames = frames
    
    # Add animation controls
    # Set transition duration to create smooth movement between frames
    # Use 80% of animation_speed for smooth transitions without overlap
    transition_duration = int(animation_speed * 0.8)
    
    fig.update_layout(
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': animation_speed, 'redraw': False},
                        'fromcurrent': True,
                        'mode': 'immediate',
                        'transition': {'duration': transition_duration, 'easing': 'linear'}
                    }]
                },
                {
                    'label': 'Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
            ],
            'x': 0.1,
            'y': 1.15,
            'xanchor': 'left',
            'yanchor': 'top'
        }],
        sliders=[{
            'active': 0,
            'steps': [
                {
                    'args': [[f.name], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }],
                    'label': f't={int(time_steps[i])}' if time_steps[i] == int(time_steps[i]) else f't={time_steps[i]:.1f}',
                    'method': 'animate'
                }
                for i, f in enumerate(frames)
            ],
            'x': 0.1,
            'len': 0.85,
            'xanchor': 'left',
            'y': 0,
            'yanchor': 'top'
        }]
    )
    
    return fig

# Visualize at specific time
def visualize_at_time(df, selected_configs, selected_objects, current_time, 
                      start_time, aggregation_type, temporal_resolution, court_type='Football',
                      use_interpolation=False, interpolation_steps=5):
    """Create visualization at specific time point"""
    fig = create_pitch_figure(court_type)
    
    # Build color map for consistency
    try:
        palette = px.colors.qualitative.Plotly
    except Exception:
        palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    color_map = {}
    ci = 0
    for config in selected_configs:
        for obj_id in selected_objects:
            color_map[(config, obj_id)] = palette[ci % len(palette)]
            ci += 1
    
    for config in selected_configs:
        config_data = df[df['config_source'] == config]
        
        for obj_id in selected_objects:
            obj_data = config_data[config_data['obj'] == obj_id]
            # Get all data from start to beyond current_time for interpolation
            obj_data_extended = obj_data[(obj_data['tst'] >= start_time)]
            obj_data_extended = obj_data_extended.sort_values('tst')
            
            if len(obj_data_extended) == 0:
                continue
            
            # Get all available points for interpolation
            all_points = obj_data_extended[['x', 'y', 'tst']].rename(columns={'tst': 'timestamp'}).to_dict('records')
            all_points = aggregate_points(all_points, aggregation_type, temporal_resolution)
            
            if len(all_points) == 0:
                continue
            
            # Apply interpolation if enabled
            if use_interpolation and len(all_points) > 1:
                all_points = interpolate_points(all_points, interpolation_steps)
            
            # Now filter to show only up to current_time
            points = [p for p in all_points if p['timestamp'] <= current_time]
            
            # If current_time is between two points, interpolate to exact current_time
            if use_interpolation and len(points) > 0 and points[-1]['timestamp'] < current_time:
                # Find the next point after current_time
                next_points = [p for p in all_points if p['timestamp'] > current_time]
                if next_points:
                    prev_point = points[-1]
                    next_point = next_points[0]
                    # Interpolate to exact current_time
                    time_diff = next_point['timestamp'] - prev_point['timestamp']
                    if time_diff > 0:
                        alpha = (current_time - prev_point['timestamp']) / time_diff
                        interpolated_point = {
                            'x': prev_point['x'] + alpha * (next_point['x'] - prev_point['x']),
                            'y': prev_point['y'] + alpha * (next_point['y'] - prev_point['y']),
                            'timestamp': current_time
                        }
                        points.append(interpolated_point)
            
            if len(points) == 0:
                continue
            
            x_coords = [p['x'] for p in points]
            y_coords = [p['y'] for p in points]
            
            color = color_map.get((config, obj_id), utils.get_color(obj_id))
            legend_group = f'{config} | Obj {obj_id}'
            
            # Draw trajectory
            fig.add_trace(go.Scatter(
                x=x_coords, y=y_coords,
                mode='lines',
                name=f'{config} - Obj {obj_id}',
                legendgroup=legend_group,
                line=dict(color=color, width=2),
                showlegend=True
            ))
            
            # Draw current position
            if points:
                current_point = points[-1]
                fig.add_trace(go.Scatter(
                    x=[current_point['x']], y=[current_point['y']],
                    mode='markers',
                    marker=dict(size=10, color=color),
                    name=f'Current Obj {obj_id}',
                    legendgroup=legend_group,
                    showlegend=False,
                    hovertemplate=f'Object {obj_id}<br>Config: {config}<br>Time: {current_time:.2f}<br>x: {current_point["x"]:.2f}m<br>y: {current_point["y"]:.2f}m<extra></extra>'
                ))
    
    return fig

# Calculate average position
def visualize_average_position(df, selected_configs, selected_objects, start_time, end_time, court_type='Football'):
    """Calculate and visualize average positions"""
    fig = create_pitch_figure(court_type)
    
    all_avg_x = []
    all_avg_y = []
    
    # Build color map for consistency with other views
    try:
        palette = px.colors.qualitative.Plotly
    except Exception:
        palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    color_map = {}
    ci = 0
    for config in selected_configs:
        for obj_id in selected_objects:
            color_map[(config, obj_id)] = palette[ci % len(palette)]
            ci += 1
    
    for config in selected_configs:
        config_data = df[df['config_source'] == config]
        
        for obj_id in selected_objects:
            obj_data = config_data[config_data['obj'] == obj_id]
            obj_data = obj_data[(obj_data['tst'] >= start_time) & (obj_data['tst'] <= end_time)]
            
            if len(obj_data) > 0:
                avg_x = obj_data['x'].mean()
                avg_y = obj_data['y'].mean()
                
                all_avg_x.append(avg_x)
                all_avg_y.append(avg_y)
                
                color = color_map.get((config, obj_id), utils.get_color(obj_id))
                legend_group = f'{config} | Obj {obj_id}'
                
                fig.add_trace(go.Scatter(
                    x=[avg_x], y=[avg_y],
                    mode='markers+text',
                    marker=dict(size=15, color=color),
                    text=[f'Obj {obj_id}'],
                    textposition="top center",
                    name=f'{config} - Obj {obj_id} Avg',
                    legendgroup=legend_group,
                    hovertemplate=(
                        f'Avg Object {obj_id}<br>Config: {config}<br>'
                        f'x: {avg_x:.2f}m<br>y: {avg_y:.2f}m<extra></extra>'
                    )
                ))
    
    # Overall average
    if all_avg_x:
        overall_avg_x = np.mean(all_avg_x)
        overall_avg_y = np.mean(all_avg_y)
        
        fig.add_trace(go.Scatter(
            x=[overall_avg_x], y=[overall_avg_y],
            mode='markers+text',
            marker=dict(size=20, color='black', symbol='star'),
            text=['Overall Avg'],
            textposition="top center",
            name='Overall Average',
            hovertemplate=f'Overall Average<br>x: {overall_avg_x:.2f}m<br>y: {overall_avg_y:.2f}m<extra></extra>'
        ))
    
    return fig

# ============================================================================
# DTW (DYNAMIC TIME WARPING) FUNCTION
# ============================================================================

def dtw_distance(traj1, traj2):
    """
    Compute Dynamic Time Warping distance between two trajectories.
    
    Parameters:
    -----------
    traj1 : numpy array of shape (n, 2) - first trajectory with x, y coordinates
    traj2 : numpy array of shape (m, 2) - second trajectory with x, y coordinates
    
    Returns:
    --------
    float : DTW distance between the two trajectories
    """
    n, m = len(traj1), len(traj2)
    
    # Initialize DTW matrix with infinity
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    # Fill the DTW matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Euclidean distance between points
            cost = np.sqrt(np.sum((traj1[i-1] - traj2[j-1])**2))
            
            # Take minimum of three possible paths
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],      # insertion
                dtw_matrix[i, j-1],      # deletion
                dtw_matrix[i-1, j-1]     # match
            )
    
    return dtw_matrix[n, m]

# ============================================================================
# CLUSTERING FUNCTIONS (OLD - For Visual Exploration)
# ============================================================================

def extract_trajectory_features_old(df, obj_id, config, start_time, end_time):
    """Extract general properties features from a trajectory (old version for Visual Exploration)"""
    obj_data = df[(df['obj'] == obj_id) & 
                  (df['config_source'] == config) &
                  (df['tst'] >= start_time) & 
                  (df['tst'] <= end_time)].sort_values('tst')
    
    if len(obj_data) < 2:
        return None
    
    # Calculate features
    coords = obj_data[['x', 'y']].values
    times = obj_data['tst'].values
    
    # Distance traveled
    distances = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))
    total_distance = np.sum(distances)
    
    # Duration
    duration = times[-1] - times[0]
    
    # Average speed
    avg_speed = total_distance / duration if duration > 0 else 0
    
    # Max speed
    time_diffs = np.diff(times)
    speeds = distances / time_diffs
    speeds = speeds[time_diffs > 0]
    max_speed = np.max(speeds) if len(speeds) > 0 else 0
    
    # Displacement (straight line from start to end)
    displacement = np.sqrt((coords[-1][0] - coords[0][0])**2 + 
                          (coords[-1][1] - coords[0][1])**2)
    
    # Sinuosity (how curved the path is)
    sinuosity = total_distance / displacement if displacement > 0 else 1
    
    # Bounding box area
    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()
    bbox_area = x_range * y_range
    
    # Direction (overall bearing from start to end)
    # Calculate angle in degrees (0° = East, 90° = North, 180° = West, 270° = South)
    dx = coords[-1][0] - coords[0][0]
    dy = coords[-1][1] - coords[0][1]
    direction = np.degrees(np.arctan2(dy, dx))  # Range: -180 to 180
    # Normalize to 0-360 range
    if direction < 0:
        direction += 360
    
    # Start and end positions
    start_x, start_y = coords[0]
    end_x, end_y = coords[-1]
    
    return {
        'obj_id': obj_id,
        'config': config,
        'total_distance': total_distance,
        'duration': duration,
        'avg_speed': avg_speed,
        'max_speed': max_speed,
        'displacement': displacement,
        'sinuosity': sinuosity,
        'bbox_area': bbox_area,
        'direction': direction,
        'start_x': start_x,
        'start_y': start_y,
        'end_x': end_x,
        'end_y': end_y,
        'num_points': len(obj_data)
    }

def dtw_distance(traj1, traj2):
    """Calculate Dynamic Time Warping distance between two trajectories"""
    n, m = len(traj1), len(traj2)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = euclidean(traj1[i-1], traj2[j-1])
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    
    return dtw[n, m]

def hausdorff_distance(traj1, traj2):
    """Calculate Hausdorff distance between two trajectories"""
    distances1 = cdist(traj1, traj2, 'euclidean')
    distances2 = cdist(traj2, traj1, 'euclidean')
    
    max_min1 = np.max(np.min(distances1, axis=1))
    max_min2 = np.max(np.min(distances2, axis=1))
    
    return max(max_min1, max_min2)

def chamfer_distance(traj1, traj2):
    """
    Calculate Chamfer distance (average symmetric distance) between two trajectories.
    This is a simple, intuitive measure: for each point in one trajectory, 
    find the nearest point in the other trajectory, then average all these distances.
    Much more robust to outliers than Hausdorff distance.
    """
    distances1 = cdist(traj1, traj2, 'euclidean')
    distances2 = cdist(traj2, traj1, 'euclidean')
    
    # For each point in traj1, find nearest point in traj2
    avg_dist1 = np.mean(np.min(distances1, axis=1))
    
    # For each point in traj2, find nearest point in traj1
    avg_dist2 = np.mean(np.min(distances2, axis=1))
    
    # Symmetric average
    return (avg_dist1 + avg_dist2) / 2

def get_trajectory_coords(df, obj_id, config, start_time, end_time):
    """Get trajectory coordinates for a specific object"""
    obj_data = df[(df['obj'] == obj_id) & 
                  (df['config_source'] == config) &
                  (df['tst'] >= start_time) & 
                  (df['tst'] <= end_time)].sort_values('tst')
    
    if len(obj_data) < 2:
        return None
    
    return obj_data[['x', 'y']].values

def find_moving_flocks(df, selected_configs, selected_objects, start_time, end_time, 
                       distance_threshold, min_duration):
    """Find groups of objects moving together (flocking behavior)"""
    time_steps = sorted(df[(df['tst'] >= start_time) & (df['tst'] <= end_time)]['tst'].unique())
    
    flocks = []
    current_flocks = {}
    
    for t in time_steps:
        # Get positions of all objects at this time
        positions = {}
        for config in selected_configs:
            for obj_id in selected_objects:
                obj_data = df[(df['obj'] == obj_id) & 
                            (df['config_source'] == config) & 
                            (df['tst'] == t)]
                if len(obj_data) > 0:
                    positions[(config, obj_id)] = obj_data[['x', 'y']].values[0]
        
        if len(positions) < 2:
            continue
        
        # Find clusters at this time step using distance threshold
        obj_keys = list(positions.keys())
        coords = np.array([positions[k] for k in obj_keys])
        
        # Simple distance-based clustering
        groups = []
        used = set()
        
        for i, key1 in enumerate(obj_keys):
            if key1 in used:
                continue
            group = [key1]
            used.add(key1)
            
            for j, key2 in enumerate(obj_keys[i+1:], i+1):
                if key2 in used:
                    continue
                dist = euclidean(coords[i], coords[j])
                if dist <= distance_threshold:
                    group.append(key2)
                    used.add(key2)
            
            if len(group) >= 2:
                groups.append((t, frozenset(group)))
        
        # Track persistent groups
        for t, group in groups:
            group_id = None
            # Check if this group continues from a previous flock
            for flock_id, flock_data in current_flocks.items():
                if group == flock_data['members']:
                    group_id = flock_id
                    flock_data['end_time'] = t
                    break
            
            if group_id is None:
                # New flock
                flock_id = len(flocks)
                current_flocks[flock_id] = {
                    'members': group,
                    'start_time': t,
                    'end_time': t
                }
    
    # Filter flocks by minimum duration
    valid_flocks = []
    for flock_data in current_flocks.values():
        duration = flock_data['end_time'] - flock_data['start_time']
        if duration >= min_duration:
            valid_flocks.append(flock_data)
    
    return valid_flocks

def calculate_speed_trajectory(df, obj_id, config, start_time, end_time):
    """Calculate speed at each point in a trajectory"""
    obj_data = df[(df['obj'] == obj_id) & 
                  (df['config_source'] == config) &
                  (df['tst'] >= start_time) & 
                  (df['tst'] <= end_time)].sort_values('tst')
    
    if len(obj_data) < 2:
        return None
    
    coords = obj_data[['x', 'y']].values
    times = obj_data['tst'].values
    
    distances = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))
    time_diffs = np.diff(times)
    
    speeds = np.zeros(len(obj_data))
    speeds[1:] = distances / time_diffs
    
    return speeds

def grid_based_clustering(df, selected_configs, selected_objects, start_time, end_time, grid_size):
    """Cluster trajectories based on which grid cells they pass through"""
    from collections import defaultdict
    
    # Determine grid bounds
    x_min, x_max = df['x'].min(), df['x'].max()
    y_min, y_max = df['y'].min(), df['y'].max()
    
    # Create grid
    trajectory_grids = {}
    
    for config in selected_configs:
        for obj_id in selected_objects:
            obj_data = df[(df['obj'] == obj_id) & 
                        (df['config_source'] == config) &
                        (df['tst'] >= start_time) & 
                        (df['tst'] <= end_time)]
            
            if len(obj_data) < 2:
                continue
            
            # Determine which cells this trajectory passes through
            cells = set()
            for _, row in obj_data.iterrows():
                cell_x = int((row['x'] - x_min) / grid_size)
                cell_y = int((row['y'] - y_min) / grid_size)
                cells.add((cell_x, cell_y))
            
            trajectory_grids[(config, obj_id)] = cells
    
    return trajectory_grids

# Create heatmap
def create_heatmap(df):
    """Create pass heatmap using sender_id and receiver_id"""
    # Check if required columns exist
    if 'sender_id' not in df.columns or 'receiver_id' not in df.columns:
        st.error("CSV file must contain 'sender_id' and 'receiver_id' columns for heatmap.")
        return None
    
    # Create pass matrix
    pass_matrix = df.groupby(['sender_id', 'receiver_id']).size().reset_index(name='count')
    
    # Pivot to create matrix
    matrix = pass_matrix.pivot(index='receiver_id', columns='sender_id', values='count').fillna(0)
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix.values,
        x=matrix.columns,
        y=matrix.index,
        colorscale='Reds',
        hovertemplate='Sender: %{x}<br>Receiver: %{y}<br>Count: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Pass Frequency Heatmap',
        xaxis_title='Sender ID',
        yaxis_title='Receiver ID',
        width=800,
        height=800
    )
    
    return fig

# Main app
def main():
    st.title("Spatiotemporal Analysis and Modeling")
    
    df = st.session_state.data
    uploaded_files = None
    
    # Sidebar
    with st.sidebar:
        st.header("File Management")
        uploaded_files = st.file_uploader(
            "Upload CSV file(s)", type=['csv'], accept_multiple_files=True,
            help="Supports multiple formats:\n"
                 "• Long format: config, tst, obj, x, y\n"
                 "• Wide format: config_id, x1, y1, x2, y2, ..."
        )
        
        if uploaded_files is not None and len(uploaded_files) == 0:
            # User cleared the uploader
            if st.session_state.data is not None:
                st.session_state.data = None
                st.session_state.filename = None
                st.session_state.max_time = 0
                st.session_state.uploaded_filenames = []
                st.session_state.config_sources = []
            df = None
        elif uploaded_files:
            uploaded_names = [file.name for file in uploaded_files]
            combined_frames = []
            for file_idx, file in enumerate(uploaded_files):
                single_df = utils.load_data(file, update_state=False, show_success=False)
                if single_df is not None:
                    # Add file source information but preserve original config_source
                    file_name_base = file.name.rsplit('.', 1)[0]  # Remove .csv extension
                    single_df['file_source'] = file_name_base
                    # Prefix config_source with file name if multiple files to keep them unique
                    if len(uploaded_files) > 1:
                        single_df['config_source'] = file_name_base + "_" + single_df['config_source'].astype(str)
                    # Update config to be unique per file when multiple files
                    if len(uploaded_files) > 1:
                        single_df['config'] = single_df['config'] + (file_idx * 1000)
                        # Update rally_id to be unique across files
                        single_df['rally_id'] = single_df['rally_id'] + (file_idx * 10000)
                    combined_frames.append(single_df.copy())
            if combined_frames:
                df = pd.concat(combined_frames, ignore_index=True)
                st.session_state.data = df
                st.session_state.max_time = df['tst'].max()
                st.session_state.filename = ", ".join(uploaded_names)
                if uploaded_names != st.session_state.uploaded_filenames:
                    st.success(f"Loaded {len(uploaded_names)} file(s): {', '.join(uploaded_names)}")
                st.session_state.uploaded_filenames = uploaded_names
                st.session_state.config_sources = df['config_source'].drop_duplicates().tolist()
                
                # Initialize shared selections when new data is loaded
                config_sources = df['config_source'].drop_duplicates().tolist()
                objects = sorted(df['obj'].unique())
                st.session_state.shared_selected_configs = config_sources
                st.session_state.shared_selected_objects = objects[:min(5, len(objects))]
            else:
                st.error("No valid data found in the uploaded file(s). Please verify the format.")
                df = None
        else:
            df = st.session_state.data
        
        if df is not None:
            st.info(f"Current file(s): {st.session_state.filename}")
            
            st.header("Court Type")
            court_type = st.radio(
                "Select court type",
                ["Football", "Tennis"],
                index=0 if st.session_state.court_type == 'Football' else 1
            )
            st.session_state.court_type = court_type
            
            # --- CENTRALIZED SELECTION PANEL ---
            st.header("Data Selection")
            st.markdown("**Manage your selections here** - these selections apply to all analysis methods.")
            
            config_sources = df['config_source'].drop_duplicates().tolist()
            objects = sorted(df['obj'].unique())
            
            # Configuration selection
            st.subheader("Configurations (Rallies)")
            selected_configs = st.multiselect(
                "Select configurations to analyze",
                config_sources,
                default=st.session_state.shared_selected_configs,
                key="sidebar_configs",
                help="These configurations will be used across all analysis methods"
            )
            st.session_state.shared_selected_configs = selected_configs
            
            # Object selection
            st.subheader("Objects (Players/Entities)")
            selected_objects = st.multiselect(
                "Select objects to analyze",
                objects,
                default=st.session_state.shared_selected_objects,
                key="sidebar_objects",
                help="These objects will be used across all analysis methods"
            )
            st.session_state.shared_selected_objects = selected_objects
            
            # Display current selection summary
            with st.expander("Current Selection Summary", expanded=False):
                st.write(f"**Configurations:** {len(selected_configs)} of {len(config_sources)} selected")
                if selected_configs:
                    st.write(", ".join(map(str, selected_configs)))
                st.write(f"**Objects:** {len(selected_objects)} of {len(objects)} selected")
                if selected_objects:
                    st.write(", ".join(map(str, selected_objects)))
            
            st.divider()
            # --- END CENTRALIZED SELECTION PANEL ---
            
            st.header("Analysis Method")
            analysis_method = st.selectbox(
                "Select method",
                ["Visual Exploration", "Clustering", "Association Rules", "Sequence Analysis", "PDP Analysis", "Outlier Detection", "Heat Maps", "Extra"]
            )
    
    # Main content
    if df is None:
        st.info("Please upload a CSV file to begin.")
        st.markdown("""
        ### Expected CSV Formats
        
        **Format 1: Multiple files (each file = one configuration)**
        
        With header (5 columns):
        ```csv
        constant,timestamp,ID,x,y
        0,0,0,4.79,0.23
        0,1,0,3.76,17.73
        ...
        ```
        
        With header (6 columns, config name optional):
        ```csv
        constant,timestamp,ID,x,y,config_name
        0,0,0,4.79,0.23,Rally1
        0,1,0,3.76,17.73,Rally1
        ...
        ```
        
        Without header (5 or 6 columns):
        ```csv
        0,0,0,64.78,18.53
        0,1,0,54.26,20.68
        ...
        ```
        
        **Format 2: Single file with multiple configurations**
        
        Without header (6 columns with config names):
        ```csv
        0,0,0,4.79,0.23,Rally1
        0,1,0,3.76,17.73,Rally1
        1,0,0,5.12,0.45,Rally2
        1,1,0,4.23,18.12,Rally2
        ...
        ```
        
        Without header (5 columns, auto-named):
        ```csv
        0,0,0,4.79,0.23
        0,1,0,3.76,17.73
        1,0,0,5.12,0.45
        1,1,0,4.23,18.12
        ...
        ```
        
        **Columns:**
        - Column 0: Configuration number (same value in Format 1, different values in Format 2)
        - Column 1: Timestamp
        - Column 2: Object ID
        - Column 3: x coordinate
        - Column 4: y coordinate
        - Column 5 (optional): Configuration name
        
        **Coordinates:**
          - **Football**: 0-110m × 0-72m
          - **Tennis**: 0-10.97m × 0-23.77m
        
        **For heat maps:**
        ```csv
        pass_id,sender_id,receiver_id
        0,13,17
        1,17,18
        ...
        ```
        """)
        return
    
    # Analysis-specific interface
    if analysis_method == "Visual Exploration":
        st.header("Visual Exploration")
        
        st.info("""
        **Explore your trajectory data visually with interactive plots:**
        - **Static Trajectories:** View complete trajectory paths
        - **Animated Trajectories:** Watch movement over time
        - **Time Point View:** Examine trajectories at specific moments
        - **Average Positions:** Calculate and visualize mean positions
        
        **Tip:** Use the sidebar to select which configurations and objects to analyze.
        """)
        
        # Use selections from sidebar
        selected_configs = st.session_state.shared_selected_configs
        selected_objects = st.session_state.shared_selected_objects
        
        # Time range
        min_time = df['tst'].min()
        max_time = df['tst'].max()
        
        st.markdown("---")
        st.subheader("Time Range Settings")
        
        col3, col4 = st.columns(2)
        
        with col3:
            start_time = st.number_input(
                "Start time",
                min_value=float(min_time),
                max_value=float(max_time),
                value=float(min_time),
                step=0.01,
                format="%.2f",
                key="visual_start"
            )
        
        with col4:
            end_time = st.number_input(
                "End time",
                min_value=float(min_time),
                max_value=float(max_time),
                value=float(max_time),
                step=0.01,
                format="%.2f",
                key="visual_end"
            )
        
        # Aggregation settings
        col5, col6 = st.columns(2)
        
        with col5:
            aggregation_type = st.selectbox(
                "Aggregation type",
                ["none", "mean", "median"],
                key="visual_agg_type"
            )
        
        with col6:
            temporal_resolution = st.number_input(
                "Temporal resolution (s)",
                min_value=0.1,
                value=1.0,
                step=0.1,
                format="%.2f",
                key="visual_temp_res"
            )
        
        if not selected_configs or not selected_objects:
            st.warning("Please select at least one configuration and one object.")
        else:
            st.markdown("---")
            st.subheader("Visualization Types")
            
            # Create tabs for different visualization types
            viz_tabs = st.tabs(["Static Trajectories", "Animated Trajectories", "Time Point View", "Average Positions"])
            
            with viz_tabs[0]:
                st.markdown("### Static Trajectory View")
                st.info("Shows complete trajectory paths for selected objects and configurations.")
                
                try:
                    fig = visualize_static(
                        df, selected_configs, selected_objects,
                        start_time, end_time,
                        aggregation_type, temporal_resolution,
                        False, court_type  # translate_to_center set to False
                    )
                    render_interactive_chart(fig)
                except Exception as e:
                    st.error(f"Error creating static visualization: {str(e)}")
            
            with viz_tabs[1]:
                st.markdown("### Animated Trajectory View")
                st.info("Watch trajectories evolve over time with smooth animation.")
                
                # Add animation controls in two columns
                anim_col1, anim_col2 = st.columns(2)
                
                with anim_col1:
                    animation_speed = st.slider(
                        "Animation Speed (ms per frame)",
                        min_value=50,
                        max_value=1000,
                        value=200,
                        step=50,
                        help="Lower values = faster animation, higher values = slower animation"
                    )
                
                with anim_col2:
                    use_interpolation_anim = st.checkbox(
                        "Smooth continuous movement",
                        value=False,
                        help="Interpolate between timestamps for smooth continuous motion"
                    )
                    
                    if use_interpolation_anim:
                        interpolation_steps_anim = st.slider(
                            "Interpolation detail",
                            min_value=2,
                            max_value=20,
                            value=5,
                            help="Number of intermediate points between timestamps (higher = smoother but slower)"
                        )
                    else:
                        interpolation_steps_anim = 5
                
                try:
                    fig = visualize_animated(
                        df, selected_configs, selected_objects,
                        start_time, end_time,
                        aggregation_type, temporal_resolution,
                        court_type,
                        animation_speed,
                        use_interpolation_anim,
                        interpolation_steps_anim
                    )
                    render_interactive_chart(fig)
                except Exception as e:
                    st.error(f"Error creating animated visualization: {str(e)}")
            
            with viz_tabs[2]:
                st.markdown("### Time Point View")
                st.info("Examine trajectories up to a specific point in time.")
                
                # Add time control and interpolation option
                time_col1, time_col2 = st.columns([3, 1])
                
                with time_col1:
                    current_time = st.slider(
                        "Select time point",
                        min_value=float(start_time),
                        max_value=float(end_time),
                        value=float((start_time + end_time) / 2),
                        key="visual_current_time"
                    )
                
                with time_col2:
                    use_interpolation_time = st.checkbox(
                        "Smooth movement",
                        value=False,
                        key="time_interpolation",
                        help="Interpolate for smooth continuous motion"
                    )
                    
                    if use_interpolation_time:
                        interpolation_steps_time = st.slider(
                            "Detail level",
                            min_value=2,
                            max_value=20,
                            value=5,
                            key="time_interp_steps",
                            help="Higher values = smoother motion but more computation"
                        )
                    else:
                        interpolation_steps_time = 5
                
                try:
                    fig = visualize_at_time(
                        df, selected_configs, selected_objects,
                        current_time, start_time,
                        aggregation_type, temporal_resolution,
                        court_type,
                        use_interpolation_time,
                        interpolation_steps_time
                    )
                    render_interactive_chart(fig)
                except Exception as e:
                    st.error(f"Error creating time point visualization: {str(e)}")
            
            with viz_tabs[3]:
                st.markdown("### Average Position View")
                st.info("Calculate and visualize the mean position for each object across the selected time range.")
                
                try:
                    fig = visualize_average_position(
                        df, selected_configs, selected_objects,
                        start_time, end_time,
                        court_type
                    )
                    render_interactive_chart(fig)
                except Exception as e:
                    st.error(f"Error creating average position visualization: {str(e)}")
    
    elif analysis_method == "2SA Method":
        st.header("2SA Method - Two-Step Spatial Alignment")
        
        st.info("""
        **2SA (Two-Step Spatial Alignment) Method:**
        
        This method aligns trajectories to a common reference point, allowing you to compare 
        movement patterns independently of their absolute spatial location.
        
        **Key Feature:** Trajectories are translated so they all start at the center of the court,
        making it easier to identify similar movement patterns.
        
        **Use Cases:**
        - Compare player movements from different starting positions
        - Identify common tactical patterns
        - Analyze relative movement independent of field position
        """)
        
        # Get available configurations and objects
        config_sources = df['config_source'].drop_duplicates().tolist()
        objects = sorted(df['obj'].unique())
        
        # Synchronize widget state from shared state
        valid_configs = [c for c in st.session_state.shared_selected_configs if c in config_sources]
        valid_objects = [o for o in st.session_state.shared_selected_objects if o in objects]
        
        # Initialize widget state ONLY if it doesn't exist
        if '2sa_configs' not in st.session_state:
            if valid_configs:
                st.session_state['2sa_configs'] = valid_configs
            else:
                st.session_state['2sa_configs'] = config_sources
            
        if '2sa_objects' not in st.session_state:
            if valid_objects:
                st.session_state['2sa_objects'] = valid_objects
            else:
                st.session_state['2sa_objects'] = objects[:min(5, len(objects))]
        
        # Time range
        min_time = df['tst'].min()
        max_time = df['tst'].max()
        
        st.markdown("---")
        st.subheader("Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_configs = st.multiselect(
                "Select configuration(s)",
                config_sources,
                key="2sa_configs"
            )
            # Update shared state after user changes selection
            st.session_state.shared_selected_configs = selected_configs
        
        with col2:
            selected_objects = st.multiselect(
                "Select object(s)",
                objects,
                key="2sa_objects"
            )
            # Update shared state after user changes selection
            st.session_state.shared_selected_objects = selected_objects
        
        col3, col4 = st.columns(2)
        
        with col3:
            start_time = st.number_input(
                "Start time",
                min_value=float(min_time),
                max_value=float(max_time),
                value=float(min_time),
                step=0.01,
                format="%.2f",
                key="2sa_start"
            )
        
        with col4:
            end_time = st.number_input(
                "End time",
                min_value=float(min_time),
                max_value=float(max_time),
                value=float(max_time),
                step=0.01,
                format="%.2f",
                key="2sa_end"
            )
        
        # Aggregation settings
        col5, col6 = st.columns(2)
        
        with col5:
            aggregation_type = st.selectbox(
                "Aggregation type",
                ["none", "mean", "median"],
                key="2sa_agg_type"
            )
        
        with col6:
            temporal_resolution = st.number_input(
                "Temporal resolution (s)",
                min_value=0.1,
                value=1.0,
                step=0.1,
                format="%.2f",
                key="2sa_temp_res"
            )
        
        if not selected_configs or not selected_objects:
            st.warning("Please select at least one configuration and one object.")
        else:
            st.markdown("---")
            st.subheader("Aligned Trajectories")
            
            # Create comparison tabs
            alignment_tabs = st.tabs(["Aligned View", "Original View", "Side-by-Side Comparison"])
            
            with alignment_tabs[0]:
                st.markdown("### Center-Aligned Trajectories")
                st.info("All trajectories translated to start at the court center. This view highlights movement patterns.")
                
                try:
                    fig = visualize_static(
                        df, selected_configs, selected_objects,
                        start_time, end_time,
                        aggregation_type, temporal_resolution,
                        translate_to_center=True,  # 2SA mode ON
                        court_type=court_type
                    )
                    render_interactive_chart(fig)
                except Exception as e:
                    st.error(f"Error creating aligned visualization: {str(e)}")
            
            with alignment_tabs[1]:
                st.markdown("### Original Trajectories")
                st.info("Trajectories shown in their actual spatial positions.")
                
                try:
                    fig = visualize_static(
                        df, selected_configs, selected_objects,
                        start_time, end_time,
                        aggregation_type, temporal_resolution,
                        translate_to_center=False,  # 2SA mode OFF
                        court_type=court_type
                    )
                    render_interactive_chart(fig)
                except Exception as e:
                    st.error(f"Error creating original visualization: {str(e)}")
            
            with alignment_tabs[2]:
                st.markdown("### Side-by-Side Comparison")
                st.info("Compare aligned vs. original trajectories.")
                
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.markdown("**Center-Aligned**")
                    try:
                        fig_aligned = visualize_static(
                            df, selected_configs, selected_objects,
                            start_time, end_time,
                            aggregation_type, temporal_resolution,
                            translate_to_center=True,
                            court_type=court_type
                        )
                        # Make figure smaller for side-by-side
                        fig_aligned.update_layout(height=400)
                        render_interactive_chart(fig_aligned)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                
                with col_right:
                    st.markdown("**Original Position**")
                    try:
                        fig_original = visualize_static(
                            df, selected_configs, selected_objects,
                            start_time, end_time,
                            aggregation_type, temporal_resolution,
                            translate_to_center=False,
                            court_type=court_type
                        )
                        # Make figure smaller for side-by-side
                        fig_original.update_layout(height=400)
                        render_interactive_chart(fig_original)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            st.markdown("---")
            st.success("2SA analysis complete! Use the tabs above to compare aligned and original trajectories.")
    
    elif analysis_method == "Association Rules":
        # Call the modular association rules function
        association_rules.render_association_rules_section(
            data=st.session_state.data,
            selected_configs=st.session_state.shared_selected_configs,
            selected_objects=st.session_state.shared_selected_objects,
            create_spatial_grid_func=sequence_analysis.create_spatial_grid
        )
    
    elif analysis_method == "Sequence Analysis":
        st.header("Sequence Analysis")
        
        st.info("""
        **Translate trajectories to symbolic sequences for pattern mining and comparison:**
        - **Spatial Discretization:** Divide court into zones (A, B, C, ...)
        - **Sequence Comparison:** Edit distances and alignment (global/local)
        - **Pattern Discovery:** Find common sub-patterns across rallies
        """)
        
        # Show preview of grid concept
        with st.expander("How does spatial discretization work?", expanded=False):
            st.markdown("""
            **Spatial discretization converts continuous coordinates into discrete zone labels:**
            
            1. The court is divided into a grid of M×N zones
            2. Each zone gets a letter label (A, B, C, ...) assigned left-to-right, top-to-bottom
            3. Every trajectory point (x, y) is mapped to its zone
            4. The sequence of zones visited becomes a symbolic string
            
            **Example with 3×5 grid (15 zones):**
            - Trajectory: (2.1, 1.3) → (5.4, 2.8) → (11.2, 5.1) → ...
            - Zone mapping: A → B → H → ...
            - Compressed sequence: A B H ...
            
            **Adjust the grid resolution below** to change the granularity of spatial encoding.
            Finer grids capture more spatial detail, while coarser grids provide a more abstract representation.
            """)
        
        # Get available configurations and objects
        config_sources = df['config_source'].drop_duplicates().tolist()
        objects = sorted(df['obj'].unique())
        
        # Time range
        min_time = df['tst'].min()
        max_time = df['tst'].max()
        
        st.markdown("---")
        st.subheader("Sequence Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Grid Resolution**")
            grid_rows = st.slider("Grid rows", 2, 10, 3, key="seq_grid_rows", 
                                 help="Rows for the court area (buffer zones added automatically)")
            grid_cols = st.slider("Grid columns", 2, 10, 5, key="seq_grid_cols",
                                 help="Columns for the court area (buffer zones added automatically)")
            
            # Calculate actual grid with buffers
            actual_rows = grid_rows + 2  # +1 top buffer, +1 bottom buffer
            actual_cols = grid_cols + 2  # +1 left buffer, +1 right buffer
            total_zones = actual_rows * actual_cols
            
            st.caption(f"Court zones: {grid_rows} × {grid_cols} = {grid_rows * grid_cols}")
            st.caption(f"Total zones (with buffer): {actual_rows} × {actual_cols} = {total_zones}")
        
        # Spatial Grid Visualization - show immediately so users can see the effect of their grid choices
        st.markdown("---")
        st.subheader("Spatial Grid Visualization")
        
        st.info("""
        **Understanding the Grid:**
        - The **tennis field in the broad sense** (light gray zones) extends beyond the court boundaries to capture out-of-bounds positions
        - All zones are labeled **A, B, C,** etc. (row by row from bottom to top, filling each row left to right)
        - Each trajectory position is mapped to the zone it falls in
        - Adjust the grid resolution sliders above to see how it affects the zone layout
        """)
        
        # Create grid info for visualization
        grid_info = sequence_analysis.create_spatial_grid(
            st.session_state.court_type,
            grid_rows,
            grid_cols
        )
        
        # Show grid overlay on court
        fig_grid = create_pitch_figure(st.session_state.court_type)
        
        # Draw grid lines
        x_bins = grid_info['x_bins']
        y_bins = grid_info['y_bins']
        court_width = grid_info['court_width']
        court_height = grid_info['court_height']
        buffer = grid_info['buffer']
        
        # Add buffer zone background (light gray)
        fig_grid.add_shape(
            type="rect",
            x0=-buffer, y0=-buffer,
            x1=court_width + buffer, y1=court_height + buffer,
            fillcolor='rgba(200, 200, 200, 0.3)',
            line=dict(color='rgba(150, 150, 150, 0.5)', width=2),
            layer="below"
        )
        
        # Add grid lines on top of court (using shapes for better visibility)
        # Vertical lines
        for x in x_bins:
            fig_grid.add_shape(
                type="line",
                x0=x, y0=y_bins[0],
                x1=x, y1=y_bins[-1],
                line=dict(color='rgba(255, 0, 0, 0.6)', width=3, dash='dash'),
                layer="above"
            )
        
        # Horizontal lines
        for y in y_bins:
            fig_grid.add_shape(
                type="line",
                x0=x_bins[0], y0=y,
                x1=x_bins[-1], y1=y,
                line=dict(color='rgba(255, 0, 0, 0.6)', width=3, dash='dash'),
                layer="above"
            )
        
        # Add zone labels with background
        actual_rows_viz = grid_info['grid_rows']
        actual_cols_viz = grid_info['grid_cols']
        
        for row in range(actual_rows_viz):
            for col in range(actual_cols_viz):
                zone_idx = row * actual_cols_viz + col
                zone_label = grid_info['zone_labels'][zone_idx]
                
                x_center = (x_bins[col] + x_bins[col + 1]) / 2
                y_center = (y_bins[row] + y_bins[row + 1]) / 2
                
                # Determine if this is a buffer zone or court zone
                is_buffer = (col == 0 or col == actual_cols_viz - 1 or 
                           row == 0 or row == actual_rows_viz - 1)
                
                # Add zone label with uniform styling (no distinction between buffer and court)
                # All zones get the same appearance
                bgcolor = 'rgba(0, 0, 0, 0.7)'
                bordercolor = 'black'
                font_size = 18
                
                fig_grid.add_annotation(
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
        
        fig_grid.update_layout(
            height=700,
            xaxis=dict(range=[x_bins[0] - 0.5, x_bins[-1] + 0.5]),
            yaxis=dict(range=[y_bins[0] - 0.5, y_bins[-1] + 0.5])
        )
        
        render_interactive_chart(fig_grid)
        
        with col2:
            st.write("**Compression**")
            compress_runs = st.checkbox(
                "Run-length compression",
                value=True,
                help="AAABBB → AB",
                key="seq_compress"
            )
            
            sequence_type = st.radio(
                "Sequence type",
                ["Per-entity", "Multi-entity"],
                key="seq_type"
            )
            
            # Add detailed explanation in an expander
            with st.expander("Sequence Type Explanation"):
                st.markdown("""
                **Per-entity (individual sequences)**
                - Each player/object gets their own separate sequence
                - Ideal for: individual movement patterns, player-specific analyses
                - Example: 
                  - Player 1: `A1 → B2 → C3 → D4`
                  - Player 2: `E5 → F4 → G3 → D2`
                
                **Multi-entity (combined sequences)**
                - Combines all objects per timestamp into one token
                - Shows interactions and joint positions
                - Calculation: At each moment, the zones of all active objects are merged
                - Example at timestamp t=1: 
                  - Player 1 in zone B2, Player 2 in zone D4 → Token: `B2;D4`
                - Use for: teamwork analysis, position combinations, joint patterns
                
                **💡 Tip**: Start with Per-entity for individual analyses, use Multi-entity to discover interactions.
                """)
        
        # Use selections from sidebar
        selected_configs = st.session_state.shared_selected_configs
        selected_objects = st.session_state.shared_selected_objects
        
        st.markdown("---")
        
        col3, col4 = st.columns(2)
        
        with col3:
            start_time = st.number_input(
                "Start time",
                min_value=float(min_time),
                max_value=float(max_time),
                value=float(min_time),
                step=0.01,
                format="%.2f",
                key="seq_start"
            )
        
        with col4:
            end_time = st.number_input(
                "End time",
                min_value=float(min_time),
                max_value=float(max_time),
                value=float(max_time),
                step=0.01,
                format="%.2f",
                key="seq_end"
            )
        
        if not selected_configs:
            st.warning("Please select at least one configuration.")
        elif not selected_objects:
            st.warning("Please select at least one object.")
        else:
            # Create grid
            grid_info = sequence_analysis.create_spatial_grid(
                court_type=st.session_state.court_type,
                grid_rows=grid_rows,
                grid_cols=grid_cols
            )
            
            # Build sequences
            st.markdown("---")
            st.subheader("Generated Sequences")
            
            sequences_data = []
            
            # Store both the list (for processing) and string (for display)
            raw_sequences = []  # Store list form
            
            if sequence_type == "Per-entity":
                # Build per-entity sequences (event-based: one token per data point)
                for config in selected_configs:
                    for obj_id in selected_objects:
                        seq = sequence_analysis.build_event_based_sequence(
                            df, config, obj_id, start_time, end_time,
                            grid_info, compress=compress_runs
                        )
                        
                        if seq:
                            raw_sequences.append(seq)  # Store list
                            sequences_data.append({
                                'ID': f"{config}-Obj{obj_id}",
                                'Config': config,
                                'Object': obj_id,
                                'Sequence': '-'.join(seq),  # Display with delimiter
                                'Length': len(seq)
                            })
            else:
                # Multi-entity sequences (event-based)
                for config in selected_configs:
                    seq = sequence_analysis.build_multi_entity_sequence(
                        df, config, selected_objects, start_time, end_time,
                        grid_info, compress=compress_runs
                    )
                    if seq:
                        raw_sequences.append(seq)  # Store list
                        sequences_data.append({
                            'ID': config,
                            'Config': config,
                            'Object': 'Multi',
                            'Sequence': '; '.join(seq),  # Display with delimiter (multi-entity uses semicolons)
                            'Length': len(seq)
                        })
            
            if not sequences_data:
                st.warning("No sequences generated. Check your data and time range.")
            else:
                # Display sequences
                seq_df = pd.DataFrame(sequences_data)
                st.dataframe(seq_df, use_container_width=True, height=300)
                
                # Export sequences
                csv_export = seq_df.to_csv(index=False)
                st.download_button(
                    "Download sequences as CSV",
                    csv_export,
                    f"sequences_{grid_rows}x{grid_cols}.csv",
                    "text/csv"
                )
                
                # Trajectory Visualization on Tennis Court
                st.markdown("---")
                st.subheader("Trajectory Visualization on Tennis Court")
                st.markdown("See the actual movement patterns of the trajectories used to generate sequences")
                
                # Let user select which trajectories to visualize
                viz_mode = st.radio(
                    "Visualization mode:",
                    ["Show All", "Select Specific"],
                    horizontal=True,
                    key="seq_viz_mode"
                )
                
                trajectories_to_plot = []
                colors_list = []
                labels_list = []
                
                # Color palette for different trajectories
                color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                
                if viz_mode == "Show All":
                    max_trajs_value = min(20, len(seq_df))
                    default_trajs = min(10, len(seq_df))
                    
                    if max_trajs_value > 1:
                        max_trajs = st.slider("Max trajectories to show:", 1, max_trajs_value, default_trajs, key="seq_max_trajs")
                    else:
                        max_trajs = 1
                        st.info("Only 1 trajectory available to show.")
                    
                    for idx, row in seq_df.head(max_trajs).iterrows():
                        config_source = row['Config']
                        obj_id = row['Object']
                        
                        if obj_id != 'Multi':  # Skip multi-entity sequences
                            # Convert obj_id to int if it's stored as string
                            if isinstance(obj_id, str) and obj_id.isdigit():
                                obj_id = int(obj_id)
                            
                            # Get trajectory data - ensure config_source matches
                            traj_data = df[
                                (df['config_source'] == str(config_source)) &
                                (df['obj'] == obj_id) &
                                (df['tst'] >= start_time) &
                                (df['tst'] <= end_time)
                            ].copy()
                            
                            if not traj_data.empty and 'tst' in traj_data.columns:
                                traj_data = traj_data.sort_values('tst')
                                trajectories_to_plot.append(traj_data)
                                colors_list.append(color_palette[idx % len(color_palette)])
                                labels_list.append(f"{row['ID']} ({row['Sequence'][:30]}...)")
                
                else:  # Select Specific
                    # Filter out multi-entity sequences for selection
                    selectable_seqs = seq_df[seq_df['Object'] != 'Multi']
                    
                    if len(selectable_seqs) > 0:
                        selected_ids = st.multiselect(
                            "Select sequences to visualize:",
                            selectable_seqs['ID'].tolist(),
                            default=selectable_seqs['ID'].tolist()[:min(5, len(selectable_seqs))],
                            key="seq_selected_ids"
                        )
                        
                        for idx, traj_id in enumerate(selected_ids):
                            row = seq_df[seq_df['ID'] == traj_id].iloc[0]
                            config_source = row['Config']
                            obj_id = row['Object']
                            
                            # Convert obj_id to int if it's stored as string
                            if isinstance(obj_id, str) and obj_id.isdigit():
                                obj_id = int(obj_id)
                            
                            # Get trajectory data - ensure config_source matches
                            traj_data = df[
                                (df['config_source'] == str(config_source)) &
                                (df['obj'] == obj_id) &
                                (df['tst'] >= start_time) &
                                (df['tst'] <= end_time)
                            ].copy()
                            
                            if not traj_data.empty and 'tst' in traj_data.columns:
                                traj_data = traj_data.sort_values('tst')
                                trajectories_to_plot.append(traj_data)
                                colors_list.append(color_palette[idx % len(color_palette)])
                                labels_list.append(f"{row['ID']} ({row['Sequence'][:30]}...)")
                    else:
                        st.info("No per-entity sequences available for visualization (only multi-entity sequences)")
                
                # Plot trajectories on tennis court
                if trajectories_to_plot:
                    fig_court = create_pitch_figure(st.session_state.court_type)
                    
                    # Add each trajectory
                    for traj_data, color, label in zip(trajectories_to_plot, colors_list, labels_list):
                        # Add trajectory line
                        fig_court.add_trace(go.Scatter(
                            x=traj_data['x'],
                            y=traj_data['y'],
                            mode='lines+markers',
                            name=label,
                            line=dict(color=color, width=2),
                            marker=dict(size=4, color=color),
                            legendgroup=label,
                            hovertemplate=f'<b>{label}</b><br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>'
                        ))
                        
                        # Mark start and end
                        fig_court.add_trace(go.Scatter(
                            x=[traj_data['x'].iloc[0]],
                            y=[traj_data['y'].iloc[0]],
                            mode='markers',
                            name=f'{label} (start)',
                            marker=dict(size=12, color=color, symbol='circle', line=dict(width=2, color='white')),
                            legendgroup=label,
                            showlegend=False,
                            hovertemplate=f'<b>{label} START</b><extra></extra>'
                        ))
                        
                        fig_court.add_trace(go.Scatter(
                            x=[traj_data['x'].iloc[-1]],
                            y=[traj_data['y'].iloc[-1]],
                            mode='markers',
                            name=f'{label} (end)',
                            marker=dict(size=12, color=color, symbol='square', line=dict(width=2, color='white')),
                            legendgroup=label,
                            showlegend=False,
                            hovertemplate=f'<b>{label} END</b><extra></extra>'
                        ))
                    
                    fig_court.update_layout(
                        title=f"Trajectories on Tennis Court (Sequence Analysis)",
                        height=600,
                        showlegend=True,
                        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                    )
                    
                    render_interactive_chart(fig_court, "Trajectories used to generate sequences")
                    
                    st.caption("""
                    **Legend**: Each color represents a different trajectory | 
                    ● Circle = Start | ■ Square = End | 
                    Hover over points to see details
                    """)
                else:
                    st.info("No trajectories selected for visualization")
                
                # Create tabs for analysis
                st.markdown("---")
                seq_tab1, seq_tab2, seq_tab3 = st.tabs([
                    "Distance Matrix",
                    "Pairwise Alignment",
                    "N-gram Patterns"
                ])
                
                with seq_tab1:
                    st.subheader("Distance Matrix & Clustering")
                    
                    st.write("**Distance Metric**")
                    dist_method = st.radio(
                        "Select metric",
                        ["Levenshtein (edit distance)", "Normalized Levenshtein"],
                        key="seq_dist_method"
                    )
                    
                    # Compute distance matrix using raw sequences (lists)
                    method = 'levenshtein' if 'edit' in dist_method else 'normalized_levenshtein'
                    dist_matrix = sequence_analysis.compute_sequence_distance_matrix(raw_sequences, method=method)
                    
                    # Display matrix
                    fig_dist = go.Figure(data=go.Heatmap(
                        z=dist_matrix,
                        x=seq_df['ID'],
                        y=seq_df['ID'],
                        colorscale='Reds',
                        text=np.round(dist_matrix, 2),
                        texttemplate='%{text}',
                        textfont={"size": 10},
                        hovertemplate='%{y} → %{x}<br>Distance: %{z:.2f}<extra></extra>'
                    ))
                    
                    fig_dist.update_layout(
                        title=f"Sequence Distance Matrix ({dist_method})",
                        xaxis_title="Sequence",
                        yaxis_title="Sequence",
                        height=500
                    )
                    
                    render_interactive_chart(fig_dist, "Darker red = more different sequences")
                    
                    # Clustering
                    if len(raw_sequences) >= 2:
                        st.markdown('---')
                        
                        # ========================================
                        # Hierarchical Clustering - Dendrogram & Cluster Assignment
                        # ========================================
                        st.subheader("Hierarchical Clustering - Dendrogram & Cluster Assignment")
                        
                        st.info("""
                        **Dendrogram Visualization**: Shows the hierarchical structure of sequence clustering.
                        - Each leaf represents a sequence
                        - Height indicates dissimilarity between merged clusters
                        - Use the slider to cut the dendrogram at different heights (select number of clusters)
                        """)
                        
                        # Create linkage matrix for hierarchical clustering
                        # Convert square distance matrix to condensed form
                        from scipy.spatial.distance import squareform
                        condensed_dist = squareform(dist_matrix, checks=False)
                        linkage_matrix = linkage(condensed_dist, method='ward')
                        
                        # Create dendrogram visualization
                        st.markdown("#### Dendrogram")
                        
                        # Use scipy to create dendrogram data
                        from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
                        dendro_data = scipy_dendrogram(
                            linkage_matrix,
                            labels=[f"S{sid}" for sid in seq_df['ID']],
                            no_plot=True
                        )
                        
                        # Create plotly dendrogram
                        icoord = np.array(dendro_data['icoord'])
                        dcoord = np.array(dendro_data['dcoord'])
                        colors = dendro_data['color_list']
                        labels = dendro_data['ivl']
                        
                        # Convert matplotlib color codes to Plotly-compatible colors
                        color_map = {
                            'C0': '#1f77b4', 'C1': '#ff7f0e', 'C2': '#2ca02c', 'C3': '#d62728',
                            'C4': '#9467bd', 'C5': '#8c564b', 'C6': '#e377c2', 'C7': '#7f7f7f',
                            'C8': '#bcbd22', 'C9': '#17becf', 'b': 'blue', 'g': 'green',
                            'r': 'red', 'c': 'cyan', 'm': 'magenta', 'y': 'yellow', 'k': 'black'
                        }
                        plotly_colors = [color_map.get(c, c) for c in colors]
                        
                        fig_dendro = go.Figure()
                        
                        # Add dendrogram lines
                        for i, (xi, yi) in enumerate(zip(icoord, dcoord)):
                            fig_dendro.add_trace(go.Scatter(
                                x=xi,
                                y=yi,
                                mode='lines',
                                line=dict(color=plotly_colors[i], width=2),
                                hoverinfo='skip',
                                showlegend=False
                            ))
                        
                        # Add labels at bottom
                        n_leaves = len(labels)
                        x_positions = [5 + i * 10 for i in range(n_leaves)]
                        
                        fig_dendro.update_layout(
                            title="Hierarchical Clustering Dendrogram (Ward Linkage)",
                            xaxis=dict(
                                title="Sequence",
                                tickmode='array',
                                tickvals=x_positions,
                                ticktext=labels,
                                tickangle=-45
                            ),
                            yaxis=dict(title="Distance"),
                            height=500,
                            hovermode='closest',
                            plot_bgcolor='white',
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig_dendro, use_container_width=True)
                        
                        st.markdown("---")
                        st.markdown("#### Cluster Assignment")
                        
                        # Cluster selection controls
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Number of clusters slider
                            n_sequences = len(raw_sequences)
                            max_clusters = min(10, n_sequences - 1)
                            
                            if max_clusters > 2:
                                n_clusters = st.slider(
                                    "Number of clusters",
                                    min_value=2,
                                    max_value=max_clusters,
                                    value=min(3, max_clusters),
                                    help="Slide to select how many clusters to create",
                                    key="seq_clusters"
                                )
                            else:
                                n_clusters = 2
                                st.info(f"Using {n_clusters} clusters (only {n_sequences} sequences available)")
                        
                        with col2:
                            # Auto-detect optimal clusters button
                            if st.button("Auto-detect Optimal Clusters", help="Use elbow method to recommend optimal number of clusters.", key="seq_auto_clusters"):
                                with st.spinner("Detecting optimal number of clusters..."):
                                    optimal_k, plot_data = clustering.detect_optimal_clusters(dist_matrix, return_plot_data=True)
                                    if optimal_k is not None:
                                        st.success(f"Recommended number of clusters: **{optimal_k}**")
                                        
                                        # Display elbow plot
                                        fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
                                        
                                        fig.add_trace(go.Scatter(x=plot_data["k_values"], y=plot_data["inertias"], mode="lines+markers",
                                            name="Inertia", line=dict(color="blue", width=2), marker=dict(size=8)), secondary_y=False)
                                        
                                        fig.add_trace(go.Scatter(x=plot_data["k_values"], y=plot_data["silhouette_scores"],
                                            mode="lines+markers", name="Silhouette Score", line=dict(color="green", width=2),
                                            marker=dict(size=8)), secondary_y=True)
                                        
                                        fig.add_vline(x=optimal_k, line=dict(color="red", width=2, dash="dash"),
                                            annotation_text=f"Optimal k={optimal_k}", annotation_position="top")
                                        
                                        fig.update_xaxes(title_text="Number of Clusters (k)")
                                        fig.update_yaxes(title_text="Inertia", secondary_y=False)
                                        fig.update_yaxes(title_text="Silhouette Score", secondary_y=True)
                                        fig.update_layout(title="Elbow Plot", hovermode="x unified", height=400)
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.warning("Could not automatically detect optimal clusters. Please select manually.")
                        
                        # Assign clusters based on selected number
                        from scipy.cluster.hierarchy import fcluster
                        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
                        
                        # Display cluster assignment summary
                        st.markdown(f"**Cluster Assignment Summary** ({n_clusters} clusters)")
                        
                        # Create a dataframe showing cluster assignments
                        seq_df_clustered = seq_df.copy()
                        seq_df_clustered['Cluster'] = cluster_labels
                        
                        # Count sequences per cluster
                        cluster_counts = seq_df_clustered['Cluster'].value_counts().sort_index()
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Sequences per Cluster:**")
                            for cluster_id in sorted(cluster_counts.index):
                                count = cluster_counts[cluster_id]
                                st.write(f"• Cluster {cluster_id}: **{count}** sequences")
                        
                        with col2:
                            # Show cluster assignments table
                            st.markdown("**Cluster Assignments:**")
                            st.dataframe(
                                seq_df_clustered[['ID', 'Cluster', 'Length']].sort_values('Cluster'),
                                height=min(300, len(seq_df_clustered) * 35 + 38),
                                use_container_width=True
                            )
                        
                        # Cluster statistics
                        st.markdown("**Cluster Statistics**")
                        cluster_stats = seq_df_clustered.groupby('Cluster').agg({
                            'ID': 'count',
                            'Length': ['mean', 'std']
                        }).round(2)
                        cluster_stats.columns = ['Count', 'Avg Length', 'Std Length']
                        st.dataframe(cluster_stats, use_container_width=True)
                        
                        st.markdown('---')
                        st.success(f"Successfully assigned {n_sequences} sequences into {n_clusters} clusters using Ward linkage!")
                        
                        # ===========================
                        # TRAJECTORY VISUALIZATION BY CLUSTER
                        # ===========================
                        st.markdown("---")
                        st.markdown("### Trajectories Colored by Cluster")
                        
                        st.info("""
                        **Cluster Visualization**: View trajectories on the court, colored by their cluster assignment.
                        This helps you understand which trajectories are grouped together spatially.
                        """)
                        
                        # Create visualization of trajectories colored by cluster
                        fig_clusters = create_pitch_figure(st.session_state.court_type)
                        
                        # Define color palette for clusters
                        import plotly.express as px
                        cluster_colors = px.colors.qualitative.Set3[:n_clusters] if n_clusters <= len(px.colors.qualitative.Set3) else \
                                        px.colors.sample_colorscale("rainbow", [i/(n_clusters-1) for i in range(n_clusters)])
                        
                        # Plot trajectories grouped by cluster
                        for cluster_id in sorted(seq_df_clustered['Cluster'].unique()):
                            cluster_seqs = seq_df_clustered[seq_df_clustered['Cluster'] == cluster_id]
                            color = cluster_colors[cluster_id - 1]  # cluster_id starts at 1
                            
                            for idx, row in cluster_seqs.iterrows():
                                config_source = row['Config']
                                obj_id = row['Object']
                                
                                # Skip multi-entity sequences
                                if obj_id == 'Multi':
                                    continue
                                
                                # Convert obj_id to int if it's stored as string
                                if isinstance(obj_id, str) and obj_id.isdigit():
                                    obj_id = int(obj_id)
                                
                                # Get trajectory data
                                traj_data = df[
                                    (df['config_source'] == str(config_source)) &
                                    (df['obj'] == obj_id) &
                                    (df['tst'] >= start_time) &
                                    (df['tst'] <= end_time)
                                ].copy()
                                
                                if not traj_data.empty and 'tst' in traj_data.columns:
                                    traj_data = traj_data.sort_values('tst')
                                    
                                    # Add trajectory line
                                    fig_clusters.add_trace(go.Scatter(
                                        x=traj_data['x'],
                                        y=traj_data['y'],
                                        mode='lines',
                                        name=f"Cluster {cluster_id}",
                                        line=dict(color=color, width=2),
                                        legendgroup=f"cluster_{cluster_id}",
                                        showlegend=bool(idx == cluster_seqs.index[0]),  # Only show legend for first trajectory in cluster
                                        hovertemplate=f'<b>Cluster {cluster_id} - Seq {row["ID"]}</b><br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>'
                                    ))
                                    
                                    # Mark start point
                                    fig_clusters.add_trace(go.Scatter(
                                        x=[traj_data['x'].iloc[0]],
                                        y=[traj_data['y'].iloc[0]],
                                        mode='markers',
                                        marker=dict(size=8, color=color, symbol='circle', line=dict(width=1, color='white')),
                                        legendgroup=f"cluster_{cluster_id}",
                                        showlegend=False,
                                        hovertemplate=f'<b>Cluster {cluster_id} - Seq {row["ID"]} START</b><extra></extra>'
                                    ))
                                    
                                    # Mark end point
                                    fig_clusters.add_trace(go.Scatter(
                                        x=[traj_data['x'].iloc[-1]],
                                        y=[traj_data['y'].iloc[-1]],
                                        mode='markers',
                                        marker=dict(size=8, color=color, symbol='square', line=dict(width=1, color='white')),
                                        legendgroup=f"cluster_{cluster_id}",
                                        showlegend=False,
                                        hovertemplate=f'<b>Cluster {cluster_id} - Seq {row["ID"]} END</b><extra></extra>'
                                    ))
                        
                        fig_clusters.update_layout(
                            title=f"Trajectories Grouped by Cluster ({n_clusters} clusters)",
                            height=600,
                            showlegend=True,
                            legend=dict(
                                yanchor="top", 
                                y=0.99, 
                                xanchor="left", 
                                x=0.01,
                                title="Clusters"
                            )
                        )
                        
                        render_interactive_chart(fig_clusters, "Trajectories colored by cluster assignment")
                        
                        st.caption("""
                        **Legend**: Each color represents a different cluster | 
                        ● Circle = Start | ■ Square = End | 
                        Click legend items to show/hide clusters | 
                        Hover over trajectories to see cluster and sequence ID
                        """)
                        
                        # ===========================
                        # ANALYSIS TOOLS
                        # ===========================
                        
                        st.markdown('---')
                        st.markdown("### Analysis Tools")
                        
                        st.info("""
                        **Advanced Analysis**: Explore cluster quality and sequence relationships
                        - **MDS Visualization**: Project high-dimensional data to 2D/3D space
                        - **Similarity Search**: Find most similar sequences to a reference
                        - **Silhouette Analysis**: Evaluate cluster quality metrics
                        """)
                        
                        # Create tabs for different analysis tools
                        analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs([
                            "MDS Visualization", 
                            "Similarity Search", 
                            "Silhouette Analysis"
                        ])
                        
                        # ===========================
                        # TAB 1: MDS VISUALIZATION
                        # ===========================
                        with analysis_tab1:
                            st.markdown("#### Multidimensional Scaling (MDS)")
                            st.markdown("Visualize sequence clusters in 2D or 3D space based on their pairwise distances.")
                            
                            # MDS dimension selection
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                mds_dims = st.radio(
                                    "Dimensions",
                                    options=[2, 3],
                                    index=0,
                                    help="Choose 2D or 3D visualization",
                                    key="seq_mds_dims"
                                )
                            
                            with col2:
                                if st.button("🎨 Generate MDS Plot", help="Click to compute and visualize MDS projection", key="seq_mds_button"):
                                    with st.spinner(f"Computing {mds_dims}D MDS projection..."):
                                        from sklearn.manifold import MDS
                                        
                                        # Compute MDS
                                        mds = MDS(n_components=mds_dims, dissimilarity='precomputed', random_state=42)
                                        mds_coords = mds.fit_transform(dist_matrix)
                                        
                                        # Calculate normalized stress
                                        from scipy.spatial.distance import pdist, squareform
                                        mds_distances = squareform(pdist(mds_coords))
                                        
                                        stress_normalized = np.sqrt(np.sum((dist_matrix - mds_distances) ** 2) / np.sum(dist_matrix ** 2))
                                        
                                        # Create color palette for clusters
                                        import plotly.express as px
                                        colors = px.colors.qualitative.Plotly[:n_clusters]
                                        
                                        # Create plotly figure
                                        if mds_dims == 2:
                                            fig_mds = go.Figure()
                                            
                                            for cluster_id in range(1, n_clusters + 1):
                                                mask = cluster_labels == cluster_id
                                                cluster_sequences = seq_df_clustered[mask]['ID'].values
                                                
                                                fig_mds.add_trace(go.Scatter(
                                                    x=mds_coords[mask, 0],
                                                    y=mds_coords[mask, 1],
                                                    mode='markers+text',
                                                    marker=dict(
                                                        size=12,
                                                        color=colors[cluster_id - 1],
                                                        line=dict(width=1, color='white')
                                                    ),
                                                    text=[f"S{sid}" for sid in cluster_sequences],
                                                    textposition="top center",
                                                    textfont=dict(size=9),
                                                    name=f"Cluster {cluster_id}",
                                                    hovertemplate='<b>Sequence %{text}</b><br>Cluster: ' + str(cluster_id) + '<extra></extra>'
                                                ))
                                            
                                            fig_mds.update_layout(
                                                title="2D MDS Projection of Sequence Clusters",
                                                xaxis_title="MDS Dimension 1",
                                                yaxis_title="MDS Dimension 2",
                                                height=600,
                                                hovermode='closest',
                                                showlegend=True
                                            )
                                            
                                        else:  # 3D
                                            fig_mds = go.Figure()
                                            
                                            for cluster_id in range(1, n_clusters + 1):
                                                mask = cluster_labels == cluster_id
                                                cluster_sequences = seq_df_clustered[mask]['ID'].values
                                                
                                                fig_mds.add_trace(go.Scatter3d(
                                                    x=mds_coords[mask, 0],
                                                    y=mds_coords[mask, 1],
                                                    z=mds_coords[mask, 2],
                                                    mode='markers+text',
                                                    marker=dict(
                                                        size=8,
                                                        color=colors[cluster_id - 1],
                                                        line=dict(width=1, color='white')
                                                    ),
                                                    text=[f"S{sid}" for sid in cluster_sequences],
                                                    textposition="top center",
                                                    textfont=dict(size=8),
                                                    name=f"Cluster {cluster_id}",
                                                    hovertemplate='<b>Sequence %{text}</b><br>Cluster: ' + str(cluster_id) + '<extra></extra>'
                                                ))
                                            
                                            fig_mds.update_layout(
                                                title="3D MDS Projection of Sequence Clusters",
                                                scene=dict(
                                                    xaxis_title="MDS Dimension 1",
                                                    yaxis_title="MDS Dimension 2",
                                                    zaxis_title="MDS Dimension 3"
                                                ),
                                                height=700,
                                                hovermode='closest',
                                                showlegend=True
                                            )
                                        
                                        st.plotly_chart(fig_mds, use_container_width=True)
                                        st.success(f"{mds_dims}D MDS projection computed successfully!")
                                        st.info(f"**Normalized Stress (Kruskal's Stress-1)**: {stress_normalized:.4f} ({stress_normalized*100:.2f}%) — Lower is better: <0.05 (5%) excellent, <0.10 (10%) good, <0.20 (20%) acceptable")
                        
                        # ===========================
                        # TAB 2: SIMILARITY SEARCH
                        # ===========================
                        with analysis_tab2:
                            st.markdown("#### Top-K Similar Sequences")
                            st.markdown("Find sequences most similar to a selected reference sequence.")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Select reference sequence
                                reference_idx = st.selectbox(
                                    "Select reference sequence",
                                    options=range(len(seq_df)),
                                    format_func=lambda i: f"Sequence {seq_df.iloc[i]['ID']} (Cluster {cluster_labels[i]})",
                                    help="Choose a sequence to find similar ones",
                                    key="seq_ref_select"
                                )
                            
                            with col2:
                                # Select number of similar sequences to show
                                k_similar = st.slider(
                                    "Number of similar sequences (K)",
                                    min_value=1,
                                    max_value=min(20, len(seq_df) - 1),
                                    value=5,
                                    help="How many similar sequences to display",
                                    key="seq_k_similar"
                                )
                            
                            if st.button("Find Similar Sequences", key="seq_find_similar"):
                                with st.spinner("Searching for similar sequences..."):
                                    # Get distances from reference sequence to all others
                                    distances = dist_matrix[reference_idx].copy()
                                    
                                    # Set distance to self as infinity to exclude it
                                    distances[reference_idx] = np.inf
                                    
                                    # Find K most similar (smallest distances)
                                    similar_indices = np.argsort(distances)[:k_similar]
                                    
                                    # Create results dataframe
                                    results_df = pd.DataFrame({
                                        'Rank': range(1, k_similar + 1),
                                        'Sequence ID': [seq_df.iloc[i]['ID'] for i in similar_indices],
                                        'Cluster': [cluster_labels[i] for i in similar_indices],
                                        'Length': [seq_df.iloc[i]['Length'] for i in similar_indices],
                                        'Distance': distances[similar_indices],
                                        'Similarity Score': 1 / (1 + distances[similar_indices])
                                    })
                                    
                                    # Display reference info
                                    ref_sid = seq_df.iloc[reference_idx]['ID']
                                    ref_cluster = cluster_labels[reference_idx]
                                    ref_length = seq_df.iloc[reference_idx]['Length']
                                    
                                    st.markdown(f"**Reference Sequence**: S{ref_sid} (Cluster {ref_cluster}, Length {ref_length})")
                                    st.markdown(f"**Top {k_similar} Most Similar Sequences:**")
                                    
                                    # Format and display results
                                    st.dataframe(
                                        results_df.style.format({
                                            'Distance': '{:.4f}',
                                            'Similarity Score': '{:.4f}'
                                        }).background_gradient(subset=['Similarity Score'], cmap='Greens'),
                                        use_container_width=True,
                                        height=min(400, len(results_df) * 35 + 38)
                                    )
                                    
                                    # Cluster distribution analysis
                                    same_cluster = sum(results_df['Cluster'] == ref_cluster)
                                    st.markdown(f"**Cluster Analysis**: {same_cluster}/{k_similar} similar sequences are in the same cluster as the reference")
                                    
                                    if same_cluster == k_similar:
                                        st.success("All similar sequences are in the same cluster - excellent clustering!")
                                    elif same_cluster >= k_similar * 0.7:
                                        st.info("Most similar sequences are in the same cluster - good clustering quality")
                                    else:
                                        st.warning("Many similar sequences are in different clusters - consider adjusting cluster count")
                        
                        # ===========================
                        # TAB 3: SILHOUETTE ANALYSIS
                        # ===========================
                        with analysis_tab3:
                            st.markdown("#### Silhouette Analysis")
                            st.markdown("Evaluate cluster quality using silhouette coefficients. Values range from -1 to 1:")
                            st.markdown("- **Close to 1**: Well-clustered, far from neighboring clusters")
                            st.markdown("- **Close to 0**: Near the decision boundary between clusters")
                            st.markdown("- **Negative**: Possibly assigned to wrong cluster")
                            
                            if st.button("Calculate Silhouette Scores", key="seq_silhouette"):
                                with st.spinner("Computing silhouette analysis..."):
                                    from sklearn.metrics import silhouette_score, silhouette_samples
                                    
                                    # Compute silhouette scores
                                    overall_score = silhouette_score(dist_matrix, cluster_labels, metric='precomputed')
                                    sample_scores = silhouette_samples(dist_matrix, cluster_labels, metric='precomputed')
                                    
                                    # Display overall score
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Overall Silhouette Score", f"{overall_score:.4f}")
                                    with col2:
                                        st.metric("Number of Clusters", n_clusters)
                                    with col3:
                                        avg_cluster_size = len(cluster_labels) / n_clusters
                                        st.metric("Avg Cluster Size", f"{avg_cluster_size:.1f}")
                                    
                                    # Quality interpretation
                                    if overall_score > 0.7:
                                        st.success("**Excellent** clustering structure!")
                                    elif overall_score > 0.5:
                                        st.success("**Good** clustering quality")
                                    elif overall_score > 0.3:
                                        st.info("**Moderate** clustering quality")
                                    else:
                                        st.warning("**Poor** clustering - consider different parameters")
                                    
                                    st.markdown("---")
                                    st.markdown("**Per-Cluster Silhouette Scores:**")
                                    
                                    # Create per-cluster analysis
                                    cluster_stats_sil = []
                                    for cluster_id in range(1, n_clusters + 1):
                                        mask = cluster_labels == cluster_id
                                        cluster_scores = sample_scores[mask]
                                        
                                        cluster_stats_sil.append({
                                            'Cluster': cluster_id,
                                            'Size': mask.sum(),
                                            'Mean Score': cluster_scores.mean(),
                                            'Min Score': cluster_scores.min(),
                                            'Max Score': cluster_scores.max(),
                                            'Std Dev': cluster_scores.std()
                                        })
                                    
                                    cluster_stats_sil_df = pd.DataFrame(cluster_stats_sil)
                                    
                                    # Display cluster statistics
                                    st.dataframe(
                                        cluster_stats_sil_df.style.format({
                                            'Mean Score': '{:.4f}',
                                            'Min Score': '{:.4f}',
                                            'Max Score': '{:.4f}',
                                            'Std Dev': '{:.4f}'
                                        }).background_gradient(subset=['Mean Score'], cmap='RdYlGn'),
                                        use_container_width=True
                                    )
                                    
                                    # Create silhouette plot
                                    import plotly.express as px
                                    colors = px.colors.qualitative.Plotly[:n_clusters]
                                    
                                    fig_silhouette = go.Figure()
                                    
                                    y_lower = 10
                                    for cluster_id in range(1, n_clusters + 1):
                                        mask = cluster_labels == cluster_id
                                        cluster_scores = sample_scores[mask]
                                        cluster_scores.sort()
                                        
                                        y_upper = y_lower + len(cluster_scores)
                                        
                                        color = colors[cluster_id - 1]
                                        fig_silhouette.add_trace(go.Bar(
                                            x=cluster_scores,
                                            y=list(range(y_lower, y_upper)),
                                            orientation='h',
                                            marker=dict(color=color),
                                            name=f"Cluster {cluster_id}",
                                            hovertemplate='Silhouette: %{x:.3f}<extra></extra>'
                                        ))
                                        
                                        y_lower = y_upper + 10
                                    
                                    # Add average score line
                                    fig_silhouette.add_vline(
                                        x=overall_score,
                                        line=dict(color="red", width=2, dash="dash"),
                                        annotation_text=f"Average: {overall_score:.3f}",
                                        annotation_position="top"
                                    )
                                    
                                    fig_silhouette.update_layout(
                                        title="Silhouette Plot for Each Cluster",
                                        xaxis_title="Silhouette Coefficient",
                                        yaxis_title="Cluster",
                                        height=max(400, n_clusters * 100),
                                        showlegend=True,
                                        barmode='overlay'
                                    )
                                    
                                    st.plotly_chart(fig_silhouette, use_container_width=True)
                                    st.success("Silhouette analysis complete!")
                
                with seq_tab2:
                    st.subheader("Pairwise Sequence Alignment")
                    
                    if len(sequences_data) < 2:
                        st.info("Need at least 2 sequences for pairwise alignment.")
                    else:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            seq1_idx = st.selectbox(
                                "Sequence 1",
                                range(len(sequences_data)),
                                format_func=lambda i: sequences_data[i]['ID'],
                                key="seq1_select"
                            )
                        
                        with col2:
                            seq2_idx = st.selectbox(
                                "Sequence 2",
                                range(len(sequences_data)),
                                format_func=lambda i: sequences_data[i]['ID'],
                                index=min(1, len(sequences_data) - 1),
                                key="seq2_select"
                            )
                        
                        seq1 = raw_sequences[seq1_idx]  # Use raw list form
                        seq2 = raw_sequences[seq2_idx]  # Use raw list form
                        
                        # Alignment type
                        align_type = st.radio(
                            "Alignment type",
                            ["Global (Needleman-Wunsch)", "Local (Longest Common Substring)"],
                            help="""
                            **Global (Needleman-Wunsch)**: Aligns entire sequences from start to end, allowing matches, mismatches, and gaps. Best for similar-length sequences.
                            
                            **Local (Longest Common Substring)**: Finds the longest CONTINUOUS sequence that appears in both trajectories without interruptions. Must be 100% exact matches with no gaps within the match.
                            
                            Example for Local:
                            • Seq1: A-B-A-C-B-T-E
                            • Seq2: D-B-A-C-T-E
                            • Result: B-A-C (continuous in both, length=3)
                            • NOT B-A-C-T-E (would skip the B in middle of Seq1)
                            """,
                            key="seq_align_type"
                        )
                        
                        # Alignment parameters
                        st.write("**Alignment Parameters**")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            match_score = st.number_input("Match score", -5, 10, 2, key="align_match")
                        with col2:
                            mismatch_penalty = st.number_input("Mismatch penalty", -10, 5, -1, key="align_mismatch")
                        with col3:
                            gap_penalty = st.number_input("Gap penalty", -10, 5, -1, key="align_gap")
                        
                        # Perform alignment
                        if align_type.startswith("Global"):
                            result = sequence_analysis.needleman_wunsch(seq1, seq2, match_score, mismatch_penalty, gap_penalty)
                            align_method = "Global"
                        else:
                            result = sequence_analysis.smith_waterman(seq1, seq2, match_score, mismatch_penalty, gap_penalty)
                            align_method = "Local"
                        
                        # Display results
                        st.markdown("---")
                        st.write(f"**{align_method} Alignment Results**")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if align_method == "Local":
                                st.metric("Substring Length", f"{int(result['score'])} symbols")
                            else:
                                st.metric("Alignment Score", f"{result['score']:.1f}")
                        with col2:
                            if 'start1' in result:
                                st.metric("Start positions", f"Seq1:{result['start1']}, Seq2:{result['start2']}")
                            else:
                                matches = sum(1 for a, b in zip(result['aligned_seq1'], result['aligned_seq2']) if a == b and a != '-')
                                st.metric("Matches", matches)
                        
                        # Show the actual LCS for Local alignment
                        if align_method == "Local" and 'lcs' in result and result['lcs']:
                            st.success(f"**Longest Common Substring found:** `{'  →  '.join(result['lcs'])}`")
                        
                        # Display alignment
                        st.write("**Aligned Sequences:**")
                        
                        aligned1 = result['aligned_seq1']  # Now a list
                        aligned2 = result['aligned_seq2']  # Now a list
                        
                        # Format alignment with colors and delimiters
                        alignment_html = "<div style='font-family: monospace; font-size: 14px;'>"
                        alignment_html += f"<div><b>{sequences_data[seq1_idx]['ID']}:</b> "
                        
                        for c1, c2 in zip(aligned1, aligned2):
                            if c1 == c2 and c1 != '-':
                                color = 'green'
                            elif c1 == '-' or c2 == '-':
                                color = 'red'
                            else:
                                color = 'orange'
                            # Add delimiter after each token
                            alignment_html += f"<span style='color: {color};'>{c1}</span>-"
                        
                        # Remove trailing delimiter
                        alignment_html = alignment_html.rstrip('-')
                        alignment_html += "</div><div><b>" + f"{sequences_data[seq2_idx]['ID']}:</b> "
                        
                        for c1, c2 in zip(aligned1, aligned2):
                            if c1 == c2 and c1 != '-':
                                color = 'green'
                            elif c1 == '-' or c2 == '-':
                                color = 'red'
                            else:
                                color = 'orange'
                            # Add delimiter after each token
                            alignment_html += f"<span style='color: {color};'>{c2}</span>-"
                        
                        # Remove trailing delimiter
                        alignment_html = alignment_html.rstrip('-')
                        alignment_html += "</div></div>"
                        
                        st.markdown(alignment_html, unsafe_allow_html=True)
                        st.caption("🟢 Match | 🟠 Mismatch | 🔴 Gap")
                        
                        # Compute statistics
                        total_len = len(aligned1)
                        matches = sum(1 for a, b in zip(aligned1, aligned2) if a == b and a != '-')
                        mismatches = sum(1 for a, b in zip(aligned1, aligned2) if a != b and a != '-' and b != '-')
                        gaps = sum(1 for a, b in zip(aligned1, aligned2) if a == '-' or b == '-')
                        
                        st.write("**Alignment Statistics:**")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total length", total_len)
                        with col2:
                            st.metric("Matches", f"{matches} ({100*matches/total_len:.1f}%)")
                        with col3:
                            st.metric("Mismatches", f"{mismatches} ({100*mismatches/total_len:.1f}%)")
                        with col4:
                            st.metric("Gaps", f"{gaps} ({100*gaps/total_len:.1f}%)")
                
                with seq_tab3:
                    st.subheader("N-gram Pattern Analysis")
                    
                    n_gram_size = st.slider("N-gram size", 2, 5, 2, key="seq_ngram_size")
                    
                    # Extract n-grams from all sequences using raw form
                    all_ngrams = Counter()
                    for seq in raw_sequences:
                        if sequence_type == "Per-entity":  # Only for simple sequences
                            ngrams = sequence_analysis.extract_ngrams(seq, n_gram_size)
                            all_ngrams.update(ngrams)
                    
                    if sequence_type == "Multi-entity":
                        st.info("N-gram analysis works best with per-entity sequences. Switch to 'Per-entity' mode for detailed pattern analysis.")
                    elif not all_ngrams:
                        st.warning("No n-grams found. Sequences may be too short.")
                    else:
                        # Display top patterns
                        top_n = st.slider("Show top N patterns", 5, 50, 20, key="seq_top_ngrams")
                        
                        most_common = all_ngrams.most_common(top_n)
                        
                        # Convert tuples to delimited strings for display
                        ngram_df = pd.DataFrame(most_common, columns=['Pattern', 'Frequency'])
                        ngram_df['Pattern'] = ngram_df['Pattern'].apply(lambda x: '-'.join(x))
                        ngram_df['Percentage'] = (100 * ngram_df['Frequency'] / ngram_df['Frequency'].sum()).round(2)
                        
                        st.write(f"**Top {top_n} {n_gram_size}-grams:**")
                        st.dataframe(ngram_df, use_container_width=True, height=400)
                        
                        # Visualize frequency
                        fig_ngram = go.Figure(data=[
                            go.Bar(
                                x=ngram_df['Pattern'],
                                y=ngram_df['Frequency'],
                                text=ngram_df['Frequency'],
                                textposition='auto',
                                marker=dict(color='steelblue')
                            )
                        ])
                        
                        fig_ngram.update_layout(
                            title=f"Most Common {n_gram_size}-grams",
                            xaxis_title="Pattern",
                            yaxis_title="Frequency",
                            height=400
                        )
                        
                        render_interactive_chart(fig_ngram)
                        
                        # Per-sequence n-gram analysis
                        st.write("**Per-Sequence N-gram Breakdown:**")
                        
                        for idx, seq_data in enumerate(sequences_data):
                            seq = raw_sequences[idx]  # Use raw list form
                            # Show preview with delimiter
                            preview = '-'.join(seq[:20]) + ('...' if len(seq) > 20 else '')
                            with st.expander(f"{seq_data['ID']} - {preview}"):
                                seq_ngrams = sequence_analysis.extract_ngrams(seq, n_gram_size)
                                if seq_ngrams:
                                    seq_ngram_df = pd.DataFrame(
                                        seq_ngrams.most_common(10),
                                        columns=['Pattern', 'Count']
                                    )
                                    # Convert tuples to delimited strings for display
                                    seq_ngram_df['Pattern'] = seq_ngram_df['Pattern'].apply(lambda x: '-'.join(x))
                                    st.dataframe(seq_ngram_df, use_container_width=True)
                                else:
                                    st.info("No n-grams in this sequence.")
    
    elif analysis_method == "PDP Analysis":
        st.header("PDP (Point Descriptor Precedence)", help="""
**PDP compares trajectories using relative position relationships (qualitative calculus).**

Instead of comparing exact coordinates, PDP compares whether objects are relatively positioned to the left/right/same in x and above/below/same in y.

**Four PDP Variants:**

🔹 **Fundamental**: Basic qualitative comparison
🔹 **Buffer**: Add tolerance zones around each point
🔹 **Rough**: Allow approximate equality in comparisons
🔹 **Buffer + Rough**: Combined approach (most flexible)

**💡 Use Cases:**

- Compare tactical patterns independent of exact positions
- Find similar movement strategies across different court areas
- Robust to small measurement noise
""")
        
        # Initialize PDP session state
        pdp_analysis.initialize_pdp_session_state()
        
        # Use selections from sidebar
        selected_configs = st.session_state.shared_selected_configs
        selected_objects = st.session_state.shared_selected_objects
        
        # Time range
        min_time = float(df['tst'].min())
        max_time = float(df['tst'].max())
        
        st.markdown("---")
        st.subheader("PDP Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Time Range**")
            start_time = st.number_input(
                "Start time",
                min_value=min_time,
                max_value=max_time,
                value=min_time,
                step=0.01,
                format="%.2f",
                key="pdp_start"
            )
            end_time = st.number_input(
                "End time",
                min_value=start_time,
                max_value=max_time,
                value=max_time,
                step=0.01,
                format="%.2f",
                key="pdp_end"
            )
        
        with col2:
            st.write("**Window Settings**")
            
            # Calculate maximum window length based on actual data
            # Filter data to get unique timestamps in the selected time range and configs
            filtered_df = df[
                (df['tst'] >= start_time) & 
                (df['tst'] <= end_time) & 
                (df['config_source'].isin(selected_configs)) &
                (df['obj'].isin(selected_objects))
            ]
            max_window_length = len(filtered_df['tst'].unique()) if len(filtered_df) > 0 else 10
            max_window_length = max(1, max_window_length)  # Ensure at least 1
            
            # Set default value, capped by max available
            default_window = min(3, max_window_length)
            
            window_length = st.slider(
                "Window length (timestamps)",
                min_value=1,
                max_value=max_window_length,
                value=default_window,
                help=f"Number of consecutive time steps to analyze together (max: {max_window_length} timestamps available)",
                key="pdp_window"
            )
        
        with col3:
            st.write("**Tolerance Parameters**")
            st.caption("Used for Buffer/Rough variants")
            
            buffer_x = st.number_input(
                "Buffer X (m)",
                min_value=0.0,
                max_value=5.0,
                value=1.0,
                step=0.1,
                format="%.2f",
                help="Buffer zone size in x direction",
                key="pdp_buffer_x"
            )
            buffer_y = st.number_input(
                "Buffer Y (m)",
                min_value=0.0,
                max_value=5.0,
                value=1.0,
                step=0.1,
                format="%.2f",
                help="Buffer zone size in y direction",
                key="pdp_buffer_y"
            )
            
            rough_x = st.number_input(
                "Rough X (m)",
                min_value=0.0,
                max_value=5.0,
                value=0.5,
                step=0.1,
                format="%.2f",
                help="Tolerance for 'equal' in x direction",
                key="pdp_rough_x"
            )
            rough_y = st.number_input(
                "Rough Y (m)",
                min_value=0.0,
                max_value=5.0,
                value=0.5,
                step=0.1,
                format="%.2f",
                help="Tolerance for 'equal' in y direction",
                key="pdp_rough_y"
            )
        
        if not selected_configs or not selected_objects:
            st.warning("Please select at least one configuration and one object from the sidebar.")
        elif len(selected_configs) < 2:
            st.warning("Please select at least 2 configurations to compute distances.")
        else:
            # Check if we need to recompute
            if 'pdp_results_all' not in st.session_state:
                st.info("Click Compute to analyze all variants.")
            
            st.markdown("---")
            
            # Compute button
            if st.button("🚀 Compute PDP Analysis (All Variants)", type="primary", key="compute_pdp"):
                with st.spinner("Computing PDP distances for all variants..."):
                    results_all = {}
                    
                    # 1. Fundamental
                    dist_fund, config_ids = pdp_analysis.compute_pdp_distance_matrix(
                        df, selected_configs, selected_objects,
                        start_time, end_time,
                        window_length=window_length,
                        buffer_x=0, buffer_y=0, rough_x=0, rough_y=0,
                        pdp_variant="fundamental"
                    )
                    results_all['fundamental'] = dist_fund
                    
                    # 2. Buffer
                    dist_buff, _ = pdp_analysis.compute_pdp_distance_matrix(
                        df, selected_configs, selected_objects,
                        start_time, end_time,
                        window_length=window_length,
                        buffer_x=buffer_x, buffer_y=buffer_y, rough_x=0, rough_y=0,
                        pdp_variant="buffer"
                    )
                    results_all['buffer'] = dist_buff
                    
                    # 3. Rough
                    dist_rough, _ = pdp_analysis.compute_pdp_distance_matrix(
                        df, selected_configs, selected_objects,
                        start_time, end_time,
                        window_length=window_length,
                        buffer_x=0, buffer_y=0, rough_x=rough_x, rough_y=rough_y,
                        pdp_variant="rough"
                    )
                    results_all['rough'] = dist_rough
                    
                    # 4. Buffer + Rough
                    dist_br, _ = pdp_analysis.compute_pdp_distance_matrix(
                        df, selected_configs, selected_objects,
                        start_time, end_time,
                        window_length=window_length,
                        buffer_x=buffer_x, buffer_y=buffer_y, rough_x=rough_x, rough_y=rough_y,
                        pdp_variant="buffer_rough"
                    )
                    results_all['buffer_rough'] = dist_br
                    
                    # Store results
                    st.session_state.pdp_results_all = results_all
                    st.session_state.pdp_config_ids = config_ids
                    
                    # Set default active variant
                    st.session_state.pdp_active_variant = 'fundamental'
                    st.session_state.pdp_distance_matrix = results_all['fundamental']
                    
                    # Perform initial clustering
                    optimal_n = pdp_analysis.detect_optimal_clusters(results_all['fundamental'])
                    st.session_state.pdp_optimal_n = optimal_n
                    st.session_state.pdp_current_n = optimal_n
                    
                    cluster_labels, linkage_matrix = pdp_analysis.perform_hierarchical_clustering(
                        results_all['fundamental'], optimal_n
                    )
                    st.session_state.pdp_cluster_labels = cluster_labels
                    st.session_state.pdp_linkage_matrix = linkage_matrix
                
                st.success(f"PDP analysis computed for all variants!")
                st.rerun()
            
            # SELECTION LOGIC
            if 'pdp_results_all' in st.session_state and st.session_state.pdp_results_all:
                st.markdown("### Analysis View Settings")
                
                # Determine current selection index
                current_key = st.session_state.get('pdp_active_variant', 'fundamental')
                variant_options = ["Fundamental", "Buffer", "Rough", "Buffer + Rough"]
                variant_keys = ["fundamental", "buffer", "rough", "buffer_rough"]
                
                try:
                    default_index = variant_keys.index(current_key)
                except ValueError:
                    default_index = 0
                
                selected_view = st.radio(
                    "Select Variant to Visualize:",
                    variant_options,
                    index=default_index,
                    key="pdp_view_selector",
                    horizontal=True
                )
                
                variant_map = dict(zip(variant_options, variant_keys))
                active_variant_key = variant_map[selected_view]
                
                # Check if we need to update the active view
                if st.session_state.get('pdp_active_variant') != active_variant_key:
                    st.session_state.pdp_active_variant = active_variant_key
                    st.session_state.pdp_distance_matrix = st.session_state.pdp_results_all[active_variant_key]
                    
                    # Re-run clustering for the new matrix
                    matrix = st.session_state.pdp_distance_matrix
                    optimal_n = pdp_analysis.detect_optimal_clusters(matrix)
                    st.session_state.pdp_optimal_n = optimal_n
                    st.session_state.pdp_current_n = optimal_n
                    
                    cluster_labels, linkage_matrix = pdp_analysis.perform_hierarchical_clustering(
                        matrix, optimal_n
                    )
                    st.session_state.pdp_cluster_labels = cluster_labels
                    st.session_state.pdp_linkage_matrix = linkage_matrix
                    st.rerun()
            
            # Show results if available
            if st.session_state.pdp_distance_matrix is not None:
                
                # ========================================
                # SECTION 1: BASIC RESULTS
                # ========================================
                with st.expander("Inequality Matrices", expanded=False):
                    # 1. INEQUALITY MATRICES (Fundamental Representation)
                    st.markdown("### Inequality Matrices (Fundamental Representation)", help=f"""
**What you're seeing:**

Each row shows inequality matrices for one configuration at one time window:
- **Left matrix (X dimension)**: Horizontal spatial relationships (left/equal/right)
- **Right matrix (Y dimension)**: Vertical spatial relationships (below/equal/above)

**Matrix dimensions:**
- With {len(selected_objects)} object(s) and window_length={window_length}
- Each matrix is {len(selected_objects) * window_length}×{len(selected_objects) * window_length}
- Rows/columns labeled as "O<object>_T<time_index>"

**How to read the matrix:**
- Cell (row i, column j) compares position of point i vs point j
- **0 (Green)**: Point i is LEFT/BELOW point j
- **1 (Yellow)**: Point i is EQUAL to point j (within tolerance)
- **2 (Red)**: Point i is RIGHT/ABOVE point j

**Distance calculation:**
The PDP distance between two configurations is the **sum of differences** across ALL {max(1, max_window_length - window_length + 1)} windows. This visualization shows one of those windows.
""")
                    
                    # Configuration selector for inequality matrices
                    col_ineq1, col_ineq2 = st.columns([2, 1])
                    
                    with col_ineq1:
                        # Allow selecting multiple configurations
                        configs_to_compare = st.multiselect(
                            "Select configurations to view their inequality matrices",
                            options=st.session_state.pdp_config_ids,
                            default=st.session_state.pdp_config_ids[:min(2, len(st.session_state.pdp_config_ids))],
                            help="View the fundamental inequality matrix representation for selected configurations",
                            key="ineq_matrix_configs"
                        )
                    
                    with col_ineq2:
                        if len(configs_to_compare) > 0:
                            st.metric("Configs selected", len(configs_to_compare))
                            if len(configs_to_compare) > 5:
                                st.warning("Many configs selected - visualization may be large")
                    
                    if len(configs_to_compare) > 0:
                        # Get window information first
                        from modules.pdp_analysis import visualize_inequality_matrices
                        
                        window_info = visualize_inequality_matrices(
                            df, configs_to_compare, selected_objects,
                            start_time, end_time,
                            window_length=window_length,
                            buffer_x=buffer_x, buffer_y=buffer_y,
                            rough_x=rough_x, rough_y=rough_y,
                            window_indices=None  # Get metadata
                        )
                        
                        # Determine maximum available windows
                        max_available_windows = 0
                        if window_info:
                            max_available_windows = max([info['max_windows'] for info in window_info.values() if info['max_windows'] > 0], default=0)
                        
                        if max_available_windows == 0:
                            st.warning("Not enough timestamps to create windows with the current window_length setting.")
                        else:
                            # Window selection interface
                            st.markdown("**Select Time Windows to Display:**", help=f"""
**Sliding Window Analysis:**

With **window_length = {window_length}**, the PDP distance calculation uses a **sliding window approach** that computes inequality matrices across multiple overlapping time segments.

**Available windows**: {max_available_windows} (Window 0 to Window {max_available_windows-1})

Each window captures a snapshot of spatial relationships at different points in time:

- **Window 0**: First {window_length} timestamp(s)
- **Window 1**: Timestamps starting from the 2nd position
- ... and so on (each window slides by 1 timestamp)

**Tip**: By default, only the first window is shown to avoid overwhelming visualizations. The final PDP distance accumulates differences across **all** windows.
""")
                            
                            col_win1, col_win2 = st.columns([3, 1])
                            
                            with col_win1:
                                # Multi-select for windows
                                available_windows = list(range(max_available_windows))
                                
                                if max_available_windows <= 10:
                                    # For small number of windows, allow full selection
                                    default_windows = [0]  # Default to first window
                                    selected_windows = st.multiselect(
                                        "Choose which time windows to visualize",
                                        options=available_windows,
                                        default=default_windows,
                                        format_func=lambda x: f"Window {x}",
                                        help=f"Select one or more windows from 0 to {max_available_windows-1}",
                                        key="selected_inequality_windows"
                                    )
                                else:
                                    # For many windows, use slider to select range
                                    st.caption(f"Many windows available ({max_available_windows}). Select a range:")
                                    window_range = st.slider(
                                        "Window range",
                                        min_value=0,
                                        max_value=max_available_windows-1,
                                        value=(0, min(2, max_available_windows-1)),
                                        help="Select start and end window indices",
                                        key="inequality_window_range"
                                    )
                                    selected_windows = list(range(window_range[0], window_range[1] + 1))
                                    st.caption(f"Displaying windows: {', '.join([str(w) for w in selected_windows])}")
                            
                            with col_win2:
                                if selected_windows:
                                    st.metric("Windows to show", len(selected_windows))
                                    total_matrices = len(configs_to_compare) * len(selected_windows) * 2
                                    st.caption(f"{total_matrices} matrices total")
                                    if total_matrices > 20:
                                        st.warning("Large visualization!")
                            
                            if not selected_windows:
                                st.warning("Please select at least one window to visualize.")
                            else:
                                # Display inequality matrices for selected configurations and windows
                                fig_ineq = visualize_inequality_matrices(
                                    df, configs_to_compare, selected_objects,
                                    start_time, end_time,
                                    window_length=window_length,
                                    buffer_x=buffer_x, buffer_y=buffer_y,
                                    rough_x=rough_x, rough_y=rough_y,
                                    window_indices=selected_windows
                                )
                                
                                render_interactive_chart(fig_ineq, caption=None)
                    
                st.markdown("---")
                
                # 2. DISTANCE MATRIX (Derived from Inequality Matrices)
                st.markdown("### PDP Distance Matrix (Computed from Inequality Matrices)", help="How distances are computed: Each configuration has inequality matrices (X and Y) that capture spatial relationships. The distance between two configurations = number of differing cells in their inequality matrices. Larger distance → more different spatial relationships → more different trajectory patterns.")
                
                distance_matrix = st.session_state.pdp_distance_matrix
                config_ids = st.session_state.pdp_config_ids
                n_configs = len(config_ids)
                
                # Option to show normalized distances
                st.markdown("**Display Options:**")
                col_display1, col_display2 = st.columns([1, 2])
                
                with col_display1:
                    show_normalized = st.checkbox(
                        "Show normalized distances (0-100%)",
                        value=False,
                        key="pdp_show_normalized",
                        help="Convert raw distances to 0-100% scale for easier interpretation"
                    )
                
                with col_display2:
                    if show_normalized:
                        st.info("📊 Showing normalized distances (percentage of maximum possible difference)")
                    else:
                        st.info("📊 Showing raw PDP distances (sum of inequality matrix differences)")
                
                # Compute normalized distances if needed
                if show_normalized:
                    norm_info = pdp_analysis.compute_distance_normalization_info(
                        distance_matrix, config_ids
                    )
                    display_matrix = norm_info['normalized_matrix']
                    colorbar_title = "Distance (%)"
                else:
                    display_matrix = distance_matrix
                    colorbar_title = "Raw Distance"
                
                # Provide visualization options for large matrices
                st.markdown("**Visualization Options:**")
                col_viz_opt1, col_viz_opt2 = st.columns([2, 1])
                
                with col_viz_opt1:
                    if n_configs > 30:
                        show_text = st.checkbox(
                            "Show numeric values in heatmap", 
                            value=False, 
                            key="pdp_show_text",
                            help="⚠️ Warning: With many configurations, text may be very small or overlap. Use hover for exact values."
                        )
                    else:
                        show_text = st.checkbox(
                            "Show numeric values in heatmap", 
                            value=True, 
                            key="pdp_show_text"
                        )
                
                with col_viz_opt2:
                    # Font size adjustment when text is shown
                    if show_text:
                        # Suggest smaller text for larger matrices
                        default_size = max(4, min(10, 300 // n_configs))
                        text_size = st.slider("Text size", 2, 12, default_size, key="pdp_text_size",
                                            help=f"Suggested: {default_size}pt for {n_configs} configs")
                    else:
                        text_size = 8
                
                # Compute optimal size based on number of configurations
                # Aim for ~10-15 pixels per cell for good readability
                cell_size = max(10, min(30, 600 // n_configs))
                heatmap_size = max(500, min(1200, n_configs * cell_size))
                
                # Create heatmap - respect user's choice regardless of matrix size
                if show_text:
                    # With text annotations - user explicitly requested this
                    fig_heatmap = go.Figure(data=go.Heatmap(
                        z=display_matrix,
                        x=config_ids,
                        y=config_ids,
                        colorscale='Viridis',
                        text=display_matrix,
                        texttemplate='%{text:.1f}',
                        textfont={"size": text_size},
                        colorbar=dict(title=colorbar_title),
                        hovertemplate='From: %{y}<br>To: %{x}<br>Distance: %{z:.2f}<extra></extra>'
                    ))
                else:
                    # Without text - cleaner visualization, use hover for values
                    fig_heatmap = go.Figure(data=go.Heatmap(
                        z=display_matrix,
                        x=config_ids,
                        y=config_ids,
                        colorscale='Viridis',
                        colorbar=dict(title=colorbar_title),
                        hovertemplate='From: %{y}<br>To: %{x}<br>Distance: %{z:.2f}<extra></extra>'
                    ))
                
                # Smart layout adjustments for axis labels
                if n_configs > 100:
                    # Very large matrices: show every 20th label
                    tick_step = max(1, n_configs // 10)  # ~10 labels total
                    xaxis_config = dict(
                        title="Configuration",
                        tickmode='linear',
                        tick0=0,
                        dtick=tick_step,
                        tickangle=-90,
                        tickfont=dict(size=8)
                    )
                    yaxis_config = dict(
                        title="Configuration",
                        tickmode='linear',
                        tick0=0,
                        dtick=tick_step,
                        tickfont=dict(size=8)
                    )
                elif n_configs > 50:
                    # Large matrices: show every 10th label
                    tick_step = max(1, n_configs // 15)  # ~15 labels total
                    xaxis_config = dict(
                        title="Configuration",
                        tickmode='linear',
                        tick0=0,
                        dtick=tick_step,
                        tickangle=-45,
                        tickfont=dict(size=9)
                    )
                    yaxis_config = dict(
                        title="Configuration",
                        tickmode='linear',
                        tick0=0,
                        dtick=tick_step,
                        tickfont=dict(size=9)
                    )
                elif n_configs > 30:
                    # Medium matrices: show every Nth label
                    tick_step = max(1, n_configs // 20)
                    xaxis_config = dict(
                        title="Configuration",
                        tickmode='linear',
                        tick0=0,
                        dtick=tick_step,
                        tickangle=-45
                    )
                    yaxis_config = dict(
                        title="Configuration",
                        tickmode='linear',
                        tick0=0,
                        dtick=tick_step
                    )
                else:
                    # Small matrices: show all labels
                    xaxis_config = dict(
                        title="Configuration",
                        tickangle=-45 if n_configs > 15 else 0
                    )
                    yaxis_config = dict(title="Configuration")
                
                # Create title indicating distance type
                distance_type = "Normalized (%)" if show_normalized else "Raw"
                
                fig_heatmap.update_layout(
                    title=f"PDP Distance Matrix - {distance_type} ({st.session_state.get('pdp_active_variant', 'fundamental')}) - {n_configs} configurations",
                    xaxis=xaxis_config,
                    yaxis=yaxis_config,
                    width=heatmap_size,
                    height=heatmap_size
                )
                
                render_interactive_chart(fig_heatmap, caption="") 
                        
                # Use reusable configuration selector component
                selected_configs_inspect = configuration_selector(
                    config_ids=config_ids,
                    key_prefix="pdp_matrix_inspect",
                    default_configs=[],
                    max_selections=5,
                    label="Select configurations to visualize",
                    help_text="Choose configurations from the distance matrix to see their actual trajectories",
                    show_metrics=True
                )
                
                # Visualize selected configurations
                if len(selected_configs_inspect) > 0:
                    st.markdown("---")
                    
                    # Show distance information if 2+ configs selected
                    if len(selected_configs_inspect) >= 2:
                        st.markdown("**📏 Pairwise Distances Between Selected Configurations:**")
                        
                        # Create distance table
                        selected_indices = [config_ids.index(cfg) for cfg in selected_configs_inspect]
                        
                        dist_pairs = []
                        for i, cfg1 in enumerate(selected_configs_inspect):
                            for j, cfg2 in enumerate(selected_configs_inspect):
                                if i < j:
                                    idx1 = config_ids.index(cfg1)
                                    idx2 = config_ids.index(cfg2)
                                    dist_val = distance_matrix[idx1, idx2]
                                    
                                    if show_normalized:
                                        dist_pairs.append({
                                            'Config A': cfg1,
                                            'Config B': cfg2,
                                            'Distance': f"{norm_info['normalized_matrix'][idx1, idx2]:.1f}%",
                                            'Raw': f"{dist_val:.1f}"
                                        })
                                    else:
                                        dist_pairs.append({
                                            'Config A': cfg1,
                                            'Config B': cfg2,
                                            'Distance': f"{dist_val:.1f}"
                                        })
                        
                        if dist_pairs:
                            dist_df = pd.DataFrame(dist_pairs)
                            st.dataframe(dist_df, use_container_width=True, hide_index=True)
                    
                    # Plot trajectories
                    st.markdown("**🎾 Trajectory Visualization:**")
                    
                    # Get objects from global selection if available
                    if 'shared_selected_objects' in st.session_state and st.session_state.shared_selected_objects:
                        inspect_objects = st.session_state.shared_selected_objects
                    else:
                        inspect_objects = selected_objects
                    
                    fig_inspect = pdp_analysis.plot_trajectory_comparison(
                        df=st.session_state.data,
                        config_ids=config_ids,
                        selected_configs=selected_configs_inspect,
                        start_time=start_time,
                        end_time=end_time,
                        selected_objects=inspect_objects,
                        cluster_labels=st.session_state.pdp_cluster_labels,
                        distance_matrix=distance_matrix,
                        show_buffers=False,
                        buffer_size=0.5,
                        show_rough=False,
                        rough_tolerance=0.3
                    )
                    
                    # Create a unique key based on selections to force chart recreation
                    chart_key = f"pdp_inspect_chart_{'-'.join(map(str, selected_configs_inspect))}_{'-'.join(map(str, inspect_objects))}"
                    
                    render_interactive_chart(fig_inspect, 
                                           caption=f"Showing {len(selected_configs_inspect)} configuration(s) | " +
                                                  f"Time: {start_time:.1f}s - {end_time:.1f}s",
                                           key=chart_key)
                    
                    # Interpretation help
                    with st.expander("💡 How to interpret this visualization"):
                        st.markdown("""
                        **What you're seeing:**
                        - Each color represents a different configuration
                        - Trajectories show the movement paths of all selected objects
                        - Start points: ⭕ (circles), End points: ◼️ (squares)
                        
                        **Compare with the distance matrix:**
                        - **Small distances** → Similar trajectory patterns (paths overlap or follow similar patterns)
                        - **Large distances** → Very different trajectory patterns (paths diverge significantly)
                        
                        **Tips:**
                        - Hover over points to see configuration, object, and timestamp details
                        - Use the quick selection buttons to explore interesting patterns
                        - Compare 2-3 configurations to understand what causes high/low distances
                        """)
                
                # Summary statistics
                st.markdown("---")
                st.markdown("**Distance Matrix Statistics:**")
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                with col_stat1:
                    st.metric("Matrix Size", f"{n_configs}×{n_configs}")
                with col_stat2:
                    # Get upper triangle (exclude diagonal)
                    triu_indices = np.triu_indices_from(distance_matrix, k=1)
                    distances = distance_matrix[triu_indices]
                    st.metric("Mean Distance", f"{np.mean(distances):.1f}")
                with col_stat3:
                    st.metric("Min Distance", f"{np.min(distances):.1f}")
                with col_stat4:
                    st.metric("Max Distance", f"{np.max(distances):.1f}")
                
                # ========================================
                # DISTANCE NORMALIZATION & DISTRIBUTION
                # ========================================
                with st.expander("Distance Normalization & Distribution Analysis", expanded=False):
                    st.markdown("""
                    **Understanding Distance Scale:**
                    
                    Raw PDP distances can be hard to interpret - is a distance of 50 large or small?
                    Normalization helps by scaling distances to a 0-100 range for easier interpretation.
                    """)
                    
                    # Compute normalization info
                    norm_info = pdp_analysis.compute_distance_normalization_info(
                        distance_matrix, config_ids
                    )
                    
                    # Display formula
                    st.markdown("### 📊 Normalization Formula")
                    st.latex(r"\text{Normalized Distance} = \frac{\text{Raw Distance}}{\text{Max Possible Distance}} \times 100")
                    
                    col_formula1, col_formula2 = st.columns(2)
                    with col_formula1:
                        st.metric("Max Possible Distance", f"{norm_info['max_possible_distance']:.1f}")
                    with col_formula2:
                        st.info("This is the theoretical maximum distance observed in your dataset")
                    
                    # Example calculation
                    st.markdown("### 🔍 Example Calculation")
                    example = norm_info['example_calculation']
                    
                    st.markdown(f"""
                    **Comparing {example['config_a']} and {example['config_b']}:**
                    
                    - Raw Distance: **{example['raw_distance']:.2f}**
                    - Max Possible: **{example['max_possible']:.2f}**
                    - Calculation: `{example['formula']}`
                    - **Result: {example['normalized_distance']:.2f}%** difference
                    
                    💡 *Interpretation: These configurations differ by {example['normalized_distance']:.1f}% of the maximum possible difference.*
                    """)
                    
                    # Statistics comparison
                    st.markdown("### 📈 Distance Statistics")
                    
                    col_stats1, col_stats2 = st.columns(2)
                    
                    with col_stats1:
                        st.markdown("**Raw Distances:**")
                        raw_stats = norm_info['stats']['raw']
                        st.dataframe({
                            'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Q25', 'Q75'],
                            'Value': [
                                f"{raw_stats['mean']:.2f}",
                                f"{raw_stats['median']:.2f}",
                                f"{raw_stats['std']:.2f}",
                                f"{raw_stats['min']:.2f}",
                                f"{raw_stats['max']:.2f}",
                                f"{raw_stats['q25']:.2f}",
                                f"{raw_stats['q75']:.2f}"
                            ]
                        }, hide_index=True, use_container_width=True)
                    
                    with col_stats2:
                        st.markdown("**Normalized Distances (0-100):**")
                        norm_stats = norm_info['stats']['normalized']
                        st.dataframe({
                            'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Q25', 'Q75'],
                            'Value': [
                                f"{norm_stats['mean']:.2f}%",
                                f"{norm_stats['median']:.2f}%",
                                f"{norm_stats['std']:.2f}%",
                                f"{norm_stats['min']:.2f}%",
                                f"{norm_stats['max']:.2f}%",
                                f"{norm_stats['q25']:.2f}%",
                                f"{norm_stats['q75']:.2f}%"
                            ]
                        }, hide_index=True, use_container_width=True)
                    
                    # Distribution histogram
                    st.markdown("### 📊 Distance Distribution")
                    
                    show_normalized_hist = st.radio(
                        "Show distances as:",
                        options=["Normalized (0-100)", "Raw"],
                        horizontal=True,
                        key="pdp_hist_type"
                    )
                    
                    fig_hist = pdp_analysis.create_distance_distribution_plot(
                        norm_info['histogram_data'],
                        show_normalized=(show_normalized_hist == "Normalized (0-100)")
                    )
                    
                    render_interactive_chart(fig_hist)
                    
                    st.caption("""
                    💡 **Tip**: The distribution shape reveals dataset characteristics:
                    - **Uniform spread**: Configurations vary gradually
                    - **Bimodal (two peaks)**: Two distinct groups of configurations
                    - **Skewed right**: Most configs similar, few outliers very different
                    """)
                    
                    # Download normalized distances
                    st.markdown("### 💾 Download Data")
                    
                    # Create DataFrame with both raw and normalized
                    export_data = []
                    for i in range(len(config_ids)):
                        for j in range(i+1, len(config_ids)):
                            export_data.append({
                                'Config_A': config_ids[i],
                                'Config_B': config_ids[j],
                                'Raw_Distance': distance_matrix[i, j],
                                'Normalized_Distance': norm_info['normalized_matrix'][i, j],
                                'Normalized_%': f"{norm_info['normalized_matrix'][i, j]:.2f}%"
                            })
                    
                    export_df = pd.DataFrame(export_data)
                    csv_export = export_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download Normalized Distances (CSV)",
                        data=csv_export,
                        file_name="pdp_normalized_distances.csv",
                        mime="text/csv",
                        help="Download pairwise distances with both raw and normalized values"
                    )
                # End of Basic Results expander
                
                # ========================================
                # SECTION 2: CLUSTERING & PROJECTION
                # ========================================
                with st.expander("Clustering & Projection (Dendrogram, MDS, Trajectory Comparison)", expanded=False):
                    # Dendrogram
                    st.markdown("### Hierarchical Clustering Dendrogram")
                    
                    st.info("""
                    **How to read the dendrogram:**
                    - **Height of branches** indicates the distance between clusters (higher = more different)
                    - **Different colors** represent different clusters automatically identified by the algorithm
                    - Configurations connected at lower heights are more similar
                    - The x-axis shows each configuration label
                    """)
                    
                    if st.session_state.pdp_linkage_matrix is not None:
                        fig_dend = pdp_analysis.create_interactive_dendrogram(
                            st.session_state.pdp_linkage_matrix,
                            config_ids,
                            distance_matrix,
                            st.session_state.pdp_current_n
                        )
                        render_interactive_chart(fig_dend)
                        
                        # Cluster selection
                        st.markdown("---")
                        st.subheader("Cluster Assignment")
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            n_clusters = st.slider(
                                "Number of clusters",
                                min_value=2,
                                max_value=min(10, len(config_ids)),
                                value=st.session_state.pdp_optimal_n,
                                key="pdp_n_clusters"
                            )
                            
                            if n_clusters != st.session_state.pdp_current_n:
                                st.session_state.pdp_current_n = n_clusters
                                new_cluster_labels, _ = pdp_analysis.perform_hierarchical_clustering(
                                    distance_matrix, n_clusters
                                )
                                st.session_state.pdp_cluster_labels = new_cluster_labels
                        
                        with col2:
                            st.metric("Optimal Clusters", st.session_state.pdp_optimal_n)
                        
                        # Show cluster assignments
                        if st.session_state.pdp_cluster_labels is not None:
                            cluster_df = pd.DataFrame({
                                'Configuration': config_ids,
                                'Cluster': st.session_state.pdp_cluster_labels
                            })
                            
                            st.dataframe(
                                cluster_df.sort_values('Cluster'),
                                use_container_width=True,
                                hide_index=True
                            )
                            
                            # Cluster statistics
                            st.markdown("**Cluster Sizes:**")
                            cluster_counts = pd.Series(st.session_state.pdp_cluster_labels).value_counts().sort_index()
                            cols = st.columns(min(5, len(cluster_counts)))
                            for i, (cluster_id, count) in enumerate(cluster_counts.items()):
                                with cols[i % len(cols)]:
                                    st.metric(f"Cluster {cluster_id}", count)
                    
                    # MDS Visualization
                    st.markdown("---")
                    st.subheader("MDS Projection")
                    
                    st.info("""
                    **Multidimensional Scaling (MDS) visualization:**
                    - Reduces high-dimensional distance matrix to 2D or 3D for easy visualization
                    - **Closer points** = more similar configurations
                    - **Further apart** = more different configurations
                    - Colors represent cluster assignments
                    - **Stress metric**: measures how well the low-dimensional representation preserves distances (lower is better)
                    """)
                    
                    # Dimension selector
                    mds_dims = st.radio(
                        "Select MDS dimensions:",
                        options=[2, 3],
                        index=0,
                        horizontal=True,
                        help="Choose between 2D (easier to read) or 3D (more detail) visualization"
                    )
                    
                    if st.session_state.pdp_cluster_labels is not None:
                        if mds_dims == 2:
                            # 2D MDS
                            fig_mds, stress = pdp_analysis.create_mds_visualization(
                                distance_matrix,
                                config_ids,
                                st.session_state.pdp_cluster_labels
                            )
                            render_interactive_chart(fig_mds)
                            
                            # Show stress interpretation
                            with st.expander("Understanding the Stress Value"):
                                st.markdown(f"""
                                **Current Stress: {stress:.2f}**
                                
                                The stress value indicates how well the 2D projection preserves the original distances:
                                - **< 0.05**: Excellent representation
                                - **0.05 - 0.10**: Good representation
                                - **0.10 - 0.20**: Fair representation
                                - **> 0.20**: Poor representation (consider using 3D or more clusters)
                                
                                Lower stress means the 2D visualization more accurately represents the true distances between configurations.
                                """)
                        else:
                            # 3D MDS
                            fig_mds_3d, stress = pdp_analysis.create_mds_visualization_3d(
                                distance_matrix,
                                config_ids,
                                st.session_state.pdp_cluster_labels
                            )
                            render_interactive_chart(fig_mds_3d)
                            
                            st.success(f"💡 **Tip**: Use your mouse to rotate, zoom, and explore the 3D space! Stress: {stress:.2f}")
                            
                            # Show stress interpretation
                            with st.expander("Understanding the Stress Value"):
                                st.markdown(f"""
                                **Current Stress: {stress:.2f}**
                                
                                The stress value indicates how well the 3D projection preserves the original distances:
                                - **< 0.05**: Excellent representation
                                - **0.05 - 0.10**: Good representation
                                - **0.10 - 0.20**: Fair representation
                                - **> 0.20**: Poor representation
                                
                                3D projections typically have lower stress than 2D, providing a more accurate representation of configuration similarities.
                                """)
                    
                    
                    # Top-K Similar Configurations
                    st.markdown("---")
                    st.subheader("🔍 Find Similar Configurations")
                    
                    st.info("""
                    **Find configurations most similar to a selected one:**
                    - Select a target configuration
                    - View the K most similar configurations ranked by PDP distance
                    - Lower distance = more similar movement patterns
                    """)
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        target_config = st.selectbox(
                            "Select target configuration",
                            config_ids,
                            key="pdp_target_config"
                        )
                    
                    with col2:
                        k_similar = st.slider(
                            "Number of similar configs (K)",
                            min_value=1,
                            max_value=min(10, len(config_ids) - 1),
                            value=min(5, len(config_ids) - 1),
                            key="pdp_k_similar"
                        )
                    
                    if target_config:
                        similar_configs = pdp_analysis.find_top_k_similar(
                            distance_matrix,
                            config_ids,
                            target_config,
                            k=k_similar
                        )
                        
                        st.markdown(f"**Top {k_similar} configurations most similar to `{target_config}`:**")
                        
                        similar_df = pd.DataFrame(similar_configs, columns=['Configuration', 'PDP Distance'])
                        similar_df['Rank'] = range(1, len(similar_df) + 1)
                        similar_df = similar_df[['Rank', 'Configuration', 'PDP Distance']]
                        
                        st.dataframe(similar_df, use_container_width=True, hide_index=True)
                        
                        # Visualize on bar chart
                        fig_topk = go.Figure(data=[
                            go.Bar(
                                x=similar_df['Configuration'],
                                y=similar_df['PDP Distance'],
                                marker_color='lightblue',
                                text=similar_df['PDP Distance'],
                                textposition='outside'
                            )
                        ])
                    
                        
                        fig_topk.update_layout(
                            title=f"PDP Distances from {target_config}",
                            xaxis_title="Configuration",
                            yaxis_title="PDP Distance",
                            height=400
                        )
                        
                        render_interactive_chart(fig_topk, caption="Lower distance = more similar")
                    
                    # ===============================================================
                    # TRAJECTORY COMPARISON VISUALIZATION
                    # ===============================================================
                    st.markdown("---")
                    st.subheader("Trajectory Comparison on Tennis Court")
                    
                    st.info("""
                    **Visualize and compare actual trajectories on the tennis court.**
                    - Select configurations to overlay their movement patterns
                    - Compare similar or dissimilar configurations side-by-side
                    - Start points marked with circles ⭕, end points with squares ◼️
                    """)
                    
                    col_traj1, col_traj2 = st.columns([1, 1])
                    
                    with col_traj1:
                        # Configuration selection
                        num_configs_to_compare = st.slider(
                            "Number of configurations to compare",
                            min_value=1,
                            max_value=min(5, len(config_ids)),
                            value=min(2, len(config_ids)),
                            help="Select how many configurations to overlay"
                        )
                        
                        selected_configs_viz = st.multiselect(
                            "Select configurations to visualize",
                            options=config_ids,
                            default=config_ids[:num_configs_to_compare],
                            max_selections=5,
                            help="Choose configurations to compare their trajectories"
                        )
                    
                    with col_traj2:
                        # Object selection - respect global selection from data selection interface
                        # Get objects from global selection if available
                        if 'shared_selected_objects' in st.session_state and st.session_state.shared_selected_objects:
                            globally_selected_objects = st.session_state.shared_selected_objects
                        else:
                            globally_selected_objects = sorted(st.session_state.data['obj'].unique())
                        
                        show_all_objects = st.checkbox(
                            "Show all globally selected objects",
                            value=True,
                            help="Uncheck to further filter the objects selected in 'Data Selection Interface'"
                        )
                        
                        if not show_all_objects:
                            selected_objects_viz = st.multiselect(
                                "Select objects to show",
                                options=globally_selected_objects,
                                default=globally_selected_objects[:3] if len(globally_selected_objects) > 0 else [],
                                help="Choose which moving objects to display (from globally selected objects)"
                            )
                        else:
                            selected_objects_viz = globally_selected_objects  # Use global selection, not all objects
                    
                    # Info about object selection
                    if 'shared_selected_objects' in st.session_state and st.session_state.shared_selected_objects:
                        st.info(f"ℹUsing {len(globally_selected_objects)} object(s) from **Data Selection Interface**. To change, go to the sidebar.")
                    else:
                        st.warning("No objects selected in **Data Selection Interface**. All objects from dataset will be used.")
                    
                    # Quick selection buttons with customizable parameters
                    st.markdown("**Quick Selection:**")
                    st.caption("Use these buttons to automatically select interesting configuration pairs for comparison")
                    
                    # Parameter settings row
                    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
                    with col_p1:
                        n_top_similar = st.number_input("Top N similar", min_value=2, max_value=min(10, len(config_ids)), 
                                                       value=2, step=1, key="pdp_n_top_similar",
                                                       help="Number of most similar configurations to select")
                    with col_p2:
                        n_top_dissimilar = st.number_input("Top N dissimilar", min_value=2, max_value=min(10, len(config_ids)), 
                                                           value=2, step=1, key="pdp_n_top_dissimilar",
                                                           help="Number of most dissimilar configurations to select")
                    with col_p3:
                        if st.session_state.pdp_cluster_labels is not None:
                            n_clusters = len(set(st.session_state.pdp_cluster_labels))
                            n_centroids = st.number_input("N centroids", min_value=1, max_value=n_clusters, 
                                                         value=min(3, n_clusters), step=1, key="pdp_n_centroids",
                                                         help="Number of cluster centroids to select")
                        else:
                            st.text("(assign clusters first)")
                            n_centroids = 3
                    with col_p4:
                        n_random = st.number_input("N random", min_value=1, max_value=min(10, len(config_ids)), 
                                                  value=3, step=1, key="pdp_n_random",
                                                  help="Number of random configurations to select")
                    
                    # Button row
                    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
                    
                    with col_btn1:
                        if st.button("Top Similar", use_container_width=True, 
                                    help=f"Select the {n_top_similar} configurations with smallest PDP distances (most similar trajectories)"):
                            if len(config_ids) >= n_top_similar:
                                # Find N most similar configs by sorting all unique pairs
                                distances_with_pairs = []
                                for i in range(len(config_ids)):
                                    for j in range(i+1, len(config_ids)):
                                        distances_with_pairs.append((distance_matrix[i, j], i, j))
                                distances_with_pairs.sort()  # Sort by distance (ascending)
                                
                                # Collect unique configs from top pairs
                                selected_indices = set()
                                for dist, i, j in distances_with_pairs:
                                    selected_indices.add(i)
                                    selected_indices.add(j)
                                    if len(selected_indices) >= n_top_similar:
                                        break
                                
                                selected_indices = sorted(list(selected_indices))[:n_top_similar]
                                st.session_state['pdp_viz_selection'] = [config_ids[idx] for idx in selected_indices]
                                avg_dist = np.mean([distance_matrix[i, j] for i in selected_indices for j in selected_indices if i < j])
                                st.success(f"Selected {len(selected_indices)} most similar configs (avg distance: {avg_dist:.2f})")
                    
                    with col_btn2:
                        if st.button("Top Dissimilar", use_container_width=True,
                                    help=f"Select the {n_top_dissimilar} configurations with largest PDP distances (most different trajectories)"):
                            if len(config_ids) >= n_top_dissimilar:
                                # Find N most dissimilar configs
                                distances_with_pairs = []
                                for i in range(len(config_ids)):
                                    for j in range(i+1, len(config_ids)):
                                        distances_with_pairs.append((distance_matrix[i, j], i, j))
                                distances_with_pairs.sort(reverse=True)  # Sort by distance (descending)
                                
                                # Collect unique configs from top pairs
                                selected_indices = set()
                                for dist, i, j in distances_with_pairs:
                                    selected_indices.add(i)
                                    selected_indices.add(j)
                                    if len(selected_indices) >= n_top_dissimilar:
                                        break
                                
                                selected_indices = sorted(list(selected_indices))[:n_top_dissimilar]
                                st.session_state['pdp_viz_selection'] = [config_ids[idx] for idx in selected_indices]
                                avg_dist = np.mean([distance_matrix[i, j] for i in selected_indices for j in selected_indices if i < j])
                                st.success(f"Selected {len(selected_indices)} most dissimilar configs (avg distance: {avg_dist:.2f})")
                    
                    with col_btn3:
                        if st.session_state.pdp_cluster_labels is not None and st.button("📊 Cluster Centroids", use_container_width=True,
                                                                                         help=f"Select {n_centroids} representative configurations from different clusters"):
                            # Select one config from each cluster (the one most central to its cluster)
                            unique_clusters = sorted(set(st.session_state.pdp_cluster_labels))
                            centroid_configs = []
                            for cluster_id in unique_clusters[:min(n_centroids, len(unique_clusters))]:
                                # Find config closest to cluster center (smallest avg distance to other cluster members)
                                cluster_indices = [i for i, c in enumerate(st.session_state.pdp_cluster_labels) if c == cluster_id]
                                cluster_distances = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
                                avg_distances = cluster_distances.mean(axis=1)
                                centroid_idx = cluster_indices[np.argmin(avg_distances)]
                                centroid_configs.append(config_ids[centroid_idx])
                            st.session_state['pdp_viz_selection'] = centroid_configs
                            st.success(f"Selected {len(centroid_configs)} cluster centroids from {len(centroid_configs)} different clusters")
                    
                    with col_btn4:
                        if st.button("Random Sample", use_container_width=True,
                                    help=f"Randomly select {n_random} configurations for comparison"):
                            import random
                            n_sample = min(n_random, len(config_ids))
                            random_selection = random.sample(config_ids, n_sample)
                            st.session_state['pdp_viz_selection'] = random_selection
                            st.success(f"Randomly selected {len(random_selection)} configurations")
                    
                    # Use button selection if available, otherwise use multiselect
                    if 'pdp_viz_selection' in st.session_state and st.session_state['pdp_viz_selection']:
                        selected_configs_viz = st.session_state['pdp_viz_selection']
                    
                    # Visualization options for buffer and rough zones
                    st.markdown("---")
                    st.markdown("**🔬 Advanced Visualization Options:**")
                    st.caption("Visualize how buffer and rough parameters affect PDP computation")
                    
                    col_viz1, col_viz2 = st.columns(2)
                    
                    with col_viz1:
                        st.markdown("**Buffer Points (Data Expansion)**")
                        show_buffers = st.checkbox("Show buffer points", value=False, key="pdp_show_buffers",
                                                  help="Buffer ADDS extra data points around each original point at cardinal directions (left/right/up/down)")
                        if show_buffers:
                            buffer_size = st.slider("Buffer distance (meters)", min_value=0.1, max_value=2.0, 
                                                   value=0.5, step=0.1, key="pdp_buffer_size",
                                                   help="Distance of buffer points from original point")
                        else:
                            buffer_size = 0.5
                    
                    with col_viz2:
                        st.markdown("**Rough Tolerance (Comparison Zone)**")
                        show_rough = st.checkbox("Show rough zones", value=False, key="pdp_show_rough",
                                                help="Rough defines a TOLERANCE ZONE where points are considered 'approximately equal' in comparisons")
                        if show_rough:
                            rough_tolerance = st.slider("Rough radius (meters)", min_value=0.1, max_value=2.0, 
                                                       value=0.3, step=0.1, key="pdp_rough_tolerance",
                                                       help="Radius of tolerance zone for approximate equality")
                        else:
                            rough_tolerance = 0.3
                    
                    if show_buffers or show_rough:
                        st.info(f"""
                        **Visualization Legend:**
                        - **Buffer points** (small X markers): Actual extra data points added in 'buffer' variant (4 points per original: left, right, up, down)
                        - **Rough zones** (dashed circles): Tolerance zones for 'rough' variant - points within this radius are considered "approximately equal"
                        
                        **Key Difference:**
                        - Buffer = MORE data points (expands dataset)
                        - Rough = MORE tolerance in comparison (same dataset, different comparison logic)
                        """)
                    
                    st.markdown("---")
                    
                    # Generate visualization
                    if len(selected_configs_viz) > 0:
                        fig_traj = pdp_analysis.plot_trajectory_comparison(
                            df=st.session_state.data,
                            config_ids=config_ids,
                            selected_configs=selected_configs_viz,
                            start_time=start_time,
                            end_time=end_time,
                            selected_objects=selected_objects_viz,
                            cluster_labels=st.session_state.pdp_cluster_labels,
                            distance_matrix=distance_matrix,
                            show_buffers=show_buffers,
                            buffer_size=buffer_size,
                            show_rough=show_rough,
                            rough_tolerance=rough_tolerance
                        )
                        
                        render_interactive_chart(fig_traj, 
                                               caption=f"Comparing {len(selected_configs_viz)} configurations | " +
                                                      f"Time window: {start_time:.1f}s - {end_time:.1f}s")
                        
                        # Show pairwise similarities if 2+ configs selected
                        if len(selected_configs_viz) >= 2:
                            st.markdown("**Pairwise Similarities:**")
                            sim_data = []
                            for i in range(len(selected_configs_viz)):
                                for j in range(i+1, len(selected_configs_viz)):
                                    config_i = selected_configs_viz[i]
                                    config_j = selected_configs_viz[j]
                                    idx_i = config_ids.index(config_i)
                                    idx_j = config_ids.index(config_j)
                                    pdp_dist = distance_matrix[idx_i, idx_j]
                                    similarity = 100 - pdp_dist
                                    sim_data.append({
                                        'Config A': config_i,
                                        'Config B': config_j,
                                        'PDP Distance': f"{pdp_dist:.2f}",
                                        'Similarity %': f"{similarity:.1f}%"
                                    })
                            
                            sim_df = pd.DataFrame(sim_data)
                            st.dataframe(sim_df, use_container_width=True, hide_index=True)
                    else:
                        st.warning("Please select at least one configuration to visualize.")
                    
                    # ===============================================================
                    # EXPORT FUNCTIONALITY
                    # ===============================================================
                    st.markdown("---")
                    st.subheader("Export Results")
                    pdp_variant = st.session_state.get('pdp_active_variant', 'fundamental')
                    
                    st.info("""
                    **Download PDP analysis results for further processing.**
                    - Distance matrix (CSV format)
                    - Cluster assignments
                    - Similarity rankings for all configurations
                    """)
                    
                    col_exp1, col_exp2, col_exp3 = st.columns(3)
                    
                    with col_exp1:
                        # Export distance matrix
                        dist_csv, cluster_csv = pdp_analysis.export_pdp_results_to_csv(
                            distance_matrix=distance_matrix,
                            config_ids=config_ids,
                            cluster_labels=st.session_state.pdp_cluster_labels
                        )
                        
                        st.download_button(
                            label="Download Distance Matrix",
                            data=dist_csv,
                            file_name=f"pdp_distance_matrix_{pdp_variant}.csv",
                            mime="text/csv",
                            use_container_width=True,
                            help="Full pairwise PDP distance matrix"
                        )
                    
                    with col_exp2:
                        # Export cluster assignments
                        if cluster_csv is not None:
                            st.download_button(
                                label="Download Cluster Labels",
                                data=cluster_csv,
                                file_name=f"pdp_clusters_{pdp_variant}.csv",
                                mime="text/csv",
                                use_container_width=True,
                                help="Configuration cluster assignments"
                            )
                        else:
                            st.button(
                                "No Clusters Yet",
                                disabled=True,
                                use_container_width=True,
                                help="Assign clusters first using the slider above"
                            )
                    
                    with col_exp3:
                        # Export similarity rankings
                        top_k_export = st.number_input(
                            "Top K similar per config",
                            min_value=1,
                            max_value=len(config_ids)-1,
                            value=min(10, len(config_ids)-1),
                            help="Number of similar configs to export for each configuration"
                        )
                        
                        rankings_csv = pdp_analysis.export_similarity_rankings_to_csv(
                            config_ids=config_ids,
                            distance_matrix=distance_matrix,
                            top_k=top_k_export
                        )
                        
                        st.download_button(
                            label="📋 Download Rankings",
                            data=rankings_csv,
                            file_name=f"pdp_similarity_rankings_{pdp_variant}.csv",
                            mime="text/csv",
                            use_container_width=True,
                            help=f"Top {top_k_export} similar configs for each configuration"
                        )
                # End of Clustering & Projection expander
                
                # ===============================================================
                # SECTION 3: PARAMETER IMPACT ANALYSIS (FEATURE #2)
                # ===============================================================
                with st.expander("Parameter Impact Analysis (Compare PDP Variants)", expanded=False):
                    st.markdown("### Parameter Impact Analysis")
                    
                    st.info("""
                    **Compare how buffer and rough parameters affect PDP distances.**
                    
                    This analysis computes distances using all four PDP variants and visualizes:
                    - How distances change across variants
                    - Which configuration pairs are most affected by parameters
                    - Correlation between variants
                    
                    **Use this to:**
                    - Understand parameter sensitivity in your dataset
                    - Choose the best variant for your analysis
                    - Identify configurations that behave differently with parameters
                    """)
                    
                    st.markdown("---")
                    st.markdown("**This will compute PDP distances for ALL four variants:**")
                    st.markdown("""
                    - 🔹 Fundamental (baseline - no parameters)
                    - 🔹 Buffer (adds tolerance zones)
                    - 🔹 Rough (approximate equality)
                    - 🔹 Buffer + Rough (combined)
                    
                    **Note:** This may take a moment as it computes 4 distance matrices.
                    """)
                    
                    # Parameter configuration for comparison
                    st.markdown("---")
                    st.markdown("#### Set Parameters for Comparison")
                    st.caption("Define the buffer and rough values to use in the parametrized variants")
                    
                    col_comp1, col_comp2 = st.columns(2)
                    
                    with col_comp1:
                        st.markdown("**Buffer Parameters** (Data Expansion)")
                        buffer_x_comp = st.number_input(
                            "Buffer X (meters)",
                            min_value=0.0,
                            max_value=5.0,
                            value=0.5,
                            step=0.1,
                            key="comp_buffer_x",
                            help="Horizontal buffer distance - adds extra points left/right of each position"
                        )
                        buffer_y_comp = st.number_input(
                            "Buffer Y (meters)",
                            min_value=0.0,
                            max_value=5.0,
                            value=0.5,
                            step=0.1,
                            key="comp_buffer_y",
                            help="Vertical buffer distance - adds extra points above/below each position"
                        )
                    
                    with col_comp2:
                        st.markdown("**Rough Parameters** (Comparison Tolerance)")
                        rough_x_comp = st.number_input(
                            "Rough X (meters)",
                            min_value=0.0,
                            max_value=5.0,
                            value=0.3,
                            step=0.1,
                            key="comp_rough_x",
                            help="Horizontal tolerance - points within this distance considered equal in X"
                        )
                        rough_y_comp = st.number_input(
                            "Rough Y (meters)",
                            min_value=0.0,
                            max_value=5.0,
                            value=0.3,
                            step=0.1,
                            key="comp_rough_y",
                            help="Vertical tolerance - points within this distance considered equal in Y"
                        )
                    
                    # Show summary of what will be compared
                    st.markdown("---")
                    st.markdown("### 📋 Comparison Summary")
                    
                    comp_summary = f"""
                    | Variant | Buffer X | Buffer Y | Rough X | Rough Y |
                    |---------|----------|----------|---------|---------|
                    | � Fundamental | 0.0 m | 0.0 m | 0.0 m | 0.0 m |
                    | 🔹 Buffer | **{buffer_x_comp:.1f} m** | **{buffer_y_comp:.1f} m** | 0.0 m | 0.0 m |
                    | 🔹 Rough | 0.0 m | 0.0 m | **{rough_x_comp:.1f} m** | **{rough_y_comp:.1f} m** |
                    | 🔹 Buffer + Rough | **{buffer_x_comp:.1f} m** | **{buffer_y_comp:.1f} m** | **{rough_x_comp:.1f} m** | **{rough_y_comp:.1f} m** |
                    """
                    st.markdown(comp_summary)
                    
                    # Validation check
                    if buffer_x_comp == 0 and buffer_y_comp == 0 and rough_x_comp == 0 and rough_y_comp == 0:
                        st.warning("All parameters are set to 0. This means all variants will produce identical results. Please set buffer or rough values > 0 to see meaningful differences.")
                    
                    st.markdown("---")
                    
                    if st.button("Compare All PDP Variants", key="compare_variants", type="primary"):
                        with st.spinner("Computing distances for all 4 variants..."):
                            # Compare all variants with user-specified parameters
                            variant_results = pdp_analysis.compare_pdp_variants(
                                df=st.session_state.data,
                                selected_configs=selected_configs,
                                selected_objects=selected_objects,
                                start_time=start_time,
                                end_time=end_time,
                                window_length=window_length,
                                buffer_x=buffer_x_comp,
                                buffer_y=buffer_y_comp,
                                rough_x=rough_x_comp,
                                rough_y=rough_y_comp
                            )
                            
                            st.session_state['variant_comparison_results'] = variant_results
                        
                        st.success("Variant comparison complete!")
                        st.rerun()
                    
                    # Show results if available
                    if 'variant_comparison_results' in st.session_state and st.session_state['variant_comparison_results']:
                        variant_results = st.session_state['variant_comparison_results']
                        
                        st.markdown("---")
                        st.markdown("### Comparison Results")
                        
                        # Statistics table
                        st.markdown("#### Distance Statistics by Variant")
                        
                        stats_data = []
                        for variant_name in ['fundamental', 'buffer', 'rough', 'buffer_rough']:
                            data = variant_results[variant_name]
                            stats_data.append({
                                'Variant': {
                                    'fundamental': 'Fundamental',
                                    'buffer': 'Buffer',
                                    'rough': 'Rough',
                                    'buffer_rough': 'Buffer + Rough'
                                }[variant_name],
                                'Mean': f"{data['mean']:.2f}",
                                'Median': f"{data['median']:.2f}",
                                'Std Dev': f"{data['std']:.2f}",
                                'Min': f"{data['min']:.2f}",
                                'Max': f"{data['max']:.2f}"
                            })
                        
                        stats_df = pd.DataFrame(stats_data)
                        st.dataframe(stats_df, use_container_width=True, hide_index=True)
                        
                        # Box plot comparison
                        st.markdown("---")
                        st.markdown("#### Distribution Comparison")
                        st.caption("Box plots show the distribution of pairwise distances for each variant")
                        
                        fig_box = pdp_analysis.create_parameter_comparison_plot(variant_results)
                        render_interactive_chart(fig_box)
                        
                        # Interpretation
                        fund_mean = variant_results['fundamental']['mean']
                        buffer_mean = variant_results['buffer']['mean']
                        rough_mean = variant_results['rough']['mean']
                        br_mean = variant_results['buffer_rough']['mean']
                        
                        col_interp1, col_interp2 = st.columns(2)
                        
                        with col_interp1:
                            buffer_change = ((buffer_mean - fund_mean) / fund_mean * 100)
                            if buffer_change > 0:
                                st.metric("Buffer Impact", f"+{buffer_change:.1f}%", 
                                         delta="Increases distances", delta_color="normal")
                            else:
                                st.metric("Buffer Impact", f"{buffer_change:.1f}%",
                                         delta="Decreases distances", delta_color="inverse")
                        
                        with col_interp2:
                            rough_change = ((rough_mean - fund_mean) / fund_mean * 100)
                            if rough_change > 0:
                                st.metric("Rough Impact", f"+{rough_change:.1f}%",
                                         delta="Increases distances", delta_color="normal")
                            else:
                                st.metric("Rough Impact", f"{rough_change:.1f}%",
                                         delta="Decreases distances", delta_color="inverse")
                        
                        # Scatter plots
                        st.markdown("---")
                        st.markdown("#### Pairwise Distance Comparison")
                        st.caption("Each point is a configuration pair. Points above diagonal: parameter increases distance; below: decreases distance")
                        
                        fig_scatter = pdp_analysis.create_parameter_sensitivity_scatter(variant_results)
                        render_interactive_chart(fig_scatter)
                        
                        # Correlation heatmap
                        st.markdown("---")
                        st.markdown("#### Variant Correlation")
                        st.caption("How strongly do variants agree on distance rankings?")
                        
                        fig_corr = pdp_analysis.create_correlation_heatmap(variant_results)
                        render_interactive_chart(fig_corr)
                        
                        # Interpretation guide
                        with st.expander("How to Interpret These Results"):
                            st.markdown("""
                            **Statistics Table:**
                            - **Mean/Median**: Average distance between configurations
                            - Higher values → configurations are more different on average
                            - Compare across variants to see parameter impact
                            
                            **Box Plot:**
                            - **Box**: Middle 50% of distances (25th to 75th percentile)
                            - **Line in box**: Median distance
                            - **Diamond**: Mean distance
                            - **Whiskers**: Range of typical distances
                            - **Outliers**: Unusual configuration pairs
                            
                            **Scatter Plots:**
                            - **Points above diagonal**: Parameter makes distances LARGER
                            - **Points below diagonal**: Parameter makes distances SMALLER
                            - **Points on diagonal**: Parameter has no effect on that pair
                            - Look for systematic patterns or outlier points
                            
                            **Correlation Heatmap:**
                            - **High correlation (red, ~1.0)**: Variants rank configuration pairs similarly
                            - **Low correlation (blue, ~0)**: Variants disagree on rankings
                            - Fundamental vs Buffer+Rough: How much do ALL parameters change results?
                            
                            **What to look for:**
                            1. **Large differences in mean**: Parameters significantly affect your data
                            2. **Points far from diagonal**: Some pairs very sensitive to parameters
                            3. **Low correlations**: Different variants capture different aspects
                            4. **High correlations**: Variants largely agree, choose simplest (fundamental)
                            """)
                # End of Parameter Impact Analysis expander
                
                # =================================================================
                # SECTION 4: CONFIGURATION SIMILARITY EXPLORER (FEATURE #5)
                # =================================================================
                with st.expander("Configuration Similarity Explorer (Find Similar/Dissimilar Configs)", expanded=False):
                    st.markdown("### Configuration Similarity Explorer")
                    st.caption("Explore neighborhoods and find similar/dissimilar configurations")
                    st.markdown("""
                This tool helps you understand the **similarity landscape** of your configurations:
                
                **Use Cases:**
                - **Find similar configs**: Identify configurations with comparable trajectory behavior
                - **Find dissimilar configs**: Discover configurations that produce very different results
                - **Explore neighborhoods**: Visualize which configurations cluster together
                - **Validate parameter choices**: See if parameter changes create meaningful distinctions
                
                **How it works:**
                1. Select a target configuration to analyze
                2. The tool finds its k nearest and k farthest neighbors in distance space
                3. Visualizations show the configuration's position relative to others
                
                **When to use:**
                - After computing PDP distances to explore the configuration space
                - To understand if certain parameters create distinct behavior patterns
                - To select representative configurations for further analysis
                """)
                
                    # Check if distance matrix is available
                    if 'pdp_distance_matrix' in st.session_state and st.session_state.pdp_distance_matrix is not None:
                        distance_matrix = st.session_state.pdp_distance_matrix
                        config_ids = st.session_state.get('pdp_config_ids', [])
                        
                        if len(config_ids) < 2:
                            st.warning("Need at least 2 configurations for similarity analysis.")
                        else:
                            # Configuration selection
                            st.markdown("### 🎯 Select Target Configuration")
                            
                            col_target, col_k = st.columns([3, 1])
                            
                            with col_target:
                                target_config = st.selectbox(
                                    "Target Configuration",
                                    options=config_ids,
                                    key="similarity_target_config",
                                    help="Configuration to analyze"
                                )
                            
                            with col_k:
                                k_neighbors = st.number_input(
                                    "Number of Neighbors",
                                    min_value=1,
                                    max_value=min(20, len(config_ids) - 1),
                                    value=min(5, len(config_ids) - 1),
                                    key="similarity_k",
                                    help="How many similar/dissimilar configs to show"
                                )
                            
                            if target_config:
                                # Find similar and dissimilar configurations
                                similarity_results = pdp_analysis.find_similar_and_dissimilar_configs(
                                    distance_matrix=distance_matrix,
                                    config_ids=config_ids,
                                    target_config=target_config,
                                    k=k_neighbors
                                )
                                
                                # Display results in two columns
                                st.markdown("### Similar & Dissimilar Configurations")
                                
                                col_sim, col_dissim = st.columns(2)
                                
                                with col_sim:
                                    st.markdown("#### Most Similar")
                                    st.caption(f"Top {k_neighbors} configurations closest to {target_config}")
                                    
                                    similar_data = []
                                    for rank, (config, dist) in enumerate(similarity_results['similar'], 1):
                                        similar_data.append({
                                            'Rank': f"#{rank}",
                                            'Config': config,
                                            'Distance': f"{dist:.2f}"
                                        })
                                    
                                    if similar_data:
                                        similar_df = pd.DataFrame(similar_data)
                                        st.dataframe(similar_df, use_container_width=True, hide_index=True)
                                    else:
                                        st.info("No similar configurations found")
                                
                                with col_dissim:
                                    st.markdown("#### Most Dissimilar")
                                    st.caption(f"Top {k_neighbors} configurations farthest from {target_config}")
                                    
                                    dissimilar_data = []
                                    for rank, (config, dist) in enumerate(similarity_results['dissimilar'], 1):
                                        dissimilar_data.append({
                                            'Rank': f"#{rank}",
                                            'Config': config,
                                            'Distance': f"{dist:.2f}"
                                        })
                                    
                                    if dissimilar_data:
                                        dissimilar_df = pd.DataFrame(dissimilar_data)
                                        st.dataframe(dissimilar_df, use_container_width=True, hide_index=True)
                                    else:
                                        st.info("No dissimilar configurations found")
                                
                                # Neighborhood visualization
                                st.markdown("---")
                                st.markdown("### Neighborhood Graph")
                                st.caption(f"MDS projection showing {target_config}'s neighborhood. Lines connect to {k_neighbors} nearest neighbors.")
                                
                                # Check if cluster labels are available
                                cluster_labels = st.session_state.get('cluster_labels', None)
                                
                                fig_neighborhood = pdp_analysis.create_neighborhood_visualization(
                                    distance_matrix=distance_matrix,
                                    config_ids=config_ids,
                                    target_config=target_config,
                                    cluster_labels=cluster_labels,
                                    k=k_neighbors
                                )
                                render_interactive_chart(fig_neighborhood)
                                
                                st.markdown("""
                                **How to read this:**
                                - **Red star**: Your selected target configuration
                                - **Orange circles**: Nearest neighbors
                                - **Orange lines**: Connections to neighbors
                                - **Gray/colored dots**: All other configurations
                                - **Closer in 2D space** → More similar trajectories
                                """)
                                
                                # Radial distance chart
                                st.markdown("---")
                                st.markdown("### Distance Radial View")
                                st.caption(f"Polar chart showing distances from {target_config} to its {k_neighbors} nearest neighbors")
                                
                                fig_radial = pdp_analysis.create_distance_radial_chart(
                                    distance_matrix=distance_matrix,
                                    config_ids=config_ids,
                                    target_config=target_config,
                                    top_k=k_neighbors
                                )
                                render_interactive_chart(fig_radial)
                                
                                st.markdown("""
                                **How to read this:**
                                - **Radius (distance from center)**: Larger = more different from target
                                - **Color**: Darker = closer, lighter = farther
                                - **Angular position**: Arbitrary (for layout only)
                                - Compare bar heights to see relative similarities
                                """)
                                
                                # Interpretation guide
                                with st.expander("Interpretation Guide"):
                                    st.markdown(f"""
                                    **Understanding {target_config}'s Position:**
                                    
                                    1. **Similar Configurations (small distances):**
                                       - These configs produce trajectory patterns very close to {target_config}
                                       - Parameters likely have similar effects on trajectory behavior
                                       - Could be grouped together for analysis
                                       - Consider if parameters differ: meaningful or redundant?
                                    
                                    2. **Dissimilar Configurations (large distances):**
                                       - These configs produce very different trajectory patterns
                                       - Parameters create distinct behavior
                                       - Useful for understanding parameter impact boundaries
                                       - May represent different "regimes" of trajectory behavior
                                    
                                    3. **Neighborhood Graph:**
                                       - **Dense clusters**: Groups of similar configurations
                                       - **Isolated points**: Unique or extreme parameter combinations
                                       - **Bridge positions**: Configs connecting different clusters
                                       - If clusters align with parameter ranges → parameters matter
                                    
                                    4. **What to look for:**
                                       - **Tight neighborhood**: {target_config} in dense cluster → robust pattern
                                       - **Isolated position**: {target_config} is unique → extreme parameters?
                                       - **Gradual distances**: Smooth transitions between configs
                                       - **Distance jumps**: Sudden changes suggest parameter thresholds
                                    
                                    5. **Next steps:**
                                       - Compare parameter values of similar configs
                                       - Investigate what makes dissimilar configs different
                                       - Use this to select representative configurations
                                       - Validate that distance metric captures meaningful differences
                                    """)
                        # End of Configuration Similarity Explorer expander
                    
                    else:
                        st.info("Compute a distance matrix first (above) to use the Similarity Explorer.")
                
                # =================================================================
                # SECTION 5: CLUSTER QUALITY METRICS (FEATURE #8)
                # =================================================================
                with st.expander("Cluster Quality Metrics (Evaluate Clustering Quality)", expanded=False):
                    st.markdown("### Cluster Quality Metrics")
                    st.caption("Evaluate clustering quality and find optimal number of clusters")
                    st.markdown("""
                This tool provides **comprehensive evaluation** of clustering quality using multiple metrics:
                
                **Quality Metrics:**
                1. **Silhouette Score** (-1 to 1, higher is better)
                   - Measures how similar objects are to their own cluster vs. other clusters
                   - > 0.7: Strong structure
                   - 0.5-0.7: Reasonable structure
                   - < 0.5: Weak structure
                
                2. **Davies-Bouldin Index** (≥0, lower is better)
                   - Ratio of within-cluster to between-cluster distances
                   - Lower values = better separation between clusters
                
                3. **Calinski-Harabasz Score** (≥0, higher is better)
                   - Ratio of between-cluster to within-cluster dispersion
                   - Higher values = denser, better-separated clusters
                
                4. **Elbow Method (Inertia)**
                   - Within-cluster sum of squares
                   - Look for "elbow" where adding clusters yields diminishing returns
                
                **Use Cases:**
                - Determine optimal number of clusters (k)
                - Validate clustering results
                - Compare different clustering approaches
                - Identify if natural clusters exist in your data
                
                    **How it works:**
                    Tests clustering quality for k=2 to k=max_k and identifies optimal k for each metric.
                    """)
                    
                    # Check if distance matrix and clustering are available
                    if 'pdp_distance_matrix' in st.session_state and st.session_state.pdp_distance_matrix is not None:
                        distance_matrix = st.session_state.pdp_distance_matrix
                        config_ids = st.session_state.get('pdp_config_ids', [])
                        
                        n_configs = len(config_ids)
                        
                        if n_configs < 2:
                            st.warning("Need at least 2 configurations for cluster quality analysis.")
                        else:
                            st.markdown("### Configuration")
                            
                            col_minK, col_maxK, col_compute = st.columns([1, 1, 2])
                            
                            with col_minK:
                                min_k = st.number_input(
                                    "Min k (clusters)",
                                        min_value=2,
                                        max_value=max(2, n_configs - 1),
                                        value=2,
                                        key="cluster_quality_min_k",
                                        help="Minimum number of clusters to test"
                                )
                            
                            with col_maxK:
                                max_possible_k = min(10, n_configs - 1)
                                max_k = st.number_input(
                                    "Max k (clusters)",
                                    min_value=min_k,
                                    max_value=max_possible_k,
                                    value=min(5, max_possible_k),
                                    key="cluster_quality_max_k",
                                    help="Maximum number of clusters to test"
                                )
                            
                            with col_compute:
                                st.write("")  # Spacing
                                st.write("")
                                if st.button("🧪 Evaluate Cluster Quality", type="primary", key="compute_cluster_quality"):
                                    with st.spinner(f"Computing quality metrics for k={min_k} to k={max_k}..."):
                                        metrics_results = pdp_analysis.compute_cluster_quality_metrics(
                                            distance_matrix=distance_matrix,
                                            min_k=min_k,
                                            max_k=max_k
                                        )
                                        
                                        st.session_state['cluster_quality_metrics'] = metrics_results
                                    
                                    st.success("Quality evaluation complete!")
                                    st.rerun()
                            
                            # Show results if available
                            if 'cluster_quality_metrics' in st.session_state and st.session_state['cluster_quality_metrics']:
                                metrics_results = st.session_state['cluster_quality_metrics']
                                
                                st.markdown("---")
                                st.markdown("### 📈 Quality Metrics Comparison")
                                
                                # Create and show the multi-panel plot
                                fig_metrics = pdp_analysis.create_quality_metrics_plot(metrics_results)
                                render_interactive_chart(fig_metrics)
                                
                                st.markdown("""
                                **How to read this:**
                                - **Red stars**: Optimal k for each metric
                                - Look for **agreement** across metrics
                                - **Silhouette & CH**: Higher peaks = better clustering
                                - **Davies-Bouldin**: Lower valleys = better clustering
                                - **Inertia**: Look for "elbow" where curve flattens
                                """)
                                
                                # Get optimal k consensus
                                st.markdown("---")
                                st.markdown("### 🎯 Optimal Cluster Recommendations")
                                
                                consensus = pdp_analysis.get_optimal_k_consensus(metrics_results)
                                
                                col_consensus, col_details = st.columns([1, 2])
                                
                                with col_consensus:
                                    st.metric(
                                        "Consensus Recommendation",
                                        f"k = {consensus['consensus_k']}",
                                        delta=consensus['consensus_strength'],
                                        help="Most commonly recommended k across all metrics"
                                    )
                                
                                with col_details:
                                    st.markdown("#### Optimal k by Metric")
                                    optimal_k_data = []
                                    for metric_name, k_val in consensus['optimal_k_per_metric'].items():
                                        optimal_k_data.append({
                                            'Metric': {
                                                'silhouette': '🔹 Silhouette Score',
                                                'davies_bouldin': '🔹 Davies-Bouldin Index',
                                                'calinski_harabasz': '🔹 Calinski-Harabasz Score',
                                                'elbow': '🔹 Elbow Method'
                                            }[metric_name],
                                            'Optimal k': k_val,
                                            'Match': '✓' if k_val == consensus['consensus_k'] else ''
                                        })
                                    
                                    optimal_df = pd.DataFrame(optimal_k_data)
                                    st.dataframe(optimal_df, use_container_width=True, hide_index=True)
                                
                                # Detailed silhouette analysis for current clustering
                                if 'pdp_cluster_labels' in st.session_state and st.session_state['pdp_cluster_labels'] is not None:
                                    st.markdown("---")
                                    st.markdown("### 🔬 Detailed Silhouette Analysis")
                                    st.caption("Per-configuration silhouette scores for current clustering")
                                    
                                    cluster_labels = st.session_state['pdp_cluster_labels']
                                    
                                    fig_silhouette = pdp_analysis.create_silhouette_per_cluster_plot(
                                        distance_matrix=distance_matrix,
                                        cluster_labels=cluster_labels,
                                        config_ids=config_ids
                                    )
                                    render_interactive_chart(fig_silhouette)
                                    
                                    st.markdown("""
                                    **Interpretation:**
                                    - **Each bar**: One configuration's silhouette score
                                    - **Grouped by cluster**: Colors show different clusters
                                    - **Red dashed line**: Average silhouette across all configs
                                    - **Wide bars above average**: Well-clustered configurations
                                    - **Bars below zero**: Configs may be in wrong cluster
                                    - **Uneven cluster sizes**: Some clusters may dominate
                                    """)
                                
                                # Interpretation guide
                                with st.expander("📚 Understanding the Metrics"):
                                    st.markdown(f"""
                                    **Metric Interpretations:**
                                    
                                    1. **Silhouette Score:**
                                       - Range: -1 (wrong cluster) to +1 (perfect cluster)
                                       - > 0.7: Strong, well-separated clusters
                                       - 0.5-0.7: Reasonable structure
                                       - 0.25-0.5: Weak structure, clusters overlap
                                       - < 0.25: No substantial structure
                                    
                                    2. **Davies-Bouldin Index:**
                                       - Lower is better (minimum = 0)
                                       - Measures cluster separation
                                       - < 1.0: Good separation
                                       - 1.0-2.0: Moderate separation
                                       - > 2.0: Poor separation
                                    
                                    3. **Calinski-Harabasz Score:**
                                       - Higher is better
                                       - Ratio of between/within cluster dispersion
                                       - No absolute threshold, compare across k values
                                       - Sharp peak suggests natural cluster count
                                    
                                    4. **Elbow Method (Inertia):**
                                       - Always decreases as k increases
                                       - Look for "elbow" point where improvement slows
                                       - Point of diminishing returns
                                       - Subjective interpretation
                                    
                                    **When Metrics Disagree:**
                                    - Common in real data (no perfect clusters)
                                    - Consider consensus recommendation
                                    - Test multiple k values in your analysis
                                    - Think about domain knowledge: does k make sense?
                                    - Look at dendrogram and MDS for visual validation
                                    
                                    **Red Flags:**
                                    - All metrics suggest k=2: May indicate one outlier cluster
                                    - Metrics wildly disagree: Weak cluster structure
                                    - Silhouette < 0.25 for all k: Configurations may not cluster naturally
                                    - Monotonic trends (no peaks/elbows): Consider other analysis methods
                                    
                                    **Next Steps:**
                                    1. Use recommended k in clustering analysis
                                    2. Examine dendrogram with this k in mind
                                    3. Check MDS projection for visual cluster separation
                                    4. Investigate configurations with low silhouette scores
                                    5. Consider if parameter choices drive clustering
                                    """)
                        # End of Cluster Quality Metrics expander
                    
                    else:
                        st.info("📊 Compute a distance matrix first (above) to use Cluster Quality Metrics.")
    
    elif analysis_method == "Outlier Detection":
        # Call the modular outlier detection function
        outlier_detection.render_outlier_detection_section(
            data=st.session_state.data,
            selected_configs=st.session_state.shared_selected_configs,
            selected_objects=st.session_state.shared_selected_objects
        )
    
    elif analysis_method == "Heat Maps":
        st.header("🔥 Heat Maps")
        try:
            heatmap_df = None
            if uploaded_files:
                first_file = uploaded_files[0]
                first_file.seek(0)
                heatmap_df = pd.read_csv(first_file)
            if heatmap_df is not None:
                fig = create_heatmap(heatmap_df)
                if fig:
                    render_interactive_chart(fig)
            else:
                st.info("Upload at least one CSV file containing sender and receiver identifiers to generate a heat map.")
        except Exception as e:
            st.error(f"Error creating heatmap: {str(e)}")
    
    elif analysis_method == "Clustering":
        st.header("🔍 Hierarchical Clustering Methods")
        
        st.info("""
        **All clustering methods use Hierarchical Ward Linkage clustering.**
        Choose a distance metric based on what aspects of trajectories you want to group:
        - **Features:** General properties (speed, distance, duration, etc.)
        - **Spatial:** Shape and location similarity (Chamfer distance)
        - **Spatiotemporal:** Time-synchronized similarity (DTW distance)
        """)
        
        # Initialize clustering session state
        clustering.initialize_clustering_session_state()
        
        # Method selection
        clustering_method = st.radio(
            "Select Distance Metric:",
            ["Features (Euclidean)", "Spatial (Chamfer)", "Spatiotemporal (DTW)"],
            key="clustering_method_radio",
            horizontal=True
        )

        st.markdown('---')
        
        # Use selections from sidebar
        selected_configs = st.session_state.shared_selected_configs
        selected_objects = st.session_state.shared_selected_objects

        # Time range
        min_time = float(df['tst'].min())
        max_time = float(df['tst'].max())

        col1, col2 = st.columns(2)
        with col1:
            start_time = st.number_input(
                "Start time",
                min_value=min_time,
                max_value=max_time,
                value=min_time,
                step=0.01,
                format="%.2f",
                key="clustering_start"
            )
        with col2:
            end_time = st.number_input(
                "End time",
                min_value=start_time,
                max_value=max_time,
                value=max_time,
                step=0.01,
                format="%.2f",
                key="clustering_end"
            )
        
        # Check if method changed - reset state if so
        if st.session_state.clustering_method != clustering_method:
            st.session_state.clustering_method = clustering_method
            st.session_state.distance_matrix = None
            st.session_state.trajectory_ids = None
            st.session_state.linkage_matrix = None
            st.session_state.optimal_n_clusters = None
            st.session_state.current_n_clusters = None
            st.session_state.cluster_labels = None
            st.session_state.features_df = None
            st.session_state.trajectories = None
        
        st.markdown('---')
        
        # Method-specific UI
        if clustering_method == "Features (Euclidean)":
            st.subheader("🎯 Feature-Based Clustering")
            st.info("Cluster trajectories based on extracted features: distance, speed, duration, sinuosity, etc.")
            
            # Feature selection
            st.markdown("#### Select Features to Use")
            all_features = [
                'total_distance',
                'duration',
                'avg_speed',
                'net_displacement',
                'sinuosity',
                'bbox_area',
                'avg_direction',
                'max_speed'
            ]
            
            feature_labels = {
                'total_distance': '📏 Total Distance',
                'duration': '⏱️ Duration',
                'avg_speed': '🏃 Average Speed',
                'net_displacement': '📐 Net Displacement',
                'sinuosity': '🌀 Sinuosity (Path Efficiency)',
                'bbox_area': '📦 Bounding Box Area',
                'avg_direction': '🧭 Average Direction',
                'max_speed': '⚡ Maximum Speed'
            }
            
            # Initialize default selection in session state if not exists
            if 'feature_selection_default' not in st.session_state:
                st.session_state.feature_selection_default = all_features
            
            selected_features = st.multiselect(
                "Choose which features to include in the distance calculation:",
                options=all_features,
                default=st.session_state.feature_selection_default,
                format_func=lambda x: feature_labels[x],
                key="selected_features"
            )
            
            if not selected_features:
                st.warning("⚠️ Please select at least one feature to proceed.")
            else:
                st.success(f"✅ {len(selected_features)} feature(s) selected")
            
            # Compute distance matrix button
            if st.button("🔄 Compute Feature Distance Matrix", key="compute_features", disabled=not selected_features):
                with st.spinner(f"Extracting {len(selected_features)} feature(s) and computing distances..."):
                    try:
                        distance_matrix, trajectory_ids, features_df, trajectories = clustering.compute_feature_distance_matrix(
                            df, selected_configs, selected_objects, start_time, end_time, selected_features
                        )
                        
                        if distance_matrix is None:
                            st.error("❌ No valid trajectories found with the current filters.")
                        else:
                            st.session_state.distance_matrix = distance_matrix
                            st.session_state.trajectory_ids = trajectory_ids
                            st.session_state.features_df = features_df
                            st.session_state.trajectories = trajectories
                            st.success(f"✅ Computed distance matrix for {len(trajectory_ids)} trajectories using {len(selected_features)} features!")
                    except Exception as e:
                        st.error(f"Error computing distances: {str(e)}")
            
            # Show features if computed
            if st.session_state.features_df is not None:
                with st.expander("📋 Extracted Features"):
                    formatted_df = clustering.format_features_dataframe(st.session_state.features_df)
                    st.dataframe(formatted_df)
        
        elif clustering_method == "Spatial (Chamfer)":
            st.subheader("📍 Spatial Clustering (Chamfer Distance)")
            st.info("Cluster trajectories based on spatial shape and location similarity using Chamfer distance.")
            
            # Compute distance matrix button
            if st.button("🔄 Compute Chamfer Distance Matrix", key="compute_chamfer"):
                with st.spinner("Computing Chamfer distances..."):
                    try:
                        distance_matrix, trajectory_ids, trajectories = clustering.compute_chamfer_distance_matrix(
                            df, selected_configs, selected_objects, start_time, end_time
                        )
                        
                        if distance_matrix is None:
                            st.error("❌ No valid trajectories found with the current filters.")
                        else:
                            st.session_state.distance_matrix = distance_matrix
                            st.session_state.trajectory_ids = trajectory_ids
                            st.session_state.trajectories = trajectories
                            st.success(f"✅ Computed distance matrix for {len(trajectory_ids)} trajectories!")
                    except Exception as e:
                        st.error(f"Error computing distances: {str(e)}")
        
        elif clustering_method == "Spatiotemporal (DTW)":
            st.subheader("⏱️ Spatiotemporal Clustering (DTW Distance)")
            st.info("Cluster trajectories based on spatiotemporal similarity using Dynamic Time Warping (DTW).")
            
            # Compute distance matrix button
            if st.button("🔄 Compute DTW Distance Matrix", key="compute_dtw"):
                with st.spinner("Computing DTW distances... This may take a while for many trajectories."):
                    try:
                        distance_matrix, trajectory_ids, trajectories = clustering.compute_dtw_distance_matrix(
                            df, selected_configs, selected_objects, start_time, end_time
                        )
                        
                        if distance_matrix is None:
                            st.error("❌ No valid trajectories found with the current filters.")
                        else:
                            st.session_state.distance_matrix = distance_matrix
                            st.session_state.trajectory_ids = trajectory_ids
                            st.session_state.trajectories = trajectories
                            st.success(f"✅ Computed distance matrix for {len(trajectory_ids)} trajectories!")
                    except Exception as e:
                        st.error(f"Error computing distances: {str(e)}")
        
        # Show distance matrix visualization if available
        if st.session_state.distance_matrix is not None:
            st.markdown('---')
            st.subheader("📊 Distance Matrix Heatmap")
            
            distance_matrix = st.session_state.distance_matrix
            trajectory_ids = st.session_state.trajectory_ids
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=distance_matrix,
                x=trajectory_ids,
                y=trajectory_ids,
                colorscale='Viridis',
                colorbar=dict(title="Distance"),
                hovertemplate='From: %{y}<br>To: %{x}<br>Distance: %{z:.2f}<extra></extra>'
            ))
            
            fig_heatmap.update_layout(
                title="Pairwise Distance Matrix",
                xaxis_title="Trajectory",
                yaxis_title="Trajectory",
                height=min(600, max(400, len(trajectory_ids) * 20)),
                width=min(800, max(500, len(trajectory_ids) * 20))
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            st.markdown('---')
            
            # ========================================
            # STEP 4: Dendrogram & Cluster Assignment
            # ========================================
            st.subheader("🌳 Hierarchical Clustering - Dendrogram & Cluster Assignment")
            
            st.info("""
            **Dendrogram Visualization**: Shows the hierarchical structure of trajectory clustering.
            - Each leaf represents a trajectory
            - Height indicates dissimilarity between merged clusters
            - Use the slider to cut the dendrogram at different heights (select number of clusters)
            """)
            
            # Create linkage matrix for hierarchical clustering
            # Using Ward linkage for trajectory clustering
            # Convert square distance matrix to condensed form
            from scipy.spatial.distance import squareform
            condensed_dist = squareform(st.session_state.distance_matrix, checks=False)
            linkage_matrix = linkage(condensed_dist, method='ward')
            
            # Create dendrogram visualization
            st.markdown("#### Dendrogram")
            
            # Use scipy to create dendrogram data
            from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
            dendro_data = scipy_dendrogram(
                linkage_matrix,
                labels=[f"T{tid}" for tid in st.session_state.trajectory_ids],
                no_plot=True
            )
            
            # Create plotly dendrogram
            icoord = np.array(dendro_data['icoord'])
            dcoord = np.array(dendro_data['dcoord'])
            colors = dendro_data['color_list']
            labels = dendro_data['ivl']
            
            # Convert matplotlib color codes to Plotly-compatible colors
            color_map = {
                'C0': '#1f77b4', 'C1': '#ff7f0e', 'C2': '#2ca02c', 'C3': '#d62728',
                'C4': '#9467bd', 'C5': '#8c564b', 'C6': '#e377c2', 'C7': '#7f7f7f',
                'C8': '#bcbd22', 'C9': '#17becf', 'b': 'blue', 'g': 'green',
                'r': 'red', 'c': 'cyan', 'm': 'magenta', 'y': 'yellow', 'k': 'black'
            }
            plotly_colors = [color_map.get(c, c) for c in colors]
            
            fig_dendro = go.Figure()
            
            # Add dendrogram lines
            for i, (xi, yi) in enumerate(zip(icoord, dcoord)):
                fig_dendro.add_trace(go.Scatter(
                    x=xi,
                    y=yi,
                    mode='lines',
                    line=dict(color=plotly_colors[i], width=2),
                    hoverinfo='skip',
                    showlegend=False
                ))
            
            # Add labels at bottom
            n_leaves = len(labels)
            x_positions = [5 + i * 10 for i in range(n_leaves)]
            
            fig_dendro.update_layout(
                title="Hierarchical Clustering Dendrogram (Ward Linkage)",
                xaxis=dict(
                    title="Trajectory",
                    tickmode='array',
                    tickvals=x_positions,
                    ticktext=labels,
                    tickangle=-45
                ),
                yaxis=dict(title="Distance"),
                height=500,
                hovermode='closest',
                plot_bgcolor='white',
                showlegend=False
            )
            
            st.plotly_chart(fig_dendro, use_container_width=True)
            
            st.markdown("---")
            st.markdown("#### Cluster Assignment")
            
            # Cluster selection controls
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Number of clusters slider
                n_trajectories = len(st.session_state.trajectory_ids)
                max_clusters = min(20, n_trajectories - 1)
                
                
                n_clusters = st.slider(
                    "Number of clusters",
                    min_value=2,
                    max_value=max_clusters,
                    value=min(3, max_clusters),
                    help="Slide to select how many clusters to create",
                    key="n_clusters_slider"
                )
            
            with col2:
                    # Auto-detect optimal clusters button
                    if st.button("🎯 Auto-detect Optimal Clusters", help="Use elbow method to recommend optimal number of clusters."):
                        with st.spinner("Detecting optimal number of clusters..."):
                            optimal_k, plot_data = clustering.detect_optimal_clusters(st.session_state.distance_matrix, return_plot_data=True)
                            if optimal_k is not None:
                                st.success(f"✅ Recommended number of clusters: **{optimal_k}**")
                                
                                # Display elbow plot
                                fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
                                
                                fig.add_trace(go.Scatter(x=plot_data["k_values"], y=plot_data["inertias"], mode="lines+markers",
                                    name="Inertia", line=dict(color="blue", width=2), marker=dict(size=8)), secondary_y=False)
                                
                                fig.add_trace(go.Scatter(x=plot_data["k_values"], y=plot_data["silhouette_scores"],
                                    mode="lines+markers", name="Silhouette Score", line=dict(color="green", width=2),
                                    marker=dict(size=8)), secondary_y=True)
                                
                                fig.add_vline(x=optimal_k, line=dict(color="red", width=2, dash="dash"),
                                    annotation_text=f"Optimal k={optimal_k}", annotation_position="top")
                                
                                fig.update_xaxes(title_text="Number of Clusters (k)")
                                fig.update_yaxes(title_text="Inertia", secondary_y=False)
                                fig.update_yaxes(title_text="Silhouette Score", secondary_y=True)
                                fig.update_layout(title="Elbow Plot", hovermode="x unified", height=400)
                                
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Could not automatically detect optimal clusters. Please select manually.")
                
                         
            # Assign clusters based on selected number
            from scipy.cluster.hierarchy import fcluster
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            
            # Store cluster assignments in session state
            st.session_state.cluster_labels = cluster_labels
            st.session_state.n_clusters = n_clusters
            
            # Display cluster assignment summary
            st.markdown(f"**Cluster Assignment Summary** ({n_clusters} clusters)")
            
            # Create a dataframe showing cluster assignments
            cluster_df = pd.DataFrame({
                'Trajectory ID': st.session_state.trajectory_ids,
                'Cluster': cluster_labels
            })
            
            # Count trajectories per cluster
            cluster_counts = cluster_df['Cluster'].value_counts().sort_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Trajectories per Cluster:**")
                for cluster_id in sorted(cluster_counts.index):
                    count = cluster_counts[cluster_id]
                    st.write(f"• Cluster {cluster_id}: **{count}** trajectories")
            
            with col2:
                # Show cluster assignments table
                st.markdown("**Cluster Assignments:**")
                st.dataframe(
                    cluster_df.sort_values('Cluster'),
                    height=min(300, len(cluster_df) * 35 + 38),
                    use_container_width=True
                )
            
            st.markdown('---')
            st.success(f"✅ Successfully assigned {n_trajectories} trajectories into {n_clusters} clusters using Ward linkage!")
            
            # ===========================
            # ANALYSIS TOOLS
            # ===========================
            
            st.markdown('---')
            st.markdown("### 🔬 Analysis Tools")
            
            st.info("""
            **Advanced Analysis**: Explore cluster quality and trajectory relationships
            - **MDS Visualization**: Project high-dimensional data to 2D/3D space
            - **Similarity Search**: Find most similar trajectories to a reference
            - **Silhouette Analysis**: Evaluate cluster quality metrics
            """)
            
            # Create tabs for different analysis tools
            analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs([
                "📊 MDS Visualization", 
                "🔍 Similarity Search", 
                "📈 Silhouette Analysis"
            ])
            
            # ===========================
            # TAB 1: MDS VISUALIZATION
            # ===========================
            with analysis_tab1:
                st.markdown("#### Multidimensional Scaling (MDS)")
                st.markdown("Visualize trajectory clusters in 2D or 3D space based on their pairwise distances.")
                
                # MDS dimension selection
                col1, col2 = st.columns([1, 3])
                with col1:
                    mds_dims = st.radio(
                        "Dimensions",
                        options=[2, 3],
                        index=0,
                        help="Choose 2D or 3D visualization"
                    )
                
                with col2:
                    if st.button("🎨 Generate MDS Plot", help="Click to compute and visualize MDS projection"):
                        with st.spinner(f"Computing {mds_dims}D MDS projection..."):
                            from sklearn.manifold import MDS
                            
                            # Compute MDS
                            mds = MDS(n_components=mds_dims, dissimilarity='precomputed', random_state=42)
                            mds_coords = mds.fit_transform(st.session_state.distance_matrix)
                            
                            # Calculate normalized stress (Kruskal's Stress-1)
                            # This gives values between 0 and 1 (or 0-100%)
                            from scipy.spatial.distance import pdist, squareform
                            mds_distances = squareform(pdist(mds_coords))
                            original_distances = st.session_state.distance_matrix
                            
                            # Kruskal's Stress-1 formula: sqrt(sum((d_orig - d_mds)^2) / sum(d_orig^2))
                            stress_normalized = np.sqrt(np.sum((original_distances - mds_distances) ** 2) / np.sum(original_distances ** 2))
                            
                            # Create color palette for clusters
                            import plotly.express as px
                            colors = px.colors.qualitative.Plotly[:n_clusters]
                            
                            # Create plotly figure
                            if mds_dims == 2:
                                fig_mds = go.Figure()
                                
                                for cluster_id in range(1, n_clusters + 1):
                                    mask = cluster_labels == cluster_id
                                    cluster_trajectories = np.array(st.session_state.trajectory_ids)[mask]
                                    
                                    fig_mds.add_trace(go.Scatter(
                                        x=mds_coords[mask, 0],
                                        y=mds_coords[mask, 1],
                                        mode='markers+text',
                                        marker=dict(
                                            size=12,
                                            color=colors[cluster_id - 1],
                                            line=dict(width=1, color='white')
                                        ),
                                        text=[f"T{tid}" for tid in cluster_trajectories],
                                        textposition="top center",
                                        textfont=dict(size=9),
                                        name=f"Cluster {cluster_id}",
                                        hovertemplate='<b>Trajectory %{text}</b><br>Cluster: ' + str(cluster_id) + '<extra></extra>'
                                    ))
                                
                                fig_mds.update_layout(
                                    title="2D MDS Projection of Trajectory Clusters",
                                    xaxis_title="MDS Dimension 1",
                                    yaxis_title="MDS Dimension 2",
                                    height=600,
                                    hovermode='closest',
                                    showlegend=True
                                )
                                
                            else:  # 3D
                                fig_mds = go.Figure()
                                
                                for cluster_id in range(1, n_clusters + 1):
                                    mask = cluster_labels == cluster_id
                                    cluster_trajectories = np.array(st.session_state.trajectory_ids)[mask]
                                    
                                    fig_mds.add_trace(go.Scatter3d(
                                        x=mds_coords[mask, 0],
                                        y=mds_coords[mask, 1],
                                        z=mds_coords[mask, 2],
                                        mode='markers+text',
                                        marker=dict(
                                            size=8,
                                            color=colors[cluster_id - 1],
                                            line=dict(width=1, color='white')
                                        ),
                                        text=[f"T{tid}" for tid in cluster_trajectories],
                                        textposition="top center",
                                        textfont=dict(size=8),
                                        name=f"Cluster {cluster_id}",
                                        hovertemplate='<b>Trajectory %{text}</b><br>Cluster: ' + str(cluster_id) + '<extra></extra>'
                                    ))
                                
                                fig_mds.update_layout(
                                    title="3D MDS Projection of Trajectory Clusters",
                                    scene=dict(
                                        xaxis_title="MDS Dimension 1",
                                        yaxis_title="MDS Dimension 2",
                                        zaxis_title="MDS Dimension 3"
                                    ),
                                    height=700,
                                    hovermode='closest',
                                    showlegend=True
                                )
                            
                            st.plotly_chart(fig_mds, use_container_width=True)
                            st.success(f"✅ {mds_dims}D MDS projection computed successfully!")
                            st.info(f"**Normalized Stress (Kruskal's Stress-1)**: {stress_normalized:.4f} ({stress_normalized*100:.2f}%) — Lower is better: <0.05 (5%) excellent, <0.10 (10%) good, <0.20 (20%) acceptable")
            
            # ===========================
            # TAB 2: SIMILARITY SEARCH
            # ===========================
            with analysis_tab2:
                st.markdown("#### Top-K Similar Trajectories")
                st.markdown("Find trajectories most similar to a selected reference trajectory.")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Select reference trajectory
                    reference_idx = st.selectbox(
                        "Select reference trajectory",
                        options=range(len(st.session_state.trajectory_ids)),
                        format_func=lambda i: f"Trajectory {st.session_state.trajectory_ids[i]} (Cluster {cluster_labels[i]})",
                        help="Choose a trajectory to find similar ones"
                    )
                
                with col2:
                    # Select number of similar trajectories to show
                    k_similar = st.slider(
                        "Number of similar trajectories (K)",
                        min_value=1,
                        max_value=min(20, len(st.session_state.trajectory_ids) - 1),
                        value=5,
                        help="How many similar trajectories to display"
                    )
                
                if st.button("🔍 Find Similar Trajectories"):
                    with st.spinner("Searching for similar trajectories..."):
                        # Get distances from reference trajectory to all others
                        distances = st.session_state.distance_matrix[reference_idx].copy()
                        
                        # Set distance to self as infinity to exclude it
                        distances[reference_idx] = np.inf
                        
                        # Find K most similar (smallest distances)
                        similar_indices = np.argsort(distances)[:k_similar]
                        
                        # Create results dataframe
                        results_df = pd.DataFrame({
                            'Rank': range(1, k_similar + 1),
                            'Trajectory ID': [st.session_state.trajectory_ids[i] for i in similar_indices],
                            'Cluster': [cluster_labels[i] for i in similar_indices],
                            'Distance': distances[similar_indices],
                            'Similarity Score': 1 / (1 + distances[similar_indices])  # Convert distance to similarity
                        })
                        
                        # Display reference info
                        ref_tid = st.session_state.trajectory_ids[reference_idx]
                        ref_cluster = cluster_labels[reference_idx]
                        
                        st.markdown(f"**Reference Trajectory**: T{ref_tid} (Cluster {ref_cluster})")
                        st.markdown(f"**Top {k_similar} Most Similar Trajectories:**")
                        
                        # Format and display results
                        st.dataframe(
                            results_df.style.format({
                                'Distance': '{:.4f}',
                                'Similarity Score': '{:.4f}'
                            }).background_gradient(subset=['Similarity Score'], cmap='Greens'),
                            use_container_width=True,
                            height=min(400, len(results_df) * 35 + 38)
                        )
                        
                        # Cluster distribution analysis
                        same_cluster = sum(results_df['Cluster'] == ref_cluster)
                        st.markdown(f"**Cluster Analysis**: {same_cluster}/{k_similar} similar trajectories are in the same cluster as the reference")
                        
                        if same_cluster == k_similar:
                            st.success("✅ All similar trajectories are in the same cluster - excellent clustering!")
                        elif same_cluster >= k_similar * 0.7:
                            st.info("ℹ️ Most similar trajectories are in the same cluster - good clustering quality")
                        else:
                            st.warning("⚠️ Many similar trajectories are in different clusters - consider adjusting cluster count")
            
            # ===========================
            # TAB 3: SILHOUETTE ANALYSIS
            # ===========================
            with analysis_tab3:
                st.markdown("#### Silhouette Analysis")
                st.markdown("Evaluate cluster quality using silhouette coefficients. Values range from -1 to 1:")
                st.markdown("- **Close to 1**: Well-clustered, far from neighboring clusters")
                st.markdown("- **Close to 0**: Near the decision boundary between clusters")
                st.markdown("- **Negative**: Possibly assigned to wrong cluster")
                
                if st.button("📊 Calculate Silhouette Scores"):
                    with st.spinner("Computing silhouette analysis..."):
                        from sklearn.metrics import silhouette_score, silhouette_samples
                        
                        # Compute silhouette scores
                        # Convert distance matrix to similarity for silhouette calculation
                        overall_score = silhouette_score(st.session_state.distance_matrix, cluster_labels, metric='precomputed')
                        sample_scores = silhouette_samples(st.session_state.distance_matrix, cluster_labels, metric='precomputed')
                        
                        # Display overall score
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Overall Silhouette Score", f"{overall_score:.4f}")
                        with col2:
                            st.metric("Number of Clusters", n_clusters)
                        with col3:
                            avg_cluster_size = len(cluster_labels) / n_clusters
                            st.metric("Avg Cluster Size", f"{avg_cluster_size:.1f}")
                        
                        # Quality interpretation
                        if overall_score > 0.7:
                            st.success("🌟 **Excellent** clustering structure!")
                        elif overall_score > 0.5:
                            st.success("✅ **Good** clustering quality")
                        elif overall_score > 0.3:
                            st.info("ℹ️ **Moderate** clustering quality")
                        else:
                            st.warning("⚠️ **Poor** clustering - consider different parameters")
                        
                        st.markdown("---")
                        st.markdown("**Per-Cluster Silhouette Scores:**")
                        
                        # Create per-cluster analysis
                        cluster_stats = []
                        for cluster_id in range(1, n_clusters + 1):
                            mask = cluster_labels == cluster_id
                            cluster_scores = sample_scores[mask]
                            
                            cluster_stats.append({
                                'Cluster': cluster_id,
                                'Size': mask.sum(),
                                'Mean Score': cluster_scores.mean(),
                                'Min Score': cluster_scores.min(),
                                'Max Score': cluster_scores.max(),
                                'Std Dev': cluster_scores.std()
                            })
                        
                        cluster_stats_df = pd.DataFrame(cluster_stats)
                        
                        # Display cluster statistics
                        st.dataframe(
                            cluster_stats_df.style.format({
                                'Mean Score': '{:.4f}',
                                'Min Score': '{:.4f}',
                                'Max Score': '{:.4f}',
                                'Std Dev': '{:.4f}'
                            }).background_gradient(subset=['Mean Score'], cmap='RdYlGn'),
                            use_container_width=True
                        )
                        
                        # Create silhouette plot
                        import plotly.express as px
                        colors = px.colors.qualitative.Plotly[:n_clusters]
                        
                        fig_silhouette = go.Figure()
                        
                        y_lower = 10
                        for cluster_id in range(1, n_clusters + 1):
                            mask = cluster_labels == cluster_id
                            cluster_scores = sample_scores[mask]
                            cluster_scores.sort()
                            
                            y_upper = y_lower + len(cluster_scores)
                            
                            fig_silhouette.add_trace(go.Bar(
                                x=cluster_scores,
                                y=list(range(y_lower, y_upper)),
                                orientation='h',
                                name=f"Cluster {cluster_id}",
                                marker=dict(color=colors[cluster_id - 1]),
                                hovertemplate='Silhouette Score: %{x:.4f}<extra></extra>'
                            ))
                            
                            y_lower = y_upper + 10
                        
                        # Add vertical line for overall average
                        fig_silhouette.add_vline(
                            x=overall_score,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"Overall Average: {overall_score:.4f}",
                            annotation_position="top right"
                        )
                        
                        fig_silhouette.update_layout(
                            title="Silhouette Plot for All Clusters",
                            xaxis_title="Silhouette Coefficient",
                            yaxis_title="Trajectory Index (grouped by cluster)",
                            height=max(400, n_trajectories * 3),
                            showlegend=True,
                            barmode='overlay'
                        )
                        
                        st.plotly_chart(fig_silhouette, use_container_width=True)
                        
                        # Identify problematic trajectories
                        negative_scores = sample_scores < 0
                        if negative_scores.any():
                            st.warning(f"⚠️ **{negative_scores.sum()} trajectories** have negative silhouette scores (possibly misclassified)")
                            
                            problematic_df = pd.DataFrame({
                                'Trajectory ID': np.array(st.session_state.trajectory_ids)[negative_scores],
                                'Cluster': cluster_labels[negative_scores],
                                'Silhouette Score': sample_scores[negative_scores]
                            }).sort_values('Silhouette Score')
                            
                            with st.expander("Show problematic trajectories"):
                                st.dataframe(
                                    problematic_df.style.format({'Silhouette Score': '{:.4f}'}),
                                    use_container_width=True
                                )
                        else:
                            st.success("✅ All trajectories have positive silhouette scores!")
            
            st.markdown('---')
            st.success("✅ Step 5 analysis tools are ready! Use the tabs above to explore your clusters.")
            
            # ===========================
            # CLUSTER VISUALIZATIONS
            # ===========================
            
            st.markdown('---')
            st.markdown("### 🎨 Cluster Visualizations")
            
            st.info("""
            **Visualize Trajectories by Cluster**: See how trajectories are grouped spatially and temporally
            - **2D Trajectory Plots**: View trajectories colored by cluster assignment
            - **3D Spatiotemporal View**: Explore X, Y, Time dimensions with cluster colors
            - **Cluster Comparison**: Compare individual clusters side-by-side
            """)
            
            # Create tabs for different visualization types
            viz_tab1, viz_tab2, viz_tab3 = st.tabs([
                "� 2D Spatial View", 
                "🌐 3D Spatiotemporal View", 
                "🔄 Cluster Comparison"
            ])
            
            # ===========================
            # TAB 1: 2D SPATIAL VIEW
            # ===========================
            with viz_tab1:
                st.markdown("#### 2D Trajectory Visualization by Cluster")
                st.markdown("All trajectories plotted in X-Y space, colored by cluster assignment.")
                
                # Check if trajectory data is available
                if 'trajectories' not in st.session_state or st.session_state.trajectories is None:
                    st.warning("⚠️ No trajectory data available. Please compute the distance matrix first in Step 3.")
                elif 'cluster_labels' not in st.session_state or st.session_state.cluster_labels is None:
                    st.warning("⚠️ No cluster assignments available. Please assign clusters using the slider above.")
                elif st.button("🎨 Generate 2D Cluster Plot", key="btn_2d_cluster"):
                    with st.spinner("Generating 2D visualization..."):
                        import plotly.express as px
                        
                        # Get cluster data from session state
                        cluster_labels = st.session_state.cluster_labels
                        n_clusters = st.session_state.n_clusters
                        
                        # Create trajectory dictionary mapping
                        trajectories_dict = {tid: traj for tid, traj in zip(st.session_state.trajectory_ids, st.session_state.trajectories)}
                        
                        # Get unique cluster IDs that actually exist in the data
                        unique_clusters = sorted(np.unique(cluster_labels))
                        
                        # Create color palette based on actual number of clusters
                        colors = px.colors.qualitative.Plotly[:len(unique_clusters)]
                        
                        # Start with tennis court
                        fig_2d = create_tennis_court()
                        
                        # Plot each cluster (only clusters that actually exist)
                        for idx, cluster_id in enumerate(unique_clusters):
                            mask = cluster_labels == cluster_id
                            cluster_trajectory_ids = np.array(st.session_state.trajectory_ids)[mask]
                            
                            for tid in cluster_trajectory_ids:
                                # Get trajectory data
                                traj_data = trajectories_dict[tid]
                                
                                # Add trajectory line and markers
                                fig_2d.add_trace(go.Scatter(
                                    x=traj_data[:, 0],  # X coordinates
                                    y=traj_data[:, 1],  # Y coordinates
                                    mode='lines+markers',
                                    name=f"T{tid} (C{cluster_id})",
                                    line=dict(color=colors[idx], width=2),
                                    marker=dict(
                                        size=[4] * (len(traj_data) - 1) + [0],  # Hide last marker
                                        color=colors[idx]
                                    ),
                                    legendgroup=f"cluster_{cluster_id}",
                                    hovertemplate=f'<b>Trajectory {tid}</b><br>Cluster: {cluster_id}<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>'
                                ))
                                
                                # Add arrow at the end
                                if len(traj_data) >= 2:
                                    dx = traj_data[-1, 0] - traj_data[-2, 0]
                                    dy = traj_data[-1, 1] - traj_data[-2, 1]
                                    angle = np.degrees(np.arctan2(dx, dy))
                                    
                                    fig_2d.add_trace(go.Scatter(
                                        x=[traj_data[-1, 0]],
                                        y=[traj_data[-1, 1]],
                                        mode='markers',
                                        marker=dict(
                                            symbol='arrow',
                                            color=colors[idx],
                                            size=15,
                                            angle=angle
                                        ),
                                        legendgroup=f"cluster_{cluster_id}",
                                        showlegend=False,
                                        hoverinfo='skip'
                                    ))
                        
                        fig_2d.update_layout(
                            title=f"2D Trajectory Clusters (n={n_clusters})",
                            height=900,
                            hovermode='closest',
                            showlegend=True,
                            legend=dict(
                                title="Trajectories",
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=1.01
                            ),
                            uirevision='constant'
                        )
                        
                        render_interactive_chart(fig_2d)
                        
                        # Cluster statistics
                        st.markdown("**Cluster Distribution:**")
                        col1, col2, col3 = st.columns(3)
                        
                        for i, cluster_id in enumerate(range(1, n_clusters + 1)):
                            count = (cluster_labels == cluster_id).sum()
                            with [col1, col2, col3][i % 3]:
                                st.metric(
                                    f"Cluster {cluster_id}",
                                    f"{count} trajectories",
                                    delta=f"{count/len(cluster_labels)*100:.1f}%"
                                )
            
            # ===========================
            # TAB 2: 3D SPATIOTEMPORAL VIEW
            # ===========================
            with viz_tab2:
                st.markdown("#### 3D Spatiotemporal Visualization")
                st.markdown("Trajectories in 3D space (X, Y, Time), colored by cluster.")
                
                # Check if trajectory data is available
                if 'trajectories' not in st.session_state or st.session_state.trajectories is None:
                    st.warning("⚠️ No trajectory data available. Please compute the distance matrix first in Step 3.")
                elif 'cluster_labels' not in st.session_state or st.session_state.cluster_labels is None:
                    st.warning("⚠️ No cluster assignments available. Please assign clusters using the slider above.")
                else:
                    if st.button("🌐 Regenerate 3D Plot", key="btn_3d_cluster"):
                        # Clear the cached plot to force regeneration
                        if 'fig_3d_cluster' in st.session_state:
                            del st.session_state.fig_3d_cluster
                    
                    # Generate plot on first load or if regenerate button was clicked
                    if 'fig_3d_cluster' not in st.session_state:
                        with st.spinner("Generating 3D spatiotemporal visualization..."):
                            import plotly.express as px
                            
                            # Get cluster data from session state
                            cluster_labels = st.session_state.cluster_labels
                            n_clusters = st.session_state.n_clusters
                            
                            # Create trajectory dictionary mapping
                            trajectories_dict = {tid: traj for tid, traj in zip(st.session_state.trajectory_ids, st.session_state.trajectories)}
                            
                            # Get unique cluster IDs that actually exist in the data
                            unique_clusters = sorted(np.unique(cluster_labels))
                            
                            # Create color palette based on actual number of clusters
                            colors = px.colors.qualitative.Plotly[:len(unique_clusters)]
                            
                            fig_3d = go.Figure()
                            
                            # Tennis court dimensions
                            court_width = 8.23
                            court_length = 23.77
                            doubles_width = 10.97
                            doubles_alley_width = (doubles_width - court_width) / 2
                            
                            # Add tennis court as a surface at z=0
                            court_x = np.array([[-doubles_alley_width, court_width + doubles_alley_width],
                                               [-doubles_alley_width, court_width + doubles_alley_width]])
                            court_y = np.array([[0, 0],
                                               [court_length, court_length]])
                            court_z = np.array([[0, 0],
                                               [0, 0]])
                            
                            fig_3d.add_trace(go.Surface(
                                x=court_x,
                                y=court_y,
                                z=court_z,
                                colorscale=[[0, '#2ECC71'], [1, '#2ECC71']],  # Tennis court green
                                showscale=False,
                                opacity=0.7,
                                name='Tennis Court',
                                hoverinfo='skip',
                                showlegend=False
                            ))
                            
                            # Add court lines as 3D lines at z=0
                            def add_court_line_3d(x0, y0, x1, y1, color='white', width=2):
                                fig_3d.add_trace(go.Scatter3d(
                                    x=[x0, x1],
                                    y=[y0, y1],
                                    z=[0, 0],
                                    mode='lines',
                                    line=dict(color=color, width=width),
                                    showlegend=False,
                                    hoverinfo='skip'
                                ))
                            
                            # Court boundary (doubles)
                            add_court_line_3d(-doubles_alley_width, 0, court_width + doubles_alley_width, 0, width=3)
                            add_court_line_3d(-doubles_alley_width, court_length, court_width + doubles_alley_width, court_length, width=3)
                            add_court_line_3d(-doubles_alley_width, 0, -doubles_alley_width, court_length, width=3)
                            add_court_line_3d(court_width + doubles_alley_width, 0, court_width + doubles_alley_width, court_length, width=3)
                            
                            # Singles sidelines
                            add_court_line_3d(0, 0, 0, court_length)
                            add_court_line_3d(court_width, 0, court_width, court_length)
                            
                            # Net line
                            net_position = court_length / 2
                            add_court_line_3d(-doubles_alley_width, net_position, court_width + doubles_alley_width, net_position)
                            
                            # Service lines
                            service_line_distance = 6.40
                            service_line_bottom = net_position - service_line_distance
                            service_line_top = net_position + service_line_distance
                            add_court_line_3d(0, service_line_bottom, court_width, service_line_bottom)
                            add_court_line_3d(0, service_line_top, court_width, service_line_top)
                            
                            # Center service line
                            center_x = court_width / 2
                            add_court_line_3d(center_x, service_line_bottom, center_x, service_line_top)
                            
                            # Plot each cluster in 3D (only clusters that actually exist)
                            for idx, cluster_id in enumerate(unique_clusters):
                                mask = cluster_labels == cluster_id
                                cluster_trajectory_ids = np.array(st.session_state.trajectory_ids)[mask]
                            
                                for tid in cluster_trajectory_ids:
                                    traj_data = trajectories_dict[tid]
                                    
                                    # Create time dimension (assuming equal time steps)
                                    time_steps = np.arange(len(traj_data))
                                    
                                    fig_3d.add_trace(go.Scatter3d(
                                        x=traj_data[:, 0],  # X
                                        y=traj_data[:, 1],  # Y
                                        z=time_steps,        # Time
                                        mode='lines+markers',
                                        name=f"T{tid} (C{cluster_id})",
                                        line=dict(color=colors[idx], width=3),
                                        marker=dict(size=3, color=colors[idx]),
                                        legendgroup=f"cluster_{cluster_id}",
                                        hovertemplate=f'<b>Trajectory {tid}</b><br>Cluster: {cluster_id}<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<br>Time: %{{z}}<extra></extra>'
                                    ))
                        
                            fig_3d.update_layout(
                                title=f"3D Spatiotemporal Trajectory Clusters (n={n_clusters})",
                                scene=dict(
                                    xaxis_title="X Coordinate (m)",
                                    yaxis_title="Y Coordinate (m)",
                                    zaxis_title="Time Step",
                                    camera=dict(
                                        eye=dict(x=1.3, y=-1.3, z=1.0),
                                        center=dict(x=0, y=0, z=0)
                                    ),
                                    xaxis=dict(
                                        range=[-doubles_alley_width - 2, court_width + doubles_alley_width + 2]
                                    ),
                                    yaxis=dict(
                                        range=[-3, court_length + 3]
                                    ),
                                    aspectmode='manual',
                                    aspectratio=dict(x=1, y=2.5, z=1)
                                ),
                                height=900,
                                margin=dict(l=0, r=0, t=50, b=50),
                                hovermode='closest',
                                showlegend=True,
                                legend=dict(
                                    title="Trajectories",
                                    yanchor="top",
                                    y=1.0,
                                    xanchor="left",
                                    x=0.85,
                                    bgcolor="rgba(255, 255, 255, 0.9)"
                                )
                            )
                            
                        # Store in session state
                        st.session_state.fig_3d_cluster = fig_3d
                    
                    # Display the plot (always, since it now auto-generates)
                    render_interactive_chart(st.session_state.fig_3d_cluster)
                    st.success("✅ 3D visualization generated! Rotate and zoom to explore the spatiotemporal patterns.")            # ===========================
            # TAB 3: CLUSTER COMPARISON
            # ===========================
            with viz_tab3:
                st.markdown("#### Individual Cluster Analysis")
                st.markdown("View and compare individual clusters in detail.")
                
                # Check if trajectory data is available
                if 'trajectories' not in st.session_state or st.session_state.trajectories is None:
                    st.warning("⚠️ No trajectory data available. Please compute the distance matrix first in Step 3.")
                elif 'cluster_labels' not in st.session_state or st.session_state.cluster_labels is None:
                    st.warning("⚠️ No cluster assignments available. Please assign clusters using the slider above.")
                else:
                    # Get cluster data from session state
                    cluster_labels = st.session_state.cluster_labels
                    n_clusters = st.session_state.n_clusters
                    
                    # Get unique cluster IDs that actually exist in the data
                    unique_clusters = sorted(np.unique(cluster_labels))
                    
                    # Cluster selection
                    selected_clusters = st.multiselect(
                        "Select clusters to visualize",
                        options=unique_clusters,
                        default=[unique_clusters[0]] if len(unique_clusters) >= 1 else [],
                        format_func=lambda x: f"Cluster {x} ({(cluster_labels == x).sum()} trajectories)",
                        help="Select one or more clusters to visualize"
                    )
                    
                    if selected_clusters:
                        view_mode = st.radio(
                            "View mode",
                            options=["Overlay", "Side-by-side"],
                            horizontal=True,
                            help="Overlay: all clusters on one plot | Side-by-side: separate subplots"
                        )
                        
                        if st.button("📊 Visualize Selected Clusters", key="btn_cluster_compare"):
                            with st.spinner("Generating cluster comparison..."):
                                import plotly.express as px
                                
                                # Create trajectory dictionary mapping
                                trajectories_dict = {tid: traj for tid, traj in zip(st.session_state.trajectory_ids, st.session_state.trajectories)}
                                
                                # Create color mapping for actual clusters
                                cluster_to_idx = {cid: idx for idx, cid in enumerate(unique_clusters)}
                                colors = px.colors.qualitative.Plotly[:len(unique_clusters)]
                                
                                if view_mode == "Overlay":
                                    # Single plot with selected clusters - start with tennis court
                                    fig_compare = create_tennis_court()
                                    
                                    for cluster_id in selected_clusters:
                                        mask = cluster_labels == cluster_id
                                        cluster_trajectory_ids = np.array(st.session_state.trajectory_ids)[mask]
                                        color_idx = cluster_to_idx[cluster_id]
                                        
                                        for tid in cluster_trajectory_ids:
                                            traj_data = trajectories_dict[tid]
                                            
                                            # Add trajectory line and markers
                                            fig_compare.add_trace(go.Scatter(
                                                x=traj_data[:, 0],
                                                y=traj_data[:, 1],
                                                mode='lines+markers',
                                                name=f"T{tid} (C{cluster_id})",
                                                line=dict(color=colors[color_idx], width=2),
                                                marker=dict(
                                                    size=[4] * (len(traj_data) - 1) + [0],  # Hide last marker
                                                    color=colors[color_idx]
                                                ),
                                                legendgroup=f"cluster_{cluster_id}",
                                                hovertemplate=f'<b>Trajectory {tid}</b><br>Cluster: {cluster_id}<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>'
                                            ))
                                            
                                            # Add arrow at the end
                                            if len(traj_data) >= 2:
                                                dx = traj_data[-1, 0] - traj_data[-2, 0]
                                                dy = traj_data[-1, 1] - traj_data[-2, 1]
                                                angle = np.degrees(np.arctan2(dx, dy))
                                                
                                                fig_compare.add_trace(go.Scatter(
                                                    x=[traj_data[-1, 0]],
                                                    y=[traj_data[-1, 1]],
                                                    mode='markers',
                                                    marker=dict(
                                                        symbol='arrow',
                                                        color=colors[color_idx],
                                                        size=15,
                                                        angle=angle
                                                    ),
                                                    legendgroup=f"cluster_{cluster_id}",
                                                    showlegend=False,
                                                    hoverinfo='skip'
                                                ))
                                    
                                    fig_compare.update_layout(
                                        title=f"Cluster Comparison - Overlay View (Clusters: {selected_clusters})",
                                        height=900,
                                        hovermode='closest',
                                        showlegend=True,
                                        uirevision='constant'
                                    )
                                    
                                    render_interactive_chart(fig_compare)
                                    
                                else:  # Side-by-side
                                    # Create subplots with tennis courts
                                    n_selected = len(selected_clusters)
                                    cols = min(2, n_selected)
                                    rows = (n_selected + cols - 1) // cols
                                    
                                    fig_compare = make_subplots(
                                        rows=rows,
                                        cols=cols,
                                        subplot_titles=[f"Cluster {c} ({(cluster_labels == c).sum()} trajectories)" 
                                                       for c in selected_clusters],
                                        vertical_spacing=0.12,
                                        horizontal_spacing=0.1
                                    )
                                    
                                    # Tennis court dimensions
                                    court_width = 8.23
                                    court_length = 23.77
                                    doubles_width = 10.97
                                    doubles_alley_width = (doubles_width - court_width) / 2
                                    service_line_distance = 6.40
                                    net_position = court_length / 2
                                    service_line_bottom = net_position - service_line_distance
                                    service_line_top = net_position + service_line_distance
                                    center_x = court_width / 2
                                    
                                    for idx, cluster_id in enumerate(selected_clusters):
                                        row = idx // cols + 1
                                        col = idx % cols + 1
                                        color_idx = cluster_to_idx[cluster_id]
                                        
                                        # Add tennis court markings for this subplot
                                        # Outer boundary (doubles court)
                                        fig_compare.add_shape(
                                            type="rect", 
                                            x0=-doubles_alley_width, y0=0, 
                                            x1=court_width + doubles_alley_width, y1=court_length,
                                            line=dict(color="white", width=2),
                                            row=row, col=col
                                        )
                                        
                                        # Singles sidelines
                                        fig_compare.add_shape(
                                            type="line", x0=0, y0=0, x1=0, y1=court_length,
                                            line=dict(color="white", width=1.5),
                                            row=row, col=col
                                        )
                                        fig_compare.add_shape(
                                            type="line", x0=court_width, y0=0, x1=court_width, y1=court_length,
                                            line=dict(color="white", width=1.5),
                                            row=row, col=col
                                        )
                                        
                                        # Net
                                        fig_compare.add_shape(
                                            type="line", 
                                            x0=-doubles_alley_width, y0=net_position, 
                                            x1=court_width + doubles_alley_width, y1=net_position,
                                            line=dict(color="white", width=1.5),
                                            row=row, col=col
                                        )
                                        
                                        # Service lines
                                        fig_compare.add_shape(
                                            type="line", x0=0, y0=service_line_bottom, 
                                            x1=court_width, y1=service_line_bottom,
                                            line=dict(color="white", width=1.5),
                                            row=row, col=col
                                        )
                                        fig_compare.add_shape(
                                            type="line", x0=0, y0=service_line_top, 
                                            x1=court_width, y1=service_line_top,
                                            line=dict(color="white", width=1.5),
                                            row=row, col=col
                                        )
                                        
                                        # Center service line
                                        fig_compare.add_shape(
                                            type="line", x0=center_x, y0=service_line_bottom, 
                                            x1=center_x, y1=service_line_top,
                                            line=dict(color="white", width=1.5),
                                            row=row, col=col
                                        )
                                        
                                        # Add trajectories for this cluster
                                        mask = cluster_labels == cluster_id
                                        cluster_trajectory_ids = np.array(st.session_state.trajectory_ids)[mask]
                                        
                                        for tid in cluster_trajectory_ids:
                                            traj_data = trajectories_dict[tid]
                                            
                                            # Add trajectory line and markers
                                            fig_compare.add_trace(
                                                go.Scatter(
                                                    x=traj_data[:, 0],
                                                    y=traj_data[:, 1],
                                                    mode='lines+markers',
                                                    name=f"T{tid}",
                                                    line=dict(color=colors[color_idx], width=2),
                                                    marker=dict(
                                                        size=[4] * (len(traj_data) - 1) + [0],  # Hide last marker
                                                        color=colors[color_idx]
                                                    ),
                                                    legendgroup=f"traj_{tid}",
                                                    showlegend=False,
                                                    hovertemplate=f'<b>Trajectory {tid}</b><br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>'
                                                ),
                                                row=row,
                                                col=col
                                            )
                                            
                                            # Add arrow at the end
                                            if len(traj_data) >= 2:
                                                dx = traj_data[-1, 0] - traj_data[-2, 0]
                                                dy = traj_data[-1, 1] - traj_data[-2, 1]
                                                angle = np.degrees(np.arctan2(dx, dy))
                                                
                                                fig_compare.add_trace(
                                                    go.Scatter(
                                                        x=[traj_data[-1, 0]],
                                                        y=[traj_data[-1, 1]],
                                                        mode='markers',
                                                        marker=dict(
                                                            symbol='arrow',
                                                            color=colors[color_idx],
                                                            size=12,
                                                            angle=angle
                                                        ),
                                                        legendgroup=f"traj_{tid}",
                                                        showlegend=False,
                                                        hoverinfo='skip'
                                                    ),
                                                    row=row,
                                                    col=col
                                                )
                                        
                                        # Update axes for tennis court appearance
                                        x_margin = 2.0
                                        y_margin = 3.0
                                        
                                        fig_compare.update_xaxes(
                                            range=[-doubles_alley_width - x_margin, court_width + doubles_alley_width + x_margin],
                                            showgrid=False,
                                            zeroline=False,
                                            title_text="Court Width (m)",
                                            row=row, col=col
                                        )
                                        fig_compare.update_yaxes(
                                            range=[-y_margin, court_length + y_margin],
                                            showgrid=False,
                                            zeroline=False,
                                            title_text="Court Length (m)",
                                            scaleanchor=f"x{col if row == 1 else (row-1)*cols + col}",
                                            scaleratio=1,
                                            row=row, col=col
                                        )
                                    
                                    fig_compare.update_layout(
                                        title_text="Cluster Comparison - Side-by-side View",
                                        height=900 * rows,
                                        hovermode='closest',
                                        plot_bgcolor='#25D366',  # Tennis court green
                                        uirevision='constant'
                                    )
                                    
                                    render_interactive_chart(fig_compare)
                                
                                # Cluster statistics
                                st.markdown("---")
                                st.markdown("**Selected Cluster Statistics:**")
                                
                                stats_data = []
                                for cluster_id in selected_clusters:
                                    mask = cluster_labels == cluster_id
                                    cluster_tids = np.array(st.session_state.trajectory_ids)[mask]
                                    
                                    # Calculate average trajectory length
                                    avg_length = np.mean([len(trajectories_dict[tid]) for tid in cluster_tids])
                                    
                                    # Calculate spatial extent (bounding box)
                                    all_points = np.vstack([trajectories_dict[tid] for tid in cluster_tids])
                                    x_range = all_points[:, 0].max() - all_points[:, 0].min()
                                    y_range = all_points[:, 1].max() - all_points[:, 1].min()
                                    
                                    stats_data.append({
                                        'Cluster': cluster_id,
                                        'Trajectories': len(cluster_tids),
                                        'Avg Length': f"{avg_length:.1f}",
                                        'X Range': f"{x_range:.2f}",
                                        'Y Range': f"{y_range:.2f}",
                                        'Spatial Area': f"{x_range * y_range:.2f}"
                                    })
                                
                                stats_df = pd.DataFrame(stats_data)
                                st.dataframe(stats_df, use_container_width=True)
                    
                    else:
                        st.info("👆 Select one or more clusters above to visualize and compare them.")
            
            st.markdown('---')
            st.success("✅ Step 6 cluster visualizations complete! Explore your clustered trajectories above.")
            
            # ===========================
            # EXPORT & SUMMARY
            # ===========================
            
            st.markdown('---')
            st.markdown("### 📋 Export & Summary")
            
            st.info("""
            **Final Step**: Export your results and view comprehensive analysis summary
            - **Export Cluster Assignments**: Download cluster labels as CSV
            - **Export Distance Matrix**: Download pairwise distances
            - **Analysis Summary**: View complete statistics and methodology
            """)
            
            # Create tabs for export and summary
            export_tab1, export_tab2, export_tab3 = st.tabs([
                "💾 Export Data", 
                "📊 Analysis Summary", 
                "📖 Documentation"
            ])
            
            # ===========================
            # TAB 1: EXPORT DATA
            # ===========================
            with export_tab1:
                st.markdown("#### Export Analysis Results")
                st.markdown("Download your clustering results and data for further analysis.")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Cluster Assignments**")
                    st.markdown("Export trajectory-to-cluster mappings")
                    
                    # Create cluster assignments dataframe
                    cluster_export_df = pd.DataFrame({
                        'Trajectory_ID': st.session_state.trajectory_ids,
                        'Cluster': cluster_labels,
                        'Cluster_Size': [sum(cluster_labels == c) for c in cluster_labels]
                    })
                    
                    # Add trajectory length if available
                    if 'trajectories' in st.session_state and st.session_state.trajectories is not None:
                        cluster_export_df['Trajectory_Length'] = [len(traj) for traj in st.session_state.trajectories]
                    
                    # Preview
                    st.dataframe(cluster_export_df.head(10), use_container_width=True)
                    
                    # Download button
                    csv_clusters = cluster_export_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Cluster Assignments (CSV)",
                        data=csv_clusters,
                        file_name=f"cluster_assignments_{n_clusters}clusters.csv",
                        mime="text/csv",
                        help="Download complete cluster assignments"
                    )
                
                with col2:
                    st.markdown("**Distance Matrix**")
                    st.markdown("Export pairwise trajectory distances")
                    
                    # Create distance matrix dataframe
                    distance_df = pd.DataFrame(
                        st.session_state.distance_matrix,
                        index=st.session_state.trajectory_ids,
                        columns=st.session_state.trajectory_ids
                    )
                    
                    # Preview
                    st.dataframe(distance_df.iloc[:5, :5], use_container_width=True)
                    
                    # Download button
                    csv_distances = distance_df.to_csv()
                    st.download_button(
                        label="� Download Distance Matrix (CSV)",
                        data=csv_distances,
                        file_name="distance_matrix.csv",
                        mime="text/csv",
                        help="Download full pairwise distance matrix"
                    )
                
                st.markdown("---")
                st.markdown("**📈 Additional Exports**")
                
                col3, col4 = st.columns(2)
                
                with col3:
                    # Export cluster statistics
                    stats_data = []
                    for cluster_id in range(1, n_clusters + 1):
                        mask = cluster_labels == cluster_id
                        cluster_tids = np.array(st.session_state.trajectory_ids)[mask]
                        
                        stats_entry = {
                            'Cluster': cluster_id,
                            'Size': len(cluster_tids),
                            'Percentage': f"{len(cluster_tids)/len(cluster_labels)*100:.1f}%",
                            'Trajectory_IDs': ','.join(map(str, cluster_tids))
                        }
                        
                        # Add spatial stats if trajectories available
                        if 'trajectories' in st.session_state and st.session_state.trajectories is not None:
                            trajectories_dict = {tid: traj for tid, traj in zip(st.session_state.trajectory_ids, st.session_state.trajectories)}
                            avg_length = np.mean([len(trajectories_dict[tid]) for tid in cluster_tids])
                            stats_entry['Avg_Trajectory_Length'] = f"{avg_length:.1f}"
                        
                        stats_data.append(stats_entry)
                    
                    stats_export_df = pd.DataFrame(stats_data)
                    
                    st.markdown("**Cluster Statistics**")
                    st.dataframe(stats_export_df, use_container_width=True)
                    
                    csv_stats = stats_export_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Cluster Statistics (CSV)",
                        data=csv_stats,
                        file_name=f"cluster_statistics_{n_clusters}clusters.csv",
                        mime="text/csv"
                    )
                
                with col4:
                    # Export configuration/methodology
                    config_data = {
                        'Analysis_Date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'Clustering_Method': clustering_method,
                        'Number_of_Clusters': n_clusters,
                        'Number_of_Trajectories': len(st.session_state.trajectory_ids),
                        'Linkage_Method': 'Ward',
                        'Distance_Metric': 'Euclidean' if clustering_method == 'Features' else 'Chamfer' if clustering_method == 'Spatial (Chamfer)' else 'DTW'
                    }
                    
                    config_df = pd.DataFrame([config_data]).T
                    config_df.columns = ['Value']
                    
                    st.markdown("**Analysis Configuration**")
                    st.dataframe(config_df, use_container_width=True)
                    
                    csv_config = config_df.to_csv()
                    st.download_button(
                        label="📥 Download Configuration (CSV)",
                        data=csv_config,
                        file_name="analysis_configuration.csv",
                        mime="text/csv"
                    )
            
            # ===========================
            # TAB 2: ANALYSIS SUMMARY
            # ===========================
            with export_tab2:
                st.markdown("#### Comprehensive Analysis Summary")
                
                # Overall metrics
                st.markdown("### 📊 Overall Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Trajectories", len(st.session_state.trajectory_ids))
                
                with col2:
                    st.metric("Number of Clusters", n_clusters)
                
                with col3:
                    avg_cluster_size = len(cluster_labels) / n_clusters
                    st.metric("Avg Cluster Size", f"{avg_cluster_size:.1f}")
                
                with col4:
                    if 'trajectories' in st.session_state and st.session_state.trajectories is not None:
                        avg_traj_length = np.mean([len(traj) for traj in st.session_state.trajectories])
                        st.metric("Avg Trajectory Length", f"{avg_traj_length:.1f}")
                    else:
                        st.metric("Avg Trajectory Length", "N/A")
                
                st.markdown("---")
                
                # Cluster distribution
                st.markdown("### 🎯 Cluster Distribution")
                
                import plotly.express as px
                
                cluster_counts = pd.DataFrame({
                    'Cluster': [f"Cluster {i}" for i in range(1, n_clusters + 1)],
                    'Count': [(cluster_labels == i).sum() for i in range(1, n_clusters + 1)]
                })
                
                fig_dist = px.bar(
                    cluster_counts,
                    x='Cluster',
                    y='Count',
                    title="Trajectories per Cluster",
                    color='Count',
                    color_continuous_scale='Viridis'
                )
                
                fig_dist.update_layout(
                    xaxis_title="Cluster",
                    yaxis_title="Number of Trajectories",
                    height=400
                )
                
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # Pie chart
                fig_pie = px.pie(
                    cluster_counts,
                    values='Count',
                    names='Cluster',
                    title="Cluster Distribution (%)"
                )
                
                st.plotly_chart(fig_pie, use_container_width=True)
                
                st.markdown("---")
                
                # Distance matrix statistics
                st.markdown("### 📏 Distance Matrix Statistics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Get upper triangle (excluding diagonal)
                    triu_indices = np.triu_indices_from(st.session_state.distance_matrix, k=1)
                    distances = st.session_state.distance_matrix[triu_indices]
                    
                    st.markdown("**Overall Distance Statistics:**")
                    st.write(f"- **Mean Distance**: {distances.mean():.4f}")
                    st.write(f"- **Median Distance**: {np.median(distances):.4f}")
                    st.write(f"- **Std Deviation**: {distances.std():.4f}")
                    st.write(f"- **Min Distance**: {distances.min():.4f}")
                    st.write(f"- **Max Distance**: {distances.max():.4f}")
                
                with col2:
                    # Distance histogram
                    fig_hist = px.histogram(
                        x=distances,
                        nbins=50,
                        title="Distance Distribution",
                        labels={'x': 'Distance', 'y': 'Frequency'}
                    )
                    
                    fig_hist.update_layout(height=300)
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                st.markdown("---")
                
                # Methodology summary
                st.markdown("### 🔬 Methodology")
                
                st.markdown(f"""
                **Clustering Approach:**
                - **Method**: {clustering_method}
                - **Distance Metric**: {'Euclidean (feature-based)' if clustering_method == 'Features' else 'Chamfer (spatial)' if clustering_method == 'Spatial (Chamfer)' else 'Dynamic Time Warping'}
                - **Linkage**: Ward (minimizes within-cluster variance)
                - **Algorithm**: Hierarchical Agglomerative Clustering
                
                **Process:**
                1. Computed pairwise distances between all trajectories
                2. Built hierarchical clustering dendrogram using Ward linkage
                3. Cut dendrogram at {n_clusters} clusters
                4. Assigned each trajectory to its cluster
                5. Validated with silhouette analysis and MDS visualization
                """)
            
            # ===========================
            # TAB 3: DOCUMENTATION
            # ===========================
            with export_tab3:
                st.markdown("#### 📖 User Guide & Documentation")
                
                st.markdown("""
                ## Trajectory Clustering Analysis Tool
                
                ### Overview
                This tool provides comprehensive trajectory clustering analysis using hierarchical methods.
                It supports multiple distance metrics and visualization techniques to explore spatiotemporal patterns.
                
                ---
                
                ### Workflow Steps
                
                #### **Data Loading & Infrastructure**
                - Load trajectory data from CSV/Excel files
                - Initialize clustering algorithms and distance computation functions
                
                #### **Distance Computation**
                Choose from three distance metrics:
                - **Features**: Extract trajectory features (length, speed, angles) and compute Euclidean distance
                - **Spatial (Chamfer)**: Measure spatial shape similarity
                - **DTW**: Dynamic Time Warping for temporal alignment
                
                #### **Dendrogram & Clustering**
                - View hierarchical clustering structure
                - Select optimal number of clusters (manual or auto-detect)
                - Assign trajectories to clusters
                
                #### **Analysis Tools**
                - **MDS Visualization**: Project high-dimensional distances to 2D/3D
                - **Similarity Search**: Find most similar trajectories
                - **Silhouette Analysis**: Validate cluster quality
                
                #### **Cluster Visualizations**
                - 2D spatial plots of clustered trajectories
                - 3D spatiotemporal views
                - Side-by-side cluster comparison
                
                #### **Export & Summary**
                - Download cluster assignments, distance matrices, statistics
                - View comprehensive analysis summary
                - Access this documentation
                
                ---
                
                ### Distance Metrics Explained
                
                **1. Feature-Based (Euclidean)**
                - Extracts: length, avg speed, direction, spatial extent
                - Best for: Comparing overall trajectory characteristics
                - Fast computation
                
                **2. Spatial (Chamfer)**
                - Measures minimum point-to-point distances
                - Best for: Shape similarity regardless of timing
                - Symmetric distance metric
                
                **3. Dynamic Time Warping (DTW)**
                - Aligns trajectories temporally before measuring distance
                - Best for: Trajectories with similar patterns at different speeds
                - Handles temporal shifts
                
                ---
                
                ### Interpreting Results
                
                **Dendrogram:**
                - Height indicates dissimilarity when clusters merge
                - Longer vertical lines = more distinct clusters
                - Cut at desired height to get cluster count
                
                **Silhouette Score:**
                - Range: -1 to 1
                - > 0.7: Excellent clustering
                - > 0.5: Good clustering
                - > 0.3: Moderate clustering
                - < 0.3: Poor clustering
                
                **MDS Plot:**
                - Shows relative positions based on distances
                - Closer points = more similar trajectories
                - Color indicates cluster membership
                
                ---
                
                ### Tips & Best Practices
                
                1. **Choosing Distance Metric:**
                   - Start with Features for quick exploration
                   - Use Spatial if shape matters more than timing
                   - Use DTW for speed-invariant comparison
                
                2. **Selecting Number of Clusters:**
                   - Use auto-detect as starting point
                   - Check silhouette scores for validation
                   - Consider domain knowledge
                
                3. **Interpreting Clusters:**
                   - Examine cluster visualizations in Step 6
                   - Use similarity search to find representatives
                   - Check cluster statistics for size balance
                
                4. **Performance:**
                   - Large datasets (>1000 trajectories) may be slow
                   - DTW is most computationally expensive
                   - Features method is fastest
                
                ---
                
                ### Troubleshooting
                
                **Q: Distance matrix computation is slow**
                - Try Features method first (fastest)
                - Reduce number of trajectories with filters
                - Be patient with DTW on large datasets
                
                **Q: Clusters seem arbitrary**
                - Check silhouette scores (should be > 0.3)
                - Try different number of clusters
                - Consider different distance metric
                
                **Q: Visualizations not showing**
                - Ensure you completed distance computation in Step 3
                - Check that cluster assignment was performed in Step 4
                
                **Q: Export buttons not working**
                - Complete all previous steps first
                - Ensure clustering is performed
                
                ---
                
                ### Citation
                
                If you use this tool in your research, please cite:
                
                ```
                Trajectory Clustering Analysis Tool
                Hierarchical Clustering with Multiple Distance Metrics
                [Your Institution/Project Name]
                2025
                ```
                """)
                
                st.markdown("---")
                st.success("📚 Documentation complete! Use the tabs above to export data and view analysis summary.")
            
            st.markdown('---')
            st.success("🎉 **Analysis Complete!** All steps finished. Use the tabs above to export results and view summary.")
    
    elif analysis_method == "Extra":
        st.header("🎯 Extra Analysis Methods")

        extra_methods = [
            "Heat map animations",
            "PDP",
            "QTC"
        ]

        selected_extra_method = st.selectbox(
            "Select an extra method:",
            extra_methods
        )

        st.markdown('---')

        # Use selections from sidebar
        selected_configs = st.session_state.shared_selected_configs
        selected_objects = st.session_state.shared_selected_objects

        # Time range
        min_time = float(df['tst'].min())
        max_time = float(df['tst'].max())

        col1, col2 = st.columns(2)
        with col1:
            start_time = st.number_input(
                "Start time",
                min_value=min_time,
                max_value=max_time,
                value=min_time,
                step=0.01,
                format="%.2f",
                key="extra_start"
            )
        with col2:
            end_time = st.number_input(
                "End time",
                min_value=start_time,
                max_value=max_time,
                value=max_time,
                step=0.01,
                format="%.2f",
                key="extra_end"
            )

        st.markdown('---')

        if selected_extra_method == "Heat map animations":
            st.subheader("🔥 Heat Map Animations")
            st.info("This method creates animated heat maps showing density patterns over time.")
            
            col1, col2 = st.columns(2)
            with col1:
                grid_resolution = st.slider("Grid resolution", 10, 100, 30, key="heatmap_resolution")
            with col2:
                time_window = st.slider("Time window size", 5, 50, 10, key="heatmap_window")
            
            if st.button("Generate Heat Map Animation", key="run_heatmap_animation"):
                with st.spinner("Generating heat map animation..."):
                    court_dims = get_court_dimensions(court_type)
                    
                    # Create grid
                    x_edges = np.linspace(0, court_dims['width'], grid_resolution)
                    y_edges = np.linspace(0, court_dims['height'], grid_resolution)
                    
                    # Get time steps
                    filtered_df = df[
                        (df['config_source'].isin(selected_configs)) &
                        (df['obj'].isin(selected_objects)) &
                        (df['tst'] >= start_time) &
                        (df['tst'] <= end_time)
                    ]
                    
                    time_steps = sorted(filtered_df['tst'].unique())
                    
                    if len(time_steps) == 0:
                        st.error("No data in selected time range.")
                    else:
                        # Create frames for animation
                        frames = []
                        
                        for i in range(0, len(time_steps), time_window):
                            window_times = time_steps[i:i+time_window]
                            window_data = filtered_df[filtered_df['tst'].isin(window_times)]
                            
                            # Create 2D histogram
                            heatmap, _, _ = np.histogram2d(
                                window_data['x'],
                                window_data['y'],
                                bins=[x_edges, y_edges]
                            )
                            
                            frames.append(heatmap.T)
                        
                        if len(frames) == 0:
                            st.error("Not enough data to create animation.")
                        else:
                            # Create initial figure
                            fig = create_pitch_figure(court_type)
                            
                            # Add heatmap
                            fig.add_trace(go.Heatmap(
                                z=frames[0],
                                x=x_edges,
                                y=y_edges,
                                colorscale='Hot',
                                opacity=0.6,
                                showscale=True,
                                hovertemplate='x: %{x:.1f}<br>y: %{y:.1f}<br>density: %{z}<extra></extra>'
                            ))
                            
                            # Create animation frames
                            plot_frames = [
                                go.Frame(
                                    data=[go.Heatmap(
                                        z=frame,
                                        x=x_edges,
                                        y=y_edges,
                                        colorscale='Hot',
                                        opacity=0.6,
                                        showscale=True
                                    )],
                                    name=str(idx)
                                )
                                for idx, frame in enumerate(frames)
                            ]
                            
                            fig.frames = plot_frames
                            
                            # Add play button
                            fig.update_layout(
                                updatemenus=[{
                                    'type': 'buttons',
                                    'showactive': False,
                                    'buttons': [
                                        {
                                            'label': '▶ Play',
                                            'method': 'animate',
                                            'args': [None, {
                                                'frame': {'duration': 200, 'redraw': True},
                                                'fromcurrent': True,
                                                'mode': 'immediate'
                                            }]
                                        },
                                        {
                                            'label': '⏸ Pause',
                                            'method': 'animate',
                                            'args': [[None], {
                                                'frame': {'duration': 0, 'redraw': False},
                                                'mode': 'immediate'
                                            }]
                                        }
                                    ],
                                    'x': 0.1,
                                    'y': 1.15
                                }],
                                sliders=[{
                                    'steps': [
                                        {
                                            'args': [[f.name], {
                                                'frame': {'duration': 0, 'redraw': True},
                                                'mode': 'immediate'
                                            }],
                                            'label': f'Frame {i}',
                                            'method': 'animate'
                                        }
                                        for i, f in enumerate(plot_frames)
                                    ],
                                    'x': 0.1,
                                    'len': 0.85,
                                    'y': 0
                                }]
                            )
                            
                            render_interactive_chart(fig, "Animated density heat map")
                            
                            # Static aggregate heatmap
                            st.subheader("Aggregate Heat Map")
                            aggregate_heatmap, _, _ = np.histogram2d(
                                filtered_df['x'],
                                filtered_df['y'],
                                bins=[x_edges, y_edges]
                            )
                            
                            fig_static = create_pitch_figure(court_type)
                            fig_static.add_trace(go.Heatmap(
                                z=aggregate_heatmap.T,
                                x=x_edges,
                                y=y_edges,
                                colorscale='Hot',
                                opacity=0.6,
                                showscale=True,
                                hovertemplate='x: %{x:.1f}<br>y: %{y:.1f}<br>density: %{z}<extra></extra>'
                            ))
                            
                            render_interactive_chart(fig_static, "Overall density across entire time period")
            
        elif selected_extra_method == "PDP":
            st.subheader("📐 PDP (Pairwise Distance Profile)")
            st.info("This method analyzes pairwise distances between trajectories over time.")
            
            if st.button("Calculate PDP", key="run_pdp"):
                with st.spinner("Calculating pairwise distance profiles..."):
                    # Get trajectories with time information
                    trajectories = []
                    traj_ids = []
                    
                    for config in selected_configs:
                        for obj_id in selected_objects:
                            obj_data = df[(df['obj'] == obj_id) & 
                                        (df['config_source'] == config) &
                                        (df['tst'] >= start_time) & 
                                        (df['tst'] <= end_time)].sort_values('tst')
                            
                            if len(obj_data) >= 2:
                                trajectories.append(obj_data[['tst', 'x', 'y']].values)
                                traj_ids.append(f"{config}-Obj{obj_id}")
                    
                    if len(trajectories) < 2:
                        st.error("Need at least 2 trajectories for PDP analysis.")
                    else:
                        # Find common time steps
                        all_times = sorted(set().union(*[set(traj[:, 0]) for traj in trajectories]))
                        
                        # Calculate pairwise distances at each time step
                        n_trajs = len(trajectories)
                        pdp_data = {f"{traj_ids[i]} - {traj_ids[j]}": [] 
                                   for i in range(n_trajs) for j in range(i+1, n_trajs)}
                        
                        for t in all_times:
                            positions = {}
                            for idx, traj in enumerate(trajectories):
                                # Find position at time t (or closest)
                                time_diffs = np.abs(traj[:, 0] - t)
                                closest_idx = np.argmin(time_diffs)
                                if time_diffs[closest_idx] < 5:  # Within 5 time units
                                    positions[idx] = traj[closest_idx, 1:3]
                            
                            # Calculate pairwise distances
                            for i in range(n_trajs):
                                for j in range(i+1, n_trajs):
                                    if i in positions and j in positions:
                                        dist = euclidean(positions[i], positions[j])
                                        pdp_data[f"{traj_ids[i]} - {traj_ids[j]}"].append((t, dist))
                        
                        # Plot PDP
                        st.subheader("Pairwise Distance Profile")
                        fig = go.Figure()
                        
                        for pair_name, distances in pdp_data.items():
                            if distances:
                                times, dists = zip(*distances)
                                fig.add_trace(go.Scatter(
                                    x=times,
                                    y=dists,
                                    mode='lines',
                                    name=pair_name
                                ))
                        
                        fig.update_layout(
                            title="Distance Between Trajectory Pairs Over Time",
                            xaxis_title="Time",
                            yaxis_title="Distance (meters)",
                            height=600
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Statistics
                        st.subheader("Distance Statistics")
                        stats_data = []
                        for pair_name, distances in pdp_data.items():
                            if distances:
                                dists = [d[1] for d in distances]
                                stats_data.append({
                                    'Pair': pair_name,
                                    'Mean Distance': np.mean(dists),
                                    'Min Distance': np.min(dists),
                                    'Max Distance': np.max(dists),
                                    'Std Dev': np.std(dists)
                                })
                        
                        if stats_data:
                            st.dataframe(pd.DataFrame(stats_data).round(2))
            
        elif selected_extra_method == "QTC":
            st.subheader("� QTC (Qualitative Trajectory Calculus)")
            st.info("This method uses qualitative representations to characterize trajectory relationships.")
            
            if len(selected_objects) < 2:
                st.warning("QTC requires at least 2 objects. Please select more objects.")
            else:
                if st.button("Calculate QTC", key="run_qtc"):
                    with st.spinner("Calculating qualitative trajectory calculus..."):
                        # Get first two trajectories for demonstration
                        traj_data = []
                        traj_ids = []
                        
                        for config in selected_configs[:1]:  # Use first config
                            for obj_id in selected_objects[:2]:  # Use first two objects
                                obj_data = df[(df['obj'] == obj_id) & 
                                            (df['config_source'] == config) &
                                            (df['tst'] >= start_time) & 
                                            (df['tst'] <= end_time)].sort_values('tst')
                                
                                if len(obj_data) >= 2:
                                    traj_data.append(obj_data[['tst', 'x', 'y']].values)
                                    traj_ids.append(f"{config}-Obj{obj_id}")
                        
                        if len(traj_data) < 2:
                            st.error("Need at least 2 valid trajectories.")
                        else:
                            st.info(f"Analyzing QTC between {traj_ids[0]} and {traj_ids[1]}")
                            
                            # Find common time steps
                            times1 = set(traj_data[0][:, 0])
                            times2 = set(traj_data[1][:, 0])
                            common_times = sorted(times1 & times2)
                            
                            if len(common_times) < 2:
                                st.error("Trajectories don't overlap in time.")
                            else:
                                # Calculate QTC values
                                qtc_values = []
                                
                                for i in range(len(common_times) - 1):
                                    t1 = common_times[i]
                                    t2 = common_times[i + 1]
                                    
                                    # Get positions at both times
                                    idx1_t1 = np.where(traj_data[0][:, 0] == t1)[0][0]
                                    idx1_t2 = np.where(traj_data[0][:, 0] == t2)[0][0]
                                    idx2_t1 = np.where(traj_data[1][:, 0] == t1)[0][0]
                                    idx2_t2 = np.where(traj_data[1][:, 0] == t2)[0][0]
                                    
                                    pos1_t1 = traj_data[0][idx1_t1, 1:3]
                                    pos1_t2 = traj_data[0][idx1_t2, 1:3]
                                    pos2_t1 = traj_data[1][idx2_t1, 1:3]
                                    pos2_t2 = traj_data[1][idx2_t2, 1:3]
                                    
                                    # Calculate distance at both times
                                    dist_t1 = euclidean(pos1_t1, pos2_t1)
                                    dist_t2 = euclidean(pos1_t2, pos2_t2)
                                    
                                    # QTC Basic: are objects getting closer (-), staying same (0), or moving apart (+)
                                    threshold = 0.5  # meters
                                    if dist_t2 < dist_t1 - threshold:
                                        qtc = "-"  # Getting closer
                                    elif dist_t2 > dist_t1 + threshold:
                                        qtc = "+"  # Moving apart
                                    else:
                                        qtc = "0"  # Stable distance
                                    
                                    qtc_values.append({
                                        'time': t1,
                                        'distance_t1': dist_t1,
                                        'distance_t2': dist_t2,
                                        'qtc': qtc
                                    })
                                
                                # Display QTC sequence
                                st.subheader("QTC Sequence")
                                qtc_sequence = ''.join([v['qtc'] for v in qtc_values])
                                st.code(qtc_sequence)
                                
                                st.write("**Legend:**")
                                st.write("- `-`: Objects getting closer")
                                st.write("- `0`: Distance stable")
                                st.write("- `+`: Objects moving apart")
                                
                                # Plot distance over time
                                st.subheader("Distance Over Time")
                                fig = go.Figure()
                                
                                times = [v['time'] for v in qtc_values]
                                dists = [v['distance_t1'] for v in qtc_values]
                                qtcs = [v['qtc'] for v in qtc_values]
                                
                                # Color points by QTC value
                                colors = ['red' if q == '-' else 'gray' if q == '0' else 'blue' 
                                         for q in qtcs]
                                
                                fig.add_trace(go.Scatter(
                                    x=times,
                                    y=dists,
                                    mode='lines+markers',
                                    marker=dict(color=colors, size=8),
                                    line=dict(color='lightgray'),
                                    name='Distance',
                                    hovertemplate='Time: %{x}<br>Distance: %{y:.2f}m<extra></extra>'
                                ))
                                
                                fig.update_layout(
                                    title=f"Distance between {traj_ids[0]} and {traj_ids[1]}",
                                    xaxis_title="Time",
                                    yaxis_title="Distance (meters)",
                                    height=500
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # QTC statistics
                                st.subheader("QTC Statistics")
                                qtc_counts = pd.Series(qtcs).value_counts()
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Getting Closer", qtc_counts.get('-', 0))
                                with col2:
                                    st.metric("Stable", qtc_counts.get('0', 0))
                                with col3:
                                    st.metric("Moving Apart", qtc_counts.get('+', 0))
                                
                                # Visualize trajectories
                                st.subheader("Trajectory Visualization")
                                fig_traj = create_pitch_figure(court_type)
                                
                                for idx, traj_id in enumerate(traj_ids[:2]):
                                    config, obj_part = traj_id.split('-Obj')
                                    obj_id = int(float(obj_part))
                                    
                                    obj_data = df[(df['obj'] == obj_id) & 
                                                (df['config_source'] == config) &
                                                (df['tst'] >= start_time) & 
                                                (df['tst'] <= end_time)].sort_values('tst')
                                    
                                    color = ['blue', 'red'][idx]
                                    fig_traj.add_trace(go.Scatter(
                                        x=obj_data['x'],
                                        y=obj_data['y'],
                                        mode='lines+markers',
                                        name=traj_id,
                                        line=dict(color=color, width=2),
                                        marker=dict(size=4, color=color)
                                    ))
                                
                                render_interactive_chart(fig_traj, "Trajectories analyzed with QTC")

# Run the app
if __name__ == "__main__":
    main()
