import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

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


def render_interactive_chart(fig, caption="Use the toolbar to zoom, pan, or reset (double-click)."):
    """Render a Plotly figure with consistent interactive controls."""
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
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
        st.title("📊 Spatiotemporal Analysis and Modeling")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.info("🔒 Please enter the password provided by your instructor.")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.title("📊 Spatiotemporal Analysis and Modeling")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Incorrect password. Please try again.")
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

# Color mapping function
def get_color(obj_id):
    """Get color based on object ID"""
    try:
        obj_id = int(obj_id)
        if obj_id == 1:
            return 'blue'  # Object 1
        elif obj_id == 2:
            return 'red'  # Object 2
        elif 3 <= obj_id <= 11:
            return 'green'  # Other objects
        elif 12 <= obj_id <= 22:
            return 'black'  # Team 2
        elif obj_id == 0:
            return 'yellow'  # Ball
        else:
            return 'gray'
    except:
        return 'gray'

# Douglas-Peucker algorithm for line simplification
def perpendicular_distance(point, start, end):
    """Calculate perpendicular distance from point to line"""
    if start['x'] == end['x'] and start['y'] == end['y']:
        return np.sqrt((point['x'] - start['x'])**2 + (point['y'] - start['y'])**2)
    
    num = abs((end['y'] - start['y']) * point['x'] - 
              (end['x'] - start['x']) * point['y'] + 
              end['x'] * start['y'] - end['y'] * start['x'])
    den = np.sqrt((end['y'] - start['y'])**2 + (end['x'] - start['x'])**2)
    return num / den if den != 0 else 0

def douglas_peucker(points, tolerance):
    """Douglas-Peucker line simplification algorithm"""
    if len(points) < 3:
        return points
    
    max_distance = 0
    index = 0
    start = points[0]
    end = points[-1]
    
    for i in range(1, len(points) - 1):
        distance = perpendicular_distance(points[i], start, end)
        if distance > max_distance:
            index = i
            max_distance = distance
    
    if max_distance > tolerance:
        left = douglas_peucker(points[:index + 1], tolerance)
        right = douglas_peucker(points[index:], tolerance)
        return left[:-1] + right
    else:
        return [start, end]

def spatiotemporal_distance(point, start, end):
    """Calculate spatiotemporal distance"""
    spatial_dist = perpendicular_distance(point, start, end)
    time_diff = abs(point['timestamp'] - start['timestamp']) / \
                max(abs(end['timestamp'] - start['timestamp']), 1)
    return spatial_dist + time_diff

def douglas_peucker_spatiotemporal(points, tolerance):
    """Douglas-Peucker with spatiotemporal distance"""
    if len(points) < 3:
        return points
    
    max_distance = 0
    index = 0
    start = points[0]
    end = points[-1]
    
    for i in range(1, len(points) - 1):
        distance = spatiotemporal_distance(points[i], start, end)
        if distance > max_distance:
            index = i
            max_distance = distance
    
    if max_distance > tolerance:
        left = douglas_peucker_spatiotemporal(points[:index + 1], tolerance)
        right = douglas_peucker_spatiotemporal(points[index:], tolerance)
        return left[:-1] + right
    else:
        return [start, end]

# Load and parse CSV data
def load_data(uploaded_file, update_state=True, show_success=True):
    """Load and parse CSV file"""
    try:
        # Try to read with header first
        df_test = pd.read_csv(uploaded_file, nrows=5)
        uploaded_file.seek(0)  # Reset file pointer
        
        # Check if first row looks like a header
        has_header = False
        if len(df_test.columns) >= 5:
            first_col = str(df_test.columns[0]).lower()
            if 'con' in first_col or 'constant' in first_col or 'timestamp' in str(df_test.columns[1]).lower():
                has_header = True
        
        # Read the CSV
        if has_header:
            df = pd.read_csv(uploaded_file)
            # Map common column names to our standard names
            column_mapping = {
                'constant': 'con',
                'timestamp': 'tst',
                'ID': 'obj',
                'id': 'obj',
                'con': 'con',
                'tst': 'tst',
                'obj': 'obj',
                'x': 'x',
                'y': 'y'
            }
            # Rename columns if they match known names
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns:
                    df = df.rename(columns={old_name: new_name})
        else:
            # No header, read as before
            df = pd.read_csv(uploaded_file, header=None, names=['con', 'tst', 'obj', 'x', 'y'])
        
        # Keep only the columns we need
        required_cols = ['con', 'tst', 'obj', 'x', 'y']
        
        # Check if all required columns exist
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            st.error(f"Found columns: {', '.join(df.columns.tolist())}")
            return None
        
        df = df[required_cols]
        file_name = getattr(uploaded_file, 'name', 'uploaded data')
        df['config_source'] = file_name
        
        # Convert columns to numeric types
        df['con'] = pd.to_numeric(df['con'], errors='coerce')
        df['tst'] = pd.to_numeric(df['tst'], errors='coerce')
        df['obj'] = pd.to_numeric(df['obj'], errors='coerce')
        df['x'] = pd.to_numeric(df['x'], errors='coerce')
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        
        # Remove any rows with NaN values
        initial_rows = len(df)
        df = df.dropna()
        
        if len(df) == 0:
            st.error(f"No valid data rows found. All {initial_rows} rows had invalid or missing values.")
            return None
        
        if len(df) < initial_rows:
            st.warning(f"Removed {initial_rows - len(df)} rows with invalid data. {len(df)} rows remaining.")
        
        # Store data without constraining coordinates - let court type determine limits
        if update_state:
            st.session_state.data = df
            st.session_state.max_time = df['tst'].max()
            st.session_state.filename = file_name
            st.session_state.uploaded_filenames = [file_name]
            st.session_state.config_sources = [file_name]
        
        if show_success:
            st.success(f"✅ Loaded {len(df)} data points from {file_name} successfully!")
        
        return df
        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

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
        return douglas_peucker(points, temporal_resolution)
    
    elif aggregation_type == 'Spatiotemporal generalise':
        return douglas_peucker_spatiotemporal(points, temporal_resolution)
    
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

# Visualize static trajectories
def visualize_static(df, selected_configs, selected_objects, start_time, end_time, 
                     aggregation_type, temporal_resolution, translate_to_center=False, court_type='Football'):
    """Create static trajectory visualization"""
    fig = create_pitch_figure(court_type)
    court_dims = get_court_dimensions(court_type)
    
    center_x = court_dims['width'] / 2
    center_y = court_dims['height'] / 2
    
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
            
            color = get_color(obj_id)
            
            # Create legend group name
            legend_group = f'{config} | Obj {obj_id}'
            
            # Draw trajectory with arrow at the end
            fig.add_trace(go.Scatter(
                x=x_coords, y=y_coords,
                mode='lines+markers',
                name=f'{config} - Obj {obj_id}',
                legendgroup=legend_group,
                line=dict(color=color, width=2),
                marker=dict(size=4, color=color),
                hovertemplate=f'Object {obj_id}<br>Config {config}<br>x: %{{x:.2f}}m<br>y: %{{y:.2f}}m<extra></extra>'
            ))
            
            # Add arrow at the end as a separate trace with the same legend group
            if len(x_coords) >= 2:
                # Calculate arrow direction
                dx = x_coords[-1] - x_coords[-2]
                dy = y_coords[-1] - y_coords[-2]
                
                fig.add_trace(go.Scatter(
                    x=[x_coords[-1]], 
                    y=[y_coords[-1]],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=color,
                        symbol='arrow',
                        angle=np.degrees(np.arctan2(dy, dx)),
                        line=dict(width=0)
                    ),
                    showlegend=False,
                    legendgroup=legend_group,
                    hoverinfo='skip'
                ))
    
    return fig

# Create animated visualization with Plotly frames
def visualize_animated(df, selected_configs, selected_objects, start_time, end_time, 
                       aggregation_type, temporal_resolution, court_type='Football'):
    """Create smooth animation using Plotly's built-in animation"""
    
    # Get unique time steps from the data
    time_steps = sorted(df[(df['tst'] >= start_time) & (df['tst'] <= end_time)]['tst'].unique())
    
    if len(time_steps) == 0:
        # Fallback to linspace if no data
        time_steps = np.linspace(start_time, end_time, 50)
    
    # Initialize frames list
    frames = []
    
    # Create initial figure with fixed court dimensions
    fig = create_pitch_figure(court_type)
    
    # Prepare data for all objects at all times
    for frame_idx, current_time in enumerate(time_steps):
        frame_data = []
        
        for config in selected_configs:
            config_data = df[df['config_source'] == config]
            
            for obj_id in selected_objects:
                obj_data = config_data[config_data['obj'] == obj_id]
                obj_data = obj_data[(obj_data['tst'] >= start_time) & (obj_data['tst'] <= current_time)]
                obj_data = obj_data.sort_values('tst')
                
                if len(obj_data) == 0:
                    continue
                
                points = obj_data[['x', 'y', 'tst']].rename(columns={'tst': 'timestamp'}).to_dict('records')
                points = aggregate_points(points, aggregation_type, temporal_resolution)
                
                if len(points) == 0:
                    continue
                
                x_coords = [p['x'] for p in points]
                y_coords = [p['y'] for p in points]
                color = get_color(obj_id)
                
                # Add trajectory trace
                frame_data.append(go.Scatter(
                    x=x_coords, y=y_coords,
                    mode='lines',
                    name=f'{config} - Obj {obj_id}',
                    line=dict(color=color, width=2),
                    showlegend=(frame_idx == 0)
                ))
                
                # Add current position marker
                if points:
                    current_point = points[-1]
                    frame_data.append(go.Scatter(
                        x=[current_point['x']], y=[current_point['y']],
                        mode='markers',
                        marker=dict(size=10, color=color),
                        showlegend=False,
                        hovertemplate=f'Object {obj_id}<br>Config: {config}<br>Time: {current_time:.0f}<br>x: {current_point["x"]:.2f}m<br>y: {current_point["y"]:.2f}m<extra></extra>'
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
    fig.update_layout(
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {
                    'label': '▶ Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 200, 'redraw': False},
                        'fromcurrent': True,
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                },
                {
                    'label': '⏸ Pause',
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
                      start_time, aggregation_type, temporal_resolution, court_type='Football'):
    """Create visualization at specific time point"""
    fig = create_pitch_figure(court_type)
    
    for config in selected_configs:
        config_data = df[df['config_source'] == config]
        
        for obj_id in selected_objects:
            obj_data = config_data[config_data['obj'] == obj_id]
            obj_data = obj_data[(obj_data['tst'] >= start_time) & (obj_data['tst'] <= current_time)]
            obj_data = obj_data.sort_values('tst')
            
            if len(obj_data) == 0:
                continue
            
            points = obj_data[['x', 'y', 'tst']].rename(columns={'tst': 'timestamp'}).to_dict('records')
            points = aggregate_points(points, aggregation_type, temporal_resolution)
            
            if len(points) == 0:
                continue
            
            x_coords = [p['x'] for p in points]
            y_coords = [p['y'] for p in points]
            
            color = get_color(obj_id)
            
            # Draw trajectory
            fig.add_trace(go.Scatter(
                x=x_coords, y=y_coords,
                mode='lines',
                name=f'{config} - Obj {obj_id}',
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
                
                color = get_color(obj_id)
                
                fig.add_trace(go.Scatter(
                    x=[avg_x], y=[avg_y],
                    mode='markers+text',
                    marker=dict(size=15, color=color),
                    text=[f'Obj {obj_id}'],
                    textposition="top center",
                    name=f'{config} - Obj {obj_id} Avg',
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
            marker=dict(size=20, color='blue', symbol='star'),
            text=['Overall Avg'],
            textposition="top center",
            name='Overall Average',
            hovertemplate=f'Overall Average<br>x: {overall_avg_x:.2f}m<br>y: {overall_avg_y:.2f}m<extra></extra>'
        ))
    
    return fig

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
    st.title("📊 Spatiotemporal Analysis and Modeling")
    st.caption("Version 2.0 - Updated October 21, 2025")
    
    df = st.session_state.data
    uploaded_files = None
    
    # Sidebar
    with st.sidebar:
        st.header("📁 File Management")
        uploaded_files = st.file_uploader(
            "Upload CSV file(s)", type=['csv'], accept_multiple_files=True
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
            for file in uploaded_files:
                single_df = load_data(file, update_state=False, show_success=False)
                if single_df is not None:
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
            else:
                st.error("No valid data found in the uploaded file(s). Please verify the format.")
                df = None
        else:
            df = st.session_state.data
        
        if df is not None:
            st.info(f"📊 Current file(s): {st.session_state.filename}")
            
            st.header("Court Type")
            court_type = st.radio(
                "Select court type",
                ["Football", "Tennis"],
                index=0 if st.session_state.court_type == 'Football' else 1
            )
            st.session_state.court_type = court_type
            
            st.header("Analysis Method")
            analysis_method = st.selectbox(
                "Select method",
                ["Visual Exploration (IMO)", "2SA Method", "Heat Maps"]
            )
    
    # Main content
    if df is None:
        st.info("👆 Please upload a CSV file to begin.")
        st.markdown("""
        ### Expected CSV Format
        
        **For trajectory visualization (Option 1 - with header):**
        ```csv
        constant,timestamp,ID,x,y
        0,0,0,4.79,0.23
        0,1,0,3.76,17.73
        ...
        ```
        
        **For trajectory visualization (Option 2 - no header):**
        ```csv
        0,0,0,64.78,18.53
        0,1,0,54.26,20.68
        ...
        ```
        - Columns: Configuration, Timestamp, Object ID, x, y
        - Coordinates:
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
    if analysis_method == "Heat Maps":
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
    
    else:
        # Get unique configurations and objects
        config_sources = df['config_source'].drop_duplicates().tolist()
        if not config_sources:
            st.warning("No configurations found in the uploaded data. Please check your files.")
            return
        objects = sorted(df['obj'].unique())
        
        # Get time range with safety checks
        try:
            min_time = float(df['tst'].min())
            max_time = float(df['tst'].max())
            
            if pd.isna(min_time) or pd.isna(max_time):
                st.error("Invalid time values in data. Please check your CSV file.")
                return
                
        except (ValueError, TypeError) as e:
            st.error(f"Error processing time values: {str(e)}")
            return
        
        # Controls
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Selection")
            selected_configs = st.multiselect(
                "Select configuration(s)",
                config_sources,
                default=config_sources
            )
            
            selected_objects = st.multiselect(
                "Select object(s)",
                objects,
                default=objects[:min(5, len(objects))]
            )
        
        with col2:
            st.subheader("⏱️ Time Range")
            start_time = st.number_input(
                "Start time",
                min_value=min_time,
                max_value=max_time,
                value=min_time
            )
            
            end_time = st.number_input(
                "End time",
                min_value=start_time,
                max_value=max_time,
                value=max_time
            )
        
        # Method-specific controls
        if analysis_method == "2SA Method":
            st.subheader("🎯 2SA Method Controls")
            col1, col2 = st.columns(2)
            
            with col1:
                temporal_resolution = st.number_input("Temporal resolution", 
                                                     min_value=1, value=1, step=1)
            with col2:
                if st.button("Show S2A Start"):
                    translate_to_center = True
                else:
                    translate_to_center = False
            
            fig = visualize_static(df, selected_configs, selected_objects, 
                                 start_time, end_time, 'Skip frames', 
                                 temporal_resolution, translate_to_center, court_type)
            render_interactive_chart(fig)
        
        else:  # Visual Exploration (IMO)
            st.subheader("🎬 Animation Controls")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                aggregation_type = st.selectbox(
                    "Aggregation type",
                    ['Skip frames', 'Average locations', 'Spatially generalise', 
                     'Spatiotemporal generalise', 'Smoothing average']
                )
            
            with col2:
                temporal_resolution = st.number_input(
                    "Temporal resolution",
                    min_value=1, value=1, step=1
                )
            
            with col3:
                animation_duration = st.number_input(
                    "Animation duration (s)",
                    min_value=1, value=10, step=1
                )
            
            # Visualization mode
            viz_mode = st.radio(
                "Visualization mode",
                ["Static", "Animation", "Average Position"],
                horizontal=True
            )
            
            if viz_mode == "Static":
                fig = visualize_static(df, selected_configs, selected_objects, 
                                     start_time, end_time, aggregation_type, 
                                     temporal_resolution, False, court_type)
                render_interactive_chart(fig)
            
            elif viz_mode == "Average Position":
                fig = visualize_average_position(df, selected_configs, selected_objects, 
                                                start_time, end_time, court_type)
                render_interactive_chart(fig)
            
            else:  # Animation
                # Use Plotly's built-in animation for smooth playback
                st.info("📹 Use the ▶ Play button and slider below the chart to control the animation. The slider shows actual time steps from your data.")
                
                fig = visualize_animated(df, selected_configs, selected_objects, 
                                        start_time, end_time, aggregation_type, 
                                        temporal_resolution, court_type)
                
                render_interactive_chart(fig)

if __name__ == "__main__":
    main()
