import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist, euclidean
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.express as px

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
            
            # Draw trajectory line and markers, but hide the last marker
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

            # Add arrow at the end using a separate scatter trace
            if len(x_coords) >= 2:
                dx = x_coords[-1] - x_coords[-2]
                dy = y_coords[-1] - y_coords[-2]
                # Swapping dx and dy in arctan2 rotates the coordinate system by 90 degrees,
                # aligning the calculation's 0-degree reference (east) with the arrow's default orientation (north).
                angle = np.degrees(np.arctan2(dx, dy))

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
                    hoverinfo='skip'
                ))
    
    # Preserve zoom state on redraw
    fig.update_layout(uirevision='constant')
    
    return fig
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
            marker=dict(size=20, color='black', symbol='star'),
            text=['Overall Avg'],
            textposition="top center",
            name='Overall Average',
            hovertemplate=f'Overall Average<br>x: {overall_avg_x:.2f}m<br>y: {overall_avg_y:.2f}m<extra></extra>'
        ))
    
    return fig

# ============================================================================
# CLUSTERING FUNCTIONS
# ============================================================================

def extract_trajectory_features(df, obj_id, config, start_time, end_time):
    """Extract general properties features from a trajectory"""
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
                ["Visual Exploration (IMO)", "2SA Method", "Heat Maps", "Clustering"]
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
    
    elif analysis_method == "Clustering":
        st.header("🔍 Clustering Methods")
        
        clustering_methods = [
            "Similarities of general properties",
            "Similarities of general movements",
            "Spatial constraints",
            "Spatiotemporal constraints (moving flocks)",
            "Attribute trajectories",
            "Heat map animations",
            "PDP",
            "QTC"
        ]
        
        selected_clustering_method = st.selectbox(
            "Select a clustering method:",
            clustering_methods
        )
        
        st.divider()
        
        # Get common parameters for clustering
        config_sources = df['config_source'].drop_duplicates().tolist()
        objects = sorted(df['obj'].unique())
        
        col1, col2 = st.columns(2)
        with col1:
            selected_configs = st.multiselect(
                "Select configuration(s)",
                config_sources,
                default=config_sources,
                key="clustering_configs"
            )
        with col2:
            selected_objects = st.multiselect(
                "Select object(s)",
                objects,
                default=objects[:min(5, len(objects))],
                key="clustering_objects"
            )
        
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
        
        st.divider()
        
        # Display method-specific information and controls
        if selected_clustering_method == "Similarities of general properties":
            st.subheader("📊 Similarities of General Properties")
            st.info("This method clusters trajectories based on similar general properties such as length, duration, average speed, etc.")
            
            # Feature selection
            st.subheader("Select Features for Clustering")
            all_features = {
                'total_distance': 'Total Distance',
                'duration': 'Duration',
                'avg_speed': 'Average Speed',
                'max_speed': 'Maximum Speed',
                'displacement': 'Displacement (Start-to-End Distance)',
                'sinuosity': 'Sinuosity (Path Curvature)',
                'bbox_area': 'Bounding Box Area',
                'direction': 'Direction (Bearing)'
            }
            
            col1, col2 = st.columns(2)
            with col1:
                selected_features_left = st.multiselect(
                    "Movement characteristics",
                    options=['total_distance', 'displacement', 'sinuosity', 'direction'],
                    default=['total_distance', 'displacement', 'sinuosity'],
                    format_func=lambda x: all_features[x],
                    key="features_left"
                )
            with col2:
                selected_features_right = st.multiselect(
                    "Speed and spatial metrics",
                    options=['duration', 'avg_speed', 'max_speed', 'bbox_area'],
                    default=['avg_speed', 'max_speed'],
                    format_func=lambda x: all_features[x],
                    key="features_right"
                )
            
            selected_feature_cols = selected_features_left + selected_features_right
            
            if len(selected_feature_cols) == 0:
                st.warning("⚠️ Please select at least one feature for clustering.")
            else:
                st.success(f"✓ Selected {len(selected_feature_cols)} feature(s): {', '.join([all_features[f] for f in selected_feature_cols])}")
            
            # Clustering parameters
            st.subheader("Clustering Parameters")
            
            clustering_algo = st.selectbox(
                "Clustering algorithm",
                ["K-Means", "Hierarchical", "DBSCAN"],
                key="prop_algo"
            )
            
            # Show appropriate parameters based on algorithm
            if clustering_algo == "DBSCAN":
                col1, col2 = st.columns(2)
                with col1:
                    eps = st.slider("EPS (neighborhood radius)", 0.1, 5.0, 0.5, step=0.1, key="dbscan_eps")
                with col2:
                    min_samples = st.slider("Min samples (core point threshold)", 2, 10, 2, key="dbscan_min_samples")
                st.info("💡 DBSCAN automatically determines the number of clusters based on data density. Points labeled as -1 are outliers.")
            else:
                n_clusters = st.slider("Number of clusters", 2, 10, 3, key="prop_clusters")
            
            if st.button("Run Clustering", key="run_prop_clustering", disabled=len(selected_feature_cols) == 0):
                with st.spinner("Extracting features and clustering..."):
                    # Extract features for all trajectories
                    features_list = []
                    traj_ids = []
                    
                    for config in selected_configs:
                        for obj_id in selected_objects:
                            features = extract_trajectory_features(df, obj_id, config, start_time, end_time)
                            if features is not None:
                                features_list.append(features)
                                traj_ids.append(f"{config}-Obj{obj_id}")
                    
                    if len(features_list) < 2:
                        st.error("Not enough trajectories to cluster. Please select more objects or check your data.")
                    else:
                        # Create feature matrix with selected features
                        features_df = pd.DataFrame(features_list)
                        
                        # Check if all selected features exist in the data
                        missing_features = [f for f in selected_feature_cols if f not in features_df.columns]
                        if missing_features:
                            st.error(f"Missing features in data: {', '.join(missing_features)}")
                        else:
                            X = features_df[selected_feature_cols].values
                            
                            # Normalize features
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(X)
                            
                            # Perform clustering
                            if clustering_algo == "K-Means":
                                clusterer = KMeans(n_clusters=n_clusters, random_state=42)
                                labels = clusterer.fit_predict(X_scaled)
                            elif clustering_algo == "Hierarchical":
                                # Create linkage matrix for dendrogram
                                linkage_matrix = linkage(X_scaled, method='ward')
                                
                                # Display dendrogram
                                st.subheader("📊 Dendrogram")
                                st.info("The dendrogram shows how trajectories are grouped hierarchically. Taller branches indicate larger distances between clusters.")
                                
                                # Create dendrogram figure
                                fig_dend = go.Figure()
                                
                                # Compute dendrogram data
                                dend_data = dendrogram(linkage_matrix, labels=traj_ids, no_plot=True)
                                
                                # Extract x and y coordinates for dendrogram lines
                                icoord = np.array(dend_data['icoord'])
                                dcoord = np.array(dend_data['dcoord'])
                                
                                # Plot dendrogram branches
                                for i in range(len(icoord)):
                                    fig_dend.add_trace(go.Scatter(
                                        x=icoord[i],
                                        y=dcoord[i],
                                        mode='lines',
                                        line=dict(color='rgb(100,100,100)', width=1.5),
                                        hoverinfo='skip',
                                        showlegend=False
                                    ))
                                
                                # Get the order of leaves (trajectories) in the dendrogram
                                leaves_order = dend_data['leaves']
                                
                                # Add trajectory labels
                                fig_dend.update_layout(
                                    title="Hierarchical Clustering Dendrogram",
                                    xaxis=dict(
                                        title="Trajectory",
                                        tickmode='array',
                                        tickvals=[(i + 0.5) * 10 for i in range(len(leaves_order))],
                                        ticktext=[traj_ids[i] for i in leaves_order],
                                        tickangle=-45
                                    ),
                                    yaxis=dict(title="Distance"),
                                    height=500,
                                    showlegend=False
                                )
                                
                                render_interactive_chart(fig_dend, "Dendrogram showing hierarchical clustering structure")
                                
                                # Perform clustering with specified number of clusters
                                clusterer = AgglomerativeClustering(n_clusters=n_clusters)
                                labels = clusterer.fit_predict(X_scaled)
                            else:  # DBSCAN
                                clusterer = DBSCAN(eps=eps, min_samples=min_samples)
                                labels = clusterer.fit_predict(X_scaled)
                            
                            features_df['cluster'] = labels
                            features_df['trajectory_id'] = traj_ids
                            
                            # Display results
                            n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
                            n_outliers = list(labels).count(-1)
                            
                            if clustering_algo == "DBSCAN":
                                st.success(f"✅ Found {n_clusters_found} cluster(s) and {n_outliers} outlier(s)!")
                            else:
                                st.success(f"✅ Found {n_clusters_found} clusters!")
                            
                            # Show cluster assignments
                            st.subheader("Cluster Assignments")
                            for cluster_id in sorted(set(labels)):
                                cluster_trajs = features_df[features_df['cluster'] == cluster_id]['trajectory_id'].tolist()
                                if cluster_id == -1:
                                    st.write(f"**Outliers (noise):** {', '.join(cluster_trajs)}")
                                else:
                                    st.write(f"**Cluster {cluster_id}:** {', '.join(cluster_trajs)}")
                        
                            # Visualize clusters on court
                            st.subheader("Cluster Visualization")
                            fig = create_pitch_figure(court_type)
                            
                            colors = px.colors.qualitative.Plotly
                            for idx, row in features_df.iterrows():
                                config = row['config']
                                obj_id = row['obj_id']
                                cluster_id = row['cluster']
                                
                                # Use gray for outliers (-1), otherwise use color palette
                                if cluster_id == -1:
                                    color = 'gray'
                                    label = f"Outlier: {config}-Obj{obj_id}"
                                else:
                                    color = colors[cluster_id % len(colors)]
                                    label = f"C{cluster_id}: {config}-Obj{obj_id}"
                                
                                # Get trajectory
                                obj_data = df[(df['obj'] == obj_id) & 
                                            (df['config_source'] == config) &
                                            (df['tst'] >= start_time) & 
                                            (df['tst'] <= end_time)].sort_values('tst')
                                
                                if len(obj_data) > 0:
                                    fig.add_trace(go.Scatter(
                                        x=obj_data['x'], y=obj_data['y'],
                                        mode='lines',
                                        name=label,
                                    line=dict(color=color, width=2),
                                    legendgroup=f"cluster_{cluster_id}"
                                ))
                            
                            render_interactive_chart(fig, "Trajectories colored by cluster")
                            
                            # Feature comparison
                            st.subheader("Feature Statistics by Cluster")
                            
                            # Create feature statistics with units
                            feature_stats = features_df.groupby('cluster')[selected_feature_cols].mean()
                            
                            # Define units for each feature
                            feature_units = {
                                'total_distance': 'Total Distance (m)',
                                'duration': 'Duration (s)',
                                'avg_speed': 'Average Speed (m/s)',
                                'max_speed': 'Maximum Speed (m/s)',
                                'displacement': 'Displacement (m)',
                                'sinuosity': 'Sinuosity (ratio)',
                                'bbox_area': 'Bounding Box Area (m²)',
                                'direction': 'Direction (°)'
                            }
                            
                            # Rename index for outliers
                            feature_stats.index = ['Outliers' if idx == -1 else f'Cluster {idx}' for idx in feature_stats.index]
                            
                            # Rename columns with units
                            feature_stats.columns = [feature_units.get(col, col) for col in feature_stats.columns]
                            
                            # Format all values to 2 decimal places
                            feature_stats_formatted = feature_stats.apply(lambda x: x.map('{:.2f}'.format))
                        
                        # Display with optimized column width
                        st.dataframe(
                            feature_stats_formatted,
                            use_container_width=False,
                            column_config={
                                col: st.column_config.TextColumn(
                                    col,
                                    width="medium"
                                ) for col in feature_stats_formatted.columns
                            }
                        )
            
        elif selected_clustering_method == "Similarities of general movements":
            st.subheader("🔄 Similarities of General Movements")
            st.info("This method clusters trajectories based on similar movement patterns using distance metrics like DTW or Hausdorff distance.")
            
            # Distance metric selection
            distance_metric = st.selectbox(
                "Distance metric",
                ["Dynamic Time Warping (DTW)", "Hausdorff Distance"],
                key="movement_metric"
            )
            
            n_clusters = st.slider("Number of clusters", 2, 10, 3, key="movement_clusters")
            
            if st.button("Run Clustering", key="run_movement_clustering"):
                with st.spinner(f"Calculating {distance_metric} distances..."):
                    # Get all trajectories
                    trajectories = []
                    traj_ids = []
                    
                    for config in selected_configs:
                        for obj_id in selected_objects:
                            coords = get_trajectory_coords(df, obj_id, config, start_time, end_time)
                            if coords is not None:
                                trajectories.append(coords)
                                traj_ids.append(f"{config}-Obj{obj_id}")
                    
                    if len(trajectories) < 2:
                        st.error("Not enough trajectories to cluster.")
                    else:
                        # Calculate distance matrix
                        n = len(trajectories)
                        dist_matrix = np.zeros((n, n))
                        
                        progress_bar = st.progress(0)
                        total_pairs = n * (n - 1) // 2
                        completed = 0
                        
                        for i in range(n):
                            for j in range(i + 1, n):
                                if distance_metric == "Dynamic Time Warping (DTW)":
                                    dist = dtw_distance(trajectories[i], trajectories[j])
                                else:  # Hausdorff
                                    dist = hausdorff_distance(trajectories[i], trajectories[j])
                                
                                dist_matrix[i, j] = dist
                                dist_matrix[j, i] = dist
                                completed += 1
                                progress_bar.progress(completed / total_pairs)
                        
                        progress_bar.empty()
                        
                        # Perform hierarchical clustering
                        clusterer = AgglomerativeClustering(
                            n_clusters=n_clusters,
                            metric='precomputed',
                            linkage='average'
                        )
                        labels = clusterer.fit_predict(dist_matrix)
                        
                        # Display results
                        st.success(f"✅ Found {n_clusters} clusters!")
                        
                        # Show cluster assignments
                        st.subheader("Cluster Assignments")
                        cluster_df = pd.DataFrame({
                            'Trajectory': traj_ids,
                            'Cluster': labels
                        })
                        
                        for cluster_id in range(n_clusters):
                            cluster_trajs = cluster_df[cluster_df['Cluster'] == cluster_id]['Trajectory'].tolist()
                            st.write(f"**Cluster {cluster_id}:** {', '.join(cluster_trajs)}")
                        
                        # Visualize clusters
                        st.subheader("Cluster Visualization")
                        fig = create_pitch_figure(court_type)
                        
                        colors = px.colors.qualitative.Plotly
                        for idx, traj_id in enumerate(traj_ids):
                            cluster_id = labels[idx]
                            color = colors[cluster_id % len(colors)]
                            
                            config, obj_part = traj_id.split('-Obj')
                            obj_id = int(float(obj_part))
                            
                            obj_data = df[(df['obj'] == obj_id) & 
                                        (df['config_source'] == config) &
                                        (df['tst'] >= start_time) & 
                                        (df['tst'] <= end_time)].sort_values('tst')
                            
                            if len(obj_data) > 0:
                                fig.add_trace(go.Scatter(
                                    x=obj_data['x'], y=obj_data['y'],
                                    mode='lines',
                                    name=f"C{cluster_id}: {traj_id}",
                                    line=dict(color=color, width=2),
                                    legendgroup=f"cluster_{cluster_id}"
                                ))
                        
                        render_interactive_chart(fig, "Trajectories colored by movement similarity")
                        
                        # Distance matrix heatmap
                        st.subheader("Distance Matrix")
                        fig_dist = go.Figure(data=go.Heatmap(
                            z=dist_matrix,
                            x=traj_ids,
                            y=traj_ids,
                            colorscale='Viridis',
                            hovertemplate='%{x} to %{y}: %{z:.2f}<extra></extra>'
                        ))
                        fig_dist.update_layout(
                            title=f'{distance_metric} Distance Matrix',
                            height=600,
                            xaxis={'tickangle': 45}
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)
            
        elif selected_clustering_method == "Spatial constraints":
            st.subheader("📍 Spatial Constraints")
            st.info("This method clusters trajectories based on spatial proximity and location constraints.")
            
            method = st.radio(
                "Clustering method",
                ["Chamfer Distance", "Grid-based", "Start/End Point", "Spatial Zone"],
                key="spatial_method"
            )
            
            if method == "Chamfer Distance":
                st.info("💡 Chamfer distance calculates the average distance from each point in one trajectory to its nearest point in another trajectory. It's simple, intuitive, and robust to outliers - perfect for finding trajectories that follow similar routes.")
                
                n_clusters = st.slider("Number of clusters", 2, 10, 3, key="chamfer_clusters")
                
                if st.button("Run Clustering", key="run_chamfer_clustering"):
                    with st.spinner("Calculating Chamfer distances..."):
                        # Get all trajectories
                        trajectories = []
                        traj_ids = []
                        
                        for config in selected_configs:
                            for obj_id in selected_objects:
                                coords = get_trajectory_coords(df, obj_id, config, start_time, end_time)
                                if coords is not None:
                                    trajectories.append(coords)
                                    traj_ids.append(f"{config}-Obj{obj_id}")
                        
                        if len(trajectories) < 2:
                            st.error("Not enough trajectories to cluster.")
                        else:
                            # Calculate Chamfer distance matrix
                            n = len(trajectories)
                            dist_matrix = np.zeros((n, n))
                            
                            progress_bar = st.progress(0)
                            total_pairs = n * (n - 1) // 2
                            completed = 0
                            
                            for i in range(n):
                                for j in range(i + 1, n):
                                    dist = chamfer_distance(trajectories[i], trajectories[j])
                                    dist_matrix[i, j] = dist
                                    dist_matrix[j, i] = dist
                                    completed += 1
                                    progress_bar.progress(completed / total_pairs)
                            
                            progress_bar.empty()
                            
                            # Perform hierarchical clustering
                            clusterer = AgglomerativeClustering(
                                n_clusters=n_clusters,
                                metric='precomputed',
                                linkage='average'
                            )
                            labels = clusterer.fit_predict(dist_matrix)
                            
                            # Display results
                            st.success(f"✅ Found {n_clusters} clusters based on route similarity!")
                            
                            # Show cluster assignments
                            st.subheader("Cluster Assignments")
                            cluster_df = pd.DataFrame({
                                'Trajectory': traj_ids,
                                'Cluster': labels
                            })
                            
                            for cluster_id in range(n_clusters):
                                cluster_trajs = cluster_df[cluster_df['Cluster'] == cluster_id]['Trajectory'].tolist()
                                st.write(f"**Cluster {cluster_id}:** {', '.join(cluster_trajs)}")
                            
                            # Visualize clusters
                            st.subheader("Cluster Visualization")
                            fig = create_pitch_figure(court_type)
                            
                            colors = px.colors.qualitative.Plotly
                            for idx, traj_id in enumerate(traj_ids):
                                cluster_id = labels[idx]
                                color = colors[cluster_id % len(colors)]
                                
                                config, obj_part = traj_id.split('-Obj')
                                obj_id = int(float(obj_part))
                                
                                obj_data = df[(df['obj'] == obj_id) & 
                                            (df['config_source'] == config) &
                                            (df['tst'] >= start_time) & 
                                            (df['tst'] <= end_time)].sort_values('tst')
                                
                                if len(obj_data) > 0:
                                    fig.add_trace(go.Scatter(
                                        x=obj_data['x'], y=obj_data['y'],
                                        mode='lines',
                                        name=f"C{cluster_id}: {traj_id}",
                                        line=dict(color=color, width=2),
                                        legendgroup=f"cluster_{cluster_id}"
                                    ))
                            
                            render_interactive_chart(fig, "Trajectories colored by route similarity")
                            
                            # Distance matrix heatmap
                            st.subheader("Chamfer Distance Matrix")
                            st.info("Lower values (darker) indicate more similar routes. This shows the average distance between trajectories.")
                            
                            fig_dist = go.Figure(data=go.Heatmap(
                                z=dist_matrix,
                                x=traj_ids,
                                y=traj_ids,
                                colorscale='Viridis',
                                hovertemplate='%{x} to %{y}: %{z:.2f}m<extra></extra>'
                            ))
                            fig_dist.update_layout(
                                title='Chamfer Distance Matrix (meters)',
                                height=600,
                                xaxis={'tickangle': 45}
                            )
                            st.plotly_chart(fig_dist, use_container_width=True)
            
            elif method == "Grid-based":
                grid_size = st.slider("Grid cell size (meters)", 1.0, 10.0, 3.0, step=0.5, key="grid_size")
                
                if st.button("Run Clustering", key="run_spatial_grid"):
                    with st.spinner("Analyzing spatial patterns..."):
                        trajectory_grids = grid_based_clustering(
                            df, selected_configs, selected_objects,
                            start_time, end_time, grid_size
                        )
                        
                        if len(trajectory_grids) < 2:
                            st.error("Not enough trajectories to cluster.")
                        else:
                            # Calculate similarity based on Jaccard index
                            traj_ids = list(trajectory_grids.keys())
                            n = len(traj_ids)
                            similarity_matrix = np.zeros((n, n))
                            
                            for i in range(n):
                                for j in range(n):
                                    set1 = trajectory_grids[traj_ids[i]]
                                    set2 = trajectory_grids[traj_ids[j]]
                                    intersection = len(set1 & set2)
                                    union = len(set1 | set2)
                                    similarity_matrix[i, j] = intersection / union if union > 0 else 0
                            
                            # Convert to distance
                            dist_matrix = 1 - similarity_matrix
                            
                            # Cluster
                            n_clusters = st.slider("Number of clusters", 2, min(10, n), 3, key="spatial_n_clusters")
                            clusterer = AgglomerativeClustering(
                                n_clusters=n_clusters,
                                metric='precomputed',
                                linkage='average'
                            )
                            labels = clusterer.fit_predict(dist_matrix)
                            
                            # Display results
                            st.success(f"✅ Found {n_clusters} clusters based on spatial patterns!")
                            
                            st.subheader("Cluster Assignments")
                            for cluster_id in range(n_clusters):
                                cluster_trajs = [f"{c}-Obj{o}" for (c, o), lbl in zip(traj_ids, labels) if lbl == cluster_id]
                                st.write(f"**Cluster {cluster_id}:** {', '.join(cluster_trajs)}")
                            
                            # Visualize
                            fig = create_pitch_figure(court_type)
                            colors = px.colors.qualitative.Plotly
                            
                            for idx, (config, obj_id) in enumerate(traj_ids):
                                cluster_id = labels[idx]
                                color = colors[cluster_id % len(colors)]
                                
                                obj_data = df[(df['obj'] == obj_id) & 
                                            (df['config_source'] == config) &
                                            (df['tst'] >= start_time) & 
                                            (df['tst'] <= end_time)].sort_values('tst')
                                
                                if len(obj_data) > 0:
                                    fig.add_trace(go.Scatter(
                                        x=obj_data['x'], y=obj_data['y'],
                                        mode='lines',
                                        name=f"C{cluster_id}: {config}-Obj{obj_id}",
                                        line=dict(color=color, width=2)
                                    ))
                            
                            render_interactive_chart(fig, "Trajectories colored by spatial patterns")
            
            elif method == "Start/End Point":
                if st.button("Run Clustering", key="run_spatial_endpoints"):
                    with st.spinner("Clustering by start/end points..."):
                        # Extract start and end points
                        start_points = []
                        end_points = []
                        traj_ids = []
                        
                        for config in selected_configs:
                            for obj_id in selected_objects:
                                obj_data = df[(df['obj'] == obj_id) & 
                                            (df['config_source'] == config) &
                                            (df['tst'] >= start_time) & 
                                            (df['tst'] <= end_time)].sort_values('tst')
                                
                                if len(obj_data) >= 2:
                                    start_points.append([obj_data.iloc[0]['x'], obj_data.iloc[0]['y']])
                                    end_points.append([obj_data.iloc[-1]['x'], obj_data.iloc[-1]['y']])
                                    traj_ids.append(f"{config}-Obj{obj_id}")
                        
                        if len(start_points) < 2:
                            st.error("Not enough trajectories.")
                        else:
                            # Combine start and end points as features
                            X = np.hstack([np.array(start_points), np.array(end_points)])
                            
                            n_clusters = st.slider("Number of clusters", 2, min(10, len(X)), 3, key="endpoint_clusters")
                            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                            labels = kmeans.fit_predict(X)
                            
                            st.success(f"✅ Clustered by start/end points!")
                            
                            st.subheader("Cluster Assignments")
                            for cluster_id in range(n_clusters):
                                cluster_trajs = [traj_ids[i] for i in range(len(traj_ids)) if labels[i] == cluster_id]
                                st.write(f"**Cluster {cluster_id}:** {', '.join(cluster_trajs)}")
                            
                            # Visualize
                            fig = create_pitch_figure(court_type)
                            colors = px.colors.qualitative.Plotly
                            
                            for idx, traj_id in enumerate(traj_ids):
                                cluster_id = labels[idx]
                                color = colors[cluster_id % len(colors)]
                                
                                config, obj_part = traj_id.split('-Obj')
                                obj_id = int(float(obj_part))
                                
                                obj_data = df[(df['obj'] == obj_id) & 
                                            (df['config_source'] == config) &
                                            (df['tst'] >= start_time) & 
                                            (df['tst'] <= end_time)].sort_values('tst')
                                
                                if len(obj_data) > 0:
                                    fig.add_trace(go.Scatter(
                                        x=obj_data['x'], y=obj_data['y'],
                                        mode='lines+markers',
                                        name=f"C{cluster_id}: {traj_id}",
                                        line=dict(color=color, width=2),
                                        marker=dict(
                                            size=[10] + [4]*(len(obj_data)-2) + [10],
                                            color=color,
                                            symbol=['circle'] + ['circle']*(len(obj_data)-2) + ['square']
                                        )
                                    ))
                            
                            render_interactive_chart(fig, "Trajectories colored by start/end point clusters")
            
            else:  # Spatial Zone
                st.info("Draw a zone on the field and cluster trajectories that pass through it vs. those that don't.")
                st.warning("⚠️ Interactive zone drawing coming soon. For now, use grid-based clustering.")
            
        elif selected_clustering_method == "Spatiotemporal constraints (moving flocks)":
            st.subheader("🐦 Spatiotemporal Constraints (Moving Flocks)")
            st.info("This method identifies groups of objects moving together in space and time (flocking behavior).")
            
            col1, col2 = st.columns(2)
            with col1:
                distance_threshold = st.slider(
                    "Distance threshold (meters)",
                    1.0, 20.0, 5.0, step=0.5,
                    key="flock_distance",
                    help="Maximum distance between objects to be considered part of the same flock"
                )
            with col2:
                min_duration = st.slider(
                    "Minimum duration (time steps)",
                    1, 50, 5,
                    key="flock_duration",
                    help="Minimum number of consecutive time steps for a valid flock"
                )
            
            if st.button("Detect Flocks", key="run_flock_detection"):
                with st.spinner("Detecting moving flocks..."):
                    flocks = find_moving_flocks(
                        df, selected_configs, selected_objects,
                        start_time, end_time,
                        distance_threshold, min_duration
                    )
                    
                    if len(flocks) == 0:
                        st.warning("No flocks detected with these parameters. Try increasing the distance threshold or decreasing the minimum duration.")
                    else:
                        st.success(f"✅ Detected {len(flocks)} flock(s)!")
                        
                        # Display flock details
                        st.subheader("Detected Flocks")
                        for idx, flock in enumerate(flocks):
                            members = list(flock['members'])
                            member_names = [f"{c}-Obj{o}" for c, o in members]
                            duration = flock['end_time'] - flock['start_time']
                            
                            with st.expander(f"Flock {idx + 1}: {len(members)} members, duration {duration:.0f}"):
                                st.write(f"**Members:** {', '.join(member_names)}")
                                st.write(f"**Time range:** {flock['start_time']:.0f} to {flock['end_time']:.0f}")
                                st.write(f"**Duration:** {duration:.0f} time steps")
                        
                        # Visualize flocks
                        st.subheader("Flock Visualization")
                        fig = create_pitch_figure(court_type)
                        colors = px.colors.qualitative.Set1
                        
                        # Assign each trajectory to a flock color
                        traj_to_flock = {}
                        for idx, flock in enumerate(flocks):
                            for member in flock['members']:
                                traj_to_flock[member] = idx
                        
                        # Plot all trajectories
                        for config in selected_configs:
                            for obj_id in selected_objects:
                                obj_data = df[(df['obj'] == obj_id) & 
                                            (df['config_source'] == config) &
                                            (df['tst'] >= start_time) & 
                                            (df['tst'] <= end_time)].sort_values('tst')
                                
                                if len(obj_data) > 0:
                                    key = (config, obj_id)
                                    if key in traj_to_flock:
                                        flock_id = traj_to_flock[key]
                                        color = colors[flock_id % len(colors)]
                                        name = f"Flock {flock_id + 1}: {config}-Obj{obj_id}"
                                        width = 3
                                    else:
                                        color = 'lightgray'
                                        name = f"No flock: {config}-Obj{obj_id}"
                                        width = 1
                                    
                                    fig.add_trace(go.Scatter(
                                        x=obj_data['x'], y=obj_data['y'],
                                        mode='lines',
                                        name=name,
                                        line=dict(color=color, width=width)
                                    ))
                        
                        render_interactive_chart(fig, "Trajectories colored by flock membership (gray = no flock)")
            
        elif selected_clustering_method == "Attribute trajectories":
            st.subheader("📈 Attribute Trajectories")
            st.info("This method clusters trajectories based on temporal patterns of attributes like speed or acceleration.")
            
            attribute = st.selectbox(
                "Select attribute",
                ["Speed", "Acceleration"],
                key="attribute_type"
            )
            
            n_clusters = st.slider("Number of clusters", 2, 10, 3, key="attribute_clusters")
            
            if st.button("Run Clustering", key="run_attribute_clustering"):
                with st.spinner(f"Analyzing {attribute.lower()} patterns..."):
                    # Calculate attribute trajectories
                    speed_trajectories = []
                    traj_ids = []
                    
                    for config in selected_configs:
                        for obj_id in selected_objects:
                            speeds = calculate_speed_trajectory(df, obj_id, config, start_time, end_time)
                            if speeds is not None and len(speeds) > 2:
                                speed_trajectories.append(speeds)
                                traj_ids.append(f"{config}-Obj{obj_id}")
                    
                    if len(speed_trajectories) < 2:
                        st.error("Not enough trajectories to cluster.")
                    else:
                        # Pad/interpolate to same length for comparison
                        max_len = max(len(s) for s in speed_trajectories)
                        
                        # Simple linear interpolation to common length
                        normalized_speeds = []
                        for speeds in speed_trajectories:
                            if len(speeds) < max_len:
                                x_old = np.linspace(0, 1, len(speeds))
                                x_new = np.linspace(0, 1, max_len)
                                speeds_interp = np.interp(x_new, x_old, speeds)
                                normalized_speeds.append(speeds_interp)
                            else:
                                normalized_speeds.append(speeds[:max_len])
                        
                        X = np.array(normalized_speeds)
                        
                        # Calculate acceleration if needed
                        if attribute == "Acceleration":
                            X = np.diff(X, axis=1)
                        
                        # Cluster using K-means
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                        labels = kmeans.fit_predict(X)
                        
                        st.success(f"✅ Clustered by {attribute.lower()} patterns!")
                        
                        # Display cluster assignments
                        st.subheader("Cluster Assignments")
                        for cluster_id in range(n_clusters):
                            cluster_trajs = [traj_ids[i] for i in range(len(traj_ids)) if labels[i] == cluster_id]
                            st.write(f"**Cluster {cluster_id}:** {', '.join(cluster_trajs)}")
                        
                        # Plot attribute patterns by cluster
                        st.subheader(f"{attribute} Patterns by Cluster")
                        fig = go.Figure()
                        colors = px.colors.qualitative.Plotly
                        
                        for idx, (speeds, traj_id) in enumerate(zip(normalized_speeds, traj_ids)):
                            cluster_id = labels[idx]
                            color = colors[cluster_id % len(colors)]
                            
                            data = speeds if attribute == "Speed" else np.diff(speeds)
                            
                            fig.add_trace(go.Scatter(
                                y=data,
                                mode='lines',
                                name=f"C{cluster_id}: {traj_id}",
                                line=dict(color=color),
                                legendgroup=f"cluster_{cluster_id}"
                            ))
                        
                        fig.update_layout(
                            title=f"{attribute} Over Time",
                            xaxis_title="Time Step",
                            yaxis_title=f"{attribute} (m/s)" if attribute == "Speed" else f"{attribute} (m/s²)",
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Visualize trajectories colored by cluster
                        st.subheader("Spatial Trajectories")
                        fig_spatial = create_pitch_figure(court_type)
                        
                        for idx, traj_id in enumerate(traj_ids):
                            cluster_id = labels[idx]
                            color = colors[cluster_id % len(colors)]
                            
                            config, obj_part = traj_id.split('-Obj')
                            obj_id = int(float(obj_part))
                            
                            obj_data = df[(df['obj'] == obj_id) & 
                                        (df['config_source'] == config) &
                                        (df['tst'] >= start_time) & 
                                        (df['tst'] <= end_time)].sort_values('tst')
                            
                            if len(obj_data) > 0:
                                fig_spatial.add_trace(go.Scatter(
                                    x=obj_data['x'], y=obj_data['y'],
                                    mode='lines',
                                    name=f"C{cluster_id}: {traj_id}",
                                    line=dict(color=color, width=2)
                                ))
                        
                        render_interactive_chart(fig_spatial, f"Trajectories colored by {attribute.lower()} pattern cluster")
            
        elif selected_clustering_method == "Heat map animations":
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
            
        elif selected_clustering_method == "PDP":
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
            
        elif selected_clustering_method == "QTC":
            st.subheader("🔢 QTC (Qualitative Trajectory Calculus)")
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
                value=min_time,
                step=0.01,
                format="%.2f"
            )
            
            end_time = st.number_input(
                "End time",
                min_value=start_time,
                max_value=max_time,
                value=max_time,
                step=0.01,
                format="%.2f"
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
