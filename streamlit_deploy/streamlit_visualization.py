import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.spatial.distance import cdist, euclidean
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import plotly.express as px
from itertools import groupby
from collections import Counter

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
    """Load and parse CSV file - supports two formats:
    Format 1: Multiple files, each is one configuration
    Format 2: Single file with multiple configurations (column 0), optional config names in column 5
    """
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
        
        # Read the CSV - try to read all columns including optional column 5
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
            # No header - check number of columns
            df = pd.read_csv(uploaded_file, header=None)
            num_cols = len(df.columns)
            
            if num_cols >= 6:
                # Format with config names in column 5
                df.columns = ['con', 'tst', 'obj', 'x', 'y', 'config_name'] + [f'col_{i}' for i in range(6, num_cols)]
            elif num_cols >= 5:
                df.columns = ['con', 'tst', 'obj', 'x', 'y'] + [f'col_{i}' for i in range(5, num_cols)]
            else:
                st.error(f"Expected at least 5 columns, found {num_cols}")
                return None
        
        # Check if required columns exist
        required_cols = ['con', 'tst', 'obj', 'x', 'y']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            st.error(f"Found columns: {', '.join(df.columns.tolist())}")
            return None
        
        # Check if config_name column exists
        has_config_names = 'config_name' in df.columns
        
        # Convert columns to numeric types
        df['con'] = pd.to_numeric(df['con'], errors='coerce')
        df['tst'] = pd.to_numeric(df['tst'], errors='coerce')
        df['obj'] = pd.to_numeric(df['obj'], errors='coerce')
        df['x'] = pd.to_numeric(df['x'], errors='coerce')
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        
        # Remove rows where required numeric columns have NaN values
        initial_rows = len(df)
        df = df.dropna(subset=['con', 'tst', 'obj', 'x', 'y'])
        
        if len(df) == 0:
            st.error(f"No valid data rows found. All {initial_rows} rows had invalid or missing values.")
            st.info("Please ensure your CSV has numeric values in columns: con, tst, obj, x, y")
            return None
        
        if len(df) < initial_rows:
            st.warning(f"Removed {initial_rows - len(df)} rows with invalid data. {len(df)} rows remaining.")
        
        # Create config_source based on format
        file_name = getattr(uploaded_file, 'name', 'uploaded data')
        unique_configs = df['con'].unique()
        
        if len(unique_configs) > 1:
            # Format 2: Single file with multiple configurations
            if has_config_names:
                # Use config names from column 5
                df['config_source'] = df['config_name'].astype(str) + ' (c' + df['con'].astype(int).astype(str) + ')'
            else:
                # Use configuration numbers
                df['config_source'] = 'c' + df['con'].astype(int).astype(str) + '.csv'
            
            config_sources = df['config_source'].unique().tolist()
            if show_success:
                st.success(f"✅ Loaded {len(df)} data points from {file_name} with {len(unique_configs)} configurations!")
                st.info(f"Configurations found: {', '.join(config_sources)}")
        else:
            # Format 1: Single configuration file
            df['config_source'] = file_name
            config_sources = [file_name]
            if show_success:
                st.success(f"✅ Loaded {len(df)} data points from {file_name} successfully!")
        
        # Keep only necessary columns
        keep_cols = ['con', 'tst', 'obj', 'x', 'y', 'config_source']
        df = df[keep_cols]
        
        # Store data
        if update_state:
            st.session_state.data = df
            st.session_state.max_time = df['tst'].max()
            st.session_state.filename = file_name
            st.session_state.uploaded_filenames = [file_name]
            st.session_state.config_sources = config_sources
        
        return df
        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        import traceback
        st.error(f"Details: {traceback.format_exc()}")
        return None


# ============================================================================
# CLUSTERING INFRASTRUCTURE FUNCTIONS
# ============================================================================

def format_features_dataframe(features_df):
    """
    Format features dataframe with units in column names and 2 decimal places.
    
    Parameters:
    -----------
    features_df : pd.DataFrame
        Features dataframe with original column names
    
    Returns:
    --------
    pd.DataFrame : Formatted dataframe with units and rounded values
    """
    # Define column name mapping with units
    column_units = {
        'total_distance': 'Total Distance (m)',
        'duration': 'Duration (s)',
        'avg_speed': 'Avg Speed (m/s)',
        'net_displacement': 'Net Displacement (m)',
        'sinuosity': 'Sinuosity (ratio)',
        'bbox_area': 'Bbox Area (m²)',
        'avg_direction': 'Avg Direction (rad)',
        'max_speed': 'Max Speed (m/s)'
    }
    
    # Create a copy and rename columns
    formatted_df = features_df.copy()
    formatted_df = formatted_df.rename(columns=column_units)
    
    # Format all values to exactly 2 decimal places
    for col in formatted_df.columns:
        formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2f}")
    
    return formatted_df


def extract_trajectory_features(traj_df):
    """
    Extract 8 statistical features from a single trajectory.
    
    Parameters:
    -----------
    traj_df : pd.DataFrame
        Trajectory data with columns: X, Y, tst (or x, y, tst)
    
    Returns:
    --------
    dict : Dictionary containing 8 features
    """
    # Handle both uppercase and lowercase column names
    if 'X' in traj_df.columns:
        x_col, y_col = 'X', 'Y'
    else:
        x_col, y_col = 'x', 'y'
    
    if len(traj_df) < 2:
        return {
            'total_distance': 0.0,
            'duration': 0.0,
            'avg_speed': 0.0,
            'net_displacement': 0.0,
            'sinuosity': 1.0,
            'bbox_area': 0.0,
            'avg_direction': 0.0,
            'max_speed': 0.0
        }
    
    # Sort by time
    traj_df = traj_df.sort_values('tst').reset_index(drop=True)
    
    # Total distance
    distances = np.sqrt(
        np.diff(traj_df[x_col])**2 + 
        np.diff(traj_df[y_col])**2
    )
    total_distance = np.sum(distances)
    
    # Duration
    duration = traj_df['tst'].iloc[-1] - traj_df['tst'].iloc[0]
    if duration == 0:
        duration = 1.0  # Avoid division by zero
    
    # Average speed
    avg_speed = total_distance / duration if duration > 0 else 0.0
    
    # Net displacement (start to end)
    net_displacement = np.sqrt(
        (traj_df[x_col].iloc[-1] - traj_df[x_col].iloc[0])**2 +
        (traj_df[y_col].iloc[-1] - traj_df[y_col].iloc[0])**2
    )
    
    # Sinuosity (path efficiency)
    sinuosity = total_distance / net_displacement if net_displacement > 0 else 1.0
    
    # Bounding box area
    bbox_area = (traj_df[x_col].max() - traj_df[x_col].min()) * \
                (traj_df[y_col].max() - traj_df[y_col].min())
    
    # Average direction
    dx = np.diff(traj_df[x_col])
    dy = np.diff(traj_df[y_col])
    angles = np.arctan2(dy, dx)
    avg_direction = np.mean(angles) if len(angles) > 0 else 0.0
    
    # Maximum speed
    time_diffs = np.diff(traj_df['tst'])
    time_diffs[time_diffs == 0] = 0.01  # Avoid division by zero
    speeds = distances / time_diffs
    max_speed = np.max(speeds) if len(speeds) > 0 else 0.0
    
    return {
        'total_distance': float(total_distance),
        'duration': float(duration),
        'avg_speed': float(avg_speed),
        'net_displacement': float(net_displacement),
        'sinuosity': float(sinuosity),
        'bbox_area': float(bbox_area),
        'avg_direction': float(avg_direction),
        'max_speed': float(max_speed)
    }


@st.cache_data
def compute_feature_distance_matrix(df, selected_configs, selected_objects, start_time, end_time, selected_features=None):
    """
    Compute distance matrix based on trajectory features (Method 1).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full trajectory data
    selected_configs : list
        Selected configurations
    selected_objects : list
        Selected object IDs
    start_time : float
        Start time filter
    end_time : float
        End time filter
    selected_features : list, optional
        List of feature names to use. If None, use all features.
    
    Returns:
    --------
    tuple : (distance_matrix, trajectory_ids, features_df)
    """
    # Filter data
    filtered_df = df[
        (df['config_source'].isin(selected_configs)) &
        (df['obj'].isin(selected_objects)) &
        (df['tst'] >= start_time) &
        (df['tst'] <= end_time)
    ].copy()
    
    # Group by configuration and object to get individual trajectories
    trajectory_groups = filtered_df.groupby(['config_source', 'obj'])
    
    # Extract features for each trajectory
    features_list = []
    trajectory_ids = []
    trajectories = []
    
    for (config, obj), traj_df in trajectory_groups:
        features = extract_trajectory_features(traj_df)
        features_list.append(features)
        trajectory_ids.append(f"{config}_obj{obj}")
        # Store trajectory coordinates (handle both uppercase and lowercase column names)
        if 'X' in traj_df.columns:
            traj_coords = traj_df[['X', 'Y', 'tst']].values
        else:
            traj_coords = traj_df[['x', 'y', 'tst']].values
        trajectories.append(traj_coords)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features_list, index=trajectory_ids)
    
    # Select only specified features if provided
    if selected_features is not None and len(selected_features) > 0:
        features_df = features_df[selected_features]
    
    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features_df)
    
    # Compute Euclidean distance matrix
    distance_matrix = cdist(features_normalized, features_normalized, metric='euclidean')
    
    return distance_matrix, trajectory_ids, features_df, trajectories


@st.cache_data
def compute_chamfer_distance_matrix(df, selected_configs, selected_objects, start_time, end_time):
    """
    Compute distance matrix based on Chamfer distance (Method 2).
    
    Chamfer distance measures spatial similarity between trajectory paths.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full trajectory data
    selected_configs : list
        Selected configurations
    selected_objects : list
        Selected object IDs
    start_time : float
        Start time filter
    end_time : float
        End time filter
    
    Returns:
    --------
    tuple : (distance_matrix, trajectory_ids, trajectories)
    """
    # Filter data
    filtered_df = df[
        (df['config_source'].isin(selected_configs)) &
        (df['obj'].isin(selected_objects)) &
        (df['tst'] >= start_time) &
        (df['tst'] <= end_time)
    ].copy()
    
    # Group by configuration and object
    trajectory_groups = filtered_df.groupby(['config_source', 'obj'])
    
    # Extract spatial coordinates for each trajectory
    trajectories = []
    trajectory_ids = []
    
    for (config, obj), traj_df in trajectory_groups:
        traj_df = traj_df.sort_values('tst').reset_index(drop=True)
        coords = traj_df[['x', 'y']].values
        trajectories.append(coords)
        trajectory_ids.append(f"{config}_obj{obj}")
    
    n = len(trajectories)
    distance_matrix = np.zeros((n, n))
    
    # Compute pairwise Chamfer distances
    for i in range(n):
        for j in range(i+1, n):
            traj_A = trajectories[i]
            traj_B = trajectories[j]
            
            # Chamfer distance: average of minimum distances
            # Distance from A to B
            dist_A_to_B = np.mean([np.min(cdist([a], traj_B, metric='euclidean')) for a in traj_A])
            # Distance from B to A
            dist_B_to_A = np.mean([np.min(cdist([b], traj_A, metric='euclidean')) for b in traj_B])
            # Symmetric Chamfer distance
            chamfer_dist = (dist_A_to_B + dist_B_to_A) / 2.0
            
            distance_matrix[i, j] = chamfer_dist
            distance_matrix[j, i] = chamfer_dist
    
    return distance_matrix, trajectory_ids, trajectories


@st.cache_data
def compute_dtw_distance_matrix(df, selected_configs, selected_objects, start_time, end_time):
    """
    Compute distance matrix based on Dynamic Time Warping (Method 3).
    
    DTW measures spatiotemporal similarity considering both space and time.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full trajectory data
    selected_configs : list
        Selected configurations
    selected_objects : list
        Selected object IDs
    start_time : float
        Start time filter
    end_time : float
        End time filter
    
    Returns:
    --------
    tuple : (distance_matrix, trajectory_ids, trajectories)
    """
    # Filter data
    filtered_df = df[
        (df['config_source'].isin(selected_configs)) &
        (df['obj'].isin(selected_objects)) &
        (df['tst'] >= start_time) &
        (df['tst'] <= end_time)
    ].copy()
    
    # Group by configuration and object
    trajectory_groups = filtered_df.groupby(['config_source', 'obj'])
    
    # Extract spatiotemporal coordinates for each trajectory
    trajectories = []
    trajectory_ids = []
    
    for (config, obj), traj_df in trajectory_groups:
        traj_df = traj_df.sort_values('tst').reset_index(drop=True)
        # Include x, y, time as features
        coords = traj_df[['x', 'y', 'tst']].values
        trajectories.append(coords)
        trajectory_ids.append(f"{config}_obj{obj}")
    
    n = len(trajectories)
    distance_matrix = np.zeros((n, n))
    
    # Compute pairwise DTW distances
    for i in range(n):
        for j in range(i+1, n):
            dtw_dist = dtw_distance(trajectories[i], trajectories[j])
            distance_matrix[i, j] = dtw_dist
            distance_matrix[j, i] = dtw_dist
    
    return distance_matrix, trajectory_ids, trajectories


def dtw_distance(traj_A, traj_B):
    """
    Compute Dynamic Time Warping distance between two trajectories.
    
    Parameters:
    -----------
    traj_A : np.array
        Trajectory A with shape (n, 3) - columns: x, y, time
    traj_B : np.array
        Trajectory B with shape (m, 3) - columns: x, y, time
    
    Returns:
    --------
    float : DTW distance
    """
    n, m = len(traj_A), len(traj_B)
    
    # Initialize DTW matrix
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    # Fill DTW matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Euclidean distance between points (considering x, y, time)
            cost = euclidean(traj_A[i-1], traj_B[j-1])
            
            # Take minimum of three possible paths
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],      # insertion
                dtw_matrix[i, j-1],      # deletion
                dtw_matrix[i-1, j-1]     # match
            )
    
    return dtw_matrix[n, m]


def detect_optimal_clusters(distance_matrix, max_clusters=10, return_plot_data=False):
    """
    Auto-detect optimal number of clusters using elbow method with silhouette validation.
    
    Parameters:
    -----------
    distance_matrix : np.array
        Precomputed distance matrix
    max_clusters : int
        Maximum number of clusters to try
    return_plot_data : bool
        If True, also return data for plotting the elbow curve
    
    Returns:
    --------
    int : Optimal number of clusters
    dict : (optional) Plot data if return_plot_data=True
    """
    n_samples = len(distance_matrix)
    
    # Edge cases
    if n_samples < 3:
        return 2
    if n_samples < 10:
        return min(3, n_samples - 1)
    
    max_k = min(max_clusters, n_samples - 1)
    
    inertias = []
    silhouette_scores_list = []
    
    for k in range(2, max_k + 1):
        # Perform hierarchical clustering
        # Note: Ward linkage doesn't work with precomputed distances, so we use 'average' instead
        clustering = AgglomerativeClustering(
            n_clusters=k,
            metric='precomputed',
            linkage='average'
        )
        labels = clustering.fit_predict(distance_matrix)
        
        # Calculate within-cluster sum of squares (inertia)
        inertia = 0
        for cluster_id in range(k):
            cluster_mask = labels == cluster_id
            if np.sum(cluster_mask) > 0:
                cluster_distances = distance_matrix[cluster_mask][:, cluster_mask]
                inertia += np.sum(cluster_distances) / (2 * np.sum(cluster_mask))
        inertias.append(inertia)
        
        # Calculate silhouette score
        try:
            sil_score = silhouette_score(distance_matrix, labels, metric='precomputed')
            silhouette_scores_list.append(sil_score)
        except:
            silhouette_scores_list.append(0)
    
    # Find elbow point using angle method
    if len(inertias) < 2:
        return 3
    
    # Normalize inertias to 0-1 range
    inertias_norm = np.array(inertias)
    inertias_norm = (inertias_norm - inertias_norm.min()) / (inertias_norm.max() - inertias_norm.min() + 1e-10)
    
    # Calculate angles
    angles = []
    for i in range(1, len(inertias_norm) - 1):
        p1 = np.array([i-1, inertias_norm[i-1]])
        p2 = np.array([i, inertias_norm[i]])
        p3 = np.array([i+1, inertias_norm[i+1]])
        
        v1 = p1 - p2
        v2 = p3 - p2
        
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))
        angles.append(angle)
    
    if len(angles) > 0:
        elbow_idx = np.argmax(angles) + 1  # +1 because we start from k=2
        optimal_k = elbow_idx + 2  # +2 to convert back to actual k value
    else:
        optimal_k = 3
    
    # Validate with silhouette score
    if len(silhouette_scores_list) > 0:
        if silhouette_scores_list[optimal_k - 2] < 0.25:
            # If silhouette is poor, try to find better k
            best_sil_idx = np.argmax(silhouette_scores_list)
            if silhouette_scores_list[best_sil_idx] > 0.25:
                optimal_k = best_sil_idx + 2
    
    # Ensure reasonable range
    optimal_k = max(2, min(optimal_k, max_k))
    
    if return_plot_data:
        k_values = list(range(2, max_k + 1))
        return optimal_k, {
            'k_values': k_values,
            'inertias': inertias,
            'silhouette_scores': silhouette_scores_list,
            'optimal_k': optimal_k
        }
    
    return optimal_k


def perform_hierarchical_clustering(distance_matrix, n_clusters):
    """
    Perform hierarchical clustering with average linkage (compatible with precomputed distances).
    
    Parameters:
    -----------
    distance_matrix : np.array
        Precomputed distance matrix
    n_clusters : int
        Number of clusters to form
    
    Returns:
    --------
    tuple : (cluster_labels, linkage_matrix)
    """
    # Create linkage matrix for dendrogram - use 'average' instead of 'ward' for precomputed
    linkage_matrix = linkage(distance_matrix, method='average')
    
    # Perform clustering with average linkage (compatible with precomputed distances)
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average'  # Changed from 'ward' to 'average'
    )
    cluster_labels = clustering.fit_predict(distance_matrix)
    
    return cluster_labels, linkage_matrix


def initialize_clustering_session_state():
    """
    Initialize session state variables for clustering.
    Call this at the start of clustering section.
    """
    if 'clustering_method' not in st.session_state:
        st.session_state.clustering_method = None
    
    if 'clustering_distance_matrix' not in st.session_state:
        st.session_state.clustering_distance_matrix = None
    
    if 'clustering_linkage_matrix' not in st.session_state:
        st.session_state.clustering_linkage_matrix = None
    
    if 'clustering_trajectory_ids' not in st.session_state:
        st.session_state.clustering_trajectory_ids = None
    
    if 'clustering_optimal_n' not in st.session_state:
        st.session_state.clustering_optimal_n = None
    
    if 'clustering_current_n' not in st.session_state:
        st.session_state.clustering_current_n = None
    
    if 'clustering_labels' not in st.session_state:
        st.session_state.clustering_labels = None
    
    if 'clustering_features_df' not in st.session_state:
        st.session_state.clustering_features_df = None
    
    if 'clustering_trajectories' not in st.session_state:
        st.session_state.clustering_trajectories = None


# ============================================================================
# SEQUENCE ANALYSIS FUNCTIONS
# ============================================================================

def create_spatial_grid(court_type='Tennis', grid_rows=3, grid_cols=5):
    """
    Create a spatial grid for the court and return zone mapping function.
    Includes buffer zones around the court to capture out-of-bounds positions.
    
    Parameters:
    -----------
    court_type : str
        'Tennis' or 'Football'
    grid_rows : int
        Number of rows in grid (only for the court itself)
    grid_cols : int
        Number of columns in grid (only for the court itself)
    
    Returns:
    --------
    dict with zone_labels, x_bins, y_bins, and get_zone function
    """
    dims = get_court_dimensions(court_type)
    width, height = dims['width'], dims['height']
    
    # Add buffer zones around the court (typically 3-5 meters for tennis)
    # This captures positions when players are outside court boundaries
    buffer = 5.0  # meters
    
    # Extended grid with buffer zones
    x_min, x_max = -buffer, width + buffer
    y_min, y_max = -buffer, height + buffer
    
    # Create bins - main court gets the specified grid, plus 1 column/row on each side for buffer
    actual_cols = grid_cols + 2  # +1 left buffer, +1 right buffer
    actual_rows = grid_rows + 2  # +1 top buffer, +1 bottom buffer
    
    x_bins = np.linspace(x_min, x_max, actual_cols + 1)
    y_bins = np.linspace(y_min, y_max, actual_rows + 1)
    
    # Generate zone labels using row-column notation (A1, A2, B1, B2, etc.)
    # Column letters: A, B, C, ..., Z, AA, AB, AC, ...
    # Row numbers: 1, 2, 3, ...
    def get_column_label(col_idx):
        """Generate column label (A, B, C, ..., Z, AA, AB, ...)"""
        if col_idx < 26:
            return chr(65 + col_idx)  # A-Z
        else:
            # For more than 26 columns: AA, AB, AC, ...
            return chr(65 + (col_idx // 26) - 1) + chr(65 + (col_idx % 26))
    
    zone_labels = []
    for row in range(actual_rows):
        for col in range(actual_cols):
            col_label = get_column_label(col)
            row_label = str(row + 1)  # 1-based row numbering
            zone_labels.append(f"{col_label}{row_label}")
    
    def get_zone(x, y):
        """Map (x, y) coordinate to zone label (e.g., A1, B2, C3)."""
        if pd.isna(x) or pd.isna(y):
            return None
        col_idx = np.digitize(x, x_bins) - 1
        row_idx = np.digitize(y, y_bins) - 1
        # Clamp to valid range (handle extreme outliers)
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


def build_event_based_sequence(df, config, obj_id, start_time, end_time, grid_info, compress=True):
    """
    Build event-based sequence (one token per hit/bounce).
    For tennis: use all data points as events.
    
    Parameters:
    -----------
    df : DataFrame
    config : str
        Configuration source
    obj_id : int
        Object ID
    start_time, end_time : float
        Time window
    grid_info : dict
        From create_spatial_grid()
    compress : bool
        If True, compress runs (AAABBB -> AB)
    
    Returns:
    --------
    str : sequence of zone tokens (e.g., "AABBBCCC" or "ABC" if compressed)
    """
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
        # Run-length compression: AAABBB -> AB
        tokens = [k for k, _ in groupby(tokens)]
    
    # Return as list of tokens (not concatenated string)
    return tokens


def build_interval_based_sequence(df, config, obj_id, start_time, end_time, 
                                  grid_info, delta_t=0.2, compress=True):
    """
    Build equal-interval sequence (fixed time steps).
    
    Parameters:
    -----------
    delta_t : float
        Time interval in seconds
    
    Returns:
    --------
    str : sequence of zone tokens
    """
    obj_data = df[(df['config_source'] == config) &
                  (df['obj'] == obj_id) &
                  (df['tst'] >= start_time) &
                  (df['tst'] <= end_time)].sort_values('tst')
    
    if len(obj_data) == 0:
        return ""
    
    get_zone = grid_info['get_zone']
    
    # Sample at fixed intervals
    time_points = np.arange(start_time, end_time + delta_t, delta_t)
    tokens = []
    
    for t in time_points:
        # Find closest data point to this time
        closest_idx = (obj_data['tst'] - t).abs().idxmin()
        row = obj_data.loc[closest_idx]
        zone = get_zone(row['x'], row['y'])
        if zone is not None:
            tokens.append(zone)
    
    if compress:
        tokens = [k for k, _ in groupby(tokens)]
    
    # Return as list of tokens (not concatenated string)
    return tokens


def build_multi_entity_sequence(df, config, entity_ids, start_time, end_time,
                                grid_info, mode='event', delta_t=0.2, compress=True):
    """
    Build joint sequence combining multiple entities (ball, p1, p2).
    
    Returns:
    --------
    list : joint sequence tokens like ["B:A|P1:C|P2:F", "B:B|P1:C|P2:E", ...]
    """
    # Build individual sequences (now returns lists)
    sequences = {}
    for entity_id in entity_ids:
        if mode == 'event':
            seq = build_event_based_sequence(df, config, entity_id, start_time, 
                                            end_time, grid_info, compress=False)
        else:
            seq = build_interval_based_sequence(df, config, entity_id, start_time, 
                                               end_time, grid_info, delta_t, compress=False)
        sequences[entity_id] = seq
    
    # Find max length
    max_len = max(len(s) for s in sequences.values()) if sequences else 0
    
    # Pad sequences to same length
    for eid in sequences:
        while len(sequences[eid]) < max_len:
            # Append last zone or 'X' if empty
            sequences[eid].append(sequences[eid][-1] if sequences[eid] else 'X')
    
    # Combine into joint tokens
    joint_tokens = []
    for i in range(max_len):
        token_parts = [f"{eid}:{sequences[eid][i]}" for eid in entity_ids]
        joint_tokens.append('|'.join(token_parts))
    
    if compress:
        joint_tokens = [k for k, _ in groupby(joint_tokens)]
    
    # Return as list of joint tokens
    return joint_tokens



def levenshtein_distance(seq1, seq2):
    """
    Compute Levenshtein (edit) distance between two sequences.
    
    Parameters:
    -----------
    seq1, seq2 : list
        Sequences (lists of tokens) to compare
    
    Returns:
    --------
    int : edit distance
    """
    len1, len2 = len(seq1), len(seq2)
    
    # Create DP table
    dp = np.zeros((len1 + 1, len2 + 1), dtype=int)
    
    # Initialize
    for i in range(len1 + 1):
        dp[i, 0] = i
    for j in range(len2 + 1):
        dp[0, j] = j
    
    # Fill table
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if seq1[i-1] == seq2[j-1]:
                cost = 0
            else:
                cost = 1
            
            dp[i, j] = min(
                dp[i-1, j] + 1,      # deletion
                dp[i, j-1] + 1,      # insertion
                dp[i-1, j-1] + cost  # substitution
            )
    
    return int(dp[len1, len2])


def needleman_wunsch(seq1, seq2, match=2, mismatch=-1, gap=-1):
    """
    Global alignment using Needleman-Wunsch algorithm.
    
    Parameters:
    -----------
    seq1, seq2 : list
        Sequences (lists of tokens) to align
    match : int
        Score for matching tokens
    mismatch : int
        Penalty for mismatch
    gap : int
        Penalty for gap (indel)
    
    Returns:
    --------
    dict with 'score', 'aligned_seq1', 'aligned_seq2' (both are lists)
    """
    len1, len2 = len(seq1), len(seq2)
    
    # Score matrix
    score_matrix = np.zeros((len1 + 1, len2 + 1))
    
    # Initialize
    for i in range(len1 + 1):
        score_matrix[i, 0] = gap * i
    for j in range(len2 + 1):
        score_matrix[0, j] = gap * j
    
    # Fill matrix
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if seq1[i-1] == seq2[j-1]:
                diagonal = score_matrix[i-1, j-1] + match
            else:
                diagonal = score_matrix[i-1, j-1] + mismatch
            
            score_matrix[i, j] = max(
                diagonal,
                score_matrix[i-1, j] + gap,  # deletion
                score_matrix[i, j-1] + gap   # insertion
            )
    
    # Traceback
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
    """
    Local alignment using Smith-Waterman algorithm.
    
    Parameters:
    -----------
    seq1, seq2 : list
        Sequences (lists of tokens) to align
    
    Returns:
    --------
    dict with 'score', 'aligned_seq1', 'aligned_seq2' (lists), 'start1', 'start2'
    """
    len1, len2 = len(seq1), len(seq2)
    
    # Score matrix
    score_matrix = np.zeros((len1 + 1, len2 + 1))
    
    # Fill matrix (no negative scores)
    max_score = 0
    max_pos = (0, 0)
    
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if seq1[i-1] == seq2[j-1]:
                diagonal = score_matrix[i-1, j-1] + match
            else:
                diagonal = score_matrix[i-1, j-1] + mismatch
            
            score_matrix[i, j] = max(
                0,  # Can reset to 0
                diagonal,
                score_matrix[i-1, j] + gap,
                score_matrix[i, j-1] + gap
            )
            
            if score_matrix[i, j] > max_score:
                max_score = score_matrix[i, j]
                max_pos = (i, j)
    
    # Traceback from max position
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


def extract_ngrams(sequence, n=2):
    """
    Extract n-grams from sequence.
    
    Parameters:
    -----------
    sequence : list
        Token sequence (list of zone labels)
    n : int
        N-gram size
    
    Returns:
    --------
    Counter : n-gram frequencies (n-grams are tuples of tokens)
    """
    if len(sequence) < n:
        return Counter()
    
    # Extract n-grams as tuples of tokens
    ngrams = [tuple(sequence[i:i+n]) for i in range(len(sequence) - n + 1)]
    return Counter(ngrams)


def compute_sequence_distance_matrix(sequences, method='levenshtein'):
    """
    Compute pairwise distance matrix for sequences.
    
    Parameters:
    -----------
    sequences : list of str
        List of token sequences
    method : str
        'levenshtein' or 'normalized_levenshtein'
    
    Returns:
    --------
    np.array : distance matrix
    """
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


# ============================================================================
# END SEQUENCE ANALYSIS FUNCTIONS
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
    fig = go.Figure()
    
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
    st.title("📊 Spatiotemporal Analysis and Modeling")
    
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
            st.info(f"📊 Current file(s): {st.session_state.filename}")
            
            st.header("Court Type")
            court_type = st.radio(
                "Select court type",
                ["Football", "Tennis"],
                index=0 if st.session_state.court_type == 'Football' else 1
            )
            st.session_state.court_type = court_type
            
            # --- CENTRALIZED SELECTION PANEL ---
            st.header("🎯 Data Selection")
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
            with st.expander("📋 Current Selection Summary", expanded=False):
                st.write(f"**Configurations:** {len(selected_configs)} of {len(config_sources)} selected")
                if selected_configs:
                    st.write(", ".join(selected_configs))
                st.write(f"**Objects:** {len(selected_objects)} of {len(objects)} selected")
                if selected_objects:
                    st.write(", ".join(map(str, selected_objects)))
            
            st.divider()
            # --- END CENTRALIZED SELECTION PANEL ---
            
            st.header("Analysis Method")
            analysis_method = st.selectbox(
                "Select method",
                ["Visual Exploration", "Clustering", "Sequence Analysis", "Heat Maps", "Extra"]
            )
    
    # Main content
    if df is None:
        st.info("👆 Please upload a CSV file to begin.")
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
        st.header("👁️ Visual Exploration")
        
        st.info("""
        **Explore your trajectory data visually with interactive plots:**
        - **Static Trajectories:** View complete trajectory paths
        - **Animated Trajectories:** Watch movement over time
        - **Time Point View:** Examine trajectories at specific moments
        - **Average Positions:** Calculate and visualize mean positions
        
        **💡 Tip:** Use the sidebar to select which configurations and objects to analyze.
        """)
        
        # Use selections from sidebar
        selected_configs = st.session_state.shared_selected_configs
        selected_objects = st.session_state.shared_selected_objects
        
        # Time range
        min_time = df['tst'].min()
        max_time = df['tst'].max()
        
        st.markdown("---")
        st.subheader("📊 Time Range Settings")
        
        col3, col4 = st.columns(2)
        
        with col3:
            start_time = st.number_input(
                "Start time",
                min_value=float(min_time),
                max_value=float(max_time),
                value=float(min_time),
                key="visual_start"
            )
        
        with col4:
            end_time = st.number_input(
                "End time",
                min_value=float(min_time),
                max_value=float(max_time),
                value=float(max_time),
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
                key="visual_temp_res"
            )
        
        if not selected_configs or not selected_objects:
            st.warning("⚠️ Please select at least one configuration and one object.")
        else:
            st.markdown("---")
            st.subheader("📈 Visualization Types")
            
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
                
                try:
                    fig = visualize_animated(
                        df, selected_configs, selected_objects,
                        start_time, end_time,
                        aggregation_type, temporal_resolution,
                        court_type
                    )
                    render_interactive_chart(fig)
                except Exception as e:
                    st.error(f"Error creating animated visualization: {str(e)}")
            
            with viz_tabs[2]:
                st.markdown("### Time Point View")
                st.info("Examine trajectories up to a specific point in time.")
                
                current_time = st.slider(
                    "Select time point",
                    min_value=float(start_time),
                    max_value=float(end_time),
                    value=float((start_time + end_time) / 2),
                    key="visual_current_time"
                )
                
                try:
                    fig = visualize_at_time(
                        df, selected_configs, selected_objects,
                        current_time, start_time,
                        aggregation_type, temporal_resolution,
                        court_type
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
        st.header("📐 2SA Method - Two-Step Spatial Alignment")
        
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
        st.subheader("📊 Settings")
        
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
                key="2sa_start"
            )
        
        with col4:
            end_time = st.number_input(
                "End time",
                min_value=float(min_time),
                max_value=float(max_time),
                value=float(max_time),
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
                key="2sa_temp_res"
            )
        
        if not selected_configs or not selected_objects:
            st.warning("⚠️ Please select at least one configuration and one object.")
        else:
            st.markdown("---")
            st.subheader("📈 Aligned Trajectories")
            
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
            st.success("✅ 2SA analysis complete! Use the tabs above to compare aligned and original trajectories.")
    
    elif analysis_method == "Sequence Analysis":
        st.header("🔤 Sequence Analysis")
        
        st.info("""
        **Translate trajectories to symbolic sequences for pattern mining and comparison:**
        - **Spatial Discretization:** Divide court into zones (A, B, C, ...)
        - **Temporal Sampling:** Event-based (per hit/bounce) or equal-interval
        - **Sequence Comparison:** Edit distances and alignment (global/local)
        - **Pattern Discovery:** Find common sub-patterns across rallies
        """)
        
        # Show preview of grid concept
        with st.expander("ℹ️ How does spatial discretization work?", expanded=False):
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
        st.subheader("⚙️ Sequence Configuration")
        
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
        st.subheader("🌐 Spatial Grid Visualization")
        
        st.info("""
        **Understanding the Grid:**
        - The **tennis field in the broad sense** (light gray zones) extends beyond the court boundaries to capture out-of-bounds positions
        - All zones are labeled **A, B, C,** etc. (row by row from bottom to top, filling each row left to right)
        - Each trajectory position is mapped to the zone it falls in
        - Adjust the grid resolution sliders above to see how it affects the zone layout
        """)
        
        # Create grid info for visualization
        grid_info = create_spatial_grid(
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
                
                # Add zone label with high-contrast background
                # Use different styling for buffer zones
                if is_buffer:
                    bgcolor = 'rgba(150, 150, 150, 0.7)'
                    bordercolor = 'gray'
                    font_size = 16
                else:
                    bgcolor = 'rgba(0, 0, 0, 0.7)'
                    bordercolor = 'red'
                    font_size = 20
                
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
            st.write("**Temporal Resolution**")
            sampling_mode = st.radio(
                "Sampling mode",
                ["Event-based", "Equal-interval"],
                help="Event-based: one token per data point. Equal-interval: fixed time steps.",
                key="seq_sampling"
            )
            
            if sampling_mode == "Equal-interval":
                delta_t = st.slider("Time interval (Δt)", 0.1, 2.0, 0.2, 0.1, key="seq_delta_t")
                st.caption(f"Sampling every {delta_t}s")
            else:
                delta_t = 0.2  # Not used for event-based
                st.caption("One token per data point")
        
        with col3:
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
                help="Per-entity: separate sequences for each object. Multi-entity: combined token per moment.",
                key="seq_type"
            )
        
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
                key="seq_start"
            )
        
        with col4:
            end_time = st.number_input(
                "End time",
                min_value=float(min_time),
                max_value=float(max_time),
                value=float(max_time),
                key="seq_end"
            )
        
        if not selected_configs:
            st.warning("⚠️ Please select at least one configuration.")
        elif not selected_objects:
            st.warning("⚠️ Please select at least one object.")
        else:
            # Create grid
            grid_info = create_spatial_grid(
                court_type=st.session_state.court_type,
                grid_rows=grid_rows,
                grid_cols=grid_cols
            )
            
            # Build sequences
            st.markdown("---")
            st.subheader("🔤 Generated Sequences")
            
            sequences_data = []
            mode = 'event' if sampling_mode == "Event-based" else 'interval'
            
            # Store both the list (for processing) and string (for display)
            raw_sequences = []  # Store list form
            
            if sequence_type == "Per-entity":
                # Build per-entity sequences
                for config in selected_configs:
                    for obj_id in selected_objects:
                        if mode == 'event':
                            seq = build_event_based_sequence(
                                df, config, obj_id, start_time, end_time,
                                grid_info, compress=compress_runs
                            )
                        else:
                            seq = build_interval_based_sequence(
                                df, config, obj_id, start_time, end_time,
                                grid_info, delta_t=delta_t, compress=compress_runs
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
                # Multi-entity sequences
                for config in selected_configs:
                    seq = build_multi_entity_sequence(
                        df, config, selected_objects, start_time, end_time,
                        grid_info, mode=mode, delta_t=delta_t, compress=compress_runs
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
                st.warning("⚠️ No sequences generated. Check your data and time range.")
            else:
                # Display sequences
                seq_df = pd.DataFrame(sequences_data)
                st.dataframe(seq_df, use_container_width=True, height=300)
                
                # Export sequences
                csv_export = seq_df.to_csv(index=False)
                st.download_button(
                    "📥 Download sequences as CSV",
                    csv_export,
                    f"sequences_{sampling_mode}_{grid_rows}x{grid_cols}.csv",
                    "text/csv"
                )
                
                # Create tabs for analysis
                st.markdown("---")
                seq_tab1, seq_tab2, seq_tab3 = st.tabs([
                    "📏 Distance Matrix",
                    "🔄 Pairwise Alignment",
                    "📊 N-gram Patterns"
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
                    dist_matrix = compute_sequence_distance_matrix(raw_sequences, method=method)
                    
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
                        st.subheader("🌳 Hierarchical Clustering - Dendrogram & Cluster Assignment")
                        
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
                        linkage_matrix = linkage(condensed_dist, method='average')
                        
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
                            title="Hierarchical Clustering Dendrogram (Average Linkage)",
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
                            
                            n_clusters = st.slider(
                                "Number of clusters",
                                min_value=2,
                                max_value=max_clusters,
                                value=min(3, max_clusters),
                                help="Slide to select how many clusters to create",
                                key="seq_clusters"
                            )
                        
                        with col2:
                            # Auto-detect optimal clusters button
                            if st.button("🎯 Auto-detect Optimal Clusters", help="Use elbow method to recommend optimal number of clusters.", key="seq_auto_clusters"):
                                with st.spinner("Detecting optimal number of clusters..."):
                                    optimal_k, plot_data = detect_optimal_clusters(dist_matrix, return_plot_data=True)
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
                        st.success(f"✅ Successfully assigned {n_sequences} sequences into {n_clusters} clusters using Average linkage!")
                        
                        # ===========================
                        # ANALYSIS TOOLS
                        # ===========================
                        
                        st.markdown('---')
                        st.markdown("### 🔬 Analysis Tools")
                        
                        st.info("""
                        **Advanced Analysis**: Explore cluster quality and sequence relationships
                        - **MDS Visualization**: Project high-dimensional data to 2D/3D space
                        - **Similarity Search**: Find most similar sequences to a reference
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
                                        st.success(f"✅ {mds_dims}D MDS projection computed successfully!")
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
                            
                            if st.button("🔍 Find Similar Sequences", key="seq_find_similar"):
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
                                        st.success("✅ All similar sequences are in the same cluster - excellent clustering!")
                                    elif same_cluster >= k_similar * 0.7:
                                        st.info("ℹ️ Most similar sequences are in the same cluster - good clustering quality")
                                    else:
                                        st.warning("⚠️ Many similar sequences are in different clusters - consider adjusting cluster count")
                        
                        # ===========================
                        # TAB 3: SILHOUETTE ANALYSIS
                        # ===========================
                        with analysis_tab3:
                            st.markdown("#### Silhouette Analysis")
                            st.markdown("Evaluate cluster quality using silhouette coefficients. Values range from -1 to 1:")
                            st.markdown("- **Close to 1**: Well-clustered, far from neighboring clusters")
                            st.markdown("- **Close to 0**: Near the decision boundary between clusters")
                            st.markdown("- **Negative**: Possibly assigned to wrong cluster")
                            
                            if st.button("📊 Calculate Silhouette Scores", key="seq_silhouette"):
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
                                    st.success("✅ Silhouette analysis complete!")
                
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
                            ["Global (Needleman-Wunsch)", "Local (Smith-Waterman)"],
                            help="Global: align entire sequences. Local: find best matching sub-sequences.",
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
                            result = needleman_wunsch(seq1, seq2, match_score, mismatch_penalty, gap_penalty)
                            align_method = "Global"
                        else:
                            result = smith_waterman(seq1, seq2, match_score, mismatch_penalty, gap_penalty)
                            align_method = "Local"
                        
                        # Display results
                        st.markdown("---")
                        st.write(f"**{align_method} Alignment Results**")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Alignment Score", f"{result['score']:.1f}")
                        with col2:
                            if 'start1' in result:
                                st.metric("Start positions", f"Seq1:{result['start1']}, Seq2:{result['start2']}")
                            else:
                                matches = sum(1 for a, b in zip(result['aligned_seq1'], result['aligned_seq2']) if a == b and a != '-')
                                st.metric("Matches", matches)
                        
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
                            ngrams = extract_ngrams(seq, n_gram_size)
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
                                seq_ngrams = extract_ngrams(seq, n_gram_size)
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
        initialize_clustering_session_state()
        
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
                        distance_matrix, trajectory_ids, features_df, trajectories = compute_feature_distance_matrix(
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
                    formatted_df = format_features_dataframe(st.session_state.features_df)
                    st.dataframe(formatted_df)
        
        elif clustering_method == "Spatial (Chamfer)":
            st.subheader("📍 Spatial Clustering (Chamfer Distance)")
            st.info("Cluster trajectories based on spatial shape and location similarity using Chamfer distance.")
            
            # Compute distance matrix button
            if st.button("🔄 Compute Chamfer Distance Matrix", key="compute_chamfer"):
                with st.spinner("Computing Chamfer distances..."):
                    try:
                        distance_matrix, trajectory_ids, trajectories = compute_chamfer_distance_matrix(
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
                        distance_matrix, trajectory_ids, trajectories = compute_dtw_distance_matrix(
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
            # Note: Ward linkage requires raw data, not distance matrix
            # Using 'average' linkage which works with precomputed distance matrices
            # Convert square distance matrix to condensed form
            from scipy.spatial.distance import squareform
            condensed_dist = squareform(st.session_state.distance_matrix, checks=False)
            linkage_matrix = linkage(condensed_dist, method='average')
            
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
                    if st.button("� Auto-detect Optimal Clusters", help="Use elbow method to recommend optimal number of clusters."):
                        with st.spinner("Detecting optimal number of clusters..."):
                            optimal_k, plot_data = detect_optimal_clusters(st.session_state.distance_matrix, return_plot_data=True)
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
