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
    
    # Round all values to 2 decimal places
    formatted_df = formatted_df.round(2)
    
    return formatted_df


def extract_trajectory_features(traj_df):
    """
    Extract 8 statistical features from a single trajectory.
    
    Parameters:
    -----------
    traj_df : pd.DataFrame
        Trajectory data with columns: x, y, tst
    
    Returns:
    --------
    dict : Dictionary containing 8 features
    """
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
        np.diff(traj_df['x'])**2 + 
        np.diff(traj_df['y'])**2
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
        (traj_df['x'].iloc[-1] - traj_df['x'].iloc[0])**2 +
        (traj_df['y'].iloc[-1] - traj_df['y'].iloc[0])**2
    )
    
    # Sinuosity (path efficiency)
    sinuosity = total_distance / net_displacement if net_displacement > 0 else 1.0
    
    # Bounding box area
    bbox_area = (traj_df['x'].max() - traj_df['x'].min()) * \
                (traj_df['y'].max() - traj_df['y'].min())
    
    # Average direction
    dx = np.diff(traj_df['x'])
    dy = np.diff(traj_df['y'])
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
    
    for (config, obj), traj_df in trajectory_groups:
        features = extract_trajectory_features(traj_df)
        features_list.append(features)
        trajectory_ids.append(f"{config}_obj{obj}")
    
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
    
    return distance_matrix, trajectory_ids, features_df


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


def detect_optimal_clusters(distance_matrix, max_clusters=10):
    """
    Auto-detect optimal number of clusters using elbow method with silhouette validation.
    
    Parameters:
    -----------
    distance_matrix : np.array
        Precomputed distance matrix
    max_clusters : int
        Maximum number of clusters to try
    
    Returns:
    --------
    int : Optimal number of clusters
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
        clustering = AgglomerativeClustering(
            n_clusters=k,
            metric='precomputed',
            linkage='ward'
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
    
    return optimal_k


def perform_hierarchical_clustering(distance_matrix, n_clusters):
    """
    Perform hierarchical clustering with Ward linkage.
    
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
    # Create linkage matrix for dendrogram
    linkage_matrix = linkage(distance_matrix, method='ward')
    
    # Perform clustering
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='ward'
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
                ["Visual Exploration (IMO)", "2SA Method", "Heat Maps", "Clustering", "Extra"]
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
    if analysis_method == "Visual Exploration (IMO)":
        st.header("👁️ Visual Exploration")
        
        st.info("""
        **Explore your trajectory data visually with interactive plots:**
        - **Static Trajectories:** View complete trajectory paths
        - **Animated Trajectories:** Watch movement over time
        - **Time Point View:** Examine trajectories at specific moments
        - **Average Positions:** Calculate and visualize mean positions
        """)
        
        # Get available configurations and objects
        config_sources = df['config_source'].drop_duplicates().tolist()
        objects = sorted(df['obj'].unique())
        
        # Time range
        min_time = df['tst'].min()
        max_time = df['tst'].max()
        
        st.markdown("---")
        st.subheader("📊 Visualization Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_configs = st.multiselect(
                "Select configuration(s)",
                config_sources,
                default=config_sources[:min(3, len(config_sources))],
                key="visual_configs"
            )
        
        with col2:
            selected_objects = st.multiselect(
                "Select object(s)",
                objects,
                default=objects[:min(5, len(objects))],
                key="visual_objects"
            )
        
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
                
                translate_to_center = st.checkbox(
                    "Translate to center",
                    value=False,
                    key="visual_translate"
                )
                
                try:
                    fig = visualize_static(
                        df, selected_configs, selected_objects,
                        start_time, end_time,
                        aggregation_type, temporal_resolution,
                        translate_to_center, court_type
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
                default=config_sources[:min(3, len(config_sources))],
                key="2sa_configs"
            )
        
        with col2:
            selected_objects = st.multiselect(
                "Select object(s)",
                objects,
                default=objects[:min(5, len(objects))],
                key="2sa_objects"
            )
        
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
        
        st.subheader("📊 Data Selection")

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
            
            col1, col2 = st.columns([3, 1])
            with col2:
                st.markdown("")
                st.markdown("")
                if st.button("Select All", key="select_all_features"):
                    st.session_state.feature_selection_default = all_features
                    st.rerun()
                if st.button("Clear All", key="clear_all_features"):
                    st.session_state.feature_selection_default = []
                    st.rerun()
            
            with col1:
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
                        distance_matrix, trajectory_ids, features_df = compute_feature_distance_matrix(
                            df, selected_configs, selected_objects, start_time, end_time, selected_features
                        )
                        
                        if distance_matrix is None:
                            st.error("❌ No valid trajectories found with the current filters.")
                        else:
                            st.session_state.distance_matrix = distance_matrix
                            st.session_state.trajectory_ids = trajectory_ids
                            st.session_state.features_df = features_df
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
            
            # Create linkage matrix for hierarchical clustering (Ward linkage)
            linkage_matrix = linkage(st.session_state.distance_matrix, method='ward')
            
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
                    if st.button("� Auto-detect Optimal Clusters", help="Use elbow method to find optimal number of clusters"):
                        with st.spinner("Detecting optimal number of clusters..."):
                            optimal_k = detect_optimal_clusters(st.session_state.distance_matrix)
                            if optimal_k is not None:
                                st.success(f"✅ Optimal number of clusters detected: **{optimal_k}**")
                                # Update the slider value in session state
                                st.session_state.n_clusters_slider = optimal_k
                                st.rerun()
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
                            st.info(f"**Stress value**: {mds.stress_:.4f} (lower is better, <0.1 is excellent)")
            
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
                elif st.button("🎨 Generate 2D Cluster Plot", key="btn_2d_cluster"):
                    with st.spinner("Generating 2D visualization..."):
                        import plotly.express as px
                        
                        # Create trajectory dictionary mapping
                        trajectories_dict = {tid: traj for tid, traj in zip(st.session_state.trajectory_ids, st.session_state.trajectories)}
                        
                        # Create color palette
                        colors = px.colors.qualitative.Plotly[:n_clusters]
                        
                        fig_2d = go.Figure()
                        
                        # Plot each cluster
                        for cluster_id in range(1, n_clusters + 1):
                            mask = cluster_labels == cluster_id
                            cluster_trajectory_ids = np.array(st.session_state.trajectory_ids)[mask]
                            
                            for tid in cluster_trajectory_ids:
                                # Get trajectory data
                                traj_data = trajectories_dict[tid]
                                
                                fig_2d.add_trace(go.Scatter(
                                    x=traj_data[:, 0],  # X coordinates
                                    y=traj_data[:, 1],  # Y coordinates
                                    mode='lines+markers',
                                    name=f"T{tid} (C{cluster_id})",
                                    line=dict(color=colors[cluster_id - 1], width=2),
                                    marker=dict(size=4, color=colors[cluster_id - 1]),
                                    legendgroup=f"cluster_{cluster_id}",
                                    hovertemplate=f'<b>Trajectory {tid}</b><br>Cluster: {cluster_id}<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>'
                                ))
                        
                        fig_2d.update_layout(
                            title=f"2D Trajectory Clusters (n={n_clusters})",
                            xaxis_title="X Coordinate",
                            yaxis_title="Y Coordinate",
                            height=700,
                            hovermode='closest',
                            showlegend=True,
                            legend=dict(
                                title="Trajectories",
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=1.01
                            )
                        )
                        
                        st.plotly_chart(fig_2d, use_container_width=True)
                        
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
                else:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        show_projections = st.checkbox(
                            "Show 2D projections",
                            value=False,
                            help="Display projections on XY, XZ, and YZ planes"
                        )
                    
                    with col2:
                        if st.button("🌐 Generate 3D Plot", key="btn_3d_cluster"):
                            with st.spinner("Generating 3D spatiotemporal visualization..."):
                                import plotly.express as px
                                
                                # Create trajectory dictionary mapping
                                trajectories_dict = {tid: traj for tid, traj in zip(st.session_state.trajectory_ids, st.session_state.trajectories)}
                                
                                colors = px.colors.qualitative.Plotly[:n_clusters]
                                
                                fig_3d = go.Figure()
                                
                                # Plot each cluster in 3D
                                for cluster_id in range(1, n_clusters + 1):
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
                                        line=dict(color=colors[cluster_id - 1], width=3),
                                        marker=dict(size=3, color=colors[cluster_id - 1]),
                                        legendgroup=f"cluster_{cluster_id}",
                                        hovertemplate=f'<b>Trajectory {tid}</b><br>Cluster: {cluster_id}<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<br>Time: %{{z}}<extra></extra>'
                                    ))
                            
                            fig_3d.update_layout(
                                title=f"3D Spatiotemporal Trajectory Clusters (n={n_clusters})",
                                scene=dict(
                                    xaxis_title="X Coordinate",
                                    yaxis_title="Y Coordinate",
                                    zaxis_title="Time Step",
                                    camera=dict(
                                        eye=dict(x=1.5, y=1.5, z=1.3)
                                    )
                                ),
                                height=800,
                                hovermode='closest',
                                showlegend=True,
                                legend=dict(
                                    title="Trajectories",
                                    yanchor="top",
                                    y=0.99,
                                    xanchor="left",
                                    x=0.01
                                )
                            )
                            
                            st.plotly_chart(fig_3d, use_container_width=True)
                            st.success("✅ 3D visualization generated! Rotate and zoom to explore the spatiotemporal patterns.")
            
            # ===========================
            # TAB 3: CLUSTER COMPARISON
            # ===========================
            with viz_tab3:
                st.markdown("#### Individual Cluster Analysis")
                st.markdown("View and compare individual clusters in detail.")
                
                # Check if trajectory data is available
                if 'trajectories' not in st.session_state or st.session_state.trajectories is None:
                    st.warning("⚠️ No trajectory data available. Please compute the distance matrix first in Step 3.")
                else:
                    # Cluster selection
                    selected_clusters = st.multiselect(
                        "Select clusters to visualize",
                        options=list(range(1, n_clusters + 1)),
                        default=[1] if n_clusters >= 1 else [],
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
                                
                                colors = px.colors.qualitative.Plotly[:n_clusters]
                                
                                if view_mode == "Overlay":
                                    # Single plot with selected clusters
                                    fig_compare = go.Figure()
                                    
                                    for cluster_id in selected_clusters:
                                        mask = cluster_labels == cluster_id
                                        cluster_trajectory_ids = np.array(st.session_state.trajectory_ids)[mask]
                                        
                                        for tid in cluster_trajectory_ids:
                                            traj_data = trajectories_dict[tid]
                                            
                                            fig_compare.add_trace(go.Scatter(
                                                x=traj_data[:, 0],
                                                y=traj_data[:, 1],
                                                mode='lines+markers',
                                                name=f"T{tid} (C{cluster_id})",
                                                line=dict(color=colors[cluster_id - 1], width=2),
                                                marker=dict(size=4, color=colors[cluster_id - 1]),
                                                legendgroup=f"cluster_{cluster_id}",
                                                hovertemplate=f'<b>Trajectory {tid}</b><br>Cluster: {cluster_id}<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>'
                                            ))
                                    
                                    fig_compare.update_layout(
                                        title=f"Cluster Comparison - Overlay View (Clusters: {selected_clusters})",
                                        xaxis_title="X Coordinate",
                                        yaxis_title="Y Coordinate",
                                        height=600,
                                        hovermode='closest',
                                        showlegend=True
                                    )
                                    
                                    st.plotly_chart(fig_compare, use_container_width=True)
                                    
                                else:  # Side-by-side
                                    # Create subplots
                                    from plotly.subplots import make_subplots
                                    
                                    n_selected = len(selected_clusters)
                                    cols = min(2, n_selected)
                                    rows = (n_selected + cols - 1) // cols
                                    
                                    fig_compare = make_subplots(
                                        rows=rows,
                                        cols=cols,
                                        subplot_titles=[f"Cluster {c} ({(cluster_labels == c).sum()} trajectories)" 
                                                       for c in selected_clusters],
                                        vertical_spacing=0.15,
                                        horizontal_spacing=0.1
                                    )
                                    
                                    for idx, cluster_id in enumerate(selected_clusters):
                                        row = idx // cols + 1
                                        col = idx % cols + 1
                                        
                                        mask = cluster_labels == cluster_id
                                        cluster_trajectory_ids = np.array(st.session_state.trajectory_ids)[mask]
                                        
                                        for tid in cluster_trajectory_ids:
                                            traj_data = trajectories_dict[tid]
                                            
                                            fig_compare.add_trace(
                                                go.Scatter(
                                                    x=traj_data[:, 0],
                                                    y=traj_data[:, 1],
                                                    mode='lines+markers',
                                                    name=f"T{tid}",
                                                    line=dict(color=colors[cluster_id - 1], width=2),
                                                    marker=dict(size=4, color=colors[cluster_id - 1]),
                                                    showlegend=False,
                                                    hovertemplate=f'<b>Trajectory {tid}</b><br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>'
                                                ),
                                                row=row,
                                                col=col
                                            )
                                        
                                        # Update axes labels
                                        fig_compare.update_xaxes(title_text="X Coordinate", row=row, col=col)
                                        fig_compare.update_yaxes(title_text="Y Coordinate", row=row, col=col)
                                    
                                    fig_compare.update_layout(
                                        title_text="Cluster Comparison - Side-by-side View",
                                        height=400 * rows,
                                        hovermode='closest'
                                    )
                                    
                                    st.plotly_chart(fig_compare, use_container_width=True)
                                
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

        # Get common parameters
        config_sources = df['config_source'].drop_duplicates().tolist()
        objects = sorted(df['obj'].unique())

        col1, col2 = st.columns(2)
        with col1:
            selected_configs = st.multiselect(
                "Select configuration(s)",
                config_sources,
                default=config_sources,
                key="extra_configs"
            )
        with col2:
            selected_objects = st.multiselect(
                "Select object(s)",
                objects,
                default=objects[:min(5, len(objects))],
                key="extra_objects"
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
