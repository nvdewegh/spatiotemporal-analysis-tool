"""
Streamlit GUI for PDP Analysis - SAM Version
============================================
Replaces Dash GUI with Streamlit interface.
All functionality from GUI.py but with simpler, more intuitive Streamlit components.

Run with: streamlit run GUI_streamlit.py
"""

import streamlit as st
import os
import sys
import threading
import time
import subprocess
import io
import contextlib
import importlib
from pathlib import Path
import warnings

# Set page config first (must be first Streamlit command)
st.set_page_config(
    page_title="🚀 PDP Analysis Dashboard",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set environment to skip heavy loading during import
os.environ.setdefault('AV_SKIP_LOAD', '1')

# Configure matplotlib to suppress font warnings and use available fonts
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Suppress matplotlib font warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='findfont: Font family.*not found')

# Set matplotlib to use DejaVu Sans (available on all systems) instead of Arial
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']

# Import av to get defaults
import av

# Initialize session state for persistent data
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'run_thread' not in st.session_state:
    st.session_state.run_thread = None
if 'last_status' not in st.session_state:
    st.session_state.last_status = 'idle'
if 'last_output' not in st.session_state:
    st.session_state.last_output = ''
if 'stop_requested' not in st.session_state:
    st.session_state.stop_requested = False
if 'analysis_finished' not in st.session_state:
    st.session_state.analysis_finished = False
if 'log_file_path' not in st.session_state:
    st.session_state.log_file_path = None
if 'analysis_params' not in st.session_state:
    st.session_state.analysis_params = None
if 'results_directory' not in st.session_state:
    st.session_state.results_directory = None
if 'expand_all_images' not in st.session_state:
    st.session_state.expand_all_images = True  # Default: show all images

# Password configuration (you can change this or use environment variable)
CORRECT_PASSWORD = os.environ.get('APP_PASSWORD', 'SAM2024')  # Default password or set via environment


def calculate_progress(output_text, params):
    """
    Calculate analysis progress based on completed modules and output.
    Returns (progress_percentage, current_stage, completed_tasks, total_tasks)
    """
    if not output_text:
        return 0, "Initializing...", 0, 1
    
    # Count completed modules by looking for "Time elapsed for running module" messages
    completed_modules = []
    
    # Parse all completed modules from output
    for line in output_text.split('\n'):
        if 'Time elapsed for running module' in line:
            # Extract module name
            if '"N_VA_StaticAbsolute"' in line:
                completed_modules.append('Static Absolute')
            elif '"N_VA_TennisCourt"' in line:
                completed_modules.append('Tennis Court')
            elif '"N_PDP"' in line:
                completed_modules.append('PDP Calculation')
            elif '"N_VA_HeatMap"' in line:
                completed_modules.append('Heatmap')
            elif '"N_VA_HClust"' in line:
                completed_modules.append('Hierarchical Clustering')
            elif '"N_VA_Mds"' in line:
                completed_modules.append('MDS')
            elif '"N_VA_TopK"' in line:
                completed_modules.append('Top-K Analysis')
            elif '"N_T_OB"' in line:
                completed_modules.append('Buffer Transform')
            elif '"N_VA_InequalityMatrices"' in line:
                completed_modules.append('Inequality Matrices')
    
    # Determine current stage
    if 'ALL PDP PROCESSING COMPLETE' in output_text:
        current_stage = "✅ Analysis Complete!"
        progress = 100
    elif 'STARTING PDP: BUFFER' in output_text:
        current_stage = "Processing Buffer PDP..."
        progress = 60
    elif 'STARTING PDP: FUNDAMENTAL' in output_text:
        current_stage = "Processing Fundamental PDP..."
        progress = 30
    elif 'STARTING ANALYSIS' in output_text:
        current_stage = "Initializing..."
        progress = 10
    else:
        current_stage = "Starting..."
        progress = 5
    
    # Calculate more accurate progress based on completed modules
    if len(completed_modules) > 0:
        # Estimate based on typical run (adjust based on actual modules)
        progress = min(95, 10 + (len(completed_modules) * 5))
    
    total_modules = len(set(completed_modules))
    
    return progress, current_stage, total_modules, total_modules


# Password configuration (you can change this or use environment variable)
CORRECT_PASSWORD = os.environ.get('APP_PASSWORD', 'GIST')  # Default password or set via environment


def check_password():
    """
    Returns True if the user has entered the correct password.
    Displays a login form if not authenticated.
    """
    if st.session_state.authenticated:
        return True
    
    # Show login form
    st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 40px; border-radius: 12px; color: white; margin-bottom: 20px;
                    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4); text-align: center;'>
            <h1 style='margin: 0 0 16px 0; font-size: 42px;'>🔒 SAM Analysis Platform</h1>
            <p style='font-size: 18px; opacity: 0.95; margin: 0;'>
                Please enter your password to access the dashboard
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🔐 Login")
        
        # Password input
        password = st.text_input("Password", type="password", key="password_input")
        
        col_login, col_space = st.columns([1, 1])
        with col_login:
            if st.button("🚀 Login", type="primary", use_container_width=True):
                if password == CORRECT_PASSWORD:
                    st.session_state.authenticated = True
                    st.success("✅ Login successful!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("❌ Incorrect password. Please try again.")
        
        st.markdown("---")
        st.info("💡 **Tip:** Contact your administrator if you've forgotten your password.")
        
        # Optional: Show environment hint for deployment
        if os.environ.get('APP_PASSWORD'):
            st.caption("🔒 Password is configured via environment variable")
        else:
            st.caption("⚠️ Using default password. Set APP_PASSWORD environment variable for production.")
    
    return False


# ============================================================================
# PASSWORD CHECK - Must pass before showing main app
# ============================================================================
if not check_password():
    st.stop()  # Stop execution if not authenticated

# Logout button in sidebar
with st.sidebar:
    st.markdown("---")
    if st.button("🚪 Logout", type="secondary", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()


def run_moving_objects_in_background(params):
    """
    Background worker function that runs N_Moving_Objects with given parameters.
    Writes output to a log file for real-time monitoring.
    """
    try:
        # Get results dir first
        results_dir = params.get('results_dir', os.getcwd())
        os.makedirs(results_dir, exist_ok=True)
        
        # Create status and log files
        log_file_path = os.path.join(results_dir, 'analysis_log.txt')
        status_file_path = os.path.join(results_dir, 'analysis_status.txt')
        
        # Write initial status
        with open(status_file_path, 'w', encoding='utf-8') as f:
            f.write('running')
        
        with open(log_file_path, 'w', encoding='utf-8') as debug_log:
            debug_log.write("="*60 + "\n")
            debug_log.write("🔍 DEBUG: Function started\n")
            debug_log.write(f"📂 Results dir: {results_dir}\n")
            debug_log.write(f"📊 Dataset param: {params.get('dataset_name', 'N_C_Dataset.csv')}\n")
            debug_log.write(f"🗂️ Current working dir: {os.getcwd()}\n")
            debug_log.write("="*60 + "\n")
        
        # Validate dataset file exists
        dataset_name = params.get('dataset_name', 'N_C_Dataset.csv')
        if not os.path.isfile(dataset_name):
            with open(log_file_path, 'a', encoding='utf-8') as debug_log:
                debug_log.write(f'❌ ERROR: Dataset file not found: {dataset_name}\n')
                debug_log.write(f'📂 Looking in: {os.path.abspath(dataset_name)}\n')
            with open(status_file_path, 'w', encoding='utf-8') as f:
                f.write('error')
            return
        
        # Set environment variables for av module
        if 'dataset_name' in params:
            os.environ['AV_DATASET'] = params['dataset_name']
        
        os.environ['AV_RESULTS_DIR'] = results_dir
        
        # Allow av to load dataset now
        os.environ['AV_SKIP_LOAD'] = '0'
        
        # Open log file for writing
        log_file = open(log_file_path, 'a', encoding='utf-8', buffering=1)
        
        # Redirect stdout and stderr to log file
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        class LogWriter:
            def __init__(self, log_file, original):
                self.log_file = log_file
                self.original = original
                
            def write(self, text):
                self.log_file.write(text)
                self.log_file.flush()
                
            def flush(self):
                self.log_file.flush()
        
        sys.stdout = LogWriter(log_file, old_stdout)
        sys.stderr = LogWriter(log_file, old_stderr)
        
        try:
            print("="*60)
            print("🚀 STARTING ANALYSIS")
            print("="*60)
            print(f"📊 Dataset: {params.get('dataset_name', 'N_C_Dataset.csv')}")
            print(f"📂 Results: {results_dir}")
            print("="*60)
            
            # Reload av so it processes the dataset with the provided environment
            import importlib
            importlib.reload(av)
            
            # Apply all parameters to av module
            for key, value in params.items():
                if hasattr(av, key):
                    setattr(av, key, value)
            
            print(f'✅ Configuration loaded successfully')
            print(f'🔢 Configurations: {av.con}, Timestamps: {av.tst}, Points: {av.poi}')
            print('='*60)
            
            # Import and run N_Moving_Objects
            import N_Moving_Objects
            importlib.reload(N_Moving_Objects)
            
            print('\n✅ Analysis completed successfully!')
            with open(status_file_path, 'w', encoding='utf-8') as f:
                f.write('finished')
                
        except Exception as e:
            print(f'\n❌ ERROR during analysis:\n{str(e)}')
            import traceback
            traceback.print_exc()
            with open(status_file_path, 'w', encoding='utf-8') as f:
                f.write('error')
        
        finally:
            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            log_file.close()
        
    except Exception as e:
        # Critical error - write to status file if possible
        try:
            status_file_path = os.path.join(params.get('results_dir', os.getcwd()), 'analysis_status.txt')
            with open(status_file_path, 'w', encoding='utf-8') as f:
                f.write('error')
        except:
            pass
    
    finally:
        # Reset environment
        os.environ['AV_SKIP_LOAD'] = '1'


# ============================================================================
# MAIN UI
# ============================================================================

# Header
st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 24px; border-radius: 12px; color: white; margin-bottom: 20px;
                box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);'>
        <h1 style='margin: 0 0 8px 0; font-size: 32px;'>🚀 Moving Objects Analysis Dashboard</h1>
        <p style='font-size: 16px; opacity: 0.95; margin: 0;'>
            ⚙️ Configureer parameters, selecteer modules en start de analyse
        </p>
    </div>
""", unsafe_allow_html=True)

# Create three columns for layout
col1, col2, col3 = st.columns([1, 1, 1])

# ============================================================================
# COLUMN 1: PDP Types & Core Parameters
# ============================================================================
with col1:
    st.markdown("### 📊 PDP Types")
    
    # Fundamental is always required (disabled checkbox)
    st.checkbox("🔹 Fundamental (required)", value=True, disabled=True, key='pdp_fundamental')
    
    # Optional PDP types
    pdp_buffer = st.checkbox("🔸 Buffer", value=getattr(av, 'PDPg_buffer', 1) == 1, key='pdp_buffer')
    pdp_rough = st.checkbox("🔶 Rough", value=getattr(av, 'PDPg_rough', 0) == 1, key='pdp_rough')
    pdp_bufferrough = st.checkbox("🔷 Buffer + Rough", value=getattr(av, 'PDPg_bufferrough', 0) == 1, key='pdp_bufferrough')
    
    st.markdown("### 🔢 Core Parameters")
    window_length_tst = st.number_input("⏱️ window_length_tst", value=getattr(av, 'window_length_tst', 3), min_value=1, step=1)
    
    st.markdown("### 📏 Buffer / Rough Parameters")
    col1a, col1b = st.columns(2)
    with col1a:
        buffer_x = st.number_input("↔️ buffer_x", value=getattr(av, 'buffer_x', 1), disabled=not pdp_buffer and not pdp_bufferrough)
        rough_x = st.number_input("↔️ rough_x", value=getattr(av, 'rough_x', 0), disabled=not pdp_rough and not pdp_bufferrough)
    with col1b:
        buffer_y = st.number_input("↕️ buffer_y", value=getattr(av, 'buffer_y', 1), disabled=not pdp_buffer and not pdp_bufferrough)
        rough_y = st.number_input("↕️ rough_y", value=getattr(av, 'rough_y', 0), disabled=not pdp_rough and not pdp_bufferrough)

# ============================================================================
# COLUMN 2: Visualization & Analysis Modules
# ============================================================================
with col2:
    st.markdown("### 🎨 Visualization & Analysis Modules")
    
    # Create multiselect for all modules
    module_options = {
        'N_PDP': 'PDP Calculation (Distance Matrices)',
        'N_VA_StaticAbsolute': 'Static Absolute',
        'N_VA_HeatMap': 'Heatmap',
        'N_VA_HClust': 'Hierarchical Clustering',
        'N_VA_Mds': 'MDS',
        'N_VA_InequalityMatrices': 'Inequality Matrices',
        'N_VA_TopK': 'Top K',
        'N_VA_TennisCourt': '🎾 Tennis Court',
    }
    
    # Get default selected modules
    default_selected = [k for k, v in module_options.items() if getattr(av, k, 0) == 1]
    
    selected_modules = st.multiselect(
        "Select modules to run:",
        options=list(module_options.keys()),
        default=default_selected,
        format_func=lambda x: module_options[x]
    )
    
    st.markdown("### 💾 Dataset & Output")
    
    # Dataset file selection
    dataset_name = st.text_input("📁 Dataset file (CSV path)", 
                                 value=getattr(av, 'dataset_name', 'N_C_Dataset.csv'),
                                 help="Enter full path to CSV file or use file uploader below")
    
    # File uploader as alternative to text input
    uploaded_file = st.file_uploader("Or upload a CSV file", type=['csv'], key='dataset_uploader')
    if uploaded_file is not None:
        # Save uploaded file to temp location
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            dataset_name = tmp_file.name
            st.success(f"✅ File uploaded: {uploaded_file.name}")
    
    # Results directory selection
    results_dir = st.text_input("📂 Results directory", 
                                value=os.environ.get('AV_RESULTS_DIR', os.getcwd()),
                                help="Enter full path to output directory")
    
    # Quick path suggestions
    with st.expander("💡 Common Paths"):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📁 Current Directory"):
                st.info(f"Current: {os.getcwd()}")
            if st.button("📁 User Home"):
                st.info(f"Home: {os.path.expanduser('~')}")
        with col2:
            if st.button("📁 Desktop"):
                desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
                st.info(f"Desktop: {desktop}")
            if st.button("📁 Documents"):
                docs = os.path.join(os.path.expanduser('~'), 'Documents')
                st.info(f"Documents: {docs}")

# ============================================================================
# COLUMN 3: Advanced Settings
# ============================================================================
with col3:
    with st.expander("⚙️ Advanced Settings", expanded=False):
        st.markdown("### 🎞️ Interpolation")
        num_frames = st.number_input("num_frames", value=getattr(av, 'num_frames', 20), min_value=1)
        
        st.markdown("### 🗺️ Spatial Bounds")
        col3a, col3b = st.columns(2)
        with col3a:
            min_boundary_x = st.number_input("⬅️ min_x", value=getattr(av, 'min_boundary_x', -5))
            min_boundary_y = st.number_input("⬇️ min_y", value=getattr(av, 'min_boundary_y', -5))
        with col3b:
            max_boundary_x = st.number_input("➡️ max_x", value=getattr(av, 'max_boundary_x', 15))
            max_boundary_y = st.number_input("⬆️ max_y", value=getattr(av, 'max_boundary_y', 30))
        
        st.markdown("### 🔧 Other Parameters")
        DD = st.number_input("📐 DD", value=getattr(av, 'DD', 2), min_value=1)
        des = st.number_input("📊 des", value=getattr(av, 'des', 2), min_value=1)
        dim = st.number_input("📏 dim", value=getattr(av, 'dim', 2), min_value=1)
        
        num_similar_configurations = st.number_input("🔢 num_similar_configurations", 
                                                     value=getattr(av, 'num_similar_configurations', 5), min_value=1)
        new_configuration_step = st.number_input("➕ new_configuration_step", 
                                                 value=getattr(av, 'new_configuration_step', 3), min_value=1)
        division_factor = st.number_input("➗ division_factor", 
                                         value=getattr(av, 'division_factor', 5), min_value=1)

# ============================================================================
# RUN CONTROL SECTION
# ============================================================================
st.markdown("---")
st.markdown("### 🎮 Analysis Control")

col_run1, col_run2, col_run3, col_run4 = st.columns(4)

with col_run1:
    if st.button("▶️ Run Analysis", type="primary", use_container_width=True):
        # Build params dictionary
        params = {
            # PDP types
            'PDPg_fundamental': 1,
            'PDPg_buffer': 1 if pdp_buffer else 0,
            'PDPg_rough': 1 if pdp_rough else 0,
            'PDPg_bufferrough': 1 if pdp_bufferrough else 0,
            'PDPg_fundamental_active': 1,
            'PDPg_buffer_active': 1 if pdp_buffer else 0,
            'PDPg_rough_active': 1 if pdp_rough else 0,
            'PDPg_bufferrough_active': 1 if pdp_bufferrough else 0,
            
            # Visualization modules
            **{k: (1 if k in selected_modules else 0) for k in module_options.keys()},
            
            # Auto-enable dependencies
            'N_PDP': 1 if any(mod in selected_modules for mod in ['N_VA_HeatMap', 'N_VA_HClust', 'N_VA_Mds', 'N_VA_TopK', 'N_VA_InequalityMatrices']) else (1 if 'N_PDP' in selected_modules else 0),
            
            # Numeric parameters
            'window_length_tst': int(window_length_tst),
            'num_frames': int(num_frames),
            'buffer_x': int(buffer_x),
            'buffer_y': int(buffer_y),
            'rough_x': int(rough_x),
            'rough_y': int(rough_y),
            'min_boundary_x': int(min_boundary_x),
            'max_boundary_x': int(max_boundary_x),
            'min_boundary_y': int(min_boundary_y),
            'max_boundary_y': int(max_boundary_y),
            'DD': int(DD),
            'des': int(des),
            'dim': int(dim),
            'num_similar_configurations': int(num_similar_configurations),
            'new_configuration_step': int(new_configuration_step),
            'division_factor': int(division_factor),
            'dataset_name': dataset_name,
            'results_dir': results_dir
        }
        
        # Start analysis in background thread
        if st.session_state.run_thread is None or not st.session_state.run_thread.is_alive():
            st.session_state.stop_requested = False
            st.session_state.analysis_params = params  # Store params for progress tracking
            st.session_state.results_directory = results_dir  # Store results directory
            
            st.session_state.run_thread = threading.Thread(
                target=run_moving_objects_in_background, 
                args=(params,), 
                daemon=True
            )
            st.session_state.run_thread.start()
            st.success("✅ Analysis started in background!")
            st.info(f"📂 Results will be saved to: {results_dir}")
            
            # Initialize status tracking
            status_file = os.path.join(results_dir, 'analysis_status.txt')
            with open(status_file, 'w', encoding='utf-8') as f:
                f.write('starting')
            
            time.sleep(0.5)
            st.rerun()
        else:
            st.warning("⚠️ Analysis already running!")

with col_run2:
    if st.button("⏹️ Stop Analysis", type="secondary", use_container_width=True):
        st.session_state.stop_requested = True
        st.warning("⏹️ Stop requested. Waiting for safe termination point...")
        time.sleep(0.5)
        st.rerun()

with col_run3:
    if st.button("🔄 Refresh Status", use_container_width=True):
        st.rerun()

with col_run4:
    if st.button("📈 View Results", 
                 disabled=not st.session_state.analysis_finished,
                 use_container_width=True):
        # Launch Streamlit viewer
        viewer_path = os.path.abspath(os.path.normpath(
            os.path.join(os.path.dirname(__file__), '..', '..', 'visualisations', 'app_PDP_results.py')
        ))
        
        if os.path.exists(viewer_path):
            env = os.environ.copy()
            env['AV_RESULTS_DIR'] = results_dir
            env['AV_DATASET'] = dataset_name
            
            try:
                subprocess.Popen(
                    [sys.executable, '-m', 'streamlit', 'run', viewer_path],
                    env=env,
                    creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
                )
                st.success(f"✅ Results viewer launched! Check your browser.")
            except Exception as e:
                st.error(f"❌ Could not launch viewer: {e}")
        else:
            st.error(f"❌ Viewer not found at: {viewer_path}")

# ============================================================================
# STATUS DISPLAY
# ============================================================================
st.markdown("---")
st.markdown("### 📊 Analysis Status")

# EXTENSIVE DEBUGGING
st.markdown("#### 🔍 Debug Information")
debug_col1, debug_col2 = st.columns(2)
with debug_col1:
    st.write(f"**Results Dir Set:** {st.session_state.results_directory is not None}")
    if st.session_state.results_directory:
        st.write(f"**Results Dir:** `{st.session_state.results_directory}`")
        st.write(f"**Dir Exists:** {os.path.exists(st.session_state.results_directory)}")
with debug_col2:
    st.write(f"**Analysis Params Set:** {st.session_state.analysis_params is not None}")
    st.write(f"**Thread Alive:** {st.session_state.run_thread is not None and st.session_state.run_thread.is_alive() if st.session_state.run_thread else False}")

# Read status from file if analysis was started
current_status = 'idle'
status_file_exists = False
log_file_exists = False

if st.session_state.results_directory and os.path.exists(st.session_state.results_directory):
    status_file = os.path.join(st.session_state.results_directory, 'analysis_status.txt')
    log_file = os.path.join(st.session_state.results_directory, 'analysis_log.txt')
    
    status_file_exists = os.path.exists(status_file)
    log_file_exists = os.path.exists(log_file)
    
    st.write(f"**Status File Exists:** {status_file_exists}")
    st.write(f"**Log File Exists:** {log_file_exists}")
    
    if status_file_exists:
        try:
            with open(status_file, 'r', encoding='utf-8') as f:
                current_status = f.read().strip()
                st.write(f"**Status from file:** `{current_status}`")
        except Exception as e:
            st.error(f"Error reading status: {e}")
            current_status = 'error'
    
    # Read log output
    if log_file_exists:
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                st.session_state.last_output = f.read()
                st.write(f"**Log size:** {len(st.session_state.last_output)} chars")
        except Exception as e:
            st.error(f"Error reading log: {e}")
    
    # Update analysis_finished flag
    if current_status == 'finished':
        st.session_state.analysis_finished = True
        st.session_state.last_status = 'finished'
    elif current_status == 'error':
        st.session_state.last_status = 'error'
    elif current_status == 'running':
        st.session_state.last_status = 'running'
else:
    current_status = st.session_state.last_status
    st.write(f"**Using session status:** `{current_status}`")

st.markdown("---")

# Status indicator
status_emoji = {
    'idle': '💤',
    'starting': '🔄',
    'running': '⚙️',
    'finished': '✅',
    'stopped': '⏹️',
    'error': '❌'
}

status_color = {
    'idle': 'gray',
    'starting': 'blue',
    'running': 'blue',
    'finished': 'green',
    'stopped': 'orange',
    'error': 'red'
}

st.markdown(f"**Status:** :{status_color.get(current_status, 'gray')}[{status_emoji.get(current_status, '❓')} {current_status.upper()}]")

# Manual refresh button
if st.button("🔄 Force Refresh", type="secondary"):
    st.rerun()

# Show files in results directory if it exists
if st.session_state.results_directory and os.path.exists(st.session_state.results_directory):
    with st.expander("📁 Files in Results Directory", expanded=False):
        try:
            files = os.listdir(st.session_state.results_directory)
            st.write(f"**Total files:** {len(files)}")
            for f in files[:20]:  # Show first 20
                st.text(f"  • {f}")
            if len(files) > 20:
                st.text(f"  ... and {len(files) - 20} more")
        except Exception as e:
            st.error(f"Error listing files: {e}")

# Show progress bar when running
# Show progress bar whenever we have analysis params and output (running or finishing)
if st.session_state.analysis_params and st.session_state.last_output:
    progress, current_stage, completed, total = calculate_progress(
        st.session_state.last_output, 
        st.session_state.analysis_params
    )
    
    st.markdown("### 📊 Progress")
    
    # If finished, show 100%, otherwise show calculated progress
    if current_status == 'finished':
        st.progress(1.0, text="100% - ✅ Analysis Complete!")
    else:
        st.progress(progress / 100, text=f"{progress}% - {current_stage}")
        
        # Show detailed module completion
        if completed > 0:
            st.caption(f"✅ Completed {completed} unique modules")
    
    st.markdown("---")

# Auto-refresh while running OR if thread is alive but status hasn't updated yet
should_refresh = (current_status in ['running', 'starting']) or \
                 (st.session_state.run_thread and st.session_state.run_thread.is_alive())

if should_refresh:
    # Parse timing information from output
    time_info = ""
    if st.session_state.last_output:
        # Extract elapsed time from last "Time elapsed" line
        lines = st.session_state.last_output.split('\n')
        elapsed_times = [l for l in lines if 'Time elapsed for running module' in l or 'Total time elapsed' in l]
        if elapsed_times:
            last_time = elapsed_times[-1]
            time_info = f"⏱️ {last_time.strip()}"
    
    # Show live output in expander with time info
    expander_title = "📟 Live Output Log"
    if time_info:
        expander_title += f" - {time_info}"
    
    with st.expander(expander_title, expanded=True):
        if st.session_state.last_output:
            # Show last 5000 chars for better context
            output_display = st.session_state.last_output[-5000:]
            st.code(output_display, language='text')
            
            # Add helpful info about progress
            if 'STARTING PDP: FUNDAMENTAL' in st.session_state.last_output:
                st.info("🔄 Processing Fundamental PDP - This may take a few minutes...")
            elif 'STARTING PDP: BUFFER' in st.session_state.last_output:
                st.info("🔄 Processing Buffer PDP - About halfway through...")
            elif 'ALL PDP PROCESSING COMPLETE' in st.session_state.last_output:
                st.success("✅ All processing complete!")
        else:
            st.info("⏳ Waiting for output...")
    
    time.sleep(2)  # Refresh every 2 seconds
    st.rerun()

# Output display (when not running) - Always show if there's output
elif st.session_state.last_output:
    # Determine if we should expand by default
    expand_by_default = current_status in ['finished', 'error']
    
    # Parse timing information from output for finished state
    time_info = ""
    if current_status == 'finished':
        lines = st.session_state.last_output.split('\n')
        total_time_lines = [l for l in lines if 'Total time elapsed' in l]
        if total_time_lines:
            time_info = f" - {total_time_lines[-1].strip()}"
    
    expander_title = "📄 Analysis Output Log" + time_info
    
    with st.expander(expander_title, expanded=expand_by_default):
        st.code(st.session_state.last_output, language='text')

# ============================================================================
# RESULTS VIEWER - VISUALIZATIONS
# ============================================================================
# Show results if status is finished OR if we have a results directory with files
show_results = False
if current_status == 'finished' and st.session_state.results_directory:
    show_results = True
elif st.session_state.results_directory and os.path.exists(st.session_state.results_directory):
    # Check if there are any PNG files
    try:
        files = os.listdir(st.session_state.results_directory)
        if any(f.lower().endswith('.png') for f in files):
            show_results = True
    except:
        pass

if show_results:
    st.markdown("---")
    
    # Header with expand/collapse button
    col_title, col_button = st.columns([3, 1])
    with col_title:
        st.markdown("## 🎨 Analysis Results - Visualizations")
    with col_button:
        button_label = "📦 Collapse All" if st.session_state.expand_all_images else "📂 Expand All"
        if st.button(button_label, use_container_width=True):
            st.session_state.expand_all_images = not st.session_state.expand_all_images
            st.rerun()
    
    browse_results_dir = st.session_state.results_directory
    
    if os.path.exists(browse_results_dir):
        try:
            # Collect all PNG files
            visualization_files = {
                'Static Absolute': [],
                'Heatmaps': [],
                'Hierarchical Clustering': [],
                'MDS': [],
                'Top-K': [],
                'Tennis Court': [],
                'Inequality Matrices': [],
                'Other': []
            }
            
            all_files_for_download = []
            
            for root, dirs, files in os.walk(browse_results_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, browse_results_dir)
                    file_lower = file.lower()
                    
                    # Skip Python scripts and debug files
                    if file_lower.endswith('.py') or file_lower == '_button_clicked.txt':
                        continue
                    
                    all_files_for_download.append((rel_path, file_path))
                    
                    # Categorize PNG images for display
                    if file_lower.endswith('.png') or file_lower.endswith('.jpg') or file_lower.endswith('.jpeg'):
                        # Check if in TennisCourt folder
                        if 'tenniscourt' in rel_path.lower() or 'tennis_court' in rel_path.lower():
                            visualization_files['Tennis Court'].append((file, rel_path, file_path))
                        elif 'static' in file_lower and 'absolute' in file_lower:
                            visualization_files['Static Absolute'].append((file, rel_path, file_path))
                        elif 'heatmap' in file_lower:
                            visualization_files['Heatmaps'].append((file, rel_path, file_path))
                        elif 'hclust' in file_lower or 'dendrogram' in file_lower or 'cluster' in file_lower:
                            visualization_files['Hierarchical Clustering'].append((file, rel_path, file_path))
                        elif 'mds' in file_lower:
                            visualization_files['MDS'].append((file, rel_path, file_path))
                        elif 'topk' in file_lower or 'top_k' in file_lower or 'top-k' in file_lower:
                            visualization_files['Top-K'].append((file, rel_path, file_path))
                        elif 'tennis' in file_lower or 'court' in file_lower:
                            visualization_files['Tennis Court'].append((file, rel_path, file_path))
                        elif 'inequality' in file_lower:
                            visualization_files['Inequality Matrices'].append((file, rel_path, file_path))
                        else:
                            visualization_files['Other'].append((file, rel_path, file_path))
            
            # Download button at the top
            if all_files_for_download:
                import zipfile
                from io import BytesIO
                
                col_download, col_info = st.columns([1, 3])
                with col_download:
                    # Create ZIP file
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        for rel_path, full_path in all_files_for_download:
                            zip_file.write(full_path, rel_path)
                    
                    zip_buffer.seek(0)
                    st.download_button(
                        label="� Download All Results (ZIP)",
                        data=zip_buffer,
                        file_name=f"SAM_Results_{time.strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip",
                        type="primary",
                        use_container_width=True
                    )
                
                with col_info:
                    total_size = sum(os.path.getsize(fp) for _, fp in all_files_for_download)
                    st.info(f"📊 {len(all_files_for_download)} files ready • 💾 {total_size / 1024 / 1024:.2f} MB total")
            
            st.markdown("---")
            
            # Display visualizations by category
            non_empty_categories = {k: v for k, v in visualization_files.items() if v}
            
            if non_empty_categories:
                for category, files in non_empty_categories.items():
                    # Use expander for each category
                    with st.expander(f"### {category} ({len(files)} files)", expanded=st.session_state.expand_all_images):
                        # Create columns for images (2 per row)
                        for i in range(0, len(files), 2):
                            cols = st.columns(2)
                            for j, col in enumerate(cols):
                                if i + j < len(files):
                                    filename, rel_path, file_path = files[i + j]
                                    with col:
                                        st.image(file_path, caption=filename, use_container_width=True)
                    
                    st.markdown("---")
            else:
                st.info("📭 No visualization files (PNG) found. Other results may be available in the download.")
        
        except Exception as e:
            st.error(f"❌ Error loading results: {e}")
            import traceback
            st.code(traceback.format_exc())
    else:
        st.warning(f"⚠️ Results directory does not exist: {browse_results_dir}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px;'>
        🎾 SAM Tennis Analysis Module | Built with Streamlit
    </div>
""", unsafe_allow_html=True)
