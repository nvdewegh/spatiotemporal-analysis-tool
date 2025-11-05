# Quick Reference: Using the Modular Architecture

## For Students/Researchers

### Adding a New Analysis Method

1. **Create a new module file:**
   ```bash
   cd streamlit_deploy/modules
   touch my_new_method.py
   ```

2. **Structure your module:**
   ```python
   """
   My New Analysis Method
   
   This module provides [description]
   """
   
   import numpy as np
   import pandas as pd
   import streamlit as st
   import plotly.graph_objects as go
   
   from .common import render_interactive_chart
   
   
   def my_computation_function(df, params):
       """
       Main computation logic.
       
       Args:
           df: Trajectory DataFrame
           params: Analysis parameters
       
       Returns:
           Results dictionary
       """
       # Your algorithm here
       pass
   
   
   def render_my_analysis_section(data, selected_configs, selected_objects):
       """
       Main UI function for this analysis method.
       
       Called from main streamlit_visualization.py
       """
       st.header("🎯 My New Analysis")
       st.info("Description of what this method does...")
       
       # Your Streamlit UI here
       # Use tabs, columns, expanders for organization
       
       # Call computation functions
       results = my_computation_function(data, params)
       
       # Visualize results
       fig = go.Figure(...)
       render_interactive_chart(st, fig, "Chart description")
   ```

3. **Register in `__init__.py`:**
   ```python
   __all__ = [
       'association_rules',
       'clustering',
       'sequence_analysis',
       'common',
       'utils',
       'my_new_method'  # Add this
   ]
   ```

4. **Import in main file:**
   ```python
   # In streamlit_visualization.py
   from modules import association_rules, clustering, sequence_analysis, utils, my_new_method
   ```

5. **Add to analysis method dropdown:**
   ```python
   # In main() function, sidebar section
   analysis_method = st.selectbox(
       "Select method",
       ["Visual Exploration", "Clustering", "Association Rules", 
        "Sequence Analysis", "Heat Maps", "My New Method", "Extra"]  # Add here
   )
   ```

6. **Add UI section:**
   ```python
   # In main() function, after other elif blocks
   elif analysis_method == "My New Method":
       my_new_method.render_my_analysis_section(
           data=st.session_state.data,
           selected_configs=st.session_state.shared_selected_configs,
           selected_objects=st.session_state.shared_selected_objects
       )
   ```

### Using Existing Modules

#### Clustering
```python
from modules import clustering

# Initialize session state
clustering.initialize_clustering_session_state()

# Compute distance matrix (3 methods available)
dist_matrix, ids, features, trajs = clustering.compute_feature_distance_matrix(
    df, configs, objects, start, end, selected_features=['total_distance', 'avg_speed']
)

# Or use Chamfer distance
dist_matrix, ids, trajs = clustering.compute_chamfer_distance_matrix(
    df, configs, objects, start, end
)

# Or use DTW
dist_matrix, ids, trajs = clustering.compute_dtw_distance_matrix(
    df, configs, objects, start, end
)

# Detect optimal clusters
optimal_k = clustering.detect_optimal_clusters(dist_matrix, max_clusters=10)

# Perform clustering
labels, linkage = clustering.perform_hierarchical_clustering(dist_matrix, n_clusters=optimal_k)
```

#### Sequence Analysis
```python
from modules import sequence_analysis

# Create spatial grid
grid = sequence_analysis.create_spatial_grid(court_type='Tennis', grid_rows=3, grid_cols=5)

# Build sequences
seq = sequence_analysis.build_event_based_sequence(
    df, config, obj_id, start_time, end_time, grid, compress=True
)

# Or interval-based
seq = sequence_analysis.build_interval_based_sequence(
    df, config, obj_id, start_time, end_time, grid, delta_t=0.2, compress=True
)

# Compute edit distance
distance = sequence_analysis.levenshtein_distance(seq1, seq2)

# Align sequences
alignment = sequence_analysis.needleman_wunsch(seq1, seq2, match=2, mismatch=-1, gap=-1)
# alignment = {'score': ..., 'aligned_seq1': [...], 'aligned_seq2': [...]}

# Extract patterns
ngrams = sequence_analysis.extract_ngrams(seq, n=2)
# Returns: Counter({('A', 'B'): 5, ('B', 'C'): 3, ...})
```

#### Association Rules
```python
from modules import association_rules

# This module has a complete UI, just call:
association_rules.render_association_rules_section(
    data=st.session_state.data,
    selected_configs=st.session_state.shared_selected_configs,
    selected_objects=st.session_state.shared_selected_objects,
    create_spatial_grid_func=sequence_analysis.create_spatial_grid
)
```

#### Utilities
```python
from modules import utils

# Load CSV data
df = utils.load_data(uploaded_file, update_state=True, show_success=True)

# Get consistent colors
color = utils.get_color(obj_id)  # Returns color string

# Simplify trajectories
simplified_points = utils.douglas_peucker(points, tolerance=1.0)

# Spatiotemporal simplification
simplified_points = utils.douglas_peucker_spatiotemporal(points, tolerance=1.0)

# Aggregate points
aggregated = utils.aggregate_points(points, aggregation_type='mean', temporal_resolution=1.0)

# Extract trajectory coordinates
coords = utils.get_trajectory_coords(df, obj_id, config, start_time, end_time)
```

#### Common Utilities
```python
from modules.common import render_interactive_chart, PLOTLY_CONFIG

# Render Plotly charts consistently
fig = go.Figure(...)
render_interactive_chart(st, fig, "Use toolbar to zoom and pan")

# Use standard config for charts
st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
```

## Common Patterns

### Pattern 1: Analysis with Tabs
```python
def render_my_analysis_section(data, selected_configs, selected_objects):
    st.header("📊 My Analysis")
    
    tabs = st.tabs(["Overview", "Detailed Results", "Visualization", "Export"])
    
    with tabs[0]:
        st.info("Overview information...")
    
    with tabs[1]:
        # Detailed results
        pass
    
    with tabs[2]:
        # Visualizations
        fig = create_visualization()
        render_interactive_chart(st, fig)
    
    with tabs[3]:
        # Export functionality
        pass
```

### Pattern 2: Computation with Caching
```python
import streamlit as st

@st.cache_data
def compute_expensive_result(df, params):
    """
    Cache expensive computations.
    """
    # Expensive computation here
    return result
```

### Pattern 3: Session State Management
```python
def initialize_my_session_state():
    """Initialize session state variables."""
    if 'my_param' not in st.session_state:
        st.session_state.my_param = default_value
    if 'my_results' not in st.session_state:
        st.session_state.my_results = None

# In render function:
initialize_my_session_state()

# Use session state
if st.button("Compute"):
    result = compute_something()
    st.session_state.my_results = result

if st.session_state.my_results is not None:
    display_results(st.session_state.my_results)
```

### Pattern 4: Interactive Charts
```python
import plotly.graph_objects as go
from modules.common import render_interactive_chart

def create_my_chart(data):
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Line'))
    
    # Update layout
    fig.update_layout(
        title="My Chart",
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
        hovermode='closest',
        height=600
    )
    
    return fig

# In UI:
fig = create_my_chart(data)
render_interactive_chart(st, fig, "Interactive chart - zoom, pan, or reset")
```

## File Organization Tips

1. **Keep modules focused:** One analysis method per module
2. **Use docstrings:** Document all functions with clear descriptions
3. **Type hints:** Use type hints for better code clarity
4. **Constants at top:** Define constants at module level
5. **Helper functions first:** Put computation logic before UI functions
6. **Main UI function last:** Put render_*_section() at the end

## Testing Your Module

1. **Run the app:**
   ```bash
   cd streamlit_deploy
   streamlit run streamlit_visualization.py
   ```

2. **Test your method:**
   - Upload sample data
   - Select your method from dropdown
   - Verify all UI elements work
   - Test with different parameter combinations
   - Check for errors in terminal

3. **Check for errors:**
   ```bash
   # Look for Python errors in terminal output
   # Look for JavaScript errors in browser console (F12)
   ```

## Common Issues and Solutions

### Issue: Module not found
**Solution:** Make sure:
- Module file is in `modules/` directory
- Module is listed in `modules/__init__.py`
- Import statement is correct in main file

### Issue: Function not working
**Solution:** Check:
- Function signature matches when called
- All required parameters are provided
- Data types are correct (DataFrame, list, dict, etc.)

### Issue: Chart not displaying
**Solution:**
- Use `render_interactive_chart()` from common module
- Ensure fig is a valid Plotly Figure object
- Check browser console for JavaScript errors

### Issue: Session state not persisting
**Solution:**
- Initialize in `initialize_*_session_state()` function
- Call initialization function at start of render function
- Use unique keys for all widgets

## Best Practices

1. **Always use the shared selection:**
   ```python
   selected_configs = st.session_state.shared_selected_configs
   selected_objects = st.session_state.shared_selected_objects
   ```

2. **Provide informative messages:**
   ```python
   st.info("ℹ️ Information message")
   st.warning("⚠️ Warning message")
   st.error("❌ Error message")
   st.success("✅ Success message")
   ```

3. **Use expanders for details:**
   ```python
   with st.expander("ℹ️ How does this work?", expanded=False):
       st.markdown("Detailed explanation...")
   ```

4. **Progress indicators for long operations:**
   ```python
   with st.spinner("Computing results..."):
       result = expensive_computation()
   ```

5. **Consistent formatting:**
   - Use emojis for visual clarity (📊 🎯 🔍 ⚙️)
   - Use markdown headers (##, ###)
   - Use horizontal rules (st.markdown('---'))
