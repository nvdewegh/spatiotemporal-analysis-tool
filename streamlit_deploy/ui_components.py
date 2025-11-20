"""
Reusable UI Components for Streamlit Visualization
==================================================

This module contains reusable UI components to ensure consistency
across different parts of the application.
"""

import streamlit as st


def configuration_selector(
    config_ids,
    key_prefix,
    default_configs=None,
    max_selections=5,
    label="Select configurations to visualize",
    help_text="Choose configurations to visualize their trajectories",
    show_slider=False,
    show_metrics=True,
    min_slider_value=1
):
    """
    Reusable configuration selector component.
    
    This creates a consistent configuration selection interface used throughout
    the application (PDP Analysis, Trajectory Comparison, etc.).
    
    Parameters:
    -----------
    config_ids : list
        List of available configuration IDs to choose from
    key_prefix : str
        Unique prefix for the Streamlit widget keys (e.g., "pdp_inspect", "traj_compare")
    default_configs : list, optional
        Default configurations to select. If None, starts with empty selection
    max_selections : int, default=5
        Maximum number of configurations that can be selected
    label : str, default="Select configurations to visualize"
        Label for the multiselect widget
    help_text : str, optional
        Help text shown when hovering over the widget
    show_slider : bool, default=False
        Whether to show a slider for number of configs (used in trajectory comparison)
    show_metrics : bool, default=True
        Whether to show the metric box with count
    min_slider_value : int, default=1
        Minimum value for the slider (if show_slider=True)
    
    Returns:
    --------
    list
        List of selected configuration IDs
    
    Example Usage:
    --------------
    ```python
    # Simple usage (PDP inspect individual configs)
    selected_configs = configuration_selector(
        config_ids=all_configs,
        key_prefix="pdp_inspect",
        label="Select configurations to visualize",
        help_text="Choose configurations from the distance matrix"
    )
    
    # With slider (Trajectory comparison)
    selected_configs = configuration_selector(
        config_ids=all_configs,
        key_prefix="traj_compare",
        default_configs=all_configs[:2],
        show_slider=True,
        show_metrics=True
    )
    ```
    
    Where it's used:
    ----------------
    1. PDP Analysis → "Inspect Individual Configurations" (line ~3695)
    2. PDP Analysis → "Trajectory Comparison on Tennis Court" (line ~4173)
    3. (Add more locations as you refactor)
    """
    
    if default_configs is None:
        default_configs = []
    
    # Create columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if show_slider:
            # Show slider for number of configs
            num_configs = st.slider(
                "Number of configurations to compare",
                min_value=min_slider_value,
                max_value=min(max_selections, len(config_ids)),
                value=min(len(default_configs) if default_configs else 2, len(config_ids)),
                help="Select how many configurations to overlay",
                key=f"{key_prefix}_slider"
            )
            # Adjust default to match slider
            if len(default_configs) != num_configs:
                default_configs = config_ids[:num_configs]
        
        # Multiselect widget
        selected_configs = st.multiselect(
            label,
            options=config_ids,
            default=default_configs,
            max_selections=max_selections,
            key=f"{key_prefix}_multiselect",
            help=help_text
        )
    
    with col2:
        if show_metrics and len(selected_configs) > 0:
            st.metric("Configs selected", len(selected_configs))
            if len(selected_configs) == 1:
                st.caption("💡 Select 2+ to compare")
    
    return selected_configs


def object_selector(
    all_objects,
    globally_selected_objects=None,
    key_prefix="obj_select",
    show_all_checkbox=True,
    default_show_all=True,
    max_default_objects=3,
    label="Select objects to show",
    help_text="Choose which moving objects to display"
):
    """
    Reusable object selector component.
    
    This creates a consistent object selection interface that respects
    the global selection from the sidebar.
    
    Parameters:
    -----------
    all_objects : list
        List of all available object IDs
    globally_selected_objects : list, optional
        Objects selected in the global sidebar interface
    key_prefix : str
        Unique prefix for the Streamlit widget keys
    show_all_checkbox : bool, default=True
        Whether to show the "Show all globally selected objects" checkbox
    default_show_all : bool, default=True
        Default state of the show-all checkbox
    max_default_objects : int, default=3
        Maximum number of objects to select by default (if not showing all)
    label : str
        Label for the multiselect widget
    help_text : str
        Help text for the multiselect widget
    
    Returns:
    --------
    list
        List of selected object IDs
    
    Example Usage:
    --------------
    ```python
    selected_objects = object_selector(
        all_objects=sorted(df['obj'].unique()),
        globally_selected_objects=st.session_state.shared_selected_objects,
        key_prefix="traj_objects"
    )
    ```
    """
    
    # Use globally selected objects if available, otherwise use all
    if globally_selected_objects is None or len(globally_selected_objects) == 0:
        available_objects = all_objects
    else:
        available_objects = globally_selected_objects
    
    if show_all_checkbox:
        show_all_objects = st.checkbox(
            "Show all globally selected objects",
            value=default_show_all,
            help="Uncheck to further filter the objects selected in 'Data Selection Interface'",
            key=f"{key_prefix}_show_all"
        )
        
        if not show_all_objects:
            selected_objects = st.multiselect(
                label,
                options=available_objects,
                default=available_objects[:min(max_default_objects, len(available_objects))],
                help=help_text,
                key=f"{key_prefix}_multiselect"
            )
        else:
            selected_objects = available_objects
    else:
        # No checkbox, just multiselect
        selected_objects = st.multiselect(
            label,
            options=available_objects,
            default=available_objects[:min(max_default_objects, len(available_objects))],
            help=help_text,
            key=f"{key_prefix}_multiselect"
        )
    
    # Show info about object selection
    if globally_selected_objects is not None and len(globally_selected_objects) > 0:
        st.info(f"ℹ️ Using {len(available_objects)} object(s) from **Data Selection Interface**. To change, go to the sidebar.")
    else:
        st.warning("⚠️ No objects selected in **Data Selection Interface**. All objects from dataset will be used.")
    
    return selected_objects


# Documentation of where these components are used
"""
USAGE LOCATIONS:
================

configuration_selector():
-------------------------
1. streamlit_visualization.py, line ~3695 - PDP Analysis → "Inspect Individual Configurations"
   - Used after distance matrix to select configs for trajectory visualization
   
2. streamlit_visualization.py, line ~4173 - PDP Analysis → "Trajectory Comparison on Tennis Court"
   - Used with slider to compare multiple trajectories with buffer/rough visualization

object_selector():
------------------
1. streamlit_visualization.py, line ~4189 - PDP Analysis → "Trajectory Comparison on Tennis Court"
   - Used to select which objects (players) to show in trajectory visualization

REFACTORING PLAN:
=================
To refactor existing code to use these components:

1. Import at top of streamlit_visualization.py:
   ```python
   from ui_components import configuration_selector, object_selector
   ```

2. Replace existing selection code with function calls:
   ```python
   # OLD:
   col_inspect1, col_inspect2 = st.columns([2, 1])
   with col_inspect1:
       selected_configs_inspect = st.multiselect(
           "Select configurations to visualize",
           options=config_ids,
           default=[],
           ...
       )
   with col_inspect2:
       if len(selected_configs_inspect) > 0:
           st.metric("Configs selected", len(selected_configs_inspect))
           ...
   
   # NEW:
   selected_configs_inspect = configuration_selector(
       config_ids=config_ids,
       key_prefix="pdp_inspect",
       default_configs=[],
       label="Select configurations to visualize",
       help_text="Choose configurations from the distance matrix to see their actual trajectories"
   )
   ```

3. Benefits:
   - Consistent UI across all analysis methods
   - Single place to update labels, styling, behavior
   - Easier to add new features (e.g., quick selection buttons could be added to the component)
   - Reduced code duplication
"""
