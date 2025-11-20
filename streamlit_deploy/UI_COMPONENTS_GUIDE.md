# UI Components - Quick Reference

## Overview

The `ui_components.py` module provides **reusable UI components** to ensure consistency across the application. Instead of duplicating selection interfaces throughout the code, you can now use these standardized functions.

## Why This Matters

✅ **Before (Current State):**
- Configuration selection code duplicated in multiple places
- Each location might have slightly different behavior
- Changing the UI requires updating multiple locations
- Risk of inconsistent user experience

✅ **After (With UI Components):**
- Single source of truth for selection interfaces
- Consistent behavior everywhere
- Update once, applies everywhere
- Easy to add new features globally

---

## Available Components

### 1. `configuration_selector()`

Creates a configuration selection interface (multiselect + optional slider + metrics).

**Basic Usage:**
```python
from ui_components import configuration_selector

selected_configs = configuration_selector(
    config_ids=config_ids,
    key_prefix="pdp_inspect",  # Must be unique!
    label="Select configurations to visualize",
    help_text="Choose configurations from the distance matrix"
)
```

**With Slider (for trajectory comparison):**
```python
selected_configs = configuration_selector(
    config_ids=config_ids,
    key_prefix="traj_compare",
    default_configs=config_ids[:2],
    show_slider=True,  # Shows "Number of configurations to compare" slider
    show_metrics=True
)
```

**Parameters:**
- `config_ids` (required): List of available configuration IDs
- `key_prefix` (required): Unique identifier for this selector instance
- `default_configs`: Initial selection (default: empty list)
- `max_selections`: Maximum configs to select (default: 5)
- `label`: Widget label (default: "Select configurations to visualize")
- `help_text`: Tooltip text
- `show_slider`: Show slider for number selection (default: False)
- `show_metrics`: Show count metric box (default: True)

---

### 2. `object_selector()`

Creates an object (player/entity) selection interface that respects global sidebar selections.

**Usage:**
```python
from ui_components import object_selector

selected_objects = object_selector(
    all_objects=sorted(df['obj'].unique()),
    globally_selected_objects=st.session_state.shared_selected_objects,
    key_prefix="traj_objects"
)
```

**Parameters:**
- `all_objects` (required): List of all available object IDs
- `globally_selected_objects`: Objects from sidebar (default: None)
- `key_prefix` (required): Unique identifier for this selector
- `show_all_checkbox`: Show "Show all globally selected objects" checkbox (default: True)
- `default_show_all`: Default checkbox state (default: True)
- `max_default_objects`: Max objects to select by default (default: 3)
- `label`: Widget label (default: "Select objects to show")
- `help_text`: Tooltip text

---

## Current Locations Using Selection

Here's where similar selection patterns exist in the code:

### 🔹 Location 1: PDP Analysis → "Inspect Individual Configurations"
**File:** `streamlit_visualization.py`  
**Line:** ~3695  
**Current Code:**
```python
col_inspect1, col_inspect2 = st.columns([2, 1])
with col_inspect1:
    selected_configs_inspect = st.multiselect(
        "Select configurations to visualize",
        options=config_ids,
        default=[],
        max_selections=5,
        key="pdp_matrix_inspect_configs",
        ...
    )
with col_inspect2:
    if len(selected_configs_inspect) > 0:
        st.metric("Configs selected", len(selected_configs_inspect))
        ...
```

**Refactored:**
```python
selected_configs_inspect = configuration_selector(
    config_ids=config_ids,
    key_prefix="pdp_inspect",
    default_configs=[],
    label="Select configurations to visualize",
    help_text="Choose configurations from the distance matrix to see their actual trajectories"
)
```

---

### 🔹 Location 2: PDP Analysis → "Trajectory Comparison on Tennis Court"
**File:** `streamlit_visualization.py`  
**Line:** ~4165-4180  
**Current Code:**
```python
num_configs_to_compare = st.slider(
    "Number of configurations to compare",
    min_value=1,
    max_value=min(5, len(config_ids)),
    value=min(2, len(config_ids)),
    ...
)
selected_configs_viz = st.multiselect(
    "Select configurations to visualize",
    options=config_ids,
    default=config_ids[:num_configs_to_compare],
    max_selections=5,
    ...
)
```

**Refactored:**
```python
selected_configs_viz = configuration_selector(
    config_ids=config_ids,
    key_prefix="traj_compare",
    default_configs=config_ids[:2],
    show_slider=True,
    show_metrics=True
)
```

---

## How to Refactor

### Step 1: Import the Components
Add to top of `streamlit_visualization.py`:
```python
from ui_components import configuration_selector, object_selector
```

### Step 2: Replace Existing Code
Find each location with selection UI and replace with function call.

### Step 3: Test
Run the app and verify:
- ✅ Selections work correctly
- ✅ Keys are unique (no duplicates)
- ✅ Behavior is consistent

---

## Benefits

### 🎯 Single Point of Change
Want to change the max selections from 5 to 10? Update one parameter in `ui_components.py`.

### 🎯 Consistency
All selection interfaces look and behave the same way.

### 🎯 New Features Everywhere
Add a "Select All" button to `configuration_selector()` → it appears in all locations automatically.

### 🎯 Less Code
Reduce 15-20 lines of duplicated code to 3-4 lines per location.

---

## Example: Adding Quick Selection Buttons Globally

Want to add quick selection buttons (Random, Min/Max, etc.) back - but everywhere?

```python
def configuration_selector(..., show_quick_buttons=False, distance_matrix=None):
    # ... existing code ...
    
    if show_quick_buttons and distance_matrix is not None:
        st.markdown("**Quick Selection:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🎲 Random 3", key=f"{key_prefix}_random"):
                # Random selection logic
                ...
        
        with col2:
            if st.button("📈 Min/Max", key=f"{key_prefix}_minmax"):
                # Min/max distance logic
                ...
        
        # ... more buttons ...
```

Now any location can enable quick buttons with one parameter!

---

## Next Steps

1. **Review** the `ui_components.py` file
2. **Test** one refactoring (e.g., PDP inspect configs)
3. **Gradually refactor** other locations
4. **Extend** the components with new features as needed

This approach makes your code more maintainable and scalable! 🚀
