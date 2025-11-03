# Selection Synchronization Feature

## Overview

This feature ensures that configuration and object selections are synchronized across all analysis methods in the Spatiotemporal Analysis Tool. When you select specific configurations or objects in one method, those selections are automatically maintained when switching to other methods.

## How It Works

### 1. Shared State Management

The application uses Streamlit's session state to maintain two shared variables:
- `shared_selected_configs`: List of selected configuration/rally names
- `shared_selected_objects`: List of selected object IDs

### 2. Initialization

When data is first loaded:
```python
# All configurations are selected by default
st.session_state.shared_selected_configs = config_sources

# First 5 objects are selected by default  
st.session_state.shared_selected_objects = objects[:min(5, len(objects))]
```

### 3. Synchronization Across Methods

Each analysis method now:

1. **Initializes shared state** (if it doesn't exist)
2. **Validates selections** against current data
3. **Uses validated selections** as defaults in multiselect widgets
4. **Updates shared state** whenever user makes new selections

### 4. Validation

Each method validates the shared selections to ensure they still exist in the current dataset:

```python
# Validate shared selections against current data
valid_configs = [c for c in st.session_state.shared_selected_configs if c in config_sources]
valid_objects = [o for o in st.session_state.shared_selected_objects if o in objects]

# Fallback to defaults if validation fails
if not valid_configs:
    valid_configs = config_sources
if not valid_objects:
    valid_objects = objects[:min(5, len(objects))]
```

## Updated Methods

The following analysis methods now synchronize selections:

1. **Visual Exploration**
   - Key: `visual_configs`, `visual_objects`
   
2. **2SA Method (Second-order Spatial Analysis)**
   - Key: `2sa_configs`, `2sa_objects`

3. **Sequence Analysis**
   - Key: `seq_configs`, `seq_objects`

4. **Clustering**
   - Key: `clustering_configs`, `clustering_objects`

5. **Extra Methods**
   - Key: `extra_configs`, `extra_objects`

## Example Usage Scenario

### Scenario 1: Filtering Objects
1. Load data with 10 players (objects 0-9)
2. In **Visual Exploration**, select only objects `0`, `1`, `2`
3. Switch to **Sequence Analysis**
   - Objects `0`, `1`, `2` are automatically selected
4. Switch to **Clustering**
   - Objects `0`, `1`, `2` remain selected

### Scenario 2: Filtering Configurations
1. Load data with 20 rallies (configurations)
2. In **Clustering**, select only 3 specific rallies
3. Switch to **Visual Exploration**
   - Those 3 rallies are automatically selected
4. In **Visual Exploration**, change to only 1 rally
5. Switch to **2SA Method**
   - That 1 rally is selected

### Scenario 3: Mixed Filtering
1. In **Visual Exploration**:
   - Select 3 configurations
   - Select 2 objects
2. Switch to **Sequence Analysis**:
   - Same 3 configurations selected
   - Same 2 objects selected
   - Select only 1 object
3. Switch back to **Visual Exploration**:
   - Same 3 configurations still selected
   - Now only 1 object selected (updated from Sequence Analysis)

## Benefits

1. **Consistency**: Selections remain consistent across methods
2. **Efficiency**: No need to re-select the same data in each method
3. **Workflow**: Natural analysis workflow - refine selections as you explore
4. **Extensibility**: New methods automatically inherit this behavior
5. **Data Safety**: Validation ensures selections are always valid

## Implementation Details

### Code Pattern

Each method follows this pattern:

```python
# 1. Get available data
config_sources = df['config_source'].drop_duplicates().tolist()
objects = sorted(df['obj'].unique())

# 2. Initialize shared state (if needed)
if 'shared_selected_configs' not in st.session_state:
    st.session_state.shared_selected_configs = config_sources
if 'shared_selected_objects' not in st.session_state:
    st.session_state.shared_selected_objects = objects[:min(5, len(objects))]

# 3. Validate selections
valid_configs = [c for c in st.session_state.shared_selected_configs if c in config_sources]
valid_objects = [o for o in st.session_state.shared_selected_objects if o in objects]

# 4. Fallback if needed
if not valid_configs:
    valid_configs = config_sources
if not valid_objects:
    valid_objects = objects[:min(5, len(objects))]

# 5. Create multiselect with validated defaults
selected_configs = st.multiselect(
    "Select configuration(s)",
    config_sources,
    default=valid_configs,
    key="method_specific_key_configs"
)

# 6. Update shared state
st.session_state.shared_selected_configs = selected_configs
```

## Future Enhancements

This pattern can be extended to synchronize other parameters:
- Time ranges (start_time, end_time)
- Temporal resolution
- Aggregation settings
- Visualization preferences

## Adding New Methods

When adding a new analysis method, follow this pattern:

1. Use the shared state variables for defaults
2. Update shared state when user makes selections
3. Validate selections against current data
4. Provide sensible fallbacks
5. Use unique `key` values for widgets

Example:
```python
# In your new method
selected_configs = st.multiselect(
    "Select configuration(s)",
    config_sources,
    default=valid_configs,
    key="newmethod_configs"  # Unique key!
)
st.session_state.shared_selected_configs = selected_configs  # Update shared state
```

## Testing

To verify the feature works:

1. Load sample data
2. In any method, select specific configurations and objects
3. Switch to another method
4. Verify selections are preserved
5. Change selections in the new method
6. Switch back to first method
7. Verify updated selections are reflected

## Notes

- Each multiselect widget needs a unique `key` parameter
- Shared state persists during the entire session
- New data uploads reset shared state to defaults
- Invalid selections (e.g., deleted configurations) are automatically filtered out
