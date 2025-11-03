# Selection Synchronization - Quick Reference

## How It Works

```
User Flow Example:
=================

1. Visual Exploration Method
   ┌─────────────────────────────────┐
   │ Select Configurations:          │
   │ ✓ Rally1                       │
   │ ✓ Rally2                       │
   │ ✓ Rally3                       │
   │                                 │
   │ Select Objects:                 │
   │ ✓ 0 (Player A)                 │
   │ ✓ 1 (Player B)                 │
   └─────────────────────────────────┘
              │
              │ (Switch to Sequence Analysis)
              ↓
2. Sequence Analysis Method
   ┌─────────────────────────────────┐
   │ Select Configurations:          │
   │ ✓ Rally1  ← AUTOMATICALLY       │
   │ ✓ Rally2  ← SELECTED            │
   │ ✓ Rally3  ← FROM PREVIOUS       │
   │                                 │
   │ Select Objects:                 │
   │ ✓ 0 (Player A)  ← SYNCED        │
   │ ✓ 1 (Player B)  ← FROM BEFORE   │
   │                                 │
   │ User changes to:                │
   │ ✓ 0 (Player A)  ONLY           │
   └─────────────────────────────────┘
              │
              │ (Switch to Clustering)
              ↓
3. Clustering Method
   ┌─────────────────────────────────┐
   │ Select Configurations:          │
   │ ✓ Rally1  ← STILL THE SAME     │
   │ ✓ Rally2  ← AS BEFORE           │
   │ ✓ Rally3  ← MAINTAINED          │
   │                                 │
   │ Select Objects:                 │
   │ ✓ 0 (Player A)  ← UPDATED TO    │
   │                  ← MATCH CHANGE │
   └─────────────────────────────────┘
```

## State Management

```
Session State Variables:
========================

st.session_state.shared_selected_configs
  ↓
  Used by all methods
  ↓
  [Rally1, Rally2, Rally3]


st.session_state.shared_selected_objects
  ↓
  Used by all methods
  ↓
  [0, 1]
```

## Method Keys

Each method uses unique widget keys but shares state:

| Method              | Config Key          | Object Key          |
|---------------------|---------------------|---------------------|
| Visual Exploration  | `visual_configs`    | `visual_objects`    |
| 2SA Method          | `2sa_configs`       | `2sa_objects`       |
| Sequence Analysis   | `seq_configs`       | `seq_objects`       |
| Clustering          | `clustering_configs`| `clustering_objects`|
| Extra Methods       | `extra_configs`     | `extra_objects`     |

All methods read from and write to:
- `st.session_state.shared_selected_configs`
- `st.session_state.shared_selected_objects`

## Validation Process

```
When entering a method:
=======================

1. Get current available data
   configs = [Rally1, Rally2, Rally3, Rally4]
   objects = [0, 1, 2, 3]

2. Check shared state
   shared_configs = [Rally1, Rally2, Rally5]  # Rally5 doesn't exist!
   shared_objects = [0, 1, 9]                 # Object 9 doesn't exist!

3. Validate selections
   valid_configs = [Rally1, Rally2]           # Rally5 filtered out
   valid_objects = [0, 1]                     # Object 9 filtered out

4. Use validated selections as defaults
   multiselect shows: Rally1, Rally2 selected
   multiselect shows: 0, 1 selected

5. When user changes selection
   User selects: [Rally1, Rally3]
   Update: shared_selected_configs = [Rally1, Rally3]
```

## Benefits

✅ **Consistency**: Same selections across all methods
✅ **Efficiency**: No need to reselect data in each method
✅ **Workflow**: Natural analysis flow
✅ **Safety**: Invalid selections automatically filtered
✅ **Extensible**: New methods automatically inherit behavior

## Example Workflows

### Workflow 1: Focus on Specific Rally
```
1. Visual Exploration: Select Rally1 only
2. Sequence Analysis: Rally1 auto-selected → analyze patterns
3. Clustering: Rally1 auto-selected → see clusters
4. 2SA Method: Rally1 auto-selected → compare movement
```

### Workflow 2: Focus on One Player
```
1. Clustering: Select all rallies, but only Object 0
2. Visual Exploration: Object 0 auto-selected → view trajectories
3. Sequence Analysis: Object 0 auto-selected → analyze sequences
4. 2SA Method: Object 0 auto-selected → track movement
```

### Workflow 3: Progressive Refinement
```
1. Visual Exploration: Select 10 rallies, 5 players
2. Sequence Analysis: Same 10 rallies, 5 players
   → Find interesting patterns → refine to 3 rallies
3. Clustering: Auto-updated to 3 rallies, 5 players
   → Find 2 players of interest
4. 2SA Method: Auto-updated to 3 rallies, 2 players
   → Deep dive analysis
```
