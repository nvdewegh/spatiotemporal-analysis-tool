# Sequence Analysis Feature Guide

## Overview

The **Sequence Analysis** feature translates spatiotemporal trajectory data into symbolic sequences for pattern mining and comparison. This approach is based on sequence analysis methods commonly used in bioinformatics, linguistics, and spatiotemporal data mining.

## Purpose

Instead of analyzing raw (x, y, t) coordinates, we:
1. **Discretize space** into symbolic zones (A, B, C, ...)
2. **Sample time** either at events (hits/bounces) or at fixed intervals
3. **Build sequences** of zone tokens representing movement patterns
4. **Compare sequences** using alignment algorithms and distance metrics
5. **Mine patterns** to discover common movement sub-sequences

This allows us to:
- Find similar rallies based on spatial patterns
- Discover frequent movement patterns (n-grams)
- Compare whole rallies (global alignment) or find similar sub-sequences (local alignment)
- Study scale effects by varying grid resolution and time sampling

## Features Implemented

### 1. Spatial Discretization
- **Grid Creation**: Divide the court into an M×N grid of zones
- **Zone Labeling**: Each zone gets a letter (A, B, C, ..., O for a 3×5 grid)
- **Coordinate Mapping**: Convert (x, y) positions to zone letters
- **Configurable**: Adjustable grid resolution (2-10 rows/columns)

### 2. Temporal Sampling

#### Event-Based Sampling
- One token per data point (hit/bounce)
- Preserves all movement events
- Best for irregular event sequences

#### Equal-Interval Sampling
- Fixed time step Δt (e.g., 0.2s)
- Regular sampling for uniform analysis
- Adjustable interval (0.1-2.0s)

### 3. Sequence Types

#### Per-Entity Sequences
- Separate sequence for each object (ball, player1, player2)
- Example: Ball sequence = "AABBBCCCDDD"
- Best for single-object pattern analysis

#### Multi-Entity Sequences
- Combined tokens for all entities at each moment
- Example: "Ball:A|P1:C|P2:F; Ball:B|P1:C|P2:E; ..."
- Captures relational patterns between entities

### 4. Run-Length Compression
- Optional compression of repeated zones
- "AAABBBCCCC" → "ABC"
- Reduces sequence length while preserving zone transitions
- Useful for focusing on spatial patterns vs. temporal duration

### 5. Sequence Comparison Methods

#### Levenshtein Distance (Edit Distance)
- Counts minimum edits (insertions, deletions, substitutions) to transform one sequence to another
- **Standard**: Absolute edit count
- **Normalized**: Divided by max sequence length (0-1 range)
- Used for clustering similar rallies

#### Global Alignment (Needleman-Wunsch)
- Aligns entire sequences end-to-end
- Optimizes match/mismatch/gap scoring
- Best for comparing whole rally patterns
- Configurable parameters:
  - Match score (default: +2)
  - Mismatch penalty (default: -1)
  - Gap penalty (default: -1)

#### Local Alignment (Smith-Waterman)
- Finds best matching sub-sequences
- Allows free gaps at ends
- Best for finding similar sub-patterns within different rallies
- Same scoring parameters as global alignment

### 6. Pattern Mining

#### N-gram Extraction
- Extract all n-letter sub-sequences
- Count frequencies across all rallies
- Identify most common patterns
- Configurable n (2-5)
- Per-sequence breakdown available

### 7. Visualization Tools

#### Distance Matrix Heatmap
- Visual representation of pairwise sequence distances
- Darker = more different
- Interactive hover shows exact distances

#### Sequence Alignment Display
- Color-coded alignment visualization:
  - 🟢 Green: Matching positions
  - 🟠 Orange: Mismatches
  - 🔴 Red: Gaps (insertions/deletions)
- Alignment statistics (matches, mismatches, gaps)

#### Spatial Grid Overlay
- Shows zone divisions on court
- Zone labels (A, B, C, ...)
- Grid lines for visual reference

#### Zone Visit Heatmap
- Frequency of visits to each zone
- Identifies most/least used court areas
- Color intensity = visit frequency

#### N-gram Frequency Chart
- Bar chart of most common patterns
- Shows pattern distribution
- Exportable results

### 8. Clustering
- Hierarchical clustering based on sequence distances
- Adjustable number of clusters
- Cluster assignment and statistics
- Groups similar rally patterns

## Usage Workflow

### Step 1: Configure Grid Resolution
```
Grid rows: 3 (default)
Grid columns: 5 (default)
→ Creates 15 zones (A-O)
```

**Considerations:**
- Finer grid (e.g., 5×7) = more spatial detail, longer sequences
- Coarser grid (e.g., 2×3) = more abstract patterns, shorter sequences
- Course text suggests experimenting with different resolutions (§2.8.2)

### Step 2: Choose Temporal Sampling
```
Mode: Event-based or Equal-interval
If Equal-interval: Δt = 0.2s (adjustable)
```

**Considerations:**
- Event-based: Natural for tennis (hit/bounce events)
- Equal-interval: Better for cross-comparison, uniform analysis
- Try both and compare results

### Step 3: Select Compression and Type
```
Run-length compression: ON/OFF
Sequence type: Per-entity or Multi-entity
```

**Considerations:**
- Compression ON: Focus on zone transitions (ABC vs AAABBBCCC)
- Compression OFF: Include temporal duration in zones
- Per-entity: Analyze each object separately
- Multi-entity: Capture relational patterns

### Step 4: Select Data
```
Configurations: Select rallies to analyze
Objects: Select ball, players, etc.
Time range: Focus on specific rally segments
```

### Step 5: Analyze Results

**Tab 1: Distance Matrix**
- View pairwise sequence distances
- Perform hierarchical clustering
- Identify similar rally groups

**Tab 2: Pairwise Alignment**
- Select two rallies
- Choose global or local alignment
- View aligned sequences with match visualization
- Examine alignment statistics

**Tab 3: N-gram Patterns**
- Set n-gram size (2-5)
- View most common patterns
- Explore per-sequence breakdowns
- Export pattern frequencies

**Tab 4: Spatial Grid View**
- See zone definitions on court
- View zone visit heatmap
- Understand spatial discretization

## Example Use Cases

### 1. Find Similar Rallies
1. Build sequences for all rallies
2. Compute distance matrix
3. Use clustering to group similar patterns
4. Analyze what makes rallies similar

### 2. Discover Common Patterns
1. Build per-entity sequences (e.g., ball only)
2. Extract 2-grams or 3-grams
3. Identify most frequent patterns
4. Interpret: "AB" might mean serve from zone A to zone B

### 3. Compare Rally Segments
1. Use local alignment (Smith-Waterman)
2. Find similar sub-sequences in different rallies
3. Identify common tactical patterns
4. Study recurring movement sequences

### 4. Scale Analysis
1. Run analysis with 2×3 grid
2. Re-run with 3×5 grid
3. Compare pattern stability
4. Determine appropriate resolution (as suggested in course §2.8.2)

### 5. Temporal Resolution Study
1. Compare event-based vs interval-based
2. Try different Δt values (0.1s, 0.5s, 1.0s)
3. Assess how sampling affects discovered patterns
4. Find optimal temporal resolution

## Technical Implementation

### Algorithms

#### Spatial Discretization
```python
# Create M×N grid
x_bins = linspace(0, court_width, N+1)
y_bins = linspace(0, court_height, M+1)

# Map (x,y) to zone
col = digitize(x, x_bins) - 1
row = digitize(y, y_bins) - 1
zone = chr(65 + row*N + col)  # A, B, C, ...
```

#### Levenshtein Distance (Dynamic Programming)
```python
# DP table: dp[i][j] = distance between seq1[:i] and seq2[:j]
dp[0][j] = j  # Insert j characters
dp[i][0] = i  # Delete i characters

dp[i][j] = min(
    dp[i-1][j] + 1,      # deletion
    dp[i][j-1] + 1,      # insertion
    dp[i-1][j-1] + cost  # substitution (cost=0 if match)
)
```

#### Needleman-Wunsch (Global Alignment)
```python
# Similar DP, but with scoring
score[i][j] = max(
    score[i-1][j-1] + (match if seq1[i]==seq2[j] else mismatch),
    score[i-1][j] + gap,
    score[i][j-1] + gap
)

# Traceback from bottom-right to reconstruct alignment
```

#### Smith-Waterman (Local Alignment)
```python
# Similar to NW, but:
# 1. Allow score to reset to 0
# 2. Start traceback from maximum score
# 3. Stop when score reaches 0

score[i][j] = max(
    0,  # Can reset
    score[i-1][j-1] + (match if seq1[i]==seq2[j] else mismatch),
    score[i-1][j] + gap,
    score[i][j-1] + gap
)
```

### Data Structures
```python
Grid Info:
{
    'zone_labels': ['A', 'B', 'C', ...],
    'x_bins': array([0, 2, 4, ...]),
    'y_bins': array([0, 5, 10, ...]),
    'get_zone': function(x, y) -> zone_letter,
    'grid_rows': M,
    'grid_cols': N
}

Sequence Data:
{
    'ID': 'rally1-Obj0',
    'Config': 'rally1',
    'Object': 0,
    'Sequence': 'ABCDEFFG',
    'Length': 8
}
```

## Alignment with Course Content

This implementation directly supports the course concepts:

### §2.8 Sequence Analysis
- **Spatial discretization**: Zones A, B, C, ... (§2.8.1)
- **Temporal sampling**: Event-based vs equal-interval (§2.8.1)
- **Scale effects**: Grid resolution experiments (§2.8.2)
- **Sequence comparison**: Edit distance, alignment (§2.8.3)

### Pattern Mining
- **N-grams**: Most frequent k-letter patterns
- **Clustering**: Group similar sequences
- **Visualization**: Multiple views of patterns

### Comparison Methods
- **Identity**: Exact match scoring
- **Substitution**: Mismatch penalties
- **Indels**: Insertion/deletion penalties
- **Global vs Local**: Different alignment strategies

## Export Capabilities

### Sequences Export
- CSV format with ID, Config, Object, Sequence, Length
- Filename includes sampling mode and grid size
- Ready for external analysis

### N-gram Export
- Pattern frequency tables
- Can be copied from dataframe displays

### Distance Matrix
- Can be copied from heatmap
- Used for further clustering/analysis

## Best Practices

1. **Start Simple**: Use default 3×5 grid, event-based, compressed
2. **Explore Grid Sizes**: Try 2×3, 3×5, 4×7 to see pattern stability
3. **Compare Sampling**: Run both event-based and interval-based
4. **Use Compression Wisely**: ON for pattern discovery, OFF for duration analysis
5. **Align Appropriately**: Global for whole rallies, local for sub-patterns
6. **Validate Clusters**: Check if clusters make sense spatially
7. **Document Settings**: Note which parameters produced interesting results

## Troubleshooting

**Empty sequences:**
- Check time range includes data
- Verify objects exist in selected configurations
- Ensure grid covers data extent

**All sequences identical:**
- Increase grid resolution
- Check if compression is too aggressive
- Verify spatial variation in data

**Alignment looks wrong:**
- Adjust match/mismatch/gap penalties
- Try different alignment type (global vs local)
- Check sequence lengths aren't too different

**No n-grams found:**
- Use per-entity mode (not multi-entity)
- Check sequences are long enough for n-gram size
- Reduce n-gram size if sequences are short

## Further Extensions

Possible enhancements (not yet implemented):
- PrefixSpan algorithm for sequential pattern mining
- Multiple sequence alignment (3+ sequences)
- Probabilistic models (Hidden Markov Models)
- Motif discovery algorithms
- Compression ratio analysis
- Edit distance variants (Damerau-Levenshtein)

## References

### Algorithms
- Levenshtein, V. I. (1966). Binary codes capable of correcting deletions, insertions, and reversals.
- Needleman, S. B., & Wunsch, C. D. (1970). A general method applicable to the search for similarities in the amino acid sequence of two proteins.
- Smith, T. F., & Waterman, M. S. (1981). Identification of common molecular subsequences.

### Course Connection
- Aligns with spatiotemporal analysis course materials §2.8
- Implements concepts from sequence analysis theory
- Extends trajectory comparison methods

---

**Version**: 1.0  
**Date**: November 1, 2025  
**Implementation**: Complete and tested
