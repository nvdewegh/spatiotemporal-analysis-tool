# Sequence Analysis Implementation Summary

## Implementation Complete

I have successfully implemented a comprehensive **Sequence Analysis** feature for your spatiotemporal analysis tool, directly aligned with your course text on sequence analysis methods.

## What Was Implemented

### Core Functionality

1. **Spatial Discretization** (§2.8.1 from course)
   - Divides court into M×N grid of symbolic zones (A, B, C, ...)
   - Configurable grid resolution (2-10 rows/columns)
   - Maps (x, y) coordinates to zone letters
   - Visual overlay shows grid on court

2. **Temporal Sampling** (§2.8.1)
   - **Event-based**: One token per data point (hit/bounce)
   - **Equal-interval**: Fixed time steps (Δt = 0.1-2.0s)
   - Configurable sampling rate

3. **Sequence Building**
   - **Per-entity**: Separate sequences for each object (ball, players)
   - **Multi-entity**: Combined tokens showing all entities per moment
   - **Run-length compression**: AAABBB → AB (optional)

4. **Distance Metrics**
   - **Levenshtein distance**: Edit distance between sequences
   - **Normalized Levenshtein**: Scaled by sequence length
   - **Distance matrix**: Pairwise comparisons with heatmap visualization

5. **Sequence Alignment** (§2.8.3)
   - **Global (Needleman-Wunsch)**: Align entire sequences
   - **Local (Smith-Waterman)**: Find similar sub-sequences
   - Configurable scoring (match/mismatch/gap penalties)
   - Color-coded visualization (green=match, orange=mismatch, red=gap)

6. **Pattern Mining**
   - **N-gram extraction**: Find frequent k-letter patterns
   - **Frequency analysis**: Most common patterns across rallies
   - **Per-sequence breakdown**: Pattern analysis for each rally

7. **Clustering**
   - **Hierarchical clustering**: Group similar sequences
   - **Adjustable clusters**: 2-10 clusters
   - **Cluster statistics**: Size, average length, etc.

8. **Visualizations**
   - Distance matrix heatmap
   - Aligned sequence display
   - N-gram frequency charts
   - Spatial grid overlay
   - Zone visit heatmap

## User Interface

The feature is accessible via:
```
Main Menu → Analysis Method → "Sequence Analysis"
```

### Four Analysis Tabs

1. **Distance Matrix**: Compare all sequences, perform clustering
2. **Pairwise Alignment**: Align two sequences globally or locally
3. **N-gram Patterns**: Discover frequent movement patterns
4. **Spatial Grid View**: Visualize zone definitions and coverage

## Educational Value

This implementation supports several course concepts:

- **Scale Effects** (§2.8.2): Students can experiment with different grid resolutions and temporal samplings to see how patterns change
- **Comparison Methods**: Both identity (exact match) and substitution/indel operations
- **Global vs Local Alignment**: Students can see when each is appropriate
- **Pattern Discovery**: Find recurring tactical patterns
- **Symbolic Abstraction**: Transform continuous space-time to discrete tokens

## Example Student Exercise

Using the tool, students can:

1. **Build sequences** for tennis rallies with different grid sizes (2×3, 3×5, 4×7)
2. **Compare** event-based vs. equal-interval sampling (Δt = 0.2s, 0.5s, 1.0s)
3. **Discover patterns** using n-gram analysis (what are the most common 2-grams? 3-grams?)
4. **Cluster rallies** based on sequence similarity
5. **Align rallies** to find common sub-sequences
6. **Discuss scale**: How does grid resolution affect discovered patterns?
7. **Compression effects**: Compare AB vs. AAABBB—what information is lost/gained?

## Technical Details

### Algorithms Implemented

- **Levenshtein Distance**: O(mn) dynamic programming
- **Needleman-Wunsch**: O(mn) DP with traceback
- **Smith-Waterman**: O(mn) DP with local max
- **Hierarchical Clustering**: Average linkage (compatible with precomputed distances)

### Data Flow

```
Raw data (x, y, t)
    ↓
Spatial discretization (grid zones)
    ↓  
Temporal sampling (events or intervals)
    ↓
Token sequences (ABCDEFF...)
    ↓
Optional compression (ABCDEF...)
    ↓
Analysis (distances, alignments, patterns)
```

## Documentation

Complete documentation provided in `SEQUENCE_ANALYSIS_GUIDE.md`:
- Feature overview
- Usage workflow
- Algorithm explanations
- Example use cases
- Best practices
- Troubleshooting
- Technical implementation details

## How to Use

1. **Start app**: `streamlit run streamlit_deploy/streamlit_visualization.py`
2. **Upload data**: CSV files with trajectory data
3. **Select**: Analysis Method → "Sequence Analysis"
4. **Configure**: Grid size, sampling mode, compression
5. **Select data**: Rallies and objects to analyze
6. **Explore tabs**: Distance matrix, alignment, patterns, grid view
7. **Export**: Download sequences as CSV

## Key Features

- Spatial discretization with configurable grids
- Event-based and equal-interval sampling
- Per-entity and multi-entity sequences
- Run-length compression
- Levenshtein distance (regular and normalized)
- Global alignment (Needleman-Wunsch)
- Local alignment (Smith-Waterman)
- N-gram pattern mining
- Hierarchical clustering
- Interactive visualizations
- CSV export
- Complete documentation

## Alignment with Requirements

Your request asked for:
- Spatial discretization into zones
- Event-based or equal-interval sequences
- Per-entity sequences (ball_seq, p1_seq, p2_seq)
- Optional compression (AB vs AABBB)
- Edit distances (Levenshtein)
- Global/local alignment
- Pattern mining (n-grams)
- Comparison between rallies
- Scale effects analysis
- Student exercise support

**All requirements met!**

## Files Changed

1. `streamlit_deploy/streamlit_visualization.py` (+549 lines)
   - Added sequence analysis functions
   - Added Sequence Analysis tab
   - Fixed clustering to use average linkage

2. `SEQUENCE_ANALYSIS_GUIDE.md` (new file)
   - Complete user documentation
   - Algorithm explanations
   - Usage examples

3. `SEQUENCE_ANALYSIS_SUMMARY.md` (this file)
   - Implementation summary
   - Feature list

## Issues Fixed

- Initial error with Ward linkage on precomputed distances
- Changed to average linkage (compatible with distance matrices)
- All functions tested and working

## Status

**COMPLETE AND TESTED**

The feature is fully implemented, documented, committed to git, and ready for use in your course!

Students can now:
- Build symbolic sequences from trajectory data
- Compare rallies using edit distance and alignment
- Discover common movement patterns
- Explore scale effects
- Export results for further analysis

---

**Implementation Date**: November 1, 2025  
**Version**: 1.0  
**Status**: Production Ready
