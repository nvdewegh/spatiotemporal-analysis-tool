# Trajectory Clustering Analysis Tool

A comprehensive Streamlit-based web application for analyzing and clustering spatiotemporal trajectory data using hierarchical clustering methods.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Distance Metrics](#distance-metrics)
- [Troubleshooting](#troubleshooting)
- [Requirements](#requirements)

## 🎯 Overview

This tool provides a complete workflow for trajectory clustering analysis, from data loading to result export. It's designed for researchers and students working with spatiotemporal data, such as GPS trajectories, movement patterns, or time-series spatial data.

## ✨ Features

### 7-Step Analysis Workflow

1. **Data Loading & Infrastructure**: Import trajectory data from CSV or Excel files
2. **Clustering Setup**: Initialize hierarchical clustering algorithms
3. **Distance Computation**: Choose from three distance metrics:
   - Feature-based distance
   - Spatial (Chamfer) distance
   - Dynamic Time Warping (DTW)
4. **Dendrogram Visualization**: View hierarchical clustering structure and select optimal number of clusters
5. **Analysis Tools**:
   - Multidimensional Scaling (MDS) visualization
   - Trajectory similarity search
   - Silhouette analysis for cluster quality
6. **Cluster Visualizations**:
   - 2D spatial plots
   - 3D spatiotemporal views
   - Interactive cluster comparison
7. **Export & Summary**:
   - Download results as CSV files
   - View comprehensive analysis summaries
   - Access complete documentation

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- macOS, Linux, or Windows

### Step-by-Step Installation

1. **Clone or download this repository** to your computer

2. **Open Terminal** (macOS/Linux) or Command Prompt (Windows) and navigate to the project folder:
   ```bash
   cd path/to/tennis_infer_rf
   ```

3. **Create a virtual environment**:
   ```bash
   python3 -m venv .venv
   ```

4. **Activate the virtual environment**:
   - macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```
   - Windows:
     ```bash
     .venv\Scripts\activate
     ```

5. **Install required packages**:
   ```bash
   pip install streamlit pandas numpy plotly scikit-learn umap-learn matplotlib pyarrow
   ```

## 🎬 Quick Start

1. **Activate the virtual environment** (if not already activated):
   ```bash
   source .venv/bin/activate  # macOS/Linux
   # or
   .venv\Scripts\activate     # Windows
   ```

2. **Launch the application**:
   ```bash
   streamlit run streamlit_deploy/streamlit_visualization.py
   ```

3. **Open your web browser** and navigate to:
   ```
   http://localhost:8501
   ```

4. The application will open automatically in your default browser!

## 📖 Usage Guide

### Data Format Requirements

Your trajectory data should be in **CSV or Excel format** with the following structure:

- **First column**: Trajectory ID (which trajectory each point belongs to)
- **Second column**: X coordinate (spatial dimension)
- **Third column**: Y coordinate (spatial dimension)
- **Optional columns**: Additional features (ignored by the tool)

**Example CSV:**
```csv
trajectory_id,x,y
1,10.5,20.3
1,11.2,21.1
1,12.0,22.5
2,15.1,18.2
2,15.8,19.0
```

### Workflow Steps

#### Step 1-2: Data Loading

1. Click **"📁 Upload your trajectory data"**
2. Select your CSV or Excel file
3. Verify the data preview shows correctly
4. Click **"✅ Confirm Data Upload"**

#### Step 3: Compute Distance Matrix

Choose one of three distance metrics:

- **Features**: Best for comparing overall trajectory characteristics (length, speed, direction changes)
- **Spatial (Chamfer)**: Best for comparing spatial shapes
- **DTW**: Best for comparing temporal patterns and sequences

Click the corresponding **"🔄 Compute"** button and wait for processing.

#### Step 4: View Dendrogram & Assign Clusters

1. The dendrogram shows how trajectories group together
2. Use the **slider** to select number of clusters (1-20)
3. Click **"🎯 Assign Clusters"**
4. View the cluster assignment table

#### Step 5: Analyze Results

Explore three analysis tools:

- **MDS Plot**: Visualize trajectory similarities in 2D space
- **Similarity Search**: Find trajectories similar to a selected one
- **Silhouette Analysis**: Evaluate cluster quality (higher scores = better clusters)

#### Step 6: Visualize Clusters

Generate three types of visualizations:

- **2D Spatial View**: See trajectories colored by cluster
- **3D Spatiotemporal**: Add time dimension to see evolution
- **Cluster Comparison**: Compare selected clusters side-by-side or overlayed

#### Step 7: Export & Summary

- Download 4 types of CSV files:
  - Cluster assignments
  - Distance matrix
  - Cluster statistics
  - Analysis configuration
- View comprehensive summary with charts
- Read complete documentation

## 📊 Distance Metrics

### Feature-Based Distance

Extracts statistical features from each trajectory:
- Total length
- Average/max speed
- Displacement
- Direction changes (sinuosity)
- Spatial extent (bounding box)

Best for: Comparing movement characteristics regardless of exact paths.

### Spatial (Chamfer) Distance

Measures spatial shape similarity by computing minimum distances between trajectory points.

Best for: Comparing spatial shapes when timing doesn't matter.

### Dynamic Time Warping (DTW)

Aligns trajectories in time to find optimal matching, accounting for different speeds.

Best for: Comparing temporal sequences and patterns.

## 🔧 Troubleshooting

### Issue: "command not found: streamlit"

**Solution**: Make sure you activated the virtual environment:
```bash
source .venv/bin/activate  # macOS/Linux
```

### Issue: Application won't start

**Solution**: Check that all packages are installed:
```bash
pip install streamlit pandas numpy plotly scikit-learn umap-learn matplotlib pyarrow
```

### Issue: "No trajectory data available" warning

**Solution**: You need to complete Step 3 (compute distance matrix) before proceeding to Steps 4-7.

### Issue: Dendrogram doesn't appear

**Solution**: Ensure your data file has at least 2 trajectories and that the distance matrix computation completed successfully.

### Issue: Out of memory error

**Solution**: Large datasets (>1000 trajectories) may require significant RAM. Try:
- Using feature-based distance (faster)
- Reducing the dataset size
- Using a computer with more RAM

## 📦 Requirements

```
streamlit>=1.20.0
pandas>=1.5.0
numpy>=1.23.0
plotly>=5.11.0
scikit-learn>=1.2.0
umap-learn>=0.5.3
matplotlib>=3.6.0
pyarrow>=10.0.0
scipy>=1.10.0  (installed automatically with scikit-learn)
```

## 💡 Tips for Students

1. **Start small**: Test with a small dataset (10-50 trajectories) first
2. **Save your work**: Use the export feature to download results
3. **Experiment**: Try different distance metrics to see which works best for your data
4. **Interpret carefully**: Check silhouette scores to validate cluster quality
5. **Document**: The tool generates configuration files - save these with your results

## 🎓 Educational Use

This tool is designed for teaching and learning spatiotemporal data analysis. It provides:
- Interactive visualizations for better understanding
- Multiple analysis methods to compare approaches
- Step-by-step workflow for systematic analysis
- Export capabilities for reports and presentations

## 📄 License

This tool is provided for educational and research purposes.

## 🆘 Support

If you encounter issues:
1. Check the Troubleshooting section above
2. Review the in-app Documentation tab (Step 7)
3. Verify your data format matches the requirements
4. Contact your instructor for assistance

## 🎉 Getting Started Checklist

- [ ] Python 3.10+ installed
- [ ] Virtual environment created
- [ ] All packages installed
- [ ] Application launches successfully
- [ ] Sample data file prepared
- [ ] Ready to analyze trajectories!

---

**Happy Clustering!** 🚀
