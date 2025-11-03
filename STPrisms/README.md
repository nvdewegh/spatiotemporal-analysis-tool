# Space-Time Prisms: Interactive Vibe Coding Environment

## Overview

An interactive environment for uncertainty-based trajectory analysis combining analytical modeling with qualitative "vibe coding" interpretation.

Based on Arthur Jansen's PhD thesis: *Uncertainty-based queries in trajectory sample databases*.

## Features

### Core Analytical Models
- **Space-Time Prisms**: Classical, circular, and elliptical prisms
- **Trajectory Samples**: Upload or create movement scenarios
- **Alibi Queries**: Intersection analysis for multiple objects
- **Visit Probability**: Spatiotemporal probability computation
- **Anchor Uncertainty**: Error ellipses and fuzzy regions

### Vibe Coding Layer
Interpretive overlay providing qualitative annotations:
- **Epistemic Stance**: certainty, uncertainty, possibility, impossibility
- **Constraint Level**: free, constrained, blocked
- **Evidence Tone**: strong, weak, conflicting
- **Perspective**: human-centered, data-centered, algorithmic
- **Narrative Mood**: exploratory, formal, applied, speculative

### Visualizations
- 2D and 3D space-time prisms
- Potential path areas (PPAs)
- Uncertainty regions
- Visit probability heatmaps
- Interactive time sliders and filtering

## Installation

```bash
cd STPrisms
pip install -r requirements.txt
```

## Usage

```bash
streamlit run stprisms_app.py
```

## Components

1. **Scenario Manager**: Upload/create trajectory datasets
2. **Uncertainty Modelling**: Compute prisms, chains, and queries
3. **Visualization Engine**: 2D/3D rendering with Plotly
4. **Vibe Coding Layer**: Qualitative annotation system
5. **Narrative Workspace**: Interpretive notes and annotations
6. **Export Module**: HTML, PDF, JSON outputs

## Data Formats

- CSV: `timestamp, object_id, x, y, [vmax], [error]`
- GeoJSON: Standard trajectory format
- GPX: GPS tracks

## Example Use Case

**Scenario**: Assess if two tracked individuals could have met.

1. Upload two trajectories with speed bounds
2. System computes space-time prisms
3. Alibi query finds overlap region
4. Automatic vibe assignment: "uncertainty", "weak evidence"
5. Add interpretation: "Possible meeting under optimistic assumption"
6. Export visual + qualitative report

## Research Applications

- Hybrid GeoAI reasoning (quantitative + qualitative)
- Uncertainty communication
- Comparative vibe analysis
- Didactic demonstrations
- Cross-disciplinary spatial reasoning

## License

Academic research tool - MIT License
