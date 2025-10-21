# Spatiotemporal Analysis and Modeling Tool

An interactive web application for visualizing and analyzing movement data from sports like football/soccer and tennis.

## Features

- 🏟️ **Multiple Court Types**: Switch between Football and Tennis court visualizations
- 📊 **Interactive Visualizations**: Dynamic trajectory plotting with Plotly
- 🎬 **Animation Controls**: Play/pause animations with adjustable speed
- 🔧 **Multiple Aggregation Methods**:
  - Skip frames
  - Average locations
  - Spatial generalization (Douglas-Peucker)
  - Spatiotemporal generalization
  - Smoothing average
- 📈 **Analysis Methods**:
  - Visual Exploration (IMO)
  - 2SA Method (Second-order Spatial Analysis)
  - Heat Maps
- 🎯 **Average Position Calculation**
- 📤 **Easy File Upload**: Drag and drop CSV files

## CSV Data Format

### Trajectory Data
```csv
con,tst,obj,x,y
0,0,0,64.78,18.53
0,1,0,54.26,20.68
...
```

Where:
- `con`: Configuration ID
- `tst`: Timestamp
- `obj`: Object ID
- `x,y`: Coordinates (Football: 0-110m × 0-72m, Tennis: 0-10.97m × 0-23.77m)

### Heat Map Data
```csv
pass_id,sender_id,receiver_id
0,13,17
1,17,18
...
```

## How to Use

1. Open the application
2. Upload your CSV file using the file uploader
3. Select your court type (Football or Tennis)
4. Choose your analysis method
5. Select configurations and objects to visualize
6. Adjust time range and aggregation settings
7. Explore with static view, animation, or average position mode

## Requirements

- Python 3.9+
- streamlit
- pandas
- numpy
- plotly

## Local Development

```bash
pip install -r requirements.txt
streamlit run streamlit_visualization.py
```

## License

Educational use for spatiotemporal analysis courses.

## Author

Developed for spatiotemporal analysis and modeling courses.
