# Tennis Court Visualization with Player/Ball Trajectories
# =========================================================
# Visualizes configurations on a tennis court using Plotly
# Based on N_VA_StaticAbsolute but adapted for tennis analysis

import os
import pandas as pd
import time
import plotly.graph_objects as go

# Import custom modules
import av
from tennis_court_draw import create_tennis_court

# Start time
t_start = time.time()

# Use the already-loaded dataset from av module
df = av.Df_dataset.copy()
# Reset column names to numeric indices for compatibility
df.columns = range(len(df.columns))

# Get the unique configurations
configurations = df[0].unique()

# Define colors for different players/ball
# Using distinct colors that show well on green background
if av.poi == 2:
    colors = ['yellow', 'cyan']  # Player 1 (yellow), Player 2 (cyan)
    labels = ['Speler 1', 'Speler 2']
elif av.poi == 3:
    colors = ['yellow', 'cyan', 'white']  # Player 1, Player 2, Ball
    labels = ['Speler 1', 'Speler 2', 'Bal']
else:
    # Generate colors for any number of points
    import plotly.express as px
    color_sequence = px.colors.qualitative.Set3
    colors = [color_sequence[i % len(color_sequence)] for i in range(av.poi)]
    labels = [f'Punt {i}' for i in range(av.poi)]

# Create visualizations for each configuration
for config in configurations:
    config_data = df[df[0] == config]  # Get data for current configuration
    
    # Get x/y coordinates
    x = config_data[3]
    y = config_data[4]
    
    # Create tennis court as base
    fig = create_tennis_court()
    
    # Check if there's only one timestamp
    if av.tst == 1:
        # Single timestamp: just show points
        for point_index in range(av.poi):
            x_val = x.iloc[point_index]
            y_val = y.iloc[point_index]
            
            fig.add_trace(
                go.Scatter(
                    x=[x_val],
                    y=[y_val],
                    mode='markers+text',
                    marker=dict(
                        size=15,
                        color=colors[point_index % len(colors)],
                        symbol='circle',
                        line=dict(color='black', width=2)
                    ),
                    text=[f'p{point_index}'],
                    textposition='top center',
                    textfont=dict(size=14, color='white'),
                    name=labels[point_index % len(labels)],
                    hovertemplate=f'{labels[point_index % len(labels)]}<br>x: %{{x:.2f}}m<br>y: %{{y:.2f}}m<extra></extra>'
                )
            )
    else:
        # Multiple timestamps: show trajectories with arrows
        for p in range(av.poi):  # For each point (player/ball)
            x_trajectory = []
            y_trajectory = []
            
            # Collect all positions for this point across timestamps
            for i in range(av.tst):
                idx = p + i * av.poi
                x_trajectory.append(x.iloc[idx])
                y_trajectory.append(y.iloc[idx])
            
            # Add trajectory line
            fig.add_trace(
                go.Scatter(
                    x=x_trajectory,
                    y=y_trajectory,
                    mode='lines',
                    line=dict(
                        color=colors[p % len(colors)],
                        width=3,
                        dash='solid'
                    ),
                    name=f'{labels[p % len(labels)]} traject',
                    showlegend=True,
                    hoverinfo='skip'
                )
            )
            
            # Add arrow markers for each movement segment
            for i in range(av.tst - 1):
                x1 = x_trajectory[i]
                y1 = y_trajectory[i]
                x2 = x_trajectory[i + 1]
                y2 = y_trajectory[i + 1]
                
                # Calculate arrow direction
                dx = x2 - x1
                dy = y2 - y1
                
                # Add arrow annotation
                fig.add_annotation(
                    x=x2,
                    y=y2,
                    ax=x1,
                    ay=y1,
                    xref='x',
                    yref='y',
                    axref='x',
                    ayref='y',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.5,
                    arrowwidth=2,
                    arrowcolor=colors[p % len(colors)],
                    opacity=0.8
                )
            
            # Add position markers at each timestamp
            fig.add_trace(
                go.Scatter(
                    x=x_trajectory,
                    y=y_trajectory,
                    mode='markers+text',
                    marker=dict(
                        size=12,
                        color=colors[p % len(colors)],
                        symbol='circle',
                        line=dict(color='black', width=2)
                    ),
                    text=[f'p{p}' if i == 0 else '' for i in range(len(x_trajectory))],
                    textposition='top center',
                    textfont=dict(size=14, color='white'),
                    name=f'{labels[p % len(labels)]} posities',
                    showlegend=False,
                    hovertemplate=f'{labels[p % len(labels)]}<br>Tijd: t%{{pointIndex}}<br>x: %{{x:.2f}}m<br>y: %{{y:.2f}}m<extra></extra>'
                )
            )
    
    # Update layout with configuration title
    fig.update_layout(
        title=dict(
            text=f'Configuratie {config} - Tennis Baan Analyse',
            font=dict(size=20, color='white'),
            x=0.5,
            xanchor='center'
        ),
        paper_bgcolor='#1a1a1a',  # Dark background around the court
        font=dict(color='white')
    )
    
    # Save the figure
    file_name = f"Tennis_Config_{config}.html"
    results_dir = os.environ.get('AV_RESULTS_DIR', os.getcwd())
    module_dir = os.path.join(results_dir, 'TennisCourt')
    os.makedirs(module_dir, exist_ok=True)
    out_path = os.path.join(module_dir, file_name)
    
    fig.write_html(out_path)
    
    # Also save as PNG if kaleido is available
    try:
        png_path = os.path.join(module_dir, f"Tennis_Config_{config}.png")
        fig.write_image(png_path, width=800, height=1400)
    except Exception as e:
        # Kaleido not installed or other error - skip PNG export
        pass

# End and print time
print('Time elapsed for running module "N_VA_TennisCourt": {:.3f} sec.'.format(time.time() - t_start))
print(f'Created {len(configurations)} tennis court visualizations in {module_dir}')
