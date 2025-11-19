# Load the necessary libraries
from matplotlib import cm  # Colormaps
import matplotlib.patches as patches  # For drawing shapes
import matplotlib.pyplot as plt  # Plotting library
import matplotlib as mpl  # Matplotlib settings
from matplotlib.lines import Line2D
import os
import pandas as pd
import time

# Import custom attributes
import av

# Start time
t_start = time.time()

# Set the default unit of length to centimeters
mpl.rcParams['figure.dpi'] = 2.54

# Use the already-loaded dataset from av module
df = av.Df_dataset.copy()

# Get the unique values of the first column
configurations = df['conID'].unique()

# Create a list of colors
if av.poi == 3:
    colors = ['black', 'blue', 'magenta']
else:
    colors = [plt.cm.cividis(i/av.poi) for i in range(av.poi)]

# Create the scatterplot, including arrows
for config in configurations:
    config_data = df[df['constant'] == config]  # Get the data for the current configuration
    
    # Get the x/y-values
    x = config_data['x']
    y = config_data['y']
    
    # Calculate scaling factor

    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    scaling_factor = min(x_range, y_range) / 10
    #plt.figure(figsize=(12, 8))  # Set the size of the figure
    plt.figure(figsize=(18, 18), dpi=100.0)  # Set the size of the figure
    
    # Set the limits of the axes
    plt.xlim(av.min_boundary_x, av.max_boundary_x)
    plt.ylim(av.min_boundary_y, av.max_boundary_y)
    
    # Set the labels of the axes
    plt.xlabel('X-Axis (m)', fontsize=30, fontname='monospace')  
    plt.ylabel('Y-Axis (m)', fontsize=30, fontname='monospace')
    ax = plt.gca()  # Get the current axes
    # Draw full singles tennis court
    # Outer boundary (8.23 m x 23.77 m), bottom-left at (0, 0)
    court_boundary = patches.Rectangle((0, 0), 8.23, 23.77, linewidth=2, edgecolor='green', facecolor='none')
    ax.add_patch(court_boundary)

    # Net line (horizontal line across middle of the court)
    ax.add_line(Line2D([0, 8.23], [11.885, 11.885], color='green', linewidth=1))

    # Center service line (vertical line through middle of service boxes)
    ax.add_line(Line2D([4.115, 4.115], [5.485, 18.285], color='green', linewidth=1, linestyle='dotted'))

    # Service box horizontal lines (top and bottom of service boxes)
    ax.add_line(Line2D([0, 8.23], [5.485, 5.485], color='green', linewidth=1, linestyle='dotted'))
    ax.add_line(Line2D([0, 8.23], [18.285, 18.285], color='green', linewidth=1, linestyle='dotted'))

    ax.tick_params(axis='both', labelsize=30, labelcolor='black')  # Set the size and color of the tick labels
    
    # Check if there's only one timestamp
    if av.tst == 1:
        #colors = ['blue', 'red', 'green', 'yellow', 'orange', 'purple']
        #plt.scatter(x, y, color=colors[i % 10], s=100)  # s is the marker size
        for point_index, (x_val, y_val) in enumerate(zip(x, y)):
            color_to_use = colors[point_index % len(colors)]
            plt.scatter(x_val, y_val, color=color_to_use, s=200)  # s is the marker size
    else:
        vector_index = 0
        # Add vectors between points
        for p in range(av.poi):  # for each point
            for i in range(av.tst - 1):  # for each interval
                x1 = x.iloc[p + i * av.poi]
                y1 = y.iloc[p + i * av.poi]
                x2 = x.iloc[p + i * av.poi + av.poi]
                y2 = y.iloc[p + i * av.poi + av.poi]
                # Use the custom color list to assign a un@ique color to each point
                color_to_use = colors[p % len(colors)]
                plt.arrow(x1, y1, x2 - x1, y2 - y1, length_includes_head=True, head_width=(((x.max()+1)-(x.min()-1))/40), head_length=(((x.max()+1)-(x.min()-1))/20), linewidth=10, color=color_to_use)
                
                # Add label for the first timestamp of each point
                if i == 0:
                    plt.text(x1, y1, f'p{p}', fontsize=30, ha='right')
                
                vector_index += 1
    
    """
    # Draw the tennis pitch
    plt.hlines(0, -5.485, 5.485, linewidth=0.5, colors='g', linestyles='solid')
    plt.hlines(6.4, -4.115, 4.115, linewidth=0.5, colors='g', linestyles='solid')
    plt.hlines(-6.4, -4.115, 4.115, linewidth=0.5, colors='g', linestyles='solid')
    plt.hlines(11.885, -5.485, 5.485, linewidth=0.5, colors='g', linestyles='solid')
    plt.hlines(-11.885, -5.485, 5.485, linewidth=0.5, colors='g', linestyles='solid')
    plt.vlines(0, -6.4, 6.4, linewidth=0.5, colors='g', linestyles='solid')
    plt.vlines(-5.485, -11.885, 11.885, linewidth=0.5, colors='g', linestyles='solid')
    plt.vlines(5.485, -11.885, 11.885, linewidth=0.5, colors='g', linestyles='solid')
    plt.vlines(-4.115, -11.885, 11.885, linewidth=0.5, colors='g', linestyles='solid')
    plt.vlines(4.115, -11.885, 11.885, linewidth=0.5, colors='g', linestyles='solid')
    """


    plt.title("Configuration {}".format(config),fontname="monospace", fontsize=40)
    output_folder = '/Users/olivier/Documents/Thesis/Wimbeldon_23_D_A_1080p_25fps/output/configuraties'  # <- Change to your path
    os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

    file_name = os.path.join(output_folder, "N_C_Csa{}.png".format(config))
    #file_name = "N_C_Csa{}.png".format(config) # csa from configuration static absolute
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()  # close the figure to release memory



# End and print time
print('Time elapsed for running module "N_VA_StaticAbsolute": {:.3f} sec.'.format(time.time() - t_start))