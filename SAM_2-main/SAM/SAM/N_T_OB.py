"""
v230209
FUNCTIONALITY
    Transforms the dataset from its original form (O) to a dataset with buffers (buf) in both x- and y-directions
EXPLANATION
    Transforms the dataset from its original form (O) to a dataset with buffers (buf) in x- and y-directions
INPUT
    The dataset
OUTPUT
    A dataset with buffers in x- and y-directions
    csv-file "N_C_PDPg_buffer_Dataset.csv"
INPUT PARAMETERS: buffer distance buffer_x and buffer_y
"""

import csv
import time
import os

# Import av to get buffer values and dataset path
try:
    import av
    buffer_x = av.buffer_x if hasattr(av, 'buffer_x') else 25
    buffer_y = av.buffer_y if hasattr(av, 'buffer_y') else 10
    
    # Get the original dataset path from environment or av
    dataset_path = os.environ.get('AV_DATASET', av.dataset_name if hasattr(av, 'dataset_name') else 'N_C_Dataset.csv')
except ImportError:
    print("⚠️ av module not found - using default buffer values")
    buffer_x = 25
    buffer_y = 10
    dataset_path = 'N_C_Dataset.csv'

print(f"🔧 Buffer transformation: buffer_x={buffer_x}, buffer_y={buffer_y}")
print(f"📂 Reading dataset: {dataset_path}")

# Start time
t_start = time.time()

# Check if input file exists
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"❌ Dataset file not found: {dataset_path}")

# Auto-detect header: Check if first row contains non-numeric data in first column
has_header = False
with open(dataset_path) as test_file:
    first_line = test_file.readline().strip()
    if first_line:
        first_value = first_line.split(',')[0]
        try:
            float(first_value)
            has_header = False
        except ValueError:
            has_header = True

print(f"📋 Header detected: {has_header}")

# Open the CSV
with open(dataset_path, 'r') as csv_file:
    # Read the CSV
    csv_reader = csv.reader(csv_file)
    
    # Skip header if present
    if has_header:
        next(csv_reader)

    # Add new lines with buffers in x and y directions
    lines = []
    for line in csv_reader:
        # Skip empty rows or rows with insufficient columns
        if not line or len(line) < 5:
            continue
            
        # Append original and buffered points for both x and y directions
        lines.append([line[0], line[1], round((float(line[2]) * 5 + 0), 2), round((float(line[3]) - buffer_x), 2), line[4]])
        lines.append([line[0], line[1], round((float(line[2]) * 5 + 1), 2), round((float(line[3]) + buffer_x), 2), line[4]])
        lines.append([line[0], line[1], round((float(line[2]) * 5 + 2), 2), line[3], line[4]])  # No buffer in x-direction
        lines.append([line[0], line[1], round((float(line[2]) * 5 + 3), 2), line[3], round((float(line[4]) - buffer_y), 2)])
        lines.append([line[0], line[1], round((float(line[2]) * 5 + 4), 2), line[3], round((float(line[4]) + buffer_y), 2)])

# Determine output path (write to results directory if available)
results_dir = os.environ.get('AV_RESULTS_DIR', os.getcwd())
output_path = os.path.join(results_dir, 'N_C_PDPg_buffer_Dataset.csv')

print(f"💾 Saving buffer dataset to: {output_path}")

# Save the new CSV file
with open(output_path, 'w', newline='') as new_csv_file:
    csv_writer = csv.writer(new_csv_file)

    # Write the new lines
    for line in lines:
        csv_writer.writerow(line)

# End and print time
print('Time elapsed for running module "N_T_OB": {:.3f} sec.'.format(time.time() - t_start))
