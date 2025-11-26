
import os

file_path = '/Users/nicovandeweghe/MIJNJUISTEDATA/NVdW/z/01010100 OW Omvang Cursussen(titularis)/1Lop/(VanAJ2021)SpatiotemporalAnalysisAndModelling(Code C004177)/TennisprojecT/tennis_infer_rf/streamlit_deploy/streamlit_visualization.py'

with open(file_path, 'r') as f:
    lines = f.readlines()

start_marker = 'heatmap_size = max(500, min(1200, n_configs * cell_size))'
end_marker = '# Create title indicating distance type'

start_index = -1
end_index = -1

for i, line in enumerate(lines):
    if start_marker in line:
        start_index = i
        break

if start_index == -1:
    print("Start marker not found")
    exit(1)

for i in range(start_index, len(lines)):
    if end_marker in line: # Wait, 'line' variable is from previous loop? No, I need to use lines[i]
        pass
    if end_marker in lines[i]:
        end_index = i
        break

if end_index == -1:
    print("End marker not found")
    exit(1)

print(f"Indenting lines {start_index+1} to {end_index}")

# Indent by 4 spaces
for i in range(start_index, end_index):
    lines[i] = '    ' + lines[i]

with open(file_path, 'w') as f:
    f.writelines(lines)

print("Done")
