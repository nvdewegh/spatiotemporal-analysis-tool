# Student Installation Guide

## How to Download and Install the Trajectory Clustering Tool

### Step 1: Download from GitHub

1. **Go to the GitHub repository**: https://github.com/nvdewegh/spatiotemporal-analysis-tool

2. **Download the code**:
   - Click the green **"Code"** button
   - Select **"Download ZIP"**
   - Save the ZIP file to your computer

3. **Extract the ZIP file**:
   - Find the downloaded ZIP file (usually in your Downloads folder)
   - Right-click and select "Extract All" (Windows) or double-click (macOS)
   - Move the extracted folder to a location you'll remember (e.g., Documents)

### Step 2: Install Python

**If you don't have Python installed:**

- **Windows/macOS**: Download Python 3.10 or higher from https://www.python.org/downloads/
- **Linux**: Python is usually pre-installed. Check with `python3 --version`

**Important**: During installation on Windows, check the box "Add Python to PATH"

### Step 3: Open Terminal/Command Prompt

- **macOS**: Press `Cmd + Space`, type "Terminal", press Enter
- **Windows**: Press `Win + R`, type "cmd", press Enter
- **Linux**: Press `Ctrl + Alt + T`

### Step 4: Navigate to the Project Folder

In the Terminal/Command Prompt, type:

```bash
cd path/to/tennis_infer_rf
```

**Example (macOS/Linux)**:
```bash
cd ~/Documents/tennis_infer_rf
```

**Example (Windows)**:
```bash
cd C:\Users\YourName\Documents\tennis_infer_rf
```

💡 **Tip**: You can drag the folder into Terminal (macOS/Linux) to auto-fill the path!

### Step 5: Create a Virtual Environment

**macOS/Linux**:
```bash
python3 -m venv .venv
```

**Windows**:
```bash
python -m venv .venv
```

⏱️ This takes 30-60 seconds.

### Step 6: Activate the Virtual Environment

**macOS/Linux**:
```bash
source .venv/bin/activate
```

**Windows**:
```bash
.venv\Scripts\activate
```

✅ You should see `(.venv)` at the beginning of your command prompt.

### Step 7: Install Required Packages

```bash
pip install -r requirements.txt
```

⏱️ This takes 2-5 minutes depending on your internet connection.

### Step 8: Launch the Application

```bash
streamlit run streamlit_deploy/streamlit_visualization.py
```

🎉 **Success!** The application should open automatically in your web browser at http://localhost:8501

---

## Quick Reference for Future Use

Every time you want to use the tool:

1. Open Terminal/Command Prompt
2. Navigate to the project folder: `cd path/to/tennis_infer_rf`
3. Activate virtual environment:
   - macOS/Linux: `source .venv/bin/activate`
   - Windows: `.venv\Scripts\activate`
4. Run the app: `streamlit run streamlit_deploy/streamlit_visualization.py`

---

## Troubleshooting

### "command not found: python3" or "python is not recognized"

**Solution**: Python is not installed or not in your PATH. Reinstall Python and check "Add to PATH" during installation.

### "No module named 'streamlit'"

**Solution**: You forgot to activate the virtual environment or install packages:
```bash
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### "Permission denied"

**Solution (macOS/Linux)**: You may need to use `python3` instead of `python`:
```bash
python3 -m venv .venv
```

### Application won't open in browser

**Solution**: Manually open your browser and go to: http://localhost:8501

---

## What You'll Need

- Python 3.10 or higher
- Internet connection (for package installation)
- ~500 MB of free disk space
- Your trajectory data in CSV or Excel format

---

## Getting Help

If you encounter problems:
1. Check this troubleshooting guide
2. Read the main README.md file
3. Check the Documentation tab in the application (Step 7)
4. Ask your instructor for help

---

## Data Format

Your CSV file should have at least 3 columns:
1. Trajectory ID (which trajectory the point belongs to)
2. X coordinate
3. Y coordinate

**Example**:
```csv
trajectory_id,x,y
1,10.5,20.3
1,11.2,21.1
2,15.1,18.2
2,15.8,19.0
```

---

**Ready to start analyzing trajectories!** 🚀
