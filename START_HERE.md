# SAMTool – getting started

SAMTool is a Streamlit web application for exploring and analysing spatiotemporal trajectory data (e.g. tennis, football or micro-traffic movements). It offers the following analysis methods, selectable in the sidebar:

- **Visual Exploration** – static and animated trajectories, time point view, average positions
- **Clustering** – hierarchical clustering (feature-based, Chamfer and DTW distances), dendrogram, MDS
- **Association Rules**
- **Sequence Analysis**
- **PDP Analysis** – Point Descriptor Precedence (fundamental, buffer and rough variants)
- **Outlier Detection**
- **Heat Maps**

This repository always contains the most recent version of the code – the same version that runs online at **https://samtool.streamlit.app**. If you only want to try the tool, that link is enough: no installation needed.

Please do not change this repository directly: send suggestions or problems to the author. You are welcome to experiment with your own copy (download or fork).

> The older `README.md` and the other `.md` files describe individual features in more depth, but some parts are outdated (notably the data format). Where they differ, follow this file.

---

## 1. Run it on your own computer

You need **Python 3.10 or newer**. Download the code (**Code → Download ZIP** on GitHub, or `git clone https://github.com/nvdewegh/spatiotemporal-analysis-tool.git`), open a terminal in the downloaded folder and run:

```bash
python3 -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_deploy/streamlit_visualization.py
```

The app opens in your browser (usually at http://localhost:8501). Next time, only the `source …` and `streamlit run …` lines are needed. Stop the app with **Ctrl+C** in the terminal.

## 2. Prepare your data

Upload a CSV file via the sidebar. The standard format is one row per object per timestamp, with these five columns **in this order**:

| Column | Meaning |
|---|---|
| `config` | Configuration / episode ID (e.g. a rally or traffic scene) |
| `tst` | Timestamp |
| `obj` | Object ID (e.g. player, ball, road user) |
| `x` | x coordinate |
| `y` | y coordinate |

Example:

```csv
config,tst,obj,x,y
1,0,1,3.20,0.50
1,0,2,4.10,22.80
1,1,1,3.35,0.90
1,1,2,4.05,22.10
```

A sixth column is optional: either a configuration name or an events column.

**Timestamps** are shown as given in the data (e.g. `TST 12`), with no unit assumed. You need to know your data's temporal resolution yourself (e.g. 24 Hz).

**Court type** (sidebar): choose *No selection* for data without a pitch (such as micro-traffic), or *Tennis* / *Football* to draw that court in the plots.

## 3. Where things are in the code

```
streamlit_deploy/
├── streamlit_visualization.py   main app: sidebar, page layout, visual exploration, court drawing
├── modules/
│   ├── utils.py                 data loading (load_data) and helper functions
│   ├── common.py                shared chart rendering
│   ├── clustering.py            clustering
│   ├── association_rules.py     association rules
│   ├── sequence_analysis.py     sequence analysis
│   ├── pdp_analysis.py          PDP analysis
│   └── outlier_detection.py     outlier detection
└── tests/                       tests (run from streamlit_deploy/: python -m pytest tests)
requirements.txt                 Python packages needed
```

## 4. Making changes

1. Edit the relevant file (see above).
2. With the app running, Streamlit notices the change. Click **Rerun** in the browser (or press **R**) to see the result.
3. If something breaks, the error appears in the browser and in the terminal.

The files in the top-level folder (`association_rules_functions.py`, `insert_association_rules.py`, …), `PDP/` and `STPrisms/` are older helper scripts. The app does not use them.

## 5. Common problems

- **`command not found: streamlit`** – the virtual environment is not active. Run the `source .venv/bin/activate` line first.
- **`ModuleNotFoundError`** – run `pip install -r requirements.txt` again inside the active environment.
- **Data does not load correctly** – check that the columns follow the order in section 2.
