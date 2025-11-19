# N_Moving_Objects.py
# ------------------------------------------------------------
# Unified runner for PDP variants (fundamental, buffer, rough, buffer+rough)
# Works with datasets that have 5 cols: conID, tstID, poiID, x, y
# ------------------------------------------------------------

# Notes & TODOs from your original file preserved:
# N_VA_DynamicAbsolute is off (had an error; fix later).
# N_VA_ClusterMap is off (had an error; fix later).
# Keep rough/buffer experiments clear in titles/outputs.
# Page numbers in reports, etc.

import av  # Import all variables (settings, paths, toggles)
import importlib
import numpy as np
import pandas as pd
import shutil
import time
import os
from tqdm import tqdm

t_start = time.time()

# Flag to prevent duplicate timing prints on reload
_already_printed = getattr(av, '_moving_objects_printed', False)

# Conditionally import visual modules (based on av toggles)
# Only import modules that exist in SAM folder
if av.N_VA_StaticAbsolute == 1:
    try:
        import N_VA_StaticAbsolute
    except ImportError:
        print("⚠️ N_VA_StaticAbsolute not found in SAM folder")

if av.N_VA_TennisCourt == 1:
    try:
        import N_VA_TennisCourt
    except ImportError:
        print("⚠️ N_VA_TennisCourt not found in SAM folder")


# ---------------- Helper: robust CSV loader (5 columns) ----------------
def read_config_csv(path):
    """
    Load config CSV that has 5 cols: conID, tstID, poiID, x, y

    Returns:
      Df_dataset (5 numeric cols),
      L_dataset (list of rows),
      A_dataset (np.float32 array),
      con, tst, poi (counts inferred as max+1)
    """
    # Auto-detect header: Check if first row contains non-numeric data in first column
    has_header = False
    with open(path) as test_file:
        first_line = test_file.readline().strip()
        if first_line:
            first_value = first_line.split(',')[0]
            try:
                float(first_value)
                has_header = False
            except ValueError:
                has_header = True
    
    # Read with or without header
    if has_header:
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, header=None)
    
    ncols = df.shape[1]

    if ncols == 5:
        df.columns = ['conID', 'tstID', 'poiID', 'x', 'y']
    else:
        raise ValueError(f"Unexpected number of columns ({ncols}) in {path}. Expected 5.")

    # Numeric frame for computations - convert to numeric types
    df_num = df[['conID', 'tstID', 'poiID', 'x', 'y']].copy()
    for c in ['conID', 'tstID', 'poiID', 'x', 'y']:
        df_num[c] = pd.to_numeric(df_num[c], errors='coerce')

    L_dataset = df_num.values.tolist()
    A_dataset = df_num.to_numpy(dtype=np.float32)

    con = int(df_num['conID'].max()) + 1 if len(df_num) else 0
    tst = int(df_num['tstID'].max()) + 1 if len(df_num) else 0
    poi = int(df_num['poiID'].max()) + 1 if len(df_num) else 0

    return df_num, L_dataset, A_dataset, con, tst, poi


# ---------------- Optional: legacy wrapper kept minimal ----------------
def SetDataForPDPType(data_filename, D_point_mapping, curr_point_id, window_length_tst):
    """
    Legacy function signature preserved (mapping args unused here).
    """
    Df_dataset, L_dataset, A_dataset, con, tst, poi = read_config_csv(data_filename)
    results_dir = os.environ.get('AV_RESULTS_DIR', os.getcwd())
    os.makedirs(results_dir, exist_ok=True)
    # Df_dataset.to_csv removed - not necessary for output
    if av.window_length_tst > tst:
        print("ERROR IN VALUE OF VARIABLE: window_length_tst > tst")
    return Df_dataset, A_dataset, con, tst, poi


# ---------------- PDP: Fundamental ----------------
if av.PDPg_fundamental == 1:
    print("\n" + "="*60)
    print("🚀 STARTING PDP: FUNDAMENTAL")
    print("="*60)
    av.PDPg_fundamental_active = 1

    # Copy the active dataset to results directory
    results_dir = os.environ.get('AV_RESULTS_DIR', os.getcwd())
    os.makedirs(results_dir, exist_ok=True)
    pdp_dataset_path = os.path.join(results_dir, "N_C_PDPg_fundamental_Dataset.csv")
    
    # Only copy if source and destination are different
    if os.path.abspath(av.dataset_name) != os.path.abspath(pdp_dataset_path):
        shutil.copyfile(av.dataset_name, pdp_dataset_path)
    
    av.dataset_name = pdp_dataset_path
    av.dataset_name_exclusive = os.path.splitext(av.dataset_name)[0]

    # Robust read (handles 5 columns)
    (
        av.Df_dataset,
        av.L_dataset,
        av.A_dataset,
        av.con,
        av.tst,
        av.poi,
    ) = read_config_csv(pdp_dataset_path)

    # Standard export retained (write into results dir if provided)
    results_dir = os.environ.get('AV_RESULTS_DIR', os.getcwd())
    os.makedirs(results_dir, exist_ok=True)
    # av.Df_dataset.to_csv removed - not necessary for output

    # Execute analysis stages - track successful imports
    N_PDP_imported = False
    N_VA_HeatMap_imported = False
    N_VA_HClust_imported = False
    N_VA_Mds_imported = False
    N_VA_TopK_imported = False
    
    # Debug: Print current directory and Python path
    print(f"📂 Current working directory: {os.getcwd()}")
    print(f"🐍 Python path includes: {os.getcwd() in __import__('sys').path}")
    print(f"🔧 AV_SKIP_LOAD env var: {os.environ.get('AV_SKIP_LOAD', 'NOT SET')}")
    print(f"📊 AV_DATASET env var: {os.environ.get('AV_DATASET', 'NOT SET')}")
    print(f"📂 av.dataset_name: {av.dataset_name}")
    
    # IMPORTANT: Reload av module to ensure it has the latest environment settings
    # This is crucial when running from GUI where env vars are set after initial import
    print("🔄 Reloading av module to pick up environment changes...")
    importlib.reload(av)
    print(f"✅ av reloaded. New dataset_name: {av.dataset_name}")
    
    if av.N_PDP == 1:
        try:
            import N_PDP
            N_PDP_imported = True
            print("✅ N_PDP imported successfully")
        except ImportError as e:
            print(f"⚠️ N_PDP import failed (ImportError): {e}")
        except Exception as e:
            print(f"⚠️ N_PDP import failed (Other error): {type(e).__name__}: {e}")
    if av.N_VA_HeatMap == 1:
        try:
            import N_VA_HeatMap
            N_VA_HeatMap_imported = True
        except ImportError:
            print("⚠️ N_VA_HeatMap not found in SAM folder")
    if av.N_VA_HClust == 1:
        try:
            import N_VA_HClust
            N_VA_HClust_imported = True
        except ImportError as e:
            print(f"⚠️ N_VA_HClust import failed (ImportError): {e}")
        except Exception as e:
            print(f"⚠️ N_VA_HClust import failed (Other error): {e}")
    
    if av.N_VA_Mds == 1:
        try:
            import N_VA_Mds
            N_VA_Mds_imported = True
        except ImportError as e:
            print(f"⚠️ N_VA_Mds import failed (ImportError): {e}")
        except Exception as e:
            print(f"⚠️ N_VA_Mds import failed (Other error): {e}")

    if av.N_VA_TopK == 1:
        try:
            import N_VA_TopK
            N_VA_TopK_imported = True
        except ImportError as e:
            print(f"⚠️ N_VA_TopK import failed (ImportError): {e}")
        except Exception as e:
            print(f"⚠️ N_VA_TopK import failed (Other error): {e}")
    

    av.PDPg_fundamental_active = 0


# ---------------- PDP: Buffer ----------------
if av.PDPg_buffer == 1:
    print("\n" + "="*60)
    print("🚀 STARTING PDP: BUFFER")
    print("="*60)
    av.PDPg_buffer_active = 1

    try:
        import N_T_OB  # your buffer-prep module
        
        # N_T_OB now creates the file directly in results_dir
        results_dir = os.environ.get('AV_RESULTS_DIR', os.getcwd())
        buffer_dataset_path = os.path.join(results_dir, "N_C_PDPg_buffer_Dataset.csv")
        
        if not os.path.exists(buffer_dataset_path):
            raise FileNotFoundError(f"❌ Buffer dataset not created: {buffer_dataset_path}")
        
        print(f"✅ Buffer dataset created: {buffer_dataset_path}")
        
        av.dataset_name = buffer_dataset_path
        av.dataset_name_exclusive = os.path.splitext(av.dataset_name)[0]

        (
            av.Df_dataset,
            av.L_dataset,
            av.A_dataset,
            av.con,
            av.tst,
            av.poi,
        ) = read_config_csv(buffer_dataset_path)

        results_dir = os.environ.get('AV_RESULTS_DIR', os.getcwd())
        os.makedirs(results_dir, exist_ok=True)
        # av.Df_dataset.to_csv removed - not necessary for output

        # Reload analysis modules if they were already imported in the fundamental branch
        if av.N_PDP == 1 and N_PDP_imported:
            importlib.reload(N_PDP)
        if av.N_VA_HeatMap == 1 and N_VA_HeatMap_imported:
            importlib.reload(N_VA_HeatMap)
        if av.N_VA_HClust == 1 and N_VA_HClust_imported:
            importlib.reload(N_VA_HClust)
        
        if av.N_VA_Mds == 1 and N_VA_Mds_imported:
            importlib.reload(N_VA_Mds)
        
    
        if av.N_VA_TopK == 1 and N_VA_TopK_imported:
            importlib.reload(N_VA_TopK)
        
    except ImportError as e:
        print(f"⚠️ N_T_OB import error: {e}")
        print("Skipping buffer transformation")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("Buffer transformation failed - dataset not created")
    except Exception as e:
        print(f"❌ Error during buffer transformation: {e}")
        import traceback
        traceback.print_exc()

    av.PDPg_buffer_active = 0


# ---------------- PDP: Rough ----------------
if av.PDPg_rough == 1:
    print("\n" + "="*60)
    print("🚀 STARTING PDP: ROUGH")
    print("="*60)
    av.PDPg_rough_active = 1

    # For rough, you use the fundamental dataset; roughness is applied in inequality calc
    results_dir = os.environ.get('AV_RESULTS_DIR', os.getcwd())
    pdp_dataset_path = os.path.join(results_dir, "N_C_PDPg_fundamental_Dataset.csv")
    
    av.dataset_name = pdp_dataset_path
    av.dataset_name_exclusive = os.path.splitext(av.dataset_name)[0]

    (
        av.Df_dataset,
        av.L_dataset,
        av.A_dataset,
        av.con,
        av.tst,
        av.poi,
    ) = read_config_csv(pdp_dataset_path)

    results_dir = os.environ.get('AV_RESULTS_DIR', os.getcwd())
    os.makedirs(results_dir, exist_ok=True)
    # av.Df_dataset.to_csv removed - not necessary for output

    if av.N_PDP == 1 and N_PDP_imported:
        importlib.reload(N_PDP)
    if av.N_VA_HeatMap == 1 and N_VA_HeatMap_imported:
        importlib.reload(N_VA_HeatMap)
    if av.N_VA_HClust == 1 and N_VA_HClust_imported:
        importlib.reload(N_VA_HClust)
    
    if av.N_VA_Mds == 1 and N_VA_Mds_imported:
        importlib.reload(N_VA_Mds)
    
    if av.N_VA_TopK == 1 and N_VA_TopK_imported:
        importlib.reload(N_VA_TopK)
    
    av.PDPg_rough_active = 0


# ---------------- PDP: Buffer + Rough ----------------
if av.PDPg_bufferrough == 1:
    print("\n" + "="*60)
    print("🚀 STARTING PDP: BUFFER + ROUGH")
    print("="*60)
    av.PDPg_bufferrough_active = 1

    try:
        import N_T_OB  # buffer generator (roughness applied later in metrics)

        # N_T_OB creates buffer dataset - ensure it's in results_dir
        results_dir = os.environ.get('AV_RESULTS_DIR', os.getcwd())
        os.makedirs(results_dir, exist_ok=True)
        
        # Move buffer dataset if it was created in CWD
        buffer_cwd = "N_C_PDPg_buffer_Dataset.csv"
        buffer_results = os.path.join(results_dir, "N_C_PDPg_buffer_Dataset.csv")
        if os.path.exists(buffer_cwd) and buffer_cwd != buffer_results:
            shutil.move(buffer_cwd, buffer_results)
        
        av.dataset_name = buffer_results
        av.dataset_name_exclusive = os.path.splitext(av.dataset_name)[0]

        (
            av.Df_dataset,
            av.L_dataset,
            av.A_dataset,
            av.con,
            av.tst,
            av.poi,
        ) = read_config_csv(buffer_results)

        results_dir = os.environ.get('AV_RESULTS_DIR', os.getcwd())
        os.makedirs(results_dir, exist_ok=True)
        # av.Df_dataset.to_csv removed - not necessary for output

        if av.N_PDP == 1 and N_PDP_imported:
            importlib.reload(N_PDP)
        if av.N_VA_HeatMap == 1 and N_VA_HeatMap_imported:
            importlib.reload(N_VA_HeatMap)
        if av.N_VA_HClust == 1 and N_VA_HClust_imported:
            importlib.reload(N_VA_HClust)
        
        if av.N_VA_Mds == 1 and N_VA_Mds_imported:
            importlib.reload(N_VA_Mds)
        
        if av.N_VA_TopK == 1 and N_VA_TopK_imported:
            importlib.reload(N_VA_TopK)
        
    except ImportError:
        print("⚠️ N_T_OB not found in SAM folder - skipping buffer+rough transformation")

    av.PDPg_bufferrough_active = 0


# ---------------- Final timing ----------------
if not _already_printed:
    elapsed = time.time() - t_start
    print("\n" + "="*60)
    print("✅ ALL PDP PROCESSING COMPLETE!")
    print("="*60)
    print(f'⏱️  Total time elapsed: {elapsed:.3f} sec ({elapsed/60:.2f} min)')
    print("="*60 + "\n")
    av._moving_objects_printed = True  # Set flag to prevent duplicate print on reload
