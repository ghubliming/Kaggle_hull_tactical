import json
import re

file_path = r'D:\CodeSSD\Pycharm\Kaggle_hull_tactical\Kaggle_Hull\v6\Hull_AOE_Fin.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Get the source string from the notebook
source_list = notebook['cells'][1]['source']
source_str = "".join(source_list)

# Define the old block
old_block = r"""    GLOBAL_HISTORY = pd.concat([GLOBAL_HISTORY, test_df_raw], axis=0, ignore_index=True)
    
    # 2. Features (Raw)"""

# Define the new block
new_block = r"""    GLOBAL_HISTORY = pd.concat([GLOBAL_HISTORY, test_df_raw], axis=0, ignore_index=True)
    
    # --- FIX START: FILL REVEALED TARGETS ---
    if STEP > 0:
        # The API gives us the answer for the PREVIOUS day in 'lagged_forward_returns'
        # We must put this answer into our history so 'shift(1)' works tomorrow.
        revealed_prev_return = test_df_raw['lagged_forward_returns'].values[0]
        
        # Patch the PREVIOUS row (index -2) in the 'forward_returns' column
        # Note: 'forward_returns' exists in GLOBAL_HISTORY because it started from train.csv
        if 'forward_returns' in GLOBAL_HISTORY.columns:
            col_idx = GLOBAL_HISTORY.columns.get_loc('forward_returns')
            GLOBAL_HISTORY.iloc[-2, col_idx] = revealed_prev_return
    # --- FIX END ---
    
    # 2. Features (Raw)"""

# Replace
if old_block in source_str:
    new_source = source_str.replace(old_block, new_block)
else:
    print("Error: Could not find the target block for replacement.")
    # Debug: print the area where it should be
    start_search = source_str.find("GLOBAL_HISTORY = pd.concat")
    if start_search != -1:
        print("--- Found similar code ---")
        print(source_str[start_search:start_search+200])
    else:
        print("Could not find 'GLOBAL_HISTORY = pd.concat'")
    exit(1)

# Split back into lines
new_source_lines = new_source.splitlines(keepends=True)

# Update notebook
notebook['cells'][1]['source'] = new_source_lines

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Notebook updated successfully with Target Filling Fix.")
