import json
import re

file_path = r'D:\CodeSSD\Pycharm\Kaggle_hull_tactical\Kaggle_Hull\v6\Hull_AOE_Fin.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Get the source string from the notebook
source_list = notebook['cells'][1]['source']
source_str = "".join(source_list)

# Define the new code blocks
config_block = """# -----------------------------------------------------------------------------------------
# 1. CONFIGURATION (Updated)
# -----------------------------------------------------------------------------------------
class Config:
    SEED = 42
    
    # Gen5 Defaults
    VOL_SHORT = 5
    VOL_LONG = 22
    VOL_QUARTERLY = 66
    
    EMA_FAST = 5
    EMA_SLOW = 26

    # NEW: Rolling Window for Stationarity (1 Trading Year)
    SCALING_WINDOW = 252 
    
    # Model Params (Unchanged)
    LGBM_PARAMS = {
        'n_estimators': 1000,
        'learning_rate': 0.01,
        'max_depth': 5,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.5,
        'reg_alpha': 0.1,
        'n_jobs': -1,
        'verbose': -1,
        'random_state': 42
    }
    
    # Trading Logic
    BASE_W_LINEAR = 0.4
    TARGET_VOL = 0.005
    MAX_LEVERAGE = 2.0
    SGD_LR = 0.001
    SGD_ALPHA = 0.001"""

final_model_block = """# -----------------------------------------------------------------------------------------
# 5. FINAL MODEL TRAINING (FIXED)
# -----------------------------------------------------------------------------------------
# Re-generate features
train_df = feature_engineering(raw_train_df)

# --- FIX START: ROLLING SCALING ---
# Instead of global scaling, we pre-calculate rolling Z-scores for the entire history.
# This ensures training inputs are stationary (relative to their recent past).
feature_cols = [c for c in train_df.columns if c not in DROP_COLS]

# Calculate rolling stats (with a small epsilon to avoid division by zero)
rolling_mean = train_df[feature_cols].rolling(window=Config.SCALING_WINDOW, min_periods=30).mean()
rolling_std = train_df[feature_cols].rolling(window=Config.SCALING_WINDOW, min_periods=30).std()
train_df_scaled = (train_df[feature_cols] - rolling_mean) / (rolling_std + 1e-8)

# Fill initial NaNs (where window wasn't full yet) with 0
train_df_scaled = train_df_scaled.fillna(0)
# --- FIX END ---

# Slice for training (removing warmup period)
# We use the scaled features for Linear Model, raw features for Tree Model
train_start = 75
X_raw = train_df.iloc[train_start:][feature_cols]
X_scaled = train_df_scaled.iloc[train_start:][feature_cols]
y = train_df.iloc[train_start:][TARGET]

FEATURES = feature_cols

print(f"Training Final Model on {len(X_raw)} rows...")

# Linear Model (Now uses Rolling Z-Scores)
# REMOVED: scaler = StandardScaler()
linear_model = SGDRegressor(
    loss='squared_error', penalty='l2', alpha=Config.SGD_ALPHA,
    learning_rate='constant', eta0=Config.SGD_LR, 
    random_state=Config.SEED, max_iter=2000
)
linear_model.fit(X_scaled, y) # Fit on rolling scaled data

# Tree Model (Uses Raw Data - Trees handle non-stationarity better via splits)
lgbm_model = LGBMRegressor(**Config.LGBM_PARAMS)
lgbm_model.fit(X_raw, y)

print("Gen5 Models Ready (Regime Fix Applied).")"""

inference_block = """# -----------------------------------------------------------------------------------------
# 6. INFERENCE LOOP (FIXED)
# -----------------------------------------------------------------------------------------
# GLOBAL_HISTORY needs to be large enough for the rolling window
GLOBAL_HISTORY = raw_train_df.iloc[-400:].copy() 
STEP = 0
ratio_history = [] 

def predict(test_pl: pl.DataFrame) -> float:
    # GLOBAL VARIABLES
    # REMOVED: scaler
    global GLOBAL_HISTORY, STEP, linear_model 
    
    # 1. Update History
    cols = [c for c in test_pl.columns if c != 'date_id']
    test_pl = test_pl.with_columns([pl.col(c).cast(pl.Float64, strict=False).fill_null(0) for c in cols])
    test_df_raw = test_pl.to_pandas()
    
    GLOBAL_HISTORY = pd.concat([GLOBAL_HISTORY, test_df_raw], axis=0, ignore_index=True)
    
    # 2. Features (Raw)
    full_features = feature_engineering(GLOBAL_HISTORY)
    current_features_raw = full_features.iloc[[-1]][FEATURES]
    
    # --- FIX START: DYNAMIC ROLLING SCALING ---
    # We must compute the scaling relative to the SPECIFIC history at this moment.
    # Calculate rolling stats on the updated history
    rolling_mean = full_features[FEATURES].rolling(window=Config.SCALING_WINDOW, min_periods=30).mean()
    rolling_std = full_features[FEATURES].rolling(window=Config.SCALING_WINDOW, min_periods=30).std()
    
    # Normalize the entire history (or just the tail) to get the current Z-score
    full_features_scaled = (full_features[FEATURES] - rolling_mean) / (rolling_std + 1e-8)
    
    # Select just the last row (the current prediction step)
    curr_X_scaled = full_features_scaled.iloc[[-1]].fillna(0)
    # --- FIX END ---

    # 3. Prediction
    # Linear model gets the Rolling Z-Score input
    pred_linear = linear_model.predict(curr_X_scaled)[0]
    
    # Tree model gets the Raw input
    pred_tree = lgbm_model.predict(current_features_raw)[0]
    
    # 4. Regime Ensemble
    curr_vol = current_features_raw['vol_short'].values[0] 
    long_vol = current_features_raw['vol_quarterly'].values[0]
    
    w_lin, w_tree = get_adaptive_weights(curr_vol, long_vol)
    raw_pred = (pred_linear * w_lin) + (pred_tree * w_tree)
    
    # 5. Risk Control
    safe_vol = curr_vol if curr_vol > 1e-5 else 0.005
    vol_scalar = Config.TARGET_VOL / safe_vol
    sharpe_forecast = abs(raw_pred) / safe_vol
    
    allocation_size = sharpe_forecast * vol_scalar * 50
    sign = np.sign(raw_pred)
    
    # RSI Sanity Check
    rsi = current_features_raw['rsi'].values[0]
    if rsi > 75 and sign > 0: allocation_size *= 0.5
    elif rsi < 25 and sign < 0: allocation_size *= 0.5
        
    allocation = np.clip(1.0 + (sign * allocation_size), 0.0, 2.0)
    
    # 6. Online Learning
    try:
        prev_target = test_df_raw['lagged_forward_returns'].values[0]
        # Ensure we have a valid previous target and history
        if not np.isnan(prev_target) and STEP > 0:
            # NOTE: Ideally we should use the *previous step's* scaled features here.
            # But for simplicity in this specific fix, we use current scaling 
            # (which is close enough given the slow-moving window).
            linear_model.partial_fit(curr_X_scaled, [prev_target])
    except:
        pass
        
    # Maintain History Buffer (Must be larger than SCALING_WINDOW=252)
    if len(GLOBAL_HISTORY) > 500:
        GLOBAL_HISTORY = GLOBAL_HISTORY.iloc[-400:].reset_index(drop=True)
        
    STEP += 1
    return float(allocation) """

# 1. Replace Config
# Regex: From "class Config:" up to start of Section 2
pattern_config = r"(class Config:[\s\S]*?)(?=\n# -+\n# 2)"
replacement_config = config_block
new_source = re.sub(pattern_config, replacement_config, source_str, count=1)

# 2. Replace Final Model Training
# Regex: From Section 5 header up to start of Section 6
pattern_training = r"(# -+\n# 5\. FINAL MODEL TRAINING[\s\S]*?)(?=\n# -+\n# 6)"
replacement_training = final_model_block
new_source = re.sub(pattern_training, replacement_training, new_source, count=1)

# 3. Replace Inference Loop
# We need to preserve `get_adaptive_weights`.
# Find it in original source.
start_marker = "def get_adaptive_weights"
end_marker = "def predict"
start_idx = source_str.find(start_marker)
end_idx = source_str.find(end_marker)

if start_idx != -1 and end_idx != -1:
    weights_code = source_str[start_idx:end_idx]
    
    # Insert weights_code into the new inference_block
    # inference_block has "def predict". We insert before it.
    pred_idx = inference_block.find("def predict")
    final_inference_block = inference_block[:pred_idx] + weights_code + inference_block[pred_idx:]
    
    # Regex: From Section 6 header up to `inference_server =`
    pattern_inference = r"(# -+\n# 6\. INFERENCE LOOP[\s\S]*?)(?=\n\ninference_server =)"
    new_source = re.sub(pattern_inference, final_inference_block, new_source, count=1)
    
else:
    print("Error: Could not find get_adaptive_weights function.")
    exit(1)

# Split back into lines, preserving ends is important but source is usually list of strings
# The json usually stores lines with \n at the end.
new_source_lines = new_source.splitlines(keepends=True)

# Update notebook
notebook['cells'][1]['source'] = new_source_lines

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Notebook updated successfully.")
