import json
import re

file_path = r'D:\CodeSSD\Pycharm\Kaggle_hull_tactical\Kaggle_Hull\v6\Hull_AOE_Fin.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Get the source string from the notebook
source_list = notebook['cells'][1]['source']
source_str = "".join(source_list)

# Define the new inference block using single triple quotes to contain the code which uses double quotes
inference_block = r'''# -----------------------------------------------------------------------------------------
# 6. INFERENCE LOOP (FIXED)
# -----------------------------------------------------------------------------------------
# GLOBAL_HISTORY needs to be large enough for the rolling window
GLOBAL_HISTORY = raw_train_df.iloc[-400:].copy() 
STEP = 0
ratio_history = []
# NEW: Cache for correct online learning alignment
LAST_STEP_X_SCALED = None

def get_adaptive_weights(current_vol, long_term_vol, crash_sensitivity=2.0):
    """
    crash_sensitivity: Standard Deviations (Sigma) to trigger defensive mode.
                       2.0 = Top 2.5% of violent days (Robust).
    """
    global ratio_history
    
    # Calculate current ratio
    ratio = current_vol / (long_term_vol + 1e-8)
    
    # Add to history for rolling stats
    ratio_history.append(ratio)
    if len(ratio_history) > 100: 
        ratio_history.pop(0) # Keep window fixed size
    
    w_linear = Config.BASE_W_LINEAR
    
    # --- DYNAMIC LOGIC HERE ---
    # Only calculate Z-score if we have enough history (e.g., 20 days)
    if len(ratio_history) > 20:
        rolling_series = pd.Series(ratio_history)
        
        # Calculate dynamic context
        mean = rolling_series.mean()
        std = rolling_series.std() + 1e-8
        
        # Z-Score: How many Sigmas away is today's volatility?
        z_score = (ratio - mean) / std
        
        # Decision: Use Z-Score instead of "1.3"
        if z_score > crash_sensitivity: 
            # Volatility is statistically shocking relative to recent context
            w_linear = 0.7
            print(f"Defensive Mode Triggered! Z-Score: {z_score:.2f}")
            
        elif z_score < -1.0:
            # Volatility is unusually calm
            w_linear = 0.2
            
    return w_linear, 1.0 - w_linear

def predict(test_pl: pl.DataFrame) -> float:
    # GLOBAL VARIABLES
    # REMOVED: scaler
    # ADDED: LAST_STEP_X_SCALED
    global GLOBAL_HISTORY, STEP, linear_model, LAST_STEP_X_SCALED 
    
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
    
    # 6. Online Learning (FIXED ALIGNMENT)
    try:
        prev_target = test_df_raw['lagged_forward_returns'].values[0]
        # Ensure we have a valid previous target and history
        if not np.isnan(prev_target) and STEP > 0 and LAST_STEP_X_SCALED is not None:
            # Train using the CACHED features from the PREVIOUS step (X_{t-1})
            # This aligns X_{t-1} with Y_{t-1} correctly.
            linear_model.partial_fit(LAST_STEP_X_SCALED, [prev_target])
    except:
        pass

    # Update Cache for Next Step
    LAST_STEP_X_SCALED = curr_X_scaled.copy()
        
    # Maintain History Buffer (Must be larger than SCALING_WINDOW=252)
    if len(GLOBAL_HISTORY) > 500:
        GLOBAL_HISTORY = GLOBAL_HISTORY.iloc[-400:].reset_index(drop=True)
        
    STEP += 1
    return float(allocation)'''

# Regex: From Section 6 header up to `inference_server =`
# We use a greedy match from the header to the start of the inference_server line.
pattern_inference = r'(# -+\n# 6. INFERENCE LOOP[\s\S]*?)(?=\n\ninference_server =)'
new_source = re.sub(pattern_inference, inference_block, source_str, count=1)

if new_source == source_str:
    print("Error: Could not find or replace inference block.")
    exit(1)

# Split back into lines
new_source_lines = new_source.splitlines(keepends=True)

# Update notebook
notebook['cells'][1]['source'] = new_source_lines

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Notebook updated successfully with Online Learning Fix.")
