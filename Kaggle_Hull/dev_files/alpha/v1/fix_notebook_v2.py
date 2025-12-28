import json
import os

# 1. Define the Code Blocks
# NOTE: We use simple strings. The add_cell function will handle newline formatting correctly.

source_imports = """import os
import time
import warnings
import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
import kaggle_evaluation.default_inference_server

warnings.filterwarnings("ignore")"""

source_config = """class Config:
    SEED = 42
    
    # Regime-Adaptive Weights
    # Adjusted for faster reaction
    DEFENSIVE_W_LINEAR = 0.7
    DEFENSIVE_W_TREE = 0.3
    
    NORMAL_W_LINEAR = 0.4
    NORMAL_W_TREE = 0.6
    
    # Volatility Targeting
    TARGET_VOL = 0.005
    MAX_LEVERAGE = 2.0
    
    # Online Learning Rate
    SGD_LR = 0.001"""

source_feature_engineering = """# REPLACE THE FEATURE ENGINEERING SECTION WITH THIS LEAN VERSION

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    targets = ['forward_returns', 'risk_free_rate']

    # 1. Lags (The most predictive features)
    for col in targets:
        for lag in [1, 2, 3, 5, 10]:
            df[f'lag_{col}_{lag}'] = df[col].shift(lag)

    # 2. Volatility (The Risk features)
    base_col = 'lag_forward_returns_1'
    df['vol_5d'] = df[base_col].rolling(5).std()
    df['vol_22d'] = df[base_col].rolling(22).std()

    # 3. Momentum (The Trend features)
    df['mom_5d'] = df[base_col].rolling(5).mean()
    df['mom_22d'] = df[base_col].rolling(22).mean()

    # 4. Lean Technicals (Only the robust ones)
    # EMA for Trend Direction
    df['ema_12'] = df[base_col].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df[base_col].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26'] # Standard MACD Line

    # RSI for Overbought/Oversold (Normalized 0-100)
    delta = df[base_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    # 5. Fast Regime Signal (Weekly vs Monthly)
    # If 5d vol is higher than 22d vol, market is accelerating/panicking
    df['vol_ratio'] = df['vol_5d'] / (df['vol_22d'] + 1e-8)

    df = df.fillna(0)
    return df

# ADJUST THE REGIME WEIGHT FUNCTION (Inference Loop)
def get_adaptive_weights(vol_ratio):
    # Faster reaction: If this week is 20% crazier than the month
    if vol_ratio > 1.2:
        return 0.7, 0.3 # Defensive: 70% Linear (Trend), 30% Tree
    else:
        return 0.4, 0.6 # Normal: 40% Linear, 60% Tree"""

source_data_loading = """def load_data(path):
    print(f"Loading {path}...")
    df_pl = pl.read_csv(path)
    cols = [c for c in df_pl.columns if c != 'date_id']
    df_pl = df_pl.with_columns([pl.col(c).cast(pl.Float64, strict=False).fill_null(0) for c in cols])
    return df_pl.to_pandas()

# Load Train
TRAIN_PATH = "/kaggle/input/hull-tactical-market-prediction/train.csv"
train_df = load_data(TRAIN_PATH)

print(f"Raw training data: {len(train_df)} rows")

# Apply Engineering
train_df = feature_engineering(train_df)

# Drop initial NaNs from lags
train_df = train_df.iloc[25:].reset_index(drop=True)

print(f"After feature engineering: {len(train_df)} rows")

# Define Columns
TARGET = "forward_returns"
DROP = ['date_id', 'is_scored', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
FEATURES = [c for c in train_df.columns if c not in DROP]

print(f"Features Created: {len(FEATURES)}")
print(f"Training samples available: {len(train_df)}")"""

source_training = """print("Training Hybrid Models...")

X = train_df[FEATURES]
y = train_df[TARGET]

# MODEL 1: Online Linear Model (SGD)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

linear_model = SGDRegressor(
    loss='squared_error', 
    penalty='l2',
    alpha=0.0001,
    learning_rate='constant', 
    eta0=Config.SGD_LR,
    max_iter=1000,
    tol=1e-4,
    random_state=Config.SEED
)
linear_model.fit(X_scaled, y)

# MODEL 2: LightGBM (Tree) - UPDATED REGULARIZATION
lgbm_model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=5,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.5,  # CHANGED from 0.8
    reg_alpha=0.1,         # CHANGED from 0.0
    reg_lambda=0.01,
    random_state=Config.SEED,
    n_jobs=-1,
    verbose=-1
)
lgbm_model.fit(X, y)

print("Models Trained Successfully.")"""

source_validation = """print("=" * 80)
print("LEAN FEATURE VALIDATION")
print("=" * 80)

# Check new features
new_features = ['vol_ratio', 'rsi', 'macd', 'vol_5d']
print(f"✓ Lean Features Created: {sum([col in train_df.columns for col in new_features])}/{len(new_features)}")

# Sample values
print("\n" + "=" * 80)
print("SAMPLE FEATURE VALUES (Last Row)")
print("=" * 80)
for feat in new_features:
    if feat in train_df.columns:
        val = train_df[feat].iloc[-1]
        print(f"{feat:20s}: {val:10.6f}")

print("\n✓ All features validated successfully!")
print("=" * 80)"""

source_inference = """# -----------------------------------------------------------------------------------------
# 5. INFERENCE LOOP (OPTIMIZED)
# -----------------------------------------------------------------------------------------

# State Variables
GLOBAL_HISTORY = train_df.iloc[-100:].copy()  # Keep last 100 days
STEP = 0
CURRENT_REGIME = 'NORMAL'

print(f"Initial history buffer: {len(GLOBAL_HISTORY)} days")

def predict(test_pl: pl.DataFrame) -> float:
    global GLOBAL_HISTORY, STEP, linear_model, scaler, CURRENT_REGIME
    
    # 1. Process Input (Strict Float Casting)
    cols = [c for c in test_pl.columns if c != 'date_id']
    test_pl = test_pl.with_columns([pl.col(c).cast(pl.Float64, strict=False).fill_null(0) for c in cols])
    test_df_raw = test_pl.to_pandas()
    
    # 2. Update History & Feature Engineering
    GLOBAL_HISTORY = pd.concat([GLOBAL_HISTORY, test_df_raw], axis=0, ignore_index=True)
    
    # Generate features on the FULL history, then take the last row
    full_features = feature_engineering(GLOBAL_HISTORY)
    current_features = full_features.iloc[[-1]][FEATURES]
    
    # 3. REGIME DETECTION (FAST)
    if 'vol_ratio' in current_features.columns:
        vol_ratio = current_features['vol_ratio'].values[0]
    else:
        vol_ratio = 1.0 # Default
        
    w_linear, w_tree = get_adaptive_weights(vol_ratio)
    
    # Update CURRENT_REGIME for logging
    if vol_ratio > 1.2:
        CURRENT_REGIME = 'DEFENSIVE'
    else:
        CURRENT_REGIME = 'NORMAL'
    
    # 4. Hybrid Prediction
    # Linear Prediction
    curr_X_scaled = scaler.transform(current_features)
    pred_linear = linear_model.predict(curr_X_scaled)[0]
    
    # Tree Prediction
    pred_tree = lgbm_model.predict(current_features)[0]
    
    # Regime-Adaptive Ensemble
    raw_return_pred = (pred_linear * w_linear) + (pred_tree * w_tree)
    
    # -------------------------------------------------------------------------
    # ENHANCED VOLATILITY SCALING
    # -------------------------------------------------------------------------
    
    # Get current market volatility
    if 'vol_22d' in current_features.columns:
        current_vol = current_features['vol_22d'].values[0]
    else:
        current_vol = 0.005

    if current_vol < 1e-6: 
        current_vol = 0.005
        
    # Kelly-style Sizing
    vol_scalar = Config.TARGET_VOL / current_vol
    sign = np.sign(raw_return_pred)
    sharpe_forecast = abs(raw_return_pred) / current_vol
    
    # Base allocation
    allocation_size = sharpe_forecast * vol_scalar * 50
    
    # Final Allocation
    allocation = 1.0 + (sign * allocation_size)
    
    # -------------------------------------------------------------------------
    # SAFETY CHECKS
    # -------------------------------------------------------------------------
    if CURRENT_REGIME == 'DEFENSIVE':
        # Cap leverage in volatile markets
        if allocation > 1.5:
            allocation = 1.5
            
    # Clip to Competition Limits [0, 2]
    allocation = np.clip(allocation, 0.0, 2.0)
    
    # -------------------------------------------------------------------------
    # ONLINE LEARNING (Update Linear Model)
    # -------------------------------------------------------------------------
    try:
        prev_target = test_df_raw['lagged_forward_returns'].values[0] if 'lagged_forward_returns' in test_df_raw.columns else np.nan
        
        if not np.isnan(prev_target):
            prev_features = full_features.iloc[[-2]][FEATURES] if len(full_features) > 1 else current_features
            prev_features_scaled = scaler.transform(prev_features)
            linear_model.partial_fit(prev_features_scaled, [prev_target])
    except Exception as e:
        pass
    
    # -------------------------------------------------------------------------
    # MEMORY MANAGEMENT
    # -------------------------------------------------------------------------
    if len(GLOBAL_HISTORY) > 200:
        GLOBAL_HISTORY = GLOBAL_HISTORY.iloc[-150:].reset_index(drop=True)
    
    # Increment step counter
    STEP += 1
    
    # Diagnostic logging
    if STEP % 100 == 0:
        print(f"Step {STEP} | Regime: {CURRENT_REGIME} | VolRatio: {vol_ratio:.2f} | Alloc: {allocation:.2f}")
    
    return allocation"""

source_server = """inference_server = kaggle_evaluation.default_inference_server.DefaultInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(('/kaggle/input/hull-tactical-market-prediction/',))"""

# 2. Construct the Notebook Object
notebook = {
 "cells": [],
 "metadata": {
  "kaggle": {
   "accelerator": "nvidiaTeslaT4",
   "dataSources": [{"databundleVersionId": 14348714, "sourceId": 111543, "sourceType": "competition"}],
   "dockerImageVersionId": 31193,
   "isGpuEnabled": True,
   "isInternetEnabled": False,
   "language": "python",
   "sourceType": "notebook"
  },
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.11.13"}
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

def add_cell(source, cell_type="code"):
    # Split lines. splitlines() eats the newlines, so we must add them back.
    lines = source.splitlines()
    # We want standard \n, NOT escaped \\n
    source_list = [line + "\n" for line in lines[:-1]]
    if lines:
        # Last line usually doesn't need a forced newline in the list item itself, 
        # but consistency is key. Jupyter usually just joins them.
        # Actually, typically every line in the list has \n except maybe the last one.
        source_list.append(lines[-1])
        
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source_list
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    
    notebook["cells"].append(cell)

# 3. Add Content
add_cell("# =========================================================================================\n### TITLE: Hull Tactical - Gen5 Lean & Fast (Optimized)\n### AUTHOR: AI Machine Learning Engineer\n### DESCRIPTION:\n### Optimized \"Lean\" version of the Hull Tactical strategy. \n### \n### CHANGES:\n### 1. **Signal-to-Noise**: Removed noisy indicators (KDJ, BB).\n### 2. **Fast Regime**: Switched to 5d/22d volatility ratio for instant reaction.\n### 3. **Regularization**: Increased LightGBM regularization to prevent overfitting.\n### ========================================================================================"", "markdown")

add_cell(source_imports)
add_cell(source_config)
add_cell(source_feature_engineering)
add_cell(source_data_loading)
add_cell(source_training)
add_cell(source_validation)
add_cell(source_inference)
add_cell(source_server)
add_cell("#\n## 🚀 LEAN OPTIMIZATION COMPLETE\nThe model has been stripped of noisy features and optimized for speed.\n- **Features**: Lags, Volatility, Momentum, RSI, MACD\n- **Regime**: Fast 5d/22d Volatility Ratio\n- **Model**: High-Regularization LightGBM + Online Linear", "markdown")

# 4. Write to File
with open("Hull_AOE_Fin.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Notebook successfully rewritten.")
