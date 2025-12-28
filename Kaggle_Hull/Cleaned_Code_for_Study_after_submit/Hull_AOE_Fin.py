"""
Module: Hull_AOE_Fin
Description:
    Implements a hybrid machine learning strategy combining linear and tree-based models 
    with online learning capabilities. The system features dynamic feature engineering 
    and adaptive regime detection.

Technical Architecture:
    1. Configuration:
        - Dynamic configuration class capable of runtime updates via Optuna.
        - Defines window sizes for volatility, EMA, and model parameters.

    2. Feature Engineering:
        - Technical indicators: EMA, RSI, MACD.
        - Volatility measures: Short, Long, Quarterly windows.
        - Regime features: Volatility ratios and flash crash signals.
        - Dynamic Rolling Scaling: Z-score normalization using a rolling window.

    3. Optimization (Meta-Learning):
        - Uses Optuna to optimize feature lookback windows and LGBM hyperparameters 
        - Validates using TimeSeriesSplit.

    4. Model Training:
        - Linear Model: SGDRegressor with online partial_fit capabilities.
        - Tree Model: LGBMRegressor for capturing non-linear relationships.
        - Uses rolling Z-scores for the linear model and raw features for the tree model.

    5. Inference & Online Learning:
        - Adaptive Ensemble: Dynamically weights linear vs. tree models based on volatility regimes.
        - Online Learning: Updates the SGDRegressor incrementally with revealed targets.
        - Risk Control: Adjusts allocation based on forecasted Sharpe ratio and RSI.
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor
import kaggle_evaluation.default_inference_server

# Try importing Optuna for Meta-Learning, handle offline/missing case gracefully
try:
    import optuna
    OPTUNA_AVAILABLE = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not found. Using Gen5 Defaults (Meta-Learning skipped).")

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------------------
# 1. CONFIGURATION (DYNAMIC & UPDATED)
# -----------------------------------------------------------------------------------------
class Config:
    """
    Central configuration class. 
    Some parameters here are 'Dynamic' meaning they might be overwritten by Optuna.
    """
    SEED = 42
    
    # Gen5 Defaults (Can be tuned)
    VOL_SHORT = 5
    VOL_LONG = 22
    VOL_QUARTERLY = 66
    
    EMA_FAST = 5
    EMA_SLOW = 26

    # NEW: Rolling Window for Stationarity (1 Trading Year)
    # This is used for z-scoring features relative to recent history.
    SCALING_WINDOW = 252 
    
    # Model Params (LGBM)
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
    
    # Trading Logic Parameters
    BASE_W_LINEAR = 0.4
    TARGET_VOL = 0.005
    MAX_LEVERAGE = 2.0
    SGD_LR = 0.001
    SGD_ALPHA = 0.001

# -----------------------------------------------------------------------------------------
# 2. FEATURE ENGINEERING (ADAPTIVE)
# -----------------------------------------------------------------------------------------

def calculate_ema(series, period):
    """Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()

def calculate_rsi(series, period=14):
    """Relative Strength Index (RSI)"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10) # Avoid div by zero
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    """Moving Average Convergence Divergence"""
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates technical indicators.
    Critically, relies on 'Config' class variables which might be updated by Optuna.
    """
    df = df.copy()
    targets = ['forward_returns', 'risk_free_rate']
    base_col = 'lag_forward_returns_1'
    
    # 1. Lags
    for col in targets:
        for lag in [1, 2, 3, 5, 10, 22]:
            df[f'lag_{col}_{lag}'] = df[col].shift(lag)
            
    # 2. Volatility (Dynamic Windows)
    df['vol_short'] = df[base_col].rolling(Config.VOL_SHORT).std()
    df['vol_long'] = df[base_col].rolling(Config.VOL_LONG).std()
    df['vol_quarterly'] = df[base_col].rolling(Config.VOL_QUARTERLY).std()
    
    # 3. Momentum & Tech
    df['mom_short'] = df[base_col].rolling(Config.VOL_SHORT).mean()
    df['ema_fast'] = calculate_ema(df[base_col], Config.EMA_FAST)
    df['ema_slow'] = calculate_ema(df[base_col], Config.EMA_SLOW)
    df['ema_cross'] = df['ema_fast'] - df['ema_slow']
    
    df['rsi'] = calculate_rsi(df[base_col], 14)
    df['macd'], _, _ = calculate_macd(df[base_col])
    
    # 4. Regime Features
    # Dynamic Vol Ratio: How crazy is today vs the last quarter?
    df['vol_ratio'] = df['vol_long'] / (df['vol_quarterly'] + 1e-8)
    # Flash Crash Signal: How crazy is today vs this month?
    df['flash_crash_signal'] = df['vol_short'] / (df['vol_long'] + 1e-8)
    
    # Fill NaNs
    df = df.fillna(0)
    return df

# -----------------------------------------------------------------------------------------
# 3. DATA LOADING & PREP
# -----------------------------------------------------------------------------------------
def load_data(path):
    print(f"Loading {path}...")
    df_pl = pl.read_csv(path)
    cols = [c for c in df_pl.columns if c != 'date_id']
    # Cast to float and fill nulls efficiently
    df_pl = df_pl.with_columns([pl.col(c).cast(pl.Float64, strict=False).fill_null(0) for c in cols])
    return df_pl.to_pandas()

TRAIN_PATH = "/kaggle/input/hull-tactical-market-prediction/train.csv"
try:
    raw_train_df = load_data(TRAIN_PATH)
except:
    # Fallback for local testing
    print("Kaggle data not found, generating dummy data.")
    raw_train_df = pd.DataFrame({
        'date_id': range(1000),
        'lag_forward_returns_1': np.random.randn(1000),
        'forward_returns': np.random.randn(1000),
        'risk_free_rate': np.zeros(1000),
        'market_forward_excess_returns': np.random.randn(1000)
    })

TARGET = "forward_returns"
DROP_COLS = ['date_id', 'is_scored', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']

# -----------------------------------------------------------------------------------------
# 4. META-LEARNING (OPTUNA)
# -----------------------------------------------------------------------------------------
def objective(trial):
    """
    Optuna objective function.
    Optimizes lookback windows (VOL_SHORT, etc.) and LGBM hyperparameters.
    """
    # 1. Suggest Parameters
    Config.VOL_SHORT = trial.suggest_int('vol_short', 3, 10)
    Config.VOL_LONG = trial.suggest_int('vol_long', 15, 30)
    Config.VOL_QUARTERLY = trial.suggest_int('vol_quarterly', 50, 80)
    
    lgbm_params = {
        'n_estimators': 500, # Lower for speed during optimization
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.9),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0, log=True),
        'n_jobs': -1,
        'verbose': -1,
        'random_state': Config.SEED
    }
    
    # 2. Generate Features (Dynamic based on new Config)
    df = feature_engineering(raw_train_df)
    
    # 3. Walk-Forward Validation
    train_start = 75
    
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    X = df.iloc[train_start:].drop(columns=cols_to_drop)
    y = df.iloc[train_start:][TARGET]
    
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = LGBMRegressor(**lgbm_params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        scores.append(mean_squared_error(y_val, preds))
        
    return np.mean(scores)

print("Starting Optuna Optimization...")
# Check: Optuna Available + Not Rerun (Don't retune during actual submission)
if OPTUNA_AVAILABLE and not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    try:
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20)
        
        print("Best Params:", study.best_params)
        
        # Update Config with best parameters
        Config.VOL_SHORT = study.best_params['vol_short']
        Config.VOL_LONG = study.best_params['vol_long']
        Config.VOL_QUARTERLY = study.best_params['vol_quarterly']
        
        for k, v in study.best_params.items():
            if k in Config.LGBM_PARAMS:
                Config.LGBM_PARAMS[k] = v
        Config.LGBM_PARAMS['n_estimators'] = 1000 # Increase estimators for final training
    except Exception as e:
        print(f"Optimization Failed: {e}. Using Defaults.")
else:
    print("Skipping optimization (Offline/Rerun). Using Gen5 Defaults.")

# -----------------------------------------------------------------------------------------
# 5. FINAL MODEL TRAINING
# -----------------------------------------------------------------------------------------
# Re-generate features with potentially updated Config
train_df = feature_engineering(raw_train_df)

# Pre-calculate rolling Z-scores for stationarity
# Instead of global scaling, we pre-calculate rolling Z-scores for the entire history.
# This ensures training inputs are stationary (relative to their recent past).
feature_cols = [c for c in train_df.columns if c not in DROP_COLS]

# Calculate rolling stats (with a small epsilon to avoid division by zero)
rolling_mean = train_df[feature_cols].rolling(window=Config.SCALING_WINDOW, min_periods=30).mean()
rolling_std = train_df[feature_cols].rolling(window=Config.SCALING_WINDOW, min_periods=30).std()
train_df_scaled = (train_df[feature_cols] - rolling_mean) / (rolling_std + 1e-8)

# Fill initial NaNs (where window wasn't full yet) with 0
train_df_scaled = train_df_scaled.fillna(0)

# Slice for training (removing warmup period)
# We use the scaled features for Linear Model, raw features for Tree Model
train_start = 75
X_raw = train_df.iloc[train_start:][feature_cols]
X_scaled = train_df_scaled.iloc[train_start:][feature_cols]
y = train_df.iloc[train_start:][TARGET]

FEATURES = feature_cols

print(f"Training Final Model on {len(X_raw)} rows...")

# Linear Model (Now uses Rolling Z-Scores)
# Note: SGDRegressor supports partial_fit for online learning later
linear_model = SGDRegressor(
    loss='squared_error', penalty='l2', alpha=Config.SGD_ALPHA,
    learning_rate='constant', eta0=Config.SGD_LR, 
    random_state=Config.SEED, max_iter=2000
)
linear_model.fit(X_scaled, y) # Fit on rolling scaled data

# Tree Model (Uses Raw Data - Trees handle non-stationarity better via splits)
lgbm_model = LGBMRegressor(**Config.LGBM_PARAMS)
lgbm_model.fit(X_raw, y)

print("Gen5 Models Ready (Regime Fix Applied).")

# -----------------------------------------------------------------------------------------
# 6. INFERENCE LOOP (ONLINE LEARNING)
# -----------------------------------------------------------------------------------------

# GLOBAL_HISTORY needs to be large enough for the rolling window
GLOBAL_HISTORY = raw_train_df.iloc[-400:].copy() 
STEP = 0
ratio_history = []
# NEW: Cache for correct online learning alignment
LAST_STEP_X_SCALED = None

def get_adaptive_weights(current_vol, long_term_vol, crash_sensitivity=2.0):
    """
    Determines the mixing weight between Linear and Tree models based on Volatility Regime.
    
    Args:
        current_vol: Short-term volatility.
        long_term_vol: Long-term volatility.
        crash_sensitivity: Standard Deviations (Sigma) to trigger defensive mode.
                           2.0 = Top 2.5% of violent days (Robust).
                           
    Returns:
        w_linear, w_tree
    """
    global ratio_history
    
    # Calculate current ratio
    ratio = current_vol / (long_term_vol + 1e-8)
    
    # Add to history for rolling stats
    ratio_history.append(ratio)
    if len(ratio_history) > 100: 
        ratio_history.pop(0) # Keep window fixed size
    
    w_linear = Config.BASE_W_LINEAR
    
    # --- DYNAMIC REGIME LOGIC ---
    # Only calculate Z-score if we have enough history (e.g., 20 days)
    if len(ratio_history) > 20:
        rolling_series = pd.Series(ratio_history)
        
        # Calculate dynamic context
        mean = rolling_series.mean()
        std = rolling_series.std() + 1e-8
        
        # Z-Score: How many Sigmas away is today's volatility?
        z_score = (ratio - mean) / std
        
        # Decision: Use Z-Score instead of static "1.3"
        if z_score > crash_sensitivity: 
            # Volatility is statistically shocking relative to recent context -> DEFENSIVE
            w_linear = 0.7 
            print(f"Defensive Mode Triggered! Z-Score: {z_score:.2f}")
            
        elif z_score < -1.0:
            # Volatility is unusually calm -> AGGRESSIVE (more trust in tree model)
            w_linear = 0.2
            
    return w_linear, 1.0 - w_linear

def predict(test_pl: pl.DataFrame) -> float:
    """
    Main prediction loop. Features:
    - Updates Global History.
    - Fills revealed targets for Online Learning.
    - Computes Dynamic Rolling Scaling.
    - Performs Inference (Linear + Tree).
    - Updates SGDRegressor (Online Learning).
    """
    # GLOBAL VARIABLES
    global GLOBAL_HISTORY, STEP, linear_model, LAST_STEP_X_SCALED 
    
    # 1. Update History
    cols = [c for c in test_pl.columns if c != 'date_id']
    test_pl = test_pl.with_columns([pl.col(c).cast(pl.Float64, strict=False).fill_null(0) for c in cols])
    test_df_raw = test_pl.to_pandas()
    
    GLOBAL_HISTORY = pd.concat([GLOBAL_HISTORY, test_df_raw], axis=0, ignore_index=True)
    
    # Fill revealed targets for online learning context
    if STEP > 0:
        # The API gives us the answer for the PREVIOUS day in 'lagged_forward_returns'
        # We must put this answer into our history so 'shift(1)' works tomorrow.
        revealed_prev_return = test_df_raw['lagged_forward_returns'].values[0]
        
        # Patch the PREVIOUS row (index -2) in the 'forward_returns' column
        if 'forward_returns' in GLOBAL_HISTORY.columns:
            col_idx = GLOBAL_HISTORY.columns.get_loc('forward_returns')
            GLOBAL_HISTORY.iloc[-2, col_idx] = revealed_prev_return
    
    # 2. Features (Raw)
    full_features = feature_engineering(GLOBAL_HISTORY)
    current_features_raw = full_features.iloc[[-1]][FEATURES]
    
    # Compute dynamic rolling scaling for current step
    # We must compute the scaling relative to the SPECIFIC history at this moment.
    # Calculate rolling stats on the updated history
    rolling_mean = full_features[FEATURES].rolling(window=Config.SCALING_WINDOW, min_periods=30).mean()
    rolling_std = full_features[FEATURES].rolling(window=Config.SCALING_WINDOW, min_periods=30).std()
    
    # Normalize the entire history (or just the tail) to get the current Z-score
    full_features_scaled = (full_features[FEATURES] - rolling_mean) / (rolling_std + 1e-8)
    
    # Select just the last row (the current prediction step)
    curr_X_scaled = full_features_scaled.iloc[[-1]].fillna(0)

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
    
    # 5. Risk Control & Position Sizing
    # Use a safe floor for volatility to avoid exploding leverage
    safe_vol = curr_vol if curr_vol > 1e-5 else 0.005
    vol_scalar = Config.TARGET_VOL / safe_vol
    sharpe_forecast = abs(raw_pred) / safe_vol
    
    allocation_size = sharpe_forecast * vol_scalar * 50
    sign = np.sign(raw_pred)
    
    # RSI Sanity Check
    rsi = current_features_raw['rsi'].values[0]
    if rsi > 75 and sign > 0: allocation_size *= 0.5 # Overbought
    elif rsi < 25 and sign < 0: allocation_size *= 0.5 # Oversold
        
    allocation = np.clip(1.0 + (sign * allocation_size), 0.0, 2.0)
    
    # 6. Online Learning Update
    # This is where the model learns from the *previous* day's outcome
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
    return float(allocation)

# Initialize Server
inference_server = kaggle_evaluation.default_inference_server.DefaultInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(('/kaggle/input/hull-tactical-market-prediction/',))
