import os
import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path
from gc import collect 
from tqdm import tqdm
from dataclasses import dataclass, asdict
from scipy.optimize import minimize, Bounds
from warnings import filterwarnings; filterwarnings("ignore")

from sklearn.linear_model import ElasticNet, ElasticNetCV, LinearRegression
from sklearn.linear_model import RidgeCV

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import StackingRegressor

from catboost import CatBoostRegressor, Pool
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.model_selection import TimeSeriesSplit, train_test_split, KFold
from sklearn.preprocessing import StandardScaler

import kaggle_evaluation.default_inference_server

# ==========================================
# 1. DATA LOADING & PREPROCESSING
# ==========================================
try:
    train = pd.read_csv('/kaggle/input/hull-tactical-market-prediction/train.csv').dropna()
    test = pd.read_csv('/kaggle/input/hull-tactical-market-prediction/test.csv').dropna()
except FileNotFoundError:
    # Local fallback for testing if not on Kaggle
    print("Warning: Kaggle paths not found. Checking local directory...")
    if os.path.exists('train.csv'):
        train = pd.read_csv('train.csv').dropna()
        test = pd.read_csv('test.csv').dropna()
    else:
        # Create dummy data for syntax checking if no data exists
        print("Creating dummy data for structural validation...")
        train = pd.DataFrame(np.random.rand(200, 30), columns=[f'E{i}' for i in range(1, 21)] + [f'S{i}' for i in range(1, 6)] + [f'P{i}' for i in range(8, 14)] + ['forward_returns'])
        test = train.drop(columns=['forward_returns']).copy()
        train['date_id'] = np.arange(200) # Sequential dates
        test['date_id'] = 201
        train['market_forward_excess_returns'] = np.random.rand(200)

def preprocessing(data, typ):
    main_feature = ['E1','E10', 'E11', 'E12', 'E13', 'E14', 'E15', 'E16', 'E17', 'E18', 'E19',
                    'E2', 'E20', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9',
                    "S2", "P9", "S1", "S5", "I2", "P8",
                    "P10", "P12", "P13",]
    
    # Filter to only existing columns
    cols_to_use = [c for c in main_feature if c in data.columns]
    
    if typ == "train":
        if "forward_returns" in data.columns:
             cols_to_use.append("forward_returns")
        data = data[cols_to_use]
    else:
        data = data[cols_to_use]
        
    for i in zip(data.columns, data.dtypes):
        data[i[0]].fillna(0, inplace=True)

    return data

# --- CRITICAL FIX 1: STRICT TIME-SERIES SPLIT ---
# Do NOT shuffle. We must train on past, validate on future.
# Sorting by date_id if it exists to be safe, though usually train.csv is sorted.
if 'date_id' in train.columns:
    train = train.sort_values('date_id').reset_index(drop=True)

train = preprocessing(train, "train")

split_idx = int(len(train) * 0.99) # Validating on last 1%
train_split = train.iloc[:split_idx]
val_split = train.iloc[split_idx:]

X_train = train_split.drop(columns=["forward_returns"], errors='ignore')
y_train = train_split['forward_returns'] if 'forward_returns' in train_split else np.zeros(len(train_split))

X_val = val_split.drop(columns=["forward_returns"], errors='ignore')
y_val = val_split['forward_returns'] if 'forward_returns' in val_split else np.zeros(len(val_split))

print(f"Time-Series Split :: Train: {len(X_train)} | Val: {len(X_val)}")

# ==========================================
# 2. GEN6 META-LEARNING ENGINE (Hyperparams)
# ==========================================
class MetaConfig:
    # --- Default Parameters (Will be overwritten) ---
    LGBM_LR = 0.05
    LGBM_LEAVES = 50
    XGB_LR = 0.05
    XGB_DEPTH = 6
    CAT_LR = 0.01
    CAT_DEPTH = 6
    
    # --- Thresholds & Exposures (Models 4 & 5) ---
    M4_ALPHA = 0.80007
    M5_ALPHA = 0.60013
    M5_TAU = 9.437e-05
    
    # --- Ensemble Weights ---
    WEIGHTS = [0.16, 0.16, 0.2, 0.16, 0.16, 0.16]

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    print("Optuna Detected. Phase 1: Hyperparameter Tuning (Time-Series Safe)...")
    
    def objective_hyperparams(trial):
        # 1. LGBM
        lgbm_lr = trial.suggest_float('lgbm_lr', 0.01, 0.15)
        lgbm_leaves = trial.suggest_int('lgbm_leaves', 20, 100)
        
        # 2. XGB
        xgb_lr = trial.suggest_float('xgb_lr', 0.01, 0.15)
        xgb_depth = trial.suggest_int('xgb_depth', 4, 10)
        
        # 3. CatBoost
        cat_lr = trial.suggest_float('cat_lr', 0.005, 0.1)
        cat_depth = trial.suggest_int('cat_depth', 4, 10)
        
        # Eval using LGBM as proxy or light ensemble on the Time-Series Split
        # We strictly train on X_train and evaluate on X_val
        model = LGBMRegressor(learning_rate=lgbm_lr, num_leaves=lgbm_leaves, n_estimators=50, verbosity=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return np.mean((preds - y_val)**2)

    study_hyp = optuna.create_study(direction='minimize')
    study_hyp.optimize(objective_hyperparams, n_trials=20) # Short run
    
    MetaConfig.LGBM_LR = study_hyp.best_params['lgbm_lr']
    MetaConfig.LGBM_LEAVES = study_hyp.best_params['lgbm_leaves']
    MetaConfig.XGB_LR = study_hyp.best_params['xgb_lr']
    MetaConfig.XGB_DEPTH = study_hyp.best_params['xgb_depth']
    MetaConfig.CAT_LR = study_hyp.best_params['cat_lr']
    MetaConfig.CAT_DEPTH = study_hyp.best_params['cat_depth']
    
    print("Phase 1 Complete. Best Params:", study_hyp.best_params)
    
except Exception as e:
    print(f"Hyperparam Tuning Failed ({e}). Using Defaults.")

# ==========================================
# 3. MODEL TRAINING (With Tuned Params)
# ==========================================

# 1. LGBM
model_lgbm = LGBMRegressor(learning_rate=MetaConfig.LGBM_LR, num_leaves=MetaConfig.LGBM_LEAVES, n_estimators=500, verbosity=-1)
model_lgbm.fit(X_train, y_train)

# 2. XGBoost
model_xgb = XGBRegressor(learning_rate=MetaConfig.XGB_LR, max_depth=MetaConfig.XGB_DEPTH, n_estimators=500)
model_xgb.fit(X_train, y_train)

# 3. CatBoost
model_cat = CatBoostRegressor(learning_rate=MetaConfig.CAT_LR, depth=MetaConfig.CAT_DEPTH, iterations=500, verbose=0)
model_cat.fit(X_train, y_train)

# 4. Stacking (The "Meta-Learner")
estimators = [
    ('lgbm', model_lgbm),
    ('xgb', model_xgb),
    ('cat', model_cat)
]
model_3 = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
model_3.fit(X_train, y_train)

print("Base Models & Stacking Regressor Trained.")

# ==========================================
# 4. THRESHOLD OPTIMIZATION (Phase 2)
# ==========================================
try:
    print("Phase 2: Tuning Activation Thresholds (Models 4 & 5)...")
    
    # We need predictions on Validation set to tune thresholds
    pred_val = model_3.predict(X_val)
    
    def objective_thresholds(trial):
        m4_a = trial.suggest_float('m4_alpha', 0.1, 1.5)
        m5_a = trial.suggest_float('m5_alpha', 0.1, 1.5)
        m5_t = trial.suggest_float('m5_tau', 0.0, 0.005)
        
        # Calculate PnL proxy on validation
        # Logic: We want to maximize returns, minimizing error isn't exactly the same but let's stick to MSE for stability
        # or negative correlation? Let's minimize MSE of the constrained signals against returns
        
        # Simplified: Just fit to y_val
        p4 = np.clip([m4_a if x > 0 else 0.0 for x in pred_val], 0, 2.0)
        p5 = np.clip([m5_a if x > m5_t else 0.0 for x in pred_val], 0, 2.0)
        
        # Let's say we want the ensemble of these two to be close to y_val (scaled)
        # This is a bit heuristic. In real trading we maximize Sharpe. 
        # Here we just minimize squared error to the target.
        
        combined = (p4 + p5) / 2.0
        return np.mean((combined - y_val)**2)

    if 'optuna' in locals():
        study_thresh = optuna.create_study(direction='minimize')
        study_thresh.optimize(objective_thresholds, n_trials=30)
        
        MetaConfig.M4_ALPHA = study_thresh.best_params['m4_alpha']
        MetaConfig.M5_ALPHA = study_thresh.best_params['m5_alpha']
        MetaConfig.M5_TAU = study_thresh.best_params['m5_tau']
        print(f"Phase 2 Complete. M4_ALPHA={MetaConfig.M4_ALPHA:.3f}, M5_TAU={MetaConfig.M5_TAU:.2e}")
    else:
        print("Optuna not available, skipping threshold tuning.")

except Exception as e:
    print(f"Threshold Tuning Failed ({e}). Using Defaults.")


# ==========================================
# 5. ENSEMBLE WEIGHT OPTIMIZATION (Phase 3)
# ==========================================
try:
    print("Phase 3: Tuning Ensemble Weights...")
    
    # Construct M1, M4, M5, M6 based on M3 using NEW TUNED PARAMS on Validation Set
    if 'pred_m3_val' not in locals():
        pred_m3_val = model_3.predict(X_val)
    
    pred_m1 = np.where(pred_m3_val > 0, 2.0, 0.0)
    pred_m4 = np.clip([MetaConfig.M4_ALPHA if x > 0 else 0.0 for x in pred_m3_val], 0.0, 2.0)
    pred_m5 = np.clip([MetaConfig.M5_ALPHA if x > MetaConfig.M5_TAU else 0.0 for x in pred_m3_val], 0.0, 2.0)
    pred_m6 = np.array([0.09 if x > 0 else 0.0 for x in pred_m3_val])
    pred_m2 = pred_m3_val # Simplified fallback
    
    preds_matrix = np.column_stack([pred_m1, pred_m2, pred_m3_val, pred_m4, pred_m5, pred_m6])
    
    def objective_weights(trial):
        w = [trial.suggest_float(f'w{i}', 0.0, 5.0) for i in range(6)]
        w_norm = np.array(w) / sum(w)
        final_pred = np.sum(preds_matrix * w_norm, axis=1)
        return np.mean((final_pred - y_val.values)**2)

    if 'optuna' in locals():
        study_weights = optuna.create_study(direction='minimize')
        study_weights.optimize(objective_weights, n_trials=30)
        
        best_w = [study_weights.best_params[f'w{i}'] for i in range(6)]
        total_w = sum(best_w)
        MetaConfig.WEIGHTS = [x/total_w for x in best_w]
        
        print("Weights Optimized: {}".format(np.round(MetaConfig.WEIGHTS, 3)))
    else:
        print("Optuna not available, skipping weight tuning.")
    
except Exception as e:
    print(f"Weight Tuning Failed ({e}). Using Equal Weights.")

# ==========================================
# 6. INFERENCE SERVER
# ==========================================

MIN_INVESTMENT = 0.0
MAX_INVESTMENT = 2.0

# ---- M4/M5 Helpers ----
def exposure_for_m4(r: float) -> float:
    if r <= 0.0: return 0.0
    return MetaConfig.M4_ALPHA

def exposure_for_m5(r: float) -> float:
    if r <= MetaConfig.M5_TAU: return 0.0
    return MetaConfig.M5_ALPHA

MIN_SIGNAL:        float = 0.0
MAX_SIGNAL:        float = 2.0
SIGNAL_MULTIPLIER: float = 400.0

@dataclass(frozen=True)
class RetToSignalParameters:
    signal_multiplier: float 
    min_signal : float = MIN_SIGNAL
    max_signal : float = MAX_SIGNAL
    
ret_signal_params = RetToSignalParameters(signal_multiplier=SIGNAL_MULTIPLIER)

# --- Model Functions ---

def predict_Model_1(test_pd) -> float:
    raw_pred = model_3.predict(test_pd)[0]
    return MAX_INVESTMENT if raw_pred > 0 else MIN_INVESTMENT

def predict_Model_2(test_pd) -> float:
    """
    CRITICAL FIX 3: REMOVED BROKEN LOOKUP LOGIC
    Model 2 is now purely a heuristic fallback.
    It takes the return prediction from Model 3, and scales it 
    into a signal using the predefined multiplier.
    """
    raw_pred_return = model_3.predict(test_pd)[0]
    
    # If raw_pred_return is 0.0025 -> * 400 = 1.0 -> + 1 = 2.0 (Max Investment)
    # If raw_pred_return is -0.01  -> * 400 = -4.0 -> + 1 = -3.0 (Clipped to 0)
    
    signal = raw_pred_return * ret_signal_params.signal_multiplier + 1
    return float(np.clip(signal, ret_signal_params.min_signal, ret_signal_params.max_signal))

def predict_Model_3(test_pd) -> float:
    return float(model_3.predict(test_pd)[0])

def predict_Model_4(test_pd) -> float:
    r = model_3.predict(test_pd)[0]
    return float(np.clip(exposure_for_m4(r), MIN_INVESTMENT, MAX_INVESTMENT))

def predict_Model_5(test_pd) -> float:
    r = model_3.predict(test_pd)[0]
    return float(np.clip(exposure_for_m5(r), MIN_INVESTMENT, MAX_INVESTMENT))

def predict_Model_6(test_pd) -> float:
    t = model_3.predict(test_pd)[0]
    return 0.09 if t > 0 else 0.0

def predict(test: pl.DataFrame) -> float:
    # 1. Prepare Data
    test_pd = test.to_pandas().drop(columns=["lagged_forward_returns", "date_id", "is_scored"], errors='ignore')
    test_pd = preprocessing(test_pd, "test")
    
    # 2. Get Predictions
    p1 = predict_Model_1(test_pd)
    p2 = predict_Model_2(test_pd)
    p3 = predict_Model_3(test_pd)
    p4 = predict_Model_4(test_pd)
    p5 = predict_Model_5(test_pd)
    p6 = predict_Model_6(test_pd)
    
    # 3. Weighted Ensemble
    w = MetaConfig.WEIGHTS
    pred = (p1*w[0] + p2*w[1] + p3*w[2] + p4*w[3] + p5*w[4] + p6*w[5])
    
    return pred

inference_server = kaggle_evaluation.default_inference_server.DefaultInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(('/kaggle/input/hull-tactical-market-prediction/',))