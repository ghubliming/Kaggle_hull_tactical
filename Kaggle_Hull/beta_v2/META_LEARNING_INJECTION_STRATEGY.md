# Meta-Learning Injection Strategy (Non-Destructive)

**Objective:** Integrate the "Gen6" Meta-Learning engine (Optuna) into your existing `EOS_beta_v1.ipynb` without changing your model architecture.

**Core Concept:** Replace your *hardcoded dictionaries* and *fixed constants* with dynamic variables determined by an optimization loop running at the start of the notebook.

---

## 1. Setup: The "Safe" Optimization Block

Insert this at the very top of your notebook (after imports). It ensures the code doesn't crash if Optuna isn't available.

```python
# --- META-LEARNING SETUP ---
OPTUNA_AVAILABLE = False
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING) # Keep it quiet
    OPTUNA_AVAILABLE = True
    print("✅ Optuna detected. Meta-Learning engaged.")
except ImportError:
    print("⚠️ Optuna not found. Using Static Fallback values.")

# Global Config to hold the "Winners"
class MetaConfig:
    # Default / Fallback values (Your current hardcoded values)
    # Model 3 Hyperparameters
    LGBM_LR = 0.05
    LGBM_LEAVES = 50
    XGB_LR = 0.05
    CAT_LR = 0.01
    
    # Thresholds for Models 4 & 5
    M4_ALPHA = 0.80007
    M5_ALPHA = 0.60013
    M5_TAU = 9.437e-05
    
    # Ensemble Weights (The 'scores' array)
    # [M1, M2, M3, M4, M5, M6]
    BMA_SCORES = [10.15, 8.09, 1.65, 10.16, 10.22, 10.24]
```

---

## 2. Target 1: Model 3 Hyperparameters (The "Heavy Lifters")

Replace your hardcoded dictionaries (`LGBM_R_parm`, etc.) with this logic. This allows the notebook to "tune itself" to the specific dataset it loads.

**Insert this code block BEFORE the `Model_3` training cell:**

```python
def optimize_model3(X, y):
    if not OPTUNA_AVAILABLE: return

    def objective(trial):
        # 1. Define the Search Space
        lgbm_lr = trial.suggest_float('lgbm_lr', 0.01, 0.1)
        lgbm_leaves = trial.suggest_int('lgbm_leaves', 20, 100)
        xgb_lr = trial.suggest_float('xgb_lr', 0.01, 0.1)
        
        # 2. Setup Models with these Trial params
        # (Simplified for speed - using just one fold or a subset for tuning)
        model_lgbm = LGBMRegressor(learning_rate=lgbm_lr, num_leaves=lgbm_leaves, 
                                   n_estimators=500, verbosity=-1, random_state=42)
        
        # 3. Quick Validation (CV)
        # Use negative MSE because Optuna minimizes
        scores = cross_val_score(model_lgbm, X, y, cv=3, scoring='neg_mean_squared_error')
        return -scores.mean()

    print("⚡ Tuning Model 3 Hyperparameters...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20) # keep n_trials low for Kaggle limits

    # Update the Global Config with winners
    MetaConfig.LGBM_LR = study.best_params['lgbm_lr']
    MetaConfig.LGBM_LEAVES = study.best_params['lgbm_leaves']
    if 'xgb_lr' in study.best_params: MetaConfig.XGB_LR = study.best_params['xgb_lr']
    print(f"✅ Updated Params: LGBM_LR={MetaConfig.LGBM_LR}")

# Run the optimization on your training data
optimize_model3(X_train, y_train)
```

**Then, update your dictionary definitions to use `MetaConfig`:**

```python
LGBM_R_parm = {
    "n_estimators": 1500,
    "learning_rate": MetaConfig.LGBM_LR, # <--- Dynamic now!
    "num_leaves": MetaConfig.LGBM_LEAVES, # <--- Dynamic now!
    # ... other static params
}
```

---

## 3. Target 2: Threshold Optimization (Models 4 & 5)

Your notebook currently says `# Best parameters from Optuna`. This implies they were found offline. Move that logic *online*.

**Insert this before defining `predict_Model_4` and `predict_Model_5`:**

```python
def optimize_thresholds(model3_preds, y_true):
    """Find best thresholds using the OOF predictions from Model 3"""
    if not OPTUNA_AVAILABLE: return

    def objective(trial):
        # Search for the best exposure cap
        alpha = trial.suggest_float('m4_alpha', 0.5, 1.5)
        
        # Simulate Model 4 Logic
        preds = np.clip([alpha if x > 0 else 0.0 for x in model3_preds], 0.0, 2.0)
        
        # Calculate metric (e.g., RMSE or Sharpe)
        mse = np.mean((preds - y_true)**2)
        return mse

    print("⚡ Tuning Thresholds...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    
    MetaConfig.M4_ALPHA = study.best_params['m4_alpha']
    print(f"✅ Optimized M4 Alpha: {MetaConfig.M4_ALPHA}")

# You need OOF (Out-of-Fold) predictions from Model 3 to run this effectively.
# If you don't have full OOF, you can run this on the validation split (y_test).
# optimize_thresholds(model_3.predict(X_test), y_test)
```

---

## 4. Target 3: The Ensemble Weights (BMA)

Currently, you use fixed scores: `scores = np.array([10.15, 8.09...])`.
Let the Meta-Learner find the perfect balance for *this* specific dataset.

**Insert this in the final `predict` block (or just before it):**

```python
def optimize_ensemble_weights(oof_preds_df, y_true):
    """
    oof_preds_df: DataFrame where cols are [pred_m1, pred_m2, ..., pred_m6]
    """
    if not OPTUNA_AVAILABLE: return

    def objective(trial):
        # Suggest weights for all 6 models
        w1 = trial.suggest_float('w1', 0, 10)
        w2 = trial.suggest_float('w2', 0, 10)
        w3 = trial.suggest_float('w3', 0, 10)
        w4 = trial.suggest_float('w4', 0, 10)
        w5 = trial.suggest_float('w5', 0, 10)
        w6 = trial.suggest_float('w6', 0, 10)
        
        weights = np.array([w1, w2, w3, w4, w5, w6])
        weights /= weights.sum() # Normalize
        
        # Weighted sum of predictions
        final_pred = (oof_preds_df.values * weights).sum(axis=1)
        
        return np.mean((final_pred - y_true)**2) # Minimize MSE

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    
    # Update global scores
    best = study.best_params
    MetaConfig.BMA_SCORES = [best['w1'], best['w2'], best['w3'], best['w4'], best['w5'], best['w6']]
    print(f"✅ Optimized Ensemble Weights: {MetaConfig.BMA_SCORES}")
```

---

## Summary of Changes

1.  **No Architecture Changes:** You keep Models 1-6 and the StackingRegressor exactly as they are.
2.  **Dynamic Constants:** You replace the hard numbers (`0.05`, `0.8007`, `[10.15...]`) with variables from `MetaConfig`.
3.  **Optimization Layer:** You add the 3 functions above (`optimize_model3`, `optimize_thresholds`, `optimize_ensemble_weights`) to populate `MetaConfig` *before* the final training/inference runs.

This gives you the "Self-Adapting" capability of Gen6 while preserving your specific beta strategy.