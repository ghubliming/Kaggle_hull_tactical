# Gen6 Meta-Learning & Adaptive Strategy

This document outlines the "Gen6" architecture, which shifts from hardcoded parameters to a self-tuning "Meta-Learning" system. It runs a safe optimization phase at the start of execution to adapt to the specific dataset provided.

## 1. Architecture Overview

The system operates in three phases:

1.  **Phase 1: Hyperparameter Meta-Learning**
    *   **Goal:** Find the best structural parameters for the heavy boosters (LGBM, XGBoost, CatBoost).
    *   **Method:** Optuna study on a subset of `train.csv` to tune `learning_rate`, `depth`, and `num_leaves`.
    *   **Output:** Optimized `MetaConfig` parameters.

2.  **Phase 2: Threshold & Exposure Meta-Learning**
    *   **Goal:** Determine the optimal "aggressiveness" for the threshold-based models (Model 4 & Model 5).
    *   **Method:** Optuna study using Out-of-Fold (OOF) predictions from the stacked ensemble. It finds the `alpha` (exposure) and `tau` (threshold) that maximize Sharpe or minimize MSE.
    *   **Output:** Optimized `MetaConfig.M4_ALPHA`, `MetaConfig.M5_ALPHA`, etc.

3.  **Phase 3: Ensemble Weight Meta-Learning**
    *   **Goal:** Find the optimal mixing weights for the 6-model ensemble.
    *   **Method:** Optuna study to minimize MSE on the validation set by mixing the signals from Models 1-6.
    *   **Output:** Optimized `MetaConfig.WEIGHTS`.

## 2. The Configuration Class (`MetaConfig`)

A central configuration class replaces hardcoded constants.

```python
class MetaConfig:
    # Tunable Hyperparams
    LGBM_LR = 0.05
    LGBM_LEAVES = 50
    XGB_LR = 0.05
    CAT_LR = 0.01
    
    # Tunable Thresholds
    M4_ALPHA = 0.80  # Default
    M5_ALPHA = 0.60  # Default
    M5_TAU = 1e-4    # Default
    
    # Tunable Weights
    WEIGHTS = [0.16, 0.16, 0.2, 0.16, 0.16, 0.16] # Default equal split
```

## 3. Implementation Details

### Safety Checks
All optimization blocks are wrapped in `try-except` or check for `OPTUNA_AVAILABLE`. If Optuna is missing (or errors out), the system silently falls back to safe, hardcoded defaults.

### Performance
To adhere to Kaggle time limits:
*   Tuning runs on a **subset** of data or limited folds.
*   `n_trials` is kept low (15-30).
*   Booster parameters are injected *before* the main training loop.

### Online Learning Note
True "Online Learning" (updating weights during the inference loop) is **disabled** to prevent timeout risks and state management issues in the Kaggle environment. The strategy relies on the robustness of the Meta-Learned parameters.