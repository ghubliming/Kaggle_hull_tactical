# Hull Tactical Gen6: Codebase Explanation

**Status:** Operational
**Version:** Gen6 "The Adaptive Meta-Agent"
**Focus:** Dynamic Hyperparameter Optimization & Regime-Based Ensembling

---

## 1. Overview

The **Gen6** codebase represents a paradigm shift from static parameter tuning to **Meta-Learning**. Instead of relying on fixed lookback windows (e.g., "22 days for monthly volatility"), the system uses **Bayesian Optimization (Optuna)** to discover the optimal feature parameters and model hyperparameters for the specific market regime it encounters. This is designed to run within the constrained environment of a Kaggle competition, featuring a robust "Offline Fallback" to ensure execution even when external libraries are unavailable.

## 2. Architectural Pillars

The solution is built on three main pillars:

1.  **The Config Core:** A central configuration class that holds both the *default* values (safe fallbacks) and the *search space* (for optimization).
2.  **The Feature Factory:** A function that generates technical indicators. Crucially, it is **state-dependent**, generating features based on the current values in the `Config` class.
3.  **The Meta-Learner:** An isolated optimization loop that tests thousands of parameter combinations to update the `Config` class before the final model is trained.

---

## 3. Detailed Code Breakdown

### A. Configuration & Environment Setup

The code begins by establishing a safe execution environment. It attempts to import `optuna` for the meta-learning phase but wraps this in a `try/except` block.

```python
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not found. Using Gen5 Defaults...")
```

The `Config` class acts as the **Single Source of Truth**. It defines the "Gen5 Defaults," which are proven, robust parameters used if optimization fails or is skipped.

```python
class Config:
    # Dynamic Feature Windows (Defaults)
    VOL_SHORT = 5      # 1 week
    VOL_LONG = 22      # 1 month
    VOL_QUARTERLY = 66 # 1 quarter
    
    # Trading Logic
    BASE_W_LINEAR = 0.4
    TARGET_VOL = 0.005 # Daily target volatility (0.5%)
```

### B. Dynamic Feature Engineering

The `feature_engineering` function is the engine of the strategy. Unlike traditional functions that take parameters as arguments, this function reads directly from the global `Config` class. This allows the Meta-Learner to modify the `Config` and immediately see the effect on feature quality without rewriting the pipeline.

**Key Indicators:**
*   **Volatility:** Calculated over dynamic windows (`VOL_SHORT`, `VOL_LONG`).
*   **Momentum:** Exponential Moving Averages (EMA) and MACD.
*   **Regime Signals:** Ratios of short-term to long-term volatility.

$$ \text{Vol Ratio} = \frac{\sigma_{short}}{\sigma_{quarterly}} $$

This ratio is critical for the "Flash" regime detection logic.

### C. The Meta-Learner (`objective` function)

This is the brain of Gen6. It defines the search space for **Optuna**.

1.  **Parameter Suggestion:** It suggests integers for window sizes and floats for LightGBM hyperparameters.
    ```python
    Config.VOL_SHORT = trial.suggest_int('vol_short', 3, 10)
    Config.VOL_LONG = trial.suggest_int('vol_long', 15, 30)
    ```
2.  **Validation Strategy:** It uses `TimeSeriesSplit` (Walk-Forward Validation) to ensure that we are not peeking into the future. The model is trained on past data and validated on "future" data within the training set.
3.  **Metric:** It optimizes for **Mean Squared Error (MSE)** on the validation folds.

### D. Model Training (Hybrid Ensemble)

The strategy employs a **Hybrid Ensemble** approach, combining a linear model with a gradient-boosted tree model.

1.  **Linear Model (`SGDRegressor`):** Captures the broad, linear trends in the market. It is regularized (L2) to prevent overfitting.
2.  **Tree Model (`LGBMRegressor`):** Captures complex, non-linear interactions between features. Its hyperparameters are fully optimized by the Meta-Learner.

```python
# Train Linear Model
linear_model = SGDRegressor(...)
linear_model.fit(X_scaled, y)

# Train Tree Model (with optimized params)
lgbm_model = LGBMRegressor(**Config.LGBM_PARAMS)
lgbm_model.fit(X, y)
```

### E. Inference & Regime Detection

The `predict` function is called for every new data point. It implements the **"Flash" Regime Logic**.

**The Logic:**
We calculate the ratio of short-term volatility to long-term volatility.
*   If $\frac{\sigma_{short}}{\sigma_{long}} > 1.3$: The market is turbulent. We switch to **Defensive Mode**. The weight of the Linear model ($w_{linear}$) increases to 0.7 (70%), as linear models are generally more stable in high noise.
*   If $\frac{\sigma_{short}}{\sigma_{long}} < 0.8$: The market is calm. We switch to **Aggressive Mode**. The weight of the Tree model increases ($w_{linear}$ drops to 0.2), allowing the complex model to hunt for alpha.

**Risk Control (Volatility Targeting):**
The final position size is scaled to achieve a constant daily volatility target.

$$ \text{Allocation} = \frac{\text{Target Vol}}{\text{Current Vol}} \times \frac{|\text{Prediction}|}{\text{Current Vol}} $$

This ensures that we trade smaller sizes when the market is risky and larger sizes when it is calm.

**RSI Sanity Check:**
A final guardrail prevents buying into an overbought market (RSI > 75) or selling into an oversold one (RSI < 25).

---

## 4. Summary of Flow

1.  **Initialize:** Check for Optuna.
2.  **Meta-Learn:** If possible, run `objective` loop to find best `Config` values.
3.  **Update Config:** Overwrite `Config` with best found values.
4.  **Train:** Fit final Linear and Tree models using these optimal settings.
5.  **Serve:** In the inference loop, dynamically calculate features and regime weights to generate the final allocation.

```