# MISSION BRIEF - Gen6 "The Adaptive Meta-Agent"

**Date:** December 2, 2025
**Status:** 🚀 INITIATED
**Foundation:** Gen5 "Lean & Fast" + Meta-Learning Prototype

---

## 🎯 MISSION OBJECTIVE

**Transition from "Static Optimization" to "Dynamic Adaptation".**

The Gen5 update successfully purged noise and established a robust "Lean & Fast" baseline. The Gen6 mission is to operationalize **Meta-Learning** as the standard operating procedure. The model should no longer rely on hardcoded "magic numbers" (like `vol_22d`) but instead autonomously discover the optimal feature windows and hyperparameters for the given data context.

---

## 🔬 CORE PHILOSOPHY: "The Code Finds the Alpha"

We are moving away from *us* guessing parameters to *the system* finding them.

### 1. The Meta-Learning Standard
*   **Concept:** Hyperparameters (window sizes, regularization terms, learning rates) are treated as fluid variables, not constants.
*   **Mechanism:** We utilize **Bayesian Optimization (Optuna)** to navigate the search space efficiently.
*   **Safety:** The system **MUST** have a robust "Offline Fallback" mode. If the optimization library is missing or the compute budget is tight, it instantly reverts to the proven "Gen5" static configuration.

### 2. Dynamic Feature Engineering
*   **Old Way:** Hardcoded `vol_5d` and `vol_22d`.
*   **Gen6 Way:** `vol_short` and `vol_long` are variables determined by the Meta-Learner.
*   **Logic:** In some decades, a 3-day lag might be the best "fast" signal; in others, it might be 7. The model adapts to the dataset's frequency response.

### 3. Walk-Forward Integrity
*   **Risk:** Automated tuning is prone to overfitting (Look-Ahead Bias).
*   **Solution:** Strict **Walk-Forward Validation**. The Meta-Learner optimizes on a *past* validation slice, and the model is tested on a *future* hold-out slice. We never tune on the test set.

---

## 📊 ARCHITECTURAL BLUEPRINT

### A. The "Hybrid" Code Structure
The codebase is designed to be self-contained and resilient.

1.  **The Configuration Core (`Config` Class):**
    *   Holds both the *Search Space* (for optimization) and the *Default Values* (for fallback).
2.  **The Feature Factory:**
    *   Accepts `Config` variables (e.g., `Config.VOL_SHORT`) to generate features dynamically.
3.  **The Optimizer (`objective` function):**
    *   Isolated logic that runs *only* if dependencies exist.
    *   Uses `TimeSeriesSplit` to respect temporal causality.

### B. The "Flash" Regime Logic (Refined)
We retain the Gen5 breakthrough of "Instant" regime detection but make the threshold adaptive.
*   **Signal:** `vol_ratio = vol_short / vol_quarterly`
*   **Logic:**
    *   High Ratio (> 1.3) $\rightarrow$ **Defensive Mode** (High Linear Weight, Low Leverage).
    *   Low Ratio (< 0.8) $\rightarrow$ **Aggressive Mode** (High Tree Weight, Normal Leverage).

---

## 🚀 IMPLEMENTATION GUIDELINES (For the AI Agent)

When working on Gen6, adhere to these rules:

1.  **Zero External Dependencies (Runtime):** The code *must* run even if `optuna` is not installed. Always wrap imports in `try/except` blocks.
2.  **Code as Truth:** The `MISSION_BRIEF` and the `ipynb` code are the only sources of truth. Do not assume external libraries or files exist.
3.  **Optimization Speed:** The search space in the `objective` function must be tight. We cannot afford 1000 trials. Use `n_trials=20` and narrow ranges to ensure quick convergence.
4.  **Collinearity Control:** Even with dynamic features, strictly limit the number of indicators. Do not re-introduce KDJ or Bollinger Bands unless they win in the optimization arena (they likely won't).

---

**Outcome Goal:** A single notebook that automatically "upgrades" itself when running in a capable environment, while remaining a rock-solid, error-free tactical model in restricted offline environments.
