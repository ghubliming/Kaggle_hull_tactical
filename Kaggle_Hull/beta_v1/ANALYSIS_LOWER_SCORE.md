# Post-Action Analysis: Why "Smart" Logic Can Lower Scores

**Observation:** The Gen6 Adaptive Ensemble achieved a lower score on the Local Board (LB) compared to the static Gen5 version.

**Verdict:** This is a classic case of **"Regime Mismatch"** or **"Validation Overfitting."**

Here is the technical breakdown of why this happens and how to stabilize it.

---

## 1. The "Recency Bias" Trap
**The Mechanism:**
In Phase 1, we tuned hyperparameters on the **last 20%** of the training data.
> `subset_idx = int(len(X_train) * 0.8)`

**The Risk:**
*   **Theory:** Markets evolve. Recent data is the best predictor of the future.
*   **Reality:** If the Public LB (Test Set) covers a period that behaves like the *older* 80% of the data (e.g., a low-volatility period), and we tuned strictly for the *new* 20% (e.g., a high-volatility period), our "optimized" model is actually **mis-calibrated** for the test set.
*   **Fix:** Expand the tuning window to `0.5` (50%) or use `TimeSeriesSplit` to tune across multiple time periods, not just the tail end.

## 2. Overfitting the Weights (Phase 2)
**The Mechanism:**
We used Optuna to find weights `w1..w6` that minimized MSE on `X_test` (a small hold-out slice).

**The Risk:**
*   If `X_test` is small (e.g., 100 days), Optuna might find that "Model 1" got lucky on 3 big days and assign it 50% weight.
*   This "perfect" weight distribution works great for that specific `X_test` slice but fails on the unseen LB data.
*   **Static was safer:** The original hardcoded scores were likely derived from a very long-term historical average, making them "dumber" but more robust.

## 3. The Metric Trap (MSE vs. Returns)
**The Mechanism:**
We optimized for **Mean Squared Error (MSE)**.
> `return np.mean((final_pred - y_test.values)**2)`

**The Risk:**
*   In finance, the "safest" way to minimize MSE is often to predict numbers close to 0.
*   However, profitable trading requires capturing the *direction* (sign) and *magnitude* of moves.
*   By aggressively minimizing MSE, Optuna might have favored "conservative" models that predict `0.001` over "aggressive" models (like Model 3) that predict `0.05` but are occasionally wrong.
*   **Fix:** Change the Optuna objective to maximize **Information Coefficient (IC)** (correlation) or **Sharpe Ratio**.

---

## 🚀 Immediate Action Plan (To Fix LB)

If you want to recover the score while keeping the intelligence, apply these changes to `gen6_strategy.py`:

1.  **Broaden the View:**
    Change `subset_idx = int(len(X_train) * 0.8)` to `0.5` or even `0.0` (use full data for tuning).

2.  **Stabilize Phase 2:**
    Instead of 30 trials for weights, restrict the search space.
    *   *Current:* `trial.suggest_float(..., 0.0, 5.0)` (Wild swings allowed)
    *   *Robust:* `trial.suggest_float(..., 0.8, 1.2)` (Only fine-tune the legacy weights, don't rewrite them).

3.  **Check Model 3:**
    If Model 3 was originally silenced, it might be truly bad.
    *   *Test:* Run the notebook but force `weights = [0, 0, 1, 0, 0, 0]` (Only Model 3). If the score tanks, Model 3 is the problem, and our "Optimization" just amplified a bad signal.
