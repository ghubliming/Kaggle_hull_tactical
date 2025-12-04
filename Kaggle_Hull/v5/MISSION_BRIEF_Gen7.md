# MISSION BRIEF - Gen7 "The Antifragile Agent"

**Date:** December 4, 2025
**Status:** 🟢 READY FOR DEPLOYMENT
**Foundation:** Gen6 "Adaptive Meta-Agent" + Robust Z-Score Logic

---

## 📜 RETROSPECTIVE: The "Gen6" Pivot
The Gen6 mission successfully introduced **Meta-Learning**, allowing the model to find its own optimal window sizes (`vol_short`, `vol_long`). However, a critical fragility was identified during the review:
*   **The Flaw:** The "Flash Crash" detector relied on a hardcoded ratio threshold (`1.3`).
*   **The Risk:** In high-volatility regimes (like 2020), `1.3` is normal noise. In low-vol regimes (2017), `1.3` is the apocalypse. A static number fails to adapt to the *market's changing baseline*.
*   **The Fix:** We implemented a **Dynamic Z-Score** mechanism. The model now calculates how many standard deviations (Sigmas) the current volatility is away from the *recent* mean.

---

## 🎯 MISSION OBJECTIVE: Gen7

**"From Robust to Antifragile."**
Now that the crash detector adapts to the market's baseline, the next goal is to **optimize the sensitivity** and **smooth the reaction**. We move from binary "switches" to continuous adaptation.

---

## 🏗️ ARCHITECTURAL UPGRADES

### 1. The Z-Score Core (Implemented)
The codebase now maintains a rolling history of volatility ratios.
*   **Logic:** `z_score = (current_ratio - rolling_mean) / rolling_std`
*   **Trigger:** If `z_score > crash_sensitivity` (Default: 2.0), Defensive Mode is engaged.
*   **Improvement:** This works universally across any decade or asset class, regardless of absolute volatility levels.

### 2. The "Sigmoid" Weighting (Next Step)
*   **Current:** Hard switch. If `z > 2.0`, weight = 0.7. Else 0.4.
*   **Gen7 Goal:** Implement a **Sigmoid Transfer Function**.
    *   $$ w_{linear} = \frac{1}{1 + e^{-k(z - z_0)}} $$
    *   This allows a smooth transition from "Aggressive" to "Defensive" as the Z-score rises, preventing "flickering" signal behavior at the threshold.

### 3. Hyperparameter Expansion
The Meta-Learner (`objective` function) must now tune the **Robustness Parameters** alongside the window sizes.

| Parameter | Type | Range | Description |
| :--- | :--- | :--- | :--- |
| `crash_sensitivity` | Float | `1.5` - `3.0` | How "jumpy" the defensive trigger is (in Sigmas). |
| `lookback_window` | Int | `50` - `200` | How much history to use for the Z-score baseline. |

---

## 🔬 CODE STRUCTURE & RULES

### A. The State Vector
We now have state that persists across time steps (`ratio_history`).
*   **Rule:** Ensure this history is robust to empty starts. (Already handled: defaults to `Config.BASE_W_LINEAR` if history < 20 days).

### B. Optimization Logic
*   **Constraint:** We must not optimize `crash_sensitivity` on the *training* set crashes alone (overfitting).
*   **Strategy:** Use **Purged Walk-Forward Validation**. Ensure the validation folds contain a mix of calm and chaotic periods.

### C. The "Code as Truth"
The `Hull_AOE_Fin.ipynb` file is the master record.
*   **Section 6 (Inference):** Now contains the `ratio_history` global buffer.
*   **Section 4 (Optuna):** Needs to be updated to include `trial.suggest_float('crash_sensitivity', ...)` in the next iteration.

---

## 🚀 EXECUTION PLAN

1.  **Verify Gen6 Robustness:** Ensure the new Z-score logic runs without errors on the test stream.
2.  **Update Optuna:** Add `crash_sensitivity` to the search space.
3.  **Smooth the Curve:** Replace the `if/elif` block in `get_adaptive_weights` with a continuous function (Sigmoid or Linear Interpolation) if performance tests suggest "signal flickering" is hurting Sharpe.

**Outcome Goal:** A model that doesn't just survive a regime shift, but *identifies it statistically* and re-allocates capital smoothly without human intervention.
