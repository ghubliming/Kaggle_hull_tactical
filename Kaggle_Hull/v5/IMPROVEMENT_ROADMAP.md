# Strategic Roadmap & Improvement Opportunities

**Date:** December 4, 2025
**Context:** Post-Gen7 Implementation
**Scope:** Technical Debt, Alpha Research, and Risk Engineering

---

## 1. Signal Fidelity & Smoothing (The "Jitter" Problem)
**Current State:** The regime switch uses a Z-Score threshold. While robust, it is still a *binary* switch (0.2 vs 0.7 weights). This can cause "signal flickering" (rapidly toggling modes) around the threshold, incurring transaction costs (or theoretical slippage) and destabilizing the portfolio.

**Improvement:** **Sigmoid Transfer Functions**
Instead of:
```python
if z_score > 2.0: w = 0.7
else: w = 0.2
```
Use a continuous logistic function:
$$ w_{linear} = \text{Min}_W + \frac{\text{Max}_W - \text{Min}_W}{1 + e^{-k(z - z_{threshold})}} $$
*   **Benefit:** Smooth capital reallocation. No hard edges.
*   **Risk:** Adds complexity (`k` slope parameter) to tune.

---

## 2. Advanced Regime Identification
**Current State:** We rely solely on **Volatility Ratios** (Short/Long Vol) to define the "Regime."
**Critique:** High volatility doesn't always mean "crash." It can also mean "strong breakout trend." We might be selling our winners too early by getting defensive during a strong bull run.

**Improvement:** **The Hurst Exponent & Fractal Dimension**
*   **Concept:** Measure the "roughness" of the time series.
    *   $H < 0.5$: Mean Reverting (Chippy) $\rightarrow$ Favor Reversal Models.
    *   $H > 0.5$: Trending (Persistent) $\rightarrow$ Favor Trend Following (Linear/Momentum).
*   **Action:** Integrate a fast Hurst calculation (using `hurst` library or optimized NumPy) as a feature for the Meta-Learner.

---

## 3. Validation Rigor: "The Gold Standard"
**Current State:** We use `TimeSeriesSplit`. This is "okay" but loses data (early folds are small) and doesn't test all combinations.
**Improvement:** **Combinatorial Purged Cross-Validation (CPCV)**
*   **Concept:** As defined by Marcos López de Prado.
    *   **Purging:** Remove samples immediately after the training block to prevent "bleeding" of label information.
    *   **Embargoing:** Further delay reusing data after a test block.
    *   **Combinatorial:** Train on $(N-k)$ groups, test on $k$.
*   **Benefit:** drastically reduces "False Positive" strategies that look good in backtest but fail live.

---

## 4. Model Diversity (The Ensemble)
**Current State:** Linear (SGD) + Tree (LightGBM).
**Improvement:**
*   **CatBoost:** Often handles noisy financial data better than LGBM without extensive tuning. It handles categorical features (like "Day of Week" or "Quarter") natively.
*   **TabNet (PyTorch):** A deep learning architecture designed for tabular data. It provides "feature selection" built-in.
*   **Stacking:** Instead of a weighted average (`w * Lin + (1-w) * Tree`), train a *Meta-Model* (e.g., Logistic Regression) to decide the weights based on the inputs.

---

## 5. Data Engineering & Features
**Current State:** Standard OHLCV technicals (RSI, MACD, Vol).
**Improvement:**
*   **Higher Moments:** Skewness (Crash risk) and Kurtosis (Fat tails) of returns.
*   **Interaction Features:** `Vol * RSI` (High vol + Overbought = Danger).
*   **Fourier Transforms:** Decompose the price signal into frequency components to detect dominant cycles.

---

## 6. Engineering & DevOps
**Current State:** Monolithic Notebook (`.ipynb`).
**Critique:** Hard to test, version control, or collaborate on.
**Improvement:**
*   **Modularization:** Move `Config`, `feature_engineering`, and `objective` into a `.py` module (`hull_lib.py`).
*   **Unit Testing:** Add `pytest` for the feature generation logic. Ensure `calculate_rsi` returns correct values on known inputs.
*   **CI/CD:** Automate the "lint and run" check on GitHub Actions.

---

**Prioritization Recommendation:**
1.  **Sigmoid Smoothing:** High ROI, low effort.
2.  **Hurst Exponent:** Medium ROI, medium effort (computationally expensive).
3.  **CPCV:** High ROI, high effort (complex to implement correctly).
