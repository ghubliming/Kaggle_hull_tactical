# Hull Tactical Market Prediction - Methodology

## Overview
This implementation uses **industry-standard best practices** for financial time-series forecasting, replacing the basic ElasticNet approach with a more robust ensemble method.

---

## Key Improvements

### 1. Time-Series Cross-Validation (TimeSeriesSplit)
**Why it matters:**
- Financial data has temporal dependencies
- Standard K-Fold CV violates the temporal order, leading to look-ahead bias
- TimeSeriesSplit ensures we only train on past data to predict future

**Implementation:**
```python
TimeSeriesSplit(n_splits=5)
```
- Creates 5 sequential training/validation splits
- Each split trains on all previous data and validates on the next period
- Mimics real-world trading scenario

### 2. Ensemble of Regularized Linear Models
**Models used:**
1. **Ridge Regression (L2 regularization)**
   - Shrinks coefficients toward zero
   - Handles multicollinearity well
   - Stable predictions

2. **Lasso Regression (L1 regularization)**
   - Performs automatic feature selection
   - Sets some coefficients to exactly zero
   - Identifies most important features

3. **Ordinary Least Squares (OLS)**
   - No regularization
   - Captures pure linear relationships
   - Provides baseline comparison

**Ensemble Strategy:**
- Weighted voting: Ridge (50%), Lasso (30%), OLS (20%)
- Combines strengths of each model
- More robust than any single model
- Reduces overfitting risk

### 3. Robust Scaling (RobustScaler)
**Advantages over StandardScaler:**
- Uses median and IQR instead of mean and std
- Less sensitive to outliers
- Financial data often has fat-tailed distributions
- Prevents extreme values from dominating the model

### 4. Enhanced Feature Engineering
**New features added:**
- **U3**: `S2 / (S1 + ε)` - Ratio/momentum indicator
- **U4**: `E2 * E3` - Interaction term for non-linear relationships
- **U5**: `P9 + P10 + P12` - Aggregate feature to reduce noise

**Benefits:**
- Captures non-linear patterns
- Reduces dimensionality through aggregation
- Improves signal-to-noise ratio

### 5. Hyperparameter Optimization
**Ridge alphas:** `np.logspace(-3, 3, 50)` (50 values from 0.001 to 1000)
**Lasso alphas:** `np.logspace(-4, 1, 50)` (50 values from 0.0001 to 10)

- Systematic grid search over regularization parameters
- Validated using time-series splits
- Finds optimal bias-variance tradeoff

---

## Why This Approach?

### Academic Support
1. **Ensemble Methods**: Proven to reduce variance and improve generalization (Breiman, 1996)
2. **Regularization**: Essential for high-dimensional financial data (Hastie et al., 2009)
3. **Time-Series CV**: Standard practice in quantitative finance (Lopez de Prado, 2018)

### Industry Adoption
- Used by quantitative hedge funds and trading firms
- Robust to market regime changes
- Balances complexity with interpretability
- Computationally efficient for production systems

### Practical Benefits
- **Stability**: Ensemble reduces model variance
- **Robustness**: Multiple models provide redundancy
- **Interpretability**: Linear models are transparent
- **Speed**: Fast training and inference
- **Scalability**: Easy to deploy in production

---

## Performance Metrics
The model reports:
- **MSE (Mean Squared Error)**: Primary optimization metric
- **MAE (Mean Absolute Error)**: Robust to outliers

Both metrics are evaluated on the held-out test set using proper temporal validation.

---

## References
- Breiman, L. (1996). "Bagging predictors." Machine Learning.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). "The Elements of Statistical Learning."
- Lopez de Prado, M. (2018). "Advances in Financial Machine Learning."
- Bergstra, J. & Bengio, Y. (2012). "Random Search for Hyper-Parameter Optimization."

---

## Configuration Parameters

```python
N_SPLITS = 5                    # Time-series CV splits
RIDGE_ALPHAS = logspace(-3, 3, 50)  # Ridge regularization range
LASSO_ALPHAS = logspace(-4, 1, 50)  # Lasso regularization range
USE_ROBUST_SCALER = True        # Use RobustScaler vs StandardScaler
ENSEMBLE_WEIGHTS = {            # Model weights in ensemble
    'ridge': 0.5,
    'lasso': 0.3,
    'ols': 0.2
}
```

These can be tuned based on specific market conditions and competition requirements.
