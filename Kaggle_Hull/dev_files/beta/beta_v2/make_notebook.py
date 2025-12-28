import json
import os

# Define the Markdown content
md_content = """# Hull Tactical Market Prediction (Gen6 Full Meta-Learning)

## Strategy Upgrade: "The Adaptive Ensemble"

This version implements a full **Meta-Learning Pipeline** to resolve the "Frozen Weights" issue.

### 1. Hyperparameter Optimization (Pre-Train)
Instead of just LGBM, we now use Optuna to tune **LGBM, XGBoost, and CatBoost** simultaneously.

### 2. Threshold & Exposure Optimization (Post-Train)
**NEW:** We optimize the "aggressiveness" of the strategy by tuning the exposure levels (`alpha`) and activation thresholds (`tau`) for Models 4 & 5 based on validation performance.

### 3. Ensemble Weight Optimization (Final Stage)
**CRITICAL FIX:** Previously, hardcoded scores (`10.15` vs `1.65`) silenced the ML model. 
Now, we run a final optimization loop to find the optimal mixing weights (`w1` to `w6`) for the current market regime.

### 4. Dynamic Inference
The final prediction uses these learned weights and parameters, ensuring the best models actually drive the decision.
"""

# Read the Python code
with open('gen6_strategy.py', 'r', encoding='utf-8') as f:
    code_content = f.read()

# Construct the Notebook JSON structure
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in md_content.splitlines()]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in code_content.splitlines()]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.5"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Write to file
with open('EOS_beta_v1.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Notebook created successfully.")
