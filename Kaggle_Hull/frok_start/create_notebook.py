import json

# This script creates the correct notebook structure
notebook = {
    "cells": [
        # Cell 1: Imports
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Imports"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "from pathlib import Path\n",
                "import datetime\n",
                "from typing import List\n",
                "\n",
                "from tqdm import tqdm\n",
                "from dataclasses import dataclass, asdict\n",
                "\n",
                "import polars as pl \n",
                "import numpy as np\n",
                "from sklearn.linear_model import Ridge, Lasso, LinearRegression\n",
                "from sklearn.ensemble import VotingRegressor\n",
                "from sklearn.preprocessing import StandardScaler, RobustScaler\n",
                "from sklearn.model_selection import TimeSeriesSplit\n",
                "from sklearn.metrics import mean_squared_error\n",
                "\n",
                "import kaggle_evaluation.default_inference_server"
            ]
        },
        # Cell 2: Directory Structure
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Project Directory Structure"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# This Python 3 environment comes with many helpful analytics libraries installed\n",
                "# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python\n",
                "# For example, here's several helpful packages to load\n",
                "\n",
                "# Input data files are available in the read-only \"../input/\" directory\n",
                "# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory\n",
                "\n",
                "import os\n",
                "for dirname, _, filenames in os.walk('/kaggle/input'):\n",
                "    for filename in filenames:\n",
                "        print(os.path.join(dirname, filename))\n",
                "\n",
                "# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using \"Save & Run All\" \n",
                "# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session"
            ]
        },
        # Cell 3: Configurations
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Configurations"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ============ PATHS ============\n",
                "DATA_PATH: Path = Path('/kaggle/input/hull-tactical-market-prediction/')\n",
                "\n",
                "# ============ RETURNS TO SIGNAL CONFIGS ============\n",
                "MIN_SIGNAL: float = 0.0                         # Minimum value for the daily signal \n",
                "MAX_SIGNAL: float = 2.0                         # Maximum value for the daily signal \n",
                "SIGNAL_MULTIPLIER: float = 400.0                # Multiplier of the OLS market forward excess returns predictions to signal \n",
                "\n",
                "# ============ MODEL CONFIGS ============\n",
                "N_SPLITS: int = 5                               # Number of time series cross validation splits\n",
                "RIDGE_ALPHAS: np.ndarray = np.logspace(-3, 3, 50)  # Ridge regularization parameters to test\n",
                "LASSO_ALPHAS: np.ndarray = np.logspace(-4, 1, 50)  # Lasso regularization parameters to test\n",
                "USE_ROBUST_SCALER: bool = True                  # Use RobustScaler (better for outliers) vs StandardScaler\n",
                "ENSEMBLE_WEIGHTS: dict = {'ridge': 0.5, 'lasso': 0.3, 'ols': 0.2}  # Weights for ensemble"
            ]
        },
        # Continue with remaining cells...
    ],
    "metadata": {
        "kaggle": {
            "accelerator": "none",
            "dataSources": [
                {
                    "databundleVersionId": 13750964,
                    "sourceId": 111543,
                    "sourceType": "competition"
                }
            ],
            "isGpuEnabled": False,
            "isInternetEnabled": False,
            "language": "python",
            "sourceType": "notebook"
        },
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
            "version": "3.11.13"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Write to file
with open('hull-notebook-1.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook created successfully!")
