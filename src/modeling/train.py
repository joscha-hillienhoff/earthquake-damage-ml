"""Utilities for building and training machine learning models.

This module provides helper functions to create scikit-learn pipelines
and perform Bayesian hyperparameter optimization. It currently supports
both Random Forest and XGBoost models.
"""

from typing import Any, Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
import typer
from xgboost import XGBClassifier

app = typer.Typer()


def build_rf_pipeline(rf_params: Dict[str, Any] | None = None):
    """
    Build a scikit-learn pipeline with a StandardScaler and RandomForestClassifier.

    The StandardScaler is configured with `with_mean=False` to support sparse
    matrices (e.g., after one-hot encoding).

    Args:
        rf_params (dict | None): Optional dictionary of hyperparameters for RandomForest.

    Returns:
        sklearn.pipeline.Pipeline: Configured pipeline for Random Forest.
    """
    pipe = make_pipeline(
        # One-Hot -> sparse-kompatibel; funktioniert auch mit dichten DataFrames
        StandardScaler(with_mean=False),
        RandomForestClassifier(rf_params),
    )
    return pipe


def build_xgboost_pipeline(xgboost_params: Dict[str, Any] | None = None):
    """
    Build a scikit-learn pipeline with a StandardScaler and XGBClassifier.

    The StandardScaler is configured with `with_mean=False` to support sparse
    matrices (e.g., after one-hot encoding).

    Args:
        xgboost_params (dict | None): Optional dictionary of hyperparameters for XGBoost.

    Returns:
        sklearn.pipeline.Pipeline: Configured pipeline for XGBoost.
    """
    pipe = make_pipeline(
        # One-Hot -> sparse-kompatibel; funktioniert auch mit dichten DataFrames
        StandardScaler(with_mean=False),
        (XGBClassifier(xgboost_params)),
    )
    return pipe


def smbo(pipe, X_train, y_train, search_spaces, n_iter, cv):
    """
    Perform Bayesian hyperparameter optimization using BayesSearchCV.

    Args:
        pipe (Pipeline): Model pipeline to optimize.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series or pd.DataFrame): Training labels (expects column 'damage_grade').
        search_spaces (dict): Search space for hyperparameters.
        n_iter (int): Number of iterations for the search.
        cv (int): Number of cross-validation folds.

    Returns:
        BayesSearchCV: Fitted optimizer with best parameters and estimator.
    """
    opt = BayesSearchCV(
        pipe,
        search_spaces=search_spaces,
        n_iter=n_iter,
        cv=cv,
        scoring=make_scorer(f1_score, average="micro"),
        random_state=123,
    )
    # Modell auf Trainingsdaten anpassen
    np.int = int
    opt.fit(X_train, y_train["damage_grade"].values.ravel())

    return opt


# @app.command()
# def main(
#     # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
#     features_path: Path = PROCESSED_DATA_DIR / "features.csv",
#     labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
#     model_path: Path = MODELS_DIR / "model.pkl",
#     # -----------------------------------------
# ):
#     # ---- REPLACE THIS WITH YOUR OWN CODE ----
#     logger.info("Training some model...")
#     for i in tqdm(range(10), total=10):
#         if i == 5:
#             logger.info("Something happened for iteration 5.")
#     logger.success("Modeling training complete.")
#     # -----------------------------------------


if __name__ == "__main__":
    app()
