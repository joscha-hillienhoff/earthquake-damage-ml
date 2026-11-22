import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.modeling import train


def _toy_data(n=60):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.DataFrame({"damage_grade": np.arange(n) % 3})
    return X, y


def test_build_rf_pipeline_passes_hyperparameters():
    pipe = train.build_rf_pipeline({"n_estimators": 7, "max_depth": 2})

    rf = pipe.named_steps["randomforestclassifier"]
    assert isinstance(rf, RandomForestClassifier)
    assert rf.n_estimators == 7 and rf.max_depth == 2


def test_build_xgboost_pipeline_passes_hyperparameters():
    pipe = train.build_xgboost_pipeline({"n_estimators": 5, "max_depth": 2})

    xgb = pipe.named_steps["xgbclassifier"]
    assert isinstance(xgb, XGBClassifier)
    assert xgb.n_estimators == 5 and xgb.max_depth == 2


def test_default_pipelines_use_library_defaults():
    rf = train.build_rf_pipeline().named_steps["randomforestclassifier"]

    assert rf.n_estimators == RandomForestClassifier().n_estimators


def test_smbo_returns_fitted_optimizer():
    X, y = _toy_data()
    pipe = train.build_rf_pipeline({"random_state": 0})
    space = {"randomforestclassifier__n_estimators": (5, 10)}

    opt = train.smbo(pipe, X, y, space, n_iter=2, cv=2)

    assert 5 <= opt.best_params_["randomforestclassifier__n_estimators"] <= 10
    assert opt.predict(X).shape == (len(X),)
