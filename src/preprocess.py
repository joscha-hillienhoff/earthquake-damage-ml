"""Utility functions for data preprocessing in the earthquake damage prediction project.

Includes helpers for encoding, splitting, aligning features, outlier removal,
and class balancing with SMOTE.
"""

from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.model_selection import train_test_split


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """Apply one-hot encoding to categorical features."""
    return pd.get_dummies(df)


def split_train_val(X, y, test_size=0.2, random_state=42):
    """Split into train and validation sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def align_features(train_df, test_df):
    """Ensure train and test sets have the same features."""
    missing_cols = set(train_df.columns) - set(test_df.columns)
    for c in missing_cols:
        test_df[c] = 0
    return test_df[train_df.columns]


def remove_outliers(df: pd.DataFrame, column: str, threshold: float):
    """Remove rows where values in column exceed threshold."""
    return df[df[column] <= threshold]


def smote_oversample(X_train, y_train, random_state=42):
    """Apply SMOTE oversampling to balance classes."""
    sm = SMOTE(random_state=random_state)
    X_resampled, y_resampled = sm.fit_resample(X_train, y_train.values.ravel())
    return X_resampled, y_resampled
