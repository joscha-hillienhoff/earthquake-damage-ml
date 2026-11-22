import pandas as pd
import pytest

from src import preprocess


@pytest.fixture
def train_features():
    return pd.DataFrame(
        {
            "age": [5, 10, 15, 20],
            "roof_type": ["n", "q", "x", "n"],
            "land_surface_condition": ["t", "o", "n", "t"],
        }
    )


@pytest.fixture
def test_features():
    # Lacks the roof level "x" and the land surface level "o", has an unseen level "z".
    return pd.DataFrame(
        {
            "age": [7, 12],
            "roof_type": ["n", "z"],
            "land_surface_condition": ["t", "n"],
        }
    )


def test_one_hot_encode_expands_categoricals_and_keeps_numerics(train_features):
    encoded = preprocess.one_hot_encode(train_features)

    assert "age" in encoded.columns
    assert {"roof_type_n", "roof_type_q", "roof_type_x"} <= set(encoded.columns)
    assert "roof_type" not in encoded.columns
    assert len(encoded) == len(train_features)


def test_one_hot_encode_produces_different_columns_for_train_and_test(
    train_features, test_features
):
    # The mismatch that makes feature alignment necessary.
    train_encoded = preprocess.one_hot_encode(train_features)
    test_encoded = preprocess.one_hot_encode(test_features)

    assert set(train_encoded.columns) != set(test_encoded.columns)


def test_align_features_matches_train_columns_and_order(train_features, test_features):
    train_encoded = preprocess.one_hot_encode(train_features)
    test_encoded = preprocess.one_hot_encode(test_features)

    aligned = preprocess.align_features(train_encoded, test_encoded)

    assert list(aligned.columns) == list(train_encoded.columns)
    assert len(aligned) == len(test_features)


def test_align_features_fills_missing_levels_with_zero_and_drops_unseen(
    train_features, test_features
):
    train_encoded = preprocess.one_hot_encode(train_features)
    test_encoded = preprocess.one_hot_encode(test_features)

    aligned = preprocess.align_features(train_encoded, test_encoded)

    assert (aligned["roof_type_x"] == 0).all()
    assert (aligned["land_surface_condition_o"] == 0).all()
    assert "roof_type_z" not in aligned.columns
    assert aligned["roof_type_n"].tolist() == [True, False]
    assert aligned["age"].tolist() == [7, 12]


def test_align_features_does_not_modify_its_inputs(train_features, test_features):
    train_encoded = preprocess.one_hot_encode(train_features)
    test_encoded = preprocess.one_hot_encode(test_features)
    columns_before = list(test_encoded.columns)

    preprocess.align_features(train_encoded, test_encoded)

    assert list(test_encoded.columns) == columns_before


def test_split_train_val_sizes_and_reproducibility():
    X = pd.DataFrame({"a": range(10)})
    y = pd.DataFrame({"damage_grade": [1, 2, 3, 2, 2, 1, 3, 2, 2, 3]})

    X_train, X_val, y_train, y_val = preprocess.split_train_val(X, y, test_size=0.2)
    again = preprocess.split_train_val(X, y, test_size=0.2)

    assert len(X_train) == 8 and len(X_val) == 2
    assert X_train.index.equals(y_train.index)
    assert X_val.index.equals(y_val.index)
    assert X_val.index.equals(again[1].index)


def test_remove_outliers_keeps_rows_up_to_threshold():
    df = pd.DataFrame({"age": [5, 30, 995, 100]})

    result = preprocess.remove_outliers(df, "age", threshold=100)

    assert result["age"].tolist() == [5, 30, 100]


def test_smote_oversample_balances_classes():
    # SMOTE needs more minority samples than its default of 5 neighbours.
    X = pd.DataFrame({"a": [float(i) for i in range(16)], "b": [1.0] * 16})
    y = pd.DataFrame({"damage_grade": [1] * 10 + [3] * 6})

    X_res, y_res = preprocess.smote_oversample(X, y, random_state=0)

    counts = pd.Series(y_res).value_counts()
    assert counts[1] == counts[3] == 10
    assert len(X_res) == len(y_res) == 20
