import numpy as np
import pandas as pd

from src.modeling.predict import create_submission


class FakeModel:
    """Predicts a fixed class and exposes the feature names it was trained on."""

    feature_names_in_ = np.array(["age", "roof_type_n", "roof_type_x"])

    def __init__(self, label):
        self.label = label
        self.seen_columns = None

    def predict(self, X):
        self.seen_columns = list(X.columns)
        return np.full(len(X), self.label)


def _write_inputs(tmp_path):
    test_values = pd.DataFrame(
        {"building_id": [101, 102, 103], "age": [5, 10, 15], "roof_type": ["n", "n", "q"]}
    )
    test_values.to_csv(tmp_path / "test_values.csv", index=False)
    pd.DataFrame({"building_id": [101, 102, 103], "damage_grade": [1, 1, 1]}).to_csv(
        tmp_path / "submission_format.csv", index=False
    )


def test_create_submission_aligns_features_and_writes_format(tmp_path):
    _write_inputs(tmp_path)
    model = FakeModel(label=2)
    output = tmp_path / "out" / "submission.csv"

    create_submission(
        model,
        tmp_path / "test_values.csv",
        output,
        submission_format_path=tmp_path / "submission_format.csv",
    )

    assert model.seen_columns == ["age", "roof_type_n", "roof_type_x"]
    submission = pd.read_csv(output)
    assert list(submission.columns) == ["building_id", "damage_grade"]
    assert submission["building_id"].tolist() == [101, 102, 103]
    assert submission["damage_grade"].tolist() == [2, 2, 2]


def test_create_submission_label_offset_shifts_zero_based_predictions(tmp_path):
    _write_inputs(tmp_path)
    output = tmp_path / "submission.csv"

    create_submission(
        FakeModel(label=0),
        tmp_path / "test_values.csv",
        output,
        submission_format_path=tmp_path / "submission_format.csv",
        label_offset=1,
    )

    assert pd.read_csv(output)["damage_grade"].tolist() == [1, 1, 1]
