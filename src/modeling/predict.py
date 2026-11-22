"""Generates competition submission files from a trained model.

Loads test data, aligns features, makes predictions, and saves them
in the required submission format.
"""

from pathlib import Path

from loguru import logger
import pandas as pd

from src.config import RAW_DATA_DIR

SUBMISSION_FORMAT_PATH = RAW_DATA_DIR / "competition" / "submission_format.csv"


def create_submission(
    model,
    test_values_path: Path,
    output_path: Path,
    submission_format_path: Path = SUBMISSION_FORMAT_PATH,
    label_offset: int = 0,
):
    """Generate a competition submission file using a trained model.

    This function:
    - Loads the test feature dataset.
    - One-hot encodes categorical features.
    - Aligns columns with the training features (if the model stores feature names).
    - Generates predictions using the trained model.
    - Loads the official submission format.
    - Saves the predictions in the required submission format.

    Args:
        model: Trained machine learning model with a `predict` method.
        test_values_path (Path): Path to the test feature CSV file.
        output_path (Path): Destination path for the generated submission file.
        submission_format_path (Path): Path to the official submission format CSV.
        label_offset (int): Added to the predictions before saving. Use 1 for models
            trained on zero-based labels (e.g., XGBoost with classes 0-2) so that the
            submission contains the competition's damage grades 1-3.

    Returns:
        None
    """
    logger.info("Loading test values...")
    test_values = pd.read_csv(test_values_path, index_col="building_id")
    test_values = pd.get_dummies(test_values)

    if hasattr(model, "feature_names_in_"):
        test_values = test_values.reindex(columns=model.feature_names_in_, fill_value=0)

    logger.info("Predicting...")
    predictions = model.predict(test_values) + label_offset

    submission_format = pd.read_csv(submission_format_path, index_col="building_id")

    submission = pd.DataFrame(
        data=predictions, columns=submission_format.columns, index=submission_format.index
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path)
    logger.success(f"Saved submission to {output_path}")
