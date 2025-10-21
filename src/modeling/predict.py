"""Generates competition submission files from a trained model.

Loads test data, aligns features, makes predictions, and saves them
in the required submission format.
"""

from pathlib import Path

from loguru import logger
import pandas as pd
import typer

app = typer.Typer()


def create_submission(
    model,
    test_values_path: Path,
    output_path: Path,
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

    Returns:
        None
    """
    logger.info("Loading test values...")
    test_values = pd.read_csv(test_values_path, index_col="building_id")
    test_values = pd.get_dummies(test_values)

    if hasattr(model, "feature_names_in_"):
        test_values = test_values.reindex(columns=model.feature_names_in_, fill_value=0)

    logger.info("Predicting...")
    predictions = model.predict(test_values)

    submission_format = pd.read_csv(
        "../data/raw/competition/submission_format.csv", index_col="building_id"
    )

    my_submission = pd.DataFrame(
        data=predictions, columns=submission_format.columns, index=submission_format.index
    )

    my_submission.to_csv(output_path)
    logger.success(f"Saved submission to {output_path}")


# @app.command()
# def main(
#     # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
#     features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
#     model_path: Path = MODELS_DIR / "model.pkl",
#     predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
#     # -----------------------------------------
# ):
#     # ---- REPLACE THIS WITH YOUR OWN CODE ----
#     logger.info("Performing inference for model...")
#     for i in tqdm(range(10), total=10):
#         if i == 5:
#             logger.info("Something happened for iteration 5.")
#     logger.success("Inference complete.")
#     # -----------------------------------------


if __name__ == "__main__":
    app()
