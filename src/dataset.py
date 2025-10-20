"""Utility module for loading and preparing both competition and original datasets.

Provides functions to:
- Load raw competition (train, labels, test) and original data.
- Merge train with labels and structure with ownership.
- Generate interim artifacts for downstream analysis.
- Run as a CLI to create interim parquet files.

Usage:
    poetry run python src/dataset.py make-interim-comp
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from loguru import logger
import pandas as pd
import typer

from config import INTERIM_DATA_DIR, RAW_DATA_DIR

app = typer.Typer(help="Load raw data and create interim artifacts.")


# --------- reusable loaders ---------
def load_competition_raw(raw_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load competition data (train, labels, and test) from the given raw data directory.

    Args:
        raw_dir (Path): Path to the raw data directory (e.g., data/raw).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            - train (pd.DataFrame): Training feature set.
            - labels (pd.DataFrame): Corresponding labels for the training set.
            - test (pd.DataFrame): Test feature set.
    """
    comp = raw_dir / "competition"
    train = pd.read_csv(comp / "train_values.csv")
    labels = pd.read_csv(comp / "train_labels.csv")
    test = pd.read_csv(comp / "test_values.csv")
    return train, labels, test


def load_original_raw(raw_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the original auxiliary data tables (damage assessment, structure, ownership) from the given raw data directory.

    Args:
        raw_dir (Path): Path to the raw data directory (e.g., data/raw).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            - damage (pd.DataFrame): Damage assessment data.
            - structure (pd.DataFrame): Building structure data.
            - owneruse (pd.DataFrame): Ownership and use data.
    """
    og = raw_dir / "original"
    damage = pd.read_csv(og / "csv_building_damage_assessment.csv")
    structure = pd.read_csv(og / "csv_building_structure.csv")
    owneruse = pd.read_csv(og / "csv_building_ownership_and_use.csv")
    return damage, structure, owneruse


def build_competition_tables(train: pd.DataFrame, labels: pd.DataFrame, test: pd.DataFrame):
    """Build interim datasets for the competition dataset by merging training features with labels and keeping the test set unchanged.

    Args:
        train (pd.DataFrame): Raw training feature set.
        labels (pd.DataFrame): Corresponding labels.
        test (pd.DataFrame): Raw test feature set.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - df (pd.DataFrame): Merged training set with labels.
            - test (pd.DataFrame): Test set (unchanged).
    """
    df = train.merge(labels, on="building_id", how="left")
    return df, test


def build_original_table(structure: pd.DataFrame, owner: pd.DataFrame) -> pd.DataFrame:
    """Merge structure and ownership tables from the original dataset.

    Args:
        structure (pd.DataFrame): Building structure data.
        owner (pd.DataFrame): Building ownership and use data.

    Returns:
        pd.DataFrame: Merged original dataset.
    """
    df = structure.merge(owner, on="building_id", how="left")

    return df


# --------- CLI command ---------
@app.command()
def make_interim_comp(
    raw_dir: Path = RAW_DATA_DIR,
    out_dir: Path = INTERIM_DATA_DIR,
    train_out: str = "train_interim.parquet",
    test_out: str = "test_interim.parquet",
):
    """CLI command to create interim data artifacts for the competition dataset.

    Loads raw train/test data, merges labels with train,
    and writes the resulting interim train and test sets to disk as Parquet files.

    Args:
        raw_dir (Path): Directory containing the raw data (default: RAW_DATA_DIR).
        out_dir (Path): Directory where interim data will be saved (default: INTERIM_DIR).
        train_out (str): Output filename for interim train data.
        test_out (str): Output filename for interim test data.
    """
    logger.info(f"Loading raw data from {raw_dir}")
    train, labels, test = load_competition_raw(raw_dir)

    logger.info("Building interim artifacts")
    df_i, test_i = build_competition_tables(train, labels, test)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / train_out).write_bytes(df_i.to_parquet(index=False))
    (out_dir / test_out).write_bytes(test_i.to_parquet(index=False))
    logger.success("Interim artifacts written.")


if __name__ == "__main__":
    app()
