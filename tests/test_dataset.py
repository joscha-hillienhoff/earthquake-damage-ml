import pandas as pd

from src import dataset


def _write_competition_files(raw_dir):
    comp = raw_dir / "competition"
    comp.mkdir(parents=True)
    pd.DataFrame({"building_id": [1, 2], "age": [5, 10]}).to_csv(
        comp / "train_values.csv", index=False
    )
    pd.DataFrame({"building_id": [1, 2], "damage_grade": [2, 3]}).to_csv(
        comp / "train_labels.csv", index=False
    )
    pd.DataFrame({"building_id": [3], "age": [7]}).to_csv(comp / "test_values.csv", index=False)


def test_build_competition_tables_merges_labels_into_train():
    train = pd.DataFrame({"building_id": [1, 2], "age": [5, 10]})
    labels = pd.DataFrame({"building_id": [2, 1], "damage_grade": [3, 2]})
    test = pd.DataFrame({"building_id": [3], "age": [7]})

    merged, test_out = dataset.build_competition_tables(train, labels, test)

    assert merged.set_index("building_id")["damage_grade"].to_dict() == {1: 2, 2: 3}
    assert test_out.equals(test)


def test_build_original_table_left_joins_ownership_on_building_id():
    structure = pd.DataFrame({"building_id": [1, 2], "floors": [2, 3]})
    owner = pd.DataFrame({"building_id": [1], "has_repair_started": [True]})

    merged = dataset.build_original_table(structure, owner)

    assert len(merged) == 2
    assert merged.loc[merged["building_id"] == 1, "has_repair_started"].item()


def test_load_competition_raw_reads_the_three_files(tmp_path):
    _write_competition_files(tmp_path)

    train, labels, test = dataset.load_competition_raw(tmp_path)

    assert list(train.columns) == ["building_id", "age"]
    assert list(labels.columns) == ["building_id", "damage_grade"]
    assert len(test) == 1
