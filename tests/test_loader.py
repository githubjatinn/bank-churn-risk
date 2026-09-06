import pytest

from src.data.loader import SchemaValidationError, _validate_schema, load_raw


def test_validate_schema_accepts_correct_columns(sample_raw_df):
    _validate_schema(sample_raw_df)  # should not raise


def test_validate_schema_rejects_missing_column(sample_raw_df):
    broken = sample_raw_df.drop(columns=["Balance"])
    with pytest.raises(SchemaValidationError, match="missing expected columns"):
        _validate_schema(broken)


def test_validate_schema_rejects_null_target(sample_raw_df):
    broken = sample_raw_df.copy()
    broken.loc[0, "Exited"] = None
    with pytest.raises(SchemaValidationError, match="null target"):
        _validate_schema(broken)


def test_validate_schema_rejects_non_binary_target(sample_raw_df):
    broken = sample_raw_df.copy()
    broken.loc[0, "Exited"] = 2
    with pytest.raises(SchemaValidationError, match="non-binary"):
        _validate_schema(broken)


def test_validate_schema_tolerates_extra_columns(sample_raw_df, caplog):
    extra = sample_raw_df.copy()
    extra["SomeNewColumn"] = 1
    _validate_schema(extra)  # logs a warning, doesn't raise


def test_load_raw_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path / "does_not_exist.csv")


def test_load_raw_reads_valid_file(tmp_path, sample_raw_df):
    path = tmp_path / "raw.csv"
    sample_raw_df.to_csv(path, index=False)
    loaded = load_raw(path)
    assert len(loaded) == len(sample_raw_df)
    assert list(loaded.columns) == list(sample_raw_df.columns)
