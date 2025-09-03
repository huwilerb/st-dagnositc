import polars as pl

from st_dagnositc.core.models import ColumnMapping, DataType, FileFormat, ImportResult


class TestDataType:
    """Test cases for DataType enum"""

    def test_data_type_values(self):
        """Test that all expected data type values exist"""
        assert DataType.AUTO.value == "auto"
        assert DataType.STRING.value == "string"
        assert DataType.INTEGER.value == "integer"
        assert DataType.FLOAT.value == "float"
        assert DataType.BOOLEAN.value == "boolean"
        assert DataType.DATETIME.value == "datetime"
        assert DataType.DATE.value == "date"
        assert DataType.TIME.value == "time"

    def test_data_type_count(self):
        """Test that we have the expected number of data types"""
        assert len(DataType) == 8


class TestFileFormat:
    """Test cases for FileFormat enum"""

    def test_file_format_values(self):
        """Test that all expected file format values exist"""
        assert FileFormat.CSV.value == "csv"
        assert FileFormat.EXCEL.value == "excel"
        assert FileFormat.JSON.value == "json"
        assert FileFormat.PARQUET.value == "parquet"

    def test_file_format_count(self):
        """Test that we have the expected number of file formats"""
        assert len(FileFormat) == 4


class TestColumnMapping:
    """Test cases for ColumnMapping dataclass"""

    def test_column_mapping_creation(self):
        """Test basic ColumnMapping creation"""
        mapping = ColumnMapping(
            source_column="test_source", target_column="test_target"
        )

        assert mapping.source_column == "test_source"
        assert mapping.target_column == "test_target"
        assert mapping.data_type == DataType.AUTO
        assert mapping.include is True
        assert mapping.required is False
        assert mapping.validation_rules == []
        assert mapping.transformation_func is None

    def test_column_mapping_with_all_fields(self):
        """Test ColumnMapping creation with all fields specified"""

        def transform_func(x):
            return str(x).upper()

        mapping = ColumnMapping(
            source_column="name",
            target_column="full_name",
            data_type=DataType.STRING,
            include=True,
            required=True,
            validation_rules=["not_empty", "max_length_100"],
            transformation_func=transform_func,
        )

        assert mapping.source_column == "name"
        assert mapping.target_column == "full_name"
        assert mapping.data_type == DataType.STRING
        assert mapping.include is True
        assert mapping.required is True
        assert mapping.validation_rules == ["not_empty", "max_length_100"]
        assert mapping.transformation_func == transform_func

    def test_column_mapping_default_values(self):
        """Test that default values are set correctly"""
        mapping = ColumnMapping("src", "target")

        assert mapping.validation_rules == []
        assert isinstance(mapping.validation_rules, list)
        assert mapping.transformation_func is None

    def test_column_mapping_equality(self):
        """Test ColumnMapping equality comparison"""
        mapping1 = ColumnMapping("src", "target", DataType.STRING)
        mapping2 = ColumnMapping("src", "target", DataType.STRING)
        mapping3 = ColumnMapping("src", "target", DataType.INTEGER)

        assert mapping1 == mapping2
        assert mapping1 != mapping3


class TestImportResult:
    """Test cases for ImportResult dataclass"""

    def test_import_result_creation(self):
        """Test basic ImportResult creation"""
        result = ImportResult(success=True)

        assert result.success is True
        assert result.data is None
        assert result.mappings == []
        assert result.errors == []
        assert result.warnings == []
        assert result.metadata == {}

    def test_import_result_with_data(self):
        """Test ImportResult with data"""
        df = pl.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        mappings = [ColumnMapping("col1", "column1")]

        result = ImportResult(
            success=True,
            data=df,
            mappings=mappings,
            errors=["error1"],
            warnings=["warning1"],
            metadata={"rows": 3, "columns": 2},
        )

        assert result.success is True
        assert result.data.equals(df)
        assert len(result.mappings) == 1
        assert result.mappings[0].source_column == "col1"
        assert result.errors == ["error1"]
        assert result.warnings == ["warning1"]
        assert result.metadata == {"rows": 3, "columns": 2}

    def test_import_result_failure(self):
        """Test ImportResult for failure case"""
        result = ImportResult(
            success=False, errors=["File not found", "Invalid format"]
        )

        assert result.success is False
        assert result.data is None
        assert result.errors == ["File not found", "Invalid format"]
        assert result.warnings == []
        assert result.mappings == []

    def test_import_result_default_collections(self):
        """Test that default collections are properly initialized"""
        result = ImportResult(success=True)

        # Test that we can append to the lists
        result.errors.append("test error")
        result.warnings.append("test warning")
        result.metadata["key"] = "value"

        assert result.errors == ["test error"]
        assert result.warnings == ["test warning"]
        assert result.metadata == {"key": "value"}

    def test_import_result_with_polars_dataframe(self):
        """Test ImportResult specifically with Polars DataFrame"""
        df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "score": [95.5, 87.2, 92.8],
            }
        )

        result = ImportResult(success=True, data=df)

        assert result.success is True
        assert isinstance(result.data, pl.DataFrame)
        assert result.data.shape == (3, 3)
        assert list(result.data.columns) == ["id", "name", "score"]
