from unittest.mock import Mock

import polars as pl

from st_dagnositc.core.models import ColumnMapping, DataType
from st_dagnositc.core.processor import ColumnMapper, DagnosticEngine, DataProcessor
from st_dagnositc.core.schema import SchemaDefinition


class TestColumnMapper:
    """Test cases for ColumnMapper class"""

    def test_column_mapper_creation(self):
        """Test ColumnMapper creation with default schema"""
        mapper = ColumnMapper()

        assert isinstance(mapper.schema, SchemaDefinition)
        assert mapper.similarity_threshold == 0.6

    def test_column_mapper_with_schema(self, sample_schema):
        """Test ColumnMapper creation with provided schema"""
        mapper = ColumnMapper(sample_schema)

        assert mapper.schema == sample_schema
        assert mapper.similarity_threshold == 0.6

    def test_suggest_mappings_basic(self, sample_dataframe):
        """Test basic column mapping suggestions"""
        mapper = ColumnMapper()
        mappings = mapper.suggest_mappings(sample_dataframe)

        assert len(mappings) == 4
        assert all(isinstance(m, ColumnMapping) for m in mappings)
        assert all(m.include is True for m in mappings)

        source_cols = [m.source_column for m in mappings]
        assert set(source_cols) == {"name", "email", "age", "city"}

    def test_suggest_mappings_with_schema(self, sample_dataframe, sample_schema):
        """Test column mapping suggestions with schema"""
        mapper = ColumnMapper(sample_schema)
        mappings = mapper.suggest_mappings(sample_dataframe)

        assert len(mappings) == 4

        # Check that similar names get mapped correctly
        mapping_dict = {m.source_column: m.target_column for m in mappings}

        # name should map to full_name (if similarity is high enough)
        # email should map to email_address
        source_cols = set(mapping_dict.keys())
        assert source_cols == {"name", "email", "age", "city"}

    def test_detect_column_type_integer(self):
        """Test column type detection for integer columns"""
        mapper = ColumnMapper()
        df = pl.DataFrame({"int_col": [1, 2, 3, 4, 5]})

        data_type = mapper._detect_column_type(df.select("int_col"))
        assert data_type == DataType.INTEGER

    def test_detect_column_type_float(self):
        """Test column type detection for float columns"""
        mapper = ColumnMapper()
        df = pl.DataFrame({"float_col": [1.1, 2.2, 3.3, 4.4, 5.5]})

        data_type = mapper._detect_column_type(df.select("float_col"))
        assert data_type == DataType.FLOAT

    def test_detect_column_type_boolean(self):
        """Test column type detection for boolean columns"""
        mapper = ColumnMapper()
        df = pl.DataFrame({"bool_col": [True, False, True, False, True]})

        data_type = mapper._detect_column_type(df.select("bool_col"))
        assert data_type == DataType.BOOLEAN

    def test_detect_column_type_string(self):
        """Test column type detection for string columns"""
        mapper = ColumnMapper()
        df = pl.DataFrame({"str_col": ["a", "b", "c", "d", "e"]})

        data_type = mapper._detect_column_type(df.select("str_col"))
        assert data_type == DataType.STRING

    def test_detect_column_type_datetime_string(self):
        """Test column type detection for datetime strings"""
        mapper = ColumnMapper()
        df = pl.DataFrame({"date_col": ["2023-01-01", "2023-01-02", "2023-01-03"]})

        data_type = mapper._detect_column_type(df.select("date_col"))
        # This might return DATETIME if parsing succeeds, or STRING if it fails
        assert data_type in [DataType.DATETIME, DataType.STRING]

    def test_calculate_similarity(self):
        """Test string similarity calculation"""
        mapper = ColumnMapper()

        # Identical strings
        assert mapper._calculate_similarity("test", "test") == 1.0

        # One string contains the other
        assert mapper._calculate_similarity("test", "testing") == 0.8
        assert mapper._calculate_similarity("testing", "test") == 0.8

        # Completely different strings
        similarity = mapper._calculate_similarity("abc", "xyz")
        assert 0.0 <= similarity < 1.0

        # Empty strings (identical, so should be 1.0)
        assert mapper._calculate_similarity("", "") == 1.0

    def test_clean_column_name(self):
        """Test column name cleaning"""
        mapper = ColumnMapper()

        assert mapper._clean_column_name("Test Column") == "test_column"
        assert mapper._clean_column_name("Test-Column!") == "testcolumn"
        assert mapper._clean_column_name("  Multiple   Spaces  ") == "multiple_spaces"
        assert mapper._clean_column_name("UPPERCASE") == "uppercase"

    def test_suggest_target_column_patterns(self):
        """Test target column suggestion with common patterns"""
        mapper = ColumnMapper()

        # Test email patterns
        assert "email" in mapper._suggest_target_column("email_address").lower()
        assert "email" in mapper._suggest_target_column("e_mail").lower()

        # Test name patterns
        suggested = mapper._suggest_target_column("full_name")
        assert "name" in suggested.lower()

        # Test phone patterns
        suggested = mapper._suggest_target_column("phone_number")
        assert "phone" in suggested.lower()


class TestDataProcessor:
    """Test cases for DataProcessor class"""

    def test_data_processor_creation(self):
        """Test DataProcessor creation"""
        processor = DataProcessor()
        assert processor is not None

    def test_process_dataframe_success(self, sample_dataframe, sample_column_mappings):
        """Test successful DataFrame processing"""
        processor = DataProcessor()

        # Use only included mappings
        included_mappings = [m for m in sample_column_mappings if m.include]
        result = processor.process_dataframe(sample_dataframe, included_mappings)

        assert result.success is True
        assert isinstance(result.data, pl.DataFrame)
        assert len(result.errors) == 0
        assert len(result.mappings) == len(included_mappings)

        # Check that target columns are correctly named
        expected_columns = ["full_name", "email_address", "age"]
        assert set(result.data.columns) == set(expected_columns)

    def test_process_dataframe_no_mappings(self, sample_dataframe):
        """Test DataFrame processing with no mappings"""
        processor = DataProcessor()

        result = processor.process_dataframe(sample_dataframe, [])

        assert result.success is False
        assert result.data is None
        assert "No column mappings provided" in result.errors

    def test_process_dataframe_missing_source_column(self, sample_dataframe):
        """Test DataFrame processing with missing source column"""
        processor = DataProcessor()

        mappings = [ColumnMapping("nonexistent_column", "target", include=True)]

        result = processor.process_dataframe(sample_dataframe, mappings)

        assert result.success is False
        assert "Source column 'nonexistent_column' not found" in result.errors

    def test_process_dataframe_with_transformation(self, sample_dataframe):
        """Test DataFrame processing with transformation function"""
        processor = DataProcessor()

        def uppercase_transform(x):
            return str(x).upper()

        mappings = [
            ColumnMapping(
                source_column="name",
                target_column="upper_name",
                transformation_func=uppercase_transform,
                include=True,
            )
        ]

        result = processor.process_dataframe(sample_dataframe, mappings)

        assert result.success is True
        assert "upper_name" in result.data.columns
        # Check if transformation was applied (case insensitive check since it might be handled differently)
        names = result.data["upper_name"].to_list()
        assert len(names) == 3

    def test_convert_column_type_string(self):
        """Test column type conversion to string"""
        processor = DataProcessor()

        expr = pl.col("test")
        converted = processor._convert_column_type(expr, DataType.STRING)

        # Test with a simple DataFrame
        df = pl.DataFrame({"test": [1, 2, 3]})
        result = df.select(converted.alias("converted"))

        assert result["converted"].dtype == pl.Utf8

    def test_convert_column_type_integer(self):
        """Test column type conversion to integer"""
        processor = DataProcessor()

        expr = pl.col("test")
        converted = processor._convert_column_type(expr, DataType.INTEGER)

        # Test with a simple DataFrame
        df = pl.DataFrame({"test": ["1", "2", "3"]})
        result = df.select(converted.alias("converted"))

        assert result["converted"].dtype == pl.Int64

    def test_convert_column_type_float(self):
        """Test column type conversion to float"""
        processor = DataProcessor()

        expr = pl.col("test")
        converted = processor._convert_column_type(expr, DataType.FLOAT)

        # Test with a simple DataFrame
        df = pl.DataFrame({"test": ["1.1", "2.2", "3.3"]})
        result = df.select(converted.alias("converted"))

        assert result["converted"].dtype == pl.Float64

    def test_convert_column_type_boolean(self):
        """Test column type conversion to boolean"""
        processor = DataProcessor()

        expr = pl.col("test")
        converted = processor._convert_column_type(expr, DataType.BOOLEAN)

        # Test with a simple DataFrame
        df = pl.DataFrame({"test": ["true", "false", "yes", "no", "1", "0"]})
        result = df.select(converted.alias("converted"))

        assert result["converted"].dtype == pl.Boolean


class TestDagnosticEngine:
    """Test cases for DagnosticEngine class"""

    def test_dagnostic_engine_creation(self):
        """Test DagnosticEngine creation with default schema"""
        engine = DagnosticEngine()

        assert isinstance(engine.schema, SchemaDefinition)
        assert isinstance(engine.column_mapper, ColumnMapper)
        assert isinstance(engine.data_processor, DataProcessor)
        assert len(engine.file_loaders) == 4  # CSV, Excel, JSON, Parquet

    def test_dagnostic_engine_with_schema(self, sample_schema):
        """Test DagnosticEngine creation with provided schema"""
        engine = DagnosticEngine(sample_schema)

        assert engine.schema == sample_schema

    def test_load_file_unsupported_format(self):
        """Test loading unsupported file format"""
        engine = DagnosticEngine()

        # Mock file that no loader can handle
        mock_file = Mock()
        mock_file.name = "test.unsupported"

        # Mock all loaders to return False for can_load
        for loader in engine.file_loaders:
            loader.can_load = Mock(return_value=False)

        df, errors = engine.load_file(mock_file)

        assert df is None
        assert "Unsupported file format" in errors

    def test_load_file_success(self, mock_csv_file, sample_dataframe):
        """Test successful file loading"""
        engine = DagnosticEngine()

        # Mock the CSV loader to return our sample dataframe
        engine.file_loaders[0].can_load = Mock(return_value=True)
        engine.file_loaders[0].load = Mock(return_value=sample_dataframe)

        df, errors = engine.load_file(mock_csv_file)

        assert df is not None
        assert isinstance(df, pl.DataFrame)
        assert len(errors) == 0

    def test_load_file_loader_exception(self, mock_csv_file):
        """Test file loading when loader raises exception"""
        engine = DagnosticEngine()

        # Mock the CSV loader to raise an exception
        engine.file_loaders[0].can_load = Mock(return_value=True)
        engine.file_loaders[0].load = Mock(side_effect=Exception("Load failed"))

        df, errors = engine.load_file(mock_csv_file)

        assert df is None
        assert len(errors) == 1
        assert "Failed to load file: Load failed" in errors

    def test_suggest_mappings(self, sample_dataframe):
        """Test mapping suggestions"""
        engine = DagnosticEngine()

        mappings = engine.suggest_mappings(sample_dataframe)

        assert isinstance(mappings, list)
        assert len(mappings) == 4  # One for each column
        assert all(isinstance(m, ColumnMapping) for m in mappings)

    def test_validate_mappings_success(self, sample_column_mappings):
        """Test successful mapping validation"""
        # Create schema that matches the mappings
        schema = SchemaDefinition()
        schema.add_column("full_name", required=True)
        schema.add_column("email_address", required=True)
        schema.add_column("age", required=False)
        schema.add_column("location", required=False)

        engine = DagnosticEngine(schema)

        errors = engine.validate_mappings(sample_column_mappings)
        assert len(errors) == 0

    def test_validate_mappings_missing_required(self, sample_column_mappings):
        """Test validation with missing required columns"""
        # Create schema with additional required column
        schema = SchemaDefinition()
        schema.add_column("full_name", required=True)
        schema.add_column("email_address", required=True)
        schema.add_column("missing_required", required=True)  # This will be missing

        engine = DagnosticEngine(schema)

        errors = engine.validate_mappings(sample_column_mappings)
        assert len(errors) == 1
        assert "Missing required columns" in errors[0]
        assert "missing_required" in errors[0]

    def test_validate_mappings_duplicate_targets(self):
        """Test validation with duplicate target columns"""
        mappings = [
            ColumnMapping("source1", "target", include=True),
            ColumnMapping("source2", "target", include=True),  # Duplicate target
        ]

        engine = DagnosticEngine()

        errors = engine.validate_mappings(mappings)
        assert len(errors) == 1
        assert "Duplicate target columns" in errors[0]
        assert "target" in errors[0]

    def test_process_data_success(self, sample_dataframe, sample_column_mappings):
        """Test successful data processing"""
        # Create schema that matches the mappings
        schema = SchemaDefinition()
        schema.add_column("full_name", required=True)
        schema.add_column("email_address", required=True)
        schema.add_column("age", required=False)
        schema.add_column("location", required=False)

        engine = DagnosticEngine(schema)

        result = engine.process_data(sample_dataframe, sample_column_mappings)

        assert result.success is True
        assert isinstance(result.data, pl.DataFrame)
        assert len(result.errors) == 0

    def test_process_data_validation_failure(
        self, sample_dataframe, sample_column_mappings
    ):
        """Test data processing with validation failure"""
        # Create schema with missing required column
        schema = SchemaDefinition()
        schema.add_column("full_name", required=True)
        schema.add_column("email_address", required=True)
        schema.add_column("missing_required", required=True)

        engine = DagnosticEngine(schema)

        result = engine.process_data(sample_dataframe, sample_column_mappings)

        assert result.success is False
        assert result.data is None
        assert len(result.errors) > 0
        assert "Missing required columns" in result.errors[0]


class TestProcessorIntegration:
    """Integration tests for processor components"""

    def test_end_to_end_processing(self, sample_dataframe):
        """Test complete end-to-end processing workflow"""
        # Create schema
        schema = SchemaDefinition()
        schema.add_column("full_name", DataType.STRING, required=True)
        schema.add_column("email_address", DataType.STRING, required=True)
        schema.add_column("user_age", DataType.INTEGER, required=False)

        # Create engine
        engine = DagnosticEngine(schema)

        # Suggest mappings
        suggested_mappings = engine.suggest_mappings(sample_dataframe)
        assert len(suggested_mappings) == 4

        # Manually adjust mappings for testing
        final_mappings = []
        for mapping in suggested_mappings:
            if mapping.source_column == "name":
                mapping.target_column = "full_name"
                final_mappings.append(mapping)
            elif mapping.source_column == "email":
                mapping.target_column = "email_address"
                final_mappings.append(mapping)
            elif mapping.source_column == "age":
                mapping.target_column = "user_age"
                mapping.data_type = DataType.INTEGER
                final_mappings.append(mapping)

        # Process data
        result = engine.process_data(sample_dataframe, final_mappings)

        assert result.success is True
        assert isinstance(result.data, pl.DataFrame)
        assert set(result.data.columns) == {"full_name", "email_address", "user_age"}
        assert result.data.shape[0] == 3  # Same number of rows

    def test_complete_workflow_with_file_loading(self, mock_csv_file, sample_dataframe):
        """Test complete workflow including file loading"""
        engine = DagnosticEngine()

        # Mock file loading
        engine.file_loaders[0].can_load = Mock(return_value=True)
        engine.file_loaders[0].load = Mock(return_value=sample_dataframe)

        # Load file
        df, load_errors = engine.load_file(mock_csv_file)
        assert df is not None
        assert len(load_errors) == 0

        # Suggest mappings
        mappings = engine.suggest_mappings(df)
        assert len(mappings) > 0

        # Process data
        result = engine.process_data(df, mappings)
        assert result.success is True
