from st_dagnositc.core.models import DataType
from st_dagnositc.core.schema import SchemaDefinition


class TestSchemaDefinition:
    """Test cases for SchemaDefinition class"""

    def test_schema_definition_creation(self):
        """Test basic SchemaDefinition creation"""
        schema = SchemaDefinition()

        assert schema.columns == {}
        assert isinstance(schema.columns, dict)

    def test_add_column_minimal(self):
        """Test adding a column with minimal parameters"""
        schema = SchemaDefinition()
        schema.add_column("test_column")

        assert "test_column" in schema.columns
        assert schema.columns["test_column"]["data_type"] == DataType.AUTO
        assert schema.columns["test_column"]["required"] is False
        assert schema.columns["test_column"]["description"] == ""

    def test_add_column_with_all_parameters(self):
        """Test adding a column with all parameters"""
        schema = SchemaDefinition()
        schema.add_column(
            name="user_email",
            data_type=DataType.STRING,
            required=True,
            description="User's email address",
        )

        assert "user_email" in schema.columns
        column_config = schema.columns["user_email"]
        assert column_config["data_type"] == DataType.STRING
        assert column_config["required"] is True
        assert column_config["description"] == "User's email address"

    def test_add_multiple_columns(self):
        """Test adding multiple columns"""
        schema = SchemaDefinition()

        schema.add_column("name", DataType.STRING, required=True)
        schema.add_column("age", DataType.INTEGER, required=False)
        schema.add_column("email", DataType.STRING, required=True)

        assert len(schema.columns) == 3
        assert "name" in schema.columns
        assert "age" in schema.columns
        assert "email" in schema.columns

    def test_add_column_overwrites(self):
        """Test that adding a column with the same name overwrites the previous one"""
        schema = SchemaDefinition()

        schema.add_column("test", DataType.STRING, required=False)
        schema.add_column(
            "test", DataType.INTEGER, required=True, description="Updated"
        )

        assert len(schema.columns) == 1
        column_config = schema.columns["test"]
        assert column_config["data_type"] == DataType.INTEGER
        assert column_config["required"] is True
        assert column_config["description"] == "Updated"

    def test_get_required_columns_empty(self):
        """Test get_required_columns with no required columns"""
        schema = SchemaDefinition()
        schema.add_column("optional1", required=False)
        schema.add_column("optional2", required=False)

        required = schema.get_required_columns()
        assert required == []

    def test_get_required_columns_with_required(self):
        """Test get_required_columns with mixed required/optional columns"""
        schema = SchemaDefinition()
        schema.add_column("name", DataType.STRING, required=True)
        schema.add_column("age", DataType.INTEGER, required=False)
        schema.add_column("email", DataType.STRING, required=True)
        schema.add_column("city", DataType.STRING, required=False)

        required = schema.get_required_columns()
        assert set(required) == {"name", "email"}
        assert len(required) == 2

    def test_get_required_columns_all_required(self):
        """Test get_required_columns when all columns are required"""
        schema = SchemaDefinition()
        schema.add_column("col1", required=True)
        schema.add_column("col2", required=True)
        schema.add_column("col3", required=True)

        required = schema.get_required_columns()
        assert set(required) == {"col1", "col2", "col3"}

    def test_get_optional_columns_empty(self):
        """Test get_optional_columns with no optional columns"""
        schema = SchemaDefinition()
        schema.add_column("required1", required=True)
        schema.add_column("required2", required=True)

        optional = schema.get_optional_columns()
        assert optional == []

    def test_get_optional_columns_with_optional(self):
        """Test get_optional_columns with mixed required/optional columns"""
        schema = SchemaDefinition()
        schema.add_column("name", DataType.STRING, required=True)
        schema.add_column("age", DataType.INTEGER, required=False)
        schema.add_column("email", DataType.STRING, required=True)
        schema.add_column("city", DataType.STRING, required=False)

        optional = schema.get_optional_columns()
        assert set(optional) == {"age", "city"}
        assert len(optional) == 2

    def test_get_optional_columns_all_optional(self):
        """Test get_optional_columns when all columns are optional"""
        schema = SchemaDefinition()
        schema.add_column("col1", required=False)
        schema.add_column("col2", required=False)
        schema.add_column("col3", required=False)

        optional = schema.get_optional_columns()
        assert set(optional) == {"col1", "col2", "col3"}

    def test_empty_schema_methods(self):
        """Test methods on empty schema"""
        schema = SchemaDefinition()

        assert schema.get_required_columns() == []
        assert schema.get_optional_columns() == []

    def test_schema_with_all_data_types(self):
        """Test schema with all supported data types"""
        schema = SchemaDefinition()

        schema.add_column("auto_col", DataType.AUTO)
        schema.add_column("string_col", DataType.STRING)
        schema.add_column("int_col", DataType.INTEGER)
        schema.add_column("float_col", DataType.FLOAT)
        schema.add_column("bool_col", DataType.BOOLEAN)
        schema.add_column("datetime_col", DataType.DATETIME)
        schema.add_column("date_col", DataType.DATE)
        schema.add_column("time_col", DataType.TIME)

        assert len(schema.columns) == 8
        assert schema.columns["auto_col"]["data_type"] == DataType.AUTO
        assert schema.columns["string_col"]["data_type"] == DataType.STRING
        assert schema.columns["int_col"]["data_type"] == DataType.INTEGER
        assert schema.columns["float_col"]["data_type"] == DataType.FLOAT
        assert schema.columns["bool_col"]["data_type"] == DataType.BOOLEAN
        assert schema.columns["datetime_col"]["data_type"] == DataType.DATETIME
        assert schema.columns["date_col"]["data_type"] == DataType.DATE
        assert schema.columns["time_col"]["data_type"] == DataType.TIME

    def test_schema_from_fixture(self, sample_schema):
        """Test using the sample schema fixture"""
        required = sample_schema.get_required_columns()
        optional = sample_schema.get_optional_columns()

        assert set(required) == {"full_name", "email_address"}
        assert set(optional) == {"age", "location"}
        assert len(sample_schema.columns) == 4

    def test_column_descriptions(self):
        """Test column descriptions are stored correctly"""
        schema = SchemaDefinition()

        schema.add_column(
            "user_id",
            DataType.INTEGER,
            required=True,
            description="Unique identifier for user",
        )
        schema.add_column(
            "created_at",
            DataType.DATETIME,
            required=False,
            description="Timestamp when record was created",
        )

        assert schema.columns["user_id"]["description"] == "Unique identifier for user"
        assert (
            schema.columns["created_at"]["description"]
            == "Timestamp when record was created"
        )
