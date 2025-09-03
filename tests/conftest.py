import io
from typing import Any, Dict

import polars as pl
import pytest

from st_dagnositc.core import (
    ColumnMapping,
    DagnosticEngine,
    DataType,
    SchemaDefinition,
)


@pytest.fixture
def sample_csv_data() -> str:
    """Sample CSV data for testing"""
    return """name,email,age,city
John Doe,john@example.com,30,New York
Jane Smith,jane@example.com,25,Los Angeles
Bob Johnson,bob@example.com,35,Chicago"""


@pytest.fixture
def sample_excel_data() -> Dict[str, Any]:
    """Sample Excel data structure for testing"""
    return {
        "name": ["John Doe", "Jane Smith", "Bob Johnson"],
        "email": ["john@example.com", "jane@example.com", "bob@example.com"],
        "age": [30, 25, 35],
        "city": ["New York", "Los Angeles", "Chicago"],
    }


@pytest.fixture
def sample_json_data() -> str:
    """Sample JSON data for testing"""
    return """[
        {"name": "John Doe", "email": "john@example.com", "age": 30, "city": "New York"},
        {"name": "Jane Smith", "email": "jane@example.com", "age": 25, "city": "Los Angeles"},
        {"name": "Bob Johnson", "email": "bob@example.com", "age": 35, "city": "Chicago"}
    ]"""


@pytest.fixture
def sample_dataframe() -> pl.DataFrame:
    """Sample Polars DataFrame for testing"""
    return pl.DataFrame(
        {
            "name": ["John Doe", "Jane Smith", "Bob Johnson"],
            "email": ["john@example.com", "jane@example.com", "bob@example.com"],
            "age": [30, 25, 35],
            "city": ["New York", "Los Angeles", "Chicago"],
        }
    )


@pytest.fixture
def sample_schema() -> SchemaDefinition:
    """Sample schema definition for testing"""
    schema = SchemaDefinition()
    schema.add_column("full_name", DataType.STRING, required=True)
    schema.add_column("email_address", DataType.STRING, required=True)
    schema.add_column("age", DataType.INTEGER, required=False)
    schema.add_column("location", DataType.STRING, required=False)
    return schema


@pytest.fixture
def sample_column_mappings() -> list[ColumnMapping]:
    """Sample column mappings for testing"""
    return [
        ColumnMapping(
            source_column="name",
            target_column="full_name",
            data_type=DataType.STRING,
            include=True,
        ),
        ColumnMapping(
            source_column="email",
            target_column="email_address",
            data_type=DataType.STRING,
            include=True,
        ),
        ColumnMapping(
            source_column="age",
            target_column="age",
            data_type=DataType.INTEGER,
            include=True,
        ),
        ColumnMapping(
            source_column="city",
            target_column="location",
            data_type=DataType.STRING,
            include=False,
        ),
    ]


@pytest.fixture
def dagnostic_engine() -> DagnosticEngine:
    """DagnosticEngine instance for testing"""
    return DagnosticEngine()


@pytest.fixture
def dagnostic_engine_with_schema(sample_schema) -> DagnosticEngine:
    """DagnosticEngine instance with sample schema for testing"""
    return DagnosticEngine(sample_schema)


class MockFile:
    """Mock file object for testing file loaders"""

    def __init__(self, name: str, content: str | bytes):
        self.name = name
        self._content = content
        self._position = 0

    def read(self) -> str | bytes:
        return self._content

    def getvalue(self) -> str | bytes:
        return self._content


@pytest.fixture
def mock_csv_file(sample_csv_data) -> MockFile:
    """Mock CSV file for testing"""
    return MockFile("test.csv", sample_csv_data)


@pytest.fixture
def mock_json_file(sample_json_data) -> MockFile:
    """Mock JSON file for testing"""
    return MockFile("test.json", sample_json_data)


@pytest.fixture
def mock_excel_file() -> MockFile:
    """Mock Excel file for testing"""
    import pandas as pd

    df = pd.DataFrame(
        {
            "name": ["John Doe", "Jane Smith", "Bob Johnson"],
            "email": ["john@example.com", "jane@example.com", "bob@example.com"],
            "age": [30, 25, 35],
            "city": ["New York", "Los Angeles", "Chicago"],
        }
    )

    buffer = io.BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)

    return MockFile("test.xlsx", buffer.getvalue())
