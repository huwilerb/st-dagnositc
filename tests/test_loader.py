import io
from unittest.mock import Mock, patch

import polars as pl

from st_dagnositc.core.loader import (
    CSVLoader,
    ExcelLoader,
    FileLoaderProtocol,
    JSONLoader,
    ParquetLoader,
)


class TestFileLoaderProtocol:
    """Test cases for FileLoaderProtocol"""

    def test_protocol_methods_exist(self):
        """Test that protocol defines required methods"""
        assert hasattr(FileLoaderProtocol, "can_load")
        assert hasattr(FileLoaderProtocol, "load")

    def test_protocol_instance_check(self):
        """Test that concrete loaders implement the protocol"""
        csv_loader = CSVLoader()
        excel_loader = ExcelLoader()
        json_loader = JSONLoader()
        parquet_loader = ParquetLoader()

        assert isinstance(csv_loader, FileLoaderProtocol)
        assert isinstance(excel_loader, FileLoaderProtocol)
        assert isinstance(json_loader, FileLoaderProtocol)
        assert isinstance(parquet_loader, FileLoaderProtocol)


class TestCSVLoader:
    """Test cases for CSVLoader"""

    def test_can_load_csv_file(self):
        """Test CSVLoader can detect CSV files"""
        loader = CSVLoader()

        # Test with mock file object
        mock_file = Mock()
        mock_file.name = "test.csv"
        assert loader.can_load(mock_file) is True

        mock_file.name = "TEST.CSV"
        assert loader.can_load(mock_file) is True

    def test_can_load_non_csv_file(self):
        """Test CSVLoader rejects non-CSV files"""
        loader = CSVLoader()

        mock_file = Mock()
        mock_file.name = "test.xlsx"
        assert loader.can_load(mock_file) is False

        mock_file.name = "test.json"
        assert loader.can_load(mock_file) is False

        # Test with no name attribute
        mock_file_no_name = Mock()
        del mock_file_no_name.name
        assert loader.can_load(mock_file_no_name) is False

    def test_load_csv_from_string_io(self, sample_csv_data):
        """Test loading CSV from StringIO object"""
        loader = CSVLoader()

        mock_file = Mock()
        mock_file.getvalue.return_value = sample_csv_data
        mock_file.read.return_value = sample_csv_data

        df = loader.load(mock_file)

        assert isinstance(df, pl.DataFrame)
        assert df.shape == (3, 4)
        assert list(df.columns) == ["name", "email", "age", "city"]
        assert df["name"].to_list() == ["John Doe", "Jane Smith", "Bob Johnson"]

    def test_load_csv_from_bytes(self, sample_csv_data):
        """Test loading CSV from bytes"""
        loader = CSVLoader()

        mock_file = Mock()
        mock_file.getvalue.return_value = sample_csv_data.encode("utf-8")

        df = loader.load(mock_file)

        assert isinstance(df, pl.DataFrame)
        assert df.shape == (3, 4)
        assert list(df.columns) == ["name", "email", "age", "city"]

    def test_load_csv_from_file_path(self, sample_csv_data, tmp_path):
        """Test loading CSV from file path"""
        loader = CSVLoader()

        # Create temporary CSV file
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(sample_csv_data)

        df = loader.load(str(csv_file))

        assert isinstance(df, pl.DataFrame)
        assert df.shape == (3, 4)
        assert list(df.columns) == ["name", "email", "age", "city"]

    def test_load_csv_with_kwargs(self, sample_csv_data):
        """Test loading CSV with additional kwargs"""
        loader = CSVLoader()

        mock_file = Mock()
        mock_file.getvalue.return_value = sample_csv_data

        # Test with separator parameter
        df = loader.load(mock_file, separator=",")

        assert isinstance(df, pl.DataFrame)
        assert df.shape == (3, 4)


class TestExcelLoader:
    """Test cases for ExcelLoader"""

    def test_can_load_excel_file(self):
        """Test ExcelLoader can detect Excel files"""
        loader = ExcelLoader()

        mock_file = Mock()
        mock_file.name = "test.xlsx"
        assert loader.can_load(mock_file) is True

        mock_file.name = "test.xls"
        assert loader.can_load(mock_file) is True

        mock_file.name = "TEST.XLSX"
        assert loader.can_load(mock_file) is True

    def test_can_load_non_excel_file(self):
        """Test ExcelLoader rejects non-Excel files"""
        loader = ExcelLoader()

        mock_file = Mock()
        mock_file.name = "test.csv"
        assert loader.can_load(mock_file) is False

        mock_file.name = "test.json"
        assert loader.can_load(mock_file) is False

    @patch("pandas.read_excel")
    @patch("polars.from_pandas")
    def test_load_excel_from_bytes(
        self, mock_from_pandas, mock_read_excel, sample_excel_data
    ):
        """Test loading Excel from bytes"""
        import pandas as pd

        loader = ExcelLoader()

        # Mock pandas DataFrame
        mock_pandas_df = pd.DataFrame(sample_excel_data)
        mock_read_excel.return_value = mock_pandas_df

        # Mock polars DataFrame
        mock_polars_df = Mock()
        mock_from_pandas.return_value = mock_polars_df

        mock_file = Mock()
        mock_file.getvalue.return_value = b"fake_excel_data"

        result = loader.load(mock_file)

        # Verify pandas.read_excel was called with BytesIO
        mock_read_excel.assert_called_once()
        args, kwargs = mock_read_excel.call_args
        assert isinstance(args[0], io.BytesIO)

        # Verify polars.from_pandas was called
        mock_from_pandas.assert_called_once_with(mock_pandas_df)

        assert result == mock_polars_df

    @patch("pandas.read_excel")
    @patch("polars.from_pandas")
    def test_load_excel_from_file_path(
        self, mock_from_pandas, mock_read_excel, sample_excel_data
    ):
        """Test loading Excel from file path"""
        import pandas as pd

        loader = ExcelLoader()

        # Mock pandas DataFrame
        mock_pandas_df = pd.DataFrame(sample_excel_data)
        mock_read_excel.return_value = mock_pandas_df

        # Mock polars DataFrame
        mock_polars_df = Mock()
        mock_from_pandas.return_value = mock_polars_df

        file_path = "test.xlsx"
        result = loader.load(file_path)

        # Verify pandas.read_excel was called with file path
        mock_read_excel.assert_called_once_with(file_path)
        mock_from_pandas.assert_called_once_with(mock_pandas_df)

        assert result == mock_polars_df


class TestJSONLoader:
    """Test cases for JSONLoader"""

    def test_can_load_json_file(self):
        """Test JSONLoader can detect JSON files"""
        loader = JSONLoader()

        mock_file = Mock()
        mock_file.name = "test.json"
        assert loader.can_load(mock_file) is True

        mock_file.name = "TEST.JSON"
        assert loader.can_load(mock_file) is True

    def test_can_load_non_json_file(self):
        """Test JSONLoader rejects non-JSON files"""
        loader = JSONLoader()

        mock_file = Mock()
        mock_file.name = "test.csv"
        assert loader.can_load(mock_file) is False

        mock_file.name = "test.xlsx"
        assert loader.can_load(mock_file) is False

    def test_load_json_from_string_io(self, sample_json_data):
        """Test loading JSON from StringIO object"""
        loader = JSONLoader()

        mock_file = Mock()
        mock_file.getvalue.return_value = sample_json_data
        mock_file.read.return_value = sample_json_data

        df = loader.load(mock_file)

        assert isinstance(df, pl.DataFrame)
        assert df.shape == (3, 4)
        assert list(df.columns) == ["name", "email", "age", "city"]
        assert df["name"].to_list() == ["John Doe", "Jane Smith", "Bob Johnson"]

    def test_load_json_from_bytes(self, sample_json_data):
        """Test loading JSON from bytes"""
        loader = JSONLoader()

        mock_file = Mock()
        mock_file.getvalue.return_value = sample_json_data.encode("utf-8")

        df = loader.load(mock_file)

        assert isinstance(df, pl.DataFrame)
        assert df.shape == (3, 4)
        assert list(df.columns) == ["name", "email", "age", "city"]

    def test_load_json_from_file_path(self, sample_json_data, tmp_path):
        """Test loading JSON from file path"""
        loader = JSONLoader()

        # Create temporary JSON file
        json_file = tmp_path / "test.json"
        json_file.write_text(sample_json_data)

        df = loader.load(str(json_file))

        assert isinstance(df, pl.DataFrame)
        assert df.shape == (3, 4)
        assert list(df.columns) == ["name", "email", "age", "city"]


class TestParquetLoader:
    """Test cases for ParquetLoader"""

    def test_can_load_parquet_file(self):
        """Test ParquetLoader can detect Parquet files"""
        loader = ParquetLoader()

        mock_file = Mock()
        mock_file.name = "test.parquet"
        assert loader.can_load(mock_file) is True

        mock_file.name = "TEST.PARQUET"
        assert loader.can_load(mock_file) is True

    def test_can_load_non_parquet_file(self):
        """Test ParquetLoader rejects non-Parquet files"""
        loader = ParquetLoader()

        mock_file = Mock()
        mock_file.name = "test.csv"
        assert loader.can_load(mock_file) is False

        mock_file.name = "test.json"
        assert loader.can_load(mock_file) is False

    def test_load_parquet_from_bytes(self, sample_dataframe, tmp_path):
        """Test loading Parquet from bytes"""
        loader = ParquetLoader()

        # Create temporary parquet file to get bytes
        parquet_file = tmp_path / "test.parquet"
        sample_dataframe.write_parquet(parquet_file)
        parquet_bytes = parquet_file.read_bytes()

        mock_file = Mock()
        mock_file.getvalue.return_value = parquet_bytes

        df = loader.load(mock_file)

        assert isinstance(df, pl.DataFrame)
        assert df.shape == sample_dataframe.shape
        assert list(df.columns) == list(sample_dataframe.columns)

    def test_load_parquet_from_file_path(self, sample_dataframe, tmp_path):
        """Test loading Parquet from file path"""
        loader = ParquetLoader()

        # Create temporary parquet file
        parquet_file = tmp_path / "test.parquet"
        sample_dataframe.write_parquet(parquet_file)

        df = loader.load(str(parquet_file))

        assert isinstance(df, pl.DataFrame)
        assert df.shape == sample_dataframe.shape
        assert list(df.columns) == list(sample_dataframe.columns)


class TestLoaderIntegration:
    """Integration tests for all loaders"""

    def test_all_loaders_implement_protocol(self):
        """Test that all loaders properly implement the FileLoaderProtocol"""
        loaders = [CSVLoader(), ExcelLoader(), JSONLoader(), ParquetLoader()]

        for loader in loaders:
            assert isinstance(loader, FileLoaderProtocol)
            assert hasattr(loader, "can_load")
            assert hasattr(loader, "load")
            assert callable(loader.can_load)
            assert callable(loader.load)

    def test_loader_file_type_exclusivity(self):
        """Test that each loader only accepts its specific file type"""
        loaders = {
            "csv": CSVLoader(),
            "excel": ExcelLoader(),
            "json": JSONLoader(),
            "parquet": ParquetLoader(),
        }

        test_files = {"csv": Mock(), "excel": Mock(), "json": Mock(), "parquet": Mock()}
        test_files["csv"].name = "test.csv"
        test_files["excel"].name = "test.xlsx"
        test_files["json"].name = "test.json"
        test_files["parquet"].name = "test.parquet"

        for loader_type, loader in loaders.items():
            for file_type, file_mock in test_files.items():
                # Special case for Excel loader accepting both .xlsx and .xls
                if loader_type == file_type:
                    assert loader.can_load(file_mock) is True
                elif loader_type == "excel" and file_type == "excel":
                    assert loader.can_load(file_mock) is True
                else:
                    assert loader.can_load(file_mock) is False

    def test_loader_case_insensitive_extensions(self):
        """Test that all loaders handle case-insensitive file extensions"""
        test_cases = [
            (CSVLoader(), ["test.csv", "test.CSV", "Test.Csv"]),
            (
                ExcelLoader(),
                ["test.xlsx", "test.XLSX", "Test.Xlsx", "test.xls", "TEST.XLS"],
            ),
            (JSONLoader(), ["test.json", "test.JSON", "Test.Json"]),
            (ParquetLoader(), ["test.parquet", "test.PARQUET", "Test.Parquet"]),
        ]

        for loader, file_names in test_cases:
            for file_name in file_names:
                mock_file = Mock()
                mock_file.name = file_name
                assert loader.can_load(mock_file) is True
