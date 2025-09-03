import io
from typing import Any, Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class FileLoaderProtocol(Protocol):
    """Protocol for file loaders"""

    VALID_SUFFIXES: list[str] = []

    def can_load(self, file_obj: Any) -> bool:
        """Check if this loader can handle the file"""
        ...

    def load(self, file_obj: Any, **kwargs: Any) -> pl.DataFrame:
        """Load the file and return a DataFrame"""
        ...


class BaseLoader:
    """Base loader"""

    VALID_SUFFIXES: list[str] = []

    def can_load(self, file_obj: Any) -> bool:
        if not self.has_name(file_obj):
            return False
        return self.has_suffix(file_obj)

    def load(self, file_obj: Any, **kwargs: Any) -> pl.DataFrame:
        raise NotImplementedError

    def has_name(self, file_obj: Any) -> bool:
        return hasattr(file_obj, "name")

    def has_suffix(self, file_obj: Any) -> bool:
        name = file_obj.name
        suffix = "." + name.lower().split(".")[-1]
        return suffix in self.VALID_SUFFIXES


class CSVLoader(BaseLoader):
    """Loader for CSV files"""

    VALID_SUFFIXES = [".csv"]

    def load(self, file_obj: Any, **kwargs: Any) -> pl.DataFrame:
        if hasattr(file_obj, "read"):
            content = (
                file_obj.getvalue()
                if hasattr(file_obj, "getvalue")
                else file_obj.read()
            )
            if isinstance(content, bytes):
                content = content.decode("utf-8")
            return pl.read_csv(io.StringIO(content), **kwargs)
        else:
            return pl.read_csv(file_obj, **kwargs)


class ExcelLoader(BaseLoader):
    """Loader for Excel files"""

    VALID_SUFFIXES: list[str] = [".xlsx", ".xls"]

    def load(self, file_obj: Any, **kwargs: Any) -> pl.DataFrame:
        import pandas as pd

        if hasattr(file_obj, "read"):
            content = (
                file_obj.getvalue()
                if hasattr(file_obj, "getvalue")
                else file_obj.read()
            )
            df_pandas = pd.read_excel(io.BytesIO(content), **kwargs)
        else:
            df_pandas = pd.read_excel(file_obj, **kwargs)

        return pl.from_pandas(df_pandas)  # type: ignore[no-any-return]


class JSONLoader(BaseLoader):
    """Loader for JSON files"""

    VALID_SUFFIXES: list[str] = [".json"]

    def load(self, file_obj: Any, **kwargs: Any) -> pl.DataFrame:
        if hasattr(file_obj, "read"):
            content = (
                file_obj.getvalue()
                if hasattr(file_obj, "getvalue")
                else file_obj.read()
            )
            if isinstance(content, bytes):
                content = content.decode("utf-8")
            return pl.read_json(io.StringIO(content), **kwargs)
        else:
            return pl.read_json(file_obj, **kwargs)


class ParquetLoader(BaseLoader):
    """Loader for Parquet files"""

    VALID_SUFFIXES: list[str] = [".parquet"]

    def load(self, file_obj: Any, **kwargs: Any) -> pl.DataFrame:
        if hasattr(file_obj, "read"):
            content = (
                file_obj.getvalue()
                if hasattr(file_obj, "getvalue")
                else file_obj.read()
            )
            return pl.read_parquet(io.BytesIO(content), **kwargs)
        else:
            return pl.read_parquet(file_obj, **kwargs)
