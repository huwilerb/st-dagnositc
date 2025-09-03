import io
from typing import Any, Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class FileLoaderProtocol(Protocol):
    """Protocol for file loaders"""

    def can_load(self, file_obj: Any) -> bool:
        """Check if this loader can handle the file"""
        ...

    def load(self, file_obj: Any, **kwargs: Any) -> pl.DataFrame:
        """Load the file and return a DataFrame"""
        ...


class CSVLoader:
    """Loader for CSV files"""

    def can_load(self, file_obj: Any) -> bool:
        return hasattr(file_obj, "name") and file_obj.name.lower().endswith(".csv")

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


class ExcelLoader:
    """Loader for Excel files"""

    def can_load(self, file_obj: Any) -> bool:
        if not hasattr(file_obj, "name"):
            return False
        return bool(file_obj.name.lower().endswith((".xlsx", ".xls")))

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


class JSONLoader:
    """Loader for JSON files"""

    def can_load(self, file_obj: Any) -> bool:
        return hasattr(file_obj, "name") and file_obj.name.lower().endswith(".json")

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


class ParquetLoader:
    """Loader for Parquet files"""

    def can_load(self, file_obj: Any) -> bool:
        return hasattr(file_obj, "name") and file_obj.name.lower().endswith(".parquet")

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
