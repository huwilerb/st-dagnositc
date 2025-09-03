from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import polars as pl


class DataType(Enum):
    """Supported data types for column mapping"""

    AUTO = "auto"
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    DATE = "date"
    TIME = "time"


class FileFormat(Enum):
    """Supported file formats"""

    CSV = "csv"
    EXCEL = "excel"
    JSON = "json"
    PARQUET = "parquet"


@dataclass
class ColumnMapping:
    """Represents a mapping from source column to target column"""

    source_column: str
    target_column: str
    data_type: DataType = DataType.AUTO
    include: bool = True
    required: bool = False
    validation_rules: List[str] = field(default_factory=list)
    transformation_func: Optional[Callable[[Any], Any]] = None


@dataclass
class ImportResult:
    """Result of data import process"""

    success: bool
    data: Optional[pl.DataFrame] = None
    mappings: List[ColumnMapping] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
