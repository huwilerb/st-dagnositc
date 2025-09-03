from typing import Any, Dict, List, Optional, Tuple

import polars as pl

from .loader import (
    CSVLoader,
    ExcelLoader,
    FileLoaderProtocol,
    JSONLoader,
    ParquetLoader,
)
from .models import ColumnMapping, DataType, ImportResult
from .schema import SchemaDefinition


class ColumnMapper:
    """Core engine for column mapping logic"""

    def __init__(self, schema: Optional[SchemaDefinition] = None):
        self.schema = schema or SchemaDefinition()
        self.similarity_threshold = 0.6

    def suggest_mappings(self, source_df: pl.DataFrame) -> List[ColumnMapping]:
        """Suggest automatic column mappings based on schema and heuristics"""
        mappings = []

        for source_col in source_df.columns:
            suggested_target = self._suggest_target_column(source_col)
            suggested_type = self._detect_column_type(source_df.select(source_col))

            mapping = ColumnMapping(
                source_column=source_col,
                target_column=suggested_target,
                data_type=suggested_type,
                include=True,
            )
            mappings.append(mapping)

        return mappings

    def _suggest_target_column(self, source_col: str) -> str:
        """Suggest target column name based on schema and common patterns"""
        source_lower = source_col.lower().strip()

        for target_col in self.schema.columns.keys():
            if (
                self._calculate_similarity(source_lower, target_col.lower())
                > self.similarity_threshold
            ):
                return target_col

        common_patterns = {
            "name": ["name", "full_name", "fullname", "customer_name", "user_name"],
            "email": ["email", "email_address", "e_mail", "mail"],
            "phone": ["phone", "telephone", "mobile", "cell", "phone_number"],
            "age": ["age", "years_old", "birth_year"],
            "city": ["city", "location", "town", "municipality"],
            "country": ["country", "nation", "country_code"],
            "address": ["address", "street", "location", "addr"],
            "date": ["date", "created_at", "timestamp", "time"],
            "id": ["id", "identifier", "key", "primary_key"],
        }

        for target, patterns in common_patterns.items():
            if any(pattern in source_lower for pattern in patterns):
                return target

        return self._clean_column_name(source_col)

    def _detect_column_type(self, df_col: pl.DataFrame) -> DataType:
        """Detect the most appropriate data type for a column"""
        col_name = df_col.columns[0]
        dtype = df_col.dtypes[0]

        if dtype in [
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
        ]:
            return DataType.INTEGER
        elif dtype in [pl.Float32, pl.Float64]:
            return DataType.FLOAT
        elif dtype == pl.Boolean:
            return DataType.BOOLEAN
        elif dtype in [pl.Date, pl.Datetime]:
            return DataType.DATETIME
        elif dtype == pl.Time:
            return DataType.TIME
        elif dtype == pl.Utf8:
            sample = df_col.head(100).to_series().drop_nulls()

            if len(sample) > 0:
                try:
                    df_col.with_columns(
                        pl.col(col_name).str.strptime(pl.Datetime, strict=False)
                    )
                    return DataType.DATETIME
                except Exception:
                    pass

                try:
                    numeric_result = df_col.with_columns(
                        pl.col(col_name).cast(pl.Float64, strict=False)
                    )
                    if (
                        numeric_result.select(pl.col(col_name).is_not_null())
                        .sum()
                        .item()
                        > len(sample) * 0.8
                    ):
                        return DataType.FLOAT
                except Exception:
                    pass

        return DataType.STRING

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings"""
        if str1 == str2:
            return 1.0

        if str1 in str2 or str2 in str1:
            return 0.8

        # Jaccard similarity on character level
        set1, set2 = set(str1), set(str2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def _clean_column_name(self, col_name: str) -> str:
        """Clean column name to follow naming conventions"""
        import re

        cleaned = re.sub(r"[^\w\s]", "", col_name.lower())
        cleaned = re.sub(r"\s+", "_", cleaned.strip())
        return cleaned


class DataProcessor:
    """Core data processing engine"""

    def process_dataframe(
        self, df: pl.DataFrame, mappings: List[ColumnMapping]
    ) -> ImportResult:
        """Process DataFrame according to column mappings"""
        result = ImportResult(success=False)

        try:
            included_mappings = [m for m in mappings if m.include and m.target_column]

            if not included_mappings:
                result.errors.append("No column mappings provided")
                return result

            expressions = []

            for mapping in included_mappings:
                if mapping.source_column not in df.columns:
                    result.errors.append(
                        f"Source column '{mapping.source_column}' not found"
                    )
                    continue

                expr = pl.col(mapping.source_column)

                if mapping.data_type != DataType.AUTO:
                    expr = self._convert_column_type(expr, mapping.data_type)

                if mapping.transformation_func:
                    try:
                        expr = expr.map_elements(
                            mapping.transformation_func, return_dtype=pl.Utf8
                        )
                    except Exception as e:
                        result.warnings.append(
                            f"Transformation failed for {mapping.source_column}: {e}"
                        )

                expr = expr.alias(mapping.target_column)
                expressions.append(expr)

            if expressions:
                try:
                    result.data = df.select(expressions)
                    result.mappings = included_mappings
                    result.success = True
                    result.metadata = {
                        "original_shape": df.shape,
                        "processed_shape": result.data.shape,
                        "columns_mapped": len(included_mappings),
                    }
                except Exception as e:
                    result.errors.append(f"Column transformation failed: {str(e)}")
            else:
                result.errors.append("No valid mappings to process")

        except Exception as e:
            result.errors.append(f"Processing failed: {str(e)}")

        return result

    def _convert_column_type(self, expr: pl.Expr, target_type: DataType) -> pl.Expr:
        """Convert column expression to target type"""
        try:
            if target_type == DataType.STRING:
                return expr.cast(pl.Utf8)
            elif target_type == DataType.INTEGER:
                return expr.cast(pl.Int64, strict=False)
            elif target_type == DataType.FLOAT:
                return expr.cast(pl.Float64, strict=False)
            elif target_type == DataType.BOOLEAN:
                # Handle various boolean representations
                return (
                    expr.cast(pl.Utf8)
                    .str.to_lowercase()
                    .map_elements(
                        lambda x: (
                            x in ["true", "1", "yes", "y"] if x is not None else None
                        ),
                        return_dtype=pl.Boolean,
                    )
                )
            elif target_type == DataType.DATETIME:
                return expr.str.strptime(pl.Datetime, strict=False)
            elif target_type == DataType.DATE:
                return expr.str.strptime(pl.Date, strict=False)
            elif target_type == DataType.TIME:
                return expr.str.strptime(pl.Time, strict=False)
            else:
                return expr
        except Exception:
            return expr


class DagnosticEngine:
    """Main engine that orchestrates the data import process"""

    def __init__(self, schema: Optional[SchemaDefinition] = None):
        self.schema = schema or SchemaDefinition()
        self.file_loaders: List[FileLoaderProtocol] = [
            CSVLoader(),
            ExcelLoader(),
            JSONLoader(),
            ParquetLoader(),
        ]
        self.column_mapper = ColumnMapper(self.schema)
        self.data_processor = DataProcessor()

    def load_file(
        self,
        file_obj: Any,
        **kwargs: Any,
    ) -> Tuple[Optional[pl.DataFrame], List[str]]:
        """Load file and return (DataFrame, errors)"""
        for loader in self.file_loaders:
            if loader.can_load(file_obj):
                try:
                    df = loader.load(file_obj, **kwargs)
                    return df, []
                except Exception as e:
                    return None, [f"Failed to load file: {str(e)}"]

        return None, ["Unsupported file format"]

    def suggest_mappings(self, df: pl.DataFrame) -> List[ColumnMapping]:
        """Generate suggested column mappings"""
        return self.column_mapper.suggest_mappings(df)

    def validate_mappings(self, mappings: List[ColumnMapping]) -> List[str]:
        """Validate column mappings against schema"""
        errors = []

        # Check required columns
        required_cols = self.schema.get_required_columns()
        mapped_targets = {m.target_column for m in mappings if m.include}

        missing_required = set(required_cols) - mapped_targets
        if missing_required:
            errors.append(f"Missing required columns: {missing_required}")

        # Check for duplicate target columns
        target_counts: Dict[str, int] = {}
        for mapping in mappings:
            if mapping.include and mapping.target_column:
                target_counts[mapping.target_column] = (
                    target_counts.get(mapping.target_column, 0) + 1
                )

        duplicates = [col for col, count in target_counts.items() if count > 1]
        if duplicates:
            errors.append(f"Duplicate target columns: {duplicates}")

        return errors

    def process_data(
        self, df: pl.DataFrame, mappings: List[ColumnMapping]
    ) -> ImportResult:
        """Process data with given mappings"""
        # Validate mappings first
        validation_errors = self.validate_mappings(mappings)
        if validation_errors:
            return ImportResult(success=False, errors=validation_errors)

        # Process the data
        return self.data_processor.process_dataframe(df, mappings)
