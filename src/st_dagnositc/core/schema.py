from dataclasses import dataclass, field
from typing import Any, Dict, List

from .models import DataType


@dataclass
class SchemaDefinition:
    """Defines the expected target schema"""

    columns: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def add_column(
        self,
        name: str,
        data_type: DataType = DataType.AUTO,
        required: bool = False,
        description: str = "",
    ) -> None:
        """Add a column to the schema"""
        self.columns[name] = {
            "data_type": data_type,
            "required": required,
            "description": description,
        }

    def get_required_columns(self) -> List[str]:
        """Get list of required column names"""
        return [
            name
            for name, config in self.columns.items()
            if config.get("required", False)
        ]

    def get_optional_columns(self) -> List[str]:
        """Get list of optional column names"""
        return [
            name
            for name, config in self.columns.items()
            if not config.get("required", False)
        ]
