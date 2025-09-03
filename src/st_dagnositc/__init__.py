"""
st-dagnostic: The data-agnostic importer for Streamlit

A powerful component for importing and mapping data from various sources
into your Streamlit applications with a clean, intuitive interface.
"""

from .core import (
    DataType,
    FileFormat,
    ColumnMapping,
    ImportResult,
    SchemaDefinition,
    DagnosticEngine,
)

__version__ = "0.1.0"
__author__ = "Blaise Huwiler"
__email__ = "github@blaisehuwiler.ch"

__all__ = [
    "DataType",
    "FileFormat",
    "ColumnMapping",
    "ImportResult",
    "SchemaDefinition",
    "DagnosticEngine",
]
