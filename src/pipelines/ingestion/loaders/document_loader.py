"""
Document Loader for CSV data ingestion.

This module provides functionality to load and validate CSV files containing
product review data, ensuring data quality and proper structure before
further processing in the ingestion pipeline.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

from ..exceptions import ValidationError

logger = logging.getLogger(__name__)


@dataclass
class LoaderConfig:
    """Configuration for the DocumentLoader."""
    
    file_path: Path
    encoding: str = "utf-8"
    required_columns: Optional[List[str]] = None
    
    def __post_init__(self):
        """Set default required columns if not provided."""
        if self.required_columns is None:
            self.required_columns = [
                "product_name", 
                "description", 
                "price", 
                "rating", 
                "review_title", 
                "review_text"
            ]
        
        # Convert string path to Path object if needed
        if isinstance(self.file_path, str):
            self.file_path = Path(self.file_path)


class DocumentLoader:
    """
    Loads and validates CSV data for the ingestion pipeline.
    
    The DocumentLoader is responsible for:
    - Reading CSV files with proper encoding
    - Validating file existence and structure
    - Ensuring required columns are present
    - Filtering out empty rows
    - Providing detailed error messages for debugging
    """
    
    def __init__(self, config: LoaderConfig):
        """
        Initialize the DocumentLoader with configuration.
        
        Args:
            config: LoaderConfig containing file path and validation settings
        """
        self.config = config
        logger.info(f"Initialized DocumentLoader for file: {config.file_path}")
    
    def load(self) -> pd.DataFrame:
        """
        Load CSV file and return validated DataFrame.
        
        Returns:
            pd.DataFrame: Validated DataFrame with required columns and no empty rows
            
        Raises:
            FileNotFoundError: If the specified file doesn't exist
            ValidationError: If required columns are missing
        """
        logger.info(f"Loading CSV file: {self.config.file_path}")
        
        # Validate file exists
        self._validate_file_exists()
        
        # Load the CSV file
        try:
            df = pd.read_csv(
                self.config.file_path, 
                encoding=self.config.encoding
            )
            logger.info(f"Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        except Exception as e:
            logger.error(f"Failed to read CSV file: {e}")
            raise ValidationError(f"Failed to read CSV file: {e}")
        
        # Validate required columns
        self._validate_columns(df)
        
        # Filter empty rows
        df_filtered = self._filter_empty_rows(df)
        
        logger.info(f"Validation complete. Final DataFrame has {len(df_filtered)} rows")
        return df_filtered
    
    def _validate_file_exists(self) -> None:
        """
        Validate that the specified file exists.
        
        Raises:
            FileNotFoundError: If file doesn't exist with descriptive message
        """
        if not self.config.file_path.exists():
            error_msg = f"CSV file not found: {self.config.file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        if not self.config.file_path.is_file():
            error_msg = f"Path exists but is not a file: {self.config.file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
    
    def _validate_columns(self, df: pd.DataFrame) -> None:
        """
        Validate that all required columns are present in the DataFrame.
        
        Args:
            df: DataFrame to validate
            
        Raises:
            ValidationError: If required columns are missing, with list of missing columns
        """
        missing_columns = []
        for required_col in self.config.required_columns:
            if required_col not in df.columns:
                missing_columns.append(required_col)
        
        if missing_columns:
            error_msg = f"Missing required columns: {missing_columns}"
            logger.error(error_msg)
            raise ValidationError(
                error_msg,
                details={"missing_columns": missing_columns, "available_columns": list(df.columns)}
            )
        
        logger.info("All required columns are present")
    
    def _filter_empty_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove empty rows from the DataFrame and log warnings.
        
        A row is considered empty if:
        1. All values are NaN/None, OR
        2. All string values are empty or whitespace-only (ignoring numeric NaN values)
        
        Args:
            df: DataFrame to filter
            
        Returns:
            pd.DataFrame: DataFrame with empty rows removed
        """
        initial_count = len(df)
        
        # First remove rows where all values are NaN
        df_filtered = df.dropna(how='all')
        
        # Then remove rows where all non-NaN values are empty/whitespace strings
        if len(df_filtered) > 0:
            # For each row, check if all non-NaN values are empty/whitespace strings
            def is_empty_row(row):
                non_nan_values = []
                for value in row:
                    if pd.notna(value):  # Only consider non-NaN values
                        non_nan_values.append(str(value).strip())
                
                # If there are no non-NaN values, it's already handled by dropna(how='all')
                # If all non-NaN values are empty strings after stripping, it's empty
                return len(non_nan_values) > 0 and all(v == '' for v in non_nan_values)
            
            # Apply the empty row check
            empty_mask = df_filtered.apply(is_empty_row, axis=1)
            df_filtered = df_filtered[~empty_mask]
        
        empty_rows_count = initial_count - len(df_filtered)
        
        if empty_rows_count > 0:
            logger.warning(f"Filtered out {empty_rows_count} empty rows")
        
        return df_filtered