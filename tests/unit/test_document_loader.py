"""
Property-based tests for DocumentLoader component.

These tests validate the correctness properties defined in the design document
using hypothesis for property-based testing.
"""

import tempfile
from pathlib import Path
from typing import List

import pandas as pd
import pytest
from hypothesis import given, strategies as st

from src.pipelines.ingestion.loaders import DocumentLoader, LoaderConfig
from src.pipelines.ingestion.exceptions import ValidationError


class TestDocumentLoaderProperties:
    """Property-based tests for DocumentLoader."""

    @given(
        # Generate lists of rows where some are empty
        st.lists(
            st.one_of(
                # Valid rows with all required columns
                st.fixed_dictionaries({
                    "product_name": st.text(min_size=1, max_size=50),
                    "description": st.text(min_size=1, max_size=100),
                    "price": st.floats(min_value=0.01, max_value=10000.0),
                    "rating": st.floats(min_value=1.0, max_value=5.0),
                    "review_title": st.text(min_size=1, max_size=50),
                    "review_text": st.text(min_size=1, max_size=200)
                }),
                # Empty rows (all NaN)
                st.fixed_dictionaries({
                    "product_name": st.just(None),
                    "description": st.just(None),
                    "price": st.just(None),
                    "rating": st.just(None),
                    "review_title": st.just(None),
                    "review_text": st.just(None)
                }),
                # Whitespace-only rows
                st.fixed_dictionaries({
                    "product_name": st.just("   "),
                    "description": st.just(""),
                    "price": st.just(None),
                    "rating": st.just(None),
                    "review_title": st.just("\t\n"),
                    "review_text": st.just("  ")
                })
            ),
            min_size=1,
            max_size=20
        )
    )
    def test_empty_row_filtering_property(self, rows: List[dict]):
        """
        **Feature: data-ingestion-pipeline, Property 3: Empty Row Filtering**
        
        For any CSV file containing N total rows where M rows are empty,
        the Document_Loader SHALL return a DataFrame with exactly (N - M) rows.
        
        **Validates: Requirements 1.4**
        """
        # Create a temporary CSV file with the generated data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame(rows)
            df.to_csv(f.name, index=False)
            temp_path = Path(f.name)
        
        try:
            # Count empty rows manually
            # A row is empty if all values are NaN/None or all string values are whitespace
            empty_count = 0
            for row in rows:
                # Check if all values are None/NaN
                all_none = all(v is None for v in row.values())
                
                # Check if all string values are empty/whitespace
                string_values = [str(v).strip() if v is not None else "" for v in row.values()]
                all_empty_strings = all(v == "" for v in string_values)
                
                if all_none or all_empty_strings:
                    empty_count += 1
            
            total_rows = len(rows)
            expected_rows = total_rows - empty_count
            
            # Load using DocumentLoader
            config = LoaderConfig(file_path=temp_path)
            loader = DocumentLoader(config)
            result_df = loader.load()
            
            # Verify the property: result should have exactly (N - M) rows
            assert len(result_df) == expected_rows, (
                f"Expected {expected_rows} rows (total: {total_rows}, empty: {empty_count}), "
                f"but got {len(result_df)} rows"
            )
            
        finally:
            # Clean up temporary file
            temp_path.unlink(missing_ok=True)


class TestDocumentLoaderUnitTests:
    """Unit tests for specific DocumentLoader behaviors."""
    
    def test_file_not_found_error(self):
        """Test that FileNotFoundError is raised for non-existent files."""
        config = LoaderConfig(file_path=Path("non_existent_file.csv"))
        loader = DocumentLoader(config)
        
        with pytest.raises(FileNotFoundError) as exc_info:
            loader.load()
        
        assert "CSV file not found" in str(exc_info.value)
    
    def test_missing_columns_validation_error(self):
        """Test that ValidationError is raised for missing required columns."""
        # Create CSV with missing columns
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame({
                "product_name": ["Product 1"],
                "description": ["Description 1"]
                # Missing: price, rating, review_title, review_text
            })
            df.to_csv(f.name, index=False)
            temp_path = Path(f.name)
        
        try:
            config = LoaderConfig(file_path=temp_path)
            loader = DocumentLoader(config)
            
            with pytest.raises(ValidationError) as exc_info:
                loader.load()
            
            assert "Missing required columns" in str(exc_info.value)
            # Should list the missing columns
            error_details = exc_info.value.details
            assert "missing_columns" in error_details
            missing_cols = error_details["missing_columns"]
            expected_missing = ["price", "rating", "review_title", "review_text"]
            assert all(col in missing_cols for col in expected_missing)
            
        finally:
            temp_path.unlink(missing_ok=True)
    
    def test_successful_loading_with_valid_data(self):
        """Test successful loading of valid CSV data."""
        # Create valid CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame({
                "product_name": ["Product 1", "Product 2"],
                "description": ["Desc 1", "Desc 2"],
                "price": [10.99, 20.99],
                "rating": [4.5, 3.8],
                "review_title": ["Great!", "Good"],
                "review_text": ["Love it", "Nice product"]
            })
            df.to_csv(f.name, index=False)
            temp_path = Path(f.name)
        
        try:
            config = LoaderConfig(file_path=temp_path)
            loader = DocumentLoader(config)
            result_df = loader.load()
            
            assert len(result_df) == 2
            assert list(result_df.columns) == [
                "product_name", "description", "price", 
                "rating", "review_title", "review_text"
            ]
            
        finally:
            temp_path.unlink(missing_ok=True)