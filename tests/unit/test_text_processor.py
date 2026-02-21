"""
Unit tests for TextProcessor component.

Tests the core functionality of converting DataFrame rows to Document objects,
including data sanitization, validation, and error handling.
"""

import pandas as pd
import pytest
from langchain_core.documents import Document

from src.pipelines.ingestion.processors.text_processor import TextProcessor, ProcessorConfig


class TestTextProcessor:
    """Test suite for TextProcessor class."""
    
    def test_process_valid_data(self):
        """Test processing valid DataFrame rows into Documents."""
        # Create test data
        data = {
            'product_name': ['iPhone 14', 'Samsung Galaxy'],
            'description': ['Latest iPhone', 'Android phone'],
            'price': [999.99, 799.99],
            'rating': [4.5, 4.2],
            'review_title': ['Great phone', 'Good value'],
            'review_text': ['Love this phone!', 'Works well for the price']
        }
        df = pd.DataFrame(data)
        
        # Initialize processor
        config = ProcessorConfig()
        processor = TextProcessor(config)
        
        # Process data
        documents = processor.process(df)
        
        # Verify results
        assert len(documents) == 2
        assert isinstance(documents[0], Document)
        assert documents[0].page_content == 'Product: iPhone 14 | Price: $999.99 | Rating: 4.5/5 | Review: Love this phone!'
        assert documents[0].metadata['product_name'] == 'iPhone 14'
        assert documents[0].metadata['price'] == '999.99'
        
        assert documents[1].page_content == 'Product: Samsung Galaxy | Price: $799.99 | Rating: 4.2/5 | Review: Works well for the price'
        assert documents[1].metadata['product_name'] == 'Samsung Galaxy'
    
    def test_sanitize_nan_values(self):
        """Test that NaN values are properly replaced with defaults."""
        # Create test data with NaN values
        data = {
            'product_name': ['iPhone 14'],
            'description': [None],  # Will become NaN
            'price': [float('nan')],
            'rating': [4.5],
            'review_title': [''],
            'review_text': ['Great phone!']
        }
        df = pd.DataFrame(data)
        
        # Initialize processor
        config = ProcessorConfig()
        processor = TextProcessor(config)
        
        # Process data
        documents = processor.process(df)
        
        # Verify NaN handling
        assert len(documents) == 1
        metadata = documents[0].metadata
        assert metadata['description'] == ''  # Text field gets empty string
        assert metadata['price'] == 'N/A'     # Numeric field gets "N/A"
        assert metadata['review_title'] == ''
    
    def test_skip_empty_content(self):
        """Test that records with empty or whitespace-only content are skipped."""
        # Create test data with empty/whitespace content
        data = {
            'product_name': ['iPhone 14', 'Samsung Galaxy', 'Google Pixel'],
            'description': ['Latest iPhone', 'Android phone', 'Google phone'],
            'price': [999.99, 799.99, 699.99],
            'rating': [4.5, 4.2, 4.0],
            'review_title': ['Great phone', 'Good value', 'Nice phone'],
            'review_text': ['Love this phone!', '', '   \n\t  ']  # Valid, empty, whitespace-only
        }
        df = pd.DataFrame(data)
        
        # Initialize processor
        config = ProcessorConfig()
        processor = TextProcessor(config)
        
        # Process data
        documents = processor.process(df)
        
        # Verify only valid content is processed
        assert len(documents) == 1
        assert 'Review: Love this phone!' in documents[0].page_content
        
        # Check validation report
        report = processor.get_validation_report()
        assert report['total_skipped'] == 2
        assert 'Empty or whitespace-only content' in report['skip_reasons']
    
    def test_validation_report(self):
        """Test that validation report provides accurate statistics."""
        # Create test data with some invalid records
        data = {
            'product_name': ['iPhone 14', 'Samsung Galaxy'],
            'description': ['Latest iPhone', 'Android phone'],
            'price': [999.99, 799.99],
            'rating': [4.5, 4.2],
            'review_title': ['Great phone', 'Good value'],
            'review_text': ['Love this phone!', '']  # One valid, one empty
        }
        df = pd.DataFrame(data)
        
        # Initialize processor
        config = ProcessorConfig()
        processor = TextProcessor(config)
        
        # Process data
        documents = processor.process(df)
        
        # Get validation report
        report = processor.get_validation_report()
        
        # Verify report contents
        assert report['total_skipped'] == 1
        assert len(report['skipped_records']) == 1
        assert report['skipped_records'][0]['index'] == 1
        assert 'Empty or whitespace-only content' in report['skipped_records'][0]['reason']
    
    def test_custom_config(self):
        """Test processor with custom configuration."""
        # Create test data
        data = {
            'title': ['Product A'],
            'content': ['This is the main content'],
            'category': ['Electronics'],
            'custom_field': ['Custom value']
        }
        df = pd.DataFrame(data)
        
        # Initialize processor with custom config
        config = ProcessorConfig(
            content_field='content',
            metadata_fields=['title', 'category', 'custom_field']
        )
        processor = TextProcessor(config)
        
        # Process data
        documents = processor.process(df)
        
        # Verify custom configuration is used
        assert len(documents) == 1
        assert 'Review: This is the main content' in documents[0].page_content
        assert documents[0].metadata['title'] == 'Product A'
        assert documents[0].metadata['category'] == 'Electronics'
        assert documents[0].metadata['custom_field'] == 'Custom value'


if __name__ == "__main__":
    pytest.main([__file__])