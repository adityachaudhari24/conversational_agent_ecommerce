"""Unit tests for metadata extraction workflow step logging.

This module tests that the metadata extraction step correctly logs
workflow information including filter details.

Tests cover:
- Workflow step includes all required fields
- Filter details are correctly captured
- Workflow step is added when extraction succeeds
- Workflow step is added when extraction fails
"""

import pytest
from unittest.mock import Mock, patch
from src.pipelines.retrieval.pipeline import RetrievalPipeline, RetrievalConfig
from src.pipelines.retrieval.config import RetrievalSettings
from src.pipelines.retrieval.search.vector_searcher import MetadataFilter
from src.pipelines.retrieval.processors.metadata_extractor import ExtractorConfig


@pytest.fixture
def mock_settings():
    """Create mock retrieval settings."""
    settings = Mock(spec=RetrievalSettings)
    settings.openai_api_key = "test-key"
    settings.pinecone_api_key = "test-key"
    settings.pinecone_index_name = "test-index"
    settings.pinecone_namespace = ""
    settings.embedding_model = "text-embedding-3-small"
    settings.embedding_dimension = 1536
    settings.max_query_length = 500
    settings.normalize_unicode = True
    settings.top_k = 4
    settings.fetch_k = 20
    settings.lambda_mult = 0.7
    settings.score_threshold = 0.6
    settings.search_type = "mmr"
    settings.compression_enabled = True
    settings.relevance_prompt = "test"
    settings.max_rewrite_attempts = 2
    settings.rewrite_threshold = 0.7
    settings.rewrite_prompt = "test"
    settings.format_template = "test"
    settings.format_delimiter = "\n"
    settings.include_scores = False
    settings.max_context_length = 4000
    settings.cache_enabled = True
    settings.cache_ttl_seconds = 3600
    settings.cache_max_size = 100
    settings.max_retries = 3
    settings.retry_delay_seconds = 1.0
    settings.enable_evaluation = False
    settings.metadata_extraction_enabled = True
    settings.metadata_extraction_model = "gpt-3.5-turbo"
    settings.metadata_extraction_timeout = 3
    return settings


def test_workflow_step_includes_all_filter_fields(mock_settings):
    """Test that workflow step entry includes all required filter fields.
    
    **Validates: Requirement 7.3, 7.4**
    
    When metadata extraction succeeds, the workflow step should include:
    - step name
    - iteration number
    - time taken
    - filters_extracted flag
    - product_name
    - min_price
    - max_price
    - min_rating
    """
    # Create pipeline
    pipeline = RetrievalPipeline.from_settings(mock_settings)
    
    # Mock all components
    with patch.object(pipeline, 'query_processor') as mock_qp, \
         patch.object(pipeline, 'metadata_extractor') as mock_me, \
         patch.object(pipeline, 'vector_searcher') as mock_vs, \
         patch.object(pipeline, 'context_compressor') as mock_cc, \
         patch.object(pipeline, 'query_rewriter') as mock_qr, \
         patch.object(pipeline, 'document_formatter') as mock_df:
        
        # Set up mocks
        pipeline._initialized = True
        
        # Mock query processor
        mock_processed_query = Mock()
        mock_processed_query.embedding = [0.1] * 1536
        mock_processed_query.truncated = False
        mock_qp.process.return_value = mock_processed_query
        
        # Mock metadata extractor to return filters
        mock_filter = MetadataFilter(
            product_name_pattern="iPhone 12",
            min_price=500.0,
            max_price=1000.0,
            min_rating=4.5
        )
        mock_me.extract.return_value = mock_filter
        
        # Mock vector searcher
        mock_search_result = Mock()
        mock_search_result.documents = [Mock()]
        mock_search_result.scores = [0.9]
        mock_search_result.search_metadata = {'search_type': 'mmr'}
        mock_vs.search.return_value = mock_search_result
        
        # Mock context compressor
        mock_compression_result = Mock()
        mock_compression_result.documents = [Mock()]
        mock_compression_result.relevance_scores = [0.9]
        mock_compression_result.compression_ratio = 1.0
        mock_cc.compress.return_value = mock_compression_result
        
        # Mock query rewriter
        mock_qr.should_rewrite.return_value = False
        
        # Mock document formatter
        mock_formatted = Mock()
        mock_formatted.text = "formatted text"
        mock_formatted.document_count = 1
        mock_formatted.truncated = False
        mock_df.format.return_value = mock_formatted
        
        # Execute retrieval
        result = pipeline._execute_retrieval_workflow("iPhone 12 price")
        
        # Find the metadata extraction workflow step
        extraction_steps = [
            step for step in result.metadata['workflow_steps']
            if step['step'] == 'metadata_extraction'
        ]
        
        assert len(extraction_steps) == 1, "Should have exactly one metadata extraction step"
        
        step = extraction_steps[0]
        
        # Verify all required fields are present
        assert 'step' in step
        assert step['step'] == 'metadata_extraction'
        
        assert 'iteration' in step
        assert isinstance(step['iteration'], int)
        
        assert 'time_ms' in step
        assert isinstance(step['time_ms'], (int, float))
        assert step['time_ms'] >= 0
        
        assert 'filters_extracted' in step
        assert step['filters_extracted'] is True
        
        assert 'product_name' in step
        assert step['product_name'] == "iPhone 12"
        
        assert 'min_price' in step
        assert step['min_price'] == 500.0
        
        assert 'max_price' in step
        assert step['max_price'] == 1000.0
        
        assert 'min_rating' in step
        assert step['min_rating'] == 4.5


def test_workflow_step_with_no_filters_extracted(mock_settings):
    """Test workflow step when no filters are extracted.
    
    **Validates: Requirement 7.3, 7.4**
    """
    pipeline = RetrievalPipeline.from_settings(mock_settings)
    
    with patch.object(pipeline, 'query_processor') as mock_qp, \
         patch.object(pipeline, 'metadata_extractor') as mock_me, \
         patch.object(pipeline, 'vector_searcher') as mock_vs, \
         patch.object(pipeline, 'context_compressor') as mock_cc, \
         patch.object(pipeline, 'query_rewriter') as mock_qr, \
         patch.object(pipeline, 'document_formatter') as mock_df:
        
        pipeline._initialized = True
        
        # Mock query processor
        mock_processed_query = Mock()
        mock_processed_query.embedding = [0.1] * 1536
        mock_processed_query.truncated = False
        mock_qp.process.return_value = mock_processed_query
        
        # Mock metadata extractor to return None (no filters)
        mock_me.extract.return_value = None
        
        # Mock vector searcher
        mock_search_result = Mock()
        mock_search_result.documents = [Mock()]
        mock_search_result.scores = [0.9]
        mock_search_result.search_metadata = {'search_type': 'mmr'}
        mock_vs.search.return_value = mock_search_result
        
        # Mock context compressor
        mock_compression_result = Mock()
        mock_compression_result.documents = [Mock()]
        mock_compression_result.relevance_scores = [0.9]
        mock_compression_result.compression_ratio = 1.0
        mock_cc.compress.return_value = mock_compression_result
        
        # Mock query rewriter
        mock_qr.should_rewrite.return_value = False
        
        # Mock document formatter
        mock_formatted = Mock()
        mock_formatted.text = "formatted text"
        mock_formatted.document_count = 1
        mock_formatted.truncated = False
        mock_df.format.return_value = mock_formatted
        
        # Execute retrieval
        result = pipeline._execute_retrieval_workflow("What are smartphones?")
        
        # Find the metadata extraction workflow step
        extraction_steps = [
            step for step in result.metadata['workflow_steps']
            if step['step'] == 'metadata_extraction'
        ]
        
        assert len(extraction_steps) == 1
        step = extraction_steps[0]
        
        # Verify filters_extracted is False
        assert step['filters_extracted'] is False
        
        # Verify all filter fields are None
        assert step['product_name'] is None
        assert step['min_price'] is None
        assert step['max_price'] is None
        assert step['min_rating'] is None


def test_workflow_step_with_partial_filters(mock_settings):
    """Test workflow step when only some filters are extracted.
    
    **Validates: Requirement 7.3, 7.4**
    """
    pipeline = RetrievalPipeline.from_settings(mock_settings)
    
    with patch.object(pipeline, 'query_processor') as mock_qp, \
         patch.object(pipeline, 'metadata_extractor') as mock_me, \
         patch.object(pipeline, 'vector_searcher') as mock_vs, \
         patch.object(pipeline, 'context_compressor') as mock_cc, \
         patch.object(pipeline, 'query_rewriter') as mock_qr, \
         patch.object(pipeline, 'document_formatter') as mock_df:
        
        pipeline._initialized = True
        
        # Mock query processor
        mock_processed_query = Mock()
        mock_processed_query.embedding = [0.1] * 1536
        mock_processed_query.truncated = False
        mock_qp.process.return_value = mock_processed_query
        
        # Mock metadata extractor to return partial filters (only price)
        mock_filter = MetadataFilter(
            max_price=300.0
        )
        mock_me.extract.return_value = mock_filter
        
        # Mock vector searcher
        mock_search_result = Mock()
        mock_search_result.documents = [Mock()]
        mock_search_result.scores = [0.9]
        mock_search_result.search_metadata = {'search_type': 'mmr'}
        mock_vs.search.return_value = mock_search_result
        
        # Mock context compressor
        mock_compression_result = Mock()
        mock_compression_result.documents = [Mock()]
        mock_compression_result.relevance_scores = [0.9]
        mock_compression_result.compression_ratio = 1.0
        mock_cc.compress.return_value = mock_compression_result
        
        # Mock query rewriter
        mock_qr.should_rewrite.return_value = False
        
        # Mock document formatter
        mock_formatted = Mock()
        mock_formatted.text = "formatted text"
        mock_formatted.document_count = 1
        mock_formatted.truncated = False
        mock_df.format.return_value = mock_formatted
        
        # Execute retrieval
        result = pipeline._execute_retrieval_workflow("phones under $300")
        
        # Find the metadata extraction workflow step
        extraction_steps = [
            step for step in result.metadata['workflow_steps']
            if step['step'] == 'metadata_extraction'
        ]
        
        assert len(extraction_steps) == 1
        step = extraction_steps[0]
        
        # Verify filters_extracted is True
        assert step['filters_extracted'] is True
        
        # Verify only max_price is set
        assert step['product_name'] is None
        assert step['min_price'] is None
        assert step['max_price'] == 300.0
        assert step['min_rating'] is None
