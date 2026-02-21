"""
Unit tests for RetrievalPipeline with metadata extraction integration.

Tests the retrieval pipeline orchestration including metadata extraction,
query processing, vector search with filters, and result caching.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from src.pipelines.retrieval.pipeline import (
    RetrievalPipeline,
    RetrievalConfig,
    RetrievalResult
)
from src.pipelines.retrieval.config import RetrievalSettings
from src.pipelines.retrieval.processors.query_processor import QueryConfig, ProcessedQuery
from src.pipelines.retrieval.search.vector_searcher import SearchConfig, MetadataFilter, SearchResult
from src.pipelines.retrieval.processors.context_compressor import CompressorConfig, CompressionResult
from src.pipelines.retrieval.processors.query_rewriter import RewriterConfig
from src.pipelines.retrieval.processors.document_formatter import FormatterConfig, FormattedContext
from src.pipelines.retrieval.cache.result_cache import CacheConfig
from src.pipelines.retrieval.processors.metadata_extractor import ExtractorConfig
from src.pipelines.retrieval.exceptions import ConfigurationError


class TestRetrievalPipelineMetadataExtraction:
    """Test RetrievalPipeline with metadata extraction integration."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration with metadata extraction enabled."""
        return RetrievalConfig(
            query_config=QueryConfig(),
            search_config=SearchConfig(),
            compressor_config=CompressorConfig(enabled=True),
            rewriter_config=RewriterConfig(),
            formatter_config=FormatterConfig(),
            cache_config=CacheConfig(enabled=False),  # Disable cache for testing
            extractor_config=ExtractorConfig(enabled=True)
        )
    
    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return RetrievalSettings(
            openai_api_key="test-openai-key",
            pinecone_api_key="test-pinecone-key",
            pinecone_index_name="test-index",
            metadata_extraction_enabled=True,
            metadata_extraction_model="gpt-3.5-turbo",
            metadata_extraction_timeout=3
        )
    
    @pytest.fixture
    def pipeline(self, config, settings):
        """Create RetrievalPipeline instance."""
        return RetrievalPipeline(config, settings)
    
    @pytest.fixture
    def initialized_pipeline(self, pipeline):
        """Create initialized pipeline with mocked components."""
        # Mock all components
        pipeline.query_processor = Mock()
        pipeline.metadata_extractor = Mock()
        pipeline.vector_searcher = Mock()
        pipeline.context_compressor = Mock()
        pipeline.query_rewriter = Mock()
        pipeline.document_formatter = Mock()
        pipeline.cache = Mock()
        pipeline._initialized = True
        
        # Setup default mock behaviors
        pipeline.cache.get.return_value = None  # No cache hit by default
        pipeline.cache._cache = {}  # Mock cache storage for len() calls
        pipeline.query_rewriter.reset_attempts.return_value = None
        pipeline.query_rewriter.should_rewrite.return_value = False
        
        return pipeline
    
    def test_pipeline_initialization(self, pipeline, config, settings):
        """Test that pipeline initializes correctly with all components."""
        assert pipeline.config == config
        assert pipeline.settings == settings
        assert pipeline.query_processor is None
        assert pipeline.metadata_extractor is None
        assert pipeline.vector_searcher is None
        assert pipeline._initialized is False
    
    def test_from_settings_includes_extractor_config(self, settings):
        """Test that from_settings creates pipeline with extractor config."""
        pipeline = RetrievalPipeline.from_settings(settings)
        
        assert pipeline.config.extractor_config.enabled == settings.metadata_extraction_enabled
        assert pipeline.config.extractor_config.llm_model == settings.metadata_extraction_model
        assert pipeline.config.extractor_config.timeout_seconds == settings.metadata_extraction_timeout
    
    def test_successful_metadata_extraction_with_filters(self, initialized_pipeline):
        """Test successful extraction with product name and price filters applied to search."""
        # Setup test data
        query = "iPhone 12 under $800"
        query_embedding = [0.1] * 3072
        
        # Mock extracted filters with both product name and price
        extracted_filters = MetadataFilter(
            product_name_pattern="iPhone 12",
            min_price=None,
            max_price=800.0,
            min_rating=None
        )
        
        # Mock query processor
        processed_query = ProcessedQuery(
            original=query,
            normalized=query,
            embedding=query_embedding,
            truncated=False
        )
        initialized_pipeline.query_processor.process.return_value = processed_query
        
        # Mock metadata extractor to return filters
        initialized_pipeline.metadata_extractor.extract.return_value = extracted_filters
        
        # Mock search results
        mock_docs = [
            Document(page_content="iPhone 12 review", metadata={"product_name": "iPhone 12", "price": 799.0})
        ]
        search_result = SearchResult(
            documents=mock_docs,
            scores=[0.9],
            query_embedding=query_embedding,
            filters_applied={
                "product_name": {"$regex": ".*iPhone 12.*", "$options": "i"},
                "price": {"$lte": 800.0}
            },
            search_metadata={"search_type": "mmr"}
        )
        initialized_pipeline.vector_searcher.search.return_value = search_result
        
        # Mock compression
        compression_result = CompressionResult(
            documents=mock_docs,
            filtered_count=1,
            relevance_scores=[0.9],
            compression_ratio=1.0,
            processing_time_ms=50.0
        )
        initialized_pipeline.context_compressor.compress.return_value = compression_result
        
        # Mock formatting
        formatted_context = FormattedContext(
            text="iPhone 12 review content",
            document_count=1,
            truncated=False
        )
        initialized_pipeline.document_formatter.format.return_value = formatted_context
        
        # Execute retrieval
        result = initialized_pipeline.retrieve(query)
        
        # Verify metadata extractor was called
        initialized_pipeline.metadata_extractor.extract.assert_called_once_with(query)
        
        # Verify vector searcher was called with extracted filters
        initialized_pipeline.vector_searcher.search.assert_called_once()
        call_args = initialized_pipeline.vector_searcher.search.call_args
        assert call_args[0][0] == query_embedding  # First positional arg is embedding
        assert call_args[0][1] == extracted_filters  # Second positional arg is filters
        
        # Verify result metadata includes auto_extracted_filters flag
        assert result.metadata.get('auto_extracted_filters') is True
        
        # Verify workflow steps include metadata_extraction with filter details
        workflow_steps = result.metadata.get('workflow_steps', [])
        extraction_steps = [step for step in workflow_steps if step['step'] == 'metadata_extraction']
        assert len(extraction_steps) == 1
        assert extraction_steps[0]['filters_extracted'] is True
        assert extraction_steps[0]['product_name'] == "iPhone 12"
        assert extraction_steps[0]['max_price'] == 800.0
        
        # Verify the extracted filters contain both product name and price
        assert extracted_filters.product_name_pattern == "iPhone 12"
        assert extracted_filters.max_price == 800.0
    
    def test_extraction_with_price_filters(self, initialized_pipeline):
        """Test extraction with price range filters."""
        query = "phones under $300"
        query_embedding = [0.1] * 3072
        
        # Mock extracted filters with price
        extracted_filters = MetadataFilter(
            product_name_pattern=None,
            min_price=None,
            max_price=300.0,
            min_rating=None
        )
        
        # Setup mocks
        processed_query = ProcessedQuery(
            original=query,
            normalized=query,
            embedding=query_embedding,
            truncated=False
        )
        initialized_pipeline.query_processor.process.return_value = processed_query
        initialized_pipeline.metadata_extractor.extract.return_value = extracted_filters
        
        mock_docs = [Document(page_content="Budget phone", metadata={"price": 250.0})]
        search_result = SearchResult(
            documents=mock_docs,
            scores=[0.8],
            query_embedding=query_embedding,
            filters_applied={"price": {"$lte": 300.0}},
            search_metadata={"search_type": "mmr"}
        )
        initialized_pipeline.vector_searcher.search.return_value = search_result
        
        compression_result = CompressionResult(
            documents=mock_docs,
            filtered_count=1,
            relevance_scores=[0.8],
            compression_ratio=1.0,
            processing_time_ms=50.0
        )
        initialized_pipeline.context_compressor.compress.return_value = compression_result
        
        formatted_context = FormattedContext(text="Budget phone content", document_count=1, truncated=False)
        initialized_pipeline.document_formatter.format.return_value = formatted_context
        
        # Execute retrieval
        result = initialized_pipeline.retrieve(query)
        
        # Verify filters were extracted and applied
        initialized_pipeline.metadata_extractor.extract.assert_called_once_with(query)
        call_args = initialized_pipeline.vector_searcher.search.call_args
        assert call_args[0][1].max_price == 300.0
        
        # Verify metadata includes price filter
        workflow_steps = result.metadata.get('workflow_steps', [])
        extraction_steps = [step for step in workflow_steps if step['step'] == 'metadata_extraction']
        assert extraction_steps[0]['max_price'] == 300.0
    
    def test_extraction_with_combined_filters(self, initialized_pipeline):
        """Test extraction with multiple filters (product name, price, rating)."""
        query = "highly rated Samsung phones over $400"
        query_embedding = [0.1] * 3072
        
        # Mock extracted filters with all fields
        extracted_filters = MetadataFilter(
            product_name_pattern="Samsung",
            min_price=400.0,
            max_price=None,
            min_rating=4.0
        )
        
        # Setup mocks
        processed_query = ProcessedQuery(
            original=query,
            normalized=query,
            embedding=query_embedding,
            truncated=False
        )
        initialized_pipeline.query_processor.process.return_value = processed_query
        initialized_pipeline.metadata_extractor.extract.return_value = extracted_filters
        
        mock_docs = [
            Document(
                page_content="Samsung Galaxy review",
                metadata={"product_name": "Samsung Galaxy", "price": 500.0, "rating": 4.5}
            )
        ]
        search_result = SearchResult(
            documents=mock_docs,
            scores=[0.9],
            query_embedding=query_embedding,
            filters_applied={
                "product_name": {"$regex": ".*Samsung.*", "$options": "i"},
                "price": {"$gte": 400.0},
                "rating": {"$gte": 4.0}
            },
            search_metadata={"search_type": "mmr"}
        )
        initialized_pipeline.vector_searcher.search.return_value = search_result
        
        compression_result = CompressionResult(
            documents=mock_docs,
            filtered_count=1,
            relevance_scores=[0.9],
            compression_ratio=1.0,
            processing_time_ms=50.0
        )
        initialized_pipeline.context_compressor.compress.return_value = compression_result
        
        formatted_context = FormattedContext(text="Samsung Galaxy content", document_count=1, truncated=False)
        initialized_pipeline.document_formatter.format.return_value = formatted_context
        
        # Execute retrieval
        result = initialized_pipeline.retrieve(query)
        
        # Verify all filters were applied
        call_args = initialized_pipeline.vector_searcher.search.call_args
        filters_arg = call_args[0][1]
        assert filters_arg.product_name_pattern == "Samsung"
        assert filters_arg.min_price == 400.0
        assert filters_arg.min_rating == 4.0
        
        # Verify metadata includes all filter details
        workflow_steps = result.metadata.get('workflow_steps', [])
        extraction_steps = [step for step in workflow_steps if step['step'] == 'metadata_extraction']
        assert extraction_steps[0]['product_name'] == "Samsung"
        assert extraction_steps[0]['min_price'] == 400.0
        assert extraction_steps[0]['min_rating'] == 4.0
    
    def test_no_extraction_when_filters_provided(self, initialized_pipeline):
        """Test that extraction is skipped when filters are manually provided."""
        query = "iPhone 12 price"
        query_embedding = [0.1] * 3072
        
        # Manually provided filters
        manual_filters = MetadataFilter(product_name_pattern="iPhone 13")
        
        # Setup mocks
        processed_query = ProcessedQuery(
            original=query,
            normalized=query,
            embedding=query_embedding,
            truncated=False
        )
        initialized_pipeline.query_processor.process.return_value = processed_query
        
        mock_docs = [Document(page_content="iPhone 13 review", metadata={"product_name": "iPhone 13"})]
        search_result = SearchResult(
            documents=mock_docs,
            scores=[0.9],
            query_embedding=query_embedding,
            filters_applied={"product_name": {"$regex": ".*iPhone 13.*", "$options": "i"}},
            search_metadata={"search_type": "mmr"}
        )
        initialized_pipeline.vector_searcher.search.return_value = search_result
        
        compression_result = CompressionResult(
            documents=mock_docs,
            filtered_count=1,
            relevance_scores=[0.9],
            compression_ratio=1.0,
            processing_time_ms=50.0
        )
        initialized_pipeline.context_compressor.compress.return_value = compression_result
        
        formatted_context = FormattedContext(text="iPhone 13 content", document_count=1, truncated=False)
        initialized_pipeline.document_formatter.format.return_value = formatted_context
        
        # Execute retrieval with manual filters
        result = initialized_pipeline.retrieve(query, filters=manual_filters)
        
        # Verify metadata extractor was NOT called
        initialized_pipeline.metadata_extractor.extract.assert_not_called()
        
        # Verify manual filters were used
        call_args = initialized_pipeline.vector_searcher.search.call_args
        assert call_args[0][1] == manual_filters
        
        # Verify metadata does NOT include auto_extracted_filters flag
        assert 'auto_extracted_filters' not in result.metadata
    
    def test_extraction_returns_none_no_filters_applied(self, initialized_pipeline):
        """Test that when extraction returns None, no filters are applied."""
        query = "What are smartphones?"
        query_embedding = [0.1] * 3072
        
        # Setup mocks
        processed_query = ProcessedQuery(
            original=query,
            normalized=query,
            embedding=query_embedding,
            truncated=False
        )
        initialized_pipeline.query_processor.process.return_value = processed_query
        
        # Mock extractor returns None (no extractable metadata)
        initialized_pipeline.metadata_extractor.extract.return_value = None
        
        mock_docs = [Document(page_content="General smartphone info", metadata={})]
        search_result = SearchResult(
            documents=mock_docs,
            scores=[0.7],
            query_embedding=query_embedding,
            filters_applied={},
            search_metadata={"search_type": "mmr"}
        )
        initialized_pipeline.vector_searcher.search.return_value = search_result
        
        compression_result = CompressionResult(
            documents=mock_docs,
            filtered_count=1,
            relevance_scores=[0.7],
            compression_ratio=1.0,
            processing_time_ms=50.0
        )
        initialized_pipeline.context_compressor.compress.return_value = compression_result
        
        formatted_context = FormattedContext(text="General info", document_count=1, truncated=False)
        initialized_pipeline.document_formatter.format.return_value = formatted_context
        
        # Execute retrieval
        result = initialized_pipeline.retrieve(query)
        
        # Verify extractor was called
        initialized_pipeline.metadata_extractor.extract.assert_called_once_with(query)
        
        # Verify search was called with None filters
        call_args = initialized_pipeline.vector_searcher.search.call_args
        assert call_args[0][1] is None
        
        # Verify metadata does NOT include auto_extracted_filters flag
        assert 'auto_extracted_filters' not in result.metadata
        
        # Verify workflow step shows no filters extracted
        workflow_steps = result.metadata.get('workflow_steps', [])
        extraction_steps = [step for step in workflow_steps if step['step'] == 'metadata_extraction']
        assert extraction_steps[0]['filters_extracted'] is False
    
    def test_extraction_disabled_no_extractor_initialized(self):
        """Test that when extraction is disabled, metadata_extractor is None and no extraction occurs."""
        # Create config with extraction disabled
        config = RetrievalConfig(
            query_config=QueryConfig(),
            search_config=SearchConfig(),
            compressor_config=CompressorConfig(enabled=True),
            rewriter_config=RewriterConfig(),
            formatter_config=FormatterConfig(),
            cache_config=CacheConfig(enabled=False),
            extractor_config=ExtractorConfig(enabled=False)  # Disabled
        )
        
        settings = RetrievalSettings(
            openai_api_key="test-openai-key",
            pinecone_api_key="test-pinecone-key",
            pinecone_index_name="test-index",
            metadata_extraction_enabled=False,  # Disabled in settings
            metadata_extraction_model="gpt-3.5-turbo",
            metadata_extraction_timeout=3
        )
        
        # Create pipeline
        pipeline = RetrievalPipeline(config, settings)
        
        # Mock all components except metadata_extractor
        pipeline.query_processor = Mock()
        pipeline.vector_searcher = Mock()
        pipeline.context_compressor = Mock()
        pipeline.query_rewriter = Mock()
        pipeline.document_formatter = Mock()
        pipeline.cache = Mock()
        pipeline._initialized = True
        
        # Verify metadata_extractor is None (not initialized)
        assert pipeline.metadata_extractor is None
        
        # Setup test data
        query = "iPhone 12 price"
        query_embedding = [0.1] * 3072
        
        # Setup mocks
        processed_query = ProcessedQuery(
            original=query,
            normalized=query,
            embedding=query_embedding,
            truncated=False
        )
        pipeline.query_processor.process.return_value = processed_query
        
        pipeline.cache.get.return_value = None
        pipeline.cache._cache = {}
        pipeline.query_rewriter.reset_attempts.return_value = None
        pipeline.query_rewriter.should_rewrite.return_value = False
        
        mock_docs = [Document(page_content="iPhone info", metadata={})]
        search_result = SearchResult(
            documents=mock_docs,
            scores=[0.8],
            query_embedding=query_embedding,
            filters_applied={},
            search_metadata={"search_type": "mmr"}
        )
        pipeline.vector_searcher.search.return_value = search_result
        
        compression_result = CompressionResult(
            documents=mock_docs,
            filtered_count=1,
            relevance_scores=[0.8],
            compression_ratio=1.0,
            processing_time_ms=50.0
        )
        pipeline.context_compressor.compress.return_value = compression_result
        
        formatted_context = FormattedContext(text="iPhone info", document_count=1, truncated=False)
        pipeline.document_formatter.format.return_value = formatted_context
        
        # Execute retrieval
        result = pipeline.retrieve(query)
        
        # Verify search was called with None filters (no extraction occurred)
        call_args = pipeline.vector_searcher.search.call_args
        assert call_args[0][1] is None
        
        # Verify metadata does NOT include auto_extracted_filters flag
        assert 'auto_extracted_filters' not in result.metadata
        
        # Verify workflow steps do NOT include metadata_extraction step
        workflow_steps = result.metadata.get('workflow_steps', [])
        extraction_steps = [step for step in workflow_steps if step['step'] == 'metadata_extraction']
        assert len(extraction_steps) == 0
    
    def test_extraction_failure_graceful_degradation(self, initialized_pipeline, caplog):
        """Test that extraction failures are handled gracefully and pipeline continues without filters."""
        import logging
        
        # Setup test data
        query = "iPhone 12 price"
        query_embedding = [0.1] * 3072
        
        # Setup mocks
        processed_query = ProcessedQuery(
            original=query,
            normalized=query,
            embedding=query_embedding,
            truncated=False
        )
        initialized_pipeline.query_processor.process.return_value = processed_query
        
        # Mock metadata extractor to raise an exception
        extraction_error = Exception("OpenAI API error: Rate limit exceeded")
        initialized_pipeline.metadata_extractor.extract.side_effect = extraction_error
        
        # Mock search results (should work without filters)
        mock_docs = [
            Document(page_content="iPhone 12 review", metadata={"product_name": "iPhone 12"}),
            Document(page_content="iPhone 13 review", metadata={"product_name": "iPhone 13"})
        ]
        search_result = SearchResult(
            documents=mock_docs,
            scores=[0.9, 0.8],
            query_embedding=query_embedding,
            filters_applied={},  # No filters applied due to extraction failure
            search_metadata={"search_type": "mmr"}
        )
        initialized_pipeline.vector_searcher.search.return_value = search_result
        
        # Mock compression
        compression_result = CompressionResult(
            documents=mock_docs,
            filtered_count=2,
            relevance_scores=[0.9, 0.8],
            compression_ratio=1.0,
            processing_time_ms=50.0
        )
        initialized_pipeline.context_compressor.compress.return_value = compression_result
        
        # Mock formatting
        formatted_context = FormattedContext(
            text="iPhone 12 and 13 reviews",
            document_count=2,
            truncated=False
        )
        initialized_pipeline.document_formatter.format.return_value = formatted_context
        
        # Execute retrieval with logging capture
        with caplog.at_level(logging.WARNING):
            result = initialized_pipeline.retrieve(query)
        
        # Verify metadata extractor was called
        initialized_pipeline.metadata_extractor.extract.assert_called_once_with(query)
        
        # Verify vector searcher was called with None filters (graceful degradation)
        initialized_pipeline.vector_searcher.search.assert_called_once()
        call_args = initialized_pipeline.vector_searcher.search.call_args
        assert call_args[0][0] == query_embedding  # First positional arg is embedding
        assert call_args[0][1] is None  # Second positional arg should be None (no filters)
        
        # Verify warning was logged
        assert any("Metadata extraction failed" in record.message for record in caplog.records)
        assert any("OpenAI API error: Rate limit exceeded" in record.message for record in caplog.records)
        
        # Verify result metadata does NOT include auto_extracted_filters flag
        assert 'auto_extracted_filters' not in result.metadata
        
        # Verify pipeline continued and returned results
        assert result.formatted_context == "iPhone 12 and 13 reviews"
        assert len(result.documents) == 2
        
        # Verify the pipeline did not crash and returned valid results
        assert result.query == query
        assert result.latency_ms >= 0
    
    def test_extraction_timeout_graceful_handling(self, initialized_pipeline, caplog):
        """Test that extraction timeout is handled gracefully and pipeline continues without filters."""
        import logging
        
        # Setup test data
        query = "Samsung Galaxy S21 reviews"
        query_embedding = [0.1] * 3072
        
        # Setup mocks
        processed_query = ProcessedQuery(
            original=query,
            normalized=query,
            embedding=query_embedding,
            truncated=False
        )
        initialized_pipeline.query_processor.process.return_value = processed_query
        
        # Mock metadata extractor to raise TimeoutError
        initialized_pipeline.metadata_extractor.extract.side_effect = TimeoutError("Metadata extraction timed out")
        
        # Mock search results (should work without filters after timeout)
        mock_docs = [
            Document(page_content="Samsung Galaxy S21 review", metadata={"product_name": "Samsung Galaxy S21"}),
            Document(page_content="Samsung Galaxy S20 review", metadata={"product_name": "Samsung Galaxy S20"})
        ]
        search_result = SearchResult(
            documents=mock_docs,
            scores=[0.9, 0.85],
            query_embedding=query_embedding,
            filters_applied={},  # No filters applied due to timeout
            search_metadata={"search_type": "mmr"}
        )
        initialized_pipeline.vector_searcher.search.return_value = search_result
        
        # Mock compression
        compression_result = CompressionResult(
            documents=mock_docs,
            filtered_count=2,
            relevance_scores=[0.9, 0.85],
            compression_ratio=1.0,
            processing_time_ms=50.0
        )
        initialized_pipeline.context_compressor.compress.return_value = compression_result
        
        # Mock formatting
        formatted_context = FormattedContext(
            text="Samsung Galaxy S21 and S20 reviews",
            document_count=2,
            truncated=False
        )
        initialized_pipeline.document_formatter.format.return_value = formatted_context
        
        # Execute retrieval with logging capture
        with caplog.at_level(logging.WARNING):
            result = initialized_pipeline.retrieve(query)
        
        # Verify metadata extractor was called
        initialized_pipeline.metadata_extractor.extract.assert_called_once_with(query)
        
        # Verify vector searcher was called with None filters (graceful degradation after timeout)
        initialized_pipeline.vector_searcher.search.assert_called_once()
        call_args = initialized_pipeline.vector_searcher.search.call_args
        assert call_args[0][0] == query_embedding  # First positional arg is embedding
        assert call_args[0][1] is None  # Second positional arg should be None (no filters due to timeout)
        
        # Verify warning was logged about the timeout
        assert any("Metadata extraction failed" in record.message for record in caplog.records)
        assert any("timed out" in record.message.lower() for record in caplog.records)
        
        # Verify result metadata does NOT include auto_extracted_filters flag
        assert 'auto_extracted_filters' not in result.metadata
        
        # Verify pipeline continued and returned results despite timeout
        assert result.formatted_context == "Samsung Galaxy S21 and S20 reviews"
        assert len(result.documents) == 2
        
        # Verify the pipeline did not crash and returned valid results
        assert result.query == query
        assert result.latency_ms >= 0
        
        # Verify no filters were applied in the result
        assert result.metadata.get('filters_applied', {}) == {}
