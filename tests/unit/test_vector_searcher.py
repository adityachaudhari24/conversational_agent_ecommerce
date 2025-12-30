"""
Unit tests for VectorSearcher.

Tests the vector search functionality including configuration,
initialization, search operations, and filtering.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from src.pipelines.retrieval.search.vector_searcher import (
    VectorSearcher, SearchConfig, MetadataFilter, SearchResult
)
from src.pipelines.retrieval.config import RetrievalSettings
from src.pipelines.retrieval.exceptions import SearchError, ConnectionError, ConfigurationError


class TestSearchConfig:
    """Test SearchConfig dataclass."""
    
    def test_default_config(self):
        """Test SearchConfig with default values."""
        config = SearchConfig()
        
        assert config.top_k == 4
        assert config.fetch_k == 20
        assert config.lambda_mult == 0.7
        assert config.score_threshold == 0.6
        assert config.search_type == "mmr"
    
    def test_custom_config(self):
        """Test SearchConfig with custom values."""
        config = SearchConfig(
            top_k=10,
            fetch_k=50,
            lambda_mult=0.5,
            score_threshold=0.8,
            search_type="similarity"
        )
        
        assert config.top_k == 10
        assert config.fetch_k == 50
        assert config.lambda_mult == 0.5
        assert config.score_threshold == 0.8
        assert config.search_type == "similarity"


class TestMetadataFilter:
    """Test MetadataFilter dataclass."""
    
    def test_default_filter(self):
        """Test MetadataFilter with default values."""
        filter_obj = MetadataFilter()
        
        assert filter_obj.min_price is None
        assert filter_obj.max_price is None
        assert filter_obj.min_rating is None
        assert filter_obj.product_name_pattern is None
    
    def test_custom_filter(self):
        """Test MetadataFilter with custom values."""
        filter_obj = MetadataFilter(
            min_price=10.0,
            max_price=100.0,
            min_rating=4.0,
            product_name_pattern="phone"
        )
        
        assert filter_obj.min_price == 10.0
        assert filter_obj.max_price == 100.0
        assert filter_obj.min_rating == 4.0
        assert filter_obj.product_name_pattern == "phone"


class TestSearchResult:
    """Test SearchResult dataclass."""
    
    def test_search_result_creation(self):
        """Test SearchResult creation with all fields."""
        documents = [Document(page_content="test", metadata={"price": 50.0})]
        scores = [0.8]
        embedding = [0.1] * 3072
        filters = {"price": {"$gte": 10.0}}
        metadata = {"search_type": "mmr"}
        
        result = SearchResult(
            documents=documents,
            scores=scores,
            query_embedding=embedding,
            filters_applied=filters,
            search_metadata=metadata
        )
        
        assert len(result.documents) == 1
        assert len(result.scores) == 1
        assert len(result.query_embedding) == 3072
        assert result.filters_applied == filters
        assert result.search_metadata == metadata


class TestVectorSearcher:
    """Test VectorSearcher functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SearchConfig()
    
    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return RetrievalSettings(
            openai_api_key="test-openai-key",
            pinecone_api_key="test-pinecone-key",
            pinecone_index_name="test-index",
            pinecone_namespace="test-namespace"
        )
    
    @pytest.fixture
    def vector_searcher(self, config, settings):
        """Create VectorSearcher instance."""
        return VectorSearcher(config, settings)
    
    def test_initialization(self, vector_searcher, config, settings):
        """Test VectorSearcher initialization."""
        assert vector_searcher.config == config
        assert vector_searcher.settings == settings
        assert vector_searcher.vector_store is None
        assert vector_searcher.retriever is None
        assert vector_searcher._pinecone_client is None
        assert vector_searcher._embeddings_model is None
    
    def test_from_settings_class_method(self, settings):
        """Test VectorSearcher.from_settings() class method."""
        searcher = VectorSearcher.from_settings(settings)
        
        assert searcher.config.top_k == settings.top_k
        assert searcher.config.fetch_k == settings.fetch_k
        assert searcher.config.lambda_mult == settings.lambda_mult
        assert searcher.config.score_threshold == settings.score_threshold
        assert searcher.config.search_type == settings.search_type
        assert searcher.settings == settings
    
    def test_validate_config_missing_pinecone_key(self, config):
        """Test validation fails with missing Pinecone API key."""
        settings = RetrievalSettings(
            openai_api_key="test-key",
            pinecone_api_key="",  # Missing
            pinecone_index_name="test-index"
        )
        
        with pytest.raises(ConfigurationError) as exc_info:
            VectorSearcher(config, settings)
        
        assert "Pinecone API key is required" in str(exc_info.value)
    
    def test_validate_config_missing_openai_key(self, config):
        """Test validation fails with missing OpenAI API key."""
        settings = RetrievalSettings(
            openai_api_key="",  # Missing
            pinecone_api_key="test-key",
            pinecone_index_name="test-index"
        )
        
        with pytest.raises(ConfigurationError) as exc_info:
            VectorSearcher(config, settings)
        
        assert "OpenAI API key is required" in str(exc_info.value)
    
    def test_validate_config_invalid_search_type(self, settings):
        """Test validation fails with invalid search type."""
        config = SearchConfig(search_type="invalid")
        
        with pytest.raises(ConfigurationError) as exc_info:
            VectorSearcher(config, settings)
        
        assert "Invalid search_type" in str(exc_info.value)
    
    def test_validate_config_invalid_lambda_mult(self, settings):
        """Test validation fails with invalid lambda_mult."""
        config = SearchConfig(lambda_mult=1.5)  # > 1.0
        
        with pytest.raises(ConfigurationError) as exc_info:
            VectorSearcher(config, settings)
        
        assert "lambda_mult must be between 0.0 and 1.0" in str(exc_info.value)
    
    def test_validate_config_invalid_score_threshold(self, settings):
        """Test validation fails with invalid score_threshold."""
        config = SearchConfig(score_threshold=-0.1)  # < 0.0
        
        with pytest.raises(ConfigurationError) as exc_info:
            VectorSearcher(config, settings)
        
        assert "score_threshold must be between 0.0 and 1.0" in str(exc_info.value)
    
    @patch('src.pipelines.retrieval.search.vector_searcher.Pinecone')
    @patch('src.pipelines.retrieval.search.vector_searcher.OpenAIEmbeddings')
    @patch('src.pipelines.retrieval.search.vector_searcher.PineconeVectorStore')
    def test_initialize_success_mmr(self, mock_vector_store_class, mock_embeddings_class, 
                                   mock_pinecone_class, vector_searcher):
        """Test successful initialization with MMR search."""
        # Setup mocks
        mock_pc = Mock()
        mock_pinecone_class.return_value = mock_pc
        
        # Create a proper mock index with name attribute
        mock_index = Mock()
        mock_index.name = "test-index"
        mock_pc.list_indexes.return_value = [mock_index]
        
        mock_embeddings = Mock()
        mock_embeddings.embed_query.return_value = [0.1] * 3072
        mock_embeddings_class.return_value = mock_embeddings
        
        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store
        
        mock_retriever = Mock()
        mock_vector_store.as_retriever.return_value = mock_retriever
        
        # Call initialize
        vector_searcher.initialize()
        
        # Verify initialization
        assert vector_searcher._pinecone_client == mock_pc
        assert vector_searcher._embeddings_model == mock_embeddings
        assert vector_searcher.vector_store == mock_vector_store
        assert vector_searcher.retriever == mock_retriever
        
        # Verify retriever was configured for MMR
        mock_vector_store.as_retriever.assert_called_once_with(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "fetch_k": 20,
                "lambda_mult": 0.7
            }
        )
    
    @patch('src.pipelines.retrieval.search.vector_searcher.Pinecone')
    @patch('src.pipelines.retrieval.search.vector_searcher.OpenAIEmbeddings')
    @patch('src.pipelines.retrieval.search.vector_searcher.PineconeVectorStore')
    def test_initialize_success_similarity(self, mock_vector_store_class, mock_embeddings_class, 
                                          mock_pinecone_class, settings):
        """Test successful initialization with similarity search."""
        # Create searcher with similarity search
        config = SearchConfig(search_type="similarity")
        searcher = VectorSearcher(config, settings)
        
        # Setup mocks
        mock_pc = Mock()
        mock_pinecone_class.return_value = mock_pc
        
        # Create a proper mock index with name attribute
        mock_index = Mock()
        mock_index.name = "test-index"
        mock_pc.list_indexes.return_value = [mock_index]
        
        mock_embeddings = Mock()
        mock_embeddings.embed_query.return_value = [0.1] * 3072
        mock_embeddings_class.return_value = mock_embeddings
        
        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store
        
        mock_retriever = Mock()
        mock_vector_store.as_retriever.return_value = mock_retriever
        
        # Call initialize
        searcher.initialize()
        
        # Verify retriever was configured for similarity
        mock_vector_store.as_retriever.assert_called_once_with(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
    
    @patch('src.pipelines.retrieval.search.vector_searcher.Pinecone')
    def test_initialize_connection_error(self, mock_pinecone_class, vector_searcher):
        """Test initialization fails with connection error."""
        mock_pinecone_class.side_effect = Exception("Connection failed")
        
        with pytest.raises(ConnectionError) as exc_info:
            vector_searcher.initialize()
        
        assert "Unexpected error during VectorSearcher initialization" in str(exc_info.value)
        assert exc_info.value.service == "vector_searcher"
    
    @patch('src.pipelines.retrieval.search.vector_searcher.Pinecone')
    def test_initialize_index_not_found(self, mock_pinecone_class, vector_searcher):
        """Test initialization fails when index not found."""
        mock_pc = Mock()
        mock_pinecone_class.return_value = mock_pc
        
        # Create a mock index with different name
        mock_index = Mock()
        mock_index.name = "other-index"
        mock_pc.list_indexes.return_value = [mock_index]
        
        with pytest.raises(ConnectionError) as exc_info:
            vector_searcher.initialize()
        
        assert "Index 'test-index' not found" in str(exc_info.value)
        assert exc_info.value.service == "pinecone"
    
    def test_search_not_initialized(self, vector_searcher):
        """Test search fails when not initialized."""
        query_embedding = [0.1] * 3072
        
        with pytest.raises(ConnectionError) as exc_info:
            vector_searcher.search(query_embedding)
        
        assert "VectorSearcher not initialized" in str(exc_info.value)
    
    def test_search_empty_embedding(self, vector_searcher):
        """Test search fails with empty embedding."""
        vector_searcher.vector_store = Mock()
        vector_searcher.retriever = Mock()
        
        with pytest.raises(SearchError) as exc_info:
            vector_searcher.search([])
        
        assert "Query embedding cannot be empty" in str(exc_info.value)
    
    def test_search_wrong_embedding_dimension(self, vector_searcher):
        """Test search fails with wrong embedding dimension."""
        vector_searcher.vector_store = Mock()
        vector_searcher.retriever = Mock()
        
        query_embedding = [0.1] * 1536  # Wrong dimension
        
        with pytest.raises(SearchError) as exc_info:
            vector_searcher.search(query_embedding)
        
        assert "Query embedding dimension mismatch" in str(exc_info.value)
    
    def test_build_filter_dict_empty(self, vector_searcher):
        """Test building filter dict with no filters."""
        filters = MetadataFilter()
        result = vector_searcher._build_filter_dict(filters)
        assert result == {}
    
    def test_build_filter_dict_price_range(self, vector_searcher):
        """Test building filter dict with price range."""
        filters = MetadataFilter(min_price=10.0, max_price=100.0)
        result = vector_searcher._build_filter_dict(filters)
        
        expected = {
            "price": {
                "$gte": 10.0,
                "$lte": 100.0
            }
        }
        assert result == expected
    
    def test_build_filter_dict_rating(self, vector_searcher):
        """Test building filter dict with rating filter."""
        filters = MetadataFilter(min_rating=4.0)
        result = vector_searcher._build_filter_dict(filters)
        
        expected = {
            "rating": {"$gte": 4.0}
        }
        assert result == expected
    
    def test_build_filter_dict_product_name(self, vector_searcher):
        """Test building filter dict with product name pattern."""
        filters = MetadataFilter(product_name_pattern="phone")
        result = vector_searcher._build_filter_dict(filters)
        
        expected = {
            "product_name": {
                "$regex": ".*phone.*",
                "$options": "i"
            }
        }
        assert result == expected
    
    def test_build_filter_dict_all_filters(self, vector_searcher):
        """Test building filter dict with all filters."""
        filters = MetadataFilter(
            min_price=10.0,
            max_price=100.0,
            min_rating=4.0,
            product_name_pattern="phone"
        )
        result = vector_searcher._build_filter_dict(filters)
        
        expected = {
            "price": {"$gte": 10.0, "$lte": 100.0},
            "rating": {"$gte": 4.0},
            "product_name": {"$regex": ".*phone.*", "$options": "i"}
        }
        assert result == expected
    
    def test_apply_score_threshold_empty_lists(self, vector_searcher):
        """Test score threshold filtering with empty lists."""
        docs, scores = vector_searcher._apply_score_threshold([], [])
        assert docs == []
        assert scores == []
    
    def test_apply_score_threshold_all_pass(self, vector_searcher):
        """Test score threshold filtering where all documents pass."""
        documents = [
            Document(page_content="doc1", metadata={}),
            Document(page_content="doc2", metadata={})
        ]
        scores = [0.8, 0.7]  # Both above threshold of 0.6
        
        filtered_docs, filtered_scores = vector_searcher._apply_score_threshold(documents, scores)
        
        assert len(filtered_docs) == 2
        assert len(filtered_scores) == 2
        assert filtered_scores == [0.8, 0.7]
    
    def test_apply_score_threshold_some_filtered(self, vector_searcher):
        """Test score threshold filtering where some documents are filtered."""
        documents = [
            Document(page_content="doc1", metadata={}),
            Document(page_content="doc2", metadata={}),
            Document(page_content="doc3", metadata={})
        ]
        scores = [0.8, 0.5, 0.3]  # Only first passes threshold of 0.6
        
        filtered_docs, filtered_scores = vector_searcher._apply_score_threshold(documents, scores)
        
        assert len(filtered_docs) == 1
        assert len(filtered_scores) == 1
        assert filtered_scores == [0.8]
        assert filtered_docs[0].page_content == "doc1"
    
    def test_apply_score_threshold_none_pass(self, vector_searcher):
        """Test score threshold filtering where no documents pass."""
        documents = [
            Document(page_content="doc1", metadata={}),
            Document(page_content="doc2", metadata={})
        ]
        scores = [0.5, 0.3]  # Both below threshold of 0.6
        
        filtered_docs, filtered_scores = vector_searcher._apply_score_threshold(documents, scores)
        
        assert filtered_docs == []
        assert filtered_scores == []
    
    def test_apply_score_threshold_mismatched_lengths(self, vector_searcher):
        """Test score threshold filtering with mismatched document and score lengths."""
        documents = [
            Document(page_content="doc1", metadata={}),
            Document(page_content="doc2", metadata={}),
            Document(page_content="doc3", metadata={})
        ]
        scores = [0.8, 0.7]  # Only 2 scores for 3 documents
        
        filtered_docs, filtered_scores = vector_searcher._apply_score_threshold(documents, scores)
        
        # Should truncate to shorter length and process
        assert len(filtered_docs) == 2
        assert len(filtered_scores) == 2
        assert filtered_scores == [0.8, 0.7]
    
    @patch('src.pipelines.retrieval.search.vector_searcher.Pinecone')
    def test_get_index_stats_not_initialized(self, mock_pinecone_class, vector_searcher):
        """Test get_index_stats fails when not initialized."""
        with pytest.raises(ConnectionError) as exc_info:
            vector_searcher.get_index_stats()
        
        assert "Not connected to Pinecone" in str(exc_info.value)
    
    def test_get_index_stats_success(self, vector_searcher):
        """Test successful index stats retrieval."""
        # Setup mock Pinecone client and index
        mock_pc = Mock()
        mock_index = Mock()
        mock_pc.Index.return_value = mock_index
        
        expected_stats = Mock()
        expected_stats.total_vector_count = 1000
        expected_stats.dimension = 3072
        expected_stats.index_fullness = 0.5
        expected_stats.namespaces = {"test-namespace": Mock()}
        
        mock_index.describe_index_stats.return_value = expected_stats
        vector_searcher._pinecone_client = mock_pc
        
        result = vector_searcher.get_index_stats()
        
        expected_result = {
            "total_vector_count": 1000,
            "dimension": 3072,
            "index_fullness": 0.5,
            "namespaces": {"test-namespace": expected_stats.namespaces["test-namespace"]}
        }
        
        assert result == expected_result
        mock_pc.Index.assert_called_once_with("test-index")
        mock_index.describe_index_stats.assert_called_once()


class TestVectorSearcherIntegration:
    """Integration-style tests for VectorSearcher with more realistic scenarios."""
    
    @pytest.fixture
    def initialized_searcher(self):
        """Create an initialized VectorSearcher with mocked dependencies."""
        config = SearchConfig()
        settings = RetrievalSettings(
            openai_api_key="test-openai-key",
            pinecone_api_key="test-pinecone-key",
            pinecone_index_name="test-index",
            pinecone_namespace="test-namespace"
        )
        
        searcher = VectorSearcher(config, settings)
        
        # Mock the dependencies
        searcher.vector_store = Mock()
        searcher.retriever = Mock()
        searcher._pinecone_client = Mock()
        searcher._embeddings_model = Mock()
        
        return searcher
    
    def test_mmr_search_with_filters(self, initialized_searcher):
        """Test MMR search with metadata filters."""
        # Setup test data
        query_embedding = [0.1] * 3072
        filters = MetadataFilter(min_price=10.0, max_price=100.0)
        
        # Mock documents and scores
        mock_docs = [
            Document(page_content="iPhone review", metadata={"price": 50.0, "rating": 4.5}),
            Document(page_content="Samsung review", metadata={"price": 75.0, "rating": 4.0})
        ]
        mock_scores = [0.8, 0.7]
        
        # Mock the vector store methods
        initialized_searcher.vector_store.similarity_search_with_score.return_value = [
            (mock_docs[0], mock_scores[0]),
            (mock_docs[1], mock_scores[1])
        ]
        
        initialized_searcher.vector_store.max_marginal_relevance_search_by_vector.return_value = mock_docs
        initialized_searcher.vector_store.similarity_search_by_vector_with_score.return_value = [
            (mock_docs[0], mock_scores[0]),
            (mock_docs[1], mock_scores[1])
        ]

        # Perform search
        result = initialized_searcher.search(query_embedding, filters)
        
        # Verify results
        assert isinstance(result, SearchResult)
        assert len(result.documents) == 2
        assert len(result.scores) == 2
        assert result.query_embedding == query_embedding
        assert "price" in result.filters_applied
        assert result.search_metadata["search_type"] == "mmr"
    
    def test_similarity_search_no_filters(self, initialized_searcher):
        """Test similarity search without filters."""
        # Change to similarity search
        initialized_searcher.config.search_type = "similarity"
        
        query_embedding = [0.1] * 3072
        
        # Mock documents and scores
        mock_docs = [
            Document(page_content="Product review 1", metadata={"price": 30.0}),
            Document(page_content="Product review 2", metadata={"price": 40.0})
        ]
        mock_scores = [0.9, 0.8]
        
        # Mock the vector store method
        initialized_searcher.vector_store.similarity_search_by_vector_with_score.return_value = [
            (mock_docs[0], mock_scores[0]),
            (mock_docs[1], mock_scores[1])
        ]

        # Perform search
        result = initialized_searcher.search(query_embedding)
        
        # Verify results
        assert isinstance(result, SearchResult)
        assert len(result.documents) == 2
        assert len(result.scores) == 2
        assert result.filters_applied == {}
        assert result.search_metadata["search_type"] == "similarity"