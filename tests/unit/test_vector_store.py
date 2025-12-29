"""
Unit tests for VectorStoreManager.

Tests the vector storage functionality including configuration,
initialization, and document storage operations.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src.pipelines.ingestion.storage.vector_store import VectorStoreManager, VectorStoreConfig
from src.pipelines.ingestion.exceptions import ConfigurationError, ConnectionError, IngestionError


class TestVectorStoreConfig:
    """Test VectorStoreConfig dataclass."""
    
    def test_default_config(self):
        """Test VectorStoreConfig with default values."""
        config = VectorStoreConfig(api_key="test-key")
        
        assert config.api_key == "test-key"
        assert config.index_name == "ecommerce-products"
        assert config.namespace == "phone-reviews"
        assert config.dimension == 3072
        assert config.metric == "cosine"
        assert config.cloud == "aws"
        assert config.region == "us-east-1"
    
    def test_custom_config(self):
        """Test VectorStoreConfig with custom values."""
        config = VectorStoreConfig(
            api_key="custom-key",
            index_name="custom-index",
            namespace="custom-namespace",
            dimension=1536,
            metric="euclidean",
            cloud="gcp",
            region="us-central1"
        )
        
        assert config.api_key == "custom-key"
        assert config.index_name == "custom-index"
        assert config.namespace == "custom-namespace"
        assert config.dimension == 1536
        assert config.metric == "euclidean"
        assert config.cloud == "gcp"
        assert config.region == "us-central1"


class TestVectorStoreManager:
    """Test VectorStoreManager functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return VectorStoreConfig(api_key="test-api-key")
    
    @pytest.fixture
    def mock_embeddings_model(self):
        """Create mock embeddings model."""
        mock_model = Mock(spec=OpenAIEmbeddings)
        return mock_model
    
    @pytest.fixture
    def vector_store_manager(self, config, mock_embeddings_model):
        """Create VectorStoreManager instance."""
        return VectorStoreManager(config, mock_embeddings_model)
    
    def test_initialization(self, vector_store_manager, config, mock_embeddings_model):
        """Test VectorStoreManager initialization."""
        assert vector_store_manager.config == config
        assert vector_store_manager.embeddings_model == mock_embeddings_model
        assert vector_store_manager.pc is None
        assert vector_store_manager.index is None
        assert vector_store_manager.vector_store is None
    
    def test_initialize_missing_api_key(self, mock_embeddings_model):
        """Test initialization fails with missing API key."""
        config = VectorStoreConfig(api_key="")
        manager = VectorStoreManager(config, mock_embeddings_model)
        
        with pytest.raises(ConfigurationError) as exc_info:
            manager.initialize()
        
        assert "API key is required" in str(exc_info.value)
    
    @patch('src.pipelines.ingestion.storage.vector_store.Pinecone')
    @patch('src.pipelines.ingestion.storage.vector_store.PineconeVectorStore')
    def test_initialize_success(self, mock_vector_store_class, mock_pinecone_class, 
                               vector_store_manager):
        """Test successful initialization."""
        # Setup mocks
        mock_pc = Mock()
        mock_pinecone_class.return_value = mock_pc
        
        mock_index = Mock()
        mock_pc.Index.return_value = mock_index
        mock_pc.list_indexes.return_value = [Mock(name="ecommerce-products")]
        
        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store
        
        # Call initialize
        vector_store_manager.initialize()
        
        # Verify initialization
        assert vector_store_manager.pc == mock_pc
        assert vector_store_manager.index == mock_index
        assert vector_store_manager.vector_store == mock_vector_store
        
        # Verify Pinecone client was created with correct API key
        mock_pinecone_class.assert_called_once_with(api_key="test-api-key")
    
    @patch('src.pipelines.ingestion.storage.vector_store.Pinecone')
    def test_initialize_connection_error(self, mock_pinecone_class, vector_store_manager):
        """Test initialization fails with connection error."""
        mock_pinecone_class.side_effect = Exception("Connection failed")
        
        with pytest.raises(ConnectionError) as exc_info:
            vector_store_manager.initialize()
        
        assert "Failed to connect to Pinecone" in str(exc_info.value)
        assert exc_info.value.service == "pinecone"
    
    def test_store_documents_not_initialized(self, vector_store_manager):
        """Test store_documents fails when not initialized."""
        documents = [Document(page_content="test", metadata={})]
        
        with pytest.raises(IngestionError) as exc_info:
            vector_store_manager.store_documents(documents)
        
        assert "not initialized" in str(exc_info.value)
    
    def test_store_documents_empty_list(self, vector_store_manager):
        """Test store_documents with empty document list."""
        vector_store_manager.vector_store = Mock()
        
        result = vector_store_manager.store_documents([])
        
        assert result == []
    
    @patch('src.pipelines.ingestion.storage.vector_store.uuid.uuid4')
    def test_store_documents_success(self, mock_uuid, vector_store_manager):
        """Test successful document storage."""
        # Setup mocks
        mock_vector_store = Mock()
        vector_store_manager.vector_store = mock_vector_store
        
        mock_uuid.side_effect = [
            Mock(__str__=lambda x: "id-1"),
            Mock(__str__=lambda x: "id-2")
        ]
        
        mock_vector_store.add_documents.return_value = ["id-1", "id-2"]
        
        # Create test documents
        documents = [
            Document(page_content="content 1", metadata={"key": "value1"}),
            Document(page_content="content 2", metadata={"key": "value2"})
        ]
        
        # Call store_documents
        result = vector_store_manager.store_documents(documents)
        
        # Verify results
        assert result == ["id-1", "id-2"]
        
        # Verify add_documents was called correctly
        mock_vector_store.add_documents.assert_called_once()
        call_args = mock_vector_store.add_documents.call_args
        
        assert call_args[1]["ids"] == ["id-1", "id-2"]
        assert call_args[1]["namespace"] == "phone-reviews"
        assert len(call_args[1]["documents"]) == 2
    
    def test_prepare_metadata(self, vector_store_manager):
        """Test metadata preparation for Pinecone storage."""
        metadata = {
            "string_field": "test",
            "int_field": 42,
            "float_field": 3.14,
            "bool_field": True,
            "none_field": None,
            "list_field": [1, 2, 3]
        }
        
        result = vector_store_manager._prepare_metadata(metadata)
        
        expected = {
            "string_field": "test",
            "int_field": "42",
            "float_field": "3.14",
            "bool_field": "True",
            "none_field": "N/A",
            "list_field": "[1, 2, 3]"
        }
        
        assert result == expected
    
    def test_get_index_stats_not_initialized(self, vector_store_manager):
        """Test get_index_stats fails when not initialized."""
        with pytest.raises(IngestionError) as exc_info:
            vector_store_manager.get_index_stats()
        
        assert "not initialized" in str(exc_info.value)
    
    def test_get_index_stats_success(self, vector_store_manager):
        """Test successful index stats retrieval."""
        mock_index = Mock()
        vector_store_manager.index = mock_index
        
        expected_stats = {"total_vector_count": 100}
        mock_index.describe_index_stats.return_value = expected_stats
        
        result = vector_store_manager.get_index_stats()
        
        assert result == expected_stats
        mock_index.describe_index_stats.assert_called_once()