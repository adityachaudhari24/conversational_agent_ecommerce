"""
Unit tests for EmbeddingGenerator component.

Tests the core functionality of generating embeddings from Document objects,
including initialization, batch processing, and error handling.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from src.pipelines.ingestion.processors.embedding_generator import EmbeddingGenerator, EmbeddingConfig
from src.pipelines.ingestion.exceptions import ConfigurationError, IngestionError


class TestEmbeddingGenerator:
    """Test suite for EmbeddingGenerator class."""
    
    def test_initialization_with_valid_config(self):
        """Test EmbeddingGenerator initialization with valid configuration."""
        config = EmbeddingConfig(
            model_name="text-embedding-3-large",
            api_key="test-key",
            batch_size=50,
            expected_dimension=3072
        )
        
        generator = EmbeddingGenerator(config)
        
        assert generator.config == config
        assert generator.embeddings_model is None
        assert generator.failed_documents == []
    
    def test_initialization_without_api_key(self):
        """Test that initialization fails when API key is missing."""
        config = EmbeddingConfig(api_key=None)
        generator = EmbeddingGenerator(config)
        
        with pytest.raises(ConfigurationError, match="OpenAI API key is required"):
            generator.initialize()
    
    @patch('src.pipelines.ingestion.processors.embedding_generator.OpenAIEmbeddings')
    def test_successful_initialization(self, mock_openai_embeddings):
        """Test successful initialization with valid API key."""
        # Mock the OpenAI embeddings
        mock_embeddings = Mock()
        mock_embeddings.embed_query.return_value = [0.1] * 3072  # Correct dimension
        mock_openai_embeddings.return_value = mock_embeddings
        
        config = EmbeddingConfig(api_key="test-key")
        generator = EmbeddingGenerator(config)
        
        # Should not raise an exception
        generator.initialize()
        
        assert generator.embeddings_model is not None
        mock_openai_embeddings.assert_called_once_with(
            model="text-embedding-3-large",
            openai_api_key="test-key"
        )
    
    @patch('src.pipelines.ingestion.processors.embedding_generator.OpenAIEmbeddings')
    def test_initialization_with_wrong_dimension(self, mock_openai_embeddings):
        """Test initialization fails when embedding dimension is wrong."""
        # Mock the OpenAI embeddings with wrong dimension
        mock_embeddings = Mock()
        mock_embeddings.embed_query.return_value = [0.1] * 1536  # Wrong dimension
        mock_openai_embeddings.return_value = mock_embeddings
        
        config = EmbeddingConfig(api_key="test-key", expected_dimension=3072)
        generator = EmbeddingGenerator(config)
        
        with pytest.raises(ConfigurationError, match="Embedding dimension mismatch"):
            generator.initialize()
    
    def test_generate_without_initialization(self):
        """Test that generate fails when model is not initialized."""
        config = EmbeddingConfig(api_key="test-key")
        generator = EmbeddingGenerator(config)
        
        documents = [Document(page_content="test", metadata={})]
        
        with pytest.raises(IngestionError, match="Embedding model not initialized"):
            generator.generate(documents)
    
    @patch('src.pipelines.ingestion.processors.embedding_generator.OpenAIEmbeddings')
    def test_successful_embedding_generation(self, mock_openai_embeddings):
        """Test successful embedding generation for documents."""
        # Mock the OpenAI embeddings
        mock_embeddings = Mock()
        mock_embeddings.embed_query.return_value = [0.1] * 3072
        mock_embeddings.embed_documents.return_value = [
            [0.1] * 3072,  # First document embedding
            [0.2] * 3072   # Second document embedding
        ]
        mock_openai_embeddings.return_value = mock_embeddings
        
        config = EmbeddingConfig(api_key="test-key")
        generator = EmbeddingGenerator(config)
        generator.initialize()
        
        # Create test documents
        documents = [
            Document(page_content="First document", metadata={"id": 1}),
            Document(page_content="Second document", metadata={"id": 2})
        ]
        
        # Generate embeddings
        results = generator.generate(documents)
        
        # Verify results
        assert len(results) == 2
        assert len(generator.failed_documents) == 0
        
        # Check first result
        doc1, embedding1 = results[0]
        assert doc1.page_content == "First document"
        assert len(embedding1) == 3072
        assert embedding1 == [0.1] * 3072
        
        # Check second result
        doc2, embedding2 = results[1]
        assert doc2.page_content == "Second document"
        assert len(embedding2) == 3072
        assert embedding2 == [0.2] * 3072
    
    @patch('src.pipelines.ingestion.processors.embedding_generator.OpenAIEmbeddings')
    def test_batch_processing(self, mock_openai_embeddings):
        """Test that documents are processed in batches."""
        # Mock the OpenAI embeddings
        mock_embeddings = Mock()
        mock_embeddings.embed_query.return_value = [0.1] * 3072
        mock_embeddings.embed_documents.side_effect = [
            [[0.1] * 3072, [0.2] * 3072],  # First batch
            [[0.3] * 3072]                  # Second batch
        ]
        mock_openai_embeddings.return_value = mock_embeddings
        
        config = EmbeddingConfig(api_key="test-key", batch_size=2)
        generator = EmbeddingGenerator(config)
        generator.initialize()
        
        # Create test documents (3 documents, batch size 2)
        documents = [
            Document(page_content="Doc 1", metadata={}),
            Document(page_content="Doc 2", metadata={}),
            Document(page_content="Doc 3", metadata={})
        ]
        
        # Generate embeddings
        results = generator.generate(documents)
        
        # Verify batch processing
        assert len(results) == 3
        assert mock_embeddings.embed_documents.call_count == 2
        
        # Check that batches were called with correct sizes
        call_args = mock_embeddings.embed_documents.call_args_list
        assert len(call_args[0][0][0]) == 2  # First batch: 2 documents
        assert len(call_args[1][0][0]) == 1  # Second batch: 1 document
    
    @patch('src.pipelines.ingestion.processors.embedding_generator.OpenAIEmbeddings')
    def test_embedding_validation_failure(self, mock_openai_embeddings):
        """Test handling of embeddings with wrong dimensions."""
        # Mock the OpenAI embeddings
        mock_embeddings = Mock()
        mock_embeddings.embed_query.return_value = [0.1] * 3072
        mock_embeddings.embed_documents.return_value = [
            [0.1] * 1536,  # Wrong dimension
            [0.2] * 3072   # Correct dimension
        ]
        mock_openai_embeddings.return_value = mock_embeddings
        
        config = EmbeddingConfig(api_key="test-key")
        generator = EmbeddingGenerator(config)
        generator.initialize()
        
        # Create test documents
        documents = [
            Document(page_content="Doc with wrong embedding", metadata={}),
            Document(page_content="Doc with correct embedding", metadata={})
        ]
        
        # Generate embeddings
        results = generator.generate(documents)
        
        # Verify that only valid embedding is returned
        assert len(results) == 1
        assert len(generator.failed_documents) == 1
        
        # Check that the correct document succeeded
        doc, embedding = results[0]
        assert doc.page_content == "Doc with correct embedding"
        assert len(embedding) == 3072
    
    @patch('src.pipelines.ingestion.processors.embedding_generator.OpenAIEmbeddings')
    def test_api_error_handling(self, mock_openai_embeddings):
        """Test handling of API errors during embedding generation."""
        # Mock the OpenAI embeddings
        mock_embeddings = Mock()
        mock_embeddings.embed_query.return_value = [0.1] * 3072
        mock_embeddings.embed_documents.side_effect = Exception("API Error")
        mock_openai_embeddings.return_value = mock_embeddings
        
        config = EmbeddingConfig(api_key="test-key")
        generator = EmbeddingGenerator(config)
        generator.initialize()
        
        # Create test documents
        documents = [
            Document(page_content="Test document", metadata={})
        ]
        
        # Generate embeddings (should handle error gracefully)
        results = generator.generate(documents)
        
        # Verify error handling
        assert len(results) == 0
        assert len(generator.failed_documents) == 1
        assert generator.failed_documents[0].page_content == "Test document"
    
    def test_validate_embedding(self):
        """Test embedding validation logic."""
        config = EmbeddingConfig(expected_dimension=3072)
        generator = EmbeddingGenerator(config)
        
        # Test valid embedding
        valid_embedding = [0.1] * 3072
        assert generator._validate_embedding(valid_embedding) is True
        
        # Test invalid embedding (wrong dimension)
        invalid_embedding = [0.1] * 1536
        assert generator._validate_embedding(invalid_embedding) is False
        
        # Test empty embedding
        empty_embedding = []
        assert generator._validate_embedding(empty_embedding) is False
        
        # Test None embedding
        assert generator._validate_embedding(None) is False
    
    def test_get_failed_documents(self):
        """Test getting failed documents list."""
        config = EmbeddingConfig()
        generator = EmbeddingGenerator(config)
        
        # Initially should be empty
        assert generator.get_failed_documents() == []
        assert generator.get_failure_count() == 0
        
        # Add some failed documents
        failed_doc = Document(page_content="Failed", metadata={})
        generator.failed_documents.append(failed_doc)
        
        # Verify getter methods
        assert len(generator.get_failed_documents()) == 1
        assert generator.get_failure_count() == 1
        assert generator.get_failed_documents()[0].page_content == "Failed"


if __name__ == "__main__":
    pytest.main([__file__])