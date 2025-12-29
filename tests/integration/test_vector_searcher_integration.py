"""
Integration tests for VectorSearcher.

These tests demonstrate how the VectorSearcher works with more realistic
scenarios and show the complete workflow from initialization to search.
"""

import pytest
from unittest.mock import Mock, patch
from langchain_core.documents import Document

from src.pipelines.retrieval.search import VectorSearcher, SearchConfig, MetadataFilter
from src.pipelines.retrieval.config import RetrievalSettings


class TestVectorSearcherIntegration:
    """Integration tests for VectorSearcher with realistic scenarios."""
    
    @pytest.fixture
    def settings(self):
        """Create realistic test settings."""
        return RetrievalSettings(
            openai_api_key="test-openai-key",
            pinecone_api_key="test-pinecone-key",
            pinecone_index_name="ecommerce-products",
            pinecone_namespace="phone-reviews",
            top_k=4,
            fetch_k=20,
            lambda_mult=0.7,
            score_threshold=0.6
        )
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample product review documents."""
        return [
            Document(
                page_content="Great iPhone with excellent camera quality and long battery life.",
                metadata={
                    "product_name": "iPhone 15 Pro",
                    "price": 999.0,
                    "rating": 4.8,
                    "review_title": "Excellent phone"
                }
            ),
            Document(
                page_content="Samsung Galaxy has amazing display and good performance for the price.",
                metadata={
                    "product_name": "Samsung Galaxy S24",
                    "price": 799.0,
                    "rating": 4.5,
                    "review_title": "Great value"
                }
            ),
            Document(
                page_content="Google Pixel offers pure Android experience with fantastic camera.",
                metadata={
                    "product_name": "Google Pixel 8",
                    "price": 699.0,
                    "rating": 4.6,
                    "review_title": "Pure Android"
                }
            ),
            Document(
                page_content="OnePlus phone delivers flagship performance at mid-range price.",
                metadata={
                    "product_name": "OnePlus 12",
                    "price": 599.0,
                    "rating": 4.3,
                    "review_title": "Great performance"
                }
            )
        ]
    
    @patch('src.pipelines.retrieval.search.vector_searcher.Pinecone')
    @patch('src.pipelines.retrieval.search.vector_searcher.OpenAIEmbeddings')
    @patch('src.pipelines.retrieval.search.vector_searcher.PineconeVectorStore')
    def test_complete_mmr_search_workflow(self, mock_vector_store_class, mock_embeddings_class, 
                                         mock_pinecone_class, settings, sample_documents):
        """Test complete MMR search workflow from initialization to results."""
        # Create VectorSearcher
        searcher = VectorSearcher.from_settings(settings)
        
        # Setup mocks for initialization
        mock_pc = Mock()
        mock_pinecone_class.return_value = mock_pc
        
        mock_index = Mock()
        mock_index.name = "ecommerce-products"
        mock_pc.list_indexes.return_value = [mock_index]
        
        mock_embeddings = Mock()
        mock_embeddings.embed_query.return_value = [0.1] * 3072
        mock_embeddings_class.return_value = mock_embeddings
        
        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store
        
        mock_retriever = Mock()
        mock_vector_store.as_retriever.return_value = mock_retriever
        
        # Initialize the searcher
        searcher.initialize()
        
        # Verify initialization
        assert searcher.vector_store is not None
        assert searcher.retriever is not None
        assert searcher.config.search_type == "mmr"
        
        # Setup search mocks
        query_embedding = [0.2] * 3072
        
        # Mock MMR search results - simulate fetching more documents then selecting diverse ones
        all_docs_with_scores = [(doc, 0.8 - i * 0.1) for i, doc in enumerate(sample_documents)]
        
        mock_vector_store.similarity_search_with_score.return_value = all_docs_with_scores
        mock_vector_store.max_marginal_relevance_search_by_vector.return_value = [
            sample_documents[0],  # iPhone (highest relevance)
            sample_documents[1],  # Samsung (different brand)
            sample_documents[2]   # Google (another different brand)
        ]
        mock_vector_store.similarity_search_with_score_by_vector.return_value = [
            (sample_documents[0], 0.8),
            (sample_documents[1], 0.7),
            (sample_documents[2], 0.6)
        ]
        
        # Perform search
        result = searcher.search(query_embedding)
        
        # Verify search results
        assert len(result.documents) == 3  # 3 documents passed score threshold
        assert len(result.scores) == 3
        assert result.query_embedding == query_embedding
        assert result.search_metadata["search_type"] == "mmr"
        assert result.search_metadata["original_count"] == 3
        assert result.search_metadata["filtered_count"] == 3
        
        # Verify MMR was called with correct parameters
        mock_vector_store.max_marginal_relevance_search_by_vector.assert_called_once_with(
            embedding=query_embedding,
            k=4,  # top_k
            fetch_k=4,  # min of len(docs) and fetch_k
            lambda_mult=0.7,
            filter={}
        )
    
    @patch('src.pipelines.retrieval.search.vector_searcher.Pinecone')
    @patch('src.pipelines.retrieval.search.vector_searcher.OpenAIEmbeddings')
    @patch('src.pipelines.retrieval.search.vector_searcher.PineconeVectorStore')
    def test_search_with_price_filter(self, mock_vector_store_class, mock_embeddings_class, 
                                     mock_pinecone_class, settings, sample_documents):
        """Test search with price range filter."""
        # Create VectorSearcher
        searcher = VectorSearcher.from_settings(settings)
        
        # Setup mocks for initialization
        mock_pc = Mock()
        mock_pinecone_class.return_value = mock_pc
        
        mock_index = Mock()
        mock_index.name = "ecommerce-products"
        mock_pc.list_indexes.return_value = [mock_index]
        
        mock_embeddings = Mock()
        mock_embeddings.embed_query.return_value = [0.1] * 3072
        mock_embeddings_class.return_value = mock_embeddings
        
        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store
        
        # Initialize
        searcher.initialize()
        
        # Create price filter (only phones under $800)
        filters = MetadataFilter(max_price=800.0, min_rating=4.0)
        query_embedding = [0.3] * 3072
        
        # Mock filtered results - only Samsung, Pixel, and OnePlus should match
        filtered_docs = [sample_documents[1], sample_documents[2], sample_documents[3]]  # Exclude iPhone
        filtered_scores = [0.7, 0.6, 0.65]
        
        mock_vector_store.similarity_search_with_score.return_value = [
            (doc, score) for doc, score in zip(filtered_docs, filtered_scores)
        ]
        mock_vector_store.max_marginal_relevance_search_by_vector.return_value = filtered_docs
        mock_vector_store.similarity_search_with_score_by_vector.return_value = [
            (doc, score) for doc, score in zip(filtered_docs, filtered_scores)
        ]
        
        # Perform search with filters
        result = searcher.search(query_embedding, filters)
        
        # Verify results
        assert len(result.documents) == 3
        assert len(result.scores) == 3
        
        # Verify filter was applied
        expected_filter = {
            "price": {"$lte": 800.0},
            "rating": {"$gte": 4.0}
        }
        assert result.filters_applied == expected_filter
        
        # Verify all returned products are under $800
        for doc in result.documents:
            assert doc.metadata["price"] <= 800.0
            assert doc.metadata["rating"] >= 4.0
    
    @patch('src.pipelines.retrieval.search.vector_searcher.Pinecone')
    @patch('src.pipelines.retrieval.search.vector_searcher.OpenAIEmbeddings')
    @patch('src.pipelines.retrieval.search.vector_searcher.PineconeVectorStore')
    def test_similarity_search_workflow(self, mock_vector_store_class, mock_embeddings_class, 
                                       mock_pinecone_class, settings, sample_documents):
        """Test similarity search workflow (non-MMR)."""
        # Create VectorSearcher with similarity search
        config = SearchConfig(search_type="similarity")
        searcher = VectorSearcher(config, settings)
        
        # Setup mocks for initialization
        mock_pc = Mock()
        mock_pinecone_class.return_value = mock_pc
        
        mock_index = Mock()
        mock_index.name = "ecommerce-products"
        mock_pc.list_indexes.return_value = [mock_index]
        
        mock_embeddings = Mock()
        mock_embeddings.embed_query.return_value = [0.1] * 3072
        mock_embeddings_class.return_value = mock_embeddings
        
        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store
        
        # Initialize
        searcher.initialize()
        
        # Setup search
        query_embedding = [0.4] * 3072
        
        # Mock similarity search results (just top similar documents)
        mock_vector_store.similarity_search_with_score_by_vector.return_value = [
            (sample_documents[0], 0.9),  # iPhone - highest similarity
            (sample_documents[1], 0.8),  # Samsung
            (sample_documents[2], 0.7),  # Pixel
            (sample_documents[3], 0.5)   # OnePlus - below threshold
        ]
        
        # Perform search
        result = searcher.search(query_embedding)
        
        # Verify results (only 3 should pass score threshold of 0.6)
        assert len(result.documents) == 3
        assert len(result.scores) == 3
        assert result.search_metadata["search_type"] == "similarity"
        assert all(score >= 0.6 for score in result.scores)
        
        # Verify similarity search was used (not MMR)
        mock_vector_store.similarity_search_with_score_by_vector.assert_called_once_with(
            embedding=query_embedding,
            k=4,
            filter={}
        )
        
        # Verify MMR was NOT called
        mock_vector_store.max_marginal_relevance_search_by_vector.assert_not_called()
    
    def test_search_error_scenarios(self, settings):
        """Test various error scenarios during search."""
        searcher = VectorSearcher.from_settings(settings)
        
        # Test search without initialization
        with pytest.raises(Exception) as exc_info:
            searcher.search([0.1] * 3072)
        assert "not initialized" in str(exc_info.value)
        
        # Mock initialization
        searcher.vector_store = Mock()
        searcher.retriever = Mock()
        
        # Test empty embedding
        with pytest.raises(Exception) as exc_info:
            searcher.search([])
        assert "cannot be empty" in str(exc_info.value)
        
        # Test wrong embedding dimension
        with pytest.raises(Exception) as exc_info:
            searcher.search([0.1] * 1536)  # Wrong dimension
        assert "dimension mismatch" in str(exc_info.value)
    
    def test_comprehensive_filter_building(self, settings):
        """Test comprehensive filter building with all filter types."""
        searcher = VectorSearcher.from_settings(settings)
        
        # Test all filter types
        filters = MetadataFilter(
            min_price=100.0,
            max_price=1000.0,
            min_rating=4.0,
            product_name_pattern="iPhone"
        )
        
        filter_dict = searcher._build_filter_dict(filters)
        
        expected = {
            "price": {"$gte": 100.0, "$lte": 1000.0},
            "rating": {"$gte": 4.0},
            "product_name": {"$regex": ".*iPhone.*", "$options": "i"}
        }
        
        assert filter_dict == expected
    
    def test_score_threshold_scenarios(self, settings, sample_documents):
        """Test various score threshold filtering scenarios."""
        searcher = VectorSearcher.from_settings(settings)
        
        # Test all documents pass threshold
        docs = sample_documents[:2]
        scores = [0.8, 0.7]  # Both above 0.6
        
        filtered_docs, filtered_scores = searcher._apply_score_threshold(docs, scores)
        assert len(filtered_docs) == 2
        assert filtered_scores == [0.8, 0.7]
        
        # Test some documents filtered
        scores = [0.8, 0.5]  # One below threshold
        filtered_docs, filtered_scores = searcher._apply_score_threshold(docs, scores)
        assert len(filtered_docs) == 1
        assert filtered_scores == [0.8]
        
        # Test no documents pass threshold
        scores = [0.5, 0.4]  # Both below threshold
        filtered_docs, filtered_scores = searcher._apply_score_threshold(docs, scores)
        assert len(filtered_docs) == 0
        assert filtered_scores == []


def test_vector_searcher_demo():
    """
    Demo test showing how to use VectorSearcher in practice.
    
    This test demonstrates the typical workflow:
    1. Create settings
    2. Initialize searcher
    3. Perform searches with different configurations
    """
    # Step 1: Create settings (in real usage, these come from environment/config)
    settings = RetrievalSettings(
        openai_api_key="your-openai-key",
        pinecone_api_key="your-pinecone-key",
        pinecone_index_name="ecommerce-products",
        pinecone_namespace="phone-reviews"
    )
    
    # Step 2: Create and configure searcher
    searcher = VectorSearcher.from_settings(settings)
    
    # Verify configuration
    assert searcher.config.search_type == "mmr"  # Default is MMR for diversity
    assert searcher.config.top_k == 4           # Return 4 results
    assert searcher.config.fetch_k == 20        # Consider 20 candidates for MMR
    assert searcher.config.lambda_mult == 0.7   # 70% relevance, 30% diversity
    assert searcher.config.score_threshold == 0.6  # Filter low-quality results
    
    # Step 3: In real usage, you would call searcher.initialize() here
    # This connects to Pinecone and sets up the vector store
    
    # Step 4: Create search filters (optional)
    price_filter = MetadataFilter(
        min_price=200.0,
        max_price=800.0,
        min_rating=4.0
    )
    
    # Step 5: In real usage, you would perform search like this:
    # query_embedding = your_embedding_model.embed_query("best smartphone camera")
    # result = searcher.search(query_embedding, price_filter)
    
    # The result would contain:
    # - result.documents: List of relevant product documents
    # - result.scores: Similarity scores for each document
    # - result.filters_applied: The filters that were applied
    # - result.search_metadata: Information about the search process
    
    print("VectorSearcher demo completed successfully!")
    print(f"Configured for {searcher.config.search_type} search")
    print(f"Will return up to {searcher.config.top_k} results")
    print(f"Score threshold: {searcher.config.score_threshold}")


if __name__ == "__main__":
    test_vector_searcher_demo()