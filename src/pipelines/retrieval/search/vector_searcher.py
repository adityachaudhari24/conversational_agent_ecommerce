"""
Vector Searcher for the retrieval pipeline.

This module handles vector similarity search against Pinecone with support for
MMR (Maximal Marginal Relevance) search, metadata filtering, and score thresholding.
It provides the core search functionality for the retrieval pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

from ..config import RetrievalSettings
from ..exceptions import SearchError, ConnectionError, ConfigurationError
from ..logging import RetrievalLoggerMixin, log_retrieval_operation


@dataclass
class SearchConfig:
    """Configuration for vector search operations."""
    
    top_k: int = 4
    fetch_k: int = 20
    lambda_mult: float = 0.7
    score_threshold: float = 0.6
    search_type: str = "mmr"  # "similarity" or "mmr"


@dataclass
class MetadataFilter:
    """Metadata filters for search operations."""
    
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_rating: Optional[float] = None
    product_name_pattern: Optional[str] = None


@dataclass
class SearchResult:
    """Result of a vector search operation."""
    
    documents: List[Document]
    scores: List[float]
    query_embedding: List[float]
    filters_applied: Dict[str, Any] = field(default_factory=dict)
    search_metadata: Dict[str, Any] = field(default_factory=dict)


class VectorSearcher(RetrievalLoggerMixin):
    """
    Performs vector similarity search against Pinecone.
    
    The VectorSearcher is responsible for:
    - Connecting to Pinecone vector database
    - Performing similarity and MMR searches
    - Applying metadata filters to search results
    - Filtering results by similarity score threshold
    - Logging search operations and metrics
    """
    
    def __init__(self, config: SearchConfig, settings: RetrievalSettings):
        """
        Initialize the VectorSearcher with configuration.
        
        Args:
            config: SearchConfig containing search settings
            settings: RetrievalSettings containing API keys and connection info
            
        Raises:
            ConfigurationError: If required settings are missing or invalid
        """
        super().__init__()
        self.config = config
        self.settings = settings
        self.vector_store: Optional[PineconeVectorStore] = None
        self.retriever = None
        self._pinecone_client: Optional[Pinecone] = None
        self._embeddings_model: Optional[OpenAIEmbeddings] = None
        
        # Validate configuration
        self._validate_config()
    
    @classmethod
    def from_settings(cls, settings: RetrievalSettings) -> 'VectorSearcher':
        """
        Create VectorSearcher from RetrievalSettings.
        
        Args:
            settings: RetrievalSettings instance
            
        Returns:
            VectorSearcher instance
        """
        config = SearchConfig(
            top_k=settings.top_k,
            fetch_k=settings.fetch_k,
            lambda_mult=settings.lambda_mult,
            score_threshold=settings.score_threshold,
            search_type=settings.search_type
        )
        
        return cls(config, settings)
    
    def _validate_config(self) -> None:
        """
        Validate configuration settings.
        
        Raises:
            ConfigurationError: If configuration is invalid
        """
        if not self.settings.pinecone_api_key:
            raise ConfigurationError(
                "Pinecone API key is required for vector search. "
                "Please set PINECONE_API_KEY environment variable."
            )
        
        if not self.settings.openai_api_key:
            raise ConfigurationError(
                "OpenAI API key is required for embeddings. "
                "Please set OPENAI_API_KEY environment variable."
            )
        
        if not self.settings.pinecone_index_name:
            raise ConfigurationError("Pinecone index name is required")
        
        if self.config.search_type not in ["similarity", "mmr"]:
            raise ConfigurationError(
                f"Invalid search_type: {self.config.search_type}. "
                "Must be 'similarity' or 'mmr'"
            )
        
        if not (0.0 <= self.config.lambda_mult <= 1.0):
            raise ConfigurationError(
                f"lambda_mult must be between 0.0 and 1.0, got {self.config.lambda_mult}"
            )
        
        if not (0.0 <= self.config.score_threshold <= 1.0):
            raise ConfigurationError(
                f"score_threshold must be between 0.0 and 1.0, got {self.config.score_threshold}"
            )
    
    @log_retrieval_operation("vector_searcher_initialization")
    def initialize(self) -> None:
        """
        Initialize the vector store and retriever with configured settings.
        
        Raises:
            ConnectionError: If connection to Pinecone fails
            ConfigurationError: If initialization fails due to configuration issues
        """
        try:
            # Initialize Pinecone client
            self._pinecone_client = Pinecone(api_key=self.settings.pinecone_api_key)
            
            # Test connection by listing indexes
            try:
                indexes = self._pinecone_client.list_indexes()
                index_names = [idx.name for idx in indexes]
                
                if self.settings.pinecone_index_name not in index_names:
                    raise ConnectionError(
                        f"Index '{self.settings.pinecone_index_name}' not found. "
                        f"Available indexes: {index_names}",
                        service="pinecone",
                        endpoint=f"index/{self.settings.pinecone_index_name}"
                    )
                
                self.logger.info(
                    f"Successfully connected to Pinecone index: {self.settings.pinecone_index_name}"
                )
                
            except Exception as e:
                raise ConnectionError(
                    f"Failed to connect to Pinecone: {str(e)}",
                    service="pinecone"
                ) from e
            
            # Initialize embeddings model
            self._embeddings_model = OpenAIEmbeddings(
                model=self.settings.embedding_model,
                openai_api_key=self.settings.openai_api_key
            )
            
            # Test embeddings model
            try:
                test_embedding = self._embeddings_model.embed_query("test")
                if len(test_embedding) != self.settings.embedding_dimension:
                    raise ConfigurationError(
                        f"Embedding dimension mismatch. Expected {self.settings.embedding_dimension}, "
                        f"got {len(test_embedding)} for model {self.settings.embedding_model}"
                    )
                
                self.logger.info(
                    f"Successfully initialized embeddings model: {self.settings.embedding_model}"
                )
                
            except Exception as e:
                raise ConfigurationError(
                    f"Failed to initialize embeddings model: {str(e)}"
                ) from e
            
            # Initialize vector store
            self.vector_store = PineconeVectorStore(
                index_name=self.settings.pinecone_index_name,
                embedding=self._embeddings_model,
                namespace=self.settings.pinecone_namespace,
                pinecone_api_key=self.settings.pinecone_api_key
            )
            
            # Initialize retriever based on search type
            if self.config.search_type == "mmr":
                self.retriever = self.vector_store.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": self.config.top_k,
                        "fetch_k": self.config.fetch_k,
                        "lambda_mult": self.config.lambda_mult
                    }
                )
            else:  # similarity
                self.retriever = self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={
                        "k": self.config.top_k
                    }
                )
            
            self.logger.info(
                f"Successfully initialized VectorSearcher with {self.config.search_type} search",
                extra={
                    'extra_fields': {
                        'search_type': self.config.search_type,
                        'top_k': self.config.top_k,
                        'fetch_k': self.config.fetch_k,
                        'lambda_mult': self.config.lambda_mult,
                        'score_threshold': self.config.score_threshold,
                        'index_name': self.settings.pinecone_index_name,
                        'namespace': self.settings.pinecone_namespace
                    }
                }
            )
            
        except (ConnectionError, ConfigurationError):
            # Re-raise these as-is
            raise
        except Exception as e:
            raise ConnectionError(
                f"Unexpected error during VectorSearcher initialization: {str(e)}",
                service="vector_searcher"
            ) from e
    
    @log_retrieval_operation("vector_search")
    def search(
        self, 
        query_embedding: List[float],
        filters: Optional[MetadataFilter] = None
    ) -> SearchResult:
        """
        Perform similarity search with optional filters.
        
        Args:
            query_embedding: Query embedding vector
            filters: Optional metadata filters to apply
            
        Returns:
            SearchResult containing documents, scores, and metadata
            
        Raises:
            SearchError: If search operation fails
            ConnectionError: If vector store is not initialized
        """
        if self.vector_store is None or self.retriever is None:
            raise ConnectionError(
                "VectorSearcher not initialized. Call initialize() first.",
                service="vector_searcher"
            )
        
        if not query_embedding:
            raise SearchError(
                "Query embedding cannot be empty",
                search_params={"embedding_length": len(query_embedding)}
            )
        
        if len(query_embedding) != self.settings.embedding_dimension:
            raise SearchError(
                f"Query embedding dimension mismatch. Expected {self.settings.embedding_dimension}, "
                f"got {len(query_embedding)}",
                search_params={"embedding_dimension": len(query_embedding)}
            )
        
        try:
            # Build filter dictionary if filters are provided
            # Note: product_name_pattern is excluded from Pinecone filters
            # and applied post-search via _apply_product_name_filter()
            filter_dict = self._build_filter_dict(filters) if filters else None
            
            # If we have a product name pattern, fetch more results to compensate
            # for post-search filtering that will reduce the result set
            product_name_pattern = filters.product_name_pattern if filters else None
            effective_top_k = self.config.top_k
            effective_fetch_k = self.config.fetch_k
            if product_name_pattern:
                effective_top_k = self.config.top_k * 5
                effective_fetch_k = self.config.fetch_k * 3
            
            # Perform search based on type
            if self.config.search_type == "mmr":
                # For MMR search, we need to use similarity_search_with_score
                # and then apply MMR manually if needed, or use the retriever
                if filter_dict:
                    # Use similarity_search_by_vector_with_score directly for filtering support
                    docs_and_scores = self.vector_store.similarity_search_by_vector_with_score(
                        embedding=query_embedding,
                        k=effective_fetch_k,
                        filter=filter_dict
                    )
                    
                    # Apply MMR to the fetched results
                    if len(docs_and_scores) > effective_top_k:
                        # Extract documents and embeddings for MMR
                        docs = [doc for doc, _ in docs_and_scores]
                        
                        # Use vector store's MMR functionality
                        mmr_docs = self.vector_store.max_marginal_relevance_search_by_vector(
                            embedding=query_embedding,
                            k=effective_top_k,
                            fetch_k=min(len(docs), effective_fetch_k),
                            lambda_mult=self.config.lambda_mult,
                            filter=filter_dict
                        )
                        
                        # Get scores for MMR documents
                        documents = mmr_docs
                        scores = []
                        for doc in mmr_docs:
                            # Find the score for this document
                            for orig_doc, score in docs_and_scores:
                                if (orig_doc.page_content == doc.page_content and 
                                    orig_doc.metadata == doc.metadata):
                                    scores.append(score)
                                    break
                            else:
                                # If we can't find the exact score, estimate it
                                scores.append(0.7)  # Default reasonable score
                    else:
                        documents = [doc for doc, _ in docs_and_scores]
                        scores = [score for _, score in docs_and_scores]
                else:
                    # Use MMR search without filters
                    documents = self.vector_store.max_marginal_relevance_search_by_vector(
                        embedding=query_embedding,
                        k=effective_top_k,
                        fetch_k=effective_fetch_k,
                        lambda_mult=self.config.lambda_mult
                    )
                    
                    # Get scores for the documents
                    # Use similarity_search_by_vector_with_score directly with the embedding
                    docs_and_scores = self.vector_store.similarity_search_by_vector_with_score(
                        embedding=query_embedding,
                        k=len(documents)
                    )
                    
                    # Match documents to scores
                    scores = []
                    for doc in documents:
                        for orig_doc, score in docs_and_scores:
                            if (orig_doc.page_content == doc.page_content and 
                                orig_doc.metadata == doc.metadata):
                                scores.append(score)
                                break
                        else:
                            scores.append(0.7)  # Default score if not found
            
            else:  # similarity search
                docs_and_scores = self.vector_store.similarity_search_by_vector_with_score(
                    embedding=query_embedding,
                    k=effective_top_k,
                    filter=filter_dict
                )
                
                documents = [doc for doc, _ in docs_and_scores]
                scores = [score for _, score in docs_and_scores]
            
            # Apply score threshold filtering
            filtered_documents, filtered_scores = self._apply_score_threshold(documents, scores)
            
            # Apply product name filter post-search (Pinecone doesn't support substring matching)
            if product_name_pattern:
                filtered_documents, filtered_scores = self._apply_product_name_filter(
                    filtered_documents, filtered_scores, product_name_pattern
                )
            
            # Trim back to original top_k after post-filtering
            if len(filtered_documents) > self.config.top_k:
                filtered_documents = filtered_documents[:self.config.top_k]
                filtered_scores = filtered_scores[:self.config.top_k]
            
            # Prepare search metadata
            search_metadata = {
                "search_type": self.config.search_type,
                "original_count": len(documents),
                "filtered_count": len(filtered_documents),
                "score_threshold": self.config.score_threshold,
                "filters_applied": bool(filter_dict),
                "product_name_filter": product_name_pattern
            }
            
            if self.config.search_type == "mmr":
                search_metadata.update({
                    "fetch_k": self.config.fetch_k,
                    "lambda_mult": self.config.lambda_mult
                })
            
            # Log search metrics
            self.logger.info(
                f"Search completed: {len(filtered_documents)} documents retrieved",
                extra={
                    'extra_fields': {
                        'search_type': self.config.search_type,
                        'original_results': len(documents),
                        'filtered_results': len(filtered_documents),
                        'score_threshold': self.config.score_threshold,
                        'filters_applied': filter_dict or {},
                        'avg_score': sum(filtered_scores) / len(filtered_scores) if filtered_scores else 0.0,
                        'min_score': min(filtered_scores) if filtered_scores else 0.0,
                        'max_score': max(filtered_scores) if filtered_scores else 0.0
                    }
                }
            )
            
            return SearchResult(
                documents=filtered_documents,
                scores=filtered_scores,
                query_embedding=query_embedding,
                filters_applied=filter_dict or {},
                search_metadata=search_metadata
            )
            
        except (SearchError, ConnectionError):
            # Re-raise these as-is
            raise
        except Exception as e:
            raise SearchError(
                f"Vector search failed: {str(e)}",
                search_params={
                    "search_type": self.config.search_type,
                    "top_k": self.config.top_k,
                    "embedding_dimension": len(query_embedding),
                    "filters": filter_dict
                }
            ) from e
    
    def _build_filter_dict(self, filters: MetadataFilter) -> Dict[str, Any]:
        """
        Convert MetadataFilter to Pinecone filter format.
        
        Note: product_name_pattern is NOT included here because Pinecone does not
        support substring/contains matching. Product name filtering is applied
        post-search in _apply_product_name_filter() instead.
        
        Args:
            filters: MetadataFilter instance
            
        Returns:
            Dictionary in Pinecone filter format
        """
        filter_dict = {}
        
        # Price range filters
        if filters.min_price is not None or filters.max_price is not None:
            price_filter = {}
            if filters.min_price is not None:
                price_filter["$gte"] = filters.min_price
            if filters.max_price is not None:
                price_filter["$lte"] = filters.max_price
            filter_dict["price"] = price_filter
        
        # Rating filter
        if filters.min_rating is not None:
            filter_dict["rating"] = {"$gte": filters.min_rating}
        
        # product_name_pattern is handled post-search via _apply_product_name_filter()
        
        return filter_dict
    
    def _apply_product_name_filter(
        self,
        documents: List[Document],
        scores: List[float],
        pattern: str
    ) -> tuple[List[Document], List[float]]:
        """
        Filter documents by checking if the product name contains the pattern.
        
        This is done post-search because Pinecone doesn't support substring matching.
        Performs case-insensitive contains check against the product_name metadata field.
        
        Args:
            documents: List of documents from vector search
            scores: Corresponding similarity scores
            pattern: Lowercase product name pattern to match (e.g. "iphone 12")
            
        Returns:
            Tuple of (filtered_documents, filtered_scores)
        """
        if not documents or not pattern:
            return documents, scores
        
        filtered_docs = []
        filtered_scores = []
        
        for doc, score in zip(documents, scores):
            product_name = doc.metadata.get("product_name", "")
            if pattern in product_name.lower():
                filtered_docs.append(doc)
                filtered_scores.append(score)
        
        self.logger.info(
            f"Product name filter '{pattern}': {len(documents)} -> {len(filtered_docs)} documents"
        )
        
        return filtered_docs, filtered_scores
    
    def _apply_score_threshold(
        self, 
        documents: List[Document], 
        scores: List[float]
    ) -> tuple[List[Document], List[float]]:
        """
        Filter results below score threshold.
        
        Args:
            documents: List of documents
            scores: List of similarity scores
            
        Returns:
            Tuple of (filtered_documents, filtered_scores)
        """
        if not documents or not scores:
            return [], []
        
        if len(documents) != len(scores):
            self.logger.warning(
                f"Document and score count mismatch: {len(documents)} docs, {len(scores)} scores"
            )
            # Truncate to the shorter length
            min_length = min(len(documents), len(scores))
            documents = documents[:min_length]
            scores = scores[:min_length]
        
        # Filter by score threshold
        filtered_pairs = [
            (doc, score) for doc, score in zip(documents, scores)
            if score >= self.config.score_threshold
        ]
        
        if not filtered_pairs:
            self.logger.info(
                f"No documents passed score threshold {self.config.score_threshold}",
                extra={
                    'extra_fields': {
                        'score_threshold': self.config.score_threshold,
                        'original_count': len(documents),
                        'max_score': max(scores) if scores else 0.0
                    }
                }
            )
            return [], []
        
        filtered_documents, filtered_scores = zip(*filtered_pairs)
        return list(filtered_documents), list(filtered_scores)
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the Pinecone index.
        
        Returns:
            Dictionary containing index statistics
            
        Raises:
            ConnectionError: If not connected to Pinecone
        """
        if self._pinecone_client is None:
            raise ConnectionError(
                "Not connected to Pinecone. Call initialize() first.",
                service="pinecone"
            )
        
        try:
            index = self._pinecone_client.Index(self.settings.pinecone_index_name)
            stats = index.describe_index_stats()
            
            return {
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "namespaces": dict(stats.namespaces) if stats.namespaces else {}
            }
            
        except Exception as e:
            raise ConnectionError(
                f"Failed to get index statistics: {str(e)}",
                service="pinecone",
                endpoint=f"index/{self.settings.pinecone_index_name}/stats"
            ) from e