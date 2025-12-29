"""
Query Processor for the retrieval pipeline.

This module handles query validation, normalization, and embedding generation
for user queries in the retrieval pipeline. It ensures queries are properly
formatted and converted to embeddings for vector similarity search.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional

from langchain_openai import OpenAIEmbeddings

from ..config import RetrievalSettings
from ..exceptions import QueryValidationError, EmbeddingError, ConfigurationError
from ..logging import RetrievalLoggerMixin, log_retrieval_operation


@dataclass
class QueryConfig:
    """Configuration for query processing."""
    
    max_length: int = 512
    embedding_model: str = "text-embedding-3-large"
    normalize_unicode: bool = True
    expected_dimension: int = 3072


@dataclass
class ProcessedQuery:
    """Result of query processing containing original, normalized, and embedded query."""
    
    original: str
    normalized: str
    embedding: List[float]
    truncated: bool = False


class QueryProcessor(RetrievalLoggerMixin):
    """
    Processes and embeds user queries for retrieval.
    
    The QueryProcessor is responsible for:
    - Validating user queries (non-empty, reasonable length)
    - Normalizing text (whitespace, unicode normalization)
    - Truncating queries that exceed maximum length
    - Generating embeddings using OpenAI's embedding models
    - Logging query processing metrics
    """
    
    def __init__(self, config: QueryConfig, openai_api_key: str):
        """
        Initialize the QueryProcessor with configuration.
        
        Args:
            config: QueryConfig containing processing settings
            openai_api_key: OpenAI API key for embedding generation
            
        Raises:
            ConfigurationError: If API key is missing or invalid
        """
        super().__init__()
        self.config = config
        self.embeddings_model: Optional[OpenAIEmbeddings] = None
        
        if not openai_api_key:
            raise ConfigurationError(
                "OpenAI API key is required for query processing. "
                "Please set OPENAI_API_KEY environment variable."
            )
        
        try:
            self.embeddings_model = OpenAIEmbeddings(
                model=config.embedding_model,
                openai_api_key=openai_api_key
            )
            
            # Test the connection and validate dimension
            test_embedding = self.embeddings_model.embed_query("test")
            if len(test_embedding) != config.expected_dimension:
                raise ConfigurationError(
                    f"Embedding dimension mismatch. Expected {config.expected_dimension}, "
                    f"got {len(test_embedding)} for model {config.embedding_model}"
                )
            
            self.logger.info(
                f"Successfully initialized QueryProcessor with model: {config.embedding_model}"
            )
            
        except Exception as e:
            if "api" in str(e).lower() or "key" in str(e).lower():
                raise ConfigurationError(f"Invalid OpenAI API key or API error: {e}")
            else:
                raise ConfigurationError(f"Failed to initialize embedding model: {e}")
    
    @classmethod
    def from_settings(cls, settings: RetrievalSettings) -> 'QueryProcessor':
        """
        Create QueryProcessor from RetrievalSettings.
        
        Args:
            settings: RetrievalSettings instance
            
        Returns:
            QueryProcessor instance
        """
        config = QueryConfig(
            max_length=settings.max_query_length,
            embedding_model=settings.embedding_model,
            normalize_unicode=settings.normalize_unicode,
            expected_dimension=settings.embedding_dimension
        )
        
        return cls(config, settings.openai_api_key)
    
    @log_retrieval_operation("query_processing")
    def process(self, query: str) -> ProcessedQuery:
        """
        Process query: validate, normalize, and embed.
        
        Args:
            query: Raw user query string
            
        Returns:
            ProcessedQuery containing processed query and embedding
            
        Raises:
            QueryValidationError: If query validation fails
            EmbeddingError: If embedding generation fails
        """
        # Step 1: Validate the query
        self._validate(query)
        
        # Step 2: Normalize the query
        normalized = self._normalize(query)
        
        # Step 3: Truncate if necessary
        truncated_text, was_truncated = self._truncate(normalized)
        
        if was_truncated:
            self.logger.warning(
                f"Query truncated from {len(normalized)} to {len(truncated_text)} characters",
                extra={
                    'extra_fields': {
                        'original_length': len(normalized),
                        'truncated_length': len(truncated_text),
                        'max_length': self.config.max_length
                    }
                }
            )
        
        # Step 4: Generate embedding
        embedding = self._embed(truncated_text)
        
        # Log processing metrics
        self.logger.info(
            "Query processed successfully",
            extra={
                'extra_fields': {
                    'original_length': len(query),
                    'normalized_length': len(normalized),
                    'final_length': len(truncated_text),
                    'truncated': was_truncated,
                    'embedding_dimension': len(embedding)
                }
            }
        )
        
        return ProcessedQuery(
            original=query,
            normalized=truncated_text,
            embedding=embedding,
            truncated=was_truncated
        )
    
    def _validate(self, query: str) -> None:
        """
        Validate query input.
        
        Args:
            query: Query string to validate
            
        Raises:
            QueryValidationError: If query is invalid (empty, whitespace-only, etc.)
        """
        if not isinstance(query, str):
            raise QueryValidationError(
                f"Query must be a string, got {type(query).__name__}",
                query=str(query) if query is not None else None
            )
        
        # Check if query is empty or whitespace-only
        if not query or not query.strip():
            raise QueryValidationError(
                "Query cannot be empty or contain only whitespace",
                query=query
            )
        
        # Additional validation could be added here:
        # - Check for minimum length
        # - Check for valid characters
        # - Check for SQL injection patterns, etc.
    
    def _normalize(self, query: str) -> str:
        """
        Normalize whitespace and unicode characters.
        
        Args:
            query: Query string to normalize
            
        Returns:
            str: Normalized query string
        """
        # Strip leading and trailing whitespace
        normalized = query.strip()
        
        # Normalize internal whitespace (replace multiple spaces with single space)
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Unicode normalization if enabled
        if self.config.normalize_unicode:
            # Use NFC (Canonical Decomposition followed by Canonical Composition)
            # This is the most common normalization form for text processing
            normalized = unicodedata.normalize('NFC', normalized)
        
        return normalized
    
    def _truncate(self, query: str) -> tuple[str, bool]:
        """
        Truncate query if it exceeds maximum length.
        
        Args:
            query: Query string to potentially truncate
            
        Returns:
            tuple[str, bool]: (truncated_text, was_truncated)
        """
        if len(query) <= self.config.max_length:
            return query, False
        
        # Truncate to max_length characters
        # Try to truncate at word boundary if possible
        truncated = query[:self.config.max_length]
        
        # If we're in the middle of a word, try to find the last space
        if len(query) > self.config.max_length and ' ' in truncated:
            last_space = truncated.rfind(' ')
            if last_space > self.config.max_length * 0.8:  # Only if we don't lose too much
                truncated = truncated[:last_space]
        
        return truncated, True
    
    def _embed(self, text: str) -> List[float]:
        """
        Generate embedding vector for text.
        
        Args:
            text: Text to embed
            
        Returns:
            List[float]: Embedding vector
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        if self.embeddings_model is None:
            raise EmbeddingError(
                "Embedding model not initialized",
                query=text
            )
        
        try:
            embedding = self.embeddings_model.embed_query(text)
            
            # Validate embedding dimension
            if len(embedding) != self.config.expected_dimension:
                raise EmbeddingError(
                    f"Embedding dimension mismatch. Expected {self.config.expected_dimension}, "
                    f"got {len(embedding)}",
                    query=text,
                    model=self.config.embedding_model
                )
            
            return embedding
            
        except Exception as e:
            # Re-raise EmbeddingError as-is
            if isinstance(e, EmbeddingError):
                raise
            
            # Wrap other exceptions in EmbeddingError
            raise EmbeddingError(
                f"Failed to generate embedding: {str(e)}",
                query=text,
                model=self.config.embedding_model
            ) from e