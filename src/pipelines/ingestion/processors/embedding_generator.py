"""
Embedding Generator for converting documents into vector embeddings.

This module provides functionality to generate embeddings from LangChain Documents
using OpenAI's embedding models. Handles batch processing, error recovery,
and validation of embedding dimensions.
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from ..exceptions import ConfigurationError, IngestionError

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for the EmbeddingGenerator."""
    
    model_name: str = "text-embedding-3-large"
    api_key: Optional[str] = None
    batch_size: int = 100
    expected_dimension: int = 3072


class EmbeddingGenerator:
    """
    Generates embeddings for documents using OpenAI embedding models.
    
    The EmbeddingGenerator is responsible for:
    - Initializing the OpenAI embeddings model
    - Processing documents in batches for efficiency
    - Validating embedding dimensions
    - Handling API errors gracefully
    - Tracking failed documents for reporting
    """
    
    def __init__(self, config: EmbeddingConfig):
        """
        Initialize the EmbeddingGenerator with configuration.
        
        Args:
            config: EmbeddingConfig containing model settings and API key
        """
        self.config = config
        self.embeddings_model: Optional[OpenAIEmbeddings] = None
        self.failed_documents: List[Document] = []
        logger.info(f"Initialized EmbeddingGenerator with model: {config.model_name}")
    
    def initialize(self) -> None:
        """
        Initialize the embedding model.
        
        Raises:
            ConfigurationError: If API key is missing or invalid
        """
        if not self.config.api_key:
            raise ConfigurationError(
                "OpenAI API key is required for embedding generation. "
                "Please set OPENAI_API_KEY environment variable."
            )
        
        try:
            self.embeddings_model = OpenAIEmbeddings(
                model=self.config.model_name,
                openai_api_key=self.config.api_key
            )
            
            # Test the connection with a simple embedding
            test_embedding = self.embeddings_model.embed_query("test")
            
            # Validate the dimension
            if len(test_embedding) != self.config.expected_dimension:
                raise ConfigurationError(
                    f"Embedding dimension mismatch. Expected {self.config.expected_dimension}, "
                    f"got {len(test_embedding)} for model {self.config.model_name}"
                )
            
            logger.info(f"Successfully initialized OpenAI embeddings model: {self.config.model_name}")
            
        except Exception as e:
            if "api" in str(e).lower() or "key" in str(e).lower():
                raise ConfigurationError(
                    f"Invalid OpenAI API key or API error: {e}"
                )
            else:
                raise ConfigurationError(
                    f"Failed to initialize embedding model: {e}"
                )
    
    def generate(self, documents: List[Document]) -> List[Tuple[Document, List[float]]]:
        """
        Generate embeddings for all documents.
        
        Args:
            documents: List of Document objects to embed
            
        Returns:
            List[Tuple[Document, List[float]]]: List of (document, embedding) pairs
            
        Raises:
            IngestionError: If embeddings model is not initialized
        """
        if self.embeddings_model is None:
            raise IngestionError(
                "Embedding model not initialized. Call initialize() first."
            )
        
        logger.info(f"Generating embeddings for {len(documents)} documents")
        
        # Reset failed documents for this generation run
        self.failed_documents = []
        results = []
        
        # Process documents in batches
        for i in range(0, len(documents), self.config.batch_size):
            batch = documents[i:i + self.config.batch_size]
            batch_start = i + 1
            batch_end = min(i + self.config.batch_size, len(documents))
            
            logger.debug(f"Processing batch {batch_start}-{batch_end} of {len(documents)}")
            
            try:
                batch_embeddings = self._generate_batch(batch)
                
                # Pair documents with their embeddings
                for doc, embedding in zip(batch, batch_embeddings):
                    if embedding is not None:
                        results.append((doc, embedding))
                    else:
                        self.failed_documents.append(doc)
                        
            except Exception as e:
                logger.error(f"Failed to process batch {batch_start}-{batch_end}: {e}")
                # Add all documents in failed batch to failed list
                self.failed_documents.extend(batch)
                continue
        
        success_count = len(results)
        failed_count = len(self.failed_documents)
        
        logger.info(
            f"Embedding generation complete. Success: {success_count}, "
            f"Failed: {failed_count}, Total: {len(documents)}"
        )
        
        return results
    
    def _generate_batch(self, batch: List[Document]) -> List[Optional[List[float]]]:
        """
        Generate embeddings for a batch of documents.
        
        Args:
            batch: List of Document objects to embed
            
        Returns:
            List[Optional[List[float]]]: List of embeddings (None for failed documents)
        """
        try:
            # Extract text content from documents
            texts = [doc.page_content for doc in batch]
            
            # Generate embeddings for the batch
            embeddings = self.embeddings_model.embed_documents(texts)
            
            # Validate each embedding
            validated_embeddings = []
            for i, embedding in enumerate(embeddings):
                if self._validate_embedding(embedding):
                    validated_embeddings.append(embedding)
                else:
                    logger.warning(
                        f"Invalid embedding dimension for document {i} in batch. "
                        f"Expected {self.config.expected_dimension}, got {len(embedding)}"
                    )
                    validated_embeddings.append(None)
            
            return validated_embeddings
            
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            # Return None for all documents in the failed batch
            return [None] * len(batch)
    
    def _validate_embedding(self, embedding: List[float]) -> bool:
        """
        Validate embedding dimension.
        
        Args:
            embedding: The embedding vector to validate
            
        Returns:
            bool: True if embedding dimension is correct, False otherwise
        """
        if not embedding:
            return False
        
        return len(embedding) == self.config.expected_dimension
    
    def get_failed_documents(self) -> List[Document]:
        """
        Get list of documents that failed embedding generation.
        
        Returns:
            List[Document]: Documents that failed to generate embeddings
        """
        return self.failed_documents.copy()
    
    def get_failure_count(self) -> int:
        """
        Get count of documents that failed embedding generation.
        
        Returns:
            int: Number of failed documents
        """
        return len(self.failed_documents)