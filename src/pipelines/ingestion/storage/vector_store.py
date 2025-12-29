"""
Vector Store Manager for storing document embeddings in Pinecone.

This module provides functionality to store document embeddings in Pinecone
vector database, including index management, batch operations, and error handling.
"""

import hashlib
import logging
import uuid
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from ..exceptions import ConnectionError, ConfigurationError, IngestionError

logger = logging.getLogger(__name__)


@dataclass
class VectorStoreConfig:
    """Configuration for the VectorStoreManager."""
    
    api_key: str
    index_name: str = "ecommerce-products"
    namespace: str = "phone-reviews"
    dimension: int = 3072
    metric: str = "cosine"
    cloud: str = "aws"
    region: str = "us-east-1"


class VectorStoreManager:
    """
    Manages Pinecone vector storage for document embeddings.
    
    The VectorStoreManager is responsible for:
    - Initializing Pinecone client and creating indexes
    - Storing document embeddings with metadata
    - Managing batch operations for efficiency
    - Handling connection errors and retries
    - Returning document IDs for tracking
    """
    
    def __init__(self, config: VectorStoreConfig, embeddings_model: OpenAIEmbeddings):
        """
        Initialize the VectorStoreManager with configuration and embeddings model.
        
        Args:
            config: VectorStoreConfig containing Pinecone settings
            embeddings_model: OpenAI embeddings model for LangChain integration
        """
        self.config = config
        self.embeddings_model = embeddings_model
        self.pc: Optional[Pinecone] = None
        self.index = None
        self.vector_store: Optional[PineconeVectorStore] = None
        logger.info(f"Initialized VectorStoreManager for index: {config.index_name}")
    
    def initialize(self) -> None:
        """
        Initialize Pinecone client and create index if needed.
        
        Raises:
            ConfigurationError: If API key is missing or invalid
            ConnectionError: If connection to Pinecone fails
        """
        if not self.config.api_key:
            raise ConfigurationError(
                "Pinecone API key is required for vector storage. "
                "Please set PINECONE_API_KEY environment variable."
            )
        
        try:
            # Initialize Pinecone client
            self.pc = Pinecone(api_key=self.config.api_key)
            
            # Create index if it doesn't exist
            self._create_index_if_not_exists()
            
            # Get the index
            self.index = self.pc.Index(self.config.index_name)
            
            # Initialize LangChain PineconeVectorStore
            self.vector_store = PineconeVectorStore(
                index=self.index,
                embedding=self.embeddings_model,
                namespace=self.config.namespace
            )
            
            logger.info(f"Successfully initialized Pinecone vector store: {self.config.index_name}")
            
        except Exception as e:
            if "api" in str(e).lower() or "key" in str(e).lower() or "auth" in str(e).lower():
                raise ConnectionError(
                    f"Invalid Pinecone API key or authentication error: {e}",
                    service="pinecone"
                )
            else:
                raise ConnectionError(
                    f"Failed to connect to Pinecone: {e}",
                    service="pinecone"
                )
    
    def store_documents(self, documents: List[Document]) -> List[str]:
        """
        Store documents with embeddings in Pinecone.
        
        Args:
            documents: List of Document objects to store
            
        Returns:
            List[str]: List of document IDs that were successfully stored
            
        Raises:
            IngestionError: If vector store is not initialized
            ConnectionError: If storage operation fails
        """
        if self.vector_store is None:
            raise IngestionError(
                "Vector store not initialized. Call initialize() first."
            )
        
        if not documents:
            logger.warning("No documents provided for storage")
            return []
        
        logger.info(f"Storing {len(documents)} documents in Pinecone")
        
        try:
            # Prepare documents with deterministic IDs
            documents_with_ids = []
            document_ids = []
            
            for doc in documents:
                # Generate deterministic ID based on content and key metadata
                doc_id = self._generate_document_id(doc)
                document_ids.append(doc_id)
                
                # Prepare metadata for Pinecone (convert all values to strings)
                metadata = self._prepare_metadata(doc.metadata)
                
                # Create new document with prepared metadata
                prepared_doc = Document(
                    page_content=doc.page_content,
                    metadata=metadata
                )
                documents_with_ids.append(prepared_doc)
                
                # Log deterministic ID generation
                logger.debug(f"Generated deterministic ID {doc_id[:8]}... for document with "
                           f"product: {doc.metadata.get('product_name', 'N/A')}")
            
            logger.info(f"Processing {len(documents)} documents with deterministic IDs for deduplication")
            
            # Store documents using LangChain PineconeVectorStore
            stored_ids = self.vector_store.add_documents(
                documents=documents_with_ids,
                ids=document_ids,
                namespace=self.config.namespace
            )
            
            logger.info(f"Successfully stored {len(stored_ids)} documents in Pinecone with deterministic IDs")
            return stored_ids
            
        except Exception as e:
            logger.error(f"Failed to store documents in Pinecone: {e}")
            raise ConnectionError(
                f"Failed to store documents in Pinecone: {e}",
                service="pinecone"
            )
    
    def _create_index_if_not_exists(self) -> None:
        """
        Create Pinecone index if it doesn't exist.
        
        Raises:
            ConnectionError: If index creation fails
        """
        try:
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.config.index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index: {self.config.index_name}")
                
                # Create index with serverless spec
                self.pc.create_index(
                    name=self.config.index_name,
                    dimension=self.config.dimension,
                    metric=self.config.metric,
                    spec=ServerlessSpec(
                        cloud=self.config.cloud,
                        region=self.config.region
                    )
                )
                
                logger.info(f"Successfully created Pinecone index: {self.config.index_name}")
            else:
                logger.info(f"Pinecone index already exists: {self.config.index_name}")
                
        except Exception as e:
            raise ConnectionError(
                f"Failed to create Pinecone index {self.config.index_name}: {e}",
                service="pinecone"
            )
    
    def _generate_document_id(self, document: Document) -> str:
        """
        Generate a deterministic ID for a document based on content and key metadata.
        
        Uses SHA-256 hash of page_content, product_name, and review_title to ensure
        that identical documents always get the same ID, enabling deduplication.
        
        Args:
            document: Document to generate ID for
            
        Returns:
            str: Deterministic document ID based on content hash
        """
        # Extract key fields for ID generation
        page_content = document.page_content or ""
        product_name = document.metadata.get("product_name", "")
        review_title = document.metadata.get("review_title", "")
        
        # Create a string combining the key fields
        id_string = f"{page_content}|{product_name}|{review_title}"
        
        # Generate SHA-256 hash
        hash_object = hashlib.sha256(id_string.encode('utf-8'))
        document_id = hash_object.hexdigest()
        
        return document_id
        """
        Prepare metadata for Pinecone storage by converting all values to strings.
        
        Pinecone requires metadata values to be strings, numbers, or booleans.
        This method converts all values to strings for consistency.
        
        Args:
            metadata: Original metadata dictionary
            
        Returns:
            Dict[str, str]: Metadata with all values converted to strings
        """
        prepared_metadata = {}
        
        for key, value in metadata.items():
            if value is None:
                prepared_metadata[key] = "N/A"
            elif isinstance(value, (str, int, float, bool)):
                prepared_metadata[key] = str(value)
            else:
                # Convert complex types to string representation
                prepared_metadata[key] = str(value)
        
        return prepared_metadata
    def _prepare_metadata(self, metadata: Dict[str, Any]) -> Dict[str, str]:
        """
        Prepare metadata for Pinecone storage by converting all values to strings.
        
        Pinecone requires metadata values to be strings, numbers, or booleans.
        This method converts all values to strings for consistency.
        
        Args:
            metadata: Original metadata dictionary
            
        Returns:
            Dict[str, str]: Metadata with all values converted to strings
        """
        prepared_metadata = {}
        
        for key, value in metadata.items():
            if value is None:
                prepared_metadata[key] = "N/A"
            elif isinstance(value, (str, int, float, bool)):
                prepared_metadata[key] = str(value)
            else:
                # Convert complex types to string representation
                prepared_metadata[key] = str(value)
        
        return prepared_metadata
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the Pinecone index.
        
        Returns:
            Dict[str, Any]: Index statistics including vector count
            
        Raises:
            IngestionError: If index is not initialized
        """
        if self.index is None:
            raise IngestionError(
                "Index not initialized. Call initialize() first."
            )
        
        try:
            stats = self.index.describe_index_stats()
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            raise ConnectionError(
                f"Failed to get index statistics: {e}",
                service="pinecone"
            )