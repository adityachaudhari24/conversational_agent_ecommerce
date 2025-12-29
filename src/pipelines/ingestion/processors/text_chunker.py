"""
Text Chunker for splitting large documents into smaller chunks.

This module provides functionality to split LangChain Documents into smaller
chunks for optimal embedding generation. Uses LangChain's RecursiveCharacterTextSplitter
to maintain semantic coherence while respecting size constraints.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class ChunkerConfig:
    """Configuration for the TextChunker."""
    
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: Optional[List[str]] = None
    
    def __post_init__(self):
        """Set default separators if not provided."""
        if self.separators is None:
            self.separators = ["\n\n", "\n", ". ", " ", ""]


class TextChunker:
    """
    Splits documents into smaller chunks for optimal embedding.
    
    The TextChunker is responsible for:
    - Splitting large documents that exceed the configured chunk size
    - Preserving metadata across all resulting chunks
    - Using recursive character text splitting for semantic coherence
    - Leaving small documents unchanged if they fit within chunk size
    - Maintaining chunk overlap for context preservation
    """
    
    def __init__(self, config: ChunkerConfig):
        """
        Initialize the TextChunker with configuration.
        
        Args:
            config: ChunkerConfig containing chunk size, overlap, and separators
        """
        self.config = config
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators
        )
        logger.info(f"Initialized TextChunker with chunk_size={config.chunk_size}, "
                   f"chunk_overlap={config.chunk_overlap}")
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks, preserving metadata.
        
        Args:
            documents: List of Document objects to potentially chunk
            
        Returns:
            List[Document]: List of chunked documents with preserved metadata
        """
        logger.info(f"Processing {len(documents)} documents for chunking")
        
        chunked_documents = []
        total_chunks_created = 0
        
        for i, document in enumerate(documents):
            if self._needs_chunking(document):
                # Document needs chunking
                logger.debug(f"Chunking document {i} (length: {len(document.page_content)})")
                
                # Split the document using LangChain's text splitter
                chunks = self.splitter.split_documents([document])
                
                # Add chunk metadata to each chunk
                for chunk_idx, chunk in enumerate(chunks):
                    # Preserve original metadata and add chunk information
                    chunk.metadata.update({
                        "chunk_index": chunk_idx,
                        "total_chunks": len(chunks)
                    })
                
                chunked_documents.extend(chunks)
                total_chunks_created += len(chunks)
                
                logger.debug(f"Document {i} split into {len(chunks)} chunks")
            else:
                # Document is small enough, keep as-is
                logger.debug(f"Document {i} kept unchanged (length: {len(document.page_content)})")
                chunked_documents.append(document)
        
        logger.info(f"Chunking complete: {len(documents)} input documents -> "
                   f"{len(chunked_documents)} output documents "
                   f"({total_chunks_created} new chunks created)")
        
        return chunked_documents
    
    def _needs_chunking(self, document: Document) -> bool:
        """
        Check if document exceeds chunk size and needs to be split.
        
        Args:
            document: Document to check
            
        Returns:
            bool: True if document needs chunking, False otherwise
        """
        content_length = len(document.page_content)
        needs_chunking = content_length > self.config.chunk_size
        
        logger.debug(f"Document length: {content_length}, chunk_size: {self.config.chunk_size}, "
                    f"needs_chunking: {needs_chunking}")
        
        return needs_chunking