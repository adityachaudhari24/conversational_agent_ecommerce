"""
Unit and property tests for TextChunker component.

Tests the core functionality of splitting documents into chunks,
including property-based tests for chunking behavior validation.
"""

import pytest
from hypothesis import given, strategies as st
from langchain_core.documents import Document

from src.pipelines.ingestion.processors.text_chunker import TextChunker, ChunkerConfig


class TestTextChunker:
    """Test suite for TextChunker class."""
    
    def test_small_document_unchanged(self):
        """Test that small documents are returned unchanged."""
        # Create a small document
        content = "This is a short review."
        metadata = {"product_name": "iPhone", "rating": "4.5"}
        document = Document(page_content=content, metadata=metadata)
        
        # Initialize chunker with default config (chunk_size=1000)
        config = ChunkerConfig()
        chunker = TextChunker(config)
        
        # Process document
        result = chunker.chunk_documents([document])
        
        # Verify document is unchanged
        assert len(result) == 1
        assert result[0].page_content == content
        assert result[0].metadata == metadata
    
    def test_large_document_chunked(self):
        """Test that large documents are properly chunked."""
        # Create a large document (> 1000 chars)
        content = "This is a very long review. " * 50  # ~1400 chars
        metadata = {"product_name": "iPhone", "rating": "4.5"}
        document = Document(page_content=content, metadata=metadata)
        
        # Initialize chunker with default config (chunk_size=1000)
        config = ChunkerConfig()
        chunker = TextChunker(config)
        
        # Process document
        result = chunker.chunk_documents([document])
        
        # Verify document was chunked
        assert len(result) > 1
        
        # Verify each chunk has proper metadata
        for i, chunk in enumerate(result):
            assert len(chunk.page_content) <= config.chunk_size
            # Original metadata should be preserved
            assert chunk.metadata["product_name"] == "iPhone"
            assert chunk.metadata["rating"] == "4.5"
            # Chunk metadata should be added
            assert "chunk_index" in chunk.metadata
            assert "total_chunks" in chunk.metadata
            assert chunk.metadata["chunk_index"] == i
            assert chunk.metadata["total_chunks"] == len(result)
    
    def test_custom_chunk_size(self):
        """Test chunking with custom chunk size."""
        # Create document with known length
        content = "A" * 150  # 150 characters
        document = Document(page_content=content, metadata={})
        
        # Initialize chunker with small chunk size
        config = ChunkerConfig(chunk_size=100, chunk_overlap=20)
        chunker = TextChunker(config)
        
        # Process document
        result = chunker.chunk_documents([document])
        
        # Should be chunked since 150 > 100
        assert len(result) > 1
        for chunk in result:
            assert len(chunk.page_content) <= 100
    
    def test_multiple_documents_mixed_sizes(self):
        """Test processing multiple documents with mixed sizes."""
        # Create documents of different sizes
        small_doc = Document(page_content="Short", metadata={"type": "small"})
        large_doc = Document(page_content="X" * 1500, metadata={"type": "large"})
        
        config = ChunkerConfig()
        chunker = TextChunker(config)
        
        # Process documents
        result = chunker.chunk_documents([small_doc, large_doc])
        
        # Small doc should be unchanged, large doc should be chunked
        small_results = [doc for doc in result if doc.metadata.get("type") == "small"]
        large_results = [doc for doc in result if doc.metadata.get("type") == "large"]
        
        assert len(small_results) == 1
        assert len(large_results) > 1
        assert small_results[0].page_content == "Short"


class TestTextChunkerProperties:
    """Property-based tests for TextChunker."""
    
    @given(
        content=st.text(min_size=1, max_size=5000),
        chunk_size=st.integers(min_value=50, max_value=2000),
        chunk_overlap=st.integers(min_value=0, max_value=100)
    )
    def test_chunking_behavior_property(self, content, chunk_size, chunk_overlap):
        """
        Feature: data-ingestion-pipeline, Property 7: Chunking Behavior
        
        For any Document:
        - If page_content length > chunk_size: output SHALL be multiple chunks, 
          each with length <= chunk_size, and all chunks SHALL have identical 
          metadata to the original
        - If page_content length <= chunk_size: output SHALL be a single Document 
          with unchanged content
        
        **Validates: Requirements 3.1, 3.2, 3.5**
        """
        # Ensure chunk_overlap is not larger than chunk_size
        if chunk_overlap >= chunk_size:
            chunk_overlap = chunk_size // 2
        
        # Create test document with random metadata
        original_metadata = {
            "product_name": "Test Product",
            "rating": "4.5",
            "price": "99.99"
        }
        document = Document(page_content=content, metadata=original_metadata.copy())
        
        # Initialize chunker with test parameters
        config = ChunkerConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunker = TextChunker(config)
        
        # Process document
        result = chunker.chunk_documents([document])
        
        # Verify basic properties
        assert len(result) >= 1, "Should return at least one document"
        
        if len(content) <= chunk_size:
            # Small document case: should return single unchanged document
            assert len(result) == 1, f"Small document (len={len(content)}, chunk_size={chunk_size}) should return single document"
            assert result[0].page_content == content, "Small document content should be unchanged"
            
            # Metadata should be preserved (excluding chunk metadata that might be added)
            for key, value in original_metadata.items():
                assert result[0].metadata[key] == value, f"Original metadata key '{key}' should be preserved"
        
        else:
            # Large document case: should return multiple chunks
            assert len(result) > 1, f"Large document (len={len(content)}, chunk_size={chunk_size}) should be chunked"
            
            # Each chunk should respect size limit
            for i, chunk in enumerate(result):
                assert len(chunk.page_content) <= chunk_size, f"Chunk {i} exceeds chunk_size: {len(chunk.page_content)} > {chunk_size}"
                
                # Original metadata should be preserved in all chunks
                for key, value in original_metadata.items():
                    assert chunk.metadata[key] == value, f"Chunk {i} missing original metadata key '{key}'"
                
                # Chunk metadata should be added
                assert "chunk_index" in chunk.metadata, f"Chunk {i} missing chunk_index"
                assert "total_chunks" in chunk.metadata, f"Chunk {i} missing total_chunks"
                assert chunk.metadata["chunk_index"] == i, f"Chunk {i} has incorrect chunk_index"
                assert chunk.metadata["total_chunks"] == len(result), f"Chunk {i} has incorrect total_chunks"
        
        # Verify all chunks together contain the original content (approximately)
        # Note: Due to overlap and splitting behavior, we can't guarantee exact reconstruction
        # but we can verify that content is not lost entirely
        combined_content = "".join(chunk.page_content for chunk in result)
        if len(result) == 1:
            # For single chunk, content should be identical
            assert combined_content == content, "Single chunk should preserve exact content"
        else:
            # For multiple chunks, verify no chunk is empty
            for i, chunk in enumerate(result):
                assert len(chunk.page_content.strip()) > 0, f"Chunk {i} should not be empty"


if __name__ == "__main__":
    pytest.main([__file__])