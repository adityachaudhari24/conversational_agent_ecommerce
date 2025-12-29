# Implementation Plan: Data Ingestion Pipeline

## Overview

This implementation plan breaks down the Data Ingestion Pipeline into discrete coding tasks. Each task builds incrementally on previous work, ensuring no orphaned code. The pipeline will be implemented in Python using LangChain, Pinecone, and OpenAI embeddings.

## Tasks

- [x] 1. Set up project structure and core utilities
  - [x] 1.1 Create ingestion pipeline directory structure
    - Create `src/pipelines/ingestion/` with `__init__.py`
    - Create subdirectories: `loaders/`, `processors/`, `storage/`
    - _Requirements: 6.5_

  - [x] 1.2 Implement custom exception classes
    - Create `src/pipelines/ingestion/exceptions.py`
    - Implement IngestionError, ValidationError, ConfigurationError, ConnectionError, DataQualityError
    - _Requirements: 8.5_

  - [x] 1.3 Create configuration schema and loader
    - Create `src/pipelines/ingestion/config.py`
    - Implement IngestionSettings using Pydantic BaseSettings
    - Add YAML config loading support
    - _Requirements: 7.1, 7.2, 7.4, 7.5_

  - [ ]* 1.4 Write property test for missing environment variable detection
    - **Property 10: Missing Environment Variable Detection**
    - **Validates: Requirements 7.3**

- [x] 2. Implement Document Loader component
  - [x] 2.1 Create DocumentLoader class
    - Create `src/pipelines/ingestion/loaders/document_loader.py`
    - Implement LoaderConfig dataclass
    - Implement load(), _validate_file_exists(), _validate_columns(), _filter_empty_rows()
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [ ]* 2.2 Write property test for CSV loading round-trip
    - **Property 1: CSV Loading Round-Trip**
    - **Validates: Requirements 1.1**

  - [ ]* 2.3 Write property test for missing columns detection
    - **Property 2: Missing Columns Detection**
    - **Validates: Requirements 1.3**

  - [x] 2.4 Write property test for empty row filtering
    - **Property 3: Empty Row Filtering**
    - **Validates: Requirements 1.4**

- [x] 3. Implement Text Processor component
  - [x] 3.1 Create TextProcessor class
    - Create `src/pipelines/ingestion/processors/text_processor.py`
    - Implement ProcessorConfig dataclass
    - Implement process(), _create_document(), _sanitize_metadata(), _is_valid_content()
    - Implement get_validation_report()
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [ ]* 3.2 Write property test for document creation
    - **Property 4: Document Creation Preserves Data**
    - **Validates: Requirements 2.1, 2.2**

  - [ ]* 3.3 Write property test for NaN value replacement
    - **Property 5: NaN Value Replacement**
    - **Validates: Requirements 2.3**

  - [ ]* 3.4 Write property test for whitespace content rejection
    - **Property 6: Whitespace Content Rejection**
    - **Validates: Requirements 2.4**

- [x] 4. Implement Text Chunker component
  - [x] 4.1 Create TextChunker class
    - Create `src/pipelines/ingestion/processors/text_chunker.py`
    - Implement ChunkerConfig dataclass
    - Implement chunk_documents(), _needs_chunking()
    - Use LangChain RecursiveCharacterTextSplitter
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

  - [x] 4.2 Write property test for chunking behavior
    - **Property 7: Chunking Behavior**
    - **Validates: Requirements 3.1, 3.2, 3.5**

- [ ]* 5. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement Embedding Generator component
  - [x] 6.1 Create EmbeddingGenerator class
    - Create `src/pipelines/ingestion/processors/embedding_generator.py`
    - Implement EmbeddingConfig dataclass
    - Implement initialize(), generate(), _generate_batch(), _validate_embedding()
    - Integrate with OpenAI embeddings
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [ ]* 6.2 Write property test for embedding dimension consistency
    - **Property 8: Embedding Dimension Consistency**
    - **Validates: Requirements 9.3**

- [-] 7. Implement Vector Store component
  - [x] 7.1 Create VectorStoreManager class
    - Create `src/pipelines/ingestion/storage/vector_store.py`
    - Implement VectorStoreConfig dataclass
    - Implement initialize(), store_documents(), _create_index_if_not_exists(), _upsert_batch()
    - Integrate with Pinecone
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

  - [ ]* 7.2 Write property test for storage verification
    - **Property 9: Storage Verification**
    - **Validates: Requirements 5.3, 5.4**

- [ ] 8. Implement Pipeline Orchestrator
  - [x] 8.1 Create IngestionPipeline class
    - Create `src/pipelines/ingestion/pipeline.py`
    - Implement PipelineConfig dataclass
    - Implement run(), _load_stage(), _transform_stage(), _chunk_stage(), _embed_stage()
    - Implement _check_data_quality(), get_summary()
    - Wire all components together
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

  - [ ]* 8.2 Write property test for validation report accuracy
    - **Property 11: Validation Report Accuracy**
    - **Validates: Requirements 9.4**

  - [ ]* 8.3 Write property test for data quality threshold enforcement
    - **Property 12: Data Quality Threshold Enforcement**
    - **Validates: Requirements 9.5**

  - [ ]* 8.4 Write property test for pipeline summary completeness
    - **Property 13: Pipeline Summary Completeness**
    - **Validates: Requirements 6.4**

- [x] 9. Implement CLI interface
  - [x] 9.1 Create CLI entry point
    - Create `src/pipelines/ingestion/cli.py`
    - Implement command-line argument parsing
    - Add progress output and summary display
    - _Requirements: 6.5_

- [ ]* 10. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ]* 11. Create test fixtures and integration tests
  - [ ] 11.1 Create test fixtures
    - Create `tests/fixtures/sample_reviews.csv` with valid test data
    - Create `tests/fixtures/invalid_reviews.csv` with edge cases
    - Create `tests/fixtures/large_reviews.csv` for performance testing
    - _Requirements: 1.1, 2.1_

  - [ ]* 11.2 Write integration test for full pipeline
    - Test complete pipeline execution with sample data
    - Verify end-to-end data flow
    - _Requirements: 6.1_

- [ ] 12. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 13. Implement document deduplication
  - [x] 13.1 Add deterministic ID generation to VectorStoreManager
    - Modify `store_documents()` method to generate content-based IDs instead of random UUIDs
    - Implement `_generate_document_id()` method using SHA-256 hash of page_content and key metadata
    - Use product_name, review_title, and page_content for hash generation
    - Update logging to show when documents are being processed with deterministic IDs
    - _Requirements: 10.1, 10.2, 10.4_

  - [ ]* 13.2 Write property test for document deduplication consistency
    - **Property 14: Document Deduplication Consistency**
    - **Validates: Requirements 10.1, 10.2**

  - [ ]* 13.3 Write property test for idempotent pipeline execution
    - **Property 15: Idempotent Pipeline Execution**
    - **Validates: Requirements 10.3, 10.5**

  - [x] 13.4 Update pipeline statistics to track deduplication
    - Add logging to distinguish between new documents and updates
    - Update summary statistics to show deduplication metrics
    - _Requirements: 10.3_

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties
- Unit tests validate specific examples and edge cases
- External API calls (Pinecone, OpenAI) should be mocked in unit tests
