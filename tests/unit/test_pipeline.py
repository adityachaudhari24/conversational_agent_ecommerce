"""
Tests for the IngestionPipeline orchestrator.

These tests verify that the pipeline can be properly configured and
that the orchestration logic works correctly.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from langchain_core.documents import Document

from src.pipelines.ingestion.pipeline import (
    IngestionPipeline,
    PipelineConfig,
    create_pipeline_from_settings,
)
from src.pipelines.ingestion.config import IngestionSettings
from src.pipelines.ingestion.loaders.document_loader import LoaderConfig
from src.pipelines.ingestion.processors.text_processor import ProcessorConfig
from src.pipelines.ingestion.processors.text_chunker import ChunkerConfig
from src.pipelines.ingestion.processors.embedding_generator import EmbeddingConfig
from src.pipelines.ingestion.storage.vector_store import VectorStoreConfig
from src.pipelines.ingestion.exceptions import DataQualityError, IngestionError


class TestPipelineConfig:
    """Test PipelineConfig creation and validation."""
    
    def test_pipeline_config_creation(self):
        """Test that PipelineConfig can be created with all required components."""
        loader_config = LoaderConfig(file_path=Path("test.csv"))
        processor_config = ProcessorConfig()
        chunker_config = ChunkerConfig()
        embedding_config = EmbeddingConfig(api_key="test-key")
        vector_store_config = VectorStoreConfig(api_key="test-key")
        
        config = PipelineConfig(
            loader_config=loader_config,
            processor_config=processor_config,
            chunker_config=chunker_config,
            embedding_config=embedding_config,
            vector_store_config=vector_store_config,
            abort_threshold=0.3
        )
        
        assert config.loader_config == loader_config
        assert config.processor_config == processor_config
        assert config.chunker_config == chunker_config
        assert config.embedding_config == embedding_config
        assert config.vector_store_config == vector_store_config
        assert config.abort_threshold == 0.3


class TestIngestionPipeline:
    """Test IngestionPipeline orchestration logic."""
    
    @pytest.fixture
    def sample_config(self):
        """Create a sample pipeline configuration for testing."""
        return PipelineConfig(
            loader_config=LoaderConfig(file_path=Path("test.csv")),
            processor_config=ProcessorConfig(),
            chunker_config=ChunkerConfig(),
            embedding_config=EmbeddingConfig(api_key="test-key"),
            vector_store_config=VectorStoreConfig(api_key="test-key"),
            abort_threshold=0.5
        )
    
    def test_pipeline_initialization(self, sample_config):
        """Test that pipeline initializes correctly with all components."""
        pipeline = IngestionPipeline(sample_config)
        
        assert pipeline.config == sample_config
        assert pipeline.loader is not None
        assert pipeline.processor is not None
        assert pipeline.chunker is not None
        assert pipeline.embedding_generator is not None
        assert pipeline.vector_store is None  # Not initialized until run
        assert pipeline.stats["documents_loaded"] == 0
    
    def test_check_data_quality_pass(self, sample_config):
        """Test data quality check passes when failure rate is below threshold."""
        pipeline = IngestionPipeline(sample_config)
        
        # Should not raise exception when failure rate is below threshold
        pipeline._check_data_quality(total=100, failed=30)  # 30% failure rate
    
    def test_check_data_quality_fail(self, sample_config):
        """Test data quality check fails when failure rate exceeds threshold."""
        pipeline = IngestionPipeline(sample_config)
        
        # Should raise DataQualityError when failure rate exceeds threshold
        with pytest.raises(DataQualityError) as exc_info:
            pipeline._check_data_quality(total=100, failed=60)  # 60% failure rate
        
        assert "Data quality threshold exceeded" in str(exc_info.value)
        assert exc_info.value.failure_rate == 0.6
    
    def test_check_data_quality_zero_total(self, sample_config):
        """Test data quality check handles zero total gracefully."""
        pipeline = IngestionPipeline(sample_config)
        
        # Should not raise exception when total is zero
        pipeline._check_data_quality(total=0, failed=0)
    
    def test_get_summary_initial_state(self, sample_config):
        """Test get_summary returns correct initial state."""
        pipeline = IngestionPipeline(sample_config)
        
        summary = pipeline.get_summary()
        
        assert summary["pipeline_status"] == "failed"  # No end_time set
        assert summary["document_counts"]["loaded"] == 0
        assert summary["document_counts"]["processed"] == 0
        assert summary["document_counts"]["stored"] == 0
        assert summary["failure_counts"]["skipped_records"] == 0
    
    @patch('src.pipelines.ingestion.pipeline.time.time')
    def test_get_summary_with_stats(self, mock_time, sample_config):
        """Test get_summary returns correct statistics after execution."""
        # Return a consistent time value for all calls
        mock_time.return_value = 1000.0
        
        pipeline = IngestionPipeline(sample_config)
        pipeline.stats.update({
            "start_time": 1000.0,
            "end_time": 1010.0,
            "duration_seconds": 10.0,
            "documents_loaded": 100,
            "documents_processed": 95,
            "documents_chunked": 98,
            "documents_embedded": 90,
            "documents_stored": 90,
            "skipped_records": 5,
            "failed_embeddings": 8,
            "validation_report": {"total_skipped": 5},
            "deduplication_metrics": {
                "total_documents_processed": 90,
                "unique_documents": 85,
                "duplicate_documents": 5,
                "deduplication_rate": 0.056
            }
        })
        
        summary = pipeline.get_summary()
        
        assert summary["pipeline_status"] == "completed"
        assert summary["document_counts"]["loaded"] == 100
        assert summary["document_counts"]["stored"] == 90
        assert summary["execution_time"]["duration_seconds"] == 10.0
        assert "95.00%" in summary["efficiency_metrics"]["processing_efficiency"]
        assert summary["summary_stats"]["total_processed"] == 90
        
        # Check deduplication metrics
        assert "deduplication_metrics" in summary
        assert summary["deduplication_metrics"]["total_documents_processed"] == 90
        assert summary["deduplication_metrics"]["unique_documents"] == 85
        assert summary["deduplication_metrics"]["duplicate_documents"] == 5
        assert summary["deduplication_metrics"]["deduplication_rate"] == 0.056


class TestPipelineFromSettings:
    """Test creating pipeline from IngestionSettings."""
    
    def test_create_pipeline_from_settings(self):
        """Test that pipeline can be created from IngestionSettings."""
        settings = IngestionSettings(
            openai_api_key="test-openai-key",
            pinecone_api_key="test-pinecone-key",
            data_file_path="test_data.csv",
            chunk_size=500,
            chunk_overlap=100,
            batch_size=50,
            abort_threshold=0.3
        )
        
        pipeline = create_pipeline_from_settings(settings)
        
        assert isinstance(pipeline, IngestionPipeline)
        assert pipeline.config.loader_config.file_path == Path("test_data.csv")
        assert pipeline.config.chunker_config.chunk_size == 500
        assert pipeline.config.chunker_config.chunk_overlap == 100
        assert pipeline.config.embedding_config.batch_size == 50
        assert pipeline.config.abort_threshold == 0.3
        assert pipeline.config.embedding_config.api_key == "test-openai-key"
        assert pipeline.config.vector_store_config.api_key == "test-pinecone-key"


class TestPipelineStages:
    """Test individual pipeline stages with mocked components."""
    
    @pytest.fixture
    def pipeline_with_mocks(self):
        """Create pipeline with mocked components for testing."""
        config = PipelineConfig(
            loader_config=LoaderConfig(file_path=Path("test.csv")),
            processor_config=ProcessorConfig(),
            chunker_config=ChunkerConfig(),
            embedding_config=EmbeddingConfig(api_key="test-key"),
            vector_store_config=VectorStoreConfig(api_key="test-key"),
            abort_threshold=0.5
        )
        
        pipeline = IngestionPipeline(config)
        
        # Mock all components
        pipeline.loader = Mock()
        pipeline.processor = Mock()
        pipeline.chunker = Mock()
        pipeline.embedding_generator = Mock()
        
        return pipeline
    
    def test_load_stage_success(self, pipeline_with_mocks):
        """Test successful load stage execution."""
        # Setup mock
        sample_df = pd.DataFrame({"col1": [1, 2, 3]})
        pipeline_with_mocks.loader.load.return_value = sample_df
        
        # Execute stage
        result = pipeline_with_mocks._load_stage()
        
        # Verify
        assert result.equals(sample_df)
        assert pipeline_with_mocks.stats["documents_loaded"] == 3
        pipeline_with_mocks.loader.load.assert_called_once()
    
    def test_load_stage_failure(self, pipeline_with_mocks):
        """Test load stage failure handling."""
        # Setup mock to raise exception
        pipeline_with_mocks.loader.load.side_effect = Exception("Load failed")
        
        # Execute and verify exception
        with pytest.raises(IngestionError) as exc_info:
            pipeline_with_mocks._load_stage()
        
        assert "Failed to load data" in str(exc_info.value)
    
    def test_transform_stage_success(self, pipeline_with_mocks):
        """Test successful transform stage execution."""
        # Setup mocks
        sample_df = pd.DataFrame({"col1": [1, 2, 3]})
        sample_docs = [Document(page_content="doc1"), Document(page_content="doc2")]
        validation_report = {"total_skipped": 1, "skip_reasons": {}}
        
        pipeline_with_mocks.processor.process.return_value = sample_docs
        pipeline_with_mocks.processor.get_validation_report.return_value = validation_report
        
        # Execute stage
        result = pipeline_with_mocks._transform_stage(sample_df)
        
        # Verify
        assert result == sample_docs
        assert pipeline_with_mocks.stats["documents_processed"] == 2
        assert pipeline_with_mocks.stats["skipped_records"] == 1
        pipeline_with_mocks.processor.process.assert_called_once_with(sample_df)
    
    def test_transform_stage_data_quality_failure(self, pipeline_with_mocks):
        """Test transform stage fails when data quality is poor."""
        # Setup mocks - high failure rate
        sample_df = pd.DataFrame({"col1": [1, 2, 3, 4, 5]})  # 5 records
        sample_docs = [Document(page_content="doc1")]  # Only 1 success
        validation_report = {"total_skipped": 4, "skip_reasons": {}}  # 4 failures = 80%
        
        pipeline_with_mocks.processor.process.return_value = sample_docs
        pipeline_with_mocks.processor.get_validation_report.return_value = validation_report
        
        # Execute and verify exception
        with pytest.raises(DataQualityError):
            pipeline_with_mocks._transform_stage(sample_df)
    
    def test_chunk_stage_success(self, pipeline_with_mocks):
        """Test successful chunk stage execution."""
        # Setup mocks
        input_docs = [Document(page_content="doc1"), Document(page_content="doc2")]
        chunked_docs = [
            Document(page_content="chunk1"), 
            Document(page_content="chunk2"),
            Document(page_content="chunk3")
        ]
        
        pipeline_with_mocks.chunker.chunk_documents.return_value = chunked_docs
        
        # Execute stage
        result = pipeline_with_mocks._chunk_stage(input_docs)
        
        # Verify
        assert result == chunked_docs
        assert pipeline_with_mocks.stats["documents_chunked"] == 3
        pipeline_with_mocks.chunker.chunk_documents.assert_called_once_with(input_docs)