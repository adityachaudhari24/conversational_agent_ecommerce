"""
Pipeline Orchestrator for the Data Ingestion Pipeline.

This module provides the main IngestionPipeline class that coordinates all
components of the ingestion process: loading, transformation, chunking,
embedding generation, and vector storage.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
from langchain_core.documents import Document

from src.utils.logging import get_logger
from src.utils.pipeline_logging import create_pipeline_logger, track_performance
from .config import IngestionSettings, create_ingestion_settings, validate_configuration
from .exceptions import DataQualityError, IngestionError
from .loaders.document_loader import DocumentLoader, LoaderConfig
from .processors.text_processor import TextProcessor, ProcessorConfig
from .processors.text_chunker import TextChunker, ChunkerConfig
from .processors.embedding_generator import EmbeddingGenerator, EmbeddingConfig
from .storage.vector_store import VectorStoreManager, VectorStoreConfig

logger = get_logger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the complete ingestion pipeline."""
    
    loader_config: LoaderConfig
    processor_config: ProcessorConfig
    chunker_config: ChunkerConfig
    embedding_config: EmbeddingConfig
    vector_store_config: VectorStoreConfig
    abort_threshold: float = 0.5  # Abort if >50% records fail


class IngestionPipeline:
    """
    Orchestrates the complete ingestion workflow.
    
    The IngestionPipeline coordinates all components to:
    - Load CSV data and validate structure
    - Transform rows into Document objects
    - Chunk large documents for optimal embedding
    - Generate embeddings using OpenAI
    - Store embeddings in Pinecone vector database
    - Provide comprehensive reporting and error handling
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the IngestionPipeline with configuration.
        
        Args:
            config: PipelineConfig containing all component configurations
        """
        self.config = config
        
        # Initialize components
        self.loader = DocumentLoader(config.loader_config)
        self.processor = TextProcessor(config.processor_config)
        self.chunker = TextChunker(config.chunker_config)
        self.embedding_generator = EmbeddingGenerator(config.embedding_config)
        
        # Vector store will be initialized after embedding model
        self.vector_store: VectorStoreManager = None
        
        # Pipeline statistics
        self.stats: Dict[str, Any] = {
            "start_time": None,
            "end_time": None,
            "duration_seconds": 0,
            "documents_loaded": 0,
            "documents_processed": 0,
            "documents_chunked": 0,
            "documents_embedded": 0,
            "documents_stored": 0,
            "failed_embeddings": 0,
            "skipped_records": 0,
            "validation_report": {},
            "deduplication_metrics": {
                "total_documents_processed": 0,
                "unique_documents": 0,
                "duplicate_documents": 0,
                "deduplication_rate": 0.0
            }
        }
        
        logger.info("Initialized IngestionPipeline")
    
    def run(self) -> Dict[str, Any]:
        """
        Execute the complete pipeline.
        
        Returns:
            Dict[str, Any]: Pipeline execution summary with statistics
            
        Raises:
            DataQualityError: If failure rate exceeds abort threshold
            IngestionError: If any pipeline stage fails critically
        """
        logger.info("Starting ingestion pipeline execution")
        self.stats["start_time"] = time.time()
        
        try:
            # Stage 1: Load data
            df = self._load_stage()
            
            # Stage 2: Transform to documents
            documents = self._transform_stage(df)
            
            # Stage 3: Chunk documents
            chunked_documents = self._chunk_stage(documents)
            
            # Stage 4: Generate embeddings and store
            self._embed_stage(chunked_documents)
            
            # Record completion time
            self.stats["end_time"] = time.time()
            self.stats["duration_seconds"] = self.stats["end_time"] - self.stats["start_time"]
            
            # Generate final summary
            summary = self.get_summary()
            
            logger.info("Pipeline execution completed successfully")
            logger.info(f"Summary: {summary['summary_stats']}")
            
            return summary
            
        except Exception as e:
            # Record failure time
            self.stats["end_time"] = time.time()
            if self.stats["start_time"]:
                self.stats["duration_seconds"] = self.stats["end_time"] - self.stats["start_time"]
            
            logger.error(f"Pipeline execution failed: {e}")
            raise
    
    def _load_stage(self) -> pd.DataFrame:
        """
        Execute loading stage.
        
        Returns:
            pd.DataFrame: Loaded and validated DataFrame
            
        Raises:
            IngestionError: If loading fails
        """
        logger.info("Stage 1: Loading CSV data")
        
        try:
            df = self.loader.load()
            self.stats["documents_loaded"] = len(df)
            
            logger.info(f"Loading stage complete: {len(df)} rows loaded")
            return df
            
        except Exception as e:
            logger.error(f"Loading stage failed: {e}")
            raise IngestionError(f"Failed to load data: {e}")
    
    def _transform_stage(self, df: pd.DataFrame) -> List[Document]:
        """
        Execute transformation stage.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            List[Document]: Transformed Document objects
            
        Raises:
            DataQualityError: If too many records fail validation
        """
        logger.info("Stage 2: Transforming data to documents")
        
        try:
            documents = self.processor.process(df)
            
            # Get validation report
            validation_report = self.processor.get_validation_report()
            self.stats["validation_report"] = validation_report
            self.stats["skipped_records"] = validation_report["total_skipped"]
            self.stats["documents_processed"] = len(documents)
            
            # Check data quality
            total_records = len(df)
            failed_records = validation_report["total_skipped"]
            self._check_data_quality(total_records, failed_records)
            
            logger.info(f"Transformation stage complete: {len(documents)} documents created, "
                       f"{failed_records} records skipped")
            return documents
            
        except DataQualityError:
            # Re-raise data quality errors
            raise
        except Exception as e:
            logger.error(f"Transformation stage failed: {e}")
            raise IngestionError(f"Failed to transform data: {e}")
    
    def _chunk_stage(self, documents: List[Document]) -> List[Document]:
        """
        Execute chunking stage.
        
        Args:
            documents: Documents to chunk
            
        Returns:
            List[Document]: Chunked documents
            
        Raises:
            IngestionError: If chunking fails
        """
        logger.info("Stage 3: Chunking documents")
        
        try:
            chunked_documents = self.chunker.chunk_documents(documents)
            self.stats["documents_chunked"] = len(chunked_documents)
            
            logger.info(f"Chunking stage complete: {len(documents)} input documents -> "
                       f"{len(chunked_documents)} output documents")
            return chunked_documents
            
        except Exception as e:
            logger.error(f"Chunking stage failed: {e}")
            raise IngestionError(f"Failed to chunk documents: {e}")
    
    def _embed_stage(self, documents: List[Document]) -> None:
        """
        Execute embedding and storage stage.
        
        Args:
            documents: Documents to embed and store
            
        Raises:
            IngestionError: If embedding or storage fails
            DataQualityError: If too many embeddings fail
        """
        logger.info("Stage 4: Generating embeddings and storing")
        
        try:
            # Initialize embedding generator
            logger.info("Initializing embedding generator")
            self.embedding_generator.initialize()
            
            # Initialize vector store with embedding model
            logger.info("Initializing vector store")
            self.vector_store = VectorStoreManager(
                self.config.vector_store_config,
                self.embedding_generator.embeddings_model
            )
            self.vector_store.initialize()
            
            # Generate embeddings
            logger.info("Generating embeddings")
            document_embeddings = self.embedding_generator.generate(documents)
            
            # Extract documents that successfully got embeddings
            successful_documents = [doc for doc, embedding in document_embeddings]
            
            # Update statistics
            self.stats["documents_embedded"] = len(successful_documents)
            self.stats["failed_embeddings"] = self.embedding_generator.get_failure_count()
            
            # Check embedding quality
            total_documents = len(documents)
            failed_embeddings = self.stats["failed_embeddings"]
            self._check_data_quality(total_documents, failed_embeddings)
            
            # Store documents in vector database
            if successful_documents:
                logger.info(f"Storing {len(successful_documents)} documents in vector database")
                
                # Track deduplication metrics
                self.stats["deduplication_metrics"]["total_documents_processed"] = len(successful_documents)
                
                # Generate deterministic IDs to check for duplicates
                document_ids = []
                for doc in successful_documents:
                    doc_id = self.vector_store._generate_document_id(doc)
                    document_ids.append(doc_id)
                
                # Check for existing documents (this is a simplified approach)
                # In a real implementation, you might query the vector store to check existing IDs
                unique_ids = set(document_ids)
                duplicate_count = len(document_ids) - len(unique_ids)
                
                self.stats["deduplication_metrics"]["unique_documents"] = len(unique_ids)
                self.stats["deduplication_metrics"]["duplicate_documents"] = duplicate_count
                
                if len(successful_documents) > 0:
                    dedup_rate = duplicate_count / len(successful_documents)
                    self.stats["deduplication_metrics"]["deduplication_rate"] = dedup_rate
                
                # Log deduplication information
                if duplicate_count > 0:
                    logger.info(f"Deduplication detected: {duplicate_count} duplicate documents "
                               f"out of {len(successful_documents)} total documents "
                               f"({dedup_rate:.2%} duplication rate)")
                else:
                    logger.info("No duplicate documents detected - all documents are unique")
                
                stored_ids = self.vector_store.store_documents(successful_documents)
                self.stats["documents_stored"] = len(stored_ids)
                
                logger.info(f"Storage complete: {len(stored_ids)} documents stored "
                           f"(using deterministic IDs for deduplication)")
            else:
                logger.warning("No documents to store - all embedding generation failed")
                self.stats["documents_stored"] = 0
            
            logger.info(f"Embedding and storage stage complete: "
                       f"{self.stats['documents_embedded']} embedded, "
                       f"{self.stats['documents_stored']} stored, "
                       f"{failed_embeddings} failed")
            
        except DataQualityError:
            # Re-raise data quality errors
            raise
        except Exception as e:
            logger.error(f"Embedding and storage stage failed: {e}")
            raise IngestionError(f"Failed to generate embeddings or store documents: {e}")
    
    def _check_data_quality(self, total: int, failed: int) -> None:
        """
        Abort if failure rate exceeds threshold.
        
        Args:
            total: Total number of records/documents
            failed: Number of failed records/documents
            
        Raises:
            DataQualityError: If failure rate exceeds abort threshold
        """
        if total == 0:
            return
        
        failure_rate = failed / total
        
        if failure_rate > self.config.abort_threshold:
            error_msg = (
                f"Data quality threshold exceeded. "
                f"Failure rate: {failure_rate:.2%} "
                f"({failed}/{total} failed), "
                f"threshold: {self.config.abort_threshold:.2%}"
            )
            logger.error(error_msg)
            raise DataQualityError(error_msg, failure_rate=failure_rate)
        
        logger.info(f"Data quality check passed: {failure_rate:.2%} failure rate "
                   f"(threshold: {self.config.abort_threshold:.2%})")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Return pipeline execution summary.
        
        Returns:
            Dict[str, Any]: Comprehensive summary with statistics and reports
        """
        # Calculate derived statistics
        processing_efficiency = 0.0
        if self.stats["documents_loaded"] > 0:
            processing_efficiency = (
                self.stats["documents_processed"] / self.stats["documents_loaded"]
            )
        
        embedding_efficiency = 0.0
        if self.stats["documents_chunked"] > 0:
            embedding_efficiency = (
                self.stats["documents_embedded"] / self.stats["documents_chunked"]
            )
        
        storage_efficiency = 0.0
        if self.stats["documents_embedded"] > 0:
            storage_efficiency = (
                self.stats["documents_stored"] / self.stats["documents_embedded"]
            )
        
        # Create summary
        summary = {
            "pipeline_status": "completed" if self.stats["end_time"] else "failed",
            "execution_time": {
                "start_time": self.stats["start_time"],
                "end_time": self.stats["end_time"],
                "duration_seconds": self.stats["duration_seconds"],
                "duration_formatted": f"{self.stats['duration_seconds']:.2f}s"
            },
            "document_counts": {
                "loaded": self.stats["documents_loaded"],
                "processed": self.stats["documents_processed"],
                "chunked": self.stats["documents_chunked"],
                "embedded": self.stats["documents_embedded"],
                "stored": self.stats["documents_stored"]
            },
            "failure_counts": {
                "skipped_records": self.stats["skipped_records"],
                "failed_embeddings": self.stats["failed_embeddings"]
            },
            "efficiency_metrics": {
                "processing_efficiency": f"{processing_efficiency:.2%}",
                "embedding_efficiency": f"{embedding_efficiency:.2%}",
                "storage_efficiency": f"{storage_efficiency:.2%}"
            },
            "deduplication_metrics": self.stats["deduplication_metrics"],
            "validation_report": self.stats["validation_report"],
            "summary_stats": {
                "total_processed": self.stats["documents_stored"],
                "time_elapsed": f"{self.stats['duration_seconds']:.2f}s",
                "processing_rate": (
                    f"{self.stats['documents_stored'] / max(self.stats['duration_seconds'], 1):.1f} docs/sec"
                    if self.stats["duration_seconds"] > 0 else "N/A"
                )
            }
        }
        
        return summary


def create_pipeline_from_settings(settings: IngestionSettings) -> IngestionPipeline:
    """
    Create an IngestionPipeline from IngestionSettings.
    
    Args:
        settings: IngestionSettings containing all configuration
        
    Returns:
        IngestionPipeline: Configured pipeline ready for execution
    """
    # Create component configurations
    loader_config = LoaderConfig(
        file_path=Path(settings.data_file_path),
        required_columns=settings.required_columns
    )
    
    processor_config = ProcessorConfig(
        content_field="review_text",
        metadata_fields=["product_name", "description", "price", "rating", "review_title"]
    )
    
    chunker_config = ChunkerConfig(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    
    embedding_config = EmbeddingConfig(
        model_name=settings.embedding_model,
        api_key=settings.openai_api_key,
        batch_size=settings.batch_size,
        expected_dimension=settings.embedding_dimension
    )
    
    vector_store_config = VectorStoreConfig(
        api_key=settings.pinecone_api_key,
        index_name=settings.pinecone_index_name,
        namespace=settings.pinecone_namespace,
        dimension=settings.embedding_dimension,
        cloud=settings.pinecone_cloud,
        region=settings.pinecone_environment
    )
    
    # Create pipeline configuration
    pipeline_config = PipelineConfig(
        loader_config=loader_config,
        processor_config=processor_config,
        chunker_config=chunker_config,
        embedding_config=embedding_config,
        vector_store_config=vector_store_config,
        abort_threshold=settings.abort_threshold
    )
    
    return IngestionPipeline(pipeline_config)


def run_ingestion_pipeline(config_file_path: str = None) -> Dict[str, Any]:
    """
    Convenience function to run the complete ingestion pipeline.
    
    Args:
        config_file_path: Optional path to YAML configuration file
        
    Returns:
        Dict[str, Any]: Pipeline execution summary
        
    Raises:
        ConfigurationError: If configuration is invalid
        IngestionError: If pipeline execution fails
        DataQualityError: If data quality is below threshold
    """
    # Load configuration
    settings = create_ingestion_settings(config_file_path)
    validate_configuration(settings)
    
    # Create and run pipeline
    pipeline = create_pipeline_from_settings(settings)
    return pipeline.run()