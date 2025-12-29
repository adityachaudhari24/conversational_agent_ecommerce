#!/usr/bin/env python3
"""
Command-line interface for the Data Ingestion Pipeline.

Provides a CLI entry point for running the ingestion pipeline with
configurable options, progress reporting, and comprehensive output.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

from src.utils.logging import setup_logging, get_logger
from src.utils.pipeline_logging import create_pipeline_logger
from .config import create_ingestion_settings, validate_configuration
from .pipeline import create_pipeline_from_settings
from .exceptions import ConfigurationError, DataQualityError, IngestionError


def setup_cli_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Configure logging for CLI execution.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    # Use centralized logging setup
    from src.utils.logging import setup_logging as core_setup_logging
    
    core_setup_logging(
        level=log_level,
        log_file=log_file,
        use_json=log_level.upper() == "DEBUG",  # Use JSON for debug mode
        context={'component': 'ingestion_cli'}
    )


def print_banner() -> None:
    """Print CLI banner with pipeline information."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                  Data Ingestion Pipeline                     ║
║              E-commerce RAG Application                      ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_progress_header() -> None:
    """Print progress tracking header."""
    print("\n" + "="*60)
    print("PIPELINE EXECUTION PROGRESS")
    print("="*60)


def print_stage_progress(stage_name: str, status: str = "STARTING") -> None:
    """
    Print progress for a pipeline stage.
    
    Args:
        stage_name: Name of the pipeline stage
        status: Status of the stage (STARTING, COMPLETE, FAILED)
    """
    timestamp = time.strftime("%H:%M:%S")
    status_symbols = {
        "STARTING": "🔄",
        "COMPLETE": "✅", 
        "FAILED": "❌"
    }
    
    symbol = status_symbols.get(status, "ℹ️")
    print(f"[{timestamp}] {symbol} {stage_name}: {status}")


def print_summary(summary: dict) -> None:
    """
    Print comprehensive pipeline execution summary.
    
    Args:
        summary: Pipeline execution summary dictionary
    """
    print("\n" + "="*60)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*60)
    
    # Execution status
    status = summary.get("pipeline_status", "unknown")
    status_symbol = "✅" if status == "completed" else "❌"
    print(f"\nStatus: {status_symbol} {status.upper()}")
    
    # Timing information
    execution_time = summary.get("execution_time", {})
    duration = execution_time.get("duration_formatted", "N/A")
    print(f"Duration: {duration}")
    
    # Document processing counts
    print(f"\nDocument Processing:")
    doc_counts = summary.get("document_counts", {})
    print(f"  • Loaded:     {doc_counts.get('loaded', 0):,}")
    print(f"  • Processed:  {doc_counts.get('processed', 0):,}")
    print(f"  • Chunked:    {doc_counts.get('chunked', 0):,}")
    print(f"  • Embedded:   {doc_counts.get('embedded', 0):,}")
    print(f"  • Stored:     {doc_counts.get('stored', 0):,}")
    
    # Failure information
    failures = summary.get("failure_counts", {})
    if failures.get("skipped_records", 0) > 0 or failures.get("failed_embeddings", 0) > 0:
        print(f"\nFailures:")
        if failures.get("skipped_records", 0) > 0:
            print(f"  • Skipped records:    {failures['skipped_records']:,}")
        if failures.get("failed_embeddings", 0) > 0:
            print(f"  • Failed embeddings:  {failures['failed_embeddings']:,}")
    
    # Efficiency metrics
    print(f"\nEfficiency:")
    efficiency = summary.get("efficiency_metrics", {})
    print(f"  • Processing:  {efficiency.get('processing_efficiency', 'N/A')}")
    print(f"  • Embedding:   {efficiency.get('embedding_efficiency', 'N/A')}")
    print(f"  • Storage:     {efficiency.get('storage_efficiency', 'N/A')}")
    
    # Processing rate
    summary_stats = summary.get("summary_stats", {})
    processing_rate = summary_stats.get("processing_rate", "N/A")
    print(f"  • Rate:        {processing_rate}")
    
    # Validation report details
    validation_report = summary.get("validation_report", {})
    if validation_report and validation_report.get("total_skipped", 0) > 0:
        print(f"\nValidation Issues:")
        skip_reasons = validation_report.get("skip_reasons", {})
        for reason, count in skip_reasons.items():
            print(f"  • {reason}: {count}")


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure the command-line argument parser.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Data Ingestion Pipeline for E-commerce RAG Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python -m src.pipelines.ingestion.cli
  
  # Run with custom config file
  python -m src.pipelines.ingestion.cli --config config/ingestion.yaml
  
  # Run with custom data file and verbose logging
  python -m src.pipelines.ingestion.cli --data-file data/custom_reviews.csv --log-level DEBUG
  
  # Run with custom log file
  python -m src.pipelines.ingestion.cli --log-file logs/ingestion.log
        """
    )
    
    # Configuration options
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to YAML configuration file"
    )
    
    parser.add_argument(
        "--data-file", "-d",
        type=str,
        help="Path to CSV data file (overrides config)"
    )
    
    # Logging options
    parser.add_argument(
        "--log-level", "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to log file (default: stdout only)"
    )
    
    # Pipeline options
    parser.add_argument(
        "--abort-threshold",
        type=float,
        help="Failure rate threshold to abort pipeline (0.0-1.0)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for embedding generation"
    )
    
    # Output options
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output (errors still shown)"
    )
    
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Skip banner display"
    )
    
    # Validation options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running pipeline"
    )
    
    return parser


def run_pipeline_with_progress(pipeline, quiet: bool = False) -> dict:
    """
    Run the pipeline with progress reporting.
    
    Args:
        pipeline: IngestionPipeline instance
        quiet: Whether to suppress progress output
        
    Returns:
        Pipeline execution summary
    """
    # Create pipeline logger
    pipeline_logger = create_pipeline_logger("ingestion")
    
    if not quiet:
        print_progress_header()
    
    # Start pipeline tracking
    pipeline_logger.start_pipeline()
    
    # Override pipeline methods to add progress reporting
    original_load = pipeline._load_stage
    original_transform = pipeline._transform_stage
    original_chunk = pipeline._chunk_stage
    original_embed = pipeline._embed_stage
    
    def load_with_progress():
        if not quiet:
            print_stage_progress("Data Loading")
        
        with pipeline_logger.stage("data_loading"):
            result = original_load()
            pipeline_logger.log_stage_progress("data_loading", len(result), len(result))
            
        if not quiet:
            print_stage_progress("Data Loading", "COMPLETE")
            print(f"    Loaded {len(result):,} records")
        return result
    
    def transform_with_progress(df):
        if not quiet:
            print_stage_progress("Data Transformation")
        
        with pipeline_logger.stage("data_transformation"):
            result = original_transform(df)
            pipeline_logger.log_stage_progress("data_transformation", len(result), len(df))
            
        if not quiet:
            print_stage_progress("Data Transformation", "COMPLETE")
            print(f"    Created {len(result):,} documents")
        return result
    
    def chunk_with_progress(documents):
        if not quiet:
            print_stage_progress("Document Chunking")
        
        with pipeline_logger.stage("document_chunking"):
            result = original_chunk(documents)
            pipeline_logger.log_stage_progress("document_chunking", len(result), len(documents))
            
        if not quiet:
            print_stage_progress("Document Chunking", "COMPLETE")
            print(f"    Generated {len(result):,} chunks")
        return result
    
    def embed_with_progress(documents):
        if not quiet:
            print_stage_progress("Embedding & Storage")
        
        with pipeline_logger.stage("embedding_storage"):
            original_embed(documents)
            pipeline_logger.log_stage_progress(
                "embedding_storage", 
                pipeline.stats['documents_stored'], 
                len(documents)
            )
            
        if not quiet:
            print_stage_progress("Embedding & Storage", "COMPLETE")
            print(f"    Stored {pipeline.stats['documents_stored']:,} documents")
    
    # Replace methods with progress versions
    pipeline._load_stage = load_with_progress
    pipeline._transform_stage = transform_with_progress
    pipeline._chunk_stage = chunk_with_progress
    pipeline._embed_stage = embed_with_progress
    
    try:
        result = pipeline.run()
        pipeline_logger.end_pipeline(success=True)
        return result
    except Exception as e:
        pipeline_logger.end_pipeline(success=False)
        pipeline_logger.log_error("pipeline_execution", str(e))
        if not quiet:
            print_stage_progress("Pipeline Execution", "FAILED")
        raise


def main() -> int:
    """
    Main CLI entry point.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Set up logging
    setup_cli_logging(args.log_level, args.log_file)
    logger = get_logger(__name__)
    
    try:
        # Display banner
        if not args.no_banner and not args.quiet:
            print_banner()
        
        # Load configuration
        logger.info("Loading configuration...")
        settings = create_ingestion_settings(args.config)
        
        # Apply command-line overrides with validation
        if args.data_file:
            settings.data_file_path = args.data_file
        if args.abort_threshold is not None:
            if not 0.0 <= args.abort_threshold <= 1.0:
                raise ConfigurationError("abort_threshold must be between 0.0 and 1.0")
            settings.abort_threshold = args.abort_threshold
        if args.batch_size is not None:
            if args.batch_size <= 0:
                raise ConfigurationError("batch_size must be positive")
            settings.batch_size = args.batch_size
        
        # Validate configuration
        logger.info("Validating configuration...")
        validate_configuration(settings)
        
        if not args.quiet:
            print(f"\nConfiguration loaded successfully:")
            print(f"  • Data file: {settings.data_file_path}")
            print(f"  • Index: {settings.pinecone_index_name}")
            print(f"  • Namespace: {settings.pinecone_namespace}")
            print(f"  • Embedding model: {settings.embedding_model}")
            print(f"  • Chunk size: {settings.chunk_size}")
            print(f"  • Batch size: {settings.batch_size}")
            print(f"  • Abort threshold: {settings.abort_threshold:.1%}")
        
        # Dry run mode
        if args.dry_run:
            if not args.quiet:
                print("\n✅ Configuration validation successful (dry run mode)")
            return 0
        
        # Create and run pipeline
        logger.info("Creating pipeline...")
        pipeline = create_pipeline_from_settings(settings)
        
        logger.info("Starting pipeline execution...")
        summary = run_pipeline_with_progress(pipeline, args.quiet)
        
        # Display summary
        if not args.quiet:
            print_summary(summary)
        
        logger.info("Pipeline execution completed successfully")
        return 0
        
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        if not args.quiet:
            print(f"\n❌ Configuration Error: {e}")
            if hasattr(e, 'missing_keys') and e.missing_keys:
                print(f"Missing environment variables: {', '.join(e.missing_keys)}")
        return 1
        
    except DataQualityError as e:
        logger.error(f"Data quality error: {e}")
        if not args.quiet:
            print(f"\n❌ Data Quality Error: {e}")
            if hasattr(e, 'failure_rate') and e.failure_rate:
                print(f"Failure rate: {e.failure_rate:.2%}")
        return 1
        
    except IngestionError as e:
        logger.error(f"Pipeline error: {e}")
        if not args.quiet:
            print(f"\n❌ Pipeline Error: {e}")
        return 1
        
    except KeyboardInterrupt:
        logger.warning("Pipeline execution interrupted by user")
        if not args.quiet:
            print(f"\n⚠️  Pipeline execution interrupted")
        return 1
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        if not args.quiet:
            print(f"\n❌ Unexpected Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())