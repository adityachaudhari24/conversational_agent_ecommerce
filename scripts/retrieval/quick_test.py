#!/usr/bin/env python3
"""
Quick Retrieval Pipeline Test Script

A simple script for quick testing of specific retrieval pipeline features.
"""

import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from pipelines.retrieval.pipeline import RetrievalPipeline
from pipelines.retrieval.search.vector_searcher import MetadataFilter
from src.utils.logging import setup_logging, get_logger
from src.utils.pipeline_logging import create_pipeline_logger


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
        use_json=True
    )



def quick_test():
    """Quick test of the retrieval pipeline."""
    
    # Set up logging first
    setup_cli_logging("INFO", "logs/retrieval_test.log")
    
    # Create pipeline logger
    pipeline_logger = create_pipeline_logger("retrieval")
    logger = get_logger(__name__)
    
    # Check environment variables
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("PINECONE_API_KEY"):
        logger.error("❌ Please set OPENAI_API_KEY and PINECONE_API_KEY environment variables")
        print("❌ Please set OPENAI_API_KEY and PINECONE_API_KEY environment variables")
        return
    
    logger.info("🚀 Starting Quick Retrieval Pipeline Test")
    print("🚀 Quick Retrieval Pipeline Test")
    print("=" * 40)
    
    # Start pipeline tracking
    pipeline_logger.start_pipeline()
    
    try:
        # Initialize pipeline
        logger.info("⚙️ Initializing pipeline...")
        print("⚙️  Initializing pipeline...")
        
        with pipeline_logger.stage("pipeline_initialization"):
            pipeline = RetrievalPipeline.from_config_file()
            pipeline.initialize()
            
        logger.info("✅ Pipeline ready!")
        print("✅ Pipeline ready!")
        
        # Test basic query
        logger.info("🔍 Testing basic query...")
        print("\n🔍 Testing basic query...")
        query = "iPhone with good camera"
        
        with pipeline_logger.stage("basic_query_test"):
            result = pipeline.retrieve(query)
            pipeline_logger.log_stage_progress("basic_query_test", 1, 1, 
                                             documents_found=len(result.documents),
                                             latency_ms=result.latency_ms)
        
        logger.info(f"Query: '{query}' - Found {len(result.documents)} documents in {result.latency_ms:.2f}ms")
        print(f"Query: '{query}'")
        print(f"Documents found: {len(result.documents)}")
        print(f"Latency: {result.latency_ms:.2f}ms")
        print(f"From cache: {result.from_cache}")
        
        if result.documents:
            doc = result.documents[0]
            product_name = doc.metadata.get('product_name', 'N/A')
            price = doc.metadata.get('price', 'N/A')
            rating = doc.metadata.get('rating', 'N/A')
            score = result.scores[0] if result.scores else 0.0
            
            logger.info(f"Top result: {product_name} - ${price} - {rating}/5 - Score: {score:.3f}")
            print(f"\nTop result:")
            print(f"  Product: {product_name}")
            print(f"  Price: ${price}")
            print(f"  Rating: {rating}/5")
            print(f"  Score: {score:.3f}")
        
        # Test cache (same query again)
        logger.info("💾 Testing cache (same query)...")
        print("\n💾 Testing cache (same query)...")
        
        with pipeline_logger.stage("cache_test"):
            result2 = pipeline.retrieve(query)
            pipeline_logger.log_stage_progress("cache_test", 1, 1,
                                             cache_hit=result2.from_cache,
                                             latency_ms=result2.latency_ms)
        
        logger.info(f"Cache test - From cache: {result2.from_cache}, Latency: {result2.latency_ms:.2f}ms")
        print(f"From cache: {result2.from_cache}")
        print(f"Latency: {result2.latency_ms:.2f}ms")
        
        # Test with filter
        logger.info("🔍 Testing with price filter...")
        print("\n🔍 Testing with price filter...")
        filters = MetadataFilter(min_price=200.0)
        
        with pipeline_logger.stage("filter_test"):
            result3 = pipeline.retrieve("iPhone reviews", filters)
            pipeline_logger.log_stage_progress("filter_test", 1, 1,
                                             documents_found=len(result3.documents),
                                             min_price_filter=200.0)
        
        logger.info(f"Filter test - Found {len(result3.documents)} documents with price >$200")
        print(f"Documents found (>$200): {len(result3.documents)}")
        
        pipeline_logger.end_pipeline(success=True)
        logger.info("✅ Quick test completed successfully!")
        print("\n✅ Quick test completed successfully!")
        
    except Exception as e:
        pipeline_logger.end_pipeline(success=False)
        pipeline_logger.log_error("quick_test_execution", str(e))
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    quick_test()