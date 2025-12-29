#!/usr/bin/env python3
"""
Simple Retrieval Pipeline Test Script

A clean, step-by-step test script for the retrieval pipeline with individual test flags.
Each test can be run independently to understand specific components.

USAGE:
    python scripts/retrieval/simple_test.py

CUSTOMIZATION:
    Edit the TEST_FLAGS dictionary below to enable/disable specific tests:
    - Set to True to run the test
    - Set to False to skip the test
    
    Example: To run only basic retrieval and compression tests:
    TEST_FLAGS = {
        "basic_retrieval": True,
        "filtered_retrieval": False, 
        "context_compression": True,
        "query_rewriting": False,
    }
"""

import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from pipelines.retrieval.pipeline import RetrievalPipeline
from pipelines.retrieval.search.vector_searcher import MetadataFilter

# ============================================================================
# TEST CONFIGURATION - Edit these flags to control which tests to run
# ============================================================================

TEST_FLAGS = {
    "basic_retrieval": True,     # Test 1: Simple vector retrieval without filters
    "filtered_retrieval": False,  # Test 2: Retrieval with metadata filters  
    "context_compression": False,  # Test 3: Context compression functionality
    "query_rewriting": False,     # Test 4: Query rewriting functionality
}

# Test Queries - Customize these if needed
TEST_QUERIES = {
    "basic": "iPhone with good camera quality",
    "filtered": "smartphone reviews", 
    "compression": "phone battery life",
    "rewriting": "good phone"  # Intentionally vague to trigger rewriting
}

# Filter Settings for Test 2
FILTER_SETTINGS = {
    "min_price": 200.0,
    "max_price": 500.0, 
    "min_rating": 4.0
}

# ============================================================================

def print_separator(title: str):
    """Print a clean separator for test sections."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_documents(docs, scores=None, title="Documents"):
    """Print documents in a clean, readable format."""
    print(f"\n📄 {title} ({len(docs)} found):")
    print("-" * 40)
    
    if not docs:
        print("  No documents found.")
        return
    
    for i, doc in enumerate(docs):
        score_text = f" (Score: {scores[i]:.3f})" if scores and i < len(scores) else ""
        print(f"  [{i+1}]{score_text}")
        print(f"    Product: {doc.metadata.get('product_name', 'N/A')}")
        print(f"    Price: ${doc.metadata.get('price', 'N/A')}")
        print(f"    Rating: {doc.metadata.get('rating', 'N/A')}/5")
        print(f"    Review: {doc.page_content[:100]}...")
        print()

def test_1_basic_retrieval(pipeline):
    """Test 1: Simple vector retrieval without any filters."""
    if not TEST_FLAGS["basic_retrieval"]:
        return
    
    print_separator("TEST 1: Basic Vector Retrieval")
    
    query = TEST_QUERIES["basic"]
    print(f"🔍 Query: '{query}'")
    
    # Perform retrieval
    result = pipeline.retrieve(query)
    
    # Print results
    print(f"⏱️  Retrieval Time: {result.latency_ms:.0f}ms")
    print(f"💾 From Cache: {result.from_cache}")
    
    print_documents(result.documents, result.scores, "Retrieved Documents")
    
    return result

def test_2_filtered_retrieval(pipeline):
    """Test 2: Retrieval with metadata filters."""
    if not TEST_FLAGS["filtered_retrieval"]:
        return
    
    print_separator("TEST 2: Filtered Retrieval")
    
    query = TEST_QUERIES["filtered"]
    filters = MetadataFilter(
        min_price=FILTER_SETTINGS["min_price"], 
        max_price=FILTER_SETTINGS["max_price"], 
        min_rating=FILTER_SETTINGS["min_rating"]
    )
    
    print(f"🔍 Query: '{query}'")
    print(f"🔧 Filters:")
    print(f"   - Min Price: ${filters.min_price}")
    print(f"   - Max Price: ${filters.max_price}")
    print(f"   - Min Rating: {filters.min_rating}/5")
    
    # Perform filtered retrieval
    result = pipeline.retrieve(query, filters)
    
    # Print results
    print(f"⏱️  Retrieval Time: {result.latency_ms:.0f}ms")
    print(f"💾 From Cache: {result.from_cache}")
    
    print_documents(result.documents, result.scores, "Filtered Documents")
    
    return result

def test_3_context_compression(pipeline):
    """Test 3: Context compression functionality."""
    if not TEST_FLAGS["context_compression"]:
        return
    
    print_separator("TEST 3: Context Compression")
    
    query = TEST_QUERIES["compression"]
    print(f"🔍 Query: '{query}'")
    
    # Get the vector searcher and context compressor directly
    vector_searcher = pipeline.vector_searcher
    context_compressor = pipeline.context_compressor
    query_processor = pipeline.query_processor
    
    # Step 1: Process query
    processed_query = query_processor.process(query)
    print(f"✅ Query processed and embedded")
    
    # Step 2: Get raw search results (before compression)
    search_result = vector_searcher.search(processed_query.embedding)
    print_documents(search_result.documents, search_result.scores, "Before Compression")
    
    # Step 3: Apply compression
    compression_result = context_compressor.compress(query, search_result.documents)
    print_documents(compression_result.documents, title="After Compression")
    
    print(f"📊 Compression Stats:")
    print(f"   - Input Documents: {len(search_result.documents)}")
    print(f"   - Output Documents: {len(compression_result.documents)}")
    print(f"   - Filtered Out: {compression_result.filtered_count}")
    print(f"   - Compression Ratio: {compression_result.compression_ratio:.2f}")
    
    return compression_result

def test_4_query_rewriting(pipeline):
    """Test 4: Query rewriting functionality."""
    if not TEST_FLAGS["query_rewriting"]:
        return
    
    print_separator("TEST 4: Query Rewriting")
    
    # Use a vague query that should trigger rewriting
    vague_query = TEST_QUERIES["rewriting"]
    print(f"🔍 Original Query: '{vague_query}'")
    
    # Get components
    query_processor = pipeline.query_processor
    vector_searcher = pipeline.vector_searcher
    context_compressor = pipeline.context_compressor
    query_rewriter = pipeline.query_rewriter
    
    # Step 1: Process original query
    processed_query = query_processor.process(vague_query)
    
    # Step 2: Get initial results
    search_result = vector_searcher.search(processed_query.embedding)
    print_documents(search_result.documents[:2], search_result.scores[:2], "Initial Results")
    
    # Step 3: Check compression and relevance
    compression_result = context_compressor.compress(vague_query, search_result.documents)
    
    if compression_result.relevance_scores:
        avg_relevance = sum(compression_result.relevance_scores) / len(compression_result.relevance_scores)
        print(f"📊 Average Relevance Score: {avg_relevance:.3f}")
        
        # Step 4: Check if rewriting is needed
        should_rewrite = query_rewriter.should_rewrite(avg_relevance)
        print(f"🔄 Should Rewrite: {should_rewrite}")
        
        if should_rewrite:
            # Step 5: Perform rewriting
            rewrite_result = query_rewriter.rewrite(
                vague_query, 
                f"Previous search returned {len(compression_result.documents)} documents with average relevance {avg_relevance:.3f}"
            )
            
            print(f"✏️  Rewritten Query: '{rewrite_result.rewritten_query}'")
            print(f"📝 Improvement Reason: {rewrite_result.improvement_reason}")
            
            # Step 6: Test rewritten query
            rewritten_processed = query_processor.process(rewrite_result.rewritten_query)
            rewritten_search = vector_searcher.search(rewritten_processed.embedding)
            
            print_documents(rewritten_search.documents[:2], rewritten_search.scores[:2], "Rewritten Query Results")
            
            return rewrite_result
        else:
            print("✅ Query relevance is good, no rewriting needed")
    else:
        print("⚠️  No relevance scores available")
    
    return None

def main():
    """Main test execution."""
    print("🚀 Simple Retrieval Pipeline Tests")
    print(f"📋 Active Tests: {[k for k, v in TEST_FLAGS.items() if v]}")
    
    # Check environment variables
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("PINECONE_API_KEY"):
        print("❌ Please set OPENAI_API_KEY and PINECONE_API_KEY environment variables")
        return
    
    try:
        # Initialize pipeline
        print("\n⚙️  Initializing pipeline...")
        pipeline = RetrievalPipeline.from_config_file()
        pipeline.initialize()
        print("✅ Pipeline ready!")
        
        # Run tests
        results = {}
        
        if TEST_FLAGS["basic_retrieval"]:
            results["basic"] = test_1_basic_retrieval(pipeline)
        
        if TEST_FLAGS["filtered_retrieval"]:
            results["filtered"] = test_2_filtered_retrieval(pipeline)
        
        if TEST_FLAGS["context_compression"]:
            results["compression"] = test_3_context_compression(pipeline)
        
        if TEST_FLAGS["query_rewriting"]:
            results["rewriting"] = test_4_query_rewriting(pipeline)
        
        print_separator("ALL TESTS COMPLETED ✅")
        print("🎉 All enabled tests completed successfully!")
        
        # Summary
        print("\n📊 Test Summary:")
        for test_name, enabled in TEST_FLAGS.items():
            status = "✅ PASSED" if enabled else "⏭️  SKIPPED"
            print(f"   {test_name}: {status}")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()