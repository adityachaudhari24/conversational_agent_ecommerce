#!/usr/bin/env python3
"""
Retrieval Pipeline Demo Script

This script demonstrates the full capabilities of the retrieval pipeline including:
- Vector similarity search
- MMR (Maximal Marginal Relevance) search for diversity
- Contextual compression to filter irrelevant documents
- Query rewriting for improved results
- Result caching
- Metadata filtering
- Error handling and retry logic

Usage:
    python scripts/retrieval/demo_retrieval_pipeline.py
"""
#TODO : redo it
import os
import sys
import time
import json
from typing import Dict, Any, List
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from pipelines.retrieval.pipeline import RetrievalPipeline
from pipelines.retrieval.search.vector_searcher import MetadataFilter
from pipelines.retrieval.config import ConfigurationLoader
from pipelines.retrieval.exceptions import RetrievalError, ConfigurationError


class RetrievalDemo:
    """Demo class for testing retrieval pipeline functionality."""
    
    def __init__(self):
        """Initialize the demo with pipeline setup."""
        self.pipeline = None
        self.setup_pipeline()
    
    def setup_pipeline(self):
        """Set up the retrieval pipeline."""
        print("🚀 Setting up Retrieval Pipeline...")
        
        try:
            # Load configuration
            config_loader = ConfigurationLoader("config/retrieval.yaml")
            settings = config_loader.load_config()
            
            # Create pipeline
            self.pipeline = RetrievalPipeline.from_settings(settings)
            
            # Initialize pipeline
            print("⚙️  Initializing pipeline components...")
            self.pipeline.initialize()
            
            print("✅ Pipeline setup complete!")
            print()
            
        except ConfigurationError as e:
            print(f"❌ Configuration error: {e}")
            print("\n💡 Make sure you have:")
            print("   - OPENAI_API_KEY environment variable set")
            print("   - PINECONE_API_KEY environment variable set")
            print("   - Pinecone index populated with data")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            sys.exit(1)
    
    def print_separator(self, title: str):
        """Print a section separator."""
        print("\n" + "="*60)
        print(f"  {title}")
        print("="*60)
    
    def print_result(self, query: str, result, test_name: str):
        """Print retrieval result in a formatted way."""
        print(f"\n📝 Query: '{query}'")
        print(f"🔍 Test: {test_name}")
        print(f"⏱️  Latency: {result.latency_ms:.2f}ms")
        print(f"💾 From Cache: {'Yes' if result.from_cache else 'No'}")
        print(f"📄 Documents Found: {len(result.documents)}")
        
        if result.documents:
            print(f"📊 Score Range: {min(result.scores):.3f} - {max(result.scores):.3f}")
            
            # Show metadata insights
            metadata = result.metadata
            if 'rewrite_attempts' in metadata:
                print(f"✏️  Query Rewrites: {metadata['rewrite_attempts']}")
            if 'compression_applied' in metadata:
                print(f"🗜️  Compression Applied: {'Yes' if metadata['compression_applied'] else 'No'}")
            
            # Show first document as example
            print(f"\n📱 Top Result:")
            doc = result.documents[0]
            print(f"   Product: {doc.metadata.get('product_name', 'N/A')}")
            print(f"   Price: ${doc.metadata.get('price', 'N/A')}")
            print(f"   Rating: {doc.metadata.get('rating', 'N/A')}/5")
            print(f"   Review: {doc.page_content[:100]}...")
            
            # Show formatted context preview
            print(f"\n📋 Formatted Context Preview:")
            preview = result.formatted_context[:200] + "..." if len(result.formatted_context) > 200 else result.formatted_context
            print(f"   {preview}")
        else:
            print("   No documents found")
        
        print("-" * 40)
    
    def demo_basic_search(self):
        """Demo basic vector similarity search."""
        self.print_separator("1. BASIC VECTOR SIMILARITY SEARCH")
        
        queries = [
            "iPhone with good camera quality",
            "cheap smartphone under $200",
            "phone with long battery life",
            "iPhone 12 reviews"
        ]
        
        for query in queries:
            try:
                result = self.pipeline.retrieve(query)
                self.print_result(query, result, "Basic Similarity Search")
                time.sleep(1)  # Brief pause between queries
            except Exception as e:
                print(f"❌ Error with query '{query}': {e}")
    
    def demo_metadata_filtering(self):
        """Demo metadata filtering capabilities."""
        self.print_separator("2. METADATA FILTERING")
        
        test_cases = [
            {
                "query": "iPhone reviews",
                "filters": MetadataFilter(min_price=400.0),
                "description": "Premium phones (>$400)"
            },
            {
                "query": "smartphone",
                "filters": MetadataFilter(max_price=300.0),
                "description": "Budget phones (<$300)"
            },
            {
                "query": "phone reviews",
                "filters": MetadataFilter(min_rating=4.0),
                "description": "Highly rated phones (4+ stars)"
            },
            {
                "query": "iPhone",
                "filters": MetadataFilter(min_price=200.0, max_price=500.0, min_rating=3.0),
                "description": "Mid-range iPhones ($200-500, 3+ stars)"
            }
        ]
        
        for case in test_cases:
            try:
                result = self.pipeline.retrieve(case["query"], case["filters"])
                self.print_result(
                    f"{case['query']} (Filter: {case['description']})", 
                    result, 
                    "Metadata Filtering"
                )
                time.sleep(1)
            except Exception as e:
                print(f"❌ Error with filtered query: {e}")
    
    def demo_query_rewriting(self):
        """Demo query rewriting for vague queries."""
        self.print_separator("3. QUERY REWRITING FOR VAGUE QUERIES")
        
        # These queries are intentionally vague to trigger rewriting
        vague_queries = [
            "something good",  # Very vague
            "what should I buy",  # Question format
            "recommendations please",  # No specific product mentioned
            "best option available",  # Generic request
            "help me choose"  # Vague request
        ]
        
        print("🔄 Testing with intentionally vague queries to trigger rewriting...")
        
        for query in vague_queries:
            try:
                result = self.pipeline.retrieve(query)
                self.print_result(query, result, "Query Rewriting Test")
                
                # Show rewrite details if available
                if result.metadata.get('rewrite_attempts', 0) > 0:
                    print(f"   ✏️  Original query was rewritten {result.metadata['rewrite_attempts']} times")
                    for step in result.metadata.get('workflow_steps', []):
                        if step.get('step') == 'query_rewriting':
                            print(f"   📝 Improvement: {step.get('improvement_reason', 'N/A')}")
                
                time.sleep(1)
            except Exception as e:
                print(f"❌ Error with vague query '{query}': {e}")
    
    def demo_contextual_compression(self):
        """Demo contextual compression filtering."""
        self.print_separator("4. CONTEXTUAL COMPRESSION")
        
        # Queries that might return some irrelevant results
        compression_queries = [
            "iPhone camera features",  # Specific feature query
            "battery problems with phones",  # Problem-focused query
            "screen quality comparison",  # Comparison query
            "value for money smartphone"  # Subjective query
        ]
        
        print("🗜️  Testing contextual compression to filter irrelevant documents...")
        
        for query in compression_queries:
            try:
                result = self.pipeline.retrieve(query)
                self.print_result(query, result, "Contextual Compression")
                
                # Show compression details
                for step in result.metadata.get('workflow_steps', []):
                    if step.get('step') == 'contextual_compression':
                        input_docs = step.get('input_documents', 0)
                        output_docs = step.get('output_documents', 0)
                        ratio = step.get('compression_ratio', 1.0)
                        print(f"   🗜️  Compression: {input_docs} → {output_docs} docs (ratio: {ratio:.2f})")
                
                time.sleep(1)
            except Exception as e:
                print(f"❌ Error with compression query '{query}': {e}")
    
    def demo_caching(self):
        """Demo result caching functionality."""
        self.print_separator("5. RESULT CACHING")
        
        test_query = "iPhone 12 Pro reviews"
        
        print(f"🔄 Testing cache with query: '{test_query}'")
        
        # First request (cache miss)
        print("\n1️⃣  First request (should be cache miss):")
        start_time = time.time()
        result1 = self.pipeline.retrieve(test_query)
        first_time = time.time() - start_time
        self.print_result(test_query, result1, "Cache Miss")
        
        # Second request (cache hit)
        print("\n2️⃣  Second request (should be cache hit):")
        start_time = time.time()
        result2 = self.pipeline.retrieve(test_query)
        second_time = time.time() - start_time
        self.print_result(test_query, result2, "Cache Hit")
        
        # Performance comparison
        speedup = first_time / second_time if second_time > 0 else float('inf')
        print(f"\n⚡ Performance Comparison:")
        print(f"   First request: {first_time*1000:.2f}ms")
        print(f"   Second request: {second_time*1000:.2f}ms")
        print(f"   Speedup: {speedup:.1f}x faster")
        
        # Third request with different filter (cache miss)
        print("\n3️⃣  Same query with filter (should be cache miss):")
        filters = MetadataFilter(min_rating=4.0)
        result3 = self.pipeline.retrieve(test_query, filters)
        self.print_result(f"{test_query} (with filter)", result3, "Cache Miss (Different Filter)")
    
    def demo_search_types(self):
        """Demo different search types (similarity vs MMR)."""
        self.print_separator("6. SEARCH TYPE COMPARISON")
        
        query = "iPhone camera and battery reviews"
        
        print(f"📊 Comparing search types with query: '{query}'")
        print("   Current pipeline uses MMR search for diversity")
        
        try:
            result = self.pipeline.retrieve(query)
            self.print_result(query, result, "MMR Search (Diversity)")
            
            # Show diversity insights
            if len(result.documents) > 1:
                products = [doc.metadata.get('product_name', 'Unknown') for doc in result.documents]
                unique_products = len(set(products))
                print(f"   🎯 Product Diversity: {unique_products} unique products out of {len(products)} results")
                
                # Show product distribution
                from collections import Counter
                product_counts = Counter(products)
                print(f"   📊 Product Distribution:")
                for product, count in product_counts.most_common(3):
                    print(f"      - {product}: {count} reviews")
            
        except Exception as e:
            print(f"❌ Error with search type demo: {e}")
    
    def demo_error_handling(self):
        """Demo error handling and retry logic."""
        self.print_separator("7. ERROR HANDLING & RETRY LOGIC")
        
        print("🛡️  Testing error handling capabilities...")
        
        # Test with empty query (should fail gracefully)
        print("\n1️⃣  Testing empty query handling:")
        try:
            result = self.pipeline.retrieve("")
            print("   ❌ Empty query should have failed!")
        except Exception as e:
            print(f"   ✅ Correctly handled empty query: {type(e).__name__}")
        
        # Test with very long query
        print("\n2️⃣  Testing very long query (truncation):")
        long_query = "iPhone " * 200  # Very long query
        try:
            result = self.pipeline.retrieve(long_query)
            if result.metadata.get('workflow_steps'):
                for step in result.metadata['workflow_steps']:
                    if step.get('step') == 'query_processing' and step.get('query_truncated'):
                        print("   ✅ Long query was properly truncated")
                        break
        except Exception as e:
            print(f"   ⚠️  Long query error: {e}")
        
        # Test pipeline health
        print("\n3️⃣  Testing pipeline health check:")
        health = self.pipeline.health_check()
        print(f"   Pipeline Status: {health['status']}")
        if health['errors']:
            print(f"   Errors: {health['errors']}")
        else:
            print("   ✅ All components healthy")
    
    def demo_pipeline_stats(self):
        """Demo pipeline statistics and monitoring."""
        self.print_separator("8. PIPELINE STATISTICS")
        
        print("📊 Pipeline Statistics and Health Information:")
        
        # Get comprehensive stats
        stats = self.pipeline.get_pipeline_stats()
        
        print(f"\n🔧 Configuration:")
        config = stats.get('configuration', {})
        for key, value in config.items():
            print(f"   {key}: {value}")
        
        print(f"\n🏗️  Components Status:")
        components = stats.get('components', {})
        for component, status in components.items():
            if isinstance(status, dict):
                if 'error' in status:
                    print(f"   ❌ {component}: {status['error']}")
                else:
                    print(f"   ✅ {component}: Active")
                    # Show some component-specific stats
                    if component == 'cache' and 'size' in status:
                        print(f"      Cache size: {status['size']}/{status.get('max_size', 'N/A')}")
                        print(f"      Total hits: {status.get('total_hits', 0)}")
            else:
                print(f"   ℹ️  {component}: {status}")
        
        # Show cache statistics in detail
        if self.pipeline.cache:
            cache_stats = self.pipeline.cache.get_stats()
            print(f"\n💾 Cache Details:")
            print(f"   Enabled: {cache_stats['enabled']}")
            print(f"   Current size: {cache_stats['size']}")
            print(f"   Max size: {cache_stats['max_size']}")
            print(f"   TTL: {cache_stats['ttl_seconds']}s")
            print(f"   Total hits: {cache_stats['total_hits']}")
    
    def run_all_demos(self):
        """Run all demo scenarios."""
        print("🎯 RETRIEVAL PIPELINE COMPREHENSIVE DEMO")
        print("=" * 60)
        print("This demo will test all major features of the retrieval pipeline:")
        print("• Vector similarity search")
        print("• Metadata filtering")
        print("• Query rewriting")
        print("• Contextual compression")
        print("• Result caching")
        print("• Error handling")
        print("• Performance monitoring")
        
        try:
            self.demo_basic_search()
            self.demo_metadata_filtering()
            self.demo_query_rewriting()
            self.demo_contextual_compression()
            self.demo_caching()
            self.demo_search_types()
            self.demo_error_handling()
            self.demo_pipeline_stats()
            
            self.print_separator("🎉 DEMO COMPLETE!")
            print("All retrieval pipeline features have been demonstrated.")
            print("Check the logs for detailed performance metrics.")
            
        except KeyboardInterrupt:
            print("\n\n⏹️  Demo interrupted by user")
        except Exception as e:
            print(f"\n\n❌ Demo failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    def interactive_mode(self):
        """Run interactive query mode."""
        self.print_separator("🎮 INTERACTIVE MODE")
        print("Enter queries to test the retrieval pipeline.")
        print("Type 'quit' to exit, 'help' for commands.")
        
        while True:
            try:
                query = input("\n🔍 Enter query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                elif query.lower() == 'help':
                    print("\nAvailable commands:")
                    print("  help - Show this help")
                    print("  stats - Show pipeline statistics")
                    print("  clear - Clear cache")
                    print("  health - Show health status")
                    print("  quit - Exit interactive mode")
                    continue
                elif query.lower() == 'stats':
                    self.demo_pipeline_stats()
                    continue
                elif query.lower() == 'clear':
                    self.pipeline.clear_cache()
                    print("✅ Cache cleared")
                    continue
                elif query.lower() == 'health':
                    health = self.pipeline.health_check()
                    print(f"Pipeline Status: {health['status']}")
                    continue
                elif not query:
                    continue
                
                # Process the query
                start_time = time.time()
                result = self.pipeline.retrieve(query)
                
                self.print_result(query, result, "Interactive Query")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n👋 Goodbye!")


def main():
    """Main function to run the demo."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Retrieval Pipeline Demo")
    parser.add_argument(
        "--mode", 
        choices=["demo", "interactive"], 
        default="demo",
        help="Run mode: 'demo' for full demo, 'interactive' for interactive queries"
    )
    
    args = parser.parse_args()
    
    # Check environment variables
    required_env_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n💡 Please set these environment variables and try again.")
        sys.exit(1)
    
    # Create and run demo
    demo = RetrievalDemo()
    
    if args.mode == "interactive":
        demo.interactive_mode()
    else:
        demo.run_all_demos()


if __name__ == "__main__":
    main()