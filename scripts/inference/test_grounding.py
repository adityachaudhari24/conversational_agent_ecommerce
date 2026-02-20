#!/usr/bin/env python3
"""Test script to demonstrate and validate RAG grounding behavior.

This script tests the grounding mechanisms to ensure the LLM only uses
retrieved context and doesn't hallucinate product information.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pipelines.inference.pipeline import InferencePipeline
from src.pipelines.retrieval.pipeline import RetrievalPipeline


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def test_grounding():
    """Run grounding tests with various query types."""
    
    print_section("RAG GROUNDING TEST")
    print("This script tests whether the LLM stays grounded in retrieved context")
    print("and doesn't hallucinate product information.\n")
    
    # Initialize pipelines
    print("Initializing pipelines...")
    try:
        retrieval_pipeline = RetrievalPipeline.from_config_file()
        retrieval_pipeline.initialize()
        
        inference_pipeline = InferencePipeline.from_config_file(retrieval_pipeline)
        inference_pipeline.initialize()
        
        print("✓ Pipelines initialized successfully\n")
    except Exception as e:
        print(f"✗ Failed to initialize pipelines: {e}")
        return
    
    # Test cases designed to check grounding
    test_cases = [
        {
            "name": "Valid Product Query",
            "query": "What are the best phones with good cameras?",
            "expected": "Should retrieve and cite specific products from database",
            "check": "response should include product names, prices, ratings from context"
        },
        {
            "name": "Non-Existent Product",
            "query": "Tell me about the SuperMegaPhone X9000 Pro Max Ultra",
            "expected": "Should say it doesn't have information about this product",
            "check": "response should NOT make up specifications or prices"
        },
        {
            "name": "Specific Price Query",
            "query": "Show me phones under $300",
            "expected": "Should retrieve and cite products with actual prices",
            "check": "response should only mention products actually in database"
        },
        {
            "name": "General Question (No Products)",
            "query": "What is a smartphone?",
            "expected": "May provide general information or request more specific query",
            "check": "response should not recommend specific products without context"
        },
        {
            "name": "Comparison Request",
            "query": "Compare iPhone 15 vs Samsung Galaxy S24",
            "expected": "Should only compare if both products are in database",
            "check": "response should not invent specifications for missing products"
        }
    ]
    
    # Run tests
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print_section(f"Test {i}/{len(test_cases)}: {test_case['name']}")
        
        print(f"Query: {test_case['query']}")
        print(f"Expected: {test_case['expected']}")
        print(f"Check: {test_case['check']}\n")
        
        try:
            # Generate response
            result = inference_pipeline.generate(
                query=test_case['query'],
                session_id=f"grounding_test_{i}"
            )
            
            print(f"Response:\n{result.response}\n")
            
            # Show metadata
            print(f"Metadata:")
            print(f"  - Latency: {result.latency_ms:.2f}ms")
            print(f"  - Session: {result.session_id}")
            
            # Check if workflow retrieved context
            workflow_steps = result.metadata.get('workflow_steps', [])
            retrieval_step = next(
                (s for s in workflow_steps if 'retrieval' in s.get('step', '').lower()),
                None
            )
            
            if retrieval_step:
                print(f"  - Retrieved context: Yes")
            else:
                print(f"  - Retrieved context: No (direct response)")
            
            # Manual grounding check
            print(f"\nGrounding Check:")
            response_lower = result.response.lower()
            
            # Check for hallucination indicators
            hallucination_indicators = [
                ("specific price without context", "$" in result.response and not retrieval_step),
                ("rating without context", "rating" in response_lower and not retrieval_step),
                ("specific model numbers", any(x in response_lower for x in ["x9000", "pro max ultra"])),
            ]
            
            has_issues = False
            for indicator, present in hallucination_indicators:
                if present:
                    print(f"  ⚠️  Potential issue: {indicator}")
                    has_issues = True
            
            # Check for good grounding indicators
            good_indicators = [
                ("acknowledges lack of info", any(x in response_lower for x in [
                    "don't have information",
                    "not in my database", 
                    "couldn't find",
                    "no information about"
                ])),
                ("cites specific products", retrieval_step is not None),
            ]
            
            for indicator, present in good_indicators:
                if present:
                    print(f"  ✓ Good: {indicator}")
            
            if not has_issues:
                print(f"  ✓ No obvious grounding issues detected")
                results.append(("PASS", test_case['name']))
            else:
                results.append(("WARN", test_case['name']))
            
        except Exception as e:
            print(f"✗ Error during test: {e}")
            results.append(("FAIL", test_case['name']))
    
    # Summary
    print_section("TEST SUMMARY")
    
    passed = sum(1 for r in results if r[0] == "PASS")
    warned = sum(1 for r in results if r[0] == "WARN")
    failed = sum(1 for r in results if r[0] == "FAIL")
    
    print(f"Total Tests: {len(results)}")
    print(f"  ✓ Passed: {passed}")
    print(f"  ⚠️  Warnings: {warned}")
    print(f"  ✗ Failed: {failed}\n")
    
    for status, name in results:
        symbol = "✓" if status == "PASS" else "⚠️" if status == "WARN" else "✗"
        print(f"  {symbol} {name}: {status}")
    
    print("\n" + "="*80)
    print("\nGROUNDING RECOMMENDATIONS:")
    print("="*80)
    print("""
1. Review responses above for any hallucinated information
2. Verify that non-existent products are handled correctly
3. Check that prices/ratings are only cited when in context
4. Ensure fallback messages appear when appropriate

To improve grounding:
- Increase strict_grounding in config/inference.yaml
- Set require_context: true for product queries
- Improve retrieval quality (better embeddings, query rewriting)
- Monitor and log responses that seem ungrounded

See docs/GROUNDING_GUIDE.md for detailed configuration options.
""")


if __name__ == "__main__":
    test_grounding()
