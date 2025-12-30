#!/usr/bin/env python3
"""
Quick Inference Pipeline Test

A minimal test script for rapid validation of the inference pipeline.
Perfect for quick checks during development or CI/CD.

USAGE:
    python scripts/inference/quick_test.py
    
    # Test specific functionality
    python scripts/inference/quick_test.py --test basic
    python scripts/inference/quick_test.py --test conversation
    python scripts/inference/quick_test.py --test streaming
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from pipelines.inference.pipeline import InferencePipeline
from pipelines.retrieval.pipeline import RetrievalPipeline

def test_basic_inference(pipeline):
    """Quick basic inference test."""
    print("🔍 Testing basic inference...")
    
    query = "What's the best phone for photography?"
    result = pipeline.generate(query, "quick_test_session")
    
    print(f"✅ Query: {query}")
    print(f"✅ Response: {result.response[:100]}...")
    print(f"✅ Latency: {result.latency_ms:.0f}ms")
    
    return True

def test_conversation(pipeline):
    """Quick conversation test."""
    print("🔍 Testing conversation...")
    
    session_id = "conversation_test"
    pipeline.clear_session(session_id)
    
    queries = [
        "I need a new smartphone",
        "What about camera quality?",
        "Which one is most affordable?"
    ]
    
    for i, query in enumerate(queries, 1):
        result = pipeline.generate(query, session_id)
        print(f"✅ Turn {i}: {result.response[:50]}...")
    
    history = pipeline.get_session_history(session_id)
    print(f"✅ History: {len(history)} messages")
    
    return True

async def test_streaming(pipeline):
    """Quick streaming test."""
    print("🔍 Testing streaming...")
    
    query = "Tell me about iPhone features"
    session_id = "streaming_test"
    
    print("🤖 Response: ", end="")
    chunk_count = 0
    
    async for chunk in pipeline.stream(query, session_id):
        print(chunk, end="", flush=True)
        chunk_count += 1
    
    print(f"\n✅ Streaming: {chunk_count} chunks")
    return True

def test_health_check(pipeline):
    """Quick health check."""
    print("🔍 Testing health check...")
    
    health = pipeline.health_check()
    stats = pipeline.get_pipeline_stats()
    
    print(f"✅ Status: {health['status']}")
    print(f"✅ Components: {len(health['components'])} healthy")
    print(f"✅ Initialized: {stats['initialized']}")
    
    return health['status'] in ['healthy', 'degraded']

async def main():
    """Main test execution."""
    parser = argparse.ArgumentParser(description='Quick Inference Pipeline Test')
    parser.add_argument('--test', choices=['basic', 'conversation', 'streaming', 'health', 'all'], 
                       default='all', help='Test to run')
    args = parser.parse_args()
    
    print("🚀 Quick Inference Pipeline Test")
    
    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set")
        return False
    
    try:
        # Initialize pipelines
        print("⚙️  Initializing pipelines...")
        
        retrieval_pipeline = RetrievalPipeline.from_config_file()
        retrieval_pipeline.initialize()
        
        inference_pipeline = InferencePipeline.from_config_file(retrieval_pipeline)
        inference_pipeline.initialize()
        
        print("✅ Pipelines ready!")
        
        # Run tests
        success = True
        
        if args.test in ['basic', 'all']:
            success &= test_basic_inference(inference_pipeline)
        
        if args.test in ['conversation', 'all']:
            success &= test_conversation(inference_pipeline)
        
        if args.test in ['streaming', 'all']:
            success &= await test_streaming(inference_pipeline)
        
        if args.test in ['health', 'all']:
            success &= test_health_check(inference_pipeline)
        
        if success:
            print("\n🎉 All tests passed!")
        else:
            print("\n❌ Some tests failed!")
        
        return success
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)