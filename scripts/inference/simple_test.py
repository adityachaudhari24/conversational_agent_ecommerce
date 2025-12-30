#!/usr/bin/env python3
"""
Simple Inference Pipeline Test Script

A clean, step-by-step test script for the inference pipeline with individual test flags.
Each test can be run independently to understand specific components.

USAGE:
    python scripts/inference/simple_test.py

CUSTOMIZATION:
    Edit the TEST_FLAGS dictionary below to enable/disable specific tests:
    - Set to True to run the test
    - Set to False to skip the test
    
    Example: To run only basic inference and conversation tests:
    TEST_FLAGS = {
        "basic_inference": True,
        "conversation_history": True,
        "agentic_workflow": False,
        "streaming_response": False,
    }
"""

import os
import sys
import asyncio
import time
from pathlib import Path
from typing import Optional, Dict, Any

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from pipelines.inference.pipeline import InferencePipeline
from pipelines.retrieval.pipeline import RetrievalPipeline

# ============================================================================
# TEST CONFIGURATION - Edit these flags to control which tests to run
# ============================================================================

TEST_FLAGS = {
    "basic_inference": True,      # Test 1: Simple query-response without history
    "conversation_history": True, # Test 2: Multi-turn conversation with history
    "agentic_workflow": True,     # Test 3: Test different workflow routing paths
    "streaming_response": False,  # Test 4: Streaming response generation
    "error_handling": False,      # Test 5: Error handling and recovery
    "performance_test": False,    # Test 6: Performance and latency testing
}

# Test Queries - Customize these if needed
TEST_QUERIES = {
    "basic": "What are the best phones under $500?",
    "conversation": [
        "I'm looking for a new smartphone",
        "What about phones with good cameras?", 
        "Which one has the best battery life?",
        "What's the price range for these phones?"
    ],
    "routing": {
        "product_query": "Compare iPhone 14 vs Samsung Galaxy S23",
        "tool_query": "compare iPhone vs Samsung features",
        "general_query": "Hello, how are you today?"
    },
    "streaming": "Tell me about the latest iPhone models and their features",
    "error": "This is a test query for error handling",
    "performance": "Recommend phones for photography enthusiasts"
}

# Session IDs for testing
TEST_SESSIONS = {
    "basic": "test_basic_session",
    "conversation": "test_conversation_session", 
    "routing": "test_routing_session",
    "streaming": "test_streaming_session",
    "error": "test_error_session",
    "performance": "test_performance_session"
}

# ============================================================================

def print_separator(title: str):
    """Print a clean separator for test sections."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def print_inference_result(result, title="Inference Result"):
    """Print inference result in a clean, readable format."""
    print(f"\n📋 {title}:")
    print("-" * 50)
    print(f"🔍 Query: {result.query}")
    print(f"🤖 Response: {result.response}")
    print(f"📊 Session ID: {result.session_id}")
    print(f"⏱️  Latency: {result.latency_ms:.0f}ms")
    print(f"🔢 Tokens Used: {result.tokens_used}")
    print(f"🕒 Timestamp: {result.timestamp.strftime('%H:%M:%S')}")
    
    if result.metadata:
        print(f"📈 Metadata:")
        for key, value in result.metadata.items():
            if key == 'workflow_steps' and isinstance(value, list):
                print(f"   - {key}:")
                for step in value:
                    print(f"     • {step}")
            else:
                print(f"   - {key}: {value}")

def print_conversation_history(pipeline, session_id: str):
    """Print conversation history for a session."""
    try:
        history = pipeline.get_session_history(session_id)
        print(f"\n💬 Conversation History ({len(history)} messages):")
        print("-" * 40)
        
        if not history:
            print("  No messages in history.")
            return
        
        for i, message in enumerate(history, 1):
            role_emoji = "👤" if message.role == "user" else "🤖"
            print(f"  [{i}] {role_emoji} {message.role.title()}: {message.content}")
            print(f"      ⏰ {message.timestamp.strftime('%H:%M:%S')}")
            print()
            
    except Exception as e:
        print(f"❌ Error getting conversation history: {e}")

def test_1_basic_inference(pipeline):
    """Test 1: Simple query-response without conversation history."""
    if not TEST_FLAGS["basic_inference"]:
        return
    
    print_separator("TEST 1: Basic Inference")
    
    query = TEST_QUERIES["basic"]
    session_id = TEST_SESSIONS["basic"]
    
    print(f"🔍 Testing basic inference with query: '{query}'")
    print(f"📋 Session ID: {session_id}")
    
    try:
        # Clear any existing session history
        pipeline.clear_session(session_id)
        
        # Perform inference
        start_time = time.time()
        result = pipeline.generate(query, session_id)
        end_time = time.time()
        
        # Print results
        print_inference_result(result, "Basic Inference Result")
        
        # Show conversation history
        print_conversation_history(pipeline, session_id)
        
        print(f"✅ Basic inference completed successfully!")
        return result
        
    except Exception as e:
        print(f"❌ Basic inference failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_2_conversation_history(pipeline):
    """Test 2: Multi-turn conversation with history management."""
    if not TEST_FLAGS["conversation_history"]:
        return
    
    print_separator("TEST 2: Conversation History")
    
    queries = TEST_QUERIES["conversation"]
    session_id = TEST_SESSIONS["conversation"]
    
    print(f"🔍 Testing multi-turn conversation with {len(queries)} queries")
    print(f"📋 Session ID: {session_id}")
    
    try:
        # Clear session to start fresh
        pipeline.clear_session(session_id)
        
        results = []
        
        for i, query in enumerate(queries, 1):
            print(f"\n--- Turn {i} ---")
            print(f"👤 User: {query}")
            
            # Perform inference
            result = pipeline.generate(query, session_id)
            results.append(result)
            
            print(f"🤖 Assistant: {result.response}")
            print(f"⏱️  Response time: {result.latency_ms:.0f}ms")
        
        # Show final conversation history
        print_conversation_history(pipeline, session_id)
        
        print(f"✅ Conversation history test completed successfully!")
        return results
        
    except Exception as e:
        print(f"❌ Conversation history test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_3_agentic_workflow(pipeline):
    """Test 3: Test different workflow routing paths."""
    if not TEST_FLAGS["agentic_workflow"]:
        return
    
    print_separator("TEST 3: Agentic Workflow Routing")
    
    routing_queries = TEST_QUERIES["routing"]
    session_id = TEST_SESSIONS["routing"]
    
    print(f"🔍 Testing agentic workflow routing with different query types")
    print(f"📋 Session ID: {session_id}")
    
    try:
        # Clear session to start fresh
        pipeline.clear_session(session_id)
        
        results = {}
        
        for query_type, query in routing_queries.items():
            print(f"\n--- {query_type.replace('_', ' ').title()} ---")
            print(f"🔍 Query: {query}")
            
            # Perform inference
            result = pipeline.generate(query, session_id)
            results[query_type] = result
            
            print(f"🤖 Response: {result.response[:200]}...")
            print(f"⏱️  Response time: {result.latency_ms:.0f}ms")
            
            # Show routing information if available in metadata
            if 'workflow_steps' in result.metadata:
                print(f"🔄 Workflow steps:")
                for step in result.metadata['workflow_steps']:
                    print(f"   • {step}")
        
        # Show final conversation history
        print_conversation_history(pipeline, session_id)
        
        print(f"✅ Agentic workflow test completed successfully!")
        return results
        
    except Exception as e:
        print(f"❌ Agentic workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_4_streaming_response(pipeline):
    """Test 4: Streaming response generation."""
    if not TEST_FLAGS["streaming_response"]:
        return
    
    print_separator("TEST 4: Streaming Response")
    
    query = TEST_QUERIES["streaming"]
    session_id = TEST_SESSIONS["streaming"]
    
    print(f"🔍 Testing streaming response with query: '{query}'")
    print(f"📋 Session ID: {session_id}")
    
    try:
        # Clear session to start fresh
        pipeline.clear_session(session_id)
        
        print(f"🤖 Streaming response:")
        print("-" * 40)
        
        # Stream the response
        start_time = time.time()
        complete_response = ""
        chunk_count = 0
        
        async for chunk in pipeline.stream(query, session_id):
            complete_response += chunk
            chunk_count += 1
            print(chunk, end="", flush=True)
        
        end_time = time.time()
        
        print(f"\n-" * 40)
        print(f"✅ Streaming completed!")
        print(f"⏱️  Total time: {(end_time - start_time) * 1000:.0f}ms")
        print(f"📦 Chunks received: {chunk_count}")
        print(f"📝 Total response length: {len(complete_response)} characters")
        
        # Show conversation history
        print_conversation_history(pipeline, session_id)
        
        return {
            "response": complete_response,
            "chunk_count": chunk_count,
            "latency_ms": (end_time - start_time) * 1000
        }
        
    except Exception as e:
        print(f"❌ Streaming test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_5_error_handling(pipeline):
    """Test 5: Error handling and recovery."""
    if not TEST_FLAGS["error_handling"]:
        return
    
    print_separator("TEST 5: Error Handling")
    
    session_id = TEST_SESSIONS["error"]
    
    print(f"🔍 Testing error handling scenarios")
    print(f"📋 Session ID: {session_id}")
    
    try:
        # Clear session to start fresh
        pipeline.clear_session(session_id)
        
        test_scenarios = [
            ("Empty query", ""),
            ("Very long query", "x" * 10000),
            ("Invalid session", None),
        ]
        
        results = {}
        
        for scenario_name, test_input in test_scenarios:
            print(f"\n--- {scenario_name} ---")
            
            try:
                if scenario_name == "Invalid session":
                    result = pipeline.generate("Test query", test_input)
                else:
                    result = pipeline.generate(test_input, session_id)
                
                print(f"✅ Handled gracefully: {result.response[:100]}...")
                results[scenario_name] = "handled"
                
            except Exception as e:
                print(f"⚠️  Expected error caught: {type(e).__name__}: {e}")
                results[scenario_name] = f"error: {type(e).__name__}"
        
        print(f"\n📊 Error handling summary:")
        for scenario, result in results.items():
            print(f"   • {scenario}: {result}")
        
        print(f"✅ Error handling test completed!")
        return results
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_6_performance_test(pipeline):
    """Test 6: Performance and latency testing."""
    if not TEST_FLAGS["performance_test"]:
        return
    
    print_separator("TEST 6: Performance Testing")
    
    query = TEST_QUERIES["performance"]
    session_id = TEST_SESSIONS["performance"]
    
    print(f"🔍 Testing performance with multiple requests")
    print(f"📋 Session ID: {session_id}")
    
    try:
        # Clear session to start fresh
        pipeline.clear_session(session_id)
        
        num_requests = 3
        latencies = []
        
        print(f"🚀 Running {num_requests} requests...")
        
        for i in range(num_requests):
            print(f"\n--- Request {i+1}/{num_requests} ---")
            
            start_time = time.time()
            result = pipeline.generate(f"{query} (request {i+1})", session_id)
            end_time = time.time()
            
            latency = (end_time - start_time) * 1000
            latencies.append(latency)
            
            print(f"⏱️  Request {i+1} latency: {latency:.0f}ms")
            print(f"📝 Response length: {len(result.response)} characters")
        
        # Calculate statistics
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        print(f"\n📊 Performance Summary:")
        print(f"   • Average latency: {avg_latency:.0f}ms")
        print(f"   • Min latency: {min_latency:.0f}ms")
        print(f"   • Max latency: {max_latency:.0f}ms")
        print(f"   • Total requests: {num_requests}")
        
        # Show conversation history
        print_conversation_history(pipeline, session_id)
        
        print(f"✅ Performance test completed!")
        return {
            "latencies": latencies,
            "avg_latency": avg_latency,
            "min_latency": min_latency,
            "max_latency": max_latency
        }
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def print_pipeline_health(pipeline):
    """Print pipeline health and statistics."""
    print_separator("PIPELINE HEALTH CHECK")
    
    try:
        # Get health status
        health = pipeline.health_check()
        print(f"🏥 Overall Status: {health['status'].upper()}")
        print(f"🕒 Timestamp: {health['timestamp']}")
        
        if health['errors']:
            print(f"❌ Errors:")
            for error in health['errors']:
                print(f"   • {error}")
        
        print(f"\n🔧 Component Status:")
        for component, status in health['components'].items():
            status_emoji = "✅" if status == "healthy" else "❌" if "error" in str(status) else "⚠️"
            print(f"   {status_emoji} {component}: {status}")
        
        # Get pipeline statistics
        stats = pipeline.get_pipeline_stats()
        print(f"\n📊 Pipeline Statistics:")
        print(f"   • Initialized: {stats['initialized']}")
        print(f"   • Streaming enabled: {stats['configuration']['enable_streaming']}")
        print(f"   • Max retries: {stats['configuration']['max_retries']}")
        print(f"   • Timeout: {stats['configuration']['timeout_seconds']}s")
        print(f"   • LLM model: {stats['configuration']['llm_model']}")
        print(f"   • LLM provider: {stats['configuration']['llm_provider']}")
        
        if 'conversation_manager' in stats['components']:
            cm_stats = stats['components']['conversation_manager']
            print(f"   • Active sessions: {cm_stats.get('active_sessions', 0)}")
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")

async def main():
    """Main test execution."""
    print("🚀 Simple Inference Pipeline Tests")
    print(f"📋 Active Tests: {[k for k, v in TEST_FLAGS.items() if v]}")
    
    # Check environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        return
    
    try:
        # Initialize retrieval pipeline first (required for inference)
        print("\n⚙️  Initializing retrieval pipeline...")
        retrieval_pipeline = RetrievalPipeline.from_config_file()
        retrieval_pipeline.initialize()
        print("✅ Retrieval pipeline ready!")
        
        # Initialize inference pipeline
        print("\n⚙️  Initializing inference pipeline...")
        inference_pipeline = InferencePipeline.from_config_file(retrieval_pipeline)
        inference_pipeline.initialize()
        print("✅ Inference pipeline ready!")
        
        # Print pipeline health
        print_pipeline_health(inference_pipeline)
        
        # Run tests
        results = {}
        
        if TEST_FLAGS["basic_inference"]:
            results["basic"] = test_1_basic_inference(inference_pipeline)
        
        if TEST_FLAGS["conversation_history"]:
            results["conversation"] = test_2_conversation_history(inference_pipeline)
        
        if TEST_FLAGS["agentic_workflow"]:
            results["routing"] = test_3_agentic_workflow(inference_pipeline)
        
        if TEST_FLAGS["streaming_response"]:
            results["streaming"] = await test_4_streaming_response(inference_pipeline)
        
        if TEST_FLAGS["error_handling"]:
            results["error_handling"] = test_5_error_handling(inference_pipeline)
        
        if TEST_FLAGS["performance_test"]:
            results["performance"] = test_6_performance_test(inference_pipeline)
        
        print_separator("ALL TESTS COMPLETED ✅")
        print("🎉 All enabled tests completed!")
        
        # Summary
        print("\n📊 Test Summary:")
        for test_name, enabled in TEST_FLAGS.items():
            if enabled:
                result_status = "✅ PASSED" if results.get(test_name.split('_')[0]) else "❌ FAILED"
                print(f"   {test_name}: {result_status}")
            else:
                print(f"   {test_name}: ⏭️  SKIPPED")
        
        # Final health check
        print_pipeline_health(inference_pipeline)
        
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run async main function
    asyncio.run(main())