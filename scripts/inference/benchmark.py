#!/usr/bin/env python3
"""
Inference Pipeline Benchmark

A focused performance testing script for the inference pipeline.
Measures latency, throughput, and resource usage under different conditions.

USAGE:
    python scripts/inference/benchmark.py
    
    # Custom benchmark parameters
    python scripts/inference/benchmark.py --requests 10 --concurrent 2
    python scripts/inference/benchmark.py --test latency
    python scripts/inference/benchmark.py --test throughput
"""

import os
import sys
import asyncio
import argparse
import time
import statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from pipelines.inference.pipeline import InferencePipeline
from pipelines.retrieval.pipeline import RetrievalPipeline

# Benchmark test queries
BENCHMARK_QUERIES = [
    "What's the best phone for photography?",
    "Compare iPhone vs Samsung Galaxy phones",
    "I need a budget smartphone under $300",
    "Which phone has the longest battery life?",
    "What are the latest phone releases?",
    "Recommend a phone for gaming",
    "Best phones for business use",
    "Phones with good camera and storage",
    "Affordable phones with 5G support",
    "Premium phones worth the price"
]

def print_benchmark_header(test_name, description):
    """Print benchmark test header."""
    print("\n" + "="*70)
    print(f"  📊 {test_name}")
    print("="*70)
    print(f"📝 {description}")
    print("-"*70)

def print_statistics(latencies, test_name):
    """Print detailed statistics for latency measurements."""
    if not latencies:
        print("❌ No data to analyze")
        return
    
    mean_latency = statistics.mean(latencies)
    median_latency = statistics.median(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    
    if len(latencies) > 1:
        stdev_latency = statistics.stdev(latencies)
        p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
        p99_latency = sorted(latencies)[int(0.99 * len(latencies))]
    else:
        stdev_latency = 0
        p95_latency = latencies[0]
        p99_latency = latencies[0]
    
    print(f"\n📊 {test_name} Statistics:")
    print(f"   • Requests: {len(latencies)}")
    print(f"   • Mean latency: {mean_latency:.0f}ms")
    print(f"   • Median latency: {median_latency:.0f}ms")
    print(f"   • Min latency: {min_latency:.0f}ms")
    print(f"   • Max latency: {max_latency:.0f}ms")
    print(f"   • Std deviation: {stdev_latency:.0f}ms")
    print(f"   • 95th percentile: {p95_latency:.0f}ms")
    print(f"   • 99th percentile: {p99_latency:.0f}ms")
    print(f"   • Throughput: {len(latencies) / (sum(latencies) / 1000):.1f} req/sec")

def benchmark_latency(pipeline, num_requests=10):
    """Benchmark single-request latency."""
    print_benchmark_header("Latency Benchmark", 
                          f"Measuring response latency for {num_requests} sequential requests")
    
    session_id = "latency_benchmark"
    pipeline.clear_session(session_id)
    
    latencies = []
    
    print(f"🚀 Running {num_requests} sequential requests...")
    
    for i in range(num_requests):
        query = BENCHMARK_QUERIES[i % len(BENCHMARK_QUERIES)]
        
        start_time = time.time()
        result = pipeline.generate(query, f"{session_id}_{i}")
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
        
        print(f"   Request {i+1:2d}: {latency_ms:6.0f}ms - {query[:40]}...")
    
    print_statistics(latencies, "Latency")
    return latencies

async def benchmark_throughput(pipeline, num_requests=10, concurrent=3):
    """Benchmark concurrent request throughput."""
    print_benchmark_header("Throughput Benchmark", 
                          f"Measuring throughput with {concurrent} concurrent requests ({num_requests} total)")
    
    async def single_request(request_id):
        query = BENCHMARK_QUERIES[request_id % len(BENCHMARK_QUERIES)]
        session_id = f"throughput_benchmark_{request_id}"
        
        start_time = time.time()
        result = await pipeline.agenerate(query, session_id)
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000
        return latency_ms, request_id, query
    
    print(f"🚀 Running {num_requests} requests with {concurrent} concurrent workers...")
    
    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(concurrent)
    
    async def limited_request(request_id):
        async with semaphore:
            return await single_request(request_id)
    
    # Start all requests
    start_time = time.time()
    tasks = [limited_request(i) for i in range(num_requests)]
    results = await asyncio.gather(*tasks)
    total_time = time.time() - start_time
    
    # Process results
    latencies = []
    for latency_ms, request_id, query in results:
        latencies.append(latency_ms)
        print(f"   Request {request_id+1:2d}: {latency_ms:6.0f}ms - {query[:40]}...")
    
    print(f"\n⏱️  Total execution time: {total_time:.1f}s")
    print(f"🔄 Overall throughput: {num_requests / total_time:.1f} req/sec")
    
    print_statistics(latencies, "Concurrent Throughput")
    return latencies

async def benchmark_streaming(pipeline, num_requests=5):
    """Benchmark streaming response performance."""
    print_benchmark_header("Streaming Benchmark", 
                          f"Measuring streaming performance for {num_requests} requests")
    
    session_id = "streaming_benchmark"
    
    streaming_stats = []
    
    print(f"🚀 Running {num_requests} streaming requests...")
    
    for i in range(num_requests):
        query = BENCHMARK_QUERIES[i % len(BENCHMARK_QUERIES)]
        session_id_i = f"{session_id}_{i}"
        
        print(f"\n   Request {i+1}: {query[:50]}...")
        
        start_time = time.time()
        first_chunk_time = None
        chunk_count = 0
        total_chars = 0
        
        async for chunk in pipeline.stream(query, session_id_i):
            if first_chunk_time is None:
                first_chunk_time = time.time()
            chunk_count += 1
            total_chars += len(chunk)
        
        end_time = time.time()
        
        total_latency = (end_time - start_time) * 1000
        first_chunk_latency = (first_chunk_time - start_time) * 1000 if first_chunk_time else 0
        
        stats = {
            'total_latency': total_latency,
            'first_chunk_latency': first_chunk_latency,
            'chunk_count': chunk_count,
            'total_chars': total_chars,
            'chars_per_second': total_chars / (total_latency / 1000) if total_latency > 0 else 0
        }
        
        streaming_stats.append(stats)
        
        print(f"      Total: {total_latency:.0f}ms, First chunk: {first_chunk_latency:.0f}ms")
        print(f"      Chunks: {chunk_count}, Chars: {total_chars}, Speed: {stats['chars_per_second']:.0f} chars/sec")
    
    # Calculate averages
    avg_total = statistics.mean([s['total_latency'] for s in streaming_stats])
    avg_first_chunk = statistics.mean([s['first_chunk_latency'] for s in streaming_stats])
    avg_chunks = statistics.mean([s['chunk_count'] for s in streaming_stats])
    avg_chars = statistics.mean([s['total_chars'] for s in streaming_stats])
    avg_speed = statistics.mean([s['chars_per_second'] for s in streaming_stats])
    
    print(f"\n📊 Streaming Statistics:")
    print(f"   • Requests: {len(streaming_stats)}")
    print(f"   • Avg total latency: {avg_total:.0f}ms")
    print(f"   • Avg first chunk latency: {avg_first_chunk:.0f}ms")
    print(f"   • Avg chunks per response: {avg_chunks:.1f}")
    print(f"   • Avg characters per response: {avg_chars:.0f}")
    print(f"   • Avg streaming speed: {avg_speed:.0f} chars/sec")
    
    return streaming_stats

def benchmark_conversation_memory(pipeline, conversation_length=10):
    """Benchmark conversation memory and context handling."""
    print_benchmark_header("Conversation Memory Benchmark", 
                          f"Testing conversation context with {conversation_length} turns")
    
    session_id = "memory_benchmark"
    pipeline.clear_session(session_id)
    
    latencies = []
    
    print(f"🚀 Running {conversation_length}-turn conversation...")
    
    for turn in range(conversation_length):
        # Create context-dependent queries
        if turn == 0:
            query = "I'm looking for a new smartphone"
        elif turn == 1:
            query = "What about camera quality?"
        elif turn == 2:
            query = "Which one is most affordable?"
        elif turn == 3:
            query = "Tell me more about that phone"
        else:
            query = f"What else should I know? (turn {turn + 1})"
        
        start_time = time.time()
        result = pipeline.generate(query, session_id)
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
        
        # Get current history length
        history = pipeline.get_session_history(session_id)
        
        print(f"   Turn {turn+1:2d}: {latency_ms:6.0f}ms - History: {len(history)} msgs - {query}")
    
    print_statistics(latencies, "Conversation Memory")
    
    # Show final conversation state
    final_history = pipeline.get_session_history(session_id)
    print(f"\n💬 Final conversation state:")
    print(f"   • Total messages: {len(final_history)}")
    print(f"   • Session ID: {session_id}")
    
    return latencies

async def run_comprehensive_benchmark(pipeline, args):
    """Run all benchmark tests."""
    print("🏁 Comprehensive Inference Pipeline Benchmark")
    print("=" * 70)
    
    all_results = {}
    
    # Latency benchmark
    if args.test in ['latency', 'all']:
        all_results['latency'] = benchmark_latency(pipeline, args.requests)
    
    # Throughput benchmark
    if args.test in ['throughput', 'all']:
        all_results['throughput'] = await benchmark_throughput(
            pipeline, args.requests, args.concurrent
        )
    
    # Streaming benchmark
    if args.test in ['streaming', 'all']:
        all_results['streaming'] = await benchmark_streaming(pipeline, min(args.requests, 5))
    
    # Memory benchmark
    if args.test in ['memory', 'all']:
        all_results['memory'] = benchmark_conversation_memory(pipeline, 8)
    
    # Summary
    print_benchmark_header("Benchmark Summary", "Overall performance summary")
    
    for test_name, results in all_results.items():
        if test_name == 'streaming':
            avg_latency = statistics.mean([s['total_latency'] for s in results])
            print(f"   {test_name:12}: {avg_latency:6.0f}ms avg total latency")
        else:
            avg_latency = statistics.mean(results)
            print(f"   {test_name:12}: {avg_latency:6.0f}ms avg latency")
    
    return all_results

async def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(description='Inference Pipeline Benchmark')
    parser.add_argument('--test', choices=['latency', 'throughput', 'streaming', 'memory', 'all'], 
                       default='all', help='Benchmark test to run')
    parser.add_argument('--requests', type=int, default=10, help='Number of requests to test')
    parser.add_argument('--concurrent', type=int, default=3, help='Concurrent requests for throughput test')
    
    args = parser.parse_args()
    
    print("📊 Inference Pipeline Benchmark")
    print("=" * 50)
    
    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        return
    
    try:
        # Initialize pipelines
        print("⚙️  Initializing pipelines...")
        
        retrieval_pipeline = RetrievalPipeline.from_config_file()
        retrieval_pipeline.initialize()
        
        inference_pipeline = InferencePipeline.from_config_file(retrieval_pipeline)
        inference_pipeline.initialize()
        
        print("✅ Pipelines ready!")
        
        # Run benchmarks
        results = await run_comprehensive_benchmark(inference_pipeline, args)
        
        print("\n🎉 Benchmark completed successfully!")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())