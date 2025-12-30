#!/usr/bin/env python3
"""
Inference Pipeline Demo

An interactive demo showcasing the inference pipeline capabilities with realistic
e-commerce scenarios. Perfect for demonstrations and understanding the system.

USAGE:
    python scripts/inference/demo.py
    
    # Run specific demo scenario
    python scripts/inference/demo.py --scenario shopping
    python scripts/inference/demo.py --scenario comparison
    python scripts/inference/demo.py --scenario support
"""

import os
import sys
import asyncio
import argparse
import time
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from pipelines.inference.pipeline import InferencePipeline
from pipelines.retrieval.pipeline import RetrievalPipeline

# Demo scenarios with realistic conversations
DEMO_SCENARIOS = {
    "shopping": {
        "title": "🛒 Smart Shopping Assistant",
        "description": "A customer looking for a new smartphone with specific needs",
        "conversation": [
            "Hi! I'm looking for a new smartphone. Can you help me?",
            "I mainly use my phone for photography and social media. What would you recommend?",
            "What about phones under $600? I'm on a budget.",
            "Which of these has the best camera quality?",
            "Can you tell me more about the iPhone 13 vs Samsung Galaxy S22?",
            "What's the battery life like on these phones?",
            "Thanks! I think I'll go with the iPhone 13. Where can I buy it?"
        ]
    },
    "comparison": {
        "title": "⚖️ Product Comparison Expert", 
        "description": "A customer comparing different phone models",
        "conversation": [
            "I need to compare iPhone 14 Pro vs Samsung Galaxy S23 Ultra",
            "Which one has better camera features?",
            "What about performance and gaming?",
            "How do they compare in terms of battery life?",
            "Which one offers better value for money?",
            "What are the main pros and cons of each?",
            "Based on everything, which would you recommend for a photographer?"
        ]
    },
    "support": {
        "title": "🎧 Customer Support Assistant",
        "description": "A customer with questions about their phone purchase",
        "conversation": [
            "I bought a phone last week but I'm having some issues",
            "The battery seems to drain very quickly. Is this normal?",
            "What are some tips to improve battery life?",
            "Also, the camera quality doesn't seem as good as advertised",
            "Are there any camera settings I should adjust?",
            "How do I enable night mode for better low-light photos?",
            "Thank you! This has been very helpful."
        ]
    },
    "discovery": {
        "title": "🔍 Product Discovery Guide",
        "description": "A customer exploring options without specific requirements",
        "conversation": [
            "I'm not sure what phone to get. Can you show me what's popular?",
            "What are the latest phone releases this year?",
            "I'm interested in phones with good reviews. What do you recommend?",
            "What makes these phones special compared to older models?",
            "Are there any upcoming releases I should wait for?",
            "What's the price range for these recommended phones?",
            "Which brands are most reliable for long-term use?"
        ]
    }
}

def print_demo_header(scenario_info):
    """Print demo scenario header."""
    print("\n" + "="*80)
    print(f"  {scenario_info['title']}")
    print("="*80)
    print(f"📝 Scenario: {scenario_info['description']}")
    print("-"*80)

def print_conversation_turn(turn_num, total_turns, query, response, latency_ms):
    """Print a conversation turn in demo format."""
    print(f"\n💬 Turn {turn_num}/{total_turns}")
    print("-" * 50)
    print(f"👤 Customer: {query}")
    print(f"🤖 Assistant: {response}")
    print(f"⏱️  Response time: {latency_ms:.0f}ms")
    
    # Add a small delay for demo effect
    time.sleep(1)

async def run_demo_scenario(pipeline, scenario_name, scenario_info, use_streaming=False):
    """Run a complete demo scenario."""
    print_demo_header(scenario_info)
    
    session_id = f"demo_{scenario_name}"
    pipeline.clear_session(session_id)
    
    conversation = scenario_info["conversation"]
    
    print(f"\n🎬 Starting {scenario_info['title']} demo...")
    print(f"📊 {len(conversation)} conversation turns")
    
    if use_streaming:
        print("🌊 Using streaming responses for real-time experience")
    
    input("\n▶️  Press Enter to start the demo...")
    
    for turn_num, query in enumerate(conversation, 1):
        print(f"\n💬 Turn {turn_num}/{len(conversation)}")
        print("-" * 50)
        print(f"👤 Customer: {query}")
        
        if use_streaming:
            print(f"🤖 Assistant: ", end="", flush=True)
            
            start_time = time.time()
            complete_response = ""
            
            async for chunk in pipeline.stream(query, session_id):
                complete_response += chunk
                print(chunk, end="", flush=True)
            
            latency_ms = (time.time() - start_time) * 1000
            print(f"\n⏱️  Response time: {latency_ms:.0f}ms")
            
        else:
            start_time = time.time()
            result = pipeline.generate(query, session_id)
            latency_ms = (time.time() - start_time) * 1000
            
            print(f"🤖 Assistant: {result.response}")
            print(f"⏱️  Response time: {latency_ms:.0f}ms")
        
        # Pause between turns for demo effect
        if turn_num < len(conversation):
            time.sleep(2)
    
    # Show conversation summary
    history = pipeline.get_session_history(session_id)
    print(f"\n📊 Demo Summary:")
    print(f"   • Total turns: {len(conversation)}")
    print(f"   • Messages in history: {len(history)}")
    print(f"   • Session ID: {session_id}")
    
    print(f"\n✅ {scenario_info['title']} demo completed!")

def show_scenario_menu():
    """Show available demo scenarios."""
    print("\n🎭 Available Demo Scenarios:")
    print("-" * 50)
    
    for key, scenario in DEMO_SCENARIOS.items():
        print(f"  {key:12} - {scenario['title']}")
        print(f"               {scenario['description']}")
        print()

async def interactive_mode(pipeline):
    """Run interactive demo mode."""
    print("\n🎮 Interactive Demo Mode")
    print("Type 'quit' to exit, 'scenarios' to see available demos")
    
    session_id = "interactive_demo"
    pipeline.clear_session(session_id)
    
    while True:
        try:
            query = input("\n👤 You: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if query.lower() in ['scenarios', 'help']:
                show_scenario_menu()
                continue
            
            if not query:
                continue
            
            print("🤖 Assistant: ", end="", flush=True)
            
            # Use streaming for interactive mode
            async for chunk in pipeline.stream(query, session_id):
                print(chunk, end="", flush=True)
            
            print()  # New line after response
            
        except KeyboardInterrupt:
            print("\n👋 Demo interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

async def main():
    """Main demo execution."""
    parser = argparse.ArgumentParser(description='Inference Pipeline Demo')
    parser.add_argument('--scenario', choices=list(DEMO_SCENARIOS.keys()) + ['interactive', 'all'], 
                       default='interactive', help='Demo scenario to run')
    parser.add_argument('--streaming', action='store_true', help='Use streaming responses')
    parser.add_argument('--list', action='store_true', help='List available scenarios')
    
    args = parser.parse_args()
    
    if args.list:
        show_scenario_menu()
        return
    
    print("🎬 Inference Pipeline Demo")
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
        
        # Run demo
        if args.scenario == 'interactive':
            await interactive_mode(inference_pipeline)
        
        elif args.scenario == 'all':
            print("\n🎭 Running all demo scenarios...")
            
            for scenario_name, scenario_info in DEMO_SCENARIOS.items():
                await run_demo_scenario(
                    inference_pipeline, 
                    scenario_name, 
                    scenario_info, 
                    args.streaming
                )
                
                if scenario_name != list(DEMO_SCENARIOS.keys())[-1]:
                    input("\n⏸️  Press Enter to continue to next scenario...")
        
        else:
            scenario_info = DEMO_SCENARIOS[args.scenario]
            await run_demo_scenario(
                inference_pipeline, 
                args.scenario, 
                scenario_info, 
                args.streaming
            )
        
        print("\n🎉 Demo completed successfully!")
        
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())