#!/usr/bin/env python3
"""
Interactive Inference Pipeline Chat

A terminal-based interactive chat interface for the inference pipeline.
Chat directly with the AI assistant using your own queries in real-time.

USAGE:
    python scripts/inference/interactive.py
    
    # With streaming responses (default)
    python scripts/inference/interactive.py --streaming
    
    # Without streaming (faster for testing)
    python scripts/inference/interactive.py --no-streaming
    
    # Custom session ID
    python scripts/inference/interactive.py --session my_session
    
    # Debug mode with detailed information
    python scripts/inference/interactive.py --debug

COMMANDS:
    Type your questions naturally, or use these special commands:
    
    /help       - Show available commands
    /history    - Show conversation history
    /clear      - Clear conversation history
    /stats      - Show pipeline statistics
    /health     - Check pipeline health
    /session    - Show current session info
    /new        - Start a new session
    /streaming  - Toggle streaming mode
    /debug      - Toggle debug mode
    /quit       - Exit the chat
"""

import os
import sys
import asyncio
import argparse
import time
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from pipelines.inference.pipeline import InferencePipeline
from pipelines.retrieval.pipeline import RetrievalPipeline

class InteractiveChat:
    """Interactive chat interface for the inference pipeline."""
    
    def __init__(self, pipeline, session_id="interactive_chat", streaming=True, debug=False):
        self.pipeline = pipeline
        self.session_id = session_id
        self.streaming = streaming
        self.debug = debug
        self.start_time = datetime.now()
        self.query_count = 0
        
        # Colors for terminal output
        self.colors = {
            'user': '\033[94m',      # Blue
            'assistant': '\033[92m', # Green
            'system': '\033[93m',    # Yellow
            'error': '\033[91m',     # Red
            'info': '\033[96m',      # Cyan
            'reset': '\033[0m'       # Reset
        }
    
    def print_colored(self, text, color='reset', end='\n'):
        """Print colored text to terminal."""
        print(f"{self.colors.get(color, '')}{text}{self.colors['reset']}", end=end)
    
    def print_welcome(self):
        """Print welcome message and instructions."""
        self.print_colored("=" * 70, 'info')
        self.print_colored("🤖 Interactive Inference Pipeline Chat", 'info')
        self.print_colored("=" * 70, 'info')
        print()
        self.print_colored("Welcome! You can now chat directly with the AI assistant.", 'system')
        self.print_colored("Type your questions naturally, or use commands starting with '/'", 'system')
        self.print_colored("Type '/help' for available commands or '/quit' to exit.", 'system')
        print()
        self.print_colored(f"📋 Session ID: {self.session_id}", 'info')
        self.print_colored(f"🌊 Streaming: {'Enabled' if self.streaming else 'Disabled'}", 'info')
        self.print_colored(f"🔍 Debug Mode: {'Enabled' if self.debug else 'Disabled'}", 'info')
        print()
        self.print_colored("-" * 70, 'info')
    
    def print_help(self):
        """Print available commands."""
        self.print_colored("\n📚 Available Commands:", 'system')
        self.print_colored("-" * 30, 'system')
        commands = [
            ("/help", "Show this help message"),
            ("/history", "Show conversation history"),
            ("/clear", "Clear conversation history"),
            ("/stats", "Show pipeline statistics"),
            ("/health", "Check pipeline health"),
            ("/session", "Show current session info"),
            ("/new [session_id]", "Start a new session"),
            ("/streaming", "Toggle streaming mode"),
            ("/debug", "Toggle debug mode"),
            ("/quit", "Exit the chat")
        ]
        
        for cmd, desc in commands:
            self.print_colored(f"  {cmd:20} - {desc}", 'info')
        print()
    
    def print_history(self):
        """Print conversation history."""
        try:
            history = self.pipeline.get_session_history(self.session_id)
            
            if not history:
                self.print_colored("📝 No conversation history yet.", 'system')
                return
            
            self.print_colored(f"\n💬 Conversation History ({len(history)} messages):", 'system')
            self.print_colored("-" * 50, 'system')
            
            for i, message in enumerate(history, 1):
                timestamp = message.timestamp.strftime("%H:%M:%S")
                color = 'user' if message.role == 'user' else 'assistant'
                emoji = "👤" if message.role == 'user' else "🤖"
                
                self.print_colored(f"[{i:2d}] {emoji} {message.role.title()} ({timestamp}):", color)
                
                # Wrap long messages
                content = message.content
                if len(content) > 80:
                    words = content.split()
                    lines = []
                    current_line = ""
                    
                    for word in words:
                        if len(current_line + word) > 76:
                            if current_line:
                                lines.append(current_line.strip())
                                current_line = word + " "
                            else:
                                lines.append(word)
                        else:
                            current_line += word + " "
                    
                    if current_line:
                        lines.append(current_line.strip())
                    
                    for line in lines:
                        print(f"     {line}")
                else:
                    print(f"     {content}")
                print()
                
        except Exception as e:
            self.print_colored(f"❌ Error getting history: {e}", 'error')
    
    def print_stats(self):
        """Print pipeline statistics."""
        try:
            stats = self.pipeline.get_pipeline_stats()
            
            self.print_colored("\n📊 Pipeline Statistics:", 'system')
            self.print_colored("-" * 30, 'system')
            
            # Session stats
            session_time = datetime.now() - self.start_time
            self.print_colored(f"🕒 Session duration: {session_time}", 'info')
            self.print_colored(f"💬 Queries processed: {self.query_count}", 'info')
            
            # Pipeline config
            config = stats.get('configuration', {})
            self.print_colored(f"🤖 LLM Model: {config.get('llm_model', 'Unknown')}", 'info')
            self.print_colored(f"🔄 Max Retries: {config.get('max_retries', 'Unknown')}", 'info')
            self.print_colored(f"⏱️  Timeout: {config.get('timeout_seconds', 'Unknown')}s", 'info')
            self.print_colored(f"🌊 Streaming: {config.get('enable_streaming', 'Unknown')}", 'info')
            
            # Component stats
            components = stats.get('components', {})
            if 'conversation_manager' in components:
                cm_stats = components['conversation_manager']
                self.print_colored(f"📋 Active Sessions: {cm_stats.get('active_sessions', 0)}", 'info')
            
        except Exception as e:
            self.print_colored(f"❌ Error getting stats: {e}", 'error')
    
    def print_health(self):
        """Print pipeline health status."""
        try:
            health = self.pipeline.health_check()
            
            self.print_colored("\n🏥 Pipeline Health Check:", 'system')
            self.print_colored("-" * 30, 'system')
            
            status_color = 'assistant' if health['status'] == 'healthy' else 'error'
            self.print_colored(f"Overall Status: {health['status'].upper()}", status_color)
            
            if health.get('errors'):
                self.print_colored("❌ Errors:", 'error')
                for error in health['errors']:
                    self.print_colored(f"   • {error}", 'error')
            
            self.print_colored("🔧 Components:", 'info')
            for component, status in health.get('components', {}).items():
                status_emoji = "✅" if status == "healthy" else "❌" if "error" in str(status) else "⚠️"
                self.print_colored(f"   {status_emoji} {component}: {status}", 'info')
                
        except Exception as e:
            self.print_colored(f"❌ Error checking health: {e}", 'error')
    
    def print_session_info(self):
        """Print current session information."""
        self.print_colored(f"\n📋 Session Information:", 'system')
        self.print_colored("-" * 25, 'system')
        self.print_colored(f"Session ID: {self.session_id}", 'info')
        self.print_colored(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}", 'info')
        self.print_colored(f"Duration: {datetime.now() - self.start_time}", 'info')
        self.print_colored(f"Queries: {self.query_count}", 'info')
        self.print_colored(f"Streaming: {'Enabled' if self.streaming else 'Disabled'}", 'info')
        self.print_colored(f"Debug: {'Enabled' if self.debug else 'Disabled'}", 'info')
    
    def handle_command(self, command):
        """Handle special commands."""
        parts = command[1:].split()  # Remove '/' and split
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        if cmd == 'help':
            self.print_help()
        
        elif cmd == 'history':
            self.print_history()
        
        elif cmd == 'clear':
            try:
                self.pipeline.clear_session(self.session_id)
                self.print_colored("✅ Conversation history cleared.", 'system')
            except Exception as e:
                self.print_colored(f"❌ Error clearing history: {e}", 'error')
        
        elif cmd == 'stats':
            self.print_stats()
        
        elif cmd == 'health':
            self.print_health()
        
        elif cmd == 'session':
            self.print_session_info()
        
        elif cmd == 'new':
            new_session_id = args[0] if args else f"chat_{int(time.time())}"
            self.session_id = new_session_id
            self.start_time = datetime.now()
            self.query_count = 0
            self.print_colored(f"✅ Started new session: {self.session_id}", 'system')
        
        elif cmd == 'streaming':
            self.streaming = not self.streaming
            status = "enabled" if self.streaming else "disabled"
            self.print_colored(f"✅ Streaming {status}.", 'system')
        
        elif cmd == 'debug':
            self.debug = not self.debug
            status = "enabled" if self.debug else "disabled"
            self.print_colored(f"✅ Debug mode {status}.", 'system')
        
        elif cmd == 'quit':
            return False
        
        else:
            self.print_colored(f"❌ Unknown command: /{cmd}", 'error')
            self.print_colored("Type '/help' for available commands.", 'system')
        
        return True
    
    async def process_query(self, query):
        """Process a user query and return response."""
        self.query_count += 1
        
        if self.debug:
            self.print_colored(f"🔍 Processing query #{self.query_count}...", 'system')
            start_time = time.time()
        
        try:
            if self.streaming:
                # Streaming response
                self.print_colored("🤖 Assistant: ", 'assistant', end='')
                
                complete_response = ""
                chunk_count = 0
                
                async for chunk in self.pipeline.stream(query, self.session_id):
                    complete_response += chunk
                    chunk_count += 1
                    print(chunk, end='', flush=True)
                
                print()  # New line after response
                
                if self.debug:
                    elapsed = (time.time() - start_time) * 1000
                    self.print_colored(f"⏱️  Response time: {elapsed:.0f}ms ({chunk_count} chunks)", 'system')
            
            else:
                # Non-streaming response
                result = await self.pipeline.agenerate(query, self.session_id)
                
                self.print_colored("🤖 Assistant: ", 'assistant')
                print(f"     {result.response}")
                
                if self.debug:
                    self.print_colored(f"⏱️  Response time: {result.latency_ms:.0f}ms", 'system')
                    self.print_colored(f"🔢 Tokens used: {result.tokens_used}", 'system')
                    
                    if result.metadata:
                        self.print_colored("📈 Metadata:", 'system')
                        for key, value in result.metadata.items():
                            if key == 'workflow_steps':
                                self.print_colored(f"   🔄 Workflow steps: {len(value)}", 'system')
                            else:
                                self.print_colored(f"   • {key}: {value}", 'system')
        
        except Exception as e:
            self.print_colored(f"❌ Error processing query: {e}", 'error')
            if self.debug:
                import traceback
                traceback.print_exc()
    
    async def run(self):
        """Run the interactive chat loop."""
        self.print_welcome()
        
        try:
            while True:
                # Get user input
                try:
                    self.print_colored("👤 You: ", 'user', end='')
                    user_input = input().strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    self.print_colored("👋 Goodbye!", 'system')
                    break
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    if not self.handle_command(user_input):
                        break
                    continue
                
                # Process regular query
                print()  # Add space before response
                await self.process_query(user_input)
                print()  # Add space after response
        
        except Exception as e:
            self.print_colored(f"❌ Chat error: {e}", 'error')
            if self.debug:
                import traceback
                traceback.print_exc()

async def main():
    """Main interactive chat execution."""
    parser = argparse.ArgumentParser(description='Interactive Inference Pipeline Chat')
    parser.add_argument('--session', default='interactive_chat', 
                       help='Session ID for conversation tracking')
    parser.add_argument('--streaming', action='store_true', default=True,
                       help='Enable streaming responses (default)')
    parser.add_argument('--no-streaming', action='store_true',
                       help='Disable streaming responses')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with detailed information')
    
    args = parser.parse_args()
    
    # Handle streaming flag
    streaming = args.streaming and not args.no_streaming
    
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
        
        # Start interactive chat
        chat = InteractiveChat(
            pipeline=inference_pipeline,
            session_id=args.session,
            streaming=streaming,
            debug=args.debug
        )
        
        await chat.run()
        
    except KeyboardInterrupt:
        print("\n👋 Chat interrupted. Goodbye!")
    except Exception as e:
        print(f"❌ Failed to start chat: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())