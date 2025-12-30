# Inference Pipeline Test Scripts

This directory contains test scripts for the Data Inference Pipeline, designed to help you understand and validate the inference functionality.

## Scripts Overview

### `interactive.py` - Terminal Chat Interface

A terminal-based interactive chat interface for real-time conversation with the inference pipeline.

### `simple_test.py` - Comprehensive Inference Testing

A comprehensive test script that validates all aspects of the inference pipeline with configurable test flags.

### `quick_test.py` - Rapid Validation

A minimal test script for quick validation during development or CI/CD pipelines.

### `demo.py` - Interactive Demonstrations

An interactive demo showcasing realistic e-commerce scenarios with streaming responses.

### `benchmark.py` - Performance Testing

A focused performance testing script measuring latency, throughput, and resource usage.

**Features:**
- ✅ **Basic Inference**: Simple query-response testing
- ✅ **Conversation History**: Multi-turn conversation testing
- ✅ **Agentic Workflow**: Test different routing paths (retrieve/tool/respond)
- ✅ **Streaming Response**: Real-time response streaming
- ✅ **Error Handling**: Graceful error recovery testing
- ✅ **Performance Testing**: Latency and throughput measurement
- ✅ **Health Monitoring**: Pipeline component health checks

## Quick Start

### Prerequisites

1. **Environment Variables**: Set up your API keys
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export PINECONE_API_KEY="your-pinecone-api-key"
   export PINECONE_INDEX_NAME="your-index-name"
   export PINECONE_NAMESPACE="your-namespace"
   export PINECONE_ENVIRONMENT="your-environment"
   ```

2. **Dependencies**: Ensure all packages are installed
   ```bash
   uv pip install -r requirements.txt
   ```

3. **Data**: Make sure your vector database is populated (run ingestion pipeline first)

### 🚀 **Start Chatting Immediately**

The fastest way to test the inference pipeline with your own queries:

```bash
# Start interactive chat with streaming
python scripts/inference/interactive.py

# Start interactive chat without streaming (faster responses)
python scripts/inference/interactive.py --no-streaming

# Start with debug mode for detailed information
python scripts/inference/interactive.py --debug
```

**Example Chat Session:**
```
🤖 Interactive Inference Pipeline Chat
======================================================================

Welcome! You can now chat directly with the AI assistant.
Type your questions naturally, or use commands starting with '/'
Type '/help' for available commands or '/quit' to exit.

📋 Session ID: interactive_chat
🌊 Streaming: Enabled
🔍 Debug Mode: Disabled

----------------------------------------------------------------------

👤 You: What's the best phone for photography under $600?
🤖 Assistant: Based on the available product data, here are some excellent phones for photography under $600...

👤 You: What about battery life on these phones?
🤖 Assistant: Regarding battery life for the photography-focused phones I mentioned...

👤 You: /history
💬 Conversation History (4 messages):
[1] 👤 User (14:30:15): What's the best phone for photography under $600?
[2] 🤖 Assistant (14:30:17): Based on the available product data...

👤 You: /quit
👋 Goodbye!
```

### Running Tests

#### Interactive Chat (Real-time Terminal Chat)
```bash
python scripts/inference/interactive.py
python scripts/inference/interactive.py --session my_chat
python scripts/inference/interactive.py --no-streaming --debug
```

#### Comprehensive Testing
```bash
python scripts/inference/simple_test.py
```

#### Quick Validation
```bash
python scripts/inference/quick_test.py
python scripts/inference/quick_test.py --test basic
python scripts/inference/quick_test.py --test conversation
```

#### Interactive Demo
```bash
python scripts/inference/demo.py
python scripts/inference/demo.py --scenario shopping
python scripts/inference/demo.py --scenario comparison --streaming
```

#### Performance Benchmarking
```bash
python scripts/inference/benchmark.py
python scripts/inference/benchmark.py --test latency --requests 20
python scripts/inference/benchmark.py --test throughput --concurrent 5
```

#### Run Specific Tests
Edit the `TEST_FLAGS` dictionary in `simple_test.py`:

```python
TEST_FLAGS = {
    "basic_inference": True,      # Enable basic testing
    "conversation_history": True, # Enable conversation testing
    "agentic_workflow": False,    # Disable workflow testing
    "streaming_response": False,  # Disable streaming testing
    "error_handling": False,      # Disable error testing
    "performance_test": False,    # Disable performance testing
}
```

## Test Descriptions

### 0. Interactive Chat Interface
**Purpose**: Real-time terminal-based chat with the inference pipeline using your own queries.

**What it provides**:
- Direct conversation with the AI assistant
- Real-time streaming responses
- Conversation history management
- Pipeline health monitoring
- Debug information and statistics

**Key Features**:
- **Natural Chat**: Type questions naturally, get AI responses
- **Commands**: Special commands starting with `/` for control
- **Streaming**: Real-time response streaming (toggleable)
- **History**: View and manage conversation history
- **Sessions**: Multiple conversation sessions
- **Debug Mode**: Detailed timing and metadata information

**Available Commands**:
```
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
```

**Example Session**:
```
👤 You: What's the best phone for photography?
🤖 Assistant: Based on the available data, here are some excellent phones for photography...

👤 You: What about under $500?
🤖 Assistant: For photography under $500, I'd recommend...

👤 You: /history
💬 Conversation History (4 messages):
[1] 👤 User (14:30:15): What's the best phone for photography?
[2] 🤖 Assistant (14:30:17): Based on the available data...
```

### 1. Basic Inference Test
**Purpose**: Validates simple query-response functionality without conversation history.

**What it tests**:
- Pipeline initialization
- Single query processing
- Response generation
- Basic metadata collection

**Example Output**:
```
🔍 Query: What are the best phones under $500?
🤖 Response: Based on the available data, here are some excellent phones under $500...
⏱️  Latency: 1250ms
🔢 Tokens Used: 245
```

### 2. Conversation History Test
**Purpose**: Validates multi-turn conversations with context retention.

**What it tests**:
- Session management
- Conversation history storage
- Context continuity across turns
- History length limits

**Example Flow**:
```
Turn 1: "I'm looking for a new smartphone"
Turn 2: "What about phones with good cameras?"
Turn 3: "Which one has the best battery life?"
Turn 4: "What's the price range for these phones?"
```

### 3. Agentic Workflow Test
**Purpose**: Tests the LangGraph-based routing system with different query types.

**What it tests**:
- Query classification (product/tool/general)
- Retrieval pipeline integration
- Tool node execution
- Workflow routing decisions

**Query Types Tested**:
- **Product Query**: "Compare iPhone 14 vs Samsung Galaxy S23" → Routes to retriever
- **Tool Query**: "compare iPhone vs Samsung features" → Routes to tool node
- **General Query**: "Hello, how are you today?" → Routes directly to generator

### 4. Streaming Response Test
**Purpose**: Validates real-time response streaming functionality.

**What it tests**:
- Async streaming setup
- Chunk-by-chunk response delivery
- Stream completion handling
- Error recovery during streaming

**Example Output**:
```
🤖 Streaming response:
The latest iPhone models include the iPhone 15 series...
[response streams in real-time]
✅ Streaming completed!
📦 Chunks received: 45
```

### 5. Error Handling Test
**Purpose**: Tests graceful error handling and recovery mechanisms.

**What it tests**:
- Empty query handling
- Oversized query handling
- Invalid session handling
- Timeout scenarios
- API failure recovery

### 6. Performance Test
**Purpose**: Measures pipeline performance and identifies bottlenecks.

**What it tests**:
- Response latency
- Throughput under load
- Memory usage patterns
- Component performance

**Metrics Collected**:
- Average/Min/Max latency
- Tokens per second
- Requests per minute
- Component timing breakdown

## Customization

### Custom Test Queries
Modify the `TEST_QUERIES` dictionary to test with your specific use cases:

```python
TEST_QUERIES = {
    "basic": "Your custom basic query",
    "conversation": [
        "First turn query",
        "Second turn query",
        "Third turn query"
    ],
    "routing": {
        "product_query": "Your product query",
        "tool_query": "Your tool query", 
        "general_query": "Your general query"
    }
}
```

### Custom Session IDs
Modify `TEST_SESSIONS` to use different session identifiers:

```python
TEST_SESSIONS = {
    "basic": "my_basic_session",
    "conversation": "my_conversation_session",
    # ... etc
}
```

## Understanding Output

### Inference Result Format
```
📋 Inference Result:
🔍 Query: [user query]
🤖 Response: [generated response]
📊 Session ID: [session identifier]
⏱️  Latency: [response time in ms]
🔢 Tokens Used: [total tokens consumed]
🕒 Timestamp: [completion time]
📈 Metadata: [workflow steps and timing]
```

### Conversation History Format
```
💬 Conversation History (4 messages):
[1] 👤 User: I'm looking for a new smartphone
    ⏰ 14:30:15
[2] 🤖 Assistant: I'd be happy to help you find a smartphone...
    ⏰ 14:30:17
```

### Health Check Format
```
🏥 Overall Status: HEALTHY
🔧 Component Status:
   ✅ llm_client: healthy
   ✅ conversation_manager: healthy
   ✅ response_generator: healthy
   ✅ agentic_workflow: healthy
   ✅ retrieval_pipeline: healthy
```

## Troubleshooting

### Common Issues

1. **Missing API Keys**
   ```
   ❌ Please set OPENAI_API_KEY environment variable
   ```
   **Solution**: Set the required environment variables

2. **Pipeline Initialization Failed**
   ```
   ❌ Failed to initialize inference pipeline: [error details]
   ```
   **Solution**: Check configuration files and API connectivity

3. **Retrieval Pipeline Not Found**
   ```
   ❌ Retrieval pipeline initialization failed
   ```
   **Solution**: Ensure retrieval pipeline is properly configured and vector database is populated

4. **Streaming Not Supported**
   ```
   ❌ Streaming is disabled in configuration
   ```
   **Solution**: Enable streaming in `config/inference.yaml`

### Debug Mode

For detailed debugging, modify the script to enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Integration with Other Pipelines

The inference pipeline depends on:
- **Retrieval Pipeline**: For context fetching
- **Vector Database**: For document storage (Pinecone/ChromaDB)
- **LLM Provider**: For response generation (OpenAI)

Make sure these components are properly configured before running inference tests.

## Performance Expectations

**Typical Performance** (with gpt-4o-mini):
- Basic inference: 800-2000ms
- With retrieval: 1200-3000ms
- Streaming: First chunk in 200-500ms
- Conversation turns: 600-1500ms

**Factors affecting performance**:
- LLM model choice (gpt-4o vs gpt-4o-mini)
- Context length
- Retrieval complexity
- Network latency
- API rate limits

## Script Comparison

| Script | Purpose | Best For | Time Required | Interactive |
|--------|---------|----------|---------------|-------------|
| `interactive.py` | Terminal chat interface | Real-time testing, exploration | Ongoing | ✅ Yes |
| `simple_test.py` | Comprehensive testing | Development, debugging, validation | 5-10 minutes | ❌ No |
| `quick_test.py` | Rapid validation | CI/CD, quick checks | 1-2 minutes | ❌ No |
| `demo.py` | Interactive showcase | Demonstrations, understanding | 10-20 minutes | ✅ Yes |
| `benchmark.py` | Performance testing | Optimization, capacity planning | 3-5 minutes | ❌ No |

### Choosing the Right Script

**For Real-time Testing:**
- Use `interactive.py` for natural conversation testing
- Use `interactive.py --debug` for detailed analysis
- Use `interactive.py --no-streaming` for faster responses

**For Development:**
- Use `interactive.py` for exploratory testing
- Use `quick_test.py` for rapid iteration
- Use `simple_test.py` for thorough validation
- Use `demo.py` to understand user experience

**For Production:**
- Use `benchmark.py` for performance validation
- Use `quick_test.py` in CI/CD pipelines
- Use `simple_test.py` for comprehensive health checks

**For Demonstrations:**
- Use `interactive.py` for live Q&A sessions
- Use `demo.py` for realistic scenarios
- Use `simple_test.py` with specific test flags
- Use `benchmark.py` to show performance metrics

## Interactive Chat Features

The `interactive.py` script provides a full-featured terminal chat interface:

### 🎯 **Core Features**
- **Natural Conversation**: Chat naturally with the AI assistant
- **Real-time Streaming**: See responses as they're generated (toggleable)
- **Session Management**: Multiple conversation sessions with history
- **Command System**: Special commands for control and monitoring
- **Debug Mode**: Detailed timing, metadata, and pipeline information
- **Colored Output**: Easy-to-read colored terminal output

### 🎮 **Interactive Commands**

| Command | Description | Example |
|---------|-------------|---------|
| `/help` | Show all available commands | `/help` |
| `/history` | Display conversation history | `/history` |
| `/clear` | Clear current session history | `/clear` |
| `/stats` | Show pipeline statistics | `/stats` |
| `/health` | Check pipeline component health | `/health` |
| `/session` | Show current session info | `/session` |
| `/new [id]` | Start new session | `/new my_session` |
| `/streaming` | Toggle streaming on/off | `/streaming` |
| `/debug` | Toggle debug mode | `/debug` |
| `/quit` | Exit the chat | `/quit` |

### 🎨 **Visual Features**
- **Color-coded Output**: Different colors for user, assistant, system messages
- **Formatted History**: Timestamped conversation history
- **Progress Indicators**: Real-time streaming indicators
- **Status Information**: Pipeline health and statistics
- **Error Handling**: Clear error messages with optional debug details

### 💡 **Usage Tips**
- Start with `/help` to see all available commands
- Use `/debug` to see detailed response timing and metadata
- Use `/streaming` to toggle between streaming and instant responses
- Use `/new` to start fresh conversations for different topics
- Use `/history` to review your conversation
- Use `/stats` and `/health` to monitor pipeline performance

## Next Steps

After running these tests successfully:

1. **Integration Testing**: Test with your frontend application
2. **Load Testing**: Use tools like `locust` for production load testing
3. **Monitoring**: Set up logging and metrics collection
4. **Optimization**: Tune configuration based on test results

For more advanced testing scenarios, consider creating custom test scripts based on your specific use cases.