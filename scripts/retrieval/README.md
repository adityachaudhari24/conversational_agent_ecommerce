# Retrieval Pipeline Demo Scripts

This directory contains demo scripts to test and explore the retrieval pipeline functionality.

## Prerequisites

1. **Environment Variables**: Set the required API keys:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export PINECONE_API_KEY="your-pinecone-api-key"
   ```

2. **Data**: Ensure your Pinecone index is populated with phone review data (run the ingestion pipeline first).

3. **Dependencies**: Install required packages:
   ```bash
   uv pip install -r requirements.txt
   ```

## Scripts

### 1. Quick Test (`quick_test.py`)

A simple script for basic functionality testing:

```bash
python scripts/retrieval/quick_test.py
```

**What it tests:**
- Basic pipeline initialization
- Simple query retrieval
- Cache functionality
- Metadata filtering

### 2. Comprehensive Demo (`demo_retrieval_pipeline.py`)

A full-featured demo that tests all pipeline capabilities:

```bash
# Run full demo
python scripts/retrieval/demo_retrieval_pipeline.py

# Run interactive mode
python scripts/retrieval/demo_retrieval_pipeline.py --mode interactive
```

**What it demonstrates:**

#### 🔍 **Basic Vector Search**
- Semantic similarity search
- Different query types (product features, price ranges, etc.)
- Score-based ranking

#### 🎯 **Metadata Filtering**
- Price range filtering (`min_price`, `max_price`)
- Rating filtering (`min_rating`)
- Combined filters
- Product name pattern matching

#### ✏️ **Query Rewriting**
- Automatic rewriting of vague queries
- Improvement tracking
- Multiple rewrite attempts
- Fallback to original query

#### 🗜️ **Contextual Compression**
- LLM-based relevance filtering
- Document filtering metrics
- Compression ratio tracking

#### 💾 **Result Caching**
- Cache hit/miss demonstration
- Performance comparison
- TTL (Time To Live) behavior
- Filter-aware caching

#### ⚡ **Performance & Monitoring**
- Latency measurements
- Component health checks
- Pipeline statistics
- Error handling

## Example Queries to Try

### Basic Searches
```python
"iPhone with good camera quality"
"cheap smartphone under $200"
"phone with long battery life"
"iPhone 12 reviews"
```

### Filtered Searches
```python
# Premium phones
query = "iPhone reviews"
filters = MetadataFilter(min_price=400.0)

# Budget phones
query = "smartphone"
filters = MetadataFilter(max_price=300.0)

# Highly rated phones
query = "phone reviews"
filters = MetadataFilter(min_rating=4.0)

# Combined filters
query = "iPhone"
filters = MetadataFilter(min_price=200.0, max_price=500.0, min_rating=3.0)
```

### Vague Queries (to trigger rewriting)
```python
"something good"
"what should I buy"
"recommendations please"
"best option available"
"help me choose"
```

### Feature-Specific Queries (for compression testing)
```python
"iPhone camera features"
"battery problems with phones"
"screen quality comparison"
"value for money smartphone"
```

## Interactive Mode Commands

When running in interactive mode, you can use these commands:

- `help` - Show available commands
- `stats` - Display pipeline statistics
- `clear` - Clear the result cache
- `health` - Show component health status
- `quit` - Exit interactive mode

## Understanding the Output

### Query Results
```
📝 Query: 'iPhone with good camera quality'
🔍 Test: Basic Similarity Search
⏱️  Latency: 245.67ms
💾 From Cache: No
📄 Documents Found: 4
📊 Score Range: 0.756 - 0.892
✏️  Query Rewrites: 0
🗜️  Compression Applied: Yes

📱 Top Result:
   Product: Apple iPhone 12 Pro 5G, US Version, 512GB, Graphite - Unlocked (Renewed)
   Price: $620.0
   Rating: 5.0/5
   Review: Shoot amazing videos and photos with the Ultra Wide, Wide, and Telephoto cameras...
```

### Cache Performance
```
⚡ Performance Comparison:
   First request: 245.67ms
   Second request: 12.34ms
   Speedup: 19.9x faster
```

### Component Health
```
Pipeline Status: healthy
🔧 Configuration:
   max_retries: 3
   retry_delay_seconds: 1.0
   enable_evaluation: False

🏗️  Components Status:
   ✅ query_processor: Active
   ✅ vector_searcher: Active
   ✅ context_compressor: Active
   ✅ query_rewriter: Active
   ✅ document_formatter: Active
   ✅ cache: Active
      Cache size: 5/1000
      Total hits: 12
```

## Troubleshooting

### Common Issues

1. **Missing API Keys**
   ```
   ❌ Configuration error: Missing required environment variables: OPENAI_API_KEY
   ```
   **Solution**: Set the required environment variables.

2. **Pinecone Index Not Found**
   ```
   ❌ ConnectionError: Index 'ecommerce-products' not found
   ```
   **Solution**: Run the ingestion pipeline first to populate the index.

3. **No Documents Found**
   ```
   📄 Documents Found: 0
   ```
   **Solution**: Check if your Pinecone index has data and verify the namespace configuration.

4. **Slow Performance**
   - First queries are slower due to cold start
   - Subsequent queries should be faster due to caching
   - Check your internet connection for API calls

### Configuration

The pipeline uses `config/retrieval.yaml` for configuration. Key settings:

```yaml
search:
  top_k: 4              # Number of documents to retrieve
  score_threshold: 0.6  # Minimum similarity score
  search_type: mmr      # "mmr" for diversity, "similarity" for pure similarity

cache:
  enabled: true         # Enable result caching
  ttl_seconds: 300     # Cache TTL (5 minutes)
  max_size: 1000       # Maximum cache entries

compression:
  enabled: true         # Enable contextual compression

rewriter:
  max_rewrite_attempts: 2    # Maximum query rewrites
  rewrite_threshold: 0.5     # Trigger rewrite below this relevance score
```

## Advanced Usage

### Custom Pipeline Configuration

```python
from pipelines.retrieval.pipeline import RetrievalPipeline
from pipelines.retrieval.config import RetrievalSettings

# Custom settings
settings = RetrievalSettings(
    top_k=6,
    score_threshold=0.7,
    compression_enabled=False,
    cache_enabled=True
)

pipeline = RetrievalPipeline.from_settings(settings)
pipeline.initialize()
```

### Async Usage

```python
import asyncio

async def async_retrieval():
    result = await pipeline.aretrieve("iPhone reviews")
    return result

# Run async
result = asyncio.run(async_retrieval())
```

### Context Manager

```python
with RetrievalPipeline.from_config_file() as pipeline:
    result = pipeline.retrieve("iPhone camera quality")
    print(f"Found {len(result.documents)} documents")
```

## Performance Tips

1. **Use Caching**: Identical queries are served from cache for 5 minutes by default
2. **Optimize Filters**: Use metadata filters to reduce search space
3. **Batch Queries**: For multiple queries, consider the async interface
4. **Monitor Health**: Use `pipeline.health_check()` to monitor component status
5. **Clear Cache**: Use `pipeline.clear_cache()` if you need fresh results

## Next Steps

After running these demos, you can:

1. **Integrate with Frontend**: Use the pipeline in your Streamlit chat interface
2. **Add Custom Filters**: Extend MetadataFilter for your specific use cases
3. **Tune Parameters**: Adjust search parameters based on your data and requirements
4. **Add Evaluation**: Enable RAGAS evaluation for quality metrics
5. **Scale Up**: Deploy the pipeline in a production environment