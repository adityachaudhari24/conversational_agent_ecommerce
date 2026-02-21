# Metadata Filtering Integration - Design

## Architecture Overview

### Current Architecture
```
User Query
    ↓
Query Processor (embedding generation)
    ↓
Vector Searcher (semantic search only)
    ↓
Context Compressor
    ↓
Document Formatter
    ↓
Response
```

### New Architecture
```
User Query
    ↓
Query Processor (embedding generation)
    ↓
Metadata Extractor (NEW - extract filters from query)
    ↓
Vector Searcher (semantic search + metadata filtering)
    ↓
Context Compressor
    ↓
Document Formatter
    ↓
Response
```

## Component Design

### 1. RetrievalPipeline Integration

**File:** `src/pipelines/retrieval/pipeline.py`

#### Changes to RetrievalConfig
```python
@dataclass
class RetrievalConfig:
    query_config: QueryConfig
    search_config: SearchConfig
    compressor_config: CompressorConfig
    rewriter_config: RewriterConfig
    formatter_config: FormatterConfig
    cache_config: CacheConfig
    extractor_config: ExtractorConfig  # NEW
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    enable_evaluation: bool = False
```

#### Changes to RetrievalPipeline.__init__
```python
def __init__(self, config: RetrievalConfig, settings: RetrievalSettings):
    # ... existing code ...
    self.metadata_extractor: Optional[MetadataExtractor] = None  # NEW
```

#### Changes to RetrievalPipeline.initialize()
```python
def initialize(self) -> None:
    # ... existing initialization ...
    
    # Initialize metadata extractor (NEW)
    if self.config.extractor_config.enabled:
        self.logger.info("Initializing MetadataExtractor...")
        self.metadata_extractor = MetadataExtractor(
            self.config.extractor_config,
            self.settings.openai_api_key
        )
```

#### Changes to _execute_retrieval_workflow()
```python
def _execute_retrieval_workflow(
    self,
    query: str,
    filters: Optional[MetadataFilter] = None
) -> RetrievalResult:
    # ... existing code ...
    
    # NEW: Step 1.5: Extract metadata filters (if not provided)
    if filters is None and self.metadata_extractor:
        step_start = time.time()
        try:
            extracted_filters = self.metadata_extractor.extract(query)
            step_time = (time.time() - step_start) * 1000
            
            metadata['workflow_steps'].append({
                'step': 'metadata_extraction',
                'iteration': iteration,
                'time_ms': step_time,
                'filters_extracted': extracted_filters is not None
            })
            
            if extracted_filters:
                filters = extracted_filters
                metadata['auto_extracted_filters'] = True
                self.logger.info(
                    f"Auto-extracted filters: {filters.__dict__}",
                    extra={'extra_fields': filters.__dict__}
                )
        except Exception as e:
            self.logger.warning(f"Metadata extraction failed: {e}")
            # Continue without filters (graceful degradation)
    
    # Step 2: Vector search (now with potential filters)
    search_result = self.vector_searcher.search(
        processed_query.embedding,
        filters  # Pass filters here
    )
```

### 2. Configuration Updates

**File:** `config/retrieval.yaml`

Add new section:
```yaml
# Metadata Extraction Configuration
metadata_extraction:
  enabled: true
  llm_model: "gpt-3.5-turbo"
  temperature: 0.0
  max_tokens: 200
  timeout_seconds: 3
```

**File:** `src/pipelines/retrieval/config.py`

Add to RetrievalSettings:
```python
@dataclass
class RetrievalSettings:
    # ... existing fields ...
    
    # Metadata extraction settings
    metadata_extraction_enabled: bool = True
    metadata_extraction_model: str = "gpt-3.5-turbo"
    metadata_extraction_timeout: int = 3
```

### 3. ExtractorConfig Integration

**File:** `src/pipelines/retrieval/processors/metadata_extractor.py`

Update ExtractorConfig to include timeout:
```python
@dataclass
class ExtractorConfig:
    """Configuration for metadata extraction."""
    
    llm_model: str = "gpt-3.5-turbo"
    temperature: float = 0.0
    max_tokens: int = 200
    enabled: bool = True
    timeout_seconds: int = 3  # NEW
```

Add timeout handling to extract():
```python
def extract(self, query: str) -> Optional[MetadataFilter]:
    if not self.config.enabled:
        return None
    
    try:
        # Add timeout wrapper
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Metadata extraction timed out")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.config.timeout_seconds)
        
        try:
            # ... existing extraction logic ...
            result = self._do_extraction(query)
            signal.alarm(0)  # Cancel alarm
            return result
        except TimeoutError:
            self.logger.warning("Metadata extraction timed out")
            return None
    except Exception as e:
        self.logger.warning(f"Metadata extraction failed: {e}")
        return None
```

## Data Flow

### Successful Extraction Flow
```
1. User Query: "iPhone 12 price"
   ↓
2. Query Processor: Generate embedding
   ↓
3. Metadata Extractor: 
   - Call GPT-3.5: "Extract metadata from: iPhone 12 price"
   - Response: {"product_name_pattern": "iPhone 12", ...}
   - Create MetadataFilter(product_name_pattern="iPhone 12")
   ↓
4. Vector Searcher:
   - Semantic search with embedding
   - Apply Pinecone filter: {"product_name": {"$regex": ".*iPhone 12.*"}}
   - Return only iPhone 12 documents
   ↓
5. Context Compressor: Filter by relevance
   ↓
6. Document Formatter: Format results
   ↓
7. Response: Only iPhone 12 information
```

### Extraction Failure Flow
```
1. User Query: "iPhone 12 price"
   ↓
2. Query Processor: Generate embedding
   ↓
3. Metadata Extractor: 
   - Call GPT-3.5: TIMEOUT or ERROR
   - Log warning
   - Return None
   ↓
4. Vector Searcher:
   - Semantic search with embedding
   - No filters applied
   - Return semantically similar documents
   ↓
5. Continue normally (graceful degradation)
```

## Error Handling

### Extraction Errors
- **Timeout**: Log warning, continue without filters
- **API Error**: Log warning, continue without filters
- **Parse Error**: Log warning, continue without filters
- **Invalid Filter**: Log warning, continue without filters

### Search Errors with Filters
- **No Results**: Log info, retry without filters
- **Invalid Filter Format**: Log error, retry without filters
- **Pinecone Error**: Log error, retry without filters

## Performance Considerations

### Latency Impact
- Metadata extraction: ~200-500ms (LLM call)
- Mitigation: 3-second timeout
- Benefit: More precise results, less post-processing

### Caching Strategy
- Cache key includes extracted filters
- Same query with same filters = cache hit
- Different filters = cache miss (expected)

### Cost Impact
- Additional GPT-3.5 call per query
- ~$0.0015 per 1000 queries (input) + ~$0.002 per 1000 queries (output)
- Total: ~$0.0035 per 1000 queries
- Mitigation: Configurable enable/disable

## Testing Strategy

### Unit Tests
1. Test MetadataExtractor.extract() with various queries
2. Test RetrievalPipeline with extracted filters
3. Test graceful degradation on extraction failure
4. Test timeout handling

### Integration Tests
1. End-to-end test: "iPhone 12 price" → filtered results
2. End-to-end test: "phones under $300" → price-filtered results
3. End-to-end test: extraction failure → semantic results
4. End-to-end test: no matching filters → semantic results

### Manual Testing
1. Test with real Pinecone index
2. Verify filter application in Pinecone logs
3. Compare results with/without filtering
4. Measure latency impact

## Rollout Plan

### Phase 1: Configuration (Low Risk)
- Add configuration fields
- No behavior change yet
- Deploy and verify config loading

### Phase 2: Integration (Medium Risk)
- Integrate MetadataExtractor into pipeline
- Enable feature flag in config
- Deploy to staging environment
- Monitor for errors

### Phase 3: Validation (Low Risk)
- Run integration tests
- Manual testing with real queries
- Performance benchmarking
- Compare results quality

### Phase 4: Production (Medium Risk)
- Deploy to production with feature flag enabled
- Monitor error rates and latency
- A/B test if possible
- Rollback plan: disable feature flag

## Monitoring and Observability

### Metrics to Track
1. **Extraction Success Rate**: % of queries with successful extraction
2. **Extraction Latency**: P50, P95, P99 extraction time
3. **Filter Application Rate**: % of searches using filters
4. **Results Quality**: Manual evaluation of filtered vs non-filtered
5. **Error Rate**: Extraction failures, timeout rate

### Logging
- INFO: Successful extraction with filter details
- WARNING: Extraction failures, timeouts
- ERROR: Unexpected errors, invalid filters

### Alerts
- Alert if extraction error rate > 10%
- Alert if extraction latency P95 > 1000ms
- Alert if overall retrieval latency increases > 50%

## Backward Compatibility

### Existing Functionality
- All existing retrieval calls work unchanged
- `retrieve(query)` works as before (now with auto-extraction)
- `retrieve(query, filters)` still works (manual filters take precedence)

### Configuration
- Default: enabled (new behavior)
- Can disable via config (old behavior)
- No code changes needed to disable

### API Compatibility
- No API changes
- No breaking changes to function signatures
- Metadata added to results (additive only)

## Future Enhancements

1. **User-Provided Filters**: Allow API users to pass filters directly
2. **Filter Refinement**: Learn from user feedback to improve extraction
3. **Multi-Language Support**: Extract filters from non-English queries
4. **Custom Extraction Prompts**: Allow domain-specific extraction logic
5. **Filter Analytics**: Track which filters are most commonly used
6. **Smart Fallback**: If no results, automatically relax filters

## Security Considerations

1. **Input Validation**: Validate extracted filters before applying
2. **Injection Prevention**: Sanitize product name patterns for regex
3. **Rate Limiting**: Prevent abuse of extraction API
4. **API Key Security**: Ensure OpenAI key is properly secured

## Documentation Updates

1. Update README with metadata filtering feature
2. Add examples of filtered queries
3. Document configuration options
4. Add troubleshooting guide for extraction issues
