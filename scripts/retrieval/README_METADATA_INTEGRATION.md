# Metadata Filtering Integration Test

## Overview

`test_metadata_integration.py` is a comprehensive integration test script for the metadata filtering feature in the retrieval pipeline. It tests end-to-end functionality with real OpenAI API calls and validates that metadata extraction and filtering work correctly.

## Purpose

This test validates:
- Automatic metadata extraction from natural language queries
- Filter application at the vector database level
- Graceful degradation when extraction fails
- Integration between MetadataExtractor and RetrievalPipeline
- Various query types (product names, prices, ratings, combinations)

## Requirements

### Environment Variables
- `OPENAI_API_KEY`: OpenAI API key for LLM calls
- `PINECONE_API_KEY`: Pinecone API key for vector database access

### Dependencies
- Python 3.11+
- All project dependencies installed (`uv pip install -r requirements.txt`)
- Pinecone index with product data

## Usage

### Basic Usage
```bash
python scripts/retrieval/test_metadata_integration.py
```

### With UV
```bash
uv run python scripts/retrieval/test_metadata_integration.py
```

## Test Cases

The script includes the following test cases:

### 1. Product Name Filter
- **Query**: "iPhone 12 price"
- **Expected**: Extract product_name_pattern="iPhone 12"
- **Validates**: Req 1.1, 1.2, 1.3, 1.4

### 2. Price Maximum Filter
- **Query**: "phones under $300"
- **Expected**: Extract max_price=300.0
- **Validates**: Req 2.1, 2.4

### 3. Price Minimum Filter
- **Query**: "phones over $500"
- **Expected**: Extract min_price=500.0
- **Validates**: Req 2.2, 2.4

### 4. Rating Filter
- **Query**: "highly rated Samsung phones"
- **Expected**: Extract product_name_pattern="Samsung", min_rating=4.0
- **Validates**: Req 3.1, 3.2, 3.3

### 5. Combined Filters
- **Query**: "highly rated Samsung phones over $400"
- **Expected**: Extract product_name, min_price, and min_rating
- **Validates**: Req 4.1, 4.2, 4.3

### 6. No Extractable Metadata
- **Query**: "What are smartphones?"
- **Expected**: No filters extracted, semantic search only
- **Validates**: Req 5.1, 5.4

### 7. Product and Price Range
- **Query**: "iPhone 12 under $800"
- **Expected**: Extract product_name_pattern="iPhone 12", max_price=800.0
- **Validates**: Req 1.1, 2.1

## Output

The script provides detailed output including:

### Test Execution
- Query being tested
- Extracted filters (if any)
- Documents retrieved
- Execution time
- Pass/fail status

### Summary Report
- Total tests run
- Pass/fail counts
- Pass rate percentage
- Detailed results for each test

### Example Output
```
================================================================================
                      Metadata Filtering Integration Tests                      
================================================================================

--------------------------------------------------------------------------------
Test: product_name_filter
--------------------------------------------------------------------------------
Description: Test extraction of product name filter
Query: 'iPhone 12 price'
  Extracted Filters:
    • Product Name: iPhone 12

  Result Summary:
    • Documents Retrieved: 5
    • Latency: 1234ms
    • From Cache: False
    • Auto-Extracted Filters: True
    • Extraction Time: 234ms
    • Filters Extracted: True
✓ Test PASSED: Filters match expected values
```

## Exit Codes

- **0**: All tests passed
- **1**: One or more tests failed

## Customization

### Adding New Test Cases

Add new test cases to the `TEST_CASES` list:

```python
TEST_CASES.append(
    TestCase(
        name="your_test_name",
        query="your test query",
        expected_filters={
            "product_name_pattern": "Expected Product",
            "min_price": 100.0,
            "max_price": 500.0,
            "min_rating": 4.0
        },
        description="Description of what this tests",
        should_extract=True
    )
)
```

### Modifying Settings

The test initializes the pipeline with specific settings:

```python
settings = RetrievalSettings(
    metadata_extraction_enabled=True,
    metadata_extraction_model="gpt-3.5-turbo",
    metadata_extraction_timeout=3,
    cache_enabled=False  # Disable cache for testing
)
```

Modify these in the `setup()` method to test different configurations.

## Troubleshooting

### Test Failures

If tests fail, check:

1. **API Keys**: Ensure OPENAI_API_KEY and PINECONE_API_KEY are set
2. **Pinecone Index**: Verify the index exists and contains data
3. **Network**: Check internet connectivity for API calls
4. **Extraction Logic**: Review metadata_extractor.py for JSON parsing issues
5. **LLM Responses**: Check logs for actual LLM responses

### Common Issues

#### "Metadata extraction failed"
- The LLM returned invalid JSON
- Check the extraction prompt in metadata_extractor.py
- Verify the LLM model is responding correctly

#### "No documents returned"
- The Pinecone index may be empty
- Filters may be too restrictive
- Check vector search configuration

#### "Pipeline initialization failed"
- Missing API keys
- Invalid Pinecone index name
- Network connectivity issues

## Integration with CI/CD

This test can be integrated into CI/CD pipelines:

```bash
# Run as part of integration test suite
uv run python scripts/retrieval/test_metadata_integration.py

# Check exit code
if [ $? -eq 0 ]; then
    echo "Integration tests passed"
else
    echo "Integration tests failed"
    exit 1
fi
```

## Related Files

- `src/pipelines/retrieval/pipeline.py`: Main retrieval pipeline
- `src/pipelines/retrieval/processors/metadata_extractor.py`: Metadata extraction logic
- `src/pipelines/retrieval/search/vector_searcher.py`: Vector search with filtering
- `tests/unit/test_retrieval_pipeline.py`: Unit tests for pipeline
- `.kiro/specs/metadata-filtering-integration/`: Feature specification

## Notes

- This is an **integration test** that makes real API calls
- Tests may take several minutes to complete
- API costs will be incurred for OpenAI calls
- Results depend on the data in your Pinecone index
- The test validates the integration, not individual components (use unit tests for that)

## Future Enhancements

Potential improvements:
- Mock Pinecone for faster testing
- Parameterized test configurations
- Performance benchmarking
- Parallel test execution
- Test data fixtures
- Snapshot testing for LLM responses
