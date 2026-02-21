# Implementation Tasks: Metadata Filtering Integration

## Task 1: Configuration Setup

- [x] Add `metadata_extraction` section to `config/retrieval.yaml` with fields: `enabled: true`, `llm_model: "gpt-3.5-turbo"`, `temperature: 0.0`, `max_tokens: 200`, `timeout_seconds: 3`. This enables configurable control of the metadata extraction feature. (Req 6.1, 6.2, 6.3)
- [x] Add configuration fields to `RetrievalSettings` dataclass in `src/pipelines/retrieval/config.py`: `metadata_extraction_enabled: bool = True`, `metadata_extraction_model: str = "gpt-3.5-turbo"`, `metadata_extraction_timeout: int = 3`. Update `ConfigurationLoader.load_config()` to read these values from the YAML file's `metadata_extraction` section. (Req 6.1, 6.2, 6.3)
- [x] Update `ExtractorConfig` dataclass in `src/pipelines/retrieval/processors/metadata_extractor.py` to add `timeout_seconds: int = 3` field. This will be used to prevent extraction from blocking the pipeline. (Req 5.2, NFR-2)

## Task 2: Pipeline Integration - Add MetadataExtractor to RetrievalPipeline

- [x] Import `MetadataExtractor` and `ExtractorConfig` at the top of `src/pipelines/retrieval/pipeline.py`. Add `extractor_config: ExtractorConfig` field to the `RetrievalConfig` dataclass. Add `metadata_extractor: Optional[MetadataExtractor] = None` field to the `RetrievalPipeline` class `__init__` method. (Req 1.1, 2.1, 3.1, 4.1)
- [x] Update `RetrievalPipeline.from_settings()` classmethod to create an `ExtractorConfig` from settings: `extractor_config = ExtractorConfig(enabled=settings.metadata_extraction_enabled, llm_model=settings.metadata_extraction_model, timeout_seconds=settings.metadata_extraction_timeout)`. Add this config to the `RetrievalConfig` constructor. (Req 6.1, 6.2)
- [x] Update `RetrievalPipeline.initialize()` to initialize the metadata extractor after the query processor: `if self.config.extractor_config.enabled: self.logger.info("Initializing MetadataExtractor..."); self.metadata_extractor = MetadataExtractor(self.config.extractor_config, self.settings.openai_api_key)`. Add "MetadataExtractor" to the `components_initialized` list in the success log. (Req 1.1, 2.1, 3.1, 4.1)

## Task 3: Add Metadata Extraction Step to Retrieval Workflow

- [x] In `RetrievalPipeline._execute_retrieval_workflow()`, add a new extraction step after query processing (Step 1) and before vector search (Step 2). Only run extraction if `filters` parameter is `None` and `self.metadata_extractor` is not `None`. Wrap the extraction call in a try-except block to handle failures gracefully. (Req 1.1, 2.1, 3.1, 4.1, 5.1, 5.2)
- [x] Inside the extraction step, call `extracted_filters = self.metadata_extractor.extract(current_query)` and measure the time taken. If `extracted_filters` is not `None`, set `filters = extracted_filters` and add `metadata['auto_extracted_filters'] = True`. Log the extracted filters with INFO level: `self.logger.info(f"Auto-extracted filters: {filters.__dict__}", extra={'extra_fields': filters.__dict__})`. (Req 1.1, 1.2, 1.3, 1.4, 7.1, 7.2)
- [x] Add a workflow step entry to `metadata['workflow_steps']` with: `{'step': 'metadata_extraction', 'iteration': iteration, 'time_ms': step_time, 'filters_extracted': extracted_filters is not None, 'product_name': filters.product_name_pattern if filters else None, 'min_price': filters.min_price if filters else None, 'max_price': filters.max_price if filters else None, 'min_rating': filters.min_rating if filters else None}`. (Req 7.3, 7.4)
- [x] In the except block for extraction failures, log a warning: `self.logger.warning(f"Metadata extraction failed: {e}")` and continue without filters (graceful degradation). Do not re-raise the exception. (Req 5.1, 5.2, 5.3, 5.4, NFR-5, NFR-6)
- [x] Pass the `filters` variable (which may now contain extracted filters) to `self.vector_searcher.search(processed_query.embedding, filters)` in Step 2. The vector searcher already supports filters, so no changes needed there. (Req 1.1, 2.1, 3.1, 4.1)

## Task 4: Add Timeout Handling to MetadataExtractor

- [x] Update `MetadataExtractor.extract()` in `src/pipelines/retrieval/processors/metadata_extractor.py` to add timeout handling. Wrap the LLM call in a timeout mechanism using `asyncio.wait_for()` or a similar approach with `self.config.timeout_seconds`. If the timeout is exceeded, log a warning and return `None`. (Req 5.2, NFR-2, NFR-5)
- [x] Update the exception handling in `extract()` to catch `TimeoutError` specifically and log: `self.logger.warning("Metadata extraction timed out after {self.config.timeout_seconds}s")`. Return `None` on timeout to allow the pipeline to continue. (Req 5.2, NFR-2, NFR-5)

## Task 5: Testing - Unit Tests

- [x] Create or update unit tests in `tests/unit/test_retrieval_pipeline.py` (or a new file) to test metadata extraction integration. Mock the `MetadataExtractor` to return predefined filters and verify that `vector_searcher.search()` is called with those filters. (Req 1.1, 2.1, 3.1, 4.1)
- [x] Add test case for extraction disabled: set `extractor_config.enabled = False` and verify that `metadata_extractor` is `None` and no extraction occurs. (Req 6.2)
- [x] Add test case for extraction failure: mock `MetadataExtractor.extract()` to raise an exception and verify that the pipeline continues without filters and logs a warning. (Req 5.1, 5.2, 5.3, 5.4)
- [x] Add test case for extraction timeout: mock `MetadataExtractor.extract()` to raise `TimeoutError` and verify graceful handling. (Req 5.2, NFR-2)
- [x] Add test case for successful extraction: mock `MetadataExtractor.extract()` to return a `MetadataFilter` with product name and price, and verify that the result metadata includes `auto_extracted_filters: True` and the filter details. (Req 1.1, 7.2, 7.3)

## Task 6: Testing - Integration Tests

- [x] Create integration test script `scripts/retrieval/test_metadata_integration.py` that initializes a real `RetrievalPipeline` with metadata extraction enabled and tests end-to-end filtering with real OpenAI API calls (but can use a test Pinecone index or mock). (Req 1.1, 2.1, 3.1, 4.1)
- [x] Add test case for "iPhone 12 price" query: verify that the extracted filter includes `product_name_pattern="iPhone 12"` and that the retrieval result metadata shows `auto_extracted_filters: True`. (Req 1.1, 1.2, 1.3, 1.4)
- [ ] Add test case for "phones under $300" query: verify that the extracted filter includes `max_price=300` and that results are filtered accordingly. (Req 2.1, 2.2, 2.3, 2.4)
- [ ] Add test case for "highly rated Samsung phones over $400" query: verify that multiple filters are extracted (product name, price, rating) and applied together. (Req 3.1, 3.2, 3.3, 4.1, 4.2, 4.3)
- [ ] Add test case for a general query with no extractable metadata (e.g., "What are smartphones?"): verify that no filters are extracted and semantic search is used. (Req 5.1, 5.4)
- [ ] Run the existing test suite with `uv run pytest tests/ -v` to ensure no regressions were introduced by the changes. (NFR-8)

## Task 7: Documentation

- [ ] Update the main `README.md` to add a section on "Metadata Filtering" that explains the feature, how it works, and provides examples of queries that benefit from it (e.g., "iPhone 12 price", "phones under $300"). (Req 7.1, 7.2, 7.3)
- [ ] Add a configuration example to the README showing how to enable/disable metadata extraction in `config/retrieval.yaml`. Include the default settings and explain each field. (Req 6.1, 6.2, 6.3, 6.4)
- [ ] Add a troubleshooting section to the README or a separate doc explaining what to do if extraction is slow (adjust timeout), if extraction fails frequently (check OpenAI API key, check logs), or if results are too restrictive (disable feature or adjust extraction prompt). (Req 5.1, 5.2, 5.3, 5.4)
- [ ] Update `docs/GROUNDING_GUIDE.md` or create a new doc explaining how metadata filtering improves grounding by ensuring retrieved context is precisely relevant to the user's query. (Req 1.1, 2.1, 3.1, 4.1)

## Task 8: Validation and Performance Testing

- [ ] Run performance benchmarks using `scripts/inference/benchmark.py` or a similar script to measure the latency impact of metadata extraction. Compare P50, P95, and P99 latencies with and without extraction enabled. Verify that P95 latency increase is ≤ 500ms. (NFR-1, NFR-2)
- [ ] Test with various query types (product-specific, price-based, rating-based, general) and manually evaluate the quality of results. Verify that product-specific queries return 90%+ relevant results. (Success Metric 1)
- [ ] Verify that cache behavior works correctly with extracted filters: same query should hit cache, different filters should miss cache. Check cache keys include filter information. (Req 7.4, NFR-3)
- [ ] Monitor error rates and extraction success rates during testing. Verify that extraction failures don't crash the pipeline and that fallback to semantic search works 100% of the time. (NFR-4, NFR-5, Success Metric 4)
