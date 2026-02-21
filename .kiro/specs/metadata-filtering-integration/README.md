# Metadata Filtering Integration Spec

## Quick Summary

**Goal:** Automatically extract and apply metadata filters (product names, prices, ratings) from user queries to improve search precision.

**Problem:** When users search "iPhone 12 price", the system returns semantically similar products (iPhone 11, 13, etc.) instead of filtering to only iPhone 12.

**Solution:** Integrate the existing `MetadataExtractor` component into the retrieval pipeline to automatically extract filters and apply them at the vector database level.

## Key Changes

1. **Configuration** - Add metadata extraction settings to `config/retrieval.yaml`
2. **Pipeline Integration** - Add extraction step before vector search in `RetrievalPipeline`
3. **Error Handling** - Graceful fallback to semantic search if extraction fails
4. **Testing** - Comprehensive tests to ensure quality and no regressions

## Expected Benefits

- **Precision**: 90%+ relevant results for product-specific queries
- **User Experience**: More accurate answers to specific product questions
- **Efficiency**: Less post-processing needed by LLM

## Expected Costs

- **Latency**: +200-500ms per query (LLM extraction call)
- **API Cost**: ~$0.0035 per 1000 queries (GPT-3.5 calls)
- **Complexity**: Minimal - reusing existing components

## Timeline

- **Phase 1-2**: Configuration + Integration - 1.5 hours
- **Phase 3-4**: Error Handling + Testing - 1.5 hours
- **Phase 5-6**: Documentation + Validation - 1 hour
- **Total**: ~4 hours

## Files to Modify

1. `config/retrieval.yaml` - Add metadata extraction config
2. `src/pipelines/retrieval/config.py` - Add config fields
3. `src/pipelines/retrieval/pipeline.py` - Integrate extractor
4. `src/pipelines/retrieval/processors/metadata_extractor.py` - Add timeout
5. Tests - Update and add new tests

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Extraction adds too much latency | 3-second timeout, configurable disable |
| Extraction fails frequently | Graceful fallback to semantic search |
| Breaking existing functionality | Comprehensive testing, feature flag |
| Increased API costs | Configurable, monitor usage |

## Success Criteria

✅ "iPhone 12 price" returns only iPhone 12 documents  
✅ "phones under $300" returns only products ≤ $300  
✅ Extraction failures don't break retrieval  
✅ Latency increase ≤ 500ms (P95)  
✅ No regressions in existing tests  

## Next Steps

1. Review this spec
2. Get approval to proceed
3. Execute tasks in order
4. Test thoroughly
5. Deploy with monitoring

## Questions?

- Should extraction be enabled by default? **Yes**
- What if extraction is too slow? **3-second timeout + fallback**
- What if no results match filters? **Fall back to semantic search**
- Can we disable it? **Yes, via config file**
