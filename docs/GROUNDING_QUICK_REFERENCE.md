# RAG Grounding Quick Reference

## TL;DR - Prevent Hallucination

**Add to `config/inference.yaml`:**

```yaml
generator:
  strict_grounding: true      # ← LLM ONLY uses retrieved context
  require_context: true       # ← No recommendations without data
  min_context_length: 50      # ← Minimum context quality check
```

**Result:** LLM won't make up products, prices, or recommendations.

---

## Configuration Options

| Setting | Default | Recommended | Effect |
|---------|---------|-------------|--------|
| `strict_grounding` | `true` | `true` | Uses strict prompt: "ONLY use provided context" |
| `require_context` | `true` | `true` | Returns fallback if no context retrieved |
| `min_context_length` | `50` | `50-100` | Minimum chars required in context |

---

## What Each Setting Does

### `strict_grounding: true`
✅ LLM instructed to NEVER make up information  
✅ Must cite specific products from context  
✅ Says "I don't know" instead of guessing  
❌ Less flexible for general questions  

### `require_context: true`
✅ Validates context before generation  
✅ Returns safe fallback if context insufficient  
✅ Prevents ungrounded product recommendations  
❌ May refuse to answer valid general questions  

### `min_context_length: 50`
✅ Ensures minimum context quality  
✅ Catches empty/truncated retrievals  
❌ May reject valid but brief results  

---

## Quick Test

```bash
# Run grounding test
uv run python scripts/inference/test_grounding.py
```

Expected behavior:
- ✅ Cites products from database with prices/ratings
- ✅ Says "I don't have information" for non-existent products
- ✅ Refuses to recommend without context
- ❌ Never invents product names or specifications

---

## Environment Variables

Override config via environment:

```bash
export INFERENCE_GENERATOR_STRICT_GROUNDING=true
export INFERENCE_GENERATOR_REQUIRE_CONTEXT=true
export INFERENCE_GENERATOR_MIN_CONTEXT_LENGTH=100
```

---

## Programmatic Override

```python
from src.pipelines.inference.grounding import GroundingConfig

grounding = GroundingConfig(
    strict_mode=True,
    require_context=True,
    min_context_length=100,
    fallback_message="Custom fallback message"
)

generator = ResponseGenerator(config, llm_client, grounding)
```

---

## Troubleshooting

### Too many "I don't have information" responses?
- Lower `min_context_length` (try 30-50)
- Improve retrieval quality
- Check if `require_context` is too strict

### Still seeing hallucinations?
- Verify `strict_grounding: true`
- Set LLM `temperature: 0.0`
- Enable response validation
- Check system prompt is applied

### Responses too generic?
- Increase `max_context_tokens`
- Improve context formatting
- Check context truncation

---

## Best Practices

1. **Start strict:** Use all grounding features initially
2. **Monitor:** Track fallback rate and user feedback
3. **Improve retrieval:** Better context = better responses
4. **Test regularly:** Run grounding tests after changes
5. **Audit responses:** Periodically review for accuracy

---

## See Also

- **Full Guide:** `docs/GROUNDING_GUIDE.md`
- **Test Script:** `scripts/inference/test_grounding.py`
- **Config File:** `config/inference.yaml`
