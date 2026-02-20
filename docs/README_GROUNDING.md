# 🛡️ RAG Grounding: Preventing LLM Hallucination

## Why Grounding Matters

In e-commerce applications, **accuracy is critical**. Your LLM must:
- ✅ Only recommend products that exist in your database
- ✅ Cite accurate prices, ratings, and specifications
- ✅ Admit when it doesn't have information
- ❌ Never make up product details or recommendations

**Without proper grounding, your LLM may hallucinate:**
- Non-existent products
- Incorrect prices or ratings
- Outdated specifications
- Unverified claims

## Quick Setup (2 minutes)

### 1. Enable Grounding in Config

Edit `config/inference.yaml`:

```yaml
generator:
  strict_grounding: true      # LLM ONLY uses retrieved context
  require_context: true       # No recommendations without data
  min_context_length: 50      # Minimum context quality
```

### 2. Test It

```bash
uv run python scripts/inference/test_grounding.py
```

### 3. Verify Behavior

Your assistant should now:
- ✅ Cite specific products from your database
- ✅ Include actual prices and ratings
- ✅ Say "I don't have information" for unknown products
- ✅ Refuse to recommend without context

## How It Works

### 5-Layer Defense Against Hallucination

1. **Strict System Prompts** - Explicit instructions to only use context
2. **Context Validation** - Checks context quality before generation
3. **Response Validation** - Validates responses for grounding issues
4. **Fallback Messages** - Safe responses when context insufficient
5. **Configuration Control** - Fine-tune behavior per environment

### Example: Grounded vs Ungrounded

**Query:** "What's the best phone under $500?"

**❌ Ungrounded (Bad):**
```
I recommend the iPhone 15 Pro or Samsung Galaxy S24. 
They're both excellent choices with great cameras...
```
*Problem: May not be in your database, prices may be wrong*

**✅ Grounded (Good):**
```
Based on products in our database, I recommend:

1. Samsung Galaxy A54 - $449.99 (Rating: 4.5/5)
   Great camera and long battery life

2. Google Pixel 7a - $499.00 (Rating: 4.6/5)
   Excellent camera and clean Android
```
*Cites actual products with real prices from your database*

**✅ Grounded (No Context):**
```
I don't have relevant product information in my database 
to answer your question. Could you try rephrasing or 
asking about different products?
```
*Admits lack of information instead of making things up*

## Configuration Options

| Setting | Effect | Recommended |
|---------|--------|-------------|
| `strict_grounding: true` | LLM ONLY uses provided context | ✅ Yes |
| `require_context: true` | Returns fallback without context | ✅ Yes |
| `min_context_length: 50` | Minimum context chars required | ✅ 50-100 |

## Documentation

- 📖 **Full Guide:** [docs/GROUNDING_GUIDE.md](./GROUNDING_GUIDE.md)
- ⚡ **Quick Reference:** [docs/GROUNDING_QUICK_REFERENCE.md](./GROUNDING_QUICK_REFERENCE.md)
- 🧪 **Test Script:** [scripts/inference/test_grounding.py](../scripts/inference/test_grounding.py)

## Troubleshooting

### Issue: Too many "I don't have information" messages

**Solution:**
- Lower `min_context_length` to 30-50
- Improve retrieval quality (better embeddings, query rewriting)
- Check if `require_context` is too strict for your use case

### Issue: Still seeing hallucinations

**Solution:**
- Verify `strict_grounding: true` in config
- Set LLM `temperature: 0.0` for deterministic responses
- Enable response validation
- Use a more instruction-following model

### Issue: Responses lack detail

**Solution:**
- Increase `max_context_tokens` to include more information
- Improve context formatting in retrieval pipeline
- Check if context truncation is too aggressive

## Best Practices

1. ✅ **Start strict** - Use maximum grounding initially
2. ✅ **Monitor** - Track fallback rate and user feedback
3. ✅ **Improve retrieval** - Better context = better responses
4. ✅ **Test regularly** - Run grounding tests after changes
5. ✅ **Audit responses** - Periodically review for accuracy

## Summary

**Grounding ensures your e-commerce assistant provides accurate, trustworthy information by:**
- Restricting responses to retrieved context
- Validating context quality before generation
- Providing safe fallbacks when data is unavailable
- Preventing the LLM from making up product information

**Result:** Your customers get accurate product information they can trust.

---

For detailed implementation details, see [GROUNDING_GUIDE.md](./GROUNDING_GUIDE.md)
