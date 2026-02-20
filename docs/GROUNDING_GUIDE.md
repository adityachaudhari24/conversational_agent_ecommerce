# RAG Grounding Guide: Preventing LLM Hallucination

## Overview

This guide explains how to ensure your e-commerce conversational agent provides responses that are **completely grounded** in retrieved context from your RAG pipeline, preventing the LLM from making up product information or recommendations.

## The Problem: LLM Hallucination

Without proper grounding, LLMs may:
- Recommend products that don't exist in your database
- Invent prices, ratings, or specifications
- Provide outdated or incorrect product information
- Make claims not supported by your data

## The Solution: Multi-Layer Grounding Strategy

Our implementation uses a **5-layer defense** against hallucination:

### Layer 1: Strict System Prompts
The LLM receives explicit instructions to ONLY use provided context.

### Layer 2: Context Validation
Checks if retrieved context is sufficient before generation.

### Layer 3: Response Validation
Validates generated responses for grounding issues.

### Layer 4: Fallback Messages
Returns safe fallback messages when context is insufficient.

### Layer 5: Configuration Control
Allows fine-tuning grounding behavior per environment.

---

## Configuration

### Quick Start: Maximum Grounding (Recommended)

In `config/inference.yaml`:

```yaml
generator:
  strict_grounding: true      # Use strict grounding prompts
  require_context: true       # Refuse to answer without context
  min_context_length: 50      # Minimum context characters required
```

This configuration ensures:
- ✅ LLM ONLY uses retrieved product data
- ✅ No product recommendations without database context
- ✅ Safe fallback messages when data is unavailable
- ✅ Explicit citations from context

### Configuration Options

#### `strict_grounding` (boolean, default: true)

**When true:**
- Uses strict system prompt with explicit grounding rules
- LLM instructed to NEVER make up product information
- Must cite specific products from context
- Says "I don't know" rather than guessing

**When false:**
- Uses softer prompt that prioritizes but doesn't require context
- LLM can use general knowledge alongside context
- More flexible but higher hallucination risk

**Recommendation:** Keep `true` for e-commerce applications.

#### `require_context` (boolean, default: true)

**When true:**
- Validates context quality before generation
- Returns fallback message if context is insufficient
- Prevents responses without product data

**When false:**
- Allows generation even without context
- LLM may provide general responses
- Higher risk of ungrounded information

**Recommendation:** Keep `true` for product recommendations, set `false` only for general FAQ.

#### `min_context_length` (integer, default: 50)

Minimum number of characters required in retrieved context.

**Guidelines:**
- `50-100`: Basic validation (recommended minimum)
- `100-200`: Ensures substantial product information
- `200+`: Very strict, may reject valid but brief results
- `0`: Disables length check (not recommended)

**Recommendation:** Start with `50`, increase if you see quality issues.

---

## How It Works

### 1. Strict Grounding System Prompt

When `strict_grounding: true`, the LLM receives:

```
You are a helpful e-commerce assistant specializing in phone products.

CRITICAL RULES - YOU MUST FOLLOW THESE STRICTLY:
1. ONLY recommend or mention products that are explicitly provided in the CONTEXT section below
2. NEVER make up product names, prices, ratings, or specifications
3. If the context doesn't contain relevant products, clearly state: 
   "I don't have information about products matching your query in my database."
4. When recommending products, ALWAYS cite specific details from the context (name, price, rating)
5. If asked about a product not in the context, respond: 
   "I don't have information about that specific product in my current database."
6. Do NOT use your general knowledge about phones - ONLY use the provided context

CONTEXT:
{retrieved_context}

Remember: It's better to say "I don't know" than to provide ungrounded information.
```

### 2. Context Quality Validation

Before generating a response, the system checks:

```python
# Check if context exists
if not context or not context.strip():
    return fallback_message

# Check minimum length
if len(context) < min_context_length:
    return fallback_message

# Check for product indicators
if not contains_product_info(context):
    return fallback_message
```

### 3. Response Validation (Optional)

After generation, validates the response for suspicious patterns:

```python
# Patterns that indicate potential hallucination
suspicious_patterns = [
    "i recommend",
    "you should buy", 
    "the best phone",
    "price is",
    "costs $"
]

# If no context but response makes specific claims
if not context and has_suspicious_pattern(response):
    return fallback_message
```

### 4. Fallback Messages

When context is insufficient:

```
"I don't have relevant product information in my database to answer your question. 
Could you try rephrasing or asking about different products?"
```

---

## Usage Examples

### Example 1: Product Recommendation with Context

**User Query:** "What's the best phone under $500?"

**Retrieved Context:**
```
Product: Samsung Galaxy A54
Price: $449.99
Rating: 4.5/5
Reviews: Great camera, long battery life...

Product: Google Pixel 7a  
Price: $499.00
Rating: 4.6/5
Reviews: Excellent camera, clean Android...
```

**Response (Grounded):**
```
Based on the products in our database, I can recommend two excellent phones under $500:

1. Samsung Galaxy A54 - $449.99 (Rating: 4.5/5)
   Customers praise its great camera and long battery life.

2. Google Pixel 7a - $499.00 (Rating: 4.6/5)
   Known for its excellent camera and clean Android experience.

Both are highly rated options within your budget.
```

### Example 2: Query Without Context

**User Query:** "What's the best phone under $500?"

**Retrieved Context:** (empty - no products found)

**Response (Grounded):**
```
I don't have relevant product information in my database to answer your question. 
Could you try rephrasing or asking about different products?
```

**NOT:** "I recommend the iPhone 15 or Samsung Galaxy S24..." ❌

### Example 3: Specific Product Not in Database

**User Query:** "Tell me about the iPhone 16 Pro Max"

**Retrieved Context:** (contains other iPhones but not iPhone 16 Pro Max)

**Response (Grounded):**
```
I don't have information about the iPhone 16 Pro Max in my current database. 
However, I can help you with information about other iPhone models we have available.
```

---

## Advanced Configuration

### Custom Fallback Messages

Customize the message shown when context is insufficient:

```python
from src.pipelines.inference.grounding import GroundingConfig

grounding_config = GroundingConfig(
    strict_mode=True,
    require_context=True,
    fallback_message="Sorry, I couldn't find matching products. Try browsing our catalog or contact support."
)
```

### Programmatic Configuration

Override config in code:

```python
from src.pipelines.inference.generation.generator import ResponseGenerator
from src.pipelines.inference.grounding import GroundingConfig

# Create custom grounding config
grounding_config = GroundingConfig(
    strict_mode=True,           # Strict grounding
    require_context=True,       # Require context
    min_context_length=100,     # Higher threshold
    enable_validation=True      # Enable response validation
)

# Initialize generator with grounding
generator = ResponseGenerator(
    config=generator_config,
    llm_client=llm_client,
    grounding_config=grounding_config
)
```

### Environment Variables

Override via environment:

```bash
# Enable strict grounding
export INFERENCE_GENERATOR_STRICT_GROUNDING=true

# Require context for all responses
export INFERENCE_GENERATOR_REQUIRE_CONTEXT=true

# Set minimum context length
export INFERENCE_GENERATOR_MIN_CONTEXT_LENGTH=100
```

---

## Testing Grounding

### Test Script

Create `scripts/test_grounding.py`:

```python
from src.pipelines.inference.pipeline import InferencePipeline
from src.pipelines.retrieval.pipeline import RetrievalPipeline

# Initialize pipelines
retrieval_pipeline = RetrievalPipeline.from_config_file()
retrieval_pipeline.initialize()

inference_pipeline = InferencePipeline.from_config_file(retrieval_pipeline)
inference_pipeline.initialize()

# Test cases
test_queries = [
    "What's the best phone?",                    # Should retrieve context
    "Tell me about the XYZ-9000 SuperPhone",     # Non-existent product
    "What's your return policy?",                # General question
]

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    
    result = inference_pipeline.generate(query)
    print(f"Response: {result.response}")
    print(f"Metadata: {result.metadata}")
```

### Expected Behaviors

✅ **Good (Grounded):**
- Cites specific products from context
- Includes prices and ratings from context
- Says "I don't have information" when appropriate
- Refuses to recommend non-existent products

❌ **Bad (Hallucinated):**
- Recommends products not in context
- Invents prices or specifications
- Makes claims without context support
- Provides outdated information

---

## Monitoring and Debugging

### Enable Debug Logging

```yaml
# config/inference.yaml
logging:
  level: DEBUG
  log_token_usage: true
  log_latency: true
```

### Check Context Quality

Add logging to see what context is retrieved:

```python
import logging

logger = logging.getLogger(__name__)

# In your code
retrieval_result = retrieval_pipeline.retrieve(query)
logger.info(f"Retrieved context length: {len(retrieval_result.formatted_context)}")
logger.debug(f"Context preview: {retrieval_result.formatted_context[:200]}...")
```

### Monitor Fallback Rate

Track how often fallback messages are returned:

```python
# Add metrics tracking
fallback_count = 0
total_queries = 0

# In generation logic
if response == grounding_config.fallback_message:
    fallback_count += 1

fallback_rate = fallback_count / total_queries
logger.info(f"Fallback rate: {fallback_rate:.2%}")
```

---

## Best Practices

### 1. Start Strict, Relax Gradually

Begin with maximum grounding:
```yaml
strict_grounding: true
require_context: true
min_context_length: 100
```

Only relax if you have specific reasons and understand the risks.

### 2. Improve Retrieval Quality

Better retrieval = better grounding:
- Tune your embedding model
- Optimize chunk sizes
- Improve query rewriting
- Increase top_k for more context

### 3. Monitor User Feedback

Track when users report incorrect information:
- Log queries that led to issues
- Review context quality for those queries
- Adjust grounding thresholds accordingly

### 4. Use Different Settings for Different Query Types

```python
# Strict for product recommendations
product_grounding = GroundingConfig(strict_mode=True, require_context=True)

# Relaxed for general questions
general_grounding = GroundingConfig(strict_mode=False, require_context=False)
```

### 5. Regular Audits

Periodically review:
- Sample of generated responses
- Fallback message frequency
- User satisfaction metrics
- Accuracy of product information

---

## Troubleshooting

### Issue: Too Many Fallback Messages

**Symptoms:** Users frequently see "I don't have information" messages

**Solutions:**
1. Lower `min_context_length` threshold
2. Improve retrieval pipeline (better embeddings, query rewriting)
3. Expand product database
4. Check if `require_context` is too strict for your use case

### Issue: Still Seeing Hallucinations

**Symptoms:** LLM recommends non-existent products despite grounding

**Solutions:**
1. Verify `strict_grounding: true` in config
2. Check system prompt is being applied correctly
3. Enable response validation
4. Review LLM temperature (should be 0.0 for deterministic)
5. Consider using a more instruction-following model

### Issue: Responses Too Generic

**Symptoms:** Responses lack detail even with good context

**Solutions:**
1. Increase `max_context_tokens` to include more information
2. Improve context formatting in retrieval pipeline
3. Adjust system prompt to encourage detailed responses
4. Check if context truncation is too aggressive

---

## Summary

**For Maximum Grounding (Recommended for E-commerce):**

```yaml
generator:
  strict_grounding: true
  require_context: true
  min_context_length: 50
  max_context_tokens: 3000

llm:
  temperature: 0.0  # Deterministic responses
```

**Key Principles:**
1. ✅ Always validate context before generation
2. ✅ Use explicit grounding instructions in prompts
3. ✅ Prefer "I don't know" over hallucination
4. ✅ Monitor and audit responses regularly
5. ✅ Improve retrieval quality continuously

**Remember:** Grounding is not just about configuration—it's about the entire RAG pipeline working together to provide accurate, trustworthy information to your users.
