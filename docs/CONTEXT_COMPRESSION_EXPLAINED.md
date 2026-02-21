# Context Compression Explained

## Overview

Context compression is a critical step in the retrieval pipeline that **filters out irrelevant documents** before they're sent to the LLM for response generation. This improves response quality and reduces token costs.

## The Problem It Solves

When you perform vector search, you might retrieve documents that are:
- **Semantically similar** but not actually relevant to the query
- **Tangentially related** but don't help answer the question
- **Noise** that could confuse the LLM or dilute the context

**Example:**
```
Query: "What's the best phone for photography?"

Vector Search Returns:
1. iPhone 14 Pro - Great camera specs ✅ RELEVANT
2. Samsung Galaxy S23 - Excellent camera ✅ RELEVANT  
3. Phone case with camera protection ❌ NOT RELEVANT
4. Photography tips article ❌ NOT RELEVANT
5. Camera accessories ❌ NOT RELEVANT
```

Without compression, all 5 documents would be sent to the LLM, wasting tokens and potentially confusing the response.

## How It Works

### Step-by-Step Process

```
┌─────────────────────────────────────────────────────────────┐
│                    CONTEXT COMPRESSION                       │
└─────────────────────────────────────────────────────────────┘

Input: Query + Retrieved Documents (from vector search)
                        ↓
        ┌───────────────────────────────┐
        │  For Each Document:           │
        │                               │
        │  1. Create Evaluation Prompt  │
        │     "Is this doc relevant     │
        │      to answering the query?" │
        │                               │
        │  2. Send to LLM (GPT-3.5)    │
        │     - Fast & cheap model      │
        │     - Temperature = 0.0       │
        │     - Max tokens = 10         │
        │                               │
        │  3. Parse Response            │
        │     "Yes" → Keep (score 1.0)  │
        │     "No"  → Filter (score 0.0)│
        │                               │
        └───────────────────────────────┘
                        ↓
Output: Filtered Documents (only relevant ones)
```

### The Evaluation Prompt

The default prompt used for each document:

```
Given the following question and document, determine if the document 
is relevant to answering the question. Consider the document relevant 
if it contains information that could help answer the question, even 
if it doesn't contain the complete answer.

Question: {question}

Document: {context}

Is this document relevant? Answer with 'Yes' if relevant, 'No' if not relevant.
```

### LLM Configuration

The compressor uses a **lightweight, cost-effective** LLM:
- **Model**: `gpt-3.5-turbo` (fast and cheap)
- **Temperature**: `0.0` (deterministic, consistent decisions)
- **Max Tokens**: `10` (just enough for "Yes" or "No")

This keeps costs low while maintaining quality filtering.

## Real Example from Your System

From the test output you saw:

```
INFO Compression completed: 2 -> 1 documents
```

This means:
1. **Vector search** found 2 documents
2. **Compression** evaluated both documents
3. **Result**: Only 1 document was truly relevant
4. The irrelevant document was filtered out

### Another Example:

```
INFO Compression completed: 2 -> 0 documents
INFO Relevance score 0.000 below threshold 0.500, triggering rewrite
```

This means:
1. Vector search found 2 documents
2. Both were evaluated as **not relevant** (score 0.0)
3. Since no relevant docs were found, the system **triggers query rewriting**
4. The query is reformulated and search is attempted again

## Configuration Options

### Enable/Disable Compression

```yaml
# config/retrieval.yaml
compression_enabled: true  # Turn on/off
```

### Custom Relevance Prompt

```yaml
relevance_prompt: |
  Your custom prompt here...
  Question: {question}
  Document: {context}
```

### Relevance Threshold

```python
relevance_threshold: 0.5  # Minimum score to keep document
```

## Benefits

### 1. **Improved Response Quality**
- Only relevant context reaches the LLM
- Reduces confusion from irrelevant information
- More focused, accurate responses

### 2. **Cost Reduction**
- Fewer tokens sent to the expensive generation LLM
- Compression uses cheap GPT-3.5-turbo
- Can reduce costs by 30-50% in some cases

### 3. **Better Grounding**
- Filters out tangentially related content
- Reduces hallucination risk
- Ensures responses are based on truly relevant data

### 4. **Automatic Quality Control**
- Acts as a quality gate for retrieval
- Catches poor vector search results
- Triggers query rewriting when needed

## Performance Metrics

The compression result includes:

```python
CompressionResult(
    documents=[...],              # Filtered documents
    filtered_count=3,             # How many were removed
    relevance_scores=[1.0, 0.5],  # Score for each doc
    compression_ratio=0.4,        # 40% of docs kept
    processing_time_ms=840.68     # Time taken
)
```

### Typical Performance:
- **Processing time**: 200-1000ms per batch
- **Compression ratio**: 0.3-0.7 (30-70% of docs kept)
- **Cost per evaluation**: ~$0.0001 per document

## Integration with Query Rewriting

Context compression works hand-in-hand with query rewriting:

```
┌──────────────────────────────────────────────────────────┐
│  Retrieval Pipeline with Compression & Rewriting         │
└──────────────────────────────────────────────────────────┘

Query → Vector Search → Compression
                            ↓
                    Calculate Avg Score
                            ↓
                ┌───────────┴───────────┐
                ↓                       ↓
        Score >= 0.5              Score < 0.5
        Keep Results              Rewrite Query
                ↓                       ↓
        Format Context          Try Search Again
                ↓                       ↓
        Send to LLM            (Max 3 attempts)
```

## Error Handling

The compressor is designed to **fail gracefully**:

1. **If compression fails**: Returns all documents (better to have too much context than none)
2. **If evaluation unclear**: Defaults to keeping the document (err on side of inclusion)
3. **If LLM unavailable**: Disables compression and passes all docs through

## Code Example

Here's how compression is used in the retrieval pipeline:

```python
# Step 1: Vector search returns documents
search_result = vector_searcher.search(query_embedding)
# Returns: [doc1, doc2, doc3, doc4, doc5]

# Step 2: Compress to filter irrelevant docs
compression_result = context_compressor.compress(
    query="What's the best phone for photography?",
    documents=search_result.documents
)
# Returns: [doc1, doc2] (only relevant ones)

# Step 3: Check if results are good enough
avg_relevance = sum(compression_result.relevance_scores) / len(...)
if avg_relevance < 0.5:
    # Trigger query rewriting
    rewritten_query = query_rewriter.rewrite(query)
    # Try again with better query
```

## When to Adjust Settings

### Increase Filtering (More Strict)
```yaml
relevance_threshold: 0.7  # Only keep highly relevant docs
```
**Use when**: You have many documents and want only the best

### Decrease Filtering (More Lenient)
```yaml
relevance_threshold: 0.3  # Keep more documents
```
**Use when**: You have few documents and can't afford to lose any

### Disable Compression
```yaml
compression_enabled: false
```
**Use when**: 
- Testing/debugging
- Very high-quality vector search
- Cost is not a concern

## Monitoring Compression

Key metrics to watch:

1. **Compression Ratio**: Should be 0.3-0.7 typically
   - Too low (<0.2): Vector search may be poor quality
   - Too high (>0.9): Compression may not be filtering enough

2. **Processing Time**: Should be <1000ms per batch
   - Higher times indicate LLM latency issues

3. **Rewrite Triggers**: How often does low relevance trigger rewrites?
   - Frequent rewrites suggest vector search needs tuning

## Summary

**Context Compression** is like having a smart assistant who:
1. Reviews all the documents retrieved by search
2. Asks "Does this actually help answer the question?"
3. Filters out the noise
4. Only passes relevant information to the final LLM

This results in **better responses, lower costs, and more reliable grounding** in your RAG system.
