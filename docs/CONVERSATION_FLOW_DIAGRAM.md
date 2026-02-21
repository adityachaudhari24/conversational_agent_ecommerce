# Conversation Flow Diagram — End-to-End Technical Reference

> Last updated: February 2026
> Audience: Engineering team — covers every layer, data shape, and decision point.

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OFFLINE (One-Time)                                │
│                                                                             │
│   CSV Data ──► Ingestion Pipeline ──► Pinecone Vector DB                    │
│   (phones_reviews.csv)   │                  (ecommerce-products /           │
│                          │                   phone-reviews namespace)        │
│                          ▼                                                  │
│              Load → Process → Chunk → Embed → Store                         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           ONLINE (Per-Request)                              │
│                                                                             │
│   Streamlit UI ──► FastAPI Backend ──► Inference Pipeline                   │
│   (port 8501)       (port 8000)          │                                  │
│                                          ├──► Agentic Workflow (LangGraph)  │
│                                          │       │                          │
│                                          │       ├──► Retrieval Pipeline    │
│                                          │       │       │                  │
│                                          │       │       └──► Pinecone      │
│                                          │       │                          │
│                                          │       └──► OpenAI GPT-4o-mini    │
│                                          │                                  │
│                                          └──► ConversationStore (disk JSON) │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Ingestion Pipeline (Offline)

Runs once (or on data refresh) to populate the vector database.

**Entry point:** `python -m src.pipelines.ingestion` or `src/pipelines/ingestion/pipeline.py`

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Stage 1:    │    │  Stage 2:    │    │  Stage 3:    │    │  Stage 4a:   │    │  Stage 4b:   │
│  CSV Loader  │───►│  Text        │───►│  Text        │───►│  Embedding   │───►│  Vector      │
│              │    │  Processor   │    │  Chunker     │    │  Generator   │    │  Store       │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

### Stage 1 — Document Loader (`loaders/document_loader.py`)

| Aspect | Detail |
|--------|--------|
| Input | `data/phones_reviews.csv` |
| Required columns | `product_name`, `description`, `price`, `rating`, `review_title`, `review_text` |
| Validation | File existence, column presence, empty-row filtering |
| Output | `pd.DataFrame` with validated rows |
| Error | `FileNotFoundError`, `ValidationError` |

### Stage 2 — Text Processor (`processors/text_processor.py`)

| Aspect | Detail |
|--------|--------|
| Input | `pd.DataFrame` |
| Transform | Each row → LangChain `Document` |
| Content format | `"Product: {name} \| Price: ${price} \| Rating: {rating}/5 \| Review: {text}"` |
| Metadata | `product_name`, `description`, `price`, `rating`, `review_title` (NaN → `"N/A"` or `""`) |
| Skipped records | Empty/whitespace-only `review_text` rows are tracked |
| Output | `List[Document]` |
| Quality gate | Abort if skip rate > 50% (`abort_threshold`) |

### Stage 3 — Text Chunker (`processors/text_chunker.py`)

| Aspect | Detail |
|--------|--------|
| Splitter | `RecursiveCharacterTextSplitter` |
| Chunk size | 1000 chars (default) |
| Overlap | 200 chars |
| Separators | `\n\n`, `\n`, `. `, ` `, `""` |
| Metadata | Original metadata preserved; `chunk_index` and `total_chunks` added |
| Pass-through | Documents smaller than chunk size are kept as-is |
| Output | `List[Document]` (≥ input count) |

### Stage 4a — Embedding Generator (`processors/embedding_generator.py`)

| Aspect | Detail |
|--------|--------|
| Model | `text-embedding-3-large` (OpenAI) |
| Dimension | 3072 |
| Batch size | 100 documents per API call |
| Validation | Each embedding checked for correct dimension |
| Failed docs | Tracked separately; quality gate applies |
| Output | `List[Tuple[Document, List[float]]]` |

### Stage 4b — Vector Store Manager (`storage/vector_store.py`)

| Aspect | Detail |
|--------|--------|
| Database | Pinecone (serverless, AWS us-east-1) |
| Index | `ecommerce-products` |
| Namespace | `phone-reviews` |
| Metric | Cosine similarity |
| Document IDs | Deterministic SHA-256 hash of `page_content + product_name + review_title` → enables deduplication |
| Metadata stored | All fields as strings |
| Output | List of stored document IDs |

---

## 3. Application Startup (FastAPI Lifespan)

**File:** `src/api/main.py`

```
Application Start
       │
       ▼
┌─────────────────────────────────────────────────┐
│ 1. RetrievalPipeline.from_config_file()         │
│    └─ Loads config/retrieval.yaml                │
│    └─ Reads env vars: OPENAI_API_KEY,            │
│       PINECONE_API_KEY, PINECONE_INDEX_NAME      │
│                                                   │
│ 2. retrieval_pipeline.initialize()               │
│    └─ QueryProcessor (OpenAI embeddings)         │
│    └─ MetadataExtractor (gpt-3.5-turbo)          │
│    └─ VectorSearcher (Pinecone connection)       │
│    └─ ContextCompressor (gpt-3.5-turbo)          │
│    └─ QueryRewriter (gpt-3.5-turbo)              │
│    └─ DocumentFormatter                          │
│    └─ ResultCache (in-memory TTL cache)          │
│                                                   │
│ 3. create_settings_from_yaml()                   │
│    └─ Loads config/inference.yaml                │
│                                                   │
│ 4. ConversationStore(storage_dir="data/sessions")│
│    └─ File-based JSON persistence with fcntl     │
│                                                   │
│ 5. InferencePipeline(config, retrieval, store)   │
│    └─ LLMClient (OpenAI gpt-4o-mini)            │
│    └─ ResponseGenerator (with GroundingConfig)   │
│    └─ AgenticWorkflow (LangGraph StateGraph)     │
│       └─ QueryReformulator                       │
│                                                   │
│ 6. Register singletons via DI (dependencies.py)  │
│    └─ set_inference_pipeline(pipeline)           │
│    └─ set_conversation_store(store)              │
└─────────────────────────────────────────────────┘
       │
       ▼
  FastAPI ready on port 8000
  Routes: /api/chat, /api/chat/stream,
          /api/sessions, /api/health
```

---

## 4. Frontend Layer (Streamlit)

**Entry point:** `src/frontend/app.py` → runs on port 8501

```
┌─────────────────────────────────────────────────────────────────┐
│                     Streamlit Application                        │
│                                                                  │
│  ┌──────────────┐    ┌──────────────────────────────────────┐   │
│  │   Sidebar     │    │         Chat Interface                │   │
│  │              │    │                                        │   │
│  │ • TechStore  │    │  ┌──────────────────────────────────┐ │   │
│  │   branding   │    │  │  Message History Display          │ │   │
│  │              │    │  │  (st.chat_message for each msg)   │ │   │
│  │ • New Chat   │    │  └──────────────────────────────────┘ │   │
│  │   button     │    │                                        │   │
│  │              │    │  ┌──────────────────────────────────┐ │   │
│  │ • Chat       │    │  │  st.chat_input                    │ │   │
│  │   History    │    │  │  (max 500 chars)                  │ │   │
│  │   (sessions) │    │  └──────────────────────────────────┘ │   │
│  └──────────────┘    └──────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Frontend → Backend Communication (`utils/api_client.py`)

Uses `httpx` client. All calls go to `http://localhost:8009/api/...`

| Action | Method | Endpoint | Request Body | Response |
|--------|--------|----------|-------------|----------|
| Create session | POST | `/api/sessions` | — | `{session_id, created_at, ...}` |
| List sessions | GET | `/api/sessions` | — | `{sessions: [...], total}` |
| Load session | GET | `/api/sessions/{id}` | — | `{session_id, messages: [...]}` |
| Send message | POST | `/api/chat` | `{query, session_id}` | `{response, session_id, metadata}` |
| Stream message | POST | `/api/chat/stream` | `{query, session_id}` | SSE stream: `data: {chunk, done}` |
| Health check | GET | `/api/health` | — | `{status}` |

### Streaming UX Flow

```
User types message
       │
       ▼
Add user message to st.session_state.messages
       │
       ▼
Display user message with st.chat_message("user")
       │
       ▼
Open st.chat_message("assistant") with placeholder
       │
       ▼
Call api_client.stream_message(prompt, session_id)
       │
       ▼
For each SSE chunk:
  ├─ Append chunk to full_response
  ├─ Update placeholder: full_response + "▌" (cursor)
  └─ Continue until done=True
       │
       ▼
Final display without cursor
       │
       ▼
Append assistant message to st.session_state.messages
```

---

## 5. API Layer (FastAPI)

### Request Validation (Pydantic Schemas)

```python
ChatRequest:
  query: str       # 1-500 chars, required
  session_id: str  # non-empty, required

ChatResponse:
  response: str
  session_id: str
  metadata: Dict[str, Any]  # workflow steps, latency, etc.
```

### Chat Endpoint Flow (`/api/chat`)

```
POST /api/chat {query, session_id}
       │
       ▼
Pydantic validation (ChatRequest)
       │
       ▼
Verify session exists (ConversationStore.get_session)
  └─ 404 if not found
       │
       ▼
Check pipeline availability
  └─ 503 if not initialized
       │
       ▼
inference_pipeline.agenerate(query, session_id)
       │
       ▼
Return ChatResponse {response, session_id, metadata}
```

### Streaming Endpoint Flow (`/api/chat/stream`)

```
POST /api/chat/stream {query, session_id}
       │
       ▼
Same validation + session check
       │
       ▼
Return StreamingResponse (text/event-stream)
       │
       ▼
For each chunk from inference_pipeline.stream():
  yield  data: {"chunk": "...", "done": false}\n\n
       │
       ▼
Final event:
  yield  data: {"chunk": "", "done": true, "total_length": N}\n\n
```

---

## 6. Inference Pipeline — The Core Engine

**File:** `src/pipelines/inference/pipeline.py`

### High-Level Flow

```
inference_pipeline.agenerate(query, session_id)
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 0: Ensure session exists (auto-create if needed)       │
│         ConversationStore.get_session(session_id)           │
│         └─ Creates new session file if missing              │
│                                                              │
│ Step 1: Load conversation history from disk                 │
│         ConversationStore.get_langchain_messages(session_id) │
│         └─ Returns last N messages as HumanMessage/AIMessage│
│         └─ N = max_history_length (default: 10)             │
│                                                              │
│ Step 2: Execute agentic workflow                            │
│         AgenticWorkflow.arun(query, history)                │
│         └─ See Section 7 for full workflow detail           │
│                                                              │
│ Step 3: Persist messages to disk                            │
│         ConversationStore.add_message(sid, "user", query)   │
│         ConversationStore.add_message(sid, "assistant", resp)│
│         └─ Writes JSON with fcntl exclusive lock            │
│                                                              │
│ Return: InferenceResult(query, response, session_id,        │
│                         metadata, latency_ms)               │
└─────────────────────────────────────────────────────────────┘
```

### Streaming Variant

For `inference_pipeline.stream()`, the flow is identical through Step 2, but:
- The complete response from the agentic workflow is chunked into 4-character pieces
- Each piece is yielded as an async iterator
- Messages are persisted after the full response is assembled

### Timeout & Retry

- Sync: `concurrent.futures.ThreadPoolExecutor` with configurable timeout (default 30s)
- Async: `asyncio.wait_for` with same timeout
- LLM calls: exponential backoff (0.5s, 1s, 2s) up to 3 retries

---

## 7. Agentic Workflow (LangGraph)

**File:** `src/pipelines/inference/workflow/agentic.py`

This is the brain of the system. A LangGraph `StateGraph` with 5 nodes and conditional routing.

### Workflow Graph

```
                    ┌─────────────┐
                    │   START     │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Router    │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
     ┌──────────────┐ ┌────────┐ ┌──────────┐
     │ Reformulator │ │  Tool  │ │ Generator │
     └──────┬───────┘ └───┬────┘ │ (direct)  │
            │              │      └─────┬─────┘
            ▼              │            │
     ┌──────────────┐     │            │
     │  Retriever   │     │            │
     └──────┬───────┘     │            │
            │              │            │
            └──────┬───────┘            │
                   │                    │
                   ▼                    │
            ┌──────────────┐            │
            │  Generator   │◄───────────┘
            └──────┬───────┘
                   │
                   ▼
            ┌─────────────┐
            │     END     │
            └─────────────┘
```

### State Shape (`AgentState`)

```python
{
    "messages":           List[BaseMessage],  # Full conversation (managed by LangGraph)
    "context":            str,                # Retrieved product context
    "route":              str,                # "retrieve" | "tool" | "respond"
    "tool_result":        str,                # Output from tool node
    "reformulated_query": str                 # Optimized query for vector search
}
```

### Node Details

#### 7.1 Router Node

Classifies the user query into one of three routes:

```
Input: Latest HumanMessage from state
       │
       ▼
Check tool_keywords (default: ["compare"])
  └─ Match? → route = "tool"
       │
       ▼
Everything else → route = "retrieve"
  (product_keywords check removed — all non-tool queries go to retrieve)
       │
       ▼
Output: {"route": "retrieve" | "tool"}
```

> Note: There is no "respond" route in practice. The router sends virtually everything
> through the retrieval path, which is the correct behavior for a RAG system.

#### 7.2 Reformulator Node

Optimizes the query before it hits the vector database. Two modes:

```
Input: Latest HumanMessage + conversation history
       │
       ▼
Is this a follow-up query?
  ├─ Check for contextual references: "that", "this", "it", "they", "the one"...
  ├─ Check for follow-up phrases: "tell me more", "what about", "how about"...
  └─ Check if recent assistant messages contain product keywords
       │
       ├─── YES (follow-up) ──────────────────────────────────────────┐
       │    Use FOLLOWUP_PROMPT:                                       │
       │    "Rewrite follow-up queries into standalone queries         │
       │     by extracting explicit iPhone model names from history"   │
       │    Example: "tell me more about the cheaper one"              │
       │         → "Tell me more about the iPhone 11"                  │
       │                                                               │
       ├─── NO (standalone) ──────────────────────────────────────────┐
       │    Use STANDALONE_PROMPT:                                     │
       │    "Rewrite the query into an optimized search query          │
       │     for a vector database of iPhone reviews"                  │
       │    Example: "I want something good for photos"                │
       │         → "iPhone camera quality photo performance"           │
       │                                                               │
       ▼                                                               │
Output: {"reformulated_query": "optimized search string"}              │
```

**LLM used:** Same `gpt-4o-mini` client as the rest of inference.

#### 7.3 Retriever Node

Calls the Retrieval Pipeline (Section 8) with the reformulated query.

```
Input: state["reformulated_query"] (falls back to raw user message)
       │
       ▼
retrieval_pipeline.retrieve(query)
       │
       ▼
Output: {"context": formatted_context_string}
```

The context string looks like:
```
Title: Apple iPhone 12, 64GB, Blue - Unlocked (Renewed)
Price: $448.00
Rating: 4.3
Review: Great phone, battery lasts all day...

---

Title: Apple iPhone 11, 128GB, Black - Unlocked (Renewed)
Price: $302.00
Rating: 4.1
Review: Excellent value for the price...
```

#### 7.4 Tool Node (Demo/Extensibility)

Currently a placeholder for future MCP integration. Returns a demo comparison message.

#### 7.5 Generator Node

Produces the final user-facing response.

```
Input: user query + context + tool_result + conversation history
       │
       ▼
Combine context sources:
  ├─ context only → use as-is
  ├─ tool_result only → use as-is
  └─ both → "Retrieved Context:\n{context}\n\nTool Result:\n{tool_result}"
       │
       ▼
ResponseGenerator.generate(query, combined_context, history)
       │
       ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 1. Context quality check (GroundingStrategy)            │
  │    └─ Is context ≥ 50 chars and contains product info?  │
  │    └─ NO → return fallback message immediately          │
  │                                                          │
  │ 2. Build message list:                                  │
  │    [SystemMessage(grounding_prompt + context),           │
  │     ...history (HumanMessage/AIMessage pairs),           │
  │     HumanMessage(current query)]                        │
  │                                                          │
  │ 3. LLM call → OpenAI gpt-4o-mini                       │
  │    temperature: 0.0 (deterministic)                     │
  │    max_tokens: 2048                                     │
  │                                                          │
  │ 4. Response validation (GroundingStrategy)              │
  │    └─ If no context was provided but response makes     │
  │       specific product claims → return fallback         │
  │                                                          │
  │ 5. Return response.content                              │
  └─────────────────────────────────────────────────────────┘
       │
       ▼
Output: {"messages": [AIMessage(content=response)]}
```

### Grounding Strategy (`grounding.py`)

The grounding system prevents the LLM from hallucinating product information.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Grounding Decision Tree                       │
│                                                                  │
│  Context available?                                              │
│  ├─ NO → Use FALLBACK_NO_CONTEXT_PROMPT                         │
│  │       "No relevant products were found... suggest rephrasing" │
│  │                                                               │
│  └─ YES → strict_mode?                                          │
│           ├─ YES → STRICT_GROUNDING_PROMPT                      │
│           │   "ONLY recommend products from CONTEXT"             │
│           │   "NEVER make up names, prices, ratings"             │
│           │   "ALWAYS cite specific details"                     │
│           │                                                      │
│           └─ NO → Softer prompt                                 │
│               "Prioritize context, acknowledge limitations"      │
│                                                                  │
│  Post-generation validation:                                     │
│  └─ If no context was provided but response contains             │
│     "i recommend", "you should buy", "price is", etc.           │
│     → Response rejected, fallback message returned               │
└─────────────────────────────────────────────────────────────────┘
```

**GroundingConfig defaults:**
| Setting | Default | Purpose |
|---------|---------|---------|
| `strict_mode` | `True` | Use strict grounding prompt |
| `require_context` | `True` | Refuse to answer without context |
| `min_context_length` | `50` chars | Minimum context to proceed |
| `enable_validation` | `True` | Validate response after generation |

---

## 8. Retrieval Pipeline — Document Search Engine

**File:** `src/pipelines/retrieval/pipeline.py`

### Complete Retrieval Flow

```
retrieval_pipeline.retrieve(query)
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│                                                                   │
│  ┌─ Cache Check ─────────────────────────────────────────────┐   │
│  │ ResultCache.get(query, filters)                            │   │
│  │ └─ HIT → return cached RetrievalResult (skip everything)  │   │
│  │ └─ MISS → continue                                        │   │
│  └────────────────────────────────────────────────────────────┘   │
│                          │                                        │
│                          ▼                                        │
│  ┌─ Retrieval Loop (max 3 iterations) ───────────────────────┐   │
│  │                                                            │   │
│  │  Step 1: Query Processing                                 │   │
│  │  QueryProcessor.process(query)                            │   │
│  │  ├─ Validate (non-empty, ≤512 chars)                      │   │
│  │  ├─ Normalize (whitespace, Unicode)                       │   │
│  │  ├─ Truncate if needed                                    │   │
│  │  └─ Generate embedding (text-embedding-3-large, dim=3072) │   │
│  │  Output: ProcessedQuery(original, normalized, embedding)  │   │
│  │                          │                                 │   │
│  │                          ▼                                 │   │
│  │  Step 1.5: Metadata Extraction (if no filters provided)   │   │
│  │  MetadataExtractor.extract(query)                         │   │
│  │  ├─ LLM call (gpt-3.5-turbo, 3s timeout)                 │   │
│  │  ├─ Extracts: product_name, min/max_price, min_rating     │   │
│  │  ├─ Example: "iPhone 12 under $500"                       │   │
│  │  │   → {product_name: "iphone 12", max_price: 500}        │   │
│  │  └─ Graceful degradation: returns None on any failure     │   │
│  │  Output: Optional[MetadataFilter]                         │   │
│  │                          │                                 │   │
│  │                          ▼                                 │   │
│  │  Step 2: Vector Search                                    │   │
│  │  VectorSearcher.search(embedding, filters)                │   │
│  │  ├─ Search type: MMR (default) or similarity              │   │
│  │  ├─ top_k: 4, fetch_k: 20, lambda_mult: 0.7              │   │
│  │  ├─ Apply Pinecone metadata filters (price, rating)       │   │
│  │  ├─ Post-search product_name substring filtering          │   │
│  │  └─ Score threshold: 0.6 (filter low-relevance docs)     │   │
│  │  Output: SearchResult(documents, scores)                  │   │
│  │                          │                                 │   │
│  │                          ▼                                 │   │
│  │  Step 3: Contextual Compression                           │   │
│  │  ContextCompressor.compress(query, documents)             │   │
│  │  ├─ For each document: LLM evaluates relevance (Yes/No)  │   │
│  │  ├─ Uses gpt-3.5-turbo for cost efficiency                │   │
│  │  ├─ Irrelevant documents filtered out                     │   │
│  │  └─ Failed evaluations → document kept (safe default)     │   │
│  │  Output: CompressionResult(filtered_docs, scores, ratio)  │   │
│  │                          │                                 │   │
│  │                          ▼                                 │   │
│  │  Step 4: Rewrite Decision                                 │   │
│  │  avg_relevance < 0.5 AND attempts < 2?                    │   │
│  │  ├─ YES → QueryRewriter.rewrite(query, context)           │   │
│  │  │         ├─ LLM rewrites for iPhone product DB          │   │
│  │  │         ├─ Example: "good camera phone"                │   │
│  │  │         │   → "iPhone camera quality megapixel reviews" │   │
│  │  │         └─ Loop back to Step 1 with rewritten query    │   │
│  │  │                                                         │   │
│  │  └─ NO → Proceed to formatting                            │   │
│  │                                                            │   │
│  └────────────────────────────────────────────────────────────┘   │
│                          │                                        │
│                          ▼                                        │
│  Step 5: Document Formatting                                     │
│  DocumentFormatter.format(documents, scores)                     │
│  ├─ Template per document:                                       │
│  │   "Title: {product_name}\nPrice: {price}\n                    │
│  │    Rating: {rating}\nReview: {review_text}"                   │
│  ├─ Delimiter: "\n\n---\n\n"                                     │
│  ├─ Max context length: 4000 chars (truncate if exceeded)        │
│  └─ Output: FormattedContext(text, document_count, truncated)    │
│                          │                                        │
│                          ▼                                        │
│  Cache result: ResultCache.set(query, result)                    │
│                          │                                        │
│                          ▼                                        │
│  Return: RetrievalResult(query, documents, formatted_context,    │
│                          scores, metadata, latency_ms)           │
└──────────────────────────────────────────────────────────────────┘
```

### Retrieval Pipeline Components Summary

| Component | LLM Used | Purpose | Failure Mode |
|-----------|----------|---------|-------------|
| QueryProcessor | OpenAI Embeddings | Embed query to vector | Fatal — raises error |
| MetadataExtractor | gpt-3.5-turbo | Extract filters from NL | Graceful — continues without filters |
| VectorSearcher | — | Pinecone similarity search | Fatal — raises error |
| ContextCompressor | gpt-3.5-turbo | Filter irrelevant docs | Graceful — returns all docs |
| QueryRewriter | gpt-3.5-turbo | Improve low-relevance queries | Graceful — uses original query |
| DocumentFormatter | — | Format docs to text | Fatal — raises error |
| ResultCache | — | In-memory TTL cache | Graceful — cache miss = fresh search |

---

## 9. Conversation Store — Persistence Layer

**File:** `src/pipelines/inference/conversation/store.py`

### Storage Model

```
data/sessions/
├── sess_20260221_185625_7ecbdb57.json
├── sess_20260221_190239_bc4b62b4.json
└── ...
```

Each JSON file:
```json
{
  "session_id": "sess_20260221_185625_7ecbdb57",
  "created_at": "2026-02-21T18:56:25.000000Z",
  "updated_at": "2026-02-21T19:03:12.000000Z",
  "messages": [
    {"role": "user", "content": "Show me iPhones under $400", "timestamp": "..."},
    {"role": "assistant", "content": "Here are some options...", "timestamp": "..."},
    {"role": "user", "content": "Tell me more about the first one", "timestamp": "..."},
    {"role": "assistant", "content": "The iPhone 11...", "timestamp": "..."}
  ]
}
```

### Key Behaviors

| Behavior | Detail |
|----------|--------|
| Concurrency | `fcntl.LOCK_SH` for reads, `fcntl.LOCK_EX` for writes |
| History trimming | `get_langchain_messages()` returns last `max_history_length` (10) messages |
| LangChain conversion | `"user"` → `HumanMessage`, `"assistant"` → `AIMessage` |
| Malformed JSON | Returns `None` / empty list (logged as warning, not fatal) |
| Session ID format | `sess_{YYYYMMDD}_{HHMMSS}_{uuid4_hex[:8]}` |
| Role validation | Only `"user"` and `"assistant"` accepted; others raise `ValueError` |

---

## 10. Complete Request Lifecycle — Worked Example

**Scenario:** User asks "Show me iPhones with good cameras under $500" in a new session.

```
 ① User types message in Streamlit chat input
    │
 ② Frontend calls api_client.stream_message("Show me iPhones...", "sess_abc123")
    │  POST /api/chat/stream  {query: "Show me iPhones...", session_id: "sess_abc123"}
    │
 ③ FastAPI validates request (Pydantic), checks session exists, checks pipeline ready
    │
 ④ inference_pipeline.stream(query, session_id)
    │
 ⑤ ConversationStore loads history from data/sessions/sess_abc123.json
    │  → [] (empty, new session)
    │
 ⑥ AgenticWorkflow.arun("Show me iPhones with good cameras under $500", [])
    │
    ├─ Router: no tool keywords → route = "retrieve"
    │
    ├─ Reformulator: no history → standalone mode
    │  LLM optimizes: "iPhone camera quality under $500 reviews photos"
    │
    ├─ Retriever: calls retrieval_pipeline.retrieve(optimized_query)
    │  │
    │  ├─ Cache miss
    │  ├─ QueryProcessor: embed query → 3072-dim vector
    │  ├─ MetadataExtractor: extracts {max_price: 500, product_name: null}
    │  ├─ VectorSearcher: MMR search on Pinecone with price filter
    │  │   → 4 documents with scores [0.82, 0.78, 0.75, 0.71]
    │  ├─ ContextCompressor: all 4 deemed relevant
    │  ├─ avg_relevance > 0.5 → no rewrite needed
    │  ├─ DocumentFormatter: formats 4 docs into context string
    │  └─ Cache result for future queries
    │
    └─ Generator:
       ├─ Context quality check: ✓ (>50 chars, has product info)
       ├─ Build messages:
       │   [SystemMessage(STRICT_GROUNDING_PROMPT + context),
       │    HumanMessage("Show me iPhones with good cameras under $500")]
       ├─ LLM call: gpt-4o-mini, temp=0.0
       ├─ Response validation: ✓ (grounded in context)
       └─ Returns: "Based on the reviews in our database, here are some
                    great camera options under $500:
                    1. iPhone 12 ($448, 4.3★) - excellent camera...
                    2. iPhone 11 ($302, 4.1★) - great value..."
    │
 ⑦ Response chunked into 4-char pieces, streamed as SSE events
    │  data: {"chunk": "Base", "done": false}
    │  data: {"chunk": "d on", "done": false}
    │  ...
    │  data: {"chunk": "", "done": true, "total_length": 342}
    │
 ⑧ ConversationStore persists both messages to disk:
    │  add_message(sess_abc123, "user", "Show me iPhones...")
    │  add_message(sess_abc123, "assistant", "Based on the reviews...")
    │
 ⑨ Frontend displays response with typing animation (cursor "▌")
    │  Appends to st.session_state.messages
    │
 ⑩ User sees formatted response with product recommendations
```

### Follow-Up Example

User then asks: "Tell me more about the first one"

```
 ⑥ AgenticWorkflow.arun("Tell me more about the first one", [prev_messages])
    │
    ├─ Router: → route = "retrieve"
    │
    ├─ Reformulator: detects follow-up
    │  ├─ "the first one" matches contextual reference patterns
    │  ├─ Previous assistant message contains product keywords
    │  └─ LLM resolves: "Tell me more about the iPhone 12 64GB Blue Renewed"
    │
    ├─ Retriever: searches with resolved query
    │  └─ MetadataExtractor: {product_name: "iphone 12"}
    │  └─ Pinecone returns iPhone 12-specific reviews
    │
    └─ Generator: produces detailed iPhone 12 response grounded in context
```

---

## 11. Configuration Reference

### config/inference.yaml

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `llm` | `provider` | `openai` | LLM provider |
| `llm` | `model_name` | `gpt-4o-mini` | Model for generation |
| `llm` | `temperature` | `0.0` | Deterministic output |
| `llm` | `max_tokens` | `2048` | Max response length |
| `conversation` | `max_history_length` | `10` | Messages kept for LLM context |
| `conversation` | `storage_dir` | `data/sessions` | Session file directory |
| `generator` | `strict_grounding` | `true` | Prevent hallucination |
| `generator` | `require_context` | `true` | Must have context for product answers |
| `generator` | `min_context_length` | `50` | Min chars of context required |
| `workflow` | `product_keywords` | `[price, review, product, ...]` | Trigger retrieval |
| `workflow` | `tool_keywords` | `[compare]` | Trigger tool node |
| — | `enable_streaming` | `true` | SSE streaming support |
| — | `max_retries` | `3` | LLM retry attempts |
| — | `timeout_seconds` | `30` | Operation timeout |

### config/retrieval.yaml

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `search` | `top_k` | `4` | Documents to return |
| `search` | `fetch_k` | `20` | Candidates for MMR |
| `search` | `lambda_mult` | `0.7` | MMR diversity (0=diverse, 1=relevant) |
| `search` | `score_threshold` | `0.6` | Min similarity score |
| `search` | `search_type` | `mmr` | Algorithm: `mmr` or `similarity` |
| `query` | `max_query_length` | `512` | Max query chars |
| `embedding` | `model` | `text-embedding-3-large` | Embedding model |
| `embedding` | `dimension` | `3072` | Vector dimension |
| `compression` | `enabled` | `true` | LLM-based doc filtering |
| `rewriting` | `max_attempts` | `2` | Max rewrite iterations |
| `rewriting` | `threshold` | `0.5` | Rewrite trigger score |
| `cache` | `enabled` | `true` | Result caching |
| `cache` | `ttl_seconds` | `300` | Cache expiry |
| `metadata_extraction` | `enabled` | `true` | Auto-extract filters |
| `metadata_extraction` | `model` | `gpt-3.5-turbo` | Extraction LLM |
| `metadata_extraction` | `timeout` | `3` | Extraction timeout (seconds) |

### Environment Variables (`.env`)

| Variable | Required | Used By |
|----------|----------|---------|
| `OPENAI_API_KEY` | Yes | Embeddings, LLM, compression, rewriting, extraction |
| `PINECONE_API_KEY` | Yes | Vector search, ingestion storage |
| `PINECONE_INDEX_NAME` | No (default: `ecommerce-products`) | Vector DB index |
| `PINECONE_NAMESPACE` | No (default: `phone-reviews`) | Vector DB namespace |
| `PINECONE_ENVIRONMENT` | No (default: `us-east-1-aws`) | Pinecone region |

---

## 12. Error Handling & Resilience

### Exception Hierarchy

```
BaseException
└── Exception
    ├── InferenceError          — General inference failures
    │   ├── ConfigurationError  — Missing config, bad API keys
    │   ├── LLMError            — OpenAI API failures (retried)
    │   ├── SessionError        — Session not found, write failures
    │   ├── StreamingError      — Mid-stream failures (partial response saved)
    │   └── TimeoutError        — Operation exceeded timeout
    │
    ├── RetrievalError          — General retrieval failures
    │   ├── ConfigurationError  — Missing config
    │   ├── ConnectionError     — Pinecone connection failures (retried)
    │   ├── QueryValidationError— Empty/invalid queries
    │   ├── EmbeddingError      — Embedding generation failures
    │   └── SearchError         — Vector search failures
    │
    └── IngestionError          — General ingestion failures
        ├── ValidationError     — Missing columns, bad data
        ├── DataQualityError    — Failure rate > threshold
        ├── ConfigurationError  — Missing config
        └── ConnectionError     — Pinecone connection failures
```

### Graceful Degradation Pattern

Every non-critical component follows the same pattern:

```
try:
    result = component.process(input)
except Exception:
    log.warning("Component failed, using fallback")
    result = safe_default  # original query, all documents, None filters, etc.
```

This ensures the system always returns a response, even if some enrichment steps fail.

---

## 13. LLM Usage Map

Summary of every LLM call in the system and its purpose:

| Call Site | Model | Temperature | Purpose | Cost Tier |
|-----------|-------|-------------|---------|-----------|
| ResponseGenerator | gpt-4o-mini | 0.0 | Final user-facing response | Medium |
| QueryReformulator | gpt-4o-mini | 0.0 | Optimize/resolve queries | Medium |
| Router (implicit) | — | — | Keyword-based, no LLM call | Free |
| MetadataExtractor | gpt-3.5-turbo | 0.0 | Extract price/rating/name filters | Low |
| ContextCompressor | gpt-3.5-turbo | 0.0 | Yes/No relevance per document | Low |
| QueryRewriter | gpt-3.5-turbo | 0.3 | Rewrite low-relevance queries | Low |
| EmbeddingGenerator | text-embedding-3-large | — | Embed documents (ingestion) | Low |
| QueryProcessor | text-embedding-3-large | — | Embed queries (retrieval) | Low |

**Per-request LLM calls (typical):** 1 embedding + 1 metadata extraction + 1 reformulation + 4 compression evaluations + 1 generation = ~8 API calls

**Per-request LLM calls (with rewrite):** Add 1 rewrite + 1 embedding + 4 more compressions = ~14 API calls

---

## 14. Data Shapes Quick Reference

### Pinecone Vector Record
```
{
  id: "sha256_hash_of_content",
  values: [0.012, -0.034, ...],  // 3072 floats
  metadata: {
    product_name: "Apple iPhone 12, 64GB, Blue - Unlocked (Renewed)",
    description: "...",
    price: "448.00",
    rating: "4.3",
    review_title: "Great phone",
    review_text: "Battery lasts all day..."
  }
}
```

### LangChain Document (in-memory)
```python
Document(
    page_content="Product: Apple iPhone 12... | Price: $448 | Rating: 4.3/5 | Review: ...",
    metadata={"product_name": "...", "price": "448.00", "rating": "4.3", ...}
)
```

### Session File (on disk)
```json
{
  "session_id": "sess_20260221_185625_7ecbdb57",
  "created_at": "2026-02-21T18:56:25Z",
  "updated_at": "2026-02-21T19:03:12Z",
  "messages": [
    {"role": "user", "content": "...", "timestamp": "2026-02-21T18:56:30Z"},
    {"role": "assistant", "content": "...", "timestamp": "2026-02-21T18:56:35Z"}
  ]
}
```

### SSE Stream Event
```
data: {"chunk": "Here are", "done": false}\n\n
data: {"chunk": " some ", "done": false}\n\n
data: {"chunk": "", "done": true, "session_id": "sess_abc", "total_length": 342}\n\n
```
