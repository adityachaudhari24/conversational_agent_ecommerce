# Storage Comparison: SessionStore vs ConversationManager

## Current State Analysis

Based on code inspection and actual session file data, here's what's stored in each system:

---

## 1. SessionStore (File-Based) 📁

### Location
`data/sessions/*.json`

### What's Stored

```json
{
  "session_id": "sess_20251230_124854_aa19f7c3",
  "created_at": "2025-12-30T12:48:54.430013Z",
  "updated_at": "2025-12-30T12:49:18.416302Z",
  "messages": [
    {
      "role": "user",
      "content": "iphone under 200",
      "timestamp": "2025-12-30T12:49:03.254143Z"
    },
    {
      "role": "assistant",
      "content": "I currently don't have specific options...",
      "timestamp": "2025-12-30T12:49:18.416292Z"
    }
  ]
}
```

### Data Structure

```python
@dataclass
class SessionMessage:
    role: str              # "user" or "assistant"
    content: str           # Full message text
    timestamp: str         # ISO format: "2025-12-30T12:49:03.254143Z"

@dataclass
class Session:
    session_id: str        # "sess_20251230_124854_aa19f7c3"
    created_at: str        # ISO format
    updated_at: str        # ISO format
    messages: List[SessionMessage]
```

### When Data is Written

**Location**: `src/api/routes/chat.py`

```python
# In /chat endpoint (non-streaming)
@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, ...):
    # Generate response
    result = await inference_pipeline.agenerate(
        query=request.query,
        session_id=request.session_id
    )
    
    # ✅ SAVE TO SESSIONSTORE
    session_store.add_message(request.session_id, "user", request.query)
    session_store.add_message(request.session_id, "assistant", result.response)
```

```python
# In /chat/stream endpoint (streaming)
@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, ...):
    async def generate_stream():
        complete_response = ""
        
        # ✅ SAVE USER MESSAGE FIRST
        session_store.add_message(request.session_id, "user", request.query)
        
        # Stream response chunks
        async for chunk in inference_pipeline.stream(...):
            complete_response += chunk
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        
        # ✅ SAVE COMPLETE ASSISTANT RESPONSE
        session_store.add_message(request.session_id, "assistant", complete_response)
```

### Purpose
- ✅ Persistent storage (survives server restart)
- ✅ UI display (session history panel)
- ✅ Session management (list, retrieve, delete)
- ✅ Audit trail with timestamps

### Characteristics
- **Persistence**: Permanent (until manually deleted)
- **Scope**: All messages in session (no limit)
- **Format**: JSON files with file locking
- **Access**: Via SessionStore class methods

---

## 2. ConversationManager (In-Memory) 💾

### Location
In-memory dictionary: `self._sessions: Dict[str, Session]`

### What's Stored

```python
# Internal structure (not persisted)
{
    "sess_20251230_124854_aa19f7c3": Session(
        session_id="sess_20251230_124854_aa19f7c3",
        messages=[
            Message(
                role="user",
                content="iphone under 200",
                timestamp=datetime(2025, 12, 30, 12, 49, 3, 254143)
            ),
            Message(
                role="assistant",
                content="I currently don't have specific options...",
                timestamp=datetime(2025, 12, 30, 12, 49, 18, 416292)
            )
        ],
        created_at=datetime(2025, 12, 30, 12, 48, 54, 430013)
    )
}
```

### Data Structure

```python
@dataclass
class Message:
    role: str              # "user" or "assistant"
    content: str           # Full message text
    timestamp: datetime    # Python datetime object

@dataclass
class Session:
    session_id: str
    messages: List[Message]
    created_at: datetime
```

### When Data is Written

**Location**: `src/pipelines/inference/pipeline.py`

```python
def _execute_generate_workflow(self, query: str, session_id: str) -> InferenceResult:
    # Step 1: Get conversation history
    history = self.conversation_manager.get_langchain_messages(session_id)
    
    # Step 2: Execute agentic workflow
    response = self.agentic_workflow.run(query, history)
    
    # Step 3: ✅ UPDATE CONVERSATION MANAGER
    self.conversation_manager.add_message(session_id, "user", query)
    self.conversation_manager.add_message(session_id, "assistant", response)
    
    return result
```

```python
async def _execute_streaming_workflow(self, query: str, session_id: str):
    complete_response = ""
    
    # Get history from ConversationManager
    history = self.conversation_manager.get_langchain_messages(session_id)
    
    # Stream response
    async for chunk in self.response_generator.astream(...):
        complete_response += chunk
        yield chunk
    
    # ✅ UPDATE CONVERSATION MANAGER AFTER STREAMING
    self.conversation_manager.add_message(session_id, "user", query)
    self.conversation_manager.add_message(session_id, "assistant", complete_response)
```

### Purpose
- ✅ Provide conversation history to LLM
- ✅ Convert to LangChain message format
- ✅ Manage history length limits
- ✅ Fast in-memory access during inference

### Characteristics
- **Persistence**: ❌ Lost on server restart
- **Scope**: Limited by `max_history_length` config (default: 10 messages)
- **Format**: Python objects in memory
- **Access**: Via ConversationManager methods

---

## Side-by-Side Comparison

| Aspect | SessionStore | ConversationManager |
|--------|-------------|---------------------|
| **Storage Type** | File-based (JSON) | In-memory (dict) |
| **Persistence** | ✅ Permanent | ❌ Volatile |
| **Location** | `data/sessions/*.json` | RAM |
| **Data Format** | JSON with ISO timestamps | Python objects with datetime |
| **Message Limit** | ❌ None (stores all) | ✅ Yes (`max_history_length`) |
| **Survives Restart** | ✅ Yes | ❌ No |
| **Used By** | API routes, Frontend | Inference pipeline |
| **Purpose** | Persistence & UI | LLM context |
| **Write Timing** | After response complete | After response complete |
| **Read Access** | Session retrieval, UI | History for LLM |
| **Timestamp Format** | ISO string | Python datetime |
| **File Locking** | ✅ Yes (fcntl) | N/A |

---

## Data Flow Diagram

```
User Query
    ↓
┌─────────────────────────────────────────────────────────────┐
│ API Endpoint (/chat/stream)                                 │
│                                                              │
│ 1. Verify session exists in SessionStore                    │
│ 2. Call inference_pipeline.stream(query, session_id)        │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ InferencePipeline._execute_streaming_workflow               │
│                                                              │
│ 1. ✅ READ from ConversationManager                         │
│    history = conversation_manager.get_langchain_messages()  │
│                                                              │
│ 2. Pass history to AgenticWorkflow                          │
│    response = agentic_workflow.arun(query, history)         │
│                                                              │
│ 3. Stream response chunks                                   │
│                                                              │
│ 4. ✅ WRITE to ConversationManager                          │
│    conversation_manager.add_message("user", query)          │
│    conversation_manager.add_message("assistant", response)  │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ API Endpoint (continued)                                    │
│                                                              │
│ 1. ✅ WRITE to SessionStore                                 │
│    session_store.add_message("user", query)                 │
│    session_store.add_message("assistant", response)         │
│                                                              │
│ 2. Return response to frontend                              │
└─────────────────────────────────────────────────────────────┘
    ↓
Frontend Display
```

---

## The Critical Problem

### Scenario: Server Restart

**Before Restart:**
```
SessionStore (File):
  sess_123: [msg1, msg2, msg3, msg4]  ← Persisted to disk

ConversationManager (Memory):
  sess_123: [msg1, msg2, msg3, msg4]  ← In RAM
```

**After Restart:**
```
SessionStore (File):
  sess_123: [msg1, msg2, msg3, msg4]  ← Still there! ✅

ConversationManager (Memory):
  sess_123: []                         ← EMPTY! ❌
```

**Next User Message:**
```
User: "What about the second one?"  (referring to msg3)

ConversationManager.get_langchain_messages(sess_123)
  → Returns: []  ← NO HISTORY!

LLM receives:
  [SystemMessage] "You are a helpful assistant..."
  [HumanMessage] "What about the second one?"
  
LLM Response: "I'm not sure what you're referring to. Could you provide more context?"
```

**But the UI shows:**
```
Frontend (reading from SessionStore):
  msg1: "Show me phones"
  msg2: "Here are some phones..."
  msg3: "What's the price of the first one?"
  msg4: "The first one costs $299"
  msg5: "What about the second one?"  ← User sees full history
  msg6: "I'm not sure what you're referring to..."  ← Confusing!
```

---

## What Gets Stored: Detailed Breakdown

### SessionStore Stores:

1. **Session Metadata**
   - `session_id`: Unique identifier
   - `created_at`: When session was created
   - `updated_at`: Last modification time

2. **All Messages** (no limit)
   - `role`: "user" or "assistant"
   - `content`: Full message text
   - `timestamp`: When message was sent

3. **Additional Properties**
   - `preview`: First 100 chars of first user message
   - `message_count`: Total number of messages

### ConversationManager Stores:

1. **Session Metadata**
   - `session_id`: Unique identifier
   - `created_at`: When session was created (in memory)

2. **Limited Messages** (max_history_length)
   - `role`: "user" or "assistant"
   - `content`: Full message text
   - `timestamp`: Python datetime object

3. **Automatic Trimming**
   ```python
   def _trim_history(self, session: Session) -> None:
       max_length = self.config.max_history_length
       if len(session.messages) > max_length:
           # Keep only the most recent messages
           session.messages = session.messages[-max_length:]
   ```

---

## Example: What's Actually Stored

### After 15 messages in a conversation:

**SessionStore** (`data/sessions/sess_123.json`):
```json
{
  "session_id": "sess_123",
  "messages": [
    {"role": "user", "content": "msg1"},
    {"role": "assistant", "content": "response1"},
    {"role": "user", "content": "msg2"},
    {"role": "assistant", "content": "response2"},
    {"role": "user", "content": "msg3"},
    {"role": "assistant", "content": "response3"},
    {"role": "user", "content": "msg4"},
    {"role": "assistant", "content": "response4"},
    {"role": "user", "content": "msg5"},
    {"role": "assistant", "content": "response5"},
    {"role": "user", "content": "msg6"},
    {"role": "assistant", "content": "response6"},
    {"role": "user", "content": "msg7"},
    {"role": "assistant", "content": "response7"},
    {"role": "user", "content": "msg8"}
  ]
}
```
**Total**: 15 messages (all stored)

**ConversationManager** (in memory, with `max_history_length: 10`):
```python
{
    "sess_123": Session(
        messages=[
            Message(role="user", content="msg4"),
            Message(role="assistant", content="response4"),
            Message(role="user", content="msg5"),
            Message(role="assistant", content="response5"),
            Message(role="user", content="msg6"),
            Message(role="assistant", content="response6"),
            Message(role="user", content="msg7"),
            Message(role="assistant", content="response7"),
            Message(role="user", content="msg8"),
            # Only last 10 messages kept
        ]
    )
}
```
**Total**: 9 messages (trimmed to last 10, but msg8 doesn't have response yet)

---

## Summary

### SessionStore
- **Stores**: Everything, forever (until deleted)
- **Purpose**: Persistence, UI, audit trail
- **Format**: JSON files
- **Survives**: Server restarts ✅

### ConversationManager
- **Stores**: Recent messages only (configurable limit)
- **Purpose**: LLM context during inference
- **Format**: Python objects in RAM
- **Survives**: Server restarts ❌

### The Gap
These two systems are **completely independent** and **never synchronize**, causing:
1. Loss of context after server restart
2. UI showing messages that LLM can't see
3. Inconsistent user experience
4. Potential confusion when history limits differ
