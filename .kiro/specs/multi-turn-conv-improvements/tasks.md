# Implementation Tasks: Multi-Turn Conversation Improvements

## Task 1: Create ConversationStore (unified persistent storage)

- [x] Create `src/pipelines/inference/conversation/store.py` with `ConversationStore` class that combines file-based persistence (from `SessionStore`) with LangChain message conversion (from `ConversationManager`). Implement `__init__`, `create_session`, `get_session`, `list_sessions`, `delete_session` methods using JSON files in `data/sessions/` with `fcntl` file locking (shared locks for reads, exclusive locks for writes). Reuse the existing `SessionMessage` and `Session` dataclasses from `src/api/services/session_store.py`. (Req 1.1, 1.4, 1.5, 1.6)
- [x] Implement `add_message(session_id, role, content)` in `ConversationStore` that validates role is "user" or "assistant", appends the message, updates `updated_at`, and persists to disk immediately. Raise `ValueError` for invalid roles and `SessionError` if session not found or write fails. (Req 1.3, 7.2, 7.4)
- [x] Implement `get_langchain_messages(session_id, limit)` in `ConversationStore` that loads session from disk, converts messages to LangChain `HumanMessage`/`AIMessage` objects in chronological order, and returns only the most recent `limit` messages (defaulting to `max_history_length`). (Req 6.1, 6.2, 6.3, 2.3)
- [x] Implement `get_history(session_id, limit)` in `ConversationStore` that returns raw `SessionMessage` objects from disk with optional limit, and handles malformed JSON by returning empty list and logging a warning. (Req 7.1, 7.3)
- [x] Write unit tests for `ConversationStore` in `tests/unit/test_conversation_store.py` covering: session CRUD, message persistence, LangChain conversion, history trimming, file locking, malformed JSON handling, and invalid role rejection. (Req 1, 2, 6, 7)
- [x] Write property-based tests for `ConversationStore` in `tests/unit/test_conversation_store.py` using Hypothesis: (P1) For any sequence of added messages, `get_langchain_messages` returns them in the same chronological order and with correct role mapping. (P2) For any session with N messages where N > max_history_length, `get_langchain_messages` returns exactly max_history_length messages, all from the tail of the sequence. (Req 6.1, 6.2, 6.3, 2.3, 2.4)

## Task 2: Update ConversationConfig to include storage_dir

- [x] Add `storage_dir: str = "data/sessions"` field to `ConversationConfig` in `src/pipelines/inference/config.py`. Update `create_settings_from_yaml` to pass `storage_dir` from YAML config if present (currently `ConversationConfig` only has `max_history_length`). (Req 1.1)

## Task 3: Update InferencePipeline to use ConversationStore

- [x] Modify `InferencePipeline.__init__` in `src/pipelines/inference/pipeline.py` to accept a `ConversationStore` parameter instead of creating `ConversationManager` internally. Remove the `self.conversation_manager` attribute and replace with `self.conversation_store`. (Req 3.2, 3.4)
- [x] Update `InferencePipeline.initialize()` to no longer create a `ConversationManager`. The `ConversationStore` is injected and ready to use. Remove the `self.conversation_manager = ConversationManager(self.config.conversation_config)` line. (Req 3.4)
- [x] Update `_execute_generate_workflow` and `_aexecute_generate_workflow` to use `self.conversation_store.get_langchain_messages(session_id)` for loading history and `self.conversation_store.add_message(...)` for persisting messages after response generation. (Req 2.1, 3.2, 8.1)
- [x] Update `_execute_streaming_workflow` and `stream` to use `self.conversation_store` for history loading and message persistence (replace all `self.conversation_manager` references). (Req 2.1, 3.2, 8.1)
- [x] Update `get_session_history`, `clear_session`, and `get_pipeline_stats` to use `self.conversation_store` instead of `self.conversation_manager`. (Req 3.2)
- [x] Update `InferencePipeline.from_config_file` to create a `ConversationStore` instance using `ConversationConfig.storage_dir` and `ConversationConfig.max_history_length`, and pass it to the constructor. (Req 1.1, 2.2)

## Task 4: Update API layer to use ConversationStore

- [x] Update `src/api/dependencies.py`: replace `get_session_store` with `get_conversation_store` and `set_conversation_store` functions that manage a global `ConversationStore` instance. Keep `get_inference_pipeline` and `set_inference_pipeline` unchanged. (Req 3.1)
- [x] Update `src/api/main.py` lifespan: create a `ConversationStore` instance during startup, call `set_conversation_store`, and pass it to `InferencePipeline` constructor. (Req 1.6, 2.2)
- [x] Update `src/api/routes/chat.py`: replace `session_store` dependency with `conversation_store` from `get_conversation_store`. Remove the separate `session_store.add_message()` calls in both `/chat` and `/chat/stream` endpoints since the `InferencePipeline` now handles message persistence via `ConversationStore`. (Req 3.1, 3.3)
- [x] Update `src/api/routes/sessions.py` to use `get_conversation_store` instead of `get_session_store` for session listing, creation, and deletion endpoints. Update the `SessionStore` type hints to `ConversationStore`. (Req 3.1)

## Task 5: Remove old ConversationManager and SessionStore

- [x] Remove `src/pipelines/inference/conversation/manager.py` (the old in-memory `ConversationManager`). Update `src/pipelines/inference/conversation/__init__.py` to export `ConversationStore` from `store.py` instead of `ConversationManager` from `manager.py`. (Req 3.4)
- [x] Remove `src/api/services/session_store.py` (the old file-based `SessionStore`). Update `src/api/services/__init__.py` to no longer export `SessionStore`, `Session`, `SessionMessage`. (Req 3.4)
- [x] Search the codebase for any remaining imports of `ConversationManager` or `SessionStore` and update them to use `ConversationStore`. Fix any broken imports in tests and other modules. (Req 3.4)

## Task 6: Implement history-aware follow-up detection in Router

- [x] Add `_is_follow_up_query(self, query, history)` method to `AgenticWorkflow` in `src/pipelines/inference/workflow/agentic.py`. Implement follow-up detection using `CONTEXTUAL_REFERENCES` and `FOLLOW_UP_PHRASES` pattern lists as defined in the design. The method checks if the query contains contextual references AND recent assistant messages contain product-related content. (Req 4.1, 4.4)
- [x] Update `_router_node` in `AgenticWorkflow` to call `_is_follow_up_query` when no tool or product keywords are found. If a follow-up is detected, set `route="retrieve"` instead of `route="respond"`. (Req 4.1, 4.4)
- [x] Add `reformulated_query: str` field to `AgentState` TypedDict (currently has `messages`, `context`, `route`, `tool_result`). (Req 5.2)
- [x] Write unit tests for follow-up detection in `tests/unit/test_follow_up_routing.py` covering: follow-up with pronouns after product response, follow-up phrases after product response, non-follow-up general queries, and queries with product keywords (should route to retrieve regardless of follow-up detection). (Req 4.1, 4.4)

## Task 7: Implement QueryReformulator

- [x] Create `src/pipelines/inference/workflow/reformulator.py` with `QueryReformulator` class. Implement `reformulate(query, history)` that uses the LLM to rewrite a follow-up query into a standalone query with explicit product/topic references extracted from conversation history. (Req 4.2, 5.1, 5.2)
- [x] Implement `areformulate(query, history)` as the async version of `reformulate`. (Req 4.2, 5.1, 5.2)
- [x] Integrate `QueryReformulator` into `AgenticWorkflow`: initialize it in `__init__`, and add a `_reformulator_node` that runs when the router detects a follow-up. The reformulated query should be stored in `AgentState.reformulated_query` and used by the retriever node. (Req 4.2, 4.3)
- [x] Update `_retriever_node` to use `state["reformulated_query"]` when available instead of the raw user message for vector DB search. (Req 5.3)
- [x] Update `_build_workflow` to add the reformulator node and conditional edges: router → reformulator (when follow-up) → retriever → generator. Update `_route_decision` to return `"reformulator"` when route is `"retrieve_followup"`. (Req 4.2, 4.3)
- [x] Write unit tests for `QueryReformulator` in `tests/unit/test_query_reformulator.py` covering: reformulation with mocked LLM, passthrough when no history, and error handling when LLM fails. (Req 4.2, 5.1, 5.2)

## Task 8: Update existing tests

- [x] Update `tests/unit/test_inference_basic.py` to work with `ConversationStore` instead of `ConversationManager`. The file currently imports `ConversationManager`, `Message`, `Session` from `src.pipelines.inference.conversation.manager` and tests `TestConversationManager`, `TestMessage`, `TestSession` classes. Update all imports and test classes to use `ConversationStore` with a temp directory. Ensure pipeline initialization tests pass with the new constructor signature. (Req 1, 2, 3)
- [x] Run the full test suite with `uv run pytest tests/ -v` and fix any remaining import errors or test failures caused by the refactoring. Note: `tests/unit/test_pipeline.py` tests the ingestion pipeline (not inference) and should not need changes. (Req 1-7)

## Task 9: Write property-based tests for follow-up detection and routing

- [ ] Write property-based tests in `tests/unit/test_follow_up_routing.py` using Hypothesis: (P3) For any query containing a product keyword from `WorkflowConfig.product_keywords`, the router always routes to "retrieve" regardless of conversation history. (P4) For any query containing only contextual references (no product keywords) with empty history, the router routes to "respond" (not "retrieve"). (Req 4.1, 4.4)
