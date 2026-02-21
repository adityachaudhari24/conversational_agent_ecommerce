# Requirements Document

## Introduction

The e-commerce conversational agent currently suffers from a critical dual-storage desynchronization bug. Conversation history is maintained in two independent stores: an in-memory `ConversationManager` (used by the inference pipeline for LLM context) and a file-based `SessionStore` (used for UI display and persistence). On server restart, the in-memory store loses all data while the file-based store retains it, causing multi-turn conversations to break. This feature unifies conversation storage into a single source of truth, ensures conversation continuity across restarts, and improves follow-up question handling so the agentic workflow can route context-dependent queries correctly.

## Glossary

- **Conversation_Store**: The single, persistent storage component that replaces both the in-memory ConversationManager and the file-based SessionStore as the authoritative source of conversation history
- **Inference_Pipeline**: The orchestrator (`InferencePipeline`) that coordinates LLM client, conversation management, response generation, and agentic workflow execution
- **Agentic_Workflow**: The LangGraph-based workflow (`AgenticWorkflow`) that routes queries through Router, Retriever, Tool, and Generator nodes
- **Router**: The node within the Agentic_Workflow that decides whether a query should go to retrieval, tool execution, or direct generation
- **Session**: A conversation session identified by a unique session_id, containing an ordered list of messages
- **Message**: A single user or assistant turn within a Session, containing role, content, and timestamp
- **Follow_Up_Query**: A user query that references or depends on information from earlier turns in the same Session (e.g., "tell me more about that phone")
- **Vector_DB**: The vector database (ChromaDB/Pinecone) used for semantic search over product embeddings
- **LLM_Client**: The client that communicates with the OpenAI API for text generation
- **Response_Generator**: The component (`ResponseGenerator`) that constructs LLM prompts from system prompt, conversation history, retrieved context, and the current query, then calls the LLM_Client
- **AgentState**: The typed state dictionary that flows through the Agentic_Workflow graph, carrying messages, context, routing decisions, and tool results
- **Chat_Route**: The FastAPI endpoint handler (`/chat` and `/chat/stream`) that coordinates between the Inference_Pipeline and persistence

## Requirements

### Requirement 1: Unified Conversation Storage

**User Story:** As a developer, I want a single source of truth for conversation history, so that the LLM context and UI display always show the same data.

#### Acceptance Criteria

1. THE Conversation_Store SHALL persist all Session data to disk using JSON files in the `data/sessions/` directory
2. WHEN the Inference_Pipeline requests conversation history for a session_id, THE Conversation_Store SHALL return the same messages that the UI displays for that session_id
3. WHEN a Message is added to a Session, THE Conversation_Store SHALL persist the Message to disk before returning control to the caller
4. THE Conversation_Store SHALL support concurrent read access using file-level shared locks
5. THE Conversation_Store SHALL support exclusive write access using file-level exclusive locks
6. WHEN the Conversation_Store is initialized, THE Conversation_Store SHALL be capable of loading existing Session data from disk without data loss

### Requirement 2: Conversation Continuity Across Restarts

**User Story:** As a user, I want my conversation history to survive server restarts, so that I can continue multi-turn conversations without losing context.

#### Acceptance Criteria

1. WHEN the server restarts and a user sends a Follow_Up_Query in an existing Session, THE Inference_Pipeline SHALL include the previously persisted conversation history in the LLM context
2. WHEN the Inference_Pipeline initializes for a session_id that has persisted history, THE Conversation_Store SHALL load all previously stored Messages for that Session
3. WHILE a Session contains more messages than the configured `max_history_length`, THE Conversation_Store SHALL provide only the most recent messages up to the configured limit for LLM context
4. WHEN the Conversation_Store loads a Session from disk, THE Conversation_Store SHALL preserve the original message ordering and timestamps

### Requirement 3: Elimination of Dual Storage

**User Story:** As a developer, I want to remove the redundant in-memory storage, so that there is no possibility of desynchronization between stores.

#### Acceptance Criteria

1. THE Chat_Route SHALL use the Conversation_Store as the sole mechanism for adding Messages to a Session
2. THE Inference_Pipeline SHALL use the Conversation_Store as the sole mechanism for retrieving conversation history
3. WHEN a chat request is processed, THE Chat_Route SHALL add the user Message and assistant Message to the Conversation_Store exactly once each
4. THE Inference_Pipeline SHALL NOT maintain a separate in-memory copy of conversation history

### Requirement 4: History-Aware Follow-Up Routing

**User Story:** As a user, I want the system to understand follow-up questions that reference previous conversation context, so that queries like "tell me more about that phone" trigger product retrieval even without explicit product keywords.

#### Acceptance Criteria

1. WHEN a user sends a Follow_Up_Query that contains contextual references (e.g., pronouns like "that", "it", "those", or phrases like "tell me more", "what about", "how does it compare") and the previous assistant Message contained product information, THE Router SHALL route the query to the retriever node
2. WHEN the Router determines a query is a Follow_Up_Query requiring retrieval, THE Agentic_Workflow SHALL use the conversation history to reformulate the query with explicit product references before calling the Vector_DB
3. WHEN a Follow_Up_Query is detected, THE Agentic_Workflow SHALL pass the conversation history to the retriever node so that the retrieval query includes relevant context from prior turns
4. WHEN the Router analyzes a query, THE Router SHALL consider both the query text and the most recent conversation history to make routing decisions

### Requirement 5: Re-Retrieval for Context-Dependent Follow-Ups

**User Story:** As a user, I want the system to fetch fresh information from the product database when my follow-up question needs new or additional data, so that I get accurate and complete answers.

#### Acceptance Criteria

1. WHEN a Follow_Up_Query requires information not present in the existing conversation context, THE Agentic_Workflow SHALL perform a new Vector_DB retrieval using a reformulated query
2. WHEN reformulating a Follow_Up_Query for retrieval, THE Agentic_Workflow SHALL extract the relevant product or topic references from conversation history and combine them with the current query
3. WHEN the retriever node receives a reformulated query, THE retriever node SHALL return context from the Vector_DB that is relevant to the combined query

### Requirement 8: Conversation Context Injection During Inference

**User Story:** As a user, I want the system to use my full conversation history when generating responses, so that multi-turn answers are coherent and contextually aware.

#### Acceptance Criteria

1. WHEN the Inference_Pipeline generates a response for a query within an existing Session, THE Response_Generator SHALL receive the conversation history from the Conversation_Store as part of the LLM prompt
2. WHEN building the LLM prompt, THE Response_Generator SHALL include the conversation history between the system prompt and the current user query so the LLM can reference prior turns
3. WHILE a Session has conversation history, THE Agentic_Workflow SHALL pass the loaded history into the AgentState messages so that all workflow nodes (Router, Retriever, Generator) have access to prior turns
4. WHEN the Generator node produces a response, THE Generator node SHALL use both the retrieved context and the conversation history to generate a contextually coherent answer
5. IF the conversation history is empty (new Session or first message), THEN THE Inference_Pipeline SHALL proceed with generation without history and produce a valid response

### Requirement 6: Conversation History Format for LLM

**User Story:** As a developer, I want conversation history to be provided to the LLM in the correct LangChain message format, so that the LLM can properly understand the conversation context.

#### Acceptance Criteria

1. WHEN the Inference_Pipeline retrieves history for LLM context, THE Conversation_Store SHALL convert stored Messages to LangChain BaseMessage objects (HumanMessage for "user" role, AIMessage for "assistant" role)
2. THE Conversation_Store SHALL return Messages in chronological order when providing history for LLM context
3. WHILE the number of Messages in a Session exceeds `max_history_length`, THE Conversation_Store SHALL return only the most recent `max_history_length` Messages for LLM context

### Requirement 7: Error Handling for Conversation Storage

**User Story:** As a developer, I want robust error handling for conversation storage operations, so that storage failures do not crash the system or corrupt data.

#### Acceptance Criteria

1. IF the Conversation_Store fails to read a Session file from disk, THEN THE Conversation_Store SHALL return an empty Session and log the error
2. IF the Conversation_Store fails to write a Message to disk, THEN THE Conversation_Store SHALL raise a SessionError with descriptive details
3. IF a Session file contains malformed JSON, THEN THE Conversation_Store SHALL treat the Session as empty and log a warning
4. WHEN adding a Message with an invalid role (not "user" or "assistant"), THE Conversation_Store SHALL reject the Message with a validation error
