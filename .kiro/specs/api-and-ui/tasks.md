# Implementation Plan: API and UI

## Overview

This implementation plan builds the FastAPI backend first, then the Streamlit frontend. The backend exposes REST API endpoints for chat (with streaming), session management, and health checks. The frontend provides a collapsible chat interface with session history panel.

## Directory Structure

```
src/
├── api/
│   ├── __init__.py
│   ├── main.py              # FastAPI app entry point
│   ├── config.py            # API configuration
│   ├── dependencies.py      # Dependency injection
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── chat.py          # Chat endpoints
│   │   ├── sessions.py      # Session endpoints
│   │   └── health.py        # Health endpoint
│   ├── services/
│   │   ├── __init__.py
│   │   └── session_store.py # File-based session storage
│   └── models/
│       ├── __init__.py
│       └── schemas.py       # Pydantic models
├── frontend/
│   ├── __init__.py
│   ├── app.py               # Streamlit entry point
│   ├── config.py            # Frontend configuration
│   ├── components/
│   │   ├── __init__.py
│   │   ├── session_panel.py # Sidebar session list
│   │   └── chat_interface.py # Collapsible chat
│   └── utils/
│       ├── __init__.py
│       └── api_client.py    # Backend API client
```

## Tasks

### Phase 1: Backend Foundation

- [x] 1. Set up API project structure
  - [x] 1.1 Create API directory structure
    - Create `src/api/` with `__init__.py`
    - Create subdirectories: `routes/`, `services/`, `models/`
    - _Requirements: 1.1_

  - [x] 1.2 Create API configuration
    - Create `src/api/config.py`
    - Implement APISettings using Pydantic BaseSettings
    - Support host, port, cors_origins, session_dir settings
    - _Requirements: 13.1, 13.2_

  - [x] 1.3 Create Pydantic schemas
    - Create `src/api/models/schemas.py`
    - Implement ChatRequest, ChatResponse, SessionResponse, HealthResponse
    - _Requirements: 1.2, 1.3_

- [x] 2. Implement Session Store
  - [x] 2.1 Create SessionStore class
    - Create `src/api/services/session_store.py`
    - Implement Session and SessionMessage dataclasses
    - Implement create_session() with unique ID generation (timestamp + uuid)
    - Implement get_session(), list_sessions(), delete_session()
    - _Requirements: 4.4, 4.5, 4.6_

  - [x] 2.2 Implement message persistence
    - Implement add_message() that saves to file immediately
    - Implement file locking for concurrent access
    - Store as JSON in data/sessions/{session_id}.json
    - _Requirements: 10.1, 10.2, 10.3, 10.5_

  - [ ]* 2.3 Write property test for session persistence
    - **Property 3: Session Persistence**
    - **Validates: Requirements 10.1, 10.3, 10.4**

  - [ ]* 2.4 Write property test for session creation
    - **Property 4: Session Creation**
    - **Validates: Requirements 4.5, 9.6**

- [ ] 3. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

### Phase 2: Backend API Endpoints

- [x] 4. Implement Health Endpoint
  - [x] 4.1 Create health router
    - Create `src/api/routes/health.py`
    - Implement GET /api/health endpoint
    - Check inference pipeline availability
    - Return status and service details
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

  - [ ]* 4.2 Write property test for health check accuracy
    - **Property 6: Health Check Accuracy**
    - **Validates: Requirements 3.2, 3.4**

- [x] 5. Implement Session Endpoints
  - [x] 5.1 Create session router
    - Create `src/api/routes/sessions.py`
    - Implement POST /api/sessions (create new session)
    - Implement GET /api/sessions (list all sessions)
    - Implement GET /api/sessions/{session_id} (get specific session)
    - _Requirements: 4.1, 4.2_

  - [x]* 5.2 Implement delete session endpoint (optional)
    - Implement DELETE /api/sessions/{session_id}
    - _Requirements: 4.3_

  - [ ]* 5.3 Write property test for session list ordering
    - **Property 5: Session List Ordering**
    - **Validates: Requirements 9.2, 9.8**

- [x] 6. Implement Chat Endpoint
  - [x] 6.1 Create chat router
    - Create `src/api/routes/chat.py`
    - Implement POST /api/chat endpoint
    - Integrate with InferencePipeline
    - Save messages to session via SessionStore
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [ ]* 6.2 Write property test for chat request validation
    - **Property 1: Chat Request Validation**
    - **Validates: Requirements 1.2, 1.4, 1.5**

- [x] 7. Implement Streaming Chat Endpoint
  - [x] 7.1 Add streaming endpoint
    - Implement POST /api/chat/stream endpoint
    - Use StreamingResponse with SSE format
    - Stream chunks from InferencePipeline.stream()
    - Send final completion event
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

  - [ ]* 7.2 Write property test for streaming completeness
    - **Property 2: Streaming Response Completeness**
    - **Validates: Requirements 2.3, 2.4**

- [x] 8. Create FastAPI Main Application
  - [x] 8.1 Create main app entry point
    - Create `src/api/main.py`
    - Configure FastAPI app with metadata
    - Add CORS middleware
    - Include all routers
    - Initialize inference pipeline on startup
    - _Requirements: 14.1, 14.2, 14.3, 14.4_

  - [x] 8.2 Create dependencies module
    - Create `src/api/dependencies.py`
    - Implement get_inference_pipeline() dependency
    - Implement get_session_store() dependency
    - _Requirements: 1.2_

- [ ] 9. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ]* 10. Implement API Authentication (Optional)
  - [ ]* 10.1 Add API key authentication
    - Create authentication middleware
    - Validate API key from X-API-Key header
    - Allow disabling via configuration
    - _Requirements: 15.1, 15.2, 15.3, 15.4_

### Phase 3: Frontend Foundation

- [x] 11. Set up Frontend project structure
  - [x] 11.1 Create frontend directory structure
    - Create `src/frontend/` with `__init__.py`
    - Create subdirectories: `components/`, `utils/`
    - _Requirements: 5.1_

  - [x] 11.2 Create frontend configuration
    - Create `src/frontend/config.py`
    - Implement FrontendSettings with api_base_url, chat_width, max_input_length
    - _Requirements: 13.3, 13.4_

- [x] 12. Implement API Client
  - [x] 12.1 Create APIClient class
    - Create `src/frontend/utils/api_client.py`
    - Implement send_message() for non-streaming
    - Implement stream_message() for SSE streaming
    - Implement create_session(), get_sessions(), get_session()
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

  - [ ]* 12.2 Write property test for API client behavior
    - **Property 7: Error Response Structure**
    - **Validates: Requirements 12.5**

- [ ] 13. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

### Phase 4: Frontend UI Components

- [x] 14. Implement Session Panel
  - [x] 14.1 Create session panel component
    - Create `src/frontend/components/session_panel.py`
    - Implement render_session_panel() for sidebar
    - Display "New Chat" button
    - List past sessions with preview and timestamp
    - Highlight current session
    - Sort by most recent first
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8_

- [x] 15. Implement Chat Interface
  - [x] 15.1 Create chat interface component
    - Create `src/frontend/components/chat_interface.py`
    - Implement collapsible chat in right corner
    - Implement toggle button when collapsed
    - Fixed width (400px), adjustable height
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

  - [x] 15.2 Implement message display
    - Show messages in chronological order
    - Distinguish user vs assistant messages
    - Support markdown formatting
    - Display timestamps
    - Auto-scroll to latest message
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [x] 15.3 Implement user input
    - Text input at bottom of chat
    - Send via Enter key or button
    - Disable send when empty
    - Clear input after sending
    - Show character count
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [x] 15.4 Implement streaming display
    - Show typing indicator during streaming
    - Display chunks as they arrive
    - Finalize message on completion
    - Handle mid-stream errors
    - Disable input during streaming
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 6.6_

- [x] 16. Implement Main Application
  - [x] 16.1 Create main app entry point
    - Create `src/frontend/app.py`
    - Configure page settings
    - Initialize session state
    - Create new session on page load/refresh
    - Wire session panel and chat interface
    - _Requirements: 9.7_

- [x] 17. Implement Error Handling UI
  - [x] 17.1 Add error display and retry
    - Display clear error messages
    - Add retry button for failed requests
    - Show loading states
    - Show validation feedback
    - _Requirements: 12.1, 12.2, 12.3, 12.4_

- [ ]* 18. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

### Phase 5: Integration and Polish

- [-] 19. Add custom CSS styling
  - [x] 19.1 Create styles for chat interface
    - Create `src/frontend/styles.py`
    - Style collapsible chat container
    - Style message bubbles (user vs assistant)
    - Style input area
    - Add responsive behavior
    - _Requirements: 5.5, 6.2_

- [ ] 20. Create test fixtures
  - [ ]* 20.1 Create API test fixtures
    - Create `tests/fixtures/mock_chat_responses.json`
    - Create sample session data
    - _Requirements: 1.1_

  - [ ]* 20.2 Write integration test for chat flow
    - Test complete send/receive flow
    - Test streaming flow
    - Test session persistence
    - _Requirements: 1.1, 2.1, 10.1_

- [ ] 21. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

### Phase 6: Frontend Redesign (Simple UI Fix)

- [x] 22. Redesign Frontend with Simple Layout
  - [x] 22.1 Create simple main page layout
    - Redesign `src/frontend/app.py` with clean, simple layout
    - Remove complex CSS and fixed positioning
    - Create main content area with e-commerce banner/hero section
    - Use standard Streamlit columns for layout
    - _Requirements: 5.1, 9.1_

  - [x] 22.2 Implement simple sidebar for sessions
    - Redesign sidebar to show "New Chat" button prominently
    - List recent sessions with simple click-to-load functionality
    - Remove complex session management, keep it minimal
    - Show current session indicator
    - _Requirements: 9.1, 9.2, 9.5, 9.6_

  - [x] 22.3 Create simple right-column chat interface
    - Replace floating chat widget with right column chat
    - Use standard Streamlit chat components (st.chat_message, st.chat_input)
    - Implement proper message display with user/assistant distinction
    - Add simple loading indicator during API calls
    - _Requirements: 5.1, 6.1, 6.2, 7.1, 7.2_

  - [x] 22.4 Fix API integration
    - Ensure proper error handling for API calls
    - Implement retry logic for failed requests
    - Add proper session management (create/load/save)
    - Test with actual backend API endpoints
    - _Requirements: 11.1, 11.2, 12.1, 12.2_

  - [x] 22.5 Add simple e-commerce content
    - Create basic product showcase in main area
    - Add simple category sections
    - Include call-to-action to use the chat assistant
    - Keep styling minimal and clean
    - _Requirements: 5.1_

- [ ] 23. Test complete UI flow
  - [ ] 23.1 Test session creation and management
    - Verify new sessions are created properly
    - Test loading existing sessions from sidebar
    - Verify messages persist correctly
    - _Requirements: 4.5, 9.6, 10.1_

  - [ ] 23.2 Test chat functionality
    - Test sending messages and receiving responses
    - Verify error handling works properly
    - Test with both successful and failed API calls
    - _Requirements: 1.1, 1.2, 12.1_

- [ ] 24. Final UI checkpoint
  - Ensure all UI components work correctly, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Backend is implemented first (Phases 1-2), then frontend (Phases 3-4)
- Phase 6 completely redesigns the frontend with a simpler approach
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- The API depends on the inference pipeline being available
- Sessions are stored as JSON files in `data/sessions/`
- Use `uv run uvicorn src.api.main:app --reload` to run backend
- Use `uv run streamlit run src/frontend/app.py` to run frontend
