# Implementation Plan: Frontend Chatbot

## Overview

This implementation plan breaks down the Frontend Chatbot into discrete coding tasks. The frontend is a Streamlit-based application with a collapsible chat interface in the right corner, communicating with the FastAPI backend.

## Tasks

- [ ] 1. Set up frontend project structure
  - [ ] 1.1 Create frontend directory structure
    - Create `src/frontend/` with `__init__.py`
    - Create subdirectories: `components/`, `utils/`
    - Create `src/frontend/app.py` as main entry point
    - _Requirements: 1.1_

  - [ ] 1.2 Create frontend configuration
    - Create `src/frontend/config.py`
    - Implement FrontendConfig using Pydantic BaseSettings
    - _Requirements: 4.4, 5.6_

  - [ ] 1.3 Create custom CSS styles
    - Create `src/frontend/styles.py`
    - Implement CHAT_STYLES constant with all CSS
    - _Requirements: 1.1, 1.5, 2.2_

- [ ] 2. Implement Session Manager component
  - [ ] 2.1 Create SessionManager class
    - Create `src/frontend/utils/session_manager.py`
    - Implement SessionConfig, ChatMessage dataclasses
    - Implement get_session_id(), get_messages(), add_message(), clear_messages()
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

  - [ ] 2.2 Implement session timeout handling
    - Add _check_session_timeout(), _trim_history() methods
    - _Requirements: 5.6_

  - [ ]* 2.3 Write property test for session management
    - **Property 4: Session Management**
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.6**

- [ ] 3. Implement API Client component
  - [ ] 3.1 Create APIClient class
    - Create `src/frontend/utils/api_client.py`
    - Implement APIConfig, ChatRequest, ChatResponse dataclasses
    - Implement send_message(), check_health()
    - _Requirements: 4.1, 4.2, 4.4_

  - [ ] 3.2 Implement streaming support
    - Add stream_message() async method
    - _Requirements: 4.5, 6.1_

  - [ ] 3.3 Implement error handling
    - Add _handle_error(), _retry_request() methods
    - _Requirements: 4.3, 4.6_

  - [ ]* 3.4 Write property test for API client behavior
    - **Property 3: API Client Behavior**
    - **Validates: Requirements 4.2, 4.3, 4.4**

- [ ] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 5. Implement Message Display component
  - [ ] 5.1 Create MessageDisplay class
    - Create `src/frontend/components/message_display.py`
    - Implement ProductRecommendation dataclass
    - Implement render_message(), render_user_message(), render_assistant_message()
    - _Requirements: 2.1, 2.2, 2.4_

  - [ ] 5.2 Implement markdown and product card rendering
    - Add render_product_cards(), _format_markdown() methods
    - _Requirements: 2.3, 7.1, 7.2_

  - [ ] 5.3 Implement typing indicator
    - Add render_typing_indicator() method
    - _Requirements: 2.6_

  - [ ]* 5.4 Write property test for message display order
    - **Property 1: Message Display Order and Timestamps**
    - **Validates: Requirements 2.1, 2.4**

  - [ ]* 5.5 Write property test for product card rendering
    - **Property 6: Product Card Rendering**
    - **Validates: Requirements 7.2, 7.4**

- [ ] 6. Implement Input Handler component
  - [ ] 6.1 Create InputHandler class
    - Create `src/frontend/components/input_handler.py`
    - Implement InputConfig dataclass
    - Implement render(), _validate_input(), _render_character_count()
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.6_

  - [ ]* 6.2 Write property test for input handling
    - **Property 2: Input Handling**
    - **Validates: Requirements 3.3, 3.4, 3.6**

- [ ] 7. Implement Chat Interface component
  - [ ] 7.1 Create ChatInterface class
    - Create `src/frontend/components/chat_interface.py`
    - Implement ChatConfig dataclass
    - Implement render(), _render_collapsed(), _render_expanded()
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

  - [ ] 7.2 Implement chat header and controls
    - Add _render_header(), _toggle_expanded(), _handle_clear() methods
    - _Requirements: 1.2, 5.4_

  - [ ] 7.3 Implement message sending and streaming
    - Add _handle_send() method with streaming support
    - _Requirements: 6.1, 6.2, 6.3_

  - [ ]* 7.4 Write property test for streaming completion
    - **Property 5: Streaming Completion**
    - **Validates: Requirements 6.3**

- [ ] 8. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 9. Implement Main Application
  - [ ] 9.1 Create main app entry point
    - Update `src/frontend/app.py`
    - Implement main() function with page config
    - Wire all components together
    - _Requirements: 1.1_

  - [ ] 9.2 Implement main content area
    - Add render_main_content() function
    - Add placeholder content for product browsing
    - _Requirements: 1.1_

- [ ] 10. Implement Error Handling UI
  - [ ] 10.1 Create error display components
    - Add render_error_message() to MessageDisplay
    - Implement retry button functionality
    - _Requirements: 8.1, 8.2, 8.3_

  - [ ]* 10.2 Write property test for error display
    - **Property 7: Error Display**
    - **Validates: Requirements 8.1, 8.4**

- [ ] 11. Implement Responsive Design
  - [ ] 11.1 Add responsive CSS
    - Update CHAT_STYLES with media queries
    - Handle mobile layout
    - _Requirements: 9.1, 9.2, 9.3_

- [ ] 12. Implement Accessibility Features
  - [ ] 12.1 Add ARIA labels and keyboard navigation
    - Add aria-label attributes to interactive elements
    - Implement keyboard shortcuts
    - _Requirements: 10.1, 10.2, 10.5_

- [ ] 13. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 14. Create test fixtures and integration tests
  - [ ] 14.1 Create test fixtures
    - Create `tests/fixtures/mock_api_responses.json`
    - Create sample conversation data
    - _Requirements: 4.1_

  - [ ]* 14.2 Write integration test for chat flow
    - Test complete send/receive flow
    - Test session persistence
    - _Requirements: 4.1, 5.3_

- [ ] 15. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- The frontend depends on the FastAPI backend being available
- Use `streamlit run src/frontend/app.py` to test locally
