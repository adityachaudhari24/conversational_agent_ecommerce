# Design Document: Frontend Chatbot

## Overview

The Frontend is a Streamlit-based web application featuring a collapsible chat interface positioned in the right corner. The chatbot enables users to ask questions about products and receive AI-powered responses with product recommendations. The frontend communicates with the FastAPI backend via REST API and supports streaming responses.

The design prioritizes:
- **Usability**: Intuitive chat interface with clear visual hierarchy
- **Responsiveness**: Works across desktop, tablet, and mobile devices
- **Performance**: Streaming responses for real-time feedback
- **Accessibility**: WCAG AA compliant with keyboard navigation

## Architecture

```mermaid
flowchart TB
    subgraph Frontend["Streamlit Frontend"]
        APP[Main App]
        CI[Chat Interface]
        MD[Message Display]
        IH[Input Handler]
        SM[Session Manager]
        AC[API Client]
    end
    
    subgraph Backend["FastAPI Backend"]
        CHAT[/chat endpoint]
        STREAM[/chat/stream endpoint]
        HEALTH[/health endpoint]
    end
    
    subgraph State["Session State"]
        SS[Streamlit Session]
        HIST[Message History]
        SID[Session ID]
    end
    
    APP --> CI
    CI --> MD
    CI --> IH
    CI --> SM
    IH --> AC
    AC --> CHAT
    AC --> STREAM
    SM --> SS
    SS --> HIST
    SS --> SID
```

## Components and Interfaces

### 1. Main Application

Entry point for the Streamlit application.

```python
# src/frontend/app.py
import streamlit as st
from components.chat_interface import ChatInterface
from components.session_manager import SessionManager
from utils.api_client import APIClient
from config import FrontendConfig

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="E-Commerce Assistant",
        page_icon="🛒",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Initialize components
    config = FrontendConfig()
    session_manager = SessionManager()
    api_client = APIClient(config.api_base_url)
    
    # Render main page content
    render_main_content()
    
    # Render chat interface in right corner
    chat_interface = ChatInterface(
        session_manager=session_manager,
        api_client=api_client,
        config=config
    )
    chat_interface.render()

def render_main_content():
    """Render the main page content."""
    st.title("🛒 E-Commerce Product Assistant")
    st.markdown("Browse products and ask questions using the chat assistant.")
    # Additional main content here

if __name__ == "__main__":
    main()
```

### 2. Chat Interface

Main chat component with collapsible container.

```python
from dataclasses import dataclass
from typing import Optional
import streamlit as st

@dataclass
class ChatConfig:
    width: int = 400
    min_height: int = 400
    max_height: int = 600
    position: str = "right"  # "right" or "left"
    default_expanded: bool = True

class ChatInterface:
    """Collapsible chat interface component."""
    
    def __init__(
        self,
        session_manager: 'SessionManager',
        api_client: 'APIClient',
        config: ChatConfig
    ):
        self.session_manager = session_manager
        self.api_client = api_client
        self.config = config
        self.message_display = MessageDisplay()
        self.input_handler = InputHandler()
    
    def render(self) -> None:
        """Render the chat interface."""
        pass
    
    def _render_collapsed(self) -> None:
        """Render collapsed chat button."""
        pass
    
    def _render_expanded(self) -> None:
        """Render expanded chat window."""
        pass
    
    def _render_header(self) -> None:
        """Render chat header with title and controls."""
        pass
    
    def _render_messages(self) -> None:
        """Render message history."""
        pass
    
    def _render_input(self) -> None:
        """Render input area."""
        pass
    
    def _handle_send(self, message: str) -> None:
        """Handle sending a message."""
        pass
    
    def _handle_clear(self) -> None:
        """Handle clearing chat history."""
        pass
    
    def _toggle_expanded(self) -> None:
        """Toggle chat expanded/collapsed state."""
        pass
```

### 3. Message Display

Renders individual messages with formatting.

```python
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime
import streamlit as st

@dataclass
class ChatMessage:
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    recommendations: Optional[List['ProductRecommendation']] = None
    is_streaming: bool = False
    error: Optional[str] = None

@dataclass
class ProductRecommendation:
    product_name: str
    price: Optional[str] = None
    rating: Optional[str] = None
    description: Optional[str] = None

class MessageDisplay:
    """Renders chat messages with formatting."""
    
    def render_message(self, message: ChatMessage) -> None:
        """Render a single message."""
        pass
    
    def render_user_message(self, message: ChatMessage) -> None:
        """Render user message with styling."""
        pass
    
    def render_assistant_message(self, message: ChatMessage) -> None:
        """Render assistant message with markdown support."""
        pass
    
    def render_product_cards(
        self, 
        recommendations: List[ProductRecommendation]
    ) -> None:
        """Render product recommendation cards."""
        pass
    
    def render_typing_indicator(self) -> None:
        """Render typing/loading indicator."""
        pass
    
    def render_error_message(self, error: str) -> None:
        """Render error message with retry option."""
        pass
    
    def render_timestamp(self, timestamp: datetime) -> None:
        """Render message timestamp."""
        pass
    
    def _format_markdown(self, content: str) -> str:
        """Format content with markdown."""
        pass
```

### 4. Input Handler

Handles user input with validation.

```python
from dataclasses import dataclass
from typing import Callable, Optional
import streamlit as st

@dataclass
class InputConfig:
    max_length: int = 500
    placeholder: str = "Ask about products..."
    multiline: bool = True

class InputHandler:
    """Handles user input with validation."""
    
    def __init__(self, config: InputConfig = None):
        self.config = config or InputConfig()
    
    def render(
        self, 
        on_send: Callable[[str], None],
        disabled: bool = False
    ) -> None:
        """Render input field and send button."""
        pass
    
    def _validate_input(self, text: str) -> tuple[bool, Optional[str]]:
        """Validate input and return (is_valid, error_message)."""
        pass
    
    def _render_character_count(self, current: int) -> None:
        """Render character count indicator."""
        pass
    
    def _handle_key_press(self, key: str) -> None:
        """Handle keyboard shortcuts."""
        pass
```

### 5. Session Manager

Manages user sessions and conversation state.

```python
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime, timedelta
import uuid
import streamlit as st

@dataclass
class SessionConfig:
    timeout_minutes: int = 30
    max_history_length: int = 50

class SessionManager:
    """Manages user sessions and conversation state."""
    
    def __init__(self, config: SessionConfig = None):
        self.config = config or SessionConfig()
        self._initialize_session()
    
    def _initialize_session(self) -> None:
        """Initialize session state if not exists."""
        pass
    
    def get_session_id(self) -> str:
        """Get or create session ID."""
        pass
    
    def get_messages(self) -> List[ChatMessage]:
        """Get conversation history."""
        pass
    
    def add_message(self, message: ChatMessage) -> None:
        """Add message to history."""
        pass
    
    def update_last_message(self, content: str) -> None:
        """Update the last message (for streaming)."""
        pass
    
    def clear_messages(self) -> None:
        """Clear conversation history."""
        pass
    
    def is_chat_expanded(self) -> bool:
        """Check if chat is expanded."""
        pass
    
    def set_chat_expanded(self, expanded: bool) -> None:
        """Set chat expanded state."""
        pass
    
    def _check_session_timeout(self) -> bool:
        """Check if session has timed out."""
        pass
    
    def _trim_history(self) -> None:
        """Trim history if exceeds max length."""
        pass
```

### 6. API Client

Communicates with the FastAPI backend.

```python
from dataclasses import dataclass
from typing import Optional, AsyncIterator, Dict, Any
import httpx
import asyncio

@dataclass
class APIConfig:
    base_url: str = "http://localhost:8000"
    timeout_seconds: int = 30
    retry_attempts: int = 3

@dataclass
class ChatRequest:
    query: str
    session_id: str
    stream: bool = False

@dataclass
class ChatResponse:
    response: str
    session_id: str
    recommendations: List[ProductRecommendation] = None
    metadata: Dict[str, Any] = None
    error: Optional[str] = None

class APIClient:
    """Client for communicating with FastAPI backend."""
    
    def __init__(self, config: APIConfig = None):
        self.config = config or APIConfig()
        self.client = httpx.Client(timeout=self.config.timeout_seconds)
    
    def send_message(self, request: ChatRequest) -> ChatResponse:
        """Send chat message and get response."""
        pass
    
    async def stream_message(
        self, 
        request: ChatRequest
    ) -> AsyncIterator[str]:
        """Stream chat response chunks."""
        pass
    
    def check_health(self) -> bool:
        """Check if backend is available."""
        pass
    
    def _handle_error(self, error: Exception) -> ChatResponse:
        """Handle API errors gracefully."""
        pass
    
    def _retry_request(
        self, 
        func, 
        *args, 
        **kwargs
    ) -> Any:
        """Retry failed requests."""
        pass
```

## UI Layout

### Desktop Layout (>= 768px)

```
+--------------------------------------------------+
|  Header: E-Commerce Product Assistant            |
+--------------------------------------------------+
|                                    +------------+|
|  Main Content Area                 | Chat       ||
|  - Product listings                | Interface  ||
|  - Search results                  |            ||
|  - etc.                            | [Messages] ||
|                                    |            ||
|                                    | [Input]    ||
|                                    +------------+|
+--------------------------------------------------+
```

### Mobile Layout (< 768px)

```
+----------------------+
| Header               |
+----------------------+
| Main Content         |
|                      |
|                      |
+----------------------+
| [Chat Button]        |
+----------------------+

When expanded:
+----------------------+
| Chat Interface       |
| (Full Width)         |
|                      |
| [Messages]           |
|                      |
| [Input]              |
| [Close Button]       |
+----------------------+
```

## Styling

### CSS Custom Styles

```python
CHAT_STYLES = """
<style>
/* Chat container */
.chat-container {
    position: fixed;
    right: 20px;
    bottom: 20px;
    width: 400px;
    max-height: 600px;
    background: white;
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    z-index: 1000;
    display: flex;
    flex-direction: column;
}

/* Chat header */
.chat-header {
    padding: 16px;
    background: #1f77b4;
    color: white;
    border-radius: 12px 12px 0 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

/* Message container */
.message-container {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
    max-height: 400px;
}

/* User message */
.user-message {
    background: #e3f2fd;
    padding: 12px 16px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0;
    margin-left: 40px;
    max-width: 80%;
    float: right;
    clear: both;
}

/* Assistant message */
.assistant-message {
    background: #f5f5f5;
    padding: 12px 16px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px 0;
    margin-right: 40px;
    max-width: 80%;
    float: left;
    clear: both;
}

/* Product card */
.product-card {
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 12px;
    margin: 8px 0;
}

/* Input area */
.input-area {
    padding: 16px;
    border-top: 1px solid #e0e0e0;
    display: flex;
    gap: 8px;
}

/* Collapsed button */
.chat-button {
    position: fixed;
    right: 20px;
    bottom: 20px;
    width: 60px;
    height: 60px;
    border-radius: 50%;
    background: #1f77b4;
    color: white;
    border: none;
    cursor: pointer;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    z-index: 1000;
}

/* Typing indicator */
.typing-indicator {
    display: flex;
    gap: 4px;
    padding: 12px;
}

.typing-dot {
    width: 8px;
    height: 8px;
    background: #888;
    border-radius: 50%;
    animation: typing 1.4s infinite;
}

@keyframes typing {
    0%, 60%, 100% { transform: translateY(0); }
    30% { transform: translateY(-10px); }
}

/* Responsive */
@media (max-width: 768px) {
    .chat-container {
        width: 100%;
        height: 100%;
        right: 0;
        bottom: 0;
        border-radius: 0;
    }
}
</style>
"""
```

## Data Models

### Frontend Models

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class MessageModel(BaseModel):
    """Chat message model."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    recommendations: Optional[List['ProductRecommendationModel']] = None
    is_streaming: bool = False
    error: Optional[str] = None

class ProductRecommendationModel(BaseModel):
    """Product recommendation model."""
    product_name: str
    price: Optional[str] = None
    rating: Optional[str] = None
    description: Optional[str] = None

class SessionStateModel(BaseModel):
    """Session state model."""
    session_id: str
    messages: List[MessageModel] = []
    is_expanded: bool = True
    last_activity: datetime = Field(default_factory=datetime.utcnow)
```

### Configuration

```python
from pydantic import BaseSettings, Field

class FrontendConfig(BaseSettings):
    """Frontend configuration."""
    
    # API Settings
    api_base_url: str = Field(default="http://localhost:8000", env="API_BASE_URL")
    api_timeout: int = Field(default=30)
    
    # Chat Settings
    chat_width: int = Field(default=400)
    chat_max_height: int = Field(default=600)
    default_expanded: bool = Field(default=True)
    
    # Session Settings
    session_timeout_minutes: int = Field(default=30)
    max_history_length: int = Field(default=50)
    
    # Input Settings
    max_input_length: int = Field(default=500)
    enable_streaming: bool = Field(default=True)
    
    class Config:
        env_file = ".env"
```



## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Message Display Order and Timestamps

*For any* list of N messages in the conversation:
- Messages SHALL be displayed in chronological order (oldest first, newest last)
- Each message SHALL have a timestamp that is >= the previous message's timestamp

**Validates: Requirements 2.1, 2.4**

### Property 2: Input Handling

*For any* user input operation:
- If the input text is empty or whitespace-only, the Send button SHALL be disabled
- After a successful send, the input field SHALL be cleared (empty string)
- The character count display SHALL show the current length and maximum limit

**Validates: Requirements 3.3, 3.4, 3.6**

### Property 3: API Client Behavior

*For any* API request:
- The request SHALL include a valid session_id
- If the API returns an error status, a user-friendly error message SHALL be displayed
- If the request exceeds timeout_seconds, a timeout error SHALL be raised

**Validates: Requirements 4.2, 4.3, 4.4**

### Property 4: Session Management

*For any* user session:
- The session_id SHALL be unique (UUID format)
- The session_id SHALL be stored in Streamlit session state
- Messages added to the session SHALL be retrievable from session state
- After clear_messages() is called, get_messages() SHALL return an empty list
- If session age exceeds timeout_minutes, the session SHALL be considered expired

**Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.6**

### Property 5: Streaming Completion

*For any* streaming response that completes successfully:
- The final message content SHALL contain the complete response
- The message is_streaming flag SHALL be False after completion

**Validates: Requirements 6.3**

### Property 6: Product Card Rendering

*For any* response containing N product recommendations:
- N product cards SHALL be rendered
- Each card SHALL display product_name (required), and price/rating if available

**Validates: Requirements 7.2, 7.4**

### Property 7: Error Display

*For any* error condition:
- An error message SHALL be displayed to the user
- If input validation fails, specific validation feedback SHALL be shown

**Validates: Requirements 8.1, 8.4**

## Error Handling

### Error Types

```python
class FrontendError(Exception):
    """Base exception for frontend errors."""
    pass

class APIError(FrontendError):
    """Raised when API communication fails."""
    def __init__(self, message: str, status_code: int = None):
        super().__init__(message)
        self.status_code = status_code

class TimeoutError(FrontendError):
    """Raised when API request times out."""
    pass

class ValidationError(FrontendError):
    """Raised when input validation fails."""
    def __init__(self, message: str, field: str = None):
        super().__init__(message)
        self.field = field

class SessionError(FrontendError):
    """Raised when session operations fail."""
    pass
```

### Error Handling Strategy

| Component | Error Type | User Message |
|-----------|-----------|--------------|
| API Client | Connection error | "Unable to connect to server. Please try again." |
| API Client | Timeout | "Request timed out. Please try again." |
| API Client | Server error (5xx) | "Server error. Please try again later." |
| Input Handler | Empty input | "Please enter a message" |
| Input Handler | Too long | "Message exceeds maximum length" |
| Session Manager | Session expired | "Session expired. Starting new conversation." |
| Streaming | Mid-stream failure | "Response interrupted. Showing partial response." |

## Testing Strategy

### Testing Framework

- **Framework**: pytest with pytest-streamlit for Streamlit testing
- **Property Testing**: hypothesis for property-based tests
- **Mocking**: unittest.mock for API mocking
- **E2E Testing**: Selenium or Playwright for browser testing

### Unit Tests

1. **SessionManager Tests**
   - Test session ID generation (uniqueness)
   - Test message storage and retrieval
   - Test history clearing
   - Test session timeout

2. **APIClient Tests**
   - Test successful request/response
   - Test error handling
   - Test timeout handling
   - Test streaming

3. **InputHandler Tests**
   - Test input validation
   - Test character count
   - Test send button state

4. **MessageDisplay Tests**
   - Test message ordering
   - Test markdown rendering
   - Test product card rendering

### Property-Based Tests

```python
from hypothesis import given, strategies as st

@given(st.lists(
    st.fixed_dictionaries({
        'role': st.sampled_from(['user', 'assistant']),
        'content': st.text(min_size=1),
        'timestamp': st.datetimes()
    }),
    min_size=0,
    max_size=20
))
def test_message_display_order(messages):
    """
    Feature: frontend-chatbot, Property 1: Message Display Order
    Messages should be displayed in chronological order.
    """
    # Sort by timestamp and verify order
    pass

@given(st.text())
def test_input_handling(input_text):
    """
    Feature: frontend-chatbot, Property 2: Input Handling
    Empty input should disable send, non-empty should enable.
    """
    pass
```

### Integration Tests

1. **Chat Flow Integration**
   - Test complete send/receive flow
   - Test conversation continuity
   - Test streaming display

2. **Session Integration**
   - Test session persistence
   - Test clear chat functionality

### E2E Tests

1. **User Journey Tests**
   - Test chat toggle (expand/collapse)
   - Test sending messages
   - Test receiving responses
   - Test error scenarios

### Test Commands

```bash
# Run unit tests
uv run pytest tests/unit/test_frontend/ -v

# Run with coverage
uv run pytest tests/ --cov=src/frontend --cov-report=html

# Run E2E tests
uv run pytest tests/e2e/ -v --browser chromium
```
