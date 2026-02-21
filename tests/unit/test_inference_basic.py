"""Basic tests for inference pipeline components.

This module contains basic functionality tests for the inference pipeline
components that have been implemented so far.
"""

import pytest
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from src.pipelines.inference.config import ConversationConfig, LLMConfig
from src.pipelines.inference.conversation import ConversationStore, Session, SessionMessage
from src.pipelines.inference.exceptions import ConfigurationError, SessionError
from src.pipelines.inference.llm.client import LLMClient


class TestConversationStore:
    """Test conversation store functionality."""
    
    @pytest.fixture
    def temp_storage_dir(self):
        """Create a temporary directory for test storage."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup after test
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_create_session(self, temp_storage_dir):
        """Test session creation."""
        store = ConversationStore(storage_dir=temp_storage_dir, max_history_length=10)
        
        session = store.create_session()
        
        assert session.session_id.startswith("sess_")
        assert len(session.messages) == 0
        assert session.created_at is not None
        
        # Verify session was persisted to disk
        retrieved_session = store.get_session(session.session_id)
        assert retrieved_session is not None
        assert retrieved_session.session_id == session.session_id
    
    def test_add_message(self, temp_storage_dir):
        """Test adding messages to session."""
        store = ConversationStore(storage_dir=temp_storage_dir, max_history_length=10)
        
        session = store.create_session()
        session_id = session.session_id
        
        store.add_message(session_id, "user", "Hello")
        store.add_message(session_id, "assistant", "Hi there!")
        
        history = store.get_history(session_id)
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[0].content == "Hello"
        assert history[1].role == "assistant"
        assert history[1].content == "Hi there!"
    
    def test_invalid_message_role(self, temp_storage_dir):
        """Test that invalid message roles raise ValueError."""
        store = ConversationStore(storage_dir=temp_storage_dir, max_history_length=10)
        
        session = store.create_session()
        
        with pytest.raises(ValueError):
            store.add_message(session.session_id, "invalid_role", "Test message")
    
    def test_history_trimming(self, temp_storage_dir):
        """Test that history is trimmed when it exceeds max length."""
        store = ConversationStore(storage_dir=temp_storage_dir, max_history_length=3)
        
        session = store.create_session()
        session_id = session.session_id
        
        # Add more messages than the limit
        for i in range(5):
            store.add_message(session_id, "user", f"Message {i}")
        
        # get_history with limit should return only the most recent messages
        history = store.get_history(session_id, limit=3)
        assert len(history) == 3
        # Should keep the most recent messages
        assert history[0].content == "Message 2"
        assert history[1].content == "Message 3"
        assert history[2].content == "Message 4"
    
    def test_langchain_message_conversion(self, temp_storage_dir):
        """Test conversion to LangChain message format."""
        store = ConversationStore(storage_dir=temp_storage_dir, max_history_length=10)
        
        session = store.create_session()
        session_id = session.session_id
        
        store.add_message(session_id, "user", "Hello")
        store.add_message(session_id, "assistant", "Hi there!")
        
        langchain_messages = store.get_langchain_messages(session_id)
        
        assert len(langchain_messages) == 2
        assert langchain_messages[0].content == "Hello"
        assert langchain_messages[1].content == "Hi there!"
        # Check message types
        assert hasattr(langchain_messages[0], 'type')
        assert hasattr(langchain_messages[1], 'type')
    
    def test_clear_session(self, temp_storage_dir):
        """Test clearing session messages."""
        store = ConversationStore(storage_dir=temp_storage_dir, max_history_length=10)
        
        session = store.create_session()
        session_id = session.session_id
        
        store.add_message(session_id, "user", "Hello")
        store.add_message(session_id, "assistant", "Hi there!")
        
        assert len(store.get_history(session_id)) == 2
        
        store.clear_session(session_id)
        
        assert len(store.get_history(session_id)) == 0


class TestLLMClient:
    """Test LLM client functionality."""
    
    def test_initialization_without_api_key(self):
        """Test that missing API key raises ConfigurationError."""
        config = LLMConfig(api_key=None)
        client = LLMClient(config)
        
        with pytest.raises(ConfigurationError) as exc_info:
            client.initialize()
        
        assert "API key is required" in str(exc_info.value)
        assert exc_info.value.error_code == "MISSING_API_KEY"
    
    def test_initialization_with_empty_api_key(self):
        """Test that empty API key raises ConfigurationError."""
        config = LLMConfig(api_key="   ")  # Whitespace only
        client = LLMClient(config)
        
        with pytest.raises(ConfigurationError) as exc_info:
            client.initialize()
        
        assert "cannot be empty" in str(exc_info.value)
        assert exc_info.value.error_code == "EMPTY_API_KEY"
    
    def test_invoke_without_initialization(self):
        """Test that invoke raises error when not initialized."""
        config = LLMConfig(api_key="test-key")
        client = LLMClient(config)
        
        with pytest.raises(ConfigurationError) as exc_info:
            client.invoke([])
        
        assert "must be initialized" in str(exc_info.value)
        assert exc_info.value.error_code == "CLIENT_NOT_INITIALIZED"
    
    @patch('src.pipelines.inference.llm.client.OpenAI')
    @patch('src.pipelines.inference.llm.client.AsyncOpenAI')
    @patch('src.pipelines.inference.llm.client.ChatOpenAI')
    def test_successful_initialization(self, mock_chat_openai, mock_async_openai, mock_openai):
        """Test successful client initialization."""
        config = LLMConfig(api_key="test-key")
        client = LLMClient(config)
        
        # Mock the clients
        mock_openai.return_value = Mock()
        mock_async_openai.return_value = Mock()
        mock_chat_openai.return_value = Mock()
        
        client.initialize()
        
        assert client._initialized is True
        assert client.client is not None
        assert client.async_client is not None
        assert client.langchain_client is not None


class TestSessionMessage:
    """Test SessionMessage dataclass."""
    
    def test_valid_message_creation(self):
        """Test creating valid messages."""
        message = SessionMessage(role="user", content="Hello")
        
        assert message.role == "user"
        assert message.content == "Hello"
        assert message.timestamp is not None
        # Timestamp should be in ISO format with Z suffix
        assert message.timestamp.endswith("Z")


class TestSession:
    """Test Session dataclass."""
    
    def test_session_creation(self):
        """Test creating a session."""
        now = datetime.utcnow().isoformat() + "Z"
        session = Session(
            session_id="test",
            created_at=now,
            updated_at=now,
            messages=[]
        )
        
        assert session.session_id == "test"
        assert len(session.messages) == 0
        assert session.created_at == now
        assert session.updated_at == now