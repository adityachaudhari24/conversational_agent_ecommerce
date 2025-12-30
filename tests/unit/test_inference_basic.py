"""Basic tests for inference pipeline components.

This module contains basic functionality tests for the inference pipeline
components that have been implemented so far.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from src.pipelines.inference.config import ConversationConfig, LLMConfig
from src.pipelines.inference.conversation.manager import ConversationManager, Message, Session
from src.pipelines.inference.exceptions import ConfigurationError, SessionError
from src.pipelines.inference.llm.client import LLMClient


class TestConversationManager:
    """Test conversation manager functionality."""
    
    def test_create_session(self):
        """Test session creation."""
        config = ConversationConfig(max_history_length=10)
        manager = ConversationManager(config)
        
        session = manager.get_or_create_session("test_session")
        
        assert session.session_id == "test_session"
        assert len(session.messages) == 0
        assert isinstance(session.created_at, datetime)
    
    def test_add_message(self):
        """Test adding messages to session."""
        config = ConversationConfig(max_history_length=10)
        manager = ConversationManager(config)
        
        manager.add_message("test_session", "user", "Hello")
        manager.add_message("test_session", "assistant", "Hi there!")
        
        history = manager.get_history("test_session")
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[0].content == "Hello"
        assert history[1].role == "assistant"
        assert history[1].content == "Hi there!"
    
    def test_invalid_message_role(self):
        """Test that invalid message roles raise ValueError."""
        config = ConversationConfig(max_history_length=10)
        manager = ConversationManager(config)
        
        with pytest.raises(ValueError):
            manager.add_message("test_session", "invalid_role", "Test message")
    
    def test_history_trimming(self):
        """Test that history is trimmed when it exceeds max length."""
        config = ConversationConfig(max_history_length=3)
        manager = ConversationManager(config)
        
        # Add more messages than the limit
        for i in range(5):
            manager.add_message("test_session", "user", f"Message {i}")
        
        history = manager.get_history("test_session")
        assert len(history) == 3
        # Should keep the most recent messages
        assert history[0].content == "Message 2"
        assert history[1].content == "Message 3"
        assert history[2].content == "Message 4"
    
    def test_langchain_message_conversion(self):
        """Test conversion to LangChain message format."""
        config = ConversationConfig(max_history_length=10)
        manager = ConversationManager(config)
        
        manager.add_message("test_session", "user", "Hello")
        manager.add_message("test_session", "assistant", "Hi there!")
        
        langchain_messages = manager.get_langchain_messages("test_session")
        
        assert len(langchain_messages) == 2
        assert langchain_messages[0].content == "Hello"
        assert langchain_messages[1].content == "Hi there!"
        # Check message types
        assert hasattr(langchain_messages[0], 'type')
        assert hasattr(langchain_messages[1], 'type')
    
    def test_clear_session(self):
        """Test clearing session messages."""
        config = ConversationConfig(max_history_length=10)
        manager = ConversationManager(config)
        
        manager.add_message("test_session", "user", "Hello")
        manager.add_message("test_session", "assistant", "Hi there!")
        
        assert len(manager.get_history("test_session")) == 2
        
        manager.clear_session("test_session")
        
        assert len(manager.get_history("test_session")) == 0


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


class TestMessage:
    """Test Message dataclass."""
    
    def test_valid_message_creation(self):
        """Test creating valid messages."""
        message = Message(role="user", content="Hello")
        
        assert message.role == "user"
        assert message.content == "Hello"
        assert isinstance(message.timestamp, datetime)
    
    def test_invalid_message_role(self):
        """Test that invalid roles raise ValueError."""
        with pytest.raises(ValueError):
            Message(role="invalid", content="Hello")


class TestSession:
    """Test Session dataclass."""
    
    def test_session_creation(self):
        """Test creating a session."""
        session = Session(session_id="test")
        
        assert session.session_id == "test"
        assert len(session.messages) == 0
        assert isinstance(session.created_at, datetime)