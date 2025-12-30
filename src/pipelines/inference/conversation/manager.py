"""Conversation management for the inference pipeline.

This module provides conversation session management with in-memory storage,
message history tracking, and LangChain message format conversion.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from ..config import ConversationConfig
from ..exceptions import SessionError


@dataclass
class Message:
    """A single message in a conversation.
    
    Attributes:
        role: Message role ("user" or "assistant")
        content: Message content text
        timestamp: When the message was created
    """
    
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self) -> None:
        """Validate message role after initialization."""
        if self.role not in ("user", "assistant"):
            raise ValueError(f"Message role must be 'user' or 'assistant', got: {self.role}")


@dataclass
class Session:
    """A conversation session containing message history.
    
    Attributes:
        session_id: Unique identifier for the session
        messages: List of messages in chronological order
        created_at: When the session was created
    """
    
    session_id: str
    messages: List[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


class ConversationManager:
    """Manages in-memory conversation sessions.
    
    This class provides session management functionality including:
    - Creating and retrieving sessions
    - Adding messages to sessions
    - Managing conversation history with configurable limits
    - Converting to LangChain message format
    
    Sessions are stored in memory only and will be lost when the process restarts.
    For production use, consider implementing persistent storage.
    """
    
    def __init__(self, config: ConversationConfig) -> None:
        """Initialize the conversation manager.
        
        Args:
            config: Configuration for conversation management
        """
        self.config = config
        self._sessions: Dict[str, Session] = {}
    
    def get_or_create_session(self, session_id: str) -> Session:
        """Get existing session or create new one.
        
        Args:
            session_id: Unique identifier for the session
            
        Returns:
            Session object (existing or newly created)
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(session_id=session_id)
        
        return self._sessions[session_id]
    
    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Add message to session history.
        
        Args:
            session_id: Session to add message to
            role: Message role ("user" or "assistant")
            content: Message content
            
        Raises:
            ValueError: If role is not "user" or "assistant"
            SessionError: If message cannot be added to session
        """
        try:
            # Get or create session
            session = self.get_or_create_session(session_id)
            
            # Create and add message
            message = Message(role=role, content=content)
            session.messages.append(message)
            
            # Trim history if needed
            self._trim_history(session)
            
        except ValueError as e:
            raise e  # Re-raise validation errors as-is
        except Exception as e:
            raise SessionError(
                f"Failed to add message to session: {e}",
                session_id=session_id,
                details={"role": role, "content_length": len(content)}
            ) from e
    
    def get_history(self, session_id: str, limit: Optional[int] = None) -> List[Message]:
        """Get conversation history for session.
        
        Args:
            session_id: Session to get history for
            limit: Maximum number of messages to return (None for all)
            
        Returns:
            List of messages in chronological order
        """
        session = self.get_or_create_session(session_id)
        messages = session.messages
        
        if limit is not None and limit > 0:
            messages = messages[-limit:]
        
        return messages.copy()  # Return copy to prevent external modification
    
    def get_langchain_messages(self, session_id: str) -> List[BaseMessage]:
        """Convert history to LangChain message format.
        
        Args:
            session_id: Session to convert messages for
            
        Returns:
            List of LangChain BaseMessage objects
        """
        messages = self.get_history(session_id)
        langchain_messages: List[BaseMessage] = []
        
        for message in messages:
            if message.role == "user":
                langchain_messages.append(HumanMessage(content=message.content))
            elif message.role == "assistant":
                langchain_messages.append(AIMessage(content=message.content))
            # Skip messages with invalid roles (shouldn't happen due to validation)
        
        return langchain_messages
    
    def clear_session(self, session_id: str) -> None:
        """Clear all messages from session.
        
        Args:
            session_id: Session to clear
        """
        if session_id in self._sessions:
            self._sessions[session_id].messages.clear()
    
    def _trim_history(self, session: Session) -> None:
        """Remove oldest messages if history exceeds limit.
        
        Args:
            session: Session to trim
        """
        max_length = self.config.max_history_length
        if len(session.messages) > max_length:
            # Keep the most recent messages
            session.messages = session.messages[-max_length:]
    
    def get_session_count(self) -> int:
        """Get the number of active sessions.
        
        Returns:
            Number of sessions currently in memory
        """
        return len(self._sessions)
    
    def get_session_ids(self) -> List[str]:
        """Get list of all session IDs.
        
        Returns:
            List of session IDs currently in memory
        """
        return list(self._sessions.keys())
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session completely.
        
        Args:
            session_id: Session to delete
            
        Returns:
            True if session was deleted, False if it didn't exist
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False