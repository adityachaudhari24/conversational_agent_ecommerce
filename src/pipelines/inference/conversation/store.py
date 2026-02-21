"""Unified persistent conversation storage.

This module provides the ConversationStore class that combines file-based
persistence with LangChain message conversion, replacing both the in-memory
ConversationManager and the file-based SessionStore as the single source of
truth for conversation history.
"""

import json
import uuid
import fcntl
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from ..exceptions import SessionError
from ..logging import get_inference_logger


# Get logger for this module
logger = get_inference_logger(__name__)


@dataclass
class SessionMessage:
    """A single message in a conversation."""
    
    role: str  # "user" or "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


@dataclass
class Session:
    """A conversation session."""
    
    session_id: str
    created_at: str
    updated_at: str
    messages: List[SessionMessage] = field(default_factory=list)
    
    @property
    def preview(self) -> str:
        """Get preview text from first user message."""
        for msg in self.messages:
            if msg.role == "user":
                return msg.content[:100] + ("..." if len(msg.content) > 100 else "")
        return ""
    
    @property
    def message_count(self) -> int:
        """Get total message count."""
        return len(self.messages)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [asdict(m) for m in self.messages]
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        """Create Session from dictionary."""
        messages = [
            SessionMessage(**m) for m in data.get("messages", [])
        ]
        return cls(
            session_id=data["session_id"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            messages=messages
        )


class ConversationStore:
    """Unified persistent conversation storage.
    
    Replaces both ConversationManager (in-memory) and SessionStore (file-based).
    All reads go to disk. Writes persist immediately.
    
    This class provides:
    - File-based JSON persistence with fcntl locking
    - Session CRUD operations
    - Message persistence
    - LangChain message conversion
    - History trimming based on max_history_length
    
    Attributes:
        storage_dir: Directory path for storing session files
        max_history_length: Maximum number of messages to retain for LLM context
    """
    
    def __init__(self, storage_dir: str = "data/sessions", max_history_length: int = 10):
        """Initialize the conversation store.
        
        Args:
            storage_dir: Directory path for storing session files
            max_history_length: Maximum messages to return for LLM context
        """
        self.storage_dir = Path(storage_dir)
        self.max_history_length = max_history_length
        
        # Create storage directory if it doesn't exist
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def create_session(self) -> Session:
        """Create a new session with unique ID and persist to disk.
        
        Returns:
            New Session instance
        """
        timestamp = datetime.utcnow()
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        session_id = f"sess_{timestamp_str}_{unique_id}"
        
        now = timestamp.isoformat() + "Z"
        session = Session(
            session_id=session_id,
            created_at=now,
            updated_at=now,
            messages=[]
        )
        
        self._save_session(session)
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Load session from disk. Returns None if not found.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session if found, None otherwise
        """
        session_path = self._get_session_path(session_id)
        if not session_path.exists():
            return None
        
        try:
            with open(session_path, "r") as f:
                # Acquire shared lock for reading
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    data = json.load(f)
                finally:
                    # Release lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return Session.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            # Log warning and return None for malformed JSON
            # (Requirement 7.3: treat malformed JSON as empty session)
            return None
    
    def list_sessions(self) -> List[Session]:
        """List all sessions sorted by most recent first.
        
        Returns:
            List of sessions sorted by updated_at descending
        """
        sessions = []
        for session_file in self.storage_dir.glob("*.json"):
            try:
                with open(session_file, "r") as f:
                    data = json.load(f)
                sessions.append(Session.from_dict(data))
            except (json.JSONDecodeError, KeyError):
                # Skip malformed session files
                continue
        
        # Sort by updated_at descending (most recent first)
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions
    
    def delete_session(self, session_id: str) -> bool:
        """Delete session file from disk.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if deleted, False if not found
        """
        session_path = self._get_session_path(session_id)
        if session_path.exists():
            session_path.unlink()
            return True
        return False
    
    def _get_session_path(self, session_id: str) -> Path:
        """Get file path for session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Path to session JSON file
        """
        return self.storage_dir / f"{session_id}.json"
    
    def add_message(self, session_id: str, role: str, content: str) -> Optional[Session]:
        """Add message to session and persist to disk immediately.
        
        Args:
            session_id: Session identifier
            role: Message role ("user" or "assistant")
            content: Message content
            
        Returns:
            Updated session
            
        Raises:
            ValueError: If role is not "user" or "assistant"
            SessionError: If session not found or write fails
        """
        # Validate role (Requirement 7.4)
        if role not in ("user", "assistant"):
            raise ValueError(
                f"Invalid role '{role}'. Role must be 'user' or 'assistant'."
            )
        
        # Load session from disk (Requirement 7.2)
        session = self.get_session(session_id)
        if session is None:
            raise SessionError(
                f"Session not found: {session_id}",
                session_id=session_id,
                details={"operation": "add_message"}
            )
        
        # Create and append message (Requirement 1.3)
        message = SessionMessage(role=role, content=content)
        session.messages.append(message)
        
        # Update timestamp (Requirement 1.3)
        session.updated_at = datetime.utcnow().isoformat() + "Z"
        
        # Persist to disk immediately (Requirement 1.3)
        self._save_session(session)
        
        return session
    
    def get_langchain_messages(
        self, session_id: str, limit: Optional[int] = None
    ) -> List[BaseMessage]:
        """Load session from disk and convert to LangChain messages.
        
        Converts stored messages to LangChain BaseMessage objects in chronological
        order, returning only the most recent messages up to the specified limit.
        
        Args:
            session_id: Session identifier
            limit: Max messages to return (uses max_history_length if None)
            
        Returns:
            List of HumanMessage/AIMessage in chronological order
            
        Requirements:
            - Req 6.1: Convert to LangChain BaseMessage objects
            - Req 6.2: Return messages in chronological order
            - Req 6.3: Return only most recent limit messages
            - Req 2.3: Trim to max_history_length when limit not specified
        """
        # Load session from disk
        session = self.get_session(session_id)
        if session is None:
            return []
        
        # Use max_history_length if limit not specified
        if limit is None:
            limit = self.max_history_length
        
        # Get the most recent messages (trim from the tail)
        messages = session.messages[-limit:] if limit > 0 else session.messages
        
        # Convert to LangChain messages in chronological order
        langchain_messages = []
        for msg in messages:
            if msg.role == "user":
                langchain_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                langchain_messages.append(AIMessage(content=msg.content))
            # Skip messages with invalid roles (defensive programming)
        
        return langchain_messages
    
    def get_history(
        self, session_id: str, limit: Optional[int] = None
    ) -> List[SessionMessage]:
        """Get raw message history from disk.
        
        Returns raw SessionMessage objects from disk with optional limit.
        Handles malformed JSON by returning empty list and logging a warning.
        
        Args:
            session_id: Session identifier
            limit: Max messages to return (uses max_history_length if None)
            
        Returns:
            List of SessionMessage objects in chronological order
            
        Requirements:
            - Req 7.1: Return raw SessionMessage objects from disk with optional limit
            - Req 7.3: Handle malformed JSON by returning empty list and logging warning
        """
        session_path = self._get_session_path(session_id)
        
        # If session file doesn't exist, return empty list
        if not session_path.exists():
            return []
        
        try:
            with open(session_path, "r") as f:
                # Acquire shared lock for reading
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    data = json.load(f)
                finally:
                    # Release lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            
            # Parse session from JSON
            session = Session.from_dict(data)
            
            # Use max_history_length if limit not specified
            if limit is None:
                limit = self.max_history_length
            
            # Get the most recent messages (trim from the tail)
            messages = session.messages[-limit:] if limit > 0 else session.messages
            
            return messages
            
        except (json.JSONDecodeError, KeyError) as e:
            # Requirement 7.3: Handle malformed JSON by returning empty list and logging warning
            logger.warning(
                f"Malformed JSON in session file {session_id}: {e}. Returning empty history.",
                extra={"extra_fields": {
                    "session_id": session_id,
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "operation": "get_history"
                }}
            )
            return []
    
    def clear_session(self, session_id: str) -> None:
        """Clear all messages from session while keeping the session.
        
        Args:
            session_id: Session identifier to clear
            
        Raises:
            SessionError: If session not found or write fails
        """
        session = self.get_session(session_id)
        if session is None:
            raise SessionError(
                f"Session not found: {session_id}",
                session_id=session_id,
                details={"operation": "clear_session"}
            )
        
        # Clear messages and update timestamp
        session.messages = []
        session.updated_at = datetime.utcnow().isoformat() + "Z"
        
        # Persist to disk
        self._save_session(session)
    
    def get_session_count(self) -> int:
        """Get the number of sessions stored on disk.
        
        Returns:
            Number of session files in storage directory
        """
        return len(list(self.storage_dir.glob("*.json")))
    
    def get_session_ids(self) -> List[str]:
        """Get list of all session IDs.
        
        Returns:
            List of session IDs from all session files
        """
        session_ids = []
        for session_file in self.storage_dir.glob("*.json"):
            # Extract session_id from filename (remove .json extension)
            session_id = session_file.stem
            session_ids.append(session_id)
        return session_ids
    
    def _save_session(self, session: Session) -> None:
        """Save session to file with exclusive file locking.
        
        Args:
            session: Session to save
            
        Raises:
            SessionError: If write fails
        """
        session_path = self._get_session_path(session.session_id)
        
        try:
            with open(session_path, "w") as f:
                # Acquire exclusive lock for writing
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    json.dump(session.to_dict(), f, indent=2)
                finally:
                    # Release lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            raise SessionError(
                f"Failed to write session to disk: {e}",
                session_id=session.session_id,
                details={"path": str(session_path), "error": str(e)}
            ) from e
