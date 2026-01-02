"""File-based session storage for chat conversations."""

import json
import uuid
import fcntl
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional


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


class SessionStore:
    """File-based session storage."""
    
    def __init__(self, storage_dir: str = "data/sessions"):
        """Initialize session store.
        
        Args:
            storage_dir: Directory path for storing session files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def create_session(self) -> Session:
        """Create a new session with unique ID.
        
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
        """Load session from file.
        
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
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    data = json.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return Session.from_dict(data)
        except (json.JSONDecodeError, KeyError):
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
                continue
        
        # Sort by updated_at descending (most recent first)
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions
    
    def add_message(self, session_id: str, role: str, content: str) -> Optional[Session]:
        """Add message to session and save to file.
        
        Args:
            session_id: Session identifier
            role: Message role ("user" or "assistant")
            content: Message content
            
        Returns:
            Updated session if found, None otherwise
        """
        session = self.get_session(session_id)
        if session is None:
            return None
        
        message = SessionMessage(role=role, content=content)
        session.messages.append(message)
        session.updated_at = datetime.utcnow().isoformat() + "Z"
        
        self._save_session(session)
        return session
    
    def delete_session(self, session_id: str) -> bool:
        """Delete session file.
        
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
    
    def _save_session(self, session: Session) -> None:
        """Save session to file with file locking.
        
        Args:
            session: Session to save
        """
        session_path = self._get_session_path(session.session_id)
        
        with open(session_path, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(session.to_dict(), f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
