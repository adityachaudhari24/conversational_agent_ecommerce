"""Session management endpoints."""

from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException

from ..models.schemas import (
    SessionResponse,
    SessionDetailResponse,
    SessionListResponse,
    MessageResponse,
)
from ..dependencies import get_conversation_store
from src.pipelines.inference.conversation.store import ConversationStore

router = APIRouter(tags=["sessions"])


@router.post("/sessions", response_model=SessionResponse)
async def create_session(
    conversation_store: ConversationStore = Depends(get_conversation_store)
) -> SessionResponse:
    """Create a new chat session."""
    session = conversation_store.create_session()
    
    return SessionResponse(
        session_id=session.session_id,
        created_at=datetime.fromisoformat(session.created_at.rstrip("Z")),
        updated_at=datetime.fromisoformat(session.updated_at.rstrip("Z")),
        message_count=session.message_count,
        preview=session.preview
    )


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    conversation_store: ConversationStore = Depends(get_conversation_store)
) -> SessionListResponse:
    """List all sessions sorted by most recent first."""
    sessions = conversation_store.list_sessions()
    
    session_responses = [
        SessionResponse(
            session_id=s.session_id,
            created_at=datetime.fromisoformat(s.created_at.rstrip("Z")),
            updated_at=datetime.fromisoformat(s.updated_at.rstrip("Z")),
            message_count=s.message_count,
            preview=s.preview
        )
        for s in sessions
    ]
    
    return SessionListResponse(
        sessions=session_responses,
        total=len(session_responses)
    )


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session(
    session_id: str,
    conversation_store: ConversationStore = Depends(get_conversation_store)
) -> SessionDetailResponse:
    """Get a specific session with all messages."""
    session = conversation_store.get_session(session_id)
    
    if session is None:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "SESSION_NOT_FOUND", "message": f"Session {session_id} not found"}
        )
    
    messages = [
        MessageResponse(
            role=m.role,
            content=m.content,
            timestamp=datetime.fromisoformat(m.timestamp.rstrip("Z"))
        )
        for m in session.messages
    ]
    
    return SessionDetailResponse(
        session_id=session.session_id,
        created_at=datetime.fromisoformat(session.created_at.rstrip("Z")),
        updated_at=datetime.fromisoformat(session.updated_at.rstrip("Z")),
        messages=messages
    )


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    conversation_store: ConversationStore = Depends(get_conversation_store)
) -> dict:
    """Delete a session (optional endpoint)."""
    deleted = conversation_store.delete_session(session_id)
    
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "SESSION_NOT_FOUND", "message": f"Session {session_id} not found"}
        )
    
    return {"message": f"Session {session_id} deleted"}
