"""Session history panel component."""

import streamlit as st
from datetime import datetime
from typing import List, Dict, Any

from src.frontend.utils.api_client import APIClient


def render_session_panel(api_client: APIClient):
    """Render session history in sidebar.
    
    Args:
        api_client: API client instance
    """
    with st.sidebar:
        # Header with banner
        st.markdown("""
        <div style="text-align: center; padding: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 20px;">
            <h2 style="color: white; margin: 0;">🛒 E-Commerce</h2>
            <p style="color: #e0e0e0; margin: 5px 0 0 0; font-size: 14px;">AI Shopping Assistant</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.title("💬 Chat History")
        
        # New Chat button
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            _create_new_session(api_client)
            st.rerun()
        
        st.divider()
        
        # Load and display sessions
        try:
            sessions = api_client.get_sessions()
            
            if not sessions:
                st.info("No chat history yet. Start a new conversation!")
            else:
                for session in sessions:
                    _render_session_item(session, api_client)
                    
        except Exception as e:
            st.error(f"Failed to load sessions: {str(e)}")


def _render_session_item(session: Dict[str, Any], api_client: APIClient):
    """Render a single session item.
    
    Args:
        session: Session data dict
        api_client: API client instance
    """
    session_id = session.get("session_id", "")
    preview = session.get("preview", "New conversation")[:50]
    if not preview:
        preview = "New conversation"
    
    # Format timestamp
    updated_at = session.get("updated_at", "")
    try:
        dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        time_str = dt.strftime("%b %d, %H:%M")
    except:
        time_str = ""
    
    message_count = session.get("message_count", 0)
    
    # Check if this is the current session
    is_current = st.session_state.get("current_session_id") == session_id
    
    # Create container with styling
    container_style = "border-left: 3px solid #667eea;" if is_current else ""
    
    with st.container():
        col1, col2 = st.columns([4, 1])
        
        with col1:
            # Session button
            button_label = f"{'📍 ' if is_current else ''}{preview}"
            if st.button(
                button_label,
                key=f"session_{session_id}",
                use_container_width=True,
                disabled=is_current
            ):
                _load_session(session_id, api_client)
                st.rerun()
        
        with col2:
            st.caption(f"{message_count}💬")
        
        st.caption(f"📅 {time_str}")
        st.markdown("---")


def _create_new_session(api_client: APIClient):
    """Create a new session and set it as current.
    
    Args:
        api_client: API client instance
    """
    try:
        session = api_client.create_session()
        st.session_state.current_session_id = session["session_id"]
        st.session_state.messages = []
        st.session_state.is_read_only = False
    except Exception as e:
        st.error(f"Failed to create session: {str(e)}")


def _load_session(session_id: str, api_client: APIClient):
    """Load a session and display it (read-only).
    
    Args:
        session_id: Session identifier
        api_client: API client instance
    """
    try:
        session = api_client.get_session(session_id)
        if session:
            st.session_state.current_session_id = session_id
            st.session_state.messages = [
                {"role": m["role"], "content": m["content"]}
                for m in session.get("messages", [])
            ]
            # Past sessions are read-only
            st.session_state.is_read_only = True
    except Exception as e:
        st.error(f"Failed to load session: {str(e)}")
