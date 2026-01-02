"""Simple session history sidebar."""

import streamlit as st
from datetime import datetime
from typing import List, Dict, Any

from src.frontend.utils.api_client import APIClient


def render_session_history_sidebar(api_client: APIClient):
    """Render session history in sidebar."""
    with st.sidebar:
        st.markdown("### 💬 Chat History")
        
        # New Chat button
        if st.button("➕ Start New Chat", use_container_width=True, type="primary"):
            try:
                session = api_client.create_session()
                st.session_state.current_session_id = session["session_id"]
                st.session_state.messages = []
                st.success("New chat started!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to create session: {str(e)}")
        
        st.divider()
        
        # Load sessions
        try:
            sessions = api_client.get_sessions()
            
            if not sessions:
                st.info("No chat history yet.")
            else:
                st.markdown("**Recent Conversations:**")
                
                for session in sessions[:10]:  # Show last 10 sessions
                    session_id = session.get("session_id", "")
                    preview = session.get("preview", "New conversation")[:40]
                    if not preview:
                        preview = "New conversation"
                    
                    # Format timestamp
                    updated_at = session.get("updated_at", "")
                    try:
                        dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                        time_str = dt.strftime("%m/%d %H:%M")
                    except:
                        time_str = ""
                    
                    # Current session indicator
                    is_current = st.session_state.get("current_session_id") == session_id
                    prefix = "📍 " if is_current else "💬 "
                    
                    if st.button(
                        f"{prefix}{preview}...",
                        key=f"session_{session_id}",
                        help=f"Created: {time_str}",
                        disabled=is_current,
                        use_container_width=True
                    ):
                        load_session(session_id, api_client)
                        st.rerun()
                        
        except Exception as e:
            st.error(f"Failed to load sessions: {str(e)}")


def load_session(session_id: str, api_client: APIClient):
    """Load a session and display it."""
    try:
        session = api_client.get_session(session_id)
        if session:
            st.session_state.current_session_id = session_id
            st.session_state.messages = [
                {"role": m["role"], "content": m["content"]}
                for m in session.get("messages", [])
            ]
    except Exception as e:
        st.error(f"Failed to load session: {str(e)}")