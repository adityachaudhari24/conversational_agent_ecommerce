"""Collapsible chat interface component."""

import streamlit as st
from typing import Optional

from src.frontend.utils.api_client import APIClient
from src.frontend.config import settings


def render_chat_interface(api_client: APIClient):
    """Render the chat interface.
    
    Args:
        api_client: API client instance
    """
    # Check if chat is expanded
    if not st.session_state.get("chat_expanded", True):
        _render_chat_button()
    else:
        _render_chat_window(api_client)


def _render_chat_button():
    """Render collapsed chat button."""
    # Use columns to position button on the right
    col1, col2, col3 = st.columns([6, 1, 1])
    
    with col3:
        if st.button("💬", help="Open chat"):
            st.session_state.chat_expanded = True
            st.rerun()


def _render_chat_window(api_client: APIClient):
    """Render expanded chat window.
    
    Args:
        api_client: API client instance
    """
    # Header with minimize button
    _render_chat_header()
    
    # Check if we have a session
    if not st.session_state.get("current_session_id"):
        st.info("👋 Welcome! Click 'New Chat' in the sidebar to start a conversation.")
        return
    
    # Read-only indicator
    if st.session_state.get("is_read_only", False):
        st.warning("📖 Viewing past conversation (read-only). Click 'New Chat' to start a new conversation.")
    
    # Message display area
    _render_messages()
    
    # Input area (only if not read-only)
    if not st.session_state.get("is_read_only", False):
        _render_input(api_client)


def _render_chat_header():
    """Render chat header with controls."""
    col1, col2 = st.columns([5, 1])
    
    with col1:
        st.subheader("🤖 AI Assistant")
    
    with col2:
        if st.button("➖", help="Minimize chat"):
            st.session_state.chat_expanded = False
            st.rerun()


def _render_messages():
    """Render conversation messages."""
    messages = st.session_state.get("messages", [])
    
    if not messages:
        st.markdown("""
        <div style="text-align: center; padding: 40px; color: #888;">
            <p>👋 Hi! I'm your e-commerce assistant.</p>
            <p>Ask me about products, recommendations, I am trained on 2013 iphone reviews data on amazon</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Display messages
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def _render_input(api_client: APIClient):
    """Render input field and handle user input.
    
    Args:
        api_client: API client instance
    """
    # Check if currently streaming
    if st.session_state.get("is_streaming", False):
        st.info("⏳ Generating response...")
        return
    
    # Chat input
    if prompt := st.chat_input(
        "Ask about products...",
        max_chars=settings.max_input_length
    ):
        _handle_user_input(prompt, api_client)


def _handle_user_input(prompt: str, api_client: APIClient):
    """Handle user input and generate response.
    
    Args:
        prompt: User message
        api_client: API client instance
    """
    session_id = st.session_state.get("current_session_id")
    if not session_id:
        st.error("No active session. Please create a new chat.")
        return
    
    # Add user message to display
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response with streaming
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            st.session_state.is_streaming = True
            
            # Stream response
            for chunk in api_client.stream_message(prompt, session_id):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            
            # Final display without cursor
            message_placeholder.markdown(full_response)
            
            # Add to messages
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response
            })
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            message_placeholder.error(error_msg)
            
            # Add error to messages
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Sorry, I encountered an error: {str(e)}"
            })
            
            # Show retry button
            if st.button("🔄 Retry"):
                # Remove the error message and retry
                st.session_state.messages = st.session_state.messages[:-2]
                st.rerun()
        
        finally:
            st.session_state.is_streaming = False
