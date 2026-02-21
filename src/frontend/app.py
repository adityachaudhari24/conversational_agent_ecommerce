"""Streamlit frontend entry point."""

import streamlit as st

from src.frontend.config import settings
from src.frontend.utils.api_client import APIClient


def initialize_session_state(api_client: APIClient):
    """Initialize Streamlit session state."""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.messages = []
        st.session_state.is_streaming = False
        st.session_state.is_read_only = False

        # Create new session on page load
        try:
            session = api_client.create_session()
            st.session_state.current_session_id = session["session_id"]
        except Exception:
            st.session_state.current_session_id = None


def render_sidebar(api_client: APIClient):
    """Render sidebar with session management."""
    with st.sidebar:
        # Logo/Brand
        st.markdown("## 🛒 TechStore")
        st.caption("AI Shopping Assistant")

        st.divider()

        # New Chat button
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            try:
                session = api_client.create_session()
                st.session_state.current_session_id = session["session_id"]
                st.session_state.messages = []
                st.session_state.is_read_only = False
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {str(e)}")

        st.divider()
        st.markdown("**Chat History**")

        # Load and display recent sessions
        try:
            sessions = api_client.get_sessions()

            if not sessions:
                st.caption("No previous chats")
            else:
                for session in sessions[:8]:
                    session_id = session.get("session_id", "")
                    preview = session.get("preview", "New chat")[:30] or "New chat"

                    is_current = st.session_state.get("current_session_id") == session_id
                    prefix = "📍 " if is_current else "💬 "

                    if st.button(
                        f"{prefix}{preview}...",
                        key=f"session_{session_id}",
                        disabled=is_current,
                        use_container_width=True
                    ):
                        try:
                            session_data = api_client.get_session(session_id)
                            if session_data:
                                st.session_state.current_session_id = session_id
                                st.session_state.messages = [
                                    {"role": m["role"], "content": m["content"]}
                                    for m in session_data.get("messages", [])
                                ]
                                st.session_state.is_read_only = True
                                st.rerun()
                        except Exception as e:
                            st.error(f"Failed: {str(e)}")

        except Exception as e:
            st.error(f"Failed to load history: {str(e)}")


def handle_user_input(prompt: str, api_client: APIClient):
    """Handle user input and generate response."""
    session_id = st.session_state.get("current_session_id")
    if not session_id:
        st.error("No active session. Please start a new chat.")
        return

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            st.session_state.is_streaming = True

            # Try streaming first
            try:
                for chunk in api_client.stream_message(prompt, session_id):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)

            except Exception:
                # Fallback to non-streaming
                response = api_client.send_message(prompt, session_id)
                full_response = response.get("response", "Sorry, I couldn't process your request.")
                message_placeholder.markdown(full_response)

            # Add to messages
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response
            })

        except Exception as e:
            message_placeholder.error(f"Error: {str(e)}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Sorry, I encountered an error: {str(e)}"
            })
        finally:
            st.session_state.is_streaming = False


def main():
    """Main application entry point."""
    # Page configuration
    st.set_page_config(
        page_title="TechStore - AI Shopping Assistant",
        page_icon="🛒",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS with better colors
    st.markdown("""
    <style>
        /* Hide Streamlit defaults */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* Banner styling */
        .banner {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem 2rem;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 1rem;
        }
        .banner h1 {
            margin: 0;
            font-size: 2rem;
        }
        .banner p {
            margin: 0.5rem 0 0 0;
            opacity: 0.9;
        }

        /* User message - purple gradient background */
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 0.75rem;
        }
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) p {
            color: white !important;
        }

        /* Assistant message - light gray background with dark text */
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
            background: #f0f2f6;
            border: 1px solid #ddd;
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 0.75rem;
        }
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) p {
            color: #1a1a2e !important;
        }

        /* Chat input styling */
        .stChatInput > div {
            border-radius: 24px !important;
            border: 2px solid #667eea !important;
        }
        .stChatInput input {
            color: #1a1a2e !important;
        }

        /* Main container */
        .main .block-container {
            padding-top: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # Initialize API client and session state
    api_client = APIClient()
    initialize_session_state(api_client)

    # Sidebar
    render_sidebar(api_client)

    # Main content area
    # Banner
    st.markdown("""
    <div class="banner">
        <h1>🛒 TechStore</h1>
        <p>Your AI-powered shopping assistant - Ask me anything about our products!</p>
    </div>
    """, unsafe_allow_html=True)

    # Chat interface
    st.markdown("### 💬 Chat with AI Assistant")

    # Check for active session
    if not st.session_state.get("current_session_id"):
        st.info("👋 Click 'New Chat' in the sidebar to start a conversation.")
        return

    # Read-only notice
    if st.session_state.get("is_read_only", False):
        st.warning("📖 Viewing past conversation (read-only). Start a new chat to continue.")

    # Chat container
    chat_container = st.container()

    with chat_container:
        # Display messages
        messages = st.session_state.get("messages", [])

        if not messages:
            st.caption("👋 Hi! I'm your shopping assistant. Ask me about iphones I am trained on 2013 iphone reviews data on amazon!")
        else:
            for msg in messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

    # Chat input (only if not read-only)
    if not st.session_state.get("is_read_only", False):
        if st.session_state.get("is_streaming", False):
            st.info("⏳ Generating response...")
        else:
            if prompt := st.chat_input("Ask about products...", max_chars=settings.max_input_length):
                handle_user_input(prompt, api_client)


if __name__ == "__main__":
    main()
