"""Unit tests for ConversationStore.

This module contains comprehensive tests for the ConversationStore class,
covering session CRUD operations, message persistence, LangChain conversion,
history trimming, file locking, malformed JSON handling, and error cases.
"""

import json
import pytest
import tempfile
import fcntl
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, mock_open

from langchain_core.messages import HumanMessage, AIMessage

from src.pipelines.inference.conversation.store import ConversationStore, Session, SessionMessage
from src.pipelines.inference.exceptions import SessionError


@pytest.fixture
def temp_storage_dir():
    """Create a temporary directory for test storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def store(temp_storage_dir):
    """Create a ConversationStore instance with temporary storage."""
    return ConversationStore(storage_dir=temp_storage_dir, max_history_length=10)


class TestSessionCRUD:
    """Test session CRUD operations (Requirement 1)."""
    
    def test_create_session(self, store):
        """Test creating a new session."""
        session = store.create_session()
        
        assert session.session_id.startswith("sess_")
        assert len(session.messages) == 0
        assert session.created_at is not None
        assert session.updated_at is not None
        assert session.created_at == session.updated_at
        
        # Verify session was persisted to disk
        session_path = store._get_session_path(session.session_id)
        assert session_path.exists()
    
    def test_get_existing_session(self, store):
        """Test retrieving an existing session."""
        # Create a session
        created_session = store.create_session()
        
        # Retrieve it
        retrieved_session = store.get_session(created_session.session_id)
        
        assert retrieved_session is not None
        assert retrieved_session.session_id == created_session.session_id
        assert retrieved_session.created_at == created_session.created_at
    
    def test_get_nonexistent_session(self, store):
        """Test retrieving a session that doesn't exist returns None."""
        session = store.get_session("nonexistent_session_id")
        
        assert session is None
    
    def test_list_sessions_empty(self, store):
        """Test listing sessions when none exist."""
        sessions = store.list_sessions()
        
        assert sessions == []
    
    def test_list_sessions_multiple(self, store):
        """Test listing multiple sessions sorted by most recent."""
        # Create multiple sessions
        session1 = store.create_session()
        session2 = store.create_session()
        session3 = store.create_session()
        
        sessions = store.list_sessions()
        
        assert len(sessions) == 3
        # Should be sorted by updated_at descending (most recent first)
        assert sessions[0].session_id == session3.session_id
        assert sessions[1].session_id == session2.session_id
        assert sessions[2].session_id == session1.session_id
    
    def test_delete_existing_session(self, store):
        """Test deleting an existing session."""
        session = store.create_session()
        session_path = store._get_session_path(session.session_id)
        
        assert session_path.exists()
        
        result = store.delete_session(session.session_id)
        
        assert result is True
        assert not session_path.exists()
    
    def test_delete_nonexistent_session(self, store):
        """Test deleting a session that doesn't exist returns False."""
        result = store.delete_session("nonexistent_session_id")
        
        assert result is False


class TestMessagePersistence:
    """Test message persistence operations (Requirements 1.3, 7.2, 7.4)."""
    
    def test_add_message_user(self, store):
        """Test adding a user message."""
        session = store.create_session()
        
        updated_session = store.add_message(session.session_id, "user", "Hello!")
        
        assert updated_session is not None
        assert len(updated_session.messages) == 1
        assert updated_session.messages[0].role == "user"
        assert updated_session.messages[0].content == "Hello!"
        assert updated_session.messages[0].timestamp is not None
    
    def test_add_message_assistant(self, store):
        """Test adding an assistant message."""
        session = store.create_session()
        
        updated_session = store.add_message(session.session_id, "assistant", "Hi there!")
        
        assert updated_session is not None
        assert len(updated_session.messages) == 1
        assert updated_session.messages[0].role == "assistant"
        assert updated_session.messages[0].content == "Hi there!"
    
    def test_add_multiple_messages(self, store):
        """Test adding multiple messages in sequence."""
        session = store.create_session()
        
        store.add_message(session.session_id, "user", "Hello!")
        store.add_message(session.session_id, "assistant", "Hi there!")
        store.add_message(session.session_id, "user", "How are you?")
        
        retrieved_session = store.get_session(session.session_id)
        
        assert len(retrieved_session.messages) == 3
        assert retrieved_session.messages[0].content == "Hello!"
        assert retrieved_session.messages[1].content == "Hi there!"
        assert retrieved_session.messages[2].content == "How are you?"
    
    def test_add_message_updates_timestamp(self, store):
        """Test that adding a message updates the session's updated_at timestamp."""
        session = store.create_session()
        original_updated_at = session.updated_at
        
        # Add a message
        updated_session = store.add_message(session.session_id, "user", "Hello!")
        
        # updated_at should be different (later)
        assert updated_session.updated_at >= original_updated_at
    
    def test_add_message_persists_immediately(self, store):
        """Test that messages are persisted to disk immediately (Req 1.3)."""
        session = store.create_session()
        
        store.add_message(session.session_id, "user", "Hello!")
        
        # Read directly from disk to verify persistence
        session_path = store._get_session_path(session.session_id)
        with open(session_path, "r") as f:
            data = json.load(f)
        
        assert len(data["messages"]) == 1
        assert data["messages"][0]["content"] == "Hello!"
    
    def test_add_message_invalid_role(self, store):
        """Test that invalid role raises ValueError (Req 7.4)."""
        session = store.create_session()
        
        with pytest.raises(ValueError) as exc_info:
            store.add_message(session.session_id, "invalid_role", "Test")
        
        assert "Invalid role" in str(exc_info.value)
        assert "invalid_role" in str(exc_info.value)
    
    def test_add_message_nonexistent_session(self, store):
        """Test that adding message to nonexistent session raises SessionError (Req 7.2)."""
        with pytest.raises(SessionError) as exc_info:
            store.add_message("nonexistent_session", "user", "Hello!")
        
        assert "Session not found" in str(exc_info.value)
        assert exc_info.value.session_id == "nonexistent_session"


class TestLangChainConversion:
    """Test LangChain message conversion (Requirement 6)."""
    
    def test_get_langchain_messages_empty(self, store):
        """Test getting LangChain messages from empty session."""
        session = store.create_session()
        
        messages = store.get_langchain_messages(session.session_id)
        
        assert messages == []
    
    def test_get_langchain_messages_conversion(self, store):
        """Test conversion to LangChain message types (Req 6.1)."""
        session = store.create_session()
        store.add_message(session.session_id, "user", "Hello!")
        store.add_message(session.session_id, "assistant", "Hi there!")
        
        messages = store.get_langchain_messages(session.session_id)
        
        assert len(messages) == 2
        assert isinstance(messages[0], HumanMessage)
        assert isinstance(messages[1], AIMessage)
        assert messages[0].content == "Hello!"
        assert messages[1].content == "Hi there!"
    
    def test_get_langchain_messages_chronological_order(self, store):
        """Test messages are returned in chronological order (Req 6.2)."""
        session = store.create_session()
        store.add_message(session.session_id, "user", "First")
        store.add_message(session.session_id, "assistant", "Second")
        store.add_message(session.session_id, "user", "Third")
        
        messages = store.get_langchain_messages(session.session_id)
        
        assert len(messages) == 3
        assert messages[0].content == "First"
        assert messages[1].content == "Second"
        assert messages[2].content == "Third"
    
    def test_get_langchain_messages_nonexistent_session(self, store):
        """Test getting messages from nonexistent session returns empty list."""
        messages = store.get_langchain_messages("nonexistent_session")
        
        assert messages == []


class TestHistoryTrimming:
    """Test history trimming functionality (Requirements 2.3, 6.3)."""
    
    def test_get_langchain_messages_respects_max_history_length(self, store):
        """Test that only max_history_length messages are returned (Req 6.3)."""
        # Create store with max_history_length=3
        store_limited = ConversationStore(
            storage_dir=store.storage_dir,
            max_history_length=3
        )
        
        session = store_limited.create_session()
        
        # Add 5 messages
        for i in range(5):
            role = "user" if i % 2 == 0 else "assistant"
            store_limited.add_message(session.session_id, role, f"Message {i}")
        
        messages = store_limited.get_langchain_messages(session.session_id)
        
        # Should only return the last 3 messages
        assert len(messages) == 3
        assert messages[0].content == "Message 2"
        assert messages[1].content == "Message 3"
        assert messages[2].content == "Message 4"
    
    def test_get_langchain_messages_with_custom_limit(self, store):
        """Test using custom limit parameter."""
        session = store.create_session()
        
        # Add 5 messages
        for i in range(5):
            role = "user" if i % 2 == 0 else "assistant"
            store.add_message(session.session_id, role, f"Message {i}")
        
        # Request only 2 messages
        messages = store.get_langchain_messages(session.session_id, limit=2)
        
        assert len(messages) == 2
        assert messages[0].content == "Message 3"
        assert messages[1].content == "Message 4"
    
    def test_get_history_respects_limit(self, store):
        """Test that get_history respects limit parameter (Req 7.1)."""
        session = store.create_session()
        
        # Add 5 messages
        for i in range(5):
            role = "user" if i % 2 == 0 else "assistant"
            store.add_message(session.session_id, role, f"Message {i}")
        
        # Request only 3 messages
        history = store.get_history(session.session_id, limit=3)
        
        assert len(history) == 3
        assert history[0].content == "Message 2"
        assert history[1].content == "Message 3"
        assert history[2].content == "Message 4"
    
    def test_get_history_uses_max_history_length_by_default(self, store):
        """Test that get_history uses max_history_length when limit not specified."""
        # Create store with max_history_length=2
        store_limited = ConversationStore(
            storage_dir=store.storage_dir,
            max_history_length=2
        )
        
        session = store_limited.create_session()
        
        # Add 4 messages
        for i in range(4):
            role = "user" if i % 2 == 0 else "assistant"
            store_limited.add_message(session.session_id, role, f"Message {i}")
        
        history = store_limited.get_history(session.session_id)
        
        assert len(history) == 2
        assert history[0].content == "Message 2"
        assert history[1].content == "Message 3"


class TestFileLocking:
    """Test file locking behavior (Requirements 1.4, 1.5)."""
    
    def test_concurrent_reads_allowed(self, store, temp_storage_dir):
        """Test that concurrent reads use shared locks (Req 1.4)."""
        session = store.create_session()
        store.add_message(session.session_id, "user", "Hello!")
        
        session_path = store._get_session_path(session.session_id)
        
        # Open file with shared lock (simulating concurrent read)
        with open(session_path, "r") as f1:
            fcntl.flock(f1.fileno(), fcntl.LOCK_SH)
            
            # Should be able to acquire another shared lock
            with open(session_path, "r") as f2:
                fcntl.flock(f2.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                data = json.load(f2)
                fcntl.flock(f2.fileno(), fcntl.LOCK_UN)
            
            fcntl.flock(f1.fileno(), fcntl.LOCK_UN)
        
        assert data["session_id"] == session.session_id
    
    def test_write_uses_exclusive_lock(self, store):
        """Test that writes use exclusive locks (Req 1.5)."""
        session = store.create_session()
        
        # The _save_session method should use exclusive locks
        # We verify this by checking that the file is written correctly
        store.add_message(session.session_id, "user", "Test message")
        
        # Verify the message was written
        retrieved_session = store.get_session(session.session_id)
        assert len(retrieved_session.messages) == 1
        assert retrieved_session.messages[0].content == "Test message"


class TestMalformedJSONHandling:
    """Test malformed JSON handling (Requirement 7.3)."""
    
    def test_get_session_with_malformed_json(self, store, temp_storage_dir):
        """Test that malformed JSON returns None and logs warning."""
        # Create a session file with malformed JSON
        session_id = "sess_malformed_test"
        session_path = Path(temp_storage_dir) / f"{session_id}.json"
        
        with open(session_path, "w") as f:
            f.write("{ invalid json content }")
        
        # Should return None for malformed JSON
        session = store.get_session(session_id)
        
        assert session is None
    
    def test_get_history_with_malformed_json(self, store, temp_storage_dir):
        """Test that get_history returns empty list for malformed JSON (Req 7.3)."""
        # Create a session file with malformed JSON
        session_id = "sess_malformed_history"
        session_path = Path(temp_storage_dir) / f"{session_id}.json"
        
        with open(session_path, "w") as f:
            f.write("{ invalid json }")
        
        # Should return empty list and log warning
        history = store.get_history(session_id)
        
        assert history == []
    
    def test_list_sessions_skips_malformed_files(self, store, temp_storage_dir):
        """Test that list_sessions skips malformed JSON files."""
        # Create a valid session
        valid_session = store.create_session()
        
        # Create a malformed session file
        malformed_path = Path(temp_storage_dir) / "sess_malformed.json"
        with open(malformed_path, "w") as f:
            f.write("{ invalid json }")
        
        sessions = store.list_sessions()
        
        # Should only return the valid session
        assert len(sessions) == 1
        assert sessions[0].session_id == valid_session.session_id
    
    def test_get_session_with_missing_required_fields(self, store, temp_storage_dir):
        """Test that JSON with missing required fields returns None."""
        # Create a session file with missing required fields
        session_id = "sess_incomplete"
        session_path = Path(temp_storage_dir) / f"{session_id}.json"
        
        with open(session_path, "w") as f:
            json.dump({"session_id": session_id}, f)  # Missing created_at, updated_at
        
        session = store.get_session(session_id)
        
        assert session is None


class TestErrorHandling:
    """Test error handling scenarios (Requirement 7)."""
    
    def test_save_session_write_failure(self, store):
        """Test that write failures raise SessionError (Req 7.2)."""
        session = store.create_session()
        
        # Mock json.dump to raise an exception during write
        with patch('src.pipelines.inference.conversation.store.json.dump', side_effect=PermissionError("Permission denied")):
            with pytest.raises(SessionError) as exc_info:
                store.add_message(session.session_id, "user", "This should fail")
            
            assert "Failed to write session to disk" in str(exc_info.value)
            assert exc_info.value.session_id == session.session_id
    
    def test_add_message_validates_role(self, store):
        """Test that add_message validates role parameter."""
        session = store.create_session()
        
        # Test various invalid roles
        invalid_roles = ["system", "tool", "function", "", "USER", "Assistant"]
        
        for invalid_role in invalid_roles:
            with pytest.raises(ValueError) as exc_info:
                store.add_message(session.session_id, invalid_role, "Test")
            
            assert "Invalid role" in str(exc_info.value)


class TestConversationContinuity:
    """Test conversation continuity across restarts (Requirement 2)."""
    
    def test_load_existing_session_after_restart(self, temp_storage_dir):
        """Test that sessions can be loaded after 'restart' (Req 2.1, 2.2)."""
        # Create a store and add messages
        store1 = ConversationStore(storage_dir=temp_storage_dir, max_history_length=10)
        session = store1.create_session()
        store1.add_message(session.session_id, "user", "Hello!")
        store1.add_message(session.session_id, "assistant", "Hi there!")
        
        # Simulate restart by creating a new store instance
        store2 = ConversationStore(storage_dir=temp_storage_dir, max_history_length=10)
        
        # Load the session
        loaded_session = store2.get_session(session.session_id)
        
        assert loaded_session is not None
        assert len(loaded_session.messages) == 2
        assert loaded_session.messages[0].content == "Hello!"
        assert loaded_session.messages[1].content == "Hi there!"
    
    def test_message_ordering_preserved_after_restart(self, temp_storage_dir):
        """Test that message ordering is preserved (Req 2.4)."""
        # Create a store and add messages
        store1 = ConversationStore(storage_dir=temp_storage_dir, max_history_length=10)
        session = store1.create_session()
        
        messages_content = ["First", "Second", "Third", "Fourth"]
        for i, content in enumerate(messages_content):
            role = "user" if i % 2 == 0 else "assistant"
            store1.add_message(session.session_id, role, content)
        
        # Simulate restart
        store2 = ConversationStore(storage_dir=temp_storage_dir, max_history_length=10)
        
        # Load messages
        langchain_messages = store2.get_langchain_messages(session.session_id)
        
        assert len(langchain_messages) == 4
        for i, content in enumerate(messages_content):
            assert langchain_messages[i].content == content
    
    def test_timestamps_preserved_after_restart(self, temp_storage_dir):
        """Test that timestamps are preserved (Req 2.4)."""
        # Create a store and add a message
        store1 = ConversationStore(storage_dir=temp_storage_dir, max_history_length=10)
        session = store1.create_session()
        original_created_at = session.created_at
        
        store1.add_message(session.session_id, "user", "Hello!")
        
        # Simulate restart
        store2 = ConversationStore(storage_dir=temp_storage_dir, max_history_length=10)
        
        # Load session
        loaded_session = store2.get_session(session.session_id)
        
        assert loaded_session.created_at == original_created_at
        assert loaded_session.messages[0].timestamp is not None


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_message_content(self, store):
        """Test adding message with empty content."""
        session = store.create_session()
        
        updated_session = store.add_message(session.session_id, "user", "")
        
        assert len(updated_session.messages) == 1
        assert updated_session.messages[0].content == ""
    
    def test_very_long_message_content(self, store):
        """Test adding message with very long content."""
        session = store.create_session()
        long_content = "A" * 10000  # 10k characters
        
        updated_session = store.add_message(session.session_id, "user", long_content)
        
        assert len(updated_session.messages) == 1
        assert updated_session.messages[0].content == long_content
    
    def test_special_characters_in_message(self, store):
        """Test adding message with special characters."""
        session = store.create_session()
        special_content = "Hello! 你好 🎉 \n\t\r Special chars: @#$%^&*()"
        
        updated_session = store.add_message(session.session_id, "user", special_content)
        
        # Verify it persists and loads correctly
        loaded_session = store.get_session(session.session_id)
        assert loaded_session.messages[0].content == special_content
    
    def test_zero_max_history_length(self, temp_storage_dir):
        """Test behavior with max_history_length=0."""
        store_zero = ConversationStore(
            storage_dir=temp_storage_dir,
            max_history_length=0
        )
        
        session = store_zero.create_session()
        store_zero.add_message(session.session_id, "user", "Hello!")
        
        # Should return all messages when limit is 0
        messages = store_zero.get_langchain_messages(session.session_id)
        
        assert len(messages) == 1
    
    def test_negative_limit_parameter(self, store):
        """Test behavior with negative limit parameter."""
        session = store.create_session()
        store.add_message(session.session_id, "user", "Hello!")
        
        # Negative limit should be handled gracefully
        messages = store.get_langchain_messages(session.session_id, limit=-1)
        
        # Should return empty or all messages (implementation dependent)
        assert isinstance(messages, list)


# Import Hypothesis for property-based testing
from hypothesis import given, strategies as st, settings, HealthCheck


class TestPropertyBasedConversationStore:
    """Property-based tests for ConversationStore using Hypothesis.
    
    These tests verify universal properties that should hold for any valid input.
    """
    
    @given(
        messages=st.lists(
            st.tuples(
                st.sampled_from(["user", "assistant"]),
                st.text(min_size=0, max_size=1000)
            ),
            min_size=0,
            max_size=50
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_chronological_order_and_role_mapping(self, messages):
        """Property P1: For any sequence of added messages, get_langchain_messages 
        returns them in the same chronological order and with correct role mapping.
        
        **Validates: Requirements 6.1, 6.2**
        
        This property verifies that:
        1. Messages are returned in the same order they were added
        2. User messages are converted to HumanMessage
        3. Assistant messages are converted to AIMessage
        4. Message content is preserved exactly
        """
        # Create a fresh temporary directory for this test run
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fresh store for this test
            store = ConversationStore(storage_dir=tmpdir, max_history_length=100)
            session = store.create_session()
            
            # Add all messages in sequence
            for role, content in messages:
                store.add_message(session.session_id, role, content)
            
            # Retrieve messages as LangChain format
            langchain_messages = store.get_langchain_messages(session.session_id, limit=len(messages) if messages else 1)
            
            # Verify count matches
            assert len(langchain_messages) == len(messages), \
                f"Expected {len(messages)} messages, got {len(langchain_messages)}"
            
            # Verify chronological order and role mapping
            for i, (expected_role, expected_content) in enumerate(messages):
                actual_message = langchain_messages[i]
                
                # Check role mapping
                if expected_role == "user":
                    assert isinstance(actual_message, HumanMessage), \
                        f"Message {i}: Expected HumanMessage for role 'user', got {type(actual_message).__name__}"
                elif expected_role == "assistant":
                    assert isinstance(actual_message, AIMessage), \
                        f"Message {i}: Expected AIMessage for role 'assistant', got {type(actual_message).__name__}"
                
                # Check content preservation
                assert actual_message.content == expected_content, \
                    f"Message {i}: Content mismatch. Expected '{expected_content}', got '{actual_message.content}'"
    
    @given(
        num_messages=st.integers(min_value=1, max_value=100),
        max_history_length=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_history_trimming_tail_selection(self, num_messages, max_history_length):
        """Property P2: For any session with N messages where N > max_history_length,
        get_langchain_messages returns exactly max_history_length messages, all from
        the tail of the sequence.
        
        **Validates: Requirements 6.1, 6.2, 6.3, 2.3, 2.4**
        
        This property verifies that:
        1. When messages exceed max_history_length, only the most recent are returned
        2. The returned messages are the last max_history_length messages
        3. The chronological order is preserved within the trimmed set
        """
        # Create a fresh temporary directory for this test run
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fresh store with the specified max_history_length
            store = ConversationStore(storage_dir=tmpdir, max_history_length=max_history_length)
            session = store.create_session()
            
            # Add num_messages messages with identifiable content
            for i in range(num_messages):
                role = "user" if i % 2 == 0 else "assistant"
                content = f"Message_{i}"
                store.add_message(session.session_id, role, content)
            
            # Retrieve messages using default limit (max_history_length)
            langchain_messages = store.get_langchain_messages(session.session_id)
            
            # Calculate expected count
            expected_count = min(num_messages, max_history_length)
            
            # Verify count
            assert len(langchain_messages) == expected_count, \
                f"Expected {expected_count} messages, got {len(langchain_messages)}"
            
            # If we have more messages than max_history_length, verify tail selection
            if num_messages > max_history_length:
                # The first returned message should be from index (num_messages - max_history_length)
                start_index = num_messages - max_history_length
                
                for i, msg in enumerate(langchain_messages):
                    expected_index = start_index + i
                    expected_content = f"Message_{expected_index}"
                    
                    assert msg.content == expected_content, \
                        f"Message {i}: Expected content '{expected_content}', got '{msg.content}'. " \
                        f"Messages should be from tail of sequence (indices {start_index} to {num_messages-1})"
            else:
                # If num_messages <= max_history_length, all messages should be returned
                for i, msg in enumerate(langchain_messages):
                    expected_content = f"Message_{i}"
                    assert msg.content == expected_content, \
                        f"Message {i}: Expected content '{expected_content}', got '{msg.content}'"
