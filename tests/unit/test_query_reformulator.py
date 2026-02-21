"""Unit tests for QueryReformulator.

This module tests the query reformulation logic that rewrites ambiguous follow-up
queries into self-contained queries with explicit product/topic references.

Tests cover:
- Reformulation with mocked LLM responses
- Passthrough when no history is provided
- Error handling when LLM fails
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.pipelines.inference.workflow.reformulator import QueryReformulator
from src.pipelines.inference.llm.client import LLMClient, LLMResponse
from src.pipelines.inference.exceptions import LLMError


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = Mock(spec=LLMClient)
    return client


@pytest.fixture
def reformulator(mock_llm_client):
    """Create a QueryReformulator instance with mocked LLM client."""
    return QueryReformulator(llm_client=mock_llm_client)


class TestQueryReformulatorBasic:
    """Test basic reformulation functionality."""
    
    def test_reformulate_with_mocked_llm_response(self, reformulator, mock_llm_client):
        """Test successful reformulation with mocked LLM.
        
        **Validates: Requirement 4.2, 5.1, 5.2**
        
        When the LLM returns a reformulated query, the reformulator should
        return that reformulated query.
        """
        # Setup
        query = "Tell me more about it"
        history = [
            HumanMessage(content="What phones do you have?"),
            AIMessage(content="We have the Samsung Galaxy S24 for $799.")
        ]
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "Tell me more about the Samsung Galaxy S24"
        mock_llm_client.invoke.return_value = mock_response
        
        # Execute
        result = reformulator.reformulate(query, history)
        
        # Verify
        assert result == "Tell me more about the Samsung Galaxy S24"
        mock_llm_client.invoke.assert_called_once()
        
        # Verify the messages passed to LLM include system prompt and history
        call_args = mock_llm_client.invoke.call_args
        messages = call_args[0][0]
        assert len(messages) >= 2  # System message + history/query message
        assert any("reformulation" in msg.content.lower() for msg in messages if hasattr(msg, 'content'))
    
    def test_reformulate_extracts_product_name_from_history(self, reformulator, mock_llm_client):
        """Test that reformulation extracts specific product names from history."""
        query = "What about the camera?"
        history = [
            HumanMessage(content="Show me flagship phones"),
            AIMessage(content="The iPhone 15 Pro has excellent features.")
        ]
        
        mock_response = Mock()
        mock_response.content = "What about the camera on the iPhone 15 Pro?"
        mock_llm_client.invoke.return_value = mock_response
        
        result = reformulator.reformulate(query, history)
        
        assert "iPhone 15 Pro" in result
        assert result == "What about the camera on the iPhone 15 Pro?"
    
    def test_reformulate_replaces_pronouns_with_product_names(self, reformulator, mock_llm_client):
        """Test that pronouns are replaced with explicit product references."""
        query = "How does it compare to other options?"
        history = [
            HumanMessage(content="Tell me about gaming laptops"),
            AIMessage(content="The ASUS ROG Strix is a powerful gaming laptop.")
        ]
        
        mock_response = Mock()
        mock_response.content = "How does the ASUS ROG Strix compare to other gaming laptops?"
        mock_llm_client.invoke.return_value = mock_response
        
        result = reformulator.reformulate(query, history)
        
        assert "ASUS ROG Strix" in result
        assert "it" not in result.lower()
    
    def test_reformulate_with_multiple_products_in_history(self, reformulator, mock_llm_client):
        """Test reformulation when history mentions multiple products."""
        query = "Compare those two"
        history = [
            HumanMessage(content="What are good budget phones?"),
            AIMessage(content="The Pixel 7a and Samsung Galaxy A54 are excellent budget options.")
        ]
        
        mock_response = Mock()
        mock_response.content = "Compare the Pixel 7a and Samsung Galaxy A54"
        mock_llm_client.invoke.return_value = mock_response
        
        result = reformulator.reformulate(query, history)
        
        assert "Pixel 7a" in result
        assert "Samsung Galaxy A54" in result


class TestQueryReformulatorPassthrough:
    """Test passthrough behavior when reformulation is not needed."""
    
    def test_passthrough_when_no_history(self, reformulator, mock_llm_client):
        """Test that query is returned unchanged when no history is provided.
        
        **Validates: Requirement 4.2, 5.1, 5.2**
        
        When there's no conversation history, the reformulator should return
        the original query without calling the LLM.
        """
        query = "What phones do you have under $500?"
        
        # No history provided
        result = reformulator.reformulate(query, history=None)
        
        assert result == query
        mock_llm_client.invoke.assert_not_called()
    
    def test_passthrough_when_empty_history(self, reformulator, mock_llm_client):
        """Test that query is returned unchanged when history is empty list."""
        query = "Show me laptops"
        history = []
        
        result = reformulator.reformulate(query, history)
        
        assert result == query
        mock_llm_client.invoke.assert_not_called()
    
    def test_passthrough_when_llm_returns_empty_string(self, reformulator, mock_llm_client):
        """Test fallback to original query when LLM returns empty response."""
        query = "Tell me more"
        history = [
            HumanMessage(content="Show me phones"),
            AIMessage(content="We have many phones.")
        ]
        
        # Mock LLM returning empty string
        mock_response = Mock()
        mock_response.content = ""
        mock_llm_client.invoke.return_value = mock_response
        
        result = reformulator.reformulate(query, history)
        
        assert result == query
    
    def test_passthrough_when_llm_returns_whitespace_only(self, reformulator, mock_llm_client):
        """Test fallback when LLM returns only whitespace."""
        query = "What about battery life?"
        history = [
            HumanMessage(content="Show me phones"),
            AIMessage(content="The iPhone 15 is available.")
        ]
        
        # Mock LLM returning whitespace
        mock_response = Mock()
        mock_response.content = "   \n\t  "
        mock_llm_client.invoke.return_value = mock_response
        
        result = reformulator.reformulate(query, history)
        
        assert result == query


class TestQueryReformulatorErrorHandling:
    """Test error handling when LLM fails."""
    
    def test_error_handling_when_llm_raises_exception(self, reformulator, mock_llm_client):
        """Test that original query is returned when LLM raises an exception.
        
        **Validates: Requirement 4.2, 5.1, 5.2**
        
        When the LLM API call fails, the reformulator should gracefully fall back
        to the original query instead of propagating the error.
        """
        query = "Tell me more about it"
        history = [
            HumanMessage(content="What phones do you have?"),
            AIMessage(content="We have the Samsung Galaxy S24.")
        ]
        
        # Mock LLM raising an exception
        mock_llm_client.invoke.side_effect = LLMError(
            "API call failed",
            provider="openai",
            model="gpt-4"
        )
        
        # Should not raise exception, should return original query
        result = reformulator.reformulate(query, history)
        
        assert result == query
    
    def test_error_handling_with_generic_exception(self, reformulator, mock_llm_client):
        """Test error handling with generic exceptions."""
        query = "What about the price?"
        history = [
            HumanMessage(content="Show me laptops"),
            AIMessage(content="The MacBook Pro is available.")
        ]
        
        # Mock generic exception
        mock_llm_client.invoke.side_effect = Exception("Network error")
        
        result = reformulator.reformulate(query, history)
        
        assert result == query
    
    def test_error_handling_with_timeout(self, reformulator, mock_llm_client):
        """Test error handling when LLM times out."""
        query = "How does it compare?"
        history = [
            HumanMessage(content="Show me phones"),
            AIMessage(content="The Pixel 8 is excellent.")
        ]
        
        # Mock timeout exception
        mock_llm_client.invoke.side_effect = TimeoutError("Request timed out")
        
        result = reformulator.reformulate(query, history)
        
        assert result == query
    
    def test_error_handling_preserves_original_query_exactly(self, reformulator, mock_llm_client):
        """Test that error handling returns the exact original query."""
        query = "Tell me MORE about THAT one!!!"
        history = [
            HumanMessage(content="Show me phones"),
            AIMessage(content="The Samsung Galaxy S24 is available.")
        ]
        
        mock_llm_client.invoke.side_effect = Exception("Error")
        
        result = reformulator.reformulate(query, history)
        
        # Should preserve exact formatting and punctuation
        assert result == query
        assert result == "Tell me MORE about THAT one!!!"


class TestQueryReformulatorAsync:
    """Test async reformulation functionality."""
    
    @pytest.mark.asyncio
    async def test_areformulate_with_mocked_llm_response(self, reformulator, mock_llm_client):
        """Test async reformulation with mocked LLM.
        
        **Validates: Requirement 4.2, 5.1, 5.2**
        """
        query = "Tell me more about it"
        history = [
            HumanMessage(content="What phones do you have?"),
            AIMessage(content="We have the Samsung Galaxy S24.")
        ]
        
        # Mock async LLM response
        mock_response = Mock()
        mock_response.content = "Tell me more about the Samsung Galaxy S24"
        mock_llm_client.ainvoke = AsyncMock(return_value=mock_response)
        
        result = await reformulator.areformulate(query, history)
        
        assert result == "Tell me more about the Samsung Galaxy S24"
        mock_llm_client.ainvoke.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_areformulate_passthrough_when_no_history(self, reformulator, mock_llm_client):
        """Test async passthrough when no history is provided."""
        query = "What phones do you have?"
        
        mock_llm_client.ainvoke = AsyncMock()
        
        result = await reformulator.areformulate(query, history=None)
        
        assert result == query
        mock_llm_client.ainvoke.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_areformulate_error_handling(self, reformulator, mock_llm_client):
        """Test async error handling when LLM fails."""
        query = "Tell me more"
        history = [
            HumanMessage(content="Show me phones"),
            AIMessage(content="The iPhone 15 is available.")
        ]
        
        # Mock async LLM raising exception
        mock_llm_client.ainvoke = AsyncMock(side_effect=LLMError(
            "API call failed",
            provider="openai",
            model="gpt-4"
        ))
        
        result = await reformulator.areformulate(query, history)
        
        assert result == query
    
    @pytest.mark.asyncio
    async def test_areformulate_with_empty_llm_response(self, reformulator, mock_llm_client):
        """Test async fallback when LLM returns empty response."""
        query = "What about the features?"
        history = [
            HumanMessage(content="Show me laptops"),
            AIMessage(content="The Dell XPS is available.")
        ]
        
        mock_response = Mock()
        mock_response.content = ""
        mock_llm_client.ainvoke = AsyncMock(return_value=mock_response)
        
        result = await reformulator.areformulate(query, history)
        
        assert result == query


class TestQueryReformulatorMessageBuilding:
    """Test the internal message building logic."""
    
    def test_build_reformulation_messages_includes_system_prompt(self, reformulator):
        """Test that reformulation messages include the system prompt."""
        query = "Tell me more"
        history = [
            HumanMessage(content="Show me phones"),
            AIMessage(content="The Samsung Galaxy S24 is available.")
        ]
        
        messages = reformulator._build_reformulation_messages(query, history)
        
        # Should have at least 2 messages: system prompt + user message
        assert len(messages) >= 2
        
        # First message should be system message
        assert isinstance(messages[0], SystemMessage)
        assert "reformulation" in messages[0].content.lower()
    
    def test_build_reformulation_messages_includes_history(self, reformulator):
        """Test that reformulation messages include conversation history."""
        query = "What about the camera?"
        history = [
            HumanMessage(content="Show me phones"),
            AIMessage(content="The iPhone 15 Pro is available.")
        ]
        
        messages = reformulator._build_reformulation_messages(query, history)
        
        # Last message should contain the history and query
        last_message = messages[-1]
        assert isinstance(last_message, HumanMessage)
        assert "Show me phones" in last_message.content
        assert "iPhone 15 Pro" in last_message.content
        assert query in last_message.content
    
    def test_build_reformulation_messages_limits_history_length(self, reformulator):
        """Test that only recent history is included to save tokens."""
        query = "Tell me more"
        
        # Create long history (more than 6 messages)
        history = []
        for i in range(10):
            history.append(HumanMessage(content=f"Question {i}"))
            history.append(AIMessage(content=f"Answer {i}"))
        
        messages = reformulator._build_reformulation_messages(query, history)
        
        # Should limit to last 6 messages (3 turns)
        last_message = messages[-1]
        
        # Should NOT include early messages
        assert "Question 0" not in last_message.content
        assert "Answer 0" not in last_message.content
        
        # Should include recent messages
        assert "Question 9" in last_message.content
        assert "Answer 9" in last_message.content
    
    def test_build_reformulation_messages_formats_history_correctly(self, reformulator):
        """Test that history is formatted with User/Assistant labels."""
        query = "What about battery?"
        history = [
            HumanMessage(content="Show me phones"),
            AIMessage(content="The Samsung Galaxy S24 is available.")
        ]
        
        messages = reformulator._build_reformulation_messages(query, history)
        
        last_message = messages[-1]
        assert "User:" in last_message.content
        assert "Assistant:" in last_message.content
    
    def test_build_reformulation_messages_with_short_history(self, reformulator):
        """Test message building with history shorter than limit."""
        query = "Tell me more"
        history = [
            HumanMessage(content="Show me phones"),
            AIMessage(content="The iPhone 15 is available.")
        ]
        
        messages = reformulator._build_reformulation_messages(query, history)
        
        # Should include all history when it's short
        last_message = messages[-1]
        assert "Show me phones" in last_message.content
        assert "iPhone 15" in last_message.content


class TestQueryReformulatorEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_reformulate_with_empty_query(self, reformulator, mock_llm_client):
        """Test behavior with empty query string."""
        query = ""
        history = [
            HumanMessage(content="Show me phones"),
            AIMessage(content="The Samsung Galaxy S24 is available.")
        ]
        
        mock_response = Mock()
        mock_response.content = "Show me more about the Samsung Galaxy S24"
        mock_llm_client.invoke.return_value = mock_response
        
        result = reformulator.reformulate(query, history)
        
        # Should still call LLM and return reformulated query
        assert result == "Show me more about the Samsung Galaxy S24"
    
    def test_reformulate_with_very_long_query(self, reformulator, mock_llm_client):
        """Test reformulation with very long query."""
        query = "Tell me more about it " * 50  # Very long query
        history = [
            HumanMessage(content="Show me phones"),
            AIMessage(content="The iPhone 15 is available.")
        ]
        
        mock_response = Mock()
        mock_response.content = "Tell me more about the iPhone 15"
        mock_llm_client.invoke.return_value = mock_response
        
        result = reformulator.reformulate(query, history)
        
        assert result == "Tell me more about the iPhone 15"
    
    def test_reformulate_with_special_characters_in_query(self, reformulator, mock_llm_client):
        """Test reformulation with special characters."""
        query = "What's the price of that one? (the expensive one)"
        history = [
            HumanMessage(content="Show me phones"),
            AIMessage(content="The iPhone 15 Pro Max is available.")
        ]
        
        mock_response = Mock()
        mock_response.content = "What's the price of the iPhone 15 Pro Max?"
        mock_llm_client.invoke.return_value = mock_response
        
        result = reformulator.reformulate(query, history)
        
        assert "iPhone 15 Pro Max" in result
    
    def test_reformulate_with_unicode_characters(self, reformulator, mock_llm_client):
        """Test reformulation with unicode characters."""
        query = "Tell me more about it 📱"
        history = [
            HumanMessage(content="Show me phones"),
            AIMessage(content="The Samsung Galaxy S24 is available.")
        ]
        
        mock_response = Mock()
        mock_response.content = "Tell me more about the Samsung Galaxy S24 📱"
        mock_llm_client.invoke.return_value = mock_response
        
        result = reformulator.reformulate(query, history)
        
        assert "Samsung Galaxy S24" in result
        assert "📱" in result
    
    def test_reformulate_preserves_llm_response_formatting(self, reformulator, mock_llm_client):
        """Test that reformulator preserves LLM response formatting."""
        query = "tell me more"
        history = [
            HumanMessage(content="Show me phones"),
            AIMessage(content="The iPhone 15 is available.")
        ]
        
        # LLM returns with specific capitalization
        mock_response = Mock()
        mock_response.content = "Tell Me More About The iPhone 15"
        mock_llm_client.invoke.return_value = mock_response
        
        result = reformulator.reformulate(query, history)
        
        # Should preserve exact LLM response
        assert result == "Tell Me More About The iPhone 15"
