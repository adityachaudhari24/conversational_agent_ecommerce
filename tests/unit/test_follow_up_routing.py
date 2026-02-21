"""Unit tests for follow-up detection and routing in AgenticWorkflow.

This module tests the history-aware follow-up detection logic that enables
the router to identify context-dependent queries and route them to retrieval
even when they lack explicit product keywords.

Tests cover:
- Follow-up queries with pronouns after product responses
- Follow-up queries with follow-up phrases after product responses
- Non-follow-up general queries
- Queries with product keywords (should always route to retrieve)
"""

import pytest
from unittest.mock import Mock, MagicMock

from langchain_core.messages import HumanMessage, AIMessage

from src.pipelines.inference.workflow.agentic import AgenticWorkflow, CONTEXTUAL_REFERENCES, FOLLOW_UP_PHRASES
from src.pipelines.inference.config import WorkflowConfig
from src.pipelines.inference.llm.client import LLMClient
from src.pipelines.inference.generation.generator import ResponseGenerator


@pytest.fixture
def workflow_config():
    """Create a WorkflowConfig instance with default settings."""
    return WorkflowConfig(
        product_keywords=[
            "price", "review", "product", "recommend", "compare",
            "rating", "phone", "buy", "cost", "feature", "spec"
        ],
        tool_keywords=["compare"]
    )


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = Mock(spec=LLMClient)
    return client


@pytest.fixture
def mock_retrieval_pipeline():
    """Create a mock retrieval pipeline."""
    pipeline = Mock()
    pipeline.retrieve = Mock(return_value=Mock(formatted_context="Mock context"))
    return pipeline


@pytest.fixture
def mock_response_generator():
    """Create a mock response generator."""
    generator = Mock(spec=ResponseGenerator)
    generator.generate = Mock(return_value="Mock response")
    return generator


@pytest.fixture
def workflow(workflow_config, mock_llm_client, mock_retrieval_pipeline, mock_response_generator):
    """Create an AgenticWorkflow instance with mocked dependencies."""
    return AgenticWorkflow(
        config=workflow_config,
        llm_client=mock_llm_client,
        retrieval_pipeline=mock_retrieval_pipeline,
        response_generator=mock_response_generator
    )


class TestFollowUpDetection:
    """Test the _is_follow_up_query method for detecting context-dependent queries."""
    
    def test_follow_up_with_pronoun_after_product_response(self, workflow):
        """Test follow-up detection with pronouns after product-related response.
        
        **Validates: Requirement 4.1, 4.4**
        
        When a user asks "tell me more about it" after the assistant mentioned
        a phone, the query should be detected as a follow-up.
        """
        # History with product-related assistant response
        history = [
            HumanMessage(content="What phones do you have under $500?"),
            AIMessage(content="We have the Samsung Galaxy A54 for $449. It has great reviews and features a 6.4-inch display.")
        ]
        
        # Follow-up query with pronoun "it"
        query = "Tell me more about it"
        
        is_follow_up = workflow._is_follow_up_query(query, history)
        
        assert is_follow_up is True, "Query with pronoun 'it' after product response should be detected as follow-up"
    
    def test_follow_up_with_that_pronoun(self, workflow):
        """Test follow-up detection with 'that' pronoun."""
        history = [
            HumanMessage(content="Show me phones with good cameras"),
            AIMessage(content="The iPhone 15 Pro has an excellent camera system with 48MP main sensor.")
        ]
        
        query = "What's the price of that one?"
        
        is_follow_up = workflow._is_follow_up_query(query, history)
        
        assert is_follow_up is True
    
    def test_follow_up_with_these_pronoun(self, workflow):
        """Test follow-up detection with 'these' pronoun."""
        history = [
            HumanMessage(content="What are the best budget phones?"),
            AIMessage(content="The top budget phones are the Pixel 7a and Samsung Galaxy A54.")
        ]
        
        query = "Can you compare these two?"
        
        is_follow_up = workflow._is_follow_up_query(query, history)
        
        assert is_follow_up is True
    
    def test_follow_up_with_tell_me_more_phrase(self, workflow):
        """Test follow-up detection with 'tell me more' phrase.
        
        **Validates: Requirement 4.1, 4.4**
        """
        history = [
            HumanMessage(content="What's a good phone for photography?"),
            AIMessage(content="The Google Pixel 8 Pro is excellent for photography with its advanced computational photography features.")
        ]
        
        query = "Tell me more about the camera features"
        
        is_follow_up = workflow._is_follow_up_query(query, history)
        
        assert is_follow_up is True
    
    def test_follow_up_with_what_about_phrase(self, workflow):
        """Test follow-up detection with 'what about' phrase."""
        history = [
            HumanMessage(content="Show me flagship phones"),
            AIMessage(content="The Samsung Galaxy S24 Ultra is our top flagship phone.")
        ]
        
        query = "What about the battery life?"
        
        is_follow_up = workflow._is_follow_up_query(query, history)
        
        assert is_follow_up is True
    
    def test_follow_up_with_how_does_it_phrase(self, workflow):
        """Test follow-up detection with 'how does it' phrase."""
        history = [
            HumanMessage(content="Tell me about the iPhone 15"),
            AIMessage(content="The iPhone 15 features the A16 Bionic chip and improved camera system.")
        ]
        
        query = "How does it compare to the iPhone 14?"
        
        is_follow_up = workflow._is_follow_up_query(query, history)
        
        assert is_follow_up is True
    
    def test_follow_up_with_any_other_phrase(self, workflow):
        """Test follow-up detection with 'any other' phrase."""
        history = [
            HumanMessage(content="What phones have 5G?"),
            AIMessage(content="The Samsung Galaxy S24 phone has 5G connectivity.")
        ]
        
        query = "Any other options with 5G?"
        
        is_follow_up = workflow._is_follow_up_query(query, history)
        
        assert is_follow_up is True
    
    def test_non_follow_up_general_query_without_history(self, workflow):
        """Test that general queries without history are not detected as follow-ups.
        
        **Validates: Requirement 4.1, 4.4**
        """
        history = []
        
        query = "What is your return policy?"
        
        is_follow_up = workflow._is_follow_up_query(query, history)
        
        assert is_follow_up is False
    
    def test_non_follow_up_general_query_with_non_product_history(self, workflow):
        """Test that general queries after non-product responses are not follow-ups."""
        history = [
            HumanMessage(content="What are your store hours?"),
            AIMessage(content="We're open Monday through Friday, 9 AM to 6 PM.")
        ]
        
        query = "What about shipping?"
        
        is_follow_up = workflow._is_follow_up_query(query, history)
        
        assert is_follow_up is False, "Query should not be follow-up when history lacks product context"
    
    def test_non_follow_up_query_without_contextual_references(self, workflow):
        """Test that queries without contextual references are not follow-ups."""
        history = [
            HumanMessage(content="Show me phones"),
            AIMessage(content="We have many phones available including Samsung, Apple, and Google.")
        ]
        
        # Query has no pronouns or follow-up phrases
        query = "Do you offer warranty?"
        
        is_follow_up = workflow._is_follow_up_query(query, history)
        
        assert is_follow_up is False
    
    def test_contextual_reference_without_product_context(self, workflow):
        """Test that contextual references alone don't trigger follow-up without product context."""
        history = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hello! How can I help you today?")
        ]
        
        query = "Tell me more about that"
        
        is_follow_up = workflow._is_follow_up_query(query, history)
        
        assert is_follow_up is False, "Contextual reference without product context should not be follow-up"
    
    def test_follow_up_detection_checks_recent_messages(self, workflow):
        """Test that follow-up detection looks at recent assistant messages."""
        # Long history with product mention in recent message
        history = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
            HumanMessage(content="What do you sell?"),
            AIMessage(content="We sell electronics."),
            HumanMessage(content="Show me phones"),
            AIMessage(content="The Samsung Galaxy S24 is our best phone.")
        ]
        
        query = "Tell me more about it"
        
        is_follow_up = workflow._is_follow_up_query(query, history)
        
        assert is_follow_up is True
    
    def test_follow_up_with_multiple_product_keywords_in_history(self, workflow):
        """Test follow-up detection when history contains multiple product keywords."""
        history = [
            HumanMessage(content="What's the price of your phones?"),
            AIMessage(content="Our phones range from $200 to $1200. The Samsung Galaxy A54 costs $449 and has excellent reviews.")
        ]
        
        query = "What about the features of that one?"
        
        is_follow_up = workflow._is_follow_up_query(query, history)
        
        assert is_follow_up is True


class TestRouterNodeFollowUpRouting:
    """Test the _router_node method's follow-up routing logic."""
    
    def test_router_routes_follow_up_to_retrieve(self, workflow):
        """Test that router routes follow-up queries to retrieve.
        
        **Validates: Requirement 4.1, 4.4**
        """
        # State with history containing product context
        state = {
            "messages": [
                HumanMessage(content="What phones do you have?"),
                AIMessage(content="We have the Samsung Galaxy S24 phone for $799."),
                HumanMessage(content="Tell me more about it")  # Follow-up query
            ],
            "context": "",
            "route": "",
            "tool_result": "",
            "reformulated_query": ""
        }
        
        result = workflow._router_node(state)
        
        assert result["route"] == "retrieve_followup", "Follow-up query should route to retrieve_followup for reformulation"
    
    def test_router_routes_product_keyword_to_retrieve(self, workflow):
        """Test that queries with product keywords always route to retrieve.
        
        **Validates: Requirement 4.1, 4.4**
        
        Product keywords should take precedence over follow-up detection.
        """
        # State with no history but query has product keyword
        state = {
            "messages": [
                HumanMessage(content="What's the price of the iPhone 15?")
            ],
            "context": "",
            "route": "",
            "tool_result": "",
            "reformulated_query": ""
        }
        
        result = workflow._router_node(state)
        
        assert result["route"] == "retrieve", "Query with 'price' keyword should route to retrieve"
    
    def test_router_routes_product_keyword_regardless_of_history(self, workflow):
        """Test that product keywords route to retrieve even with non-product history."""
        state = {
            "messages": [
                HumanMessage(content="What are your store hours?"),
                AIMessage(content="We're open 9 AM to 6 PM."),
                HumanMessage(content="Show me phones under $500")  # Has product keyword
            ],
            "context": "",
            "route": "",
            "tool_result": "",
            "reformulated_query": ""
        }
        
        result = workflow._router_node(state)
        
        assert result["route"] == "retrieve"
    
    def test_router_routes_general_query_to_respond(self, workflow):
        """Test that general queries without keywords or follow-up context route to respond.
        
        **Validates: Requirement 4.1, 4.4**
        """
        state = {
            "messages": [
                HumanMessage(content="What is your return policy?")
            ],
            "context": "",
            "route": "",
            "tool_result": "",
            "reformulated_query": ""
        }
        
        result = workflow._router_node(state)
        
        assert result["route"] == "respond", "General query should route to respond"
    
    def test_router_routes_tool_keyword_to_tool(self, workflow):
        """Test that tool keywords route to tool node."""
        state = {
            "messages": [
                HumanMessage(content="Compare the iPhone 15 and Samsung S24")
            ],
            "context": "",
            "route": "",
            "tool_result": "",
            "reformulated_query": ""
        }
        
        result = workflow._router_node(state)
        
        assert result["route"] == "tool", "Query with 'compare' keyword should route to tool"
    
    def test_router_prioritizes_tool_over_product_keywords(self, workflow):
        """Test that tool keywords take precedence over product keywords."""
        state = {
            "messages": [
                HumanMessage(content="Compare phone prices")  # Has both 'compare' and 'phone'
            ],
            "context": "",
            "route": "",
            "tool_result": "",
            "reformulated_query": ""
        }
        
        result = workflow._router_node(state)
        
        assert result["route"] == "tool", "Tool keywords should take precedence"
    
    def test_router_handles_empty_messages(self, workflow):
        """Test router behavior with empty messages."""
        state = {
            "messages": [],
            "context": "",
            "route": "",
            "tool_result": "",
            "reformulated_query": ""
        }
        
        result = workflow._router_node(state)
        
        assert result["route"] == "respond", "Empty messages should default to respond"
    
    def test_router_ignores_system_messages_in_history(self, workflow):
        """Test that router correctly excludes system messages from history analysis."""
        from langchain_core.messages import SystemMessage
        
        state = {
            "messages": [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="Show me phones"),
                AIMessage(content="We have the Samsung Galaxy S24 phone with great features."),
                HumanMessage(content="Tell me more about it")
            ],
            "context": "",
            "route": "",
            "tool_result": "",
            "reformulated_query": ""
        }
        
        result = workflow._router_node(state)
        
        assert result["route"] == "retrieve_followup", "Should detect follow-up even with system messages"


class TestFollowUpDetectionEdgeCases:
    """Test edge cases and boundary conditions for follow-up detection."""
    
    def test_case_insensitive_keyword_matching(self, workflow):
        """Test that keyword matching is case-insensitive."""
        history = [
            HumanMessage(content="What PHONES do you have?"),
            AIMessage(content="We have the Samsung Galaxy S24 phone with great features.")
        ]
        
        query = "TELL ME MORE about IT"
        
        is_follow_up = workflow._is_follow_up_query(query, history)
        
        assert is_follow_up is True
    
    def test_partial_word_matching(self, workflow):
        """Test that keywords match within words."""
        history = [
            HumanMessage(content="smartphone recommendations"),
            AIMessage(content="The iPhone 15 is a great smartphone.")
        ]
        
        # "phone" is part of "smartphone"
        query = "what about that one?"
        
        is_follow_up = workflow._is_follow_up_query(query, history)
        
        assert is_follow_up is True
    
    def test_empty_query_string(self, workflow):
        """Test behavior with empty query string."""
        history = [
            HumanMessage(content="Show me phones"),
            AIMessage(content="We have many phones.")
        ]
        
        query = ""
        
        is_follow_up = workflow._is_follow_up_query(query, history)
        
        assert is_follow_up is False
    
    def test_whitespace_only_query(self, workflow):
        """Test behavior with whitespace-only query."""
        history = [
            HumanMessage(content="Show me phones"),
            AIMessage(content="We have the Samsung Galaxy S24.")
        ]
        
        query = "   \t\n  "
        
        is_follow_up = workflow._is_follow_up_query(query, history)
        
        assert is_follow_up is False
    
    def test_very_long_history(self, workflow):
        """Test follow-up detection with very long conversation history."""
        # Create a long history
        history = []
        for i in range(50):
            history.append(HumanMessage(content=f"Question {i}"))
            history.append(AIMessage(content=f"Answer {i}"))
        
        # Add recent product context
        history.append(HumanMessage(content="Show me phones"))
        history.append(AIMessage(content="The Samsung Galaxy S24 phone is available."))
        
        query = "Tell me more about it"
        
        is_follow_up = workflow._is_follow_up_query(query, history)
        
        assert is_follow_up is True
    
    def test_multiple_contextual_references_in_query(self, workflow):
        """Test query with multiple contextual references."""
        history = [
            HumanMessage(content="What phones do you recommend?"),
            AIMessage(content="I recommend the iPhone 15 and Samsung Galaxy S24.")
        ]
        
        query = "Can you tell me more about those two and compare them?"
        
        is_follow_up = workflow._is_follow_up_query(query, history)
        
        assert is_follow_up is True
    
    def test_follow_up_phrase_at_end_of_query(self, workflow):
        """Test follow-up phrase at the end of the query."""
        history = [
            HumanMessage(content="Show me phones"),
            AIMessage(content="The Samsung Galaxy S24 is our best phone.")
        ]
        
        query = "The camera quality, tell me more"
        
        is_follow_up = workflow._is_follow_up_query(query, history)
        
        assert is_follow_up is True
    
    def test_contextual_reference_in_middle_of_sentence(self, workflow):
        """Test contextual reference embedded in a longer sentence."""
        history = [
            HumanMessage(content="What's a good budget phone?"),
            AIMessage(content="The Pixel 7a is an excellent budget phone at $449.")
        ]
        
        query = "I'm interested in learning more about that phone's camera capabilities"
        
        is_follow_up = workflow._is_follow_up_query(query, history)
        
        assert is_follow_up is True


class TestFollowUpDetectionWithMultipleAssistantMessages:
    """Test follow-up detection when there are multiple assistant messages in history."""
    
    def test_product_context_in_most_recent_message(self, workflow):
        """Test detection when product context is in the most recent assistant message."""
        history = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi! How can I help?"),
            HumanMessage(content="Show me phones"),
            AIMessage(content="The Samsung Galaxy S24 phone is available for $799.")
        ]
        
        query = "Tell me more about it"
        
        is_follow_up = workflow._is_follow_up_query(query, history)
        
        assert is_follow_up is True
    
    def test_product_context_in_earlier_message(self, workflow):
        """Test detection when product context is in an earlier assistant message."""
        history = [
            HumanMessage(content="Show me phones"),
            AIMessage(content="The Samsung Galaxy S24 phone is available."),
            HumanMessage(content="What about warranty?"),
            AIMessage(content="We offer a 1-year warranty on all products.")
        ]
        
        query = "Tell me more about the phone"
        
        # Should still detect as follow-up because "phone" is a product keyword
        # and there's product context in recent history
        is_follow_up = workflow._is_follow_up_query(query, history)
        
        assert is_follow_up is True
    
    def test_no_product_context_in_recent_messages(self, workflow):
        """Test that follow-up is not detected when product context is too old."""
        history = [
            HumanMessage(content="Show me phones"),
            AIMessage(content="The Samsung Galaxy S24 is available."),
            HumanMessage(content="What's your return policy?"),
            AIMessage(content="30-day return policy."),
            HumanMessage(content="What about shipping?"),
            AIMessage(content="Free shipping on orders over $50."),
            HumanMessage(content="Do you have stores?"),
            AIMessage(content="Yes, we have 50 stores nationwide.")
        ]
        
        # Product context is more than 3 assistant messages ago
        query = "Tell me more about it"
        
        is_follow_up = workflow._is_follow_up_query(query, history)
        
        # Should not be follow-up because product context is too old
        # (implementation checks last 3 assistant messages)
        assert is_follow_up is False
