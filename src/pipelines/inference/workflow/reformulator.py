"""Query reformulation for follow-up queries.

This module provides the QueryReformulator class that uses the LLM to rewrite
ambiguous follow-up queries into self-contained queries with explicit product/topic
references extracted from conversation history.
"""

from typing import List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from ..llm.client import LLMClient


class QueryReformulator:
    """Reformulates follow-up queries using conversation history.
    
    Uses the LLM to rewrite ambiguous follow-up queries (e.g., "tell me more about
    that one") into self-contained queries with explicit product/topic references
    (e.g., "tell me more about the Samsung Galaxy S24 Ultra").
    
    This enables accurate vector DB retrieval for context-dependent follow-up questions.
    
    Attributes:
        llm_client: LLM client for making reformulation API calls
        REFORMULATION_PROMPT: System prompt for query reformulation
    """
    
    REFORMULATION_PROMPT = """You are a query reformulation assistant for an e-commerce system.

Your task is to rewrite follow-up queries into standalone queries by extracting explicit 
product names, brands, or topics from the conversation history.

Instructions:
- Identify what "it", "that", "this", "the one", etc. refer to in the conversation history
- Replace pronouns and vague references with specific product names or topics
- Keep the user's intent and question structure intact
- Make the query self-contained so it can be understood without conversation context
- If the query is already standalone, return it unchanged

Examples:

Conversation:
User: What phones do you have under $500?
Assistant: Here are some phones under $500: Samsung Galaxy A54, Google Pixel 7a...

Follow-up: "Tell me more about the Samsung one"
Reformulated: "Tell me more about the Samsung Galaxy A54"

Conversation:
User: Show me gaming laptops
Assistant: Here are some gaming laptops: ASUS ROG Strix, MSI Raider...

Follow-up: "How does it compare to other options?"
Reformulated: "How does the ASUS ROG Strix compare to other gaming laptops?"

Conversation:
User: I'm looking for wireless headphones
Assistant: I recommend the Sony WH-1000XM5 and Bose QuietComfort 45...

Follow-up: "What about battery life?"
Reformulated: "What about the battery life of the Sony WH-1000XM5 and Bose QuietComfort 45?"

Now reformulate the following query based on the conversation history provided."""
    
    def __init__(self, llm_client: LLMClient):
        """Initialize the query reformulator.
        
        Args:
            llm_client: Initialized LLM client for API calls
        """
        self.llm_client = llm_client
    
    def reformulate(self, query: str, history: Optional[List[BaseMessage]] = None) -> str:
        """Rewrite a follow-up query as a standalone query.
        
        Uses the LLM to analyze the conversation history and rewrite the query
        with explicit references to products or topics mentioned in prior turns.
        
        Args:
            query: The follow-up query (e.g., "tell me more about that one")
            history: Recent conversation history (list of BaseMessage objects)
            
        Returns:
            Reformulated query with explicit references
            (e.g., "tell me more about the Samsung Galaxy S24 Ultra")
            
        Raises:
            LLMError: If LLM API call fails
        """
        # If no history provided, return query unchanged
        if not history:
            return query
        
        # Build messages for reformulation
        messages = self._build_reformulation_messages(query, history)
        
        try:
            # Call LLM to reformulate
            response = self.llm_client.invoke(messages)
            reformulated = response.content.strip()
            
            # If reformulation failed or returned empty, use original query
            if not reformulated:
                return query
            
            return reformulated
            
        except Exception as e:
            # On error, fall back to original query
            # Log the error in production
            return query
    
    async def areformulate(self, query: str, history: Optional[List[BaseMessage]] = None) -> str:
        """Async version of reformulate.
        
        Uses the LLM to analyze the conversation history and rewrite the query
        with explicit references to products or topics mentioned in prior turns.
        
        Args:
            query: The follow-up query (e.g., "tell me more about that one")
            history: Recent conversation history (list of BaseMessage objects)
            
        Returns:
            Reformulated query with explicit references
            (e.g., "tell me more about the Samsung Galaxy S24 Ultra")
            
        Raises:
            LLMError: If LLM API call fails
        """
        # If no history provided, return query unchanged
        if not history:
            return query
        
        # Build messages for reformulation
        messages = self._build_reformulation_messages(query, history)
        
        try:
            # Call LLM to reformulate asynchronously
            response = await self.llm_client.ainvoke(messages)
            reformulated = response.content.strip()
            
            # If reformulation failed or returned empty, use original query
            if not reformulated:
                return query
            
            return reformulated
            
        except Exception as e:
            # On error, fall back to original query
            # Log the error in production
            return query
    
    def _build_reformulation_messages(
        self,
        query: str,
        history: List[BaseMessage]
    ) -> List[BaseMessage]:
        """Build message list for reformulation LLM call.
        
        Constructs a prompt that includes:
        1. System prompt with reformulation instructions
        2. Conversation history for context
        3. The follow-up query to reformulate
        
        Args:
            query: The follow-up query to reformulate
            history: Conversation history
            
        Returns:
            List of BaseMessage objects for LLM
        """
        messages: List[BaseMessage] = []
        
        # Add system prompt
        messages.append(SystemMessage(content=self.REFORMULATION_PROMPT))
        
        # Add conversation history (limit to recent messages to save tokens)
        # Take the last 6 messages (3 turns) for context
        recent_history = history[-6:] if len(history) > 6 else history
        
        # Format history as a single message for clarity
        history_text = "Conversation History:\n"
        for msg in recent_history:
            if isinstance(msg, HumanMessage):
                history_text += f"User: {msg.content}\n"
            elif hasattr(msg, 'content'):
                # AIMessage or other message types
                history_text += f"Assistant: {msg.content}\n"
        
        history_text += f"\nFollow-up Query: {query}\n\nReformulated Query:"
        
        messages.append(HumanMessage(content=history_text))
        
        return messages
