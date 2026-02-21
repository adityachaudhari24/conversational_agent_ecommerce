"""Query reformulation for improved retrieval.

This module provides the QueryReformulator class that uses the LLM to either:
1. Rewrite follow-up queries by resolving references from conversation history
2. Optimize standalone queries for better vector search retrieval
"""

from typing import List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from ..llm.client import LLMClient


class QueryReformulator:
    """Reformulates queries for optimal vector search retrieval.
    
    Operates in two modes:
    - Follow-up mode: resolves pronouns/references using conversation history
    - Standalone mode: optimizes the query for better vector DB retrieval
    
    Attributes:
        llm_client: LLM client for making reformulation API calls
    """
    
    FOLLOWUP_PROMPT = """You are a query reformulation assistant for an Apple iPhone product database.

Your task is to rewrite follow-up queries into standalone queries by extracting explicit 
iPhone model names or topics from the conversation history.

Instructions:
- Identify what "it", "that", "this", "the one", etc. refer to in the conversation history
- Replace pronouns and vague references with specific iPhone model names or features
- Keep the user's intent and question structure intact
- Make the query self-contained so it can be understood without conversation context
- If the query is already standalone, return it unchanged
- Return ONLY the reformulated query, nothing else

Examples:

Conversation:
User: What iPhones do you have under $500?
Assistant: We have the iPhone 12 for $448 and the iPhone 11 for $302...

Follow-up: "Tell me more about the cheaper one"
Reformulated: "Tell me more about the iPhone 11"

Conversation:
User: Show me iPhones with good cameras
Assistant: The iPhone 12 Pro has an excellent camera system with LiDAR scanner...

Follow-up: "How does it compare to other options?"
Reformulated: "How does the iPhone 12 Pro camera compare to other iPhone models?"

Now reformulate the following query based on the conversation history provided."""

    STANDALONE_PROMPT = """You are a search query optimizer for an Apple iPhone product database.
The database contains reviews, prices, and ratings for Apple iPhone models (iPhone 6 through 14, SE, X, XR, XS, XS Max, 11 Pro, 12 Pro, 13 Pro, etc.).

Your task is to rewrite the user's query into an optimized search query that will perform 
better against a vector database of iPhone reviews and specifications.

Instructions:
- Extract the core intent: what iPhone model, feature, or attribute is the user looking for?
- Expand informal language into precise iPhone terminology (e.g., "good camera" → "camera quality megapixel")
- Include relevant attributes: model name, camera, battery, display, storage, price, condition
- Remove conversational filler ("I want", "can you show me", "I'm looking for")
- Keep it concise — a focused search query, not a sentence
- Return ONLY the optimized query, nothing else

Examples:

User: "I want something good for photos"
Optimized: "iPhone camera quality photo performance"

User: "cheap iphone with good battery"
Optimized: "iPhone budget affordable battery life battery health"

User: "what's the best iphone for the money"
Optimized: "iPhone best value price performance reviews highly rated"

User: "show me iPhone 12 Pro"
Optimized: "iPhone 12 Pro reviews specifications features"

User: "best camera phone"
Optimized: "iPhone camera quality megapixel photo video performance reviews"

Now optimize the following query for vector search retrieval."""
    
    def __init__(self, llm_client: LLMClient):
        """Initialize the query reformulator.
        
        Args:
            llm_client: Initialized LLM client for API calls
        """
        self.llm_client = llm_client
    
    def reformulate(
        self,
        query: str,
        history: Optional[List[BaseMessage]] = None,
        is_follow_up: bool = False
    ) -> str:
        """Reformulate a query for optimal retrieval.
        
        Two modes:
        - Follow-up (is_follow_up=True): resolves pronouns/references using history
        - Standalone (is_follow_up=False): optimizes query for vector search
        
        Args:
            query: The user query
            history: Recent conversation history
            is_follow_up: Whether this is a follow-up to a prior product conversation
            
        Returns:
            Reformulated/optimized query string
        """
        messages = self._build_messages(query, history, is_follow_up)
        
        try:
            response = self.llm_client.invoke(messages)
            reformulated = response.content.strip()
            return reformulated if reformulated else query
        except Exception:
            return query
    
    async def areformulate(
        self,
        query: str,
        history: Optional[List[BaseMessage]] = None,
        is_follow_up: bool = False
    ) -> str:
        """Async version of reformulate.
        
        Args:
            query: The user query
            history: Recent conversation history
            is_follow_up: Whether this is a follow-up to a prior product conversation
            
        Returns:
            Reformulated/optimized query string
        """
        messages = self._build_messages(query, history, is_follow_up)
        
        try:
            response = await self.llm_client.ainvoke(messages)
            reformulated = response.content.strip()
            return reformulated if reformulated else query
        except Exception:
            return query
    
    def _build_messages(
        self,
        query: str,
        history: Optional[List[BaseMessage]],
        is_follow_up: bool
    ) -> List[BaseMessage]:
        """Build message list for the LLM call based on mode.
        
        Args:
            query: The user query
            history: Conversation history
            is_follow_up: Whether to use follow-up or standalone mode
            
        Returns:
            List of BaseMessage objects for LLM
        """
        messages: List[BaseMessage] = []
        
        if is_follow_up and history:
            # Follow-up mode: resolve references from history
            messages.append(SystemMessage(content=self.FOLLOWUP_PROMPT))
            
            recent_history = history[-6:] if len(history) > 6 else history
            history_text = "Conversation History:\n"
            for msg in recent_history:
                if isinstance(msg, HumanMessage):
                    history_text += f"User: {msg.content}\n"
                elif hasattr(msg, 'content'):
                    history_text += f"Assistant: {msg.content}\n"
            
            history_text += f"\nFollow-up Query: {query}\n\nReformulated Query:"
            messages.append(HumanMessage(content=history_text))
        else:
            # Standalone mode: optimize for vector search
            messages.append(SystemMessage(content=self.STANDALONE_PROMPT))
            messages.append(HumanMessage(content=f"User query: {query}\n\nOptimized query:"))
        
        return messages
