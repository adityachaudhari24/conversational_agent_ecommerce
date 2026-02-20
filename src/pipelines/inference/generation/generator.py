"""Response generator for the inference pipeline.

This module provides the ResponseGenerator class that constructs prompts and generates
LLM responses with context injection and conversation history management.
"""

from typing import AsyncIterator, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from ..config import GeneratorConfig
from ..grounding import GroundingConfig, GroundingStrategy
from ..llm.client import LLMClient


class ResponseGenerator:
    """Generates LLM responses with context and history injection.
    
    This class handles prompt construction by combining system prompts,
    conversation history, retrieved context, and user queries into properly
    formatted messages for the LLM.
    
    Attributes:
        config: Generator configuration
        llm_client: LLM client for making API calls
        DEFAULT_SYSTEM_PROMPT: Default e-commerce focused system prompt
    """
    
    DEFAULT_SYSTEM_PROMPT = """You are a helpful e-commerce assistant specializing in phone products.
Use the provided context to answer questions accurately.

Instructions:
- Provide accurate information based on the context
- If recommending products, include price and rating when available
- Be conversational and helpful
- If you don't have enough information, say so honestly
- Cite specific products when making recommendations

When context is provided, use it to inform your responses. When no context is available, 
respond based on the conversation history and your general knowledge."""
    
    def __init__(
        self,
        config: GeneratorConfig,
        llm_client: LLMClient,
        grounding_config: Optional[GroundingConfig] = None
    ):
        """Initialize the response generator.
        
        Args:
            config: Generator configuration
            llm_client: Initialized LLM client for API calls
            grounding_config: Optional grounding configuration for strict RAG
        """
        self.config = config
        self.llm_client = llm_client
        self.grounding_config = grounding_config or GroundingConfig(strict_mode=True)
    
    def generate(
        self,
        query: str,
        context: Optional[str] = None,
        history: Optional[List[BaseMessage]] = None
    ) -> str:
        """Generate response synchronously.
        
        Args:
            query: User query to respond to
            context: Optional retrieved context to inject into prompt
            history: Optional conversation history
            
        Returns:
            Generated response string
            
        Raises:
            LLMError: If LLM API call fails
        """
        # Check context quality if grounding is enabled
        if self.grounding_config.require_context:
            is_sufficient, msg = GroundingStrategy.create_retrieval_quality_check(
                context, self.grounding_config.min_context_length
            )
            if not is_sufficient:
                # Return fallback message instead of generating
                return self.grounding_config.fallback_message
        
        messages = self._build_messages(query, context, history)
        response = self.llm_client.invoke(messages)
        
        # Validate response grounding if enabled
        if self.grounding_config.enable_validation:
            is_valid, warning = GroundingStrategy.validate_response_grounding(
                response.content, context, self.grounding_config.strict_mode
            )
            if not is_valid:
                # Log warning or return fallback
                # For now, return fallback message
                return self.grounding_config.fallback_message
        
        return response.content
    
    async def agenerate(
        self,
        query: str,
        context: Optional[str] = None,
        history: Optional[List[BaseMessage]] = None
    ) -> str:
        """Generate response asynchronously.
        
        Args:
            query: User query to respond to
            context: Optional retrieved context to inject into prompt
            history: Optional conversation history
            
        Returns:
            Generated response string
            
        Raises:
            LLMError: If LLM API call fails
        """
        # Check context quality if grounding is enabled
        if self.grounding_config.require_context:
            is_sufficient, msg = GroundingStrategy.create_retrieval_quality_check(
                context, self.grounding_config.min_context_length
            )
            if not is_sufficient:
                return self.grounding_config.fallback_message
        
        messages = self._build_messages(query, context, history)
        response = await self.llm_client.ainvoke(messages)
        
        # Validate response grounding if enabled
        if self.grounding_config.enable_validation:
            is_valid, warning = GroundingStrategy.validate_response_grounding(
                response.content, context, self.grounding_config.strict_mode
            )
            if not is_valid:
                return self.grounding_config.fallback_message
        
        return response.content
    
    async def astream(
        self,
        query: str,
        context: Optional[str] = None,
        history: Optional[List[BaseMessage]] = None
    ) -> AsyncIterator[str]:
        """Stream response chunks asynchronously.
        
        Args:
            query: User query to respond to
            context: Optional retrieved context to inject into prompt
            history: Optional conversation history
            
        Yields:
            String chunks of the response as they arrive
            
        Raises:
            LLMError: If streaming fails
        """
        messages = self._build_messages(query, context, history)
        async for chunk in self.llm_client.astream(messages):
            yield chunk
    
    def _build_messages(
        self,
        query: str,
        context: Optional[str],
        history: Optional[List[BaseMessage]]
    ) -> List[BaseMessage]:
        """Build message list for LLM with proper prompt construction.
        
        This method constructs the complete message sequence by:
        1. Adding the system prompt (with context if available)
        2. Including conversation history
        3. Adding the current user query
        
        Args:
            query: Current user query
            context: Optional retrieved context
            history: Optional conversation history
            
        Returns:
            List of BaseMessage objects ready for LLM
        """
        messages: List[BaseMessage] = []
        
        # 1. Build system prompt with query for grounding
        system_prompt = self._build_system_prompt(context, query)
        messages.append(SystemMessage(content=system_prompt))
        
        # 2. Add conversation history (if provided)
        if history:
            # Filter out any existing system messages from history
            # to avoid duplicate system prompts
            filtered_history = [
                msg for msg in history 
                if not isinstance(msg, SystemMessage)
            ]
            messages.extend(filtered_history)
        
        # 3. Add current user query
        messages.append(HumanMessage(content=query))
        
        return messages
    
    def _build_system_prompt(self, context: Optional[str], query: str = "") -> str:
        """Build the system prompt with optional context injection.
        
        Args:
            context: Optional retrieved context to inject
            query: User query (needed for grounding strategies)
            
        Returns:
            Complete system prompt string
        """
        # Use grounding strategy if configured
        if self.grounding_config and self.grounding_config.strict_mode:
            # Use strict grounding prompt
            system_prompt = GroundingStrategy.build_grounded_system_prompt(
                context=context,
                query=query,
                strict_mode=True
            )
            return system_prompt
        
        # Use custom prompt if configured, otherwise use default
        base_prompt = (
            self.config.system_prompt 
            if self.config.system_prompt 
            else self.DEFAULT_SYSTEM_PROMPT
        )
        
        # Inject context if available
        if context and context.strip():
            # Truncate context if it's too long
            truncated_context = self._truncate_context(context)
            
            system_prompt = f"""{base_prompt}

CONTEXT:
{truncated_context}

Use the above context to inform your responses when relevant."""
        else:
            system_prompt = base_prompt
        
        return system_prompt
    
    def _truncate_context(self, context: str) -> str:
        """Truncate context to fit within token limits.
        
        This is a simple character-based truncation. In a production system,
        you might want to use a proper tokenizer for more accurate truncation.
        
        Args:
            context: Context string to truncate
            
        Returns:
            Truncated context string
        """
        # Rough approximation: 4 characters per token
        max_chars = self.config.max_context_tokens * 4
        
        if len(context) <= max_chars:
            return context
        
        # Truncate and add indicator
        truncation_message = "\n\n[Context truncated due to length...]"
        available_chars = max_chars - len(truncation_message)
        
        if available_chars <= 0:
            return "[Context too long to display]"
        
        truncated = context[:available_chars]
        return f"{truncated}{truncation_message}"