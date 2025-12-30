"""LLM client for the inference pipeline.

This module provides a configurable LLM client with support for OpenAI models,
retry logic with exponential backoff, and both synchronous and streaming responses.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import AsyncIterator, List, Optional

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from ..config import LLMConfig
from ..exceptions import ConfigurationError, LLMError


@dataclass
class LLMResponse:
    """Response from LLM invocation.
    
    Attributes:
        content: The generated text content
        model: Model name that generated the response
        tokens_used: Total tokens consumed (input + output)
        latency_ms: Response latency in milliseconds
    """
    content: str
    model: str
    tokens_used: int
    latency_ms: float


class LLMClient:
    """OpenAI LLM client with retry logic and streaming support.
    
    This client provides a unified interface for interacting with OpenAI's
    chat completion API, with built-in retry logic, error handling, and
    support for both synchronous and streaming responses.
    
    Attributes:
        config: LLM configuration
        client: Synchronous OpenAI client
        async_client: Asynchronous OpenAI client
        langchain_client: LangChain ChatOpenAI client for compatibility
    """
    
    def __init__(self, config: LLMConfig):
        """Initialize the LLM client.
        
        Args:
            config: LLM configuration including API key and model settings
        """
        self.config = config
        self.client: Optional[OpenAI] = None
        self.async_client: Optional[AsyncOpenAI] = None
        self.langchain_client: Optional[ChatOpenAI] = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize OpenAI clients with API key validation.
        
        Raises:
            ConfigurationError: If API key is missing or invalid
        """
        if self._initialized:
            return
        
        # Validate API key
        if not self.config.api_key:
            raise ConfigurationError(
                "OpenAI API key is required but not provided",
                missing_keys=["api_key"],
                error_code="MISSING_API_KEY"
            )
        
        if not self.config.api_key.strip():
            raise ConfigurationError(
                "OpenAI API key cannot be empty",
                error_code="EMPTY_API_KEY"
            )
        
        try:
            # Initialize OpenAI clients
            self.client = OpenAI(api_key=self.config.api_key)
            self.async_client = AsyncOpenAI(api_key=self.config.api_key)
            
            # Initialize LangChain client for compatibility
            self.langchain_client = ChatOpenAI(
                model=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                api_key=self.config.api_key,
            )
            
            self._initialized = True
            
        except Exception as e:
            raise ConfigurationError(
                f"Failed to initialize OpenAI client: {str(e)}",
                error_code="CLIENT_INIT_ERROR",
                details={"error": str(e)}
            )
    
    def invoke(self, messages: List[BaseMessage]) -> LLMResponse:
        """Synchronous LLM invocation.
        
        Args:
            messages: List of LangChain messages for the conversation
            
        Returns:
            LLMResponse with generated content and metadata
            
        Raises:
            ConfigurationError: If client is not initialized
            LLMError: If LLM API call fails after all retries
        """
        if not self._initialized:
            raise ConfigurationError(
                "LLM client must be initialized before use",
                error_code="CLIENT_NOT_INITIALIZED"
            )
        
        def _invoke() -> LLMResponse:
            start_time = time.time()
            
            # Convert LangChain messages to OpenAI format
            openai_messages = self._convert_messages(messages)
            
            # Make API call
            response: ChatCompletion = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=openai_messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            return LLMResponse(
                content=response.choices[0].message.content or "",
                model=response.model,
                tokens_used=response.usage.total_tokens if response.usage else 0,
                latency_ms=latency_ms
            )
        
        return self._execute_with_retry(_invoke)
    
    async def ainvoke(self, messages: List[BaseMessage]) -> LLMResponse:
        """Asynchronous LLM invocation.
        
        Args:
            messages: List of LangChain messages for the conversation
            
        Returns:
            LLMResponse with generated content and metadata
            
        Raises:
            ConfigurationError: If client is not initialized
            LLMError: If LLM API call fails after all retries
        """
        if not self._initialized:
            raise ConfigurationError(
                "LLM client must be initialized before use",
                error_code="CLIENT_NOT_INITIALIZED"
            )
        
        async def _ainvoke() -> LLMResponse:
            start_time = time.time()
            
            # Convert LangChain messages to OpenAI format
            openai_messages = self._convert_messages(messages)
            
            # Make async API call
            response: ChatCompletion = await self.async_client.chat.completions.create(
                model=self.config.model_name,
                messages=openai_messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            return LLMResponse(
                content=response.choices[0].message.content or "",
                model=response.model,
                tokens_used=response.usage.total_tokens if response.usage else 0,
                latency_ms=latency_ms
            )
        
        return await self._aexecute_with_retry(_ainvoke)
    
    async def astream(self, messages: List[BaseMessage]) -> AsyncIterator[str]:
        """Stream LLM response chunks asynchronously.
        
        Args:
            messages: List of LangChain messages for the conversation
            
        Yields:
            String chunks of the response as they arrive
            
        Raises:
            ConfigurationError: If client is not initialized
            LLMError: If streaming fails
        """
        if not self._initialized:
            raise ConfigurationError(
                "LLM client must be initialized before use",
                error_code="CLIENT_NOT_INITIALIZED"
            )
        
        try:
            # Convert LangChain messages to OpenAI format
            openai_messages = self._convert_messages(messages)
            
            # Create streaming completion
            stream = await self.async_client.chat.completions.create(
                model=self.config.model_name,
                messages=openai_messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=True,
            )
            
            async for chunk in stream:
                chunk: ChatCompletionChunk
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            raise LLMError(
                f"Streaming failed: {str(e)}",
                provider="openai",
                model=self.config.model_name,
                error_code="STREAMING_ERROR",
                details={"error": str(e)}
            )
    
    def _convert_messages(self, messages: List[BaseMessage]) -> List[dict]:
        """Convert LangChain messages to OpenAI format.
        
        Args:
            messages: List of LangChain BaseMessage objects
            
        Returns:
            List of dictionaries in OpenAI message format
        """
        openai_messages = []
        
        for message in messages:
            if hasattr(message, 'type'):
                # Handle different LangChain message types
                if message.type == "system":
                    role = "system"
                elif message.type == "human":
                    role = "user"
                elif message.type == "ai":
                    role = "assistant"
                else:
                    role = "user"  # Default fallback
            else:
                # Fallback for messages without type attribute
                role = "user"
            
            openai_messages.append({
                "role": role,
                "content": message.content
            })
        
        return openai_messages
    
    def _execute_with_retry(self, func, max_retries: Optional[int] = None):
        """Execute function with exponential backoff retry logic.
        
        Args:
            func: Function to execute
            max_retries: Maximum retry attempts (uses config default if None)
            
        Returns:
            Result of the function call
            
        Raises:
            LLMError: If all retry attempts are exhausted
        """
        if max_retries is None:
            max_retries = 3  # Default retry count
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return func()
            except Exception as e:
                last_exception = e
                
                # Don't retry on configuration errors
                if isinstance(e, ConfigurationError):
                    raise
                
                # Don't retry on final attempt
                if attempt == max_retries:
                    break
                
                # Calculate exponential backoff delay
                delay = (2 ** attempt) * 0.5  # 0.5s, 1s, 2s, 4s...
                time.sleep(delay)
        
        # All retries exhausted
        raise LLMError(
            f"LLM API call failed after {max_retries + 1} attempts: {str(last_exception)}",
            provider="openai",
            model=self.config.model_name,
            error_code="MAX_RETRIES_EXCEEDED",
            details={
                "attempts": max_retries + 1,
                "last_error": str(last_exception)
            }
        )
    
    async def _aexecute_with_retry(self, func, max_retries: Optional[int] = None):
        """Execute async function with exponential backoff retry logic.
        
        Args:
            func: Async function to execute
            max_retries: Maximum retry attempts (uses config default if None)
            
        Returns:
            Result of the function call
            
        Raises:
            LLMError: If all retry attempts are exhausted
        """
        if max_retries is None:
            max_retries = 3  # Default retry count
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return await func()
            except Exception as e:
                last_exception = e
                
                # Don't retry on configuration errors
                if isinstance(e, ConfigurationError):
                    raise
                
                # Don't retry on final attempt
                if attempt == max_retries:
                    break
                
                # Calculate exponential backoff delay
                delay = (2 ** attempt) * 0.5  # 0.5s, 1s, 2s, 4s...
                await asyncio.sleep(delay)
        
        # All retries exhausted
        raise LLMError(
            f"LLM API call failed after {max_retries + 1} attempts: {str(last_exception)}",
            provider="openai",
            model=self.config.model_name,
            error_code="MAX_RETRIES_EXCEEDED",
            details={
                "attempts": max_retries + 1,
                "last_error": str(last_exception)
            }
        )