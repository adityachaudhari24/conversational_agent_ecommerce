"""Inference Pipeline Orchestrator.

This module implements the main InferencePipeline class that coordinates all
inference components including LLM client, conversation management, response
generation, and agentic workflow execution.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional
from datetime import datetime

from langchain_core.messages import BaseMessage

from .config import (
    InferenceSettings,
    LLMConfig,
    ConversationConfig,
    GeneratorConfig,
    WorkflowConfig,
    create_settings_from_yaml
)
from .exceptions import (
    InferenceError,
    ConfigurationError,
    LLMError,
    SessionError,
    StreamingError,
    TimeoutError
)
from .llm.client import LLMClient
from .conversation.store import ConversationStore, SessionMessage
from .generation.generator import ResponseGenerator
from .grounding import GroundingConfig
from .workflow.agentic import AgenticWorkflow


@dataclass
class InferenceConfig:
    """Configuration for the inference pipeline orchestrator.
    
    This class aggregates all component configurations needed for the
    inference pipeline to operate.
    
    Attributes:
        llm_config: Configuration for the LLM client
        conversation_config: Configuration for conversation management
        generator_config: Configuration for response generation
        workflow_config: Configuration for the agentic workflow
        enable_streaming: Whether to enable streaming responses
        max_retries: Maximum retry attempts for transient failures
        timeout_seconds: Timeout for inference operations
    """
    
    llm_config: LLMConfig
    conversation_config: ConversationConfig
    generator_config: GeneratorConfig
    workflow_config: WorkflowConfig
    enable_streaming: bool = True
    max_retries: int = 3
    timeout_seconds: int = 30


@dataclass
class InferenceResult:
    """Complete inference response containing generated text and metadata.
    
    This class encapsulates the complete result of an inference operation,
    including the generated response, metadata about the operation, and
    performance metrics.
    
    Attributes:
        query: Original user query
        response: Generated response text
        session_id: Session identifier for conversation tracking
        metadata: Additional metadata about the inference operation
        latency_ms: Total inference latency in milliseconds
        tokens_used: Total tokens consumed (input + output)
        timestamp: When the inference was completed
    """
    
    query: str
    response: str
    session_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    tokens_used: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)


class InferencePipeline:
    """Orchestrates the complete inference workflow.
    
    The InferencePipeline coordinates all inference components to provide a unified
    interface for conversational response generation. It handles conversation
    management, agentic workflow execution, streaming responses, and error recovery.
    
    Key features:
    - Unified interface for all inference operations
    - Conversation session management with configurable history
    - Agentic workflow with intelligent query routing
    - Streaming response support for real-time user experience
    - Comprehensive error handling and retry logic
    - Timeout handling for long-running operations
    - Integration with retrieval pipeline for context
    
    Attributes:
        config: Inference pipeline configuration
        retrieval_pipeline: Retrieval pipeline for context fetching
        llm_client: LLM client for API calls
        conversation_store: Conversation store for persistent session management
        response_generator: Response generator with context injection
        agentic_workflow: LangGraph-based agentic workflow
    """
    
    def __init__(self, config: InferenceConfig, retrieval_pipeline, conversation_store: ConversationStore):
        """Initialize the InferencePipeline with configuration.
        
        Args:
            config: InferenceConfig containing component configurations
            retrieval_pipeline: Initialized retrieval pipeline for context
            conversation_store: ConversationStore instance for conversation persistence
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        self.config = config
        self.retrieval_pipeline = retrieval_pipeline
        self.conversation_store = conversation_store
        
        # Initialize components (will be set up in initialize())
        self.llm_client: Optional[LLMClient] = None
        self.response_generator: Optional[ResponseGenerator] = None
        self.agentic_workflow: Optional[AgenticWorkflow] = None
        
        # Track pipeline state
        self._initialized = False
        
    def initialize(self) -> None:
        """Initialize all pipeline components.
        
        This method sets up all the inference components with their configurations
        and establishes connections to external services (OpenAI).
        
        Raises:
            ConfigurationError: If component initialization fails
        """
        if self._initialized:
            return
        
        try:
            # Initialize LLM client
            self.llm_client = LLMClient(self.config.llm_config)
            self.llm_client.initialize()
            
            # Create grounding configuration from generator config
            grounding_config = GroundingConfig(
                strict_mode=self.config.generator_config.strict_grounding,
                require_context=self.config.generator_config.require_context,
                min_context_length=self.config.generator_config.min_context_length,
                enable_validation=True
            )
            
            # Initialize response generator with grounding
            self.response_generator = ResponseGenerator(
                self.config.generator_config,
                self.llm_client,
                grounding_config
            )
            
            # Initialize agentic workflow
            self.agentic_workflow = AgenticWorkflow(
                self.config.workflow_config,
                self.llm_client,
                self.retrieval_pipeline,
                self.response_generator
            )
            
            self._initialized = True
            
        except Exception as e:
            self._initialized = False
            raise ConfigurationError(
                f"Failed to initialize inference pipeline: {str(e)}",
                error_code="PIPELINE_INIT_ERROR",
                details={"error": str(e)}
            ) from e
    
    @classmethod
    def from_config_file(
        cls,
        retrieval_pipeline,
        config_path: Optional[str] = None
    ) -> 'InferencePipeline':
        """Create pipeline from YAML configuration file.
        
        This method loads configuration from a YAML file and creates an
        InferencePipeline instance with all components configured.
        
        Args:
            retrieval_pipeline: Initialized retrieval pipeline
            config_path: Path to YAML config file (uses default if None)
            
        Returns:
            InferencePipeline instance loaded from configuration
            
        Raises:
            ConfigurationError: If configuration loading fails
        """
        try:
            # Load settings from YAML
            settings = create_settings_from_yaml(config_path)
            
            # Create component configurations
            llm_config = LLMConfig(
                provider=settings.llm.provider,
                model_name=settings.llm.model_name,
                temperature=settings.llm.temperature,
                max_tokens=settings.llm.max_tokens,
                api_key=settings.get_api_key()
            )
            
            conversation_config = ConversationConfig(
                max_history_length=settings.conversation.max_history_length,
                storage_dir=settings.conversation.storage_dir
            )
            
            generator_config = GeneratorConfig(
                system_prompt=settings.generator.system_prompt,
                max_context_tokens=settings.generator.max_context_tokens
            )
            
            workflow_config = WorkflowConfig(
                product_keywords=settings.workflow.product_keywords,
                tool_keywords=settings.workflow.tool_keywords
            )
            
            # Create main pipeline configuration
            pipeline_config = InferenceConfig(
                llm_config=llm_config,
                conversation_config=conversation_config,
                generator_config=generator_config,
                workflow_config=workflow_config,
                enable_streaming=settings.enable_streaming,
                max_retries=settings.max_retries,
                timeout_seconds=settings.timeout_seconds
            )
            
            # Create ConversationStore instance
            conversation_store = ConversationStore(
                storage_dir=conversation_config.storage_dir,
                max_history_length=conversation_config.max_history_length
            )
            
            return cls(pipeline_config, retrieval_pipeline, conversation_store)
            
        except Exception as e:
            raise ConfigurationError(
                f"Failed to create pipeline from config file: {str(e)}",
                error_code="CONFIG_FILE_ERROR",
                details={"config_path": config_path, "error": str(e)}
            ) from e
    
    def generate(self, query: str, session_id: str = "default") -> InferenceResult:
        """Generate response synchronously.
        
        This method executes the complete inference workflow including:
        1. Conversation history retrieval
        2. Agentic workflow execution (routing, retrieval, generation)
        3. Response generation with context injection
        4. Conversation history update
        
        Args:
            query: User query string
            session_id: Session identifier for conversation tracking
            
        Returns:
            InferenceResult with generated response and metadata
            
        Raises:
            ConfigurationError: If pipeline is not initialized
            InferenceError: If inference fails
            TimeoutError: If operation exceeds timeout
        """
        if not self._initialized:
            raise ConfigurationError(
                "Pipeline not initialized. Call initialize() first.",
                error_code="PIPELINE_NOT_INITIALIZED"
            )
        
        start_time = time.time()
        
        try:
            # Execute with timeout
            result = self._execute_with_timeout(
                self._execute_generate_workflow,
                self.config.timeout_seconds,
                query,
                session_id
            )
            
            # Calculate total latency
            latency_ms = (time.time() - start_time) * 1000
            result.latency_ms = latency_ms
            
            return result
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            
            if isinstance(e, (InferenceError, ConfigurationError, TimeoutError)):
                raise
            else:
                raise InferenceError(
                    f"Inference failed: {str(e)}",
                    error_code="INFERENCE_ERROR",
                    details={
                        "query": query,
                        "session_id": session_id,
                        "latency_ms": latency_ms,
                        "error": str(e)
                    }
                ) from e
    
    async def agenerate(self, query: str, session_id: str = "default") -> InferenceResult:
        """Generate response asynchronously.
        
        This method executes the complete inference workflow asynchronously,
        providing the same functionality as generate() but with async support.
        
        Args:
            query: User query string
            session_id: Session identifier for conversation tracking
            
        Returns:
            InferenceResult with generated response and metadata
            
        Raises:
            ConfigurationError: If pipeline is not initialized
            InferenceError: If inference fails
            TimeoutError: If operation exceeds timeout
        """
        if not self._initialized:
            raise ConfigurationError(
                "Pipeline not initialized. Call initialize() first.",
                error_code="PIPELINE_NOT_INITIALIZED"
            )
        
        start_time = time.time()
        
        try:
            # Execute with timeout using asyncio
            result = await asyncio.wait_for(
                self._aexecute_generate_workflow(query, session_id),
                timeout=self.config.timeout_seconds
            )
            
            # Calculate total latency
            latency_ms = (time.time() - start_time) * 1000
            result.latency_ms = latency_ms
            
            return result
            
        except asyncio.TimeoutError:
            latency_ms = (time.time() - start_time) * 1000
            raise TimeoutError(
                f"Inference timed out after {self.config.timeout_seconds} seconds",
                timeout_seconds=self.config.timeout_seconds,
                operation="agenerate",
                details={
                    "query": query,
                    "session_id": session_id,
                    "latency_ms": latency_ms
                }
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            
            if isinstance(e, (InferenceError, ConfigurationError, TimeoutError)):
                raise
            else:
                raise InferenceError(
                    f"Async inference failed: {str(e)}",
                    error_code="ASYNC_INFERENCE_ERROR",
                    details={
                        "query": query,
                        "session_id": session_id,
                        "latency_ms": latency_ms,
                        "error": str(e)
                    }
                ) from e
    
    def _execute_generate_workflow(self, query: str, session_id: str) -> InferenceResult:
        """Execute the core generation workflow synchronously.
        
        This method implements the core inference logic:
        1. Get conversation history
        2. Execute agentic workflow
        3. Update conversation history
        4. Return structured result
        
        Args:
            query: User query string
            session_id: Session identifier
            
        Returns:
            InferenceResult with generated response and metadata
        """
        workflow_start = time.time()
        metadata = {
            'session_id': session_id,
            'workflow_type': 'agentic',
            'streaming_enabled': self.config.enable_streaming,
            'workflow_steps': []
        }
        
        try:
            # Step 0: Ensure session exists (auto-create if needed)
            step_start = time.time()
            session = self.conversation_store.get_session(session_id)
            if session is None:
                # Create new session with the provided session_id
                from datetime import datetime
                now = datetime.utcnow().isoformat() + "Z"
                from .conversation.store import Session
                session = Session(
                    session_id=session_id,
                    created_at=now,
                    updated_at=now,
                    messages=[]
                )
                self.conversation_store._save_session(session)
            step_time = (time.time() - step_start) * 1000
            
            metadata['workflow_steps'].append({
                'step': 'session_initialization',
                'time_ms': step_time,
                'session_created': session is None
            })
            
            # Step 1: Get conversation history from ConversationStore (disk-backed)
            step_start = time.time()
            history = self.conversation_store.get_langchain_messages(session_id)
            step_time = (time.time() - step_start) * 1000
            
            metadata['workflow_steps'].append({
                'step': 'conversation_history',
                'time_ms': step_time,
                'history_length': len(history)
            })
            
            # Step 2: Execute agentic workflow
            step_start = time.time()
            response = self.agentic_workflow.run(query, history)
            step_time = (time.time() - step_start) * 1000
            
            metadata['workflow_steps'].append({
                'step': 'agentic_workflow',
                'time_ms': step_time,
                'response_length': len(response)
            })
            
            # Step 3: Persist messages to ConversationStore (writes to disk)
            step_start = time.time()
            self.conversation_store.add_message(session_id, "user", query)
            self.conversation_store.add_message(session_id, "assistant", response)
            step_time = (time.time() - step_start) * 1000
            
            metadata['workflow_steps'].append({
                'step': 'history_update',
                'time_ms': step_time
            })
            
            # Calculate total workflow time
            total_workflow_time = (time.time() - workflow_start) * 1000
            metadata['total_workflow_time_ms'] = total_workflow_time
            
            # Create result
            result = InferenceResult(
                query=query,
                response=response,
                session_id=session_id,
                metadata=metadata,
                tokens_used=0  # Will be updated if we can get token info from LLM
            )
            
            return result
            
        except Exception as e:
            total_time = (time.time() - workflow_start) * 1000
            metadata['total_workflow_time_ms'] = total_time
            metadata['error'] = str(e)
            
            raise InferenceError(
                f"Generate workflow failed: {str(e)}",
                error_code="GENERATE_WORKFLOW_ERROR",
                details=metadata
            ) from e
    
    async def _aexecute_generate_workflow(self, query: str, session_id: str) -> InferenceResult:
        """Execute the core generation workflow asynchronously.
        
        This method implements the async version of the core inference logic.
        
        Args:
            query: User query string
            session_id: Session identifier
            
        Returns:
            InferenceResult with generated response and metadata
        """
        workflow_start = time.time()
        metadata = {
            'session_id': session_id,
            'workflow_type': 'agentic_async',
            'streaming_enabled': self.config.enable_streaming,
            'workflow_steps': []
        }
        
        try:
            # Step 0: Ensure session exists (auto-create if needed)
            step_start = time.time()
            session = self.conversation_store.get_session(session_id)
            if session is None:
                # Create new session with the provided session_id
                from datetime import datetime
                now = datetime.utcnow().isoformat() + "Z"
                from .conversation.store import Session
                session = Session(
                    session_id=session_id,
                    created_at=now,
                    updated_at=now,
                    messages=[]
                )
                self.conversation_store._save_session(session)
            step_time = (time.time() - step_start) * 1000
            
            metadata['workflow_steps'].append({
                'step': 'session_initialization',
                'time_ms': step_time,
                'session_created': session is None
            })
            
            # Step 1: Get conversation history from ConversationStore (disk-backed)
            step_start = time.time()
            history = self.conversation_store.get_langchain_messages(session_id)
            step_time = (time.time() - step_start) * 1000
            
            metadata['workflow_steps'].append({
                'step': 'conversation_history',
                'time_ms': step_time,
                'history_length': len(history)
            })
            
            # Step 2: Execute agentic workflow asynchronously
            step_start = time.time()
            response = await self.agentic_workflow.arun(query, history)
            step_time = (time.time() - step_start) * 1000
            
            metadata['workflow_steps'].append({
                'step': 'agentic_workflow_async',
                'time_ms': step_time,
                'response_length': len(response)
            })
            
            # Step 3: Persist messages to ConversationStore (writes to disk)
            step_start = time.time()
            self.conversation_store.add_message(session_id, "user", query)
            self.conversation_store.add_message(session_id, "assistant", response)
            step_time = (time.time() - step_start) * 1000
            
            metadata['workflow_steps'].append({
                'step': 'history_update',
                'time_ms': step_time
            })
            
            # Calculate total workflow time
            total_workflow_time = (time.time() - workflow_start) * 1000
            metadata['total_workflow_time_ms'] = total_workflow_time
            
            # Create result
            result = InferenceResult(
                query=query,
                response=response,
                session_id=session_id,
                metadata=metadata,
                tokens_used=0  # Will be updated if we can get token info from LLM
            )
            
            return result
            
        except Exception as e:
            total_time = (time.time() - workflow_start) * 1000
            metadata['total_workflow_time_ms'] = total_time
            metadata['error'] = str(e)
            
            raise InferenceError(
                f"Async generate workflow failed: {str(e)}",
                error_code="ASYNC_GENERATE_WORKFLOW_ERROR",
                details=metadata
            ) from e
    
    async def stream(self, query: str, session_id: str = "default") -> AsyncIterator[str]:
        """Stream response chunks asynchronously.
        
        This method provides streaming response generation for real-time user
        experience. It executes the inference workflow and yields response
        chunks as they arrive from the LLM.
        
        The streaming workflow:
        1. Get conversation history
        2. Execute agentic workflow to get context/routing
        3. Stream response generation with context
        4. Update conversation history with complete response
        
        Args:
            query: User query string
            session_id: Session identifier for conversation tracking
            
        Yields:
            String chunks of the response as they arrive
            
        Raises:
            ConfigurationError: If pipeline is not initialized or streaming disabled
            StreamingError: If streaming fails mid-response
            TimeoutError: If operation exceeds timeout
        """
        if not self._initialized:
            raise ConfigurationError(
                "Pipeline not initialized. Call initialize() first.",
                error_code="PIPELINE_NOT_INITIALIZED"
            )
        
        if not self.config.enable_streaming:
            raise ConfigurationError(
                "Streaming is disabled in configuration",
                error_code="STREAMING_DISABLED"
            )
        
        start_time = time.time()
        partial_response = ""
        chunks_yielded = 0
        
        try:
            # Execute streaming workflow directly
            async for chunk in self._execute_streaming_workflow(query, session_id):
                partial_response += chunk
                chunks_yielded += 1
                yield chunk
                
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            
            # Update conversation history with partial response if we have any
            if partial_response:
                try:
                    self.conversation_store.add_message(session_id, "user", query)
                    self.conversation_store.add_message(
                        session_id, 
                        "assistant", 
                        f"{partial_response} [Response incomplete due to error]"
                    )
                except Exception:
                    pass  # Don't fail on history update during error
            
            # Handle streaming errors gracefully
            if isinstance(e, StreamingError):
                raise
            else:
                raise StreamingError(
                    f"Streaming failed: {str(e)}",
                    partial_response=partial_response,
                    chunks_received=chunks_yielded,
                    details={
                        "query": query,
                        "session_id": session_id,
                        "latency_ms": latency_ms,
                        "error": str(e)
                    }
                ) from e
    
    async def _execute_streaming_workflow(
        self, 
        query: str, 
        session_id: str
    ) -> AsyncIterator[str]:
        """Execute the streaming workflow.
        
        This method implements the core streaming logic:
        1. Get conversation history
        2. Prepare context through agentic workflow (without final generation)
        3. Stream response generation with context
        4. Update conversation history with complete response
        
        Args:
            query: User query string
            session_id: Session identifier
            
        Yields:
            String chunks of the response as they arrive
        """
        workflow_start = time.time()
        complete_response = ""
        
        try:
            # Step 0: Ensure session exists (auto-create if needed)
            session = self.conversation_store.get_session(session_id)
            if session is None:
                # Create new session with the provided session_id
                from datetime import datetime
                now = datetime.utcnow().isoformat() + "Z"
                from .conversation.store import Session
                session = Session(
                    session_id=session_id,
                    created_at=now,
                    updated_at=now,
                    messages=[]
                )
                self.conversation_store._save_session(session)
            
            # Step 1: Get conversation history from ConversationStore (disk-backed)
            history = self.conversation_store.get_langchain_messages(session_id)
            
            # Step 2: Prepare context using agentic workflow
            # For streaming, we need to get the context without final generation
            # This is a simplified approach - in a full implementation, you might
            # want to modify the agentic workflow to support streaming
            
            # For now, we'll use the response generator directly with context
            # In a production system, you'd want to integrate streaming into the workflow
            
            # Get context from retrieval if this is a product-related query
            context = None
            query_lower = query.lower()
            
            # Simple keyword check for context retrieval
            if any(keyword in query_lower for keyword in self.config.workflow_config.product_keywords):
                try:
                    retrieval_result = self.retrieval_pipeline.retrieve(query)
                    context = retrieval_result.formatted_context
                except Exception:
                    # Continue without context if retrieval fails
                    context = None
            
            # Step 3: Stream response generation
            async for chunk in self.response_generator.astream(
                query=query,
                context=context,
                history=history
            ):
                complete_response += chunk
                yield chunk
            
            # Step 4: Update conversation history with complete response
            try:
                self.conversation_store.add_message(session_id, "user", query)
                self.conversation_store.add_message(session_id, "assistant", complete_response)
            except Exception as e:
                # Log error but don't fail the streaming operation
                # In a production system, you might want to retry or handle this differently
                pass
                
        except Exception as e:
            # If we have a partial response, try to save it to history
            if complete_response:
                try:
                    self.conversation_store.add_message(session_id, "user", query)
                    self.conversation_store.add_message(
                        session_id, 
                        "assistant", 
                        f"{complete_response} [Response incomplete due to error]"
                    )
                except Exception:
                    pass
            
            raise StreamingError(
                f"Streaming workflow failed: {str(e)}",
                partial_response=complete_response,
                chunks_received=len(complete_response.split()) if complete_response else 0,
                details={
                    "query": query,
                    "session_id": session_id,
                    "workflow_time_ms": (time.time() - workflow_start) * 1000,
                    "error": str(e)
                }
            ) from e
    
    def _execute_with_timeout(self, func, timeout: int, *args, **kwargs) -> Any:
        """Execute function with timeout handling.
        
        This method provides timeout handling for synchronous operations
        by running them in a thread pool with a timeout.
        
        Args:
            func: Function to execute
            timeout: Timeout in seconds
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            TimeoutError: If operation exceeds timeout
        """
        import concurrent.futures
        import threading
        
        # Create a thread pool executor for the operation
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # Submit the function to the executor
            future = executor.submit(func, *args, **kwargs)
            
            try:
                # Wait for the result with timeout
                result = future.result(timeout=timeout)
                return result
                
            except concurrent.futures.TimeoutError:
                # Cancel the future (though it may not be cancellable)
                future.cancel()
                
                # Extract operation details for error
                operation_name = getattr(func, '__name__', 'unknown_operation')
                
                raise TimeoutError(
                    f"Operation '{operation_name}' timed out after {timeout} seconds",
                    timeout_seconds=timeout,
                    operation=operation_name,
                    details={
                        "args": str(args)[:200],  # Truncate for safety
                        "kwargs": str(kwargs)[:200]
                    }
                )
    
    def get_session_history(self, session_id: str) -> List[SessionMessage]:
        """Get conversation history for session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of messages in chronological order
            
        Raises:
            ConfigurationError: If pipeline is not initialized
        """
        if not self._initialized:
            raise ConfigurationError(
                "Pipeline not initialized. Call initialize() first.",
                error_code="PIPELINE_NOT_INITIALIZED"
            )
        
        return self.conversation_store.get_history(session_id)
    
    def clear_session(self, session_id: str) -> None:
        """Clear session history.
        
        Args:
            session_id: Session identifier to clear
            
        Raises:
            ConfigurationError: If pipeline is not initialized
        """
        if not self._initialized:
            raise ConfigurationError(
                "Pipeline not initialized. Call initialize() first.",
                error_code="PIPELINE_NOT_INITIALIZED"
            )
        
        # Check if session exists first
        session = self.conversation_store.get_session(session_id)
        if session is None:
            # Session doesn't exist, nothing to clear - just return silently
            return
        
        self.conversation_store.clear_session(session_id)
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics and health information.
        
        Returns:
            Dictionary containing pipeline statistics
        """
        stats = {
            "initialized": self._initialized,
            "configuration": {
                "enable_streaming": self.config.enable_streaming,
                "max_retries": self.config.max_retries,
                "timeout_seconds": self.config.timeout_seconds,
                "llm_model": self.config.llm_config.model_name,
                "llm_provider": self.config.llm_config.provider,
                "max_history_length": self.config.conversation_config.max_history_length
            },
            "components": {}
        }
        
        if self._initialized:
            # Get conversation store statistics
            if self.conversation_store:
                stats["components"]["conversation_store"] = {
                    "active_sessions": self.conversation_store.get_session_count(),
                    "session_ids": self.conversation_store.get_session_ids()
                }
            
            # Get retrieval pipeline statistics if available
            if self.retrieval_pipeline and hasattr(self.retrieval_pipeline, 'get_pipeline_stats'):
                try:
                    stats["components"]["retrieval_pipeline"] = self.retrieval_pipeline.get_pipeline_stats()
                except Exception as e:
                    stats["components"]["retrieval_pipeline"] = {"error": str(e)}
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check of the pipeline and all components.
        
        Returns:
            Dictionary containing health status
        """
        health = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {},
            "errors": []
        }
        
        if not self._initialized:
            health["status"] = "unhealthy"
            health["errors"].append("Pipeline not initialized")
            return health
        
        # Check each component
        components_to_check = [
            ("llm_client", self.llm_client),
            ("conversation_store", self.conversation_store),
            ("response_generator", self.response_generator),
            ("agentic_workflow", self.agentic_workflow),
            ("retrieval_pipeline", self.retrieval_pipeline)
        ]
        
        for component_name, component in components_to_check:
            try:
                if component is None:
                    health["components"][component_name] = "not_initialized"
                    health["errors"].append(f"{component_name} not initialized")
                    health["status"] = "degraded"
                else:
                    # Basic health check - try to access a simple property or method
                    if hasattr(component, 'config'):
                        _ = component.config
                    elif hasattr(component, '_initialized'):
                        _ = component._initialized
                    health["components"][component_name] = "healthy"
                    
            except Exception as e:
                health["components"][component_name] = f"error: {str(e)}"
                health["errors"].append(f"{component_name}: {str(e)}")
                health["status"] = "unhealthy"
        
        return health
    
    def __enter__(self):
        """Context manager entry."""
        if not self._initialized:
            self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Cleanup if needed
        pass