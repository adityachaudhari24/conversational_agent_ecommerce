"""
Retrieval Pipeline Orchestrator.

This module implements the main RetrievalPipeline class that coordinates all
retrieval components including query processing, vector search, contextual
compression, query rewriting, document formatting, and result caching.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
from datetime import datetime

from langchain_core.documents import Document

from .config import RetrievalSettings, ConfigurationLoader
from .exceptions import RetrievalError, ConfigurationError, ConnectionError
from .logging import RetrievalLoggerMixin, log_retrieval_operation, RetrievalMetricsLogger

# Import all components
from .processors.query_processor import QueryProcessor, QueryConfig, ProcessedQuery
from .search.vector_searcher import VectorSearcher, SearchConfig, MetadataFilter, SearchResult
from .processors.context_compressor import ContextCompressor, CompressorConfig, CompressionResult
from .processors.query_rewriter import QueryRewriter, RewriterConfig, RewriteResult
from .processors.document_formatter import DocumentFormatter, FormatterConfig, FormattedContext
from .cache.result_cache import ResultCache, CacheConfig


@dataclass
class RetrievalConfig:
    """Configuration for the retrieval pipeline orchestrator."""
    
    query_config: QueryConfig
    search_config: SearchConfig
    compressor_config: CompressorConfig
    rewriter_config: RewriterConfig
    formatter_config: FormatterConfig
    cache_config: CacheConfig
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    enable_evaluation: bool = False


@dataclass
class RetrievalResult:
    """Complete retrieval response containing documents, metadata, and metrics."""
    
    query: str
    documents: List[Document]
    formatted_context: str
    scores: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    from_cache: bool = False
    evaluation_scores: Optional[Dict[str, float]] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


class RetrievalPipeline(RetrievalLoggerMixin):
    """
    Orchestrates the complete retrieval workflow.
    
    The RetrievalPipeline coordinates all retrieval components to provide a unified
    interface for document retrieval. It handles query processing, vector search,
    contextual compression, query rewriting when needed, document formatting,
    caching, and optional evaluation.
    
    Key features:
    - Unified interface for all retrieval operations
    - Automatic query rewriting for low-relevance results
    - Contextual compression to filter irrelevant documents
    - Result caching for improved performance
    - Comprehensive error handling and retry logic
    - Optional RAGAS evaluation integration
    - Both synchronous and asynchronous operation modes
    """
    
    def __init__(self, config: RetrievalConfig, settings: RetrievalSettings):
        """
        Initialize the RetrievalPipeline with configuration.
        
        Args:
            config: RetrievalConfig containing component configurations
            settings: RetrievalSettings containing API keys and connection info
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        super().__init__()
        self.config = config
        self.settings = settings
        
        # Initialize components (will be set up in initialize())
        self.query_processor: Optional[QueryProcessor] = None
        self.vector_searcher: Optional[VectorSearcher] = None
        self.context_compressor: Optional[ContextCompressor] = None
        self.query_rewriter: Optional[QueryRewriter] = None
        self.document_formatter: Optional[DocumentFormatter] = None
        self.cache: Optional[ResultCache] = None
        self.evaluator = None  # Will be implemented when RAGAS evaluation is added
        
        # Metrics logger
        self.metrics_logger = RetrievalMetricsLogger("RetrievalPipeline")
        
        # Track pipeline state
        self._initialized = False
        self._current_query_hash: Optional[str] = None
        
        self.logger.info(
            "RetrievalPipeline created",
            extra={
                'extra_fields': {
                    'max_retries': config.max_retries,
                    'retry_delay_seconds': config.retry_delay_seconds,
                    'enable_evaluation': config.enable_evaluation,
                    'cache_enabled': config.cache_config.enabled,
                    'compression_enabled': config.compressor_config.enabled
                }
            }
        )
    
    @classmethod
    def from_settings(cls, settings: RetrievalSettings) -> 'RetrievalPipeline':
        """
        Create RetrievalPipeline from RetrievalSettings.
        
        Args:
            settings: RetrievalSettings instance
            
        Returns:
            RetrievalPipeline instance with components configured from settings
        """
        # Create component configurations from settings
        query_config = QueryConfig(
            max_length=settings.max_query_length,
            embedding_model=settings.embedding_model,
            normalize_unicode=settings.normalize_unicode,
            expected_dimension=settings.embedding_dimension
        )
        
        search_config = SearchConfig(
            top_k=settings.top_k,
            fetch_k=settings.fetch_k,
            lambda_mult=settings.lambda_mult,
            score_threshold=settings.score_threshold,
            search_type=settings.search_type
        )
        
        compressor_config = CompressorConfig(
            enabled=settings.compression_enabled,
            relevance_prompt=settings.relevance_prompt,
            llm_model="gpt-3.5-turbo",
            temperature=0.0,
            max_tokens=10
        )
        
        rewriter_config = RewriterConfig(
            max_attempts=settings.max_rewrite_attempts,
            relevance_threshold=settings.rewrite_threshold,
            rewrite_prompt=settings.rewrite_prompt,
            llm_model="gpt-3.5-turbo",
            temperature=0.3,
            max_tokens=150
        )
        
        formatter_config = FormatterConfig(
            template=settings.format_template,
            delimiter=settings.format_delimiter,
            include_scores=settings.include_scores,
            max_context_length=settings.max_context_length,
            truncate_on_limit=True
        )
        
        cache_config = CacheConfig(
            enabled=settings.cache_enabled,
            ttl_seconds=settings.cache_ttl_seconds,
            max_size=settings.cache_max_size
        )
        
        # Create main pipeline configuration
        pipeline_config = RetrievalConfig(
            query_config=query_config,
            search_config=search_config,
            compressor_config=compressor_config,
            rewriter_config=rewriter_config,
            formatter_config=formatter_config,
            cache_config=cache_config,
            max_retries=settings.max_retries,
            retry_delay_seconds=settings.retry_delay_seconds,
            enable_evaluation=settings.enable_evaluation
        )
        
        return cls(pipeline_config, settings)
    
    @classmethod
    def from_config_file(cls, config_path: Optional[str] = None) -> 'RetrievalPipeline':
        """
        Create RetrievalPipeline from configuration file.
        
        Args:
            config_path: Path to YAML configuration file (optional)
            
        Returns:
            RetrievalPipeline instance loaded from configuration
            
        Raises:
            ConfigurationError: If configuration loading fails
        """
        loader = ConfigurationLoader(config_path)
        settings = loader.load_config()
        return cls.from_settings(settings)
    
    @log_retrieval_operation("pipeline_initialization")
    def initialize(self) -> None:
        """
        Initialize all pipeline components.
        
        This method sets up all the retrieval components with their configurations
        and establishes connections to external services (Pinecone, OpenAI).
        
        Raises:
            ConfigurationError: If component initialization fails
            ConnectionError: If external service connections fail
        """
        if self._initialized:
            self.logger.info("Pipeline already initialized, skipping")
            return
        
        try:
            # Initialize query processor
            self.logger.info("Initializing QueryProcessor...")
            self.query_processor = QueryProcessor(
                self.config.query_config,
                self.settings.openai_api_key
            )
            
            # Initialize vector searcher
            self.logger.info("Initializing VectorSearcher...")
            self.vector_searcher = VectorSearcher(
                self.config.search_config,
                self.settings
            )
            self.vector_searcher.initialize()
            
            # Initialize context compressor
            self.logger.info("Initializing ContextCompressor...")
            self.context_compressor = ContextCompressor(
                self.config.compressor_config,
                self.settings.openai_api_key
            )
            self.context_compressor.initialize()
            
            # Initialize query rewriter
            self.logger.info("Initializing QueryRewriter...")
            self.query_rewriter = QueryRewriter(
                self.config.rewriter_config,
                self.settings.openai_api_key
            )
            
            # Initialize document formatter
            self.logger.info("Initializing DocumentFormatter...")
            self.document_formatter = DocumentFormatter(self.config.formatter_config)
            
            # Initialize result cache
            self.logger.info("Initializing ResultCache...")
            self.cache = ResultCache(self.config.cache_config)
            
            # TODO: Initialize RAGAS evaluator when evaluation module is implemented
            if self.config.enable_evaluation:
                self.logger.warning("RAGAS evaluation requested but not yet implemented")
            
            self._initialized = True
            
            self.logger.info(
                "RetrievalPipeline initialization completed successfully",
                extra={
                    'extra_fields': {
                        'components_initialized': [
                            'QueryProcessor',
                            'VectorSearcher', 
                            'ContextCompressor',
                            'QueryRewriter',
                            'DocumentFormatter',
                            'ResultCache'
                        ],
                        'evaluation_enabled': self.config.enable_evaluation
                    }
                }
            )
            
        except Exception as e:
            self._initialized = False
            self.logger.error(f"Pipeline initialization failed: {e}", exc_info=True)
            raise ConfigurationError(f"Failed to initialize retrieval pipeline: {e}") from e
    
    @log_retrieval_operation("retrieval")
    def retrieve(
        self, 
        query: str,
        filters: Optional[MetadataFilter] = None
    ) -> RetrievalResult:
        """
        Execute synchronous retrieval.
        
        Args:
            query: User query string
            filters: Optional metadata filters to apply to search
            
        Returns:
            RetrievalResult containing retrieved documents and metadata
            
        Raises:
            RetrievalError: If retrieval fails
            ConfigurationError: If pipeline is not initialized
        """
        if not self._initialized:
            raise ConfigurationError("Pipeline not initialized. Call initialize() first.")
        
        start_time = time.time()
        self._current_query_hash = str(hash(query) % 10000)
        
        try:
            # Check cache first
            cached_result = self.cache.get(query, filters.__dict__ if filters else None)
            if cached_result is not None:
                cached_result.from_cache = True
                latency_ms = (time.time() - start_time) * 1000
                cached_result.latency_ms = latency_ms
                
                self.metrics_logger.log_cache_operation(
                    "hit", 
                    self._current_query_hash,
                    cache_size=len(self.cache._cache)
                )
                
                self.metrics_logger.log_pipeline_metrics(
                    total_time_ms=latency_ms,
                    query_hash=self._current_query_hash,
                    documents_retrieved=len(cached_result.documents),
                    cache_hit=True
                )
                
                return cached_result
            
            # Cache miss - log it
            self.metrics_logger.log_cache_operation(
                "miss",
                self._current_query_hash,
                cache_size=len(self.cache._cache)
            )
            
            # Execute retrieval with retry logic
            result = self._execute_with_retry(
                self._execute_retrieval_workflow,
                query,
                filters
            )
            
            # Calculate total latency
            total_latency_ms = (time.time() - start_time) * 1000
            result.latency_ms = total_latency_ms
            result.from_cache = False
            
            # Cache the result
            self.cache.set(query, result, filters.__dict__ if filters else None)
            self.metrics_logger.log_cache_operation(
                "set",
                self._current_query_hash,
                cache_size=len(self.cache._cache)
            )
            
            # Log final pipeline metrics
            self.metrics_logger.log_pipeline_metrics(
                total_time_ms=total_latency_ms,
                query_hash=self._current_query_hash,
                documents_retrieved=len(result.documents),
                cache_hit=False,
                rewrite_attempts=result.metadata.get('rewrite_attempts', 0),
                compression_applied=result.metadata.get('compression_applied', False)
            )
            
            return result
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.logger.error(
                f"Retrieval failed after {latency_ms:.2f}ms: {e}",
                extra={
                    'extra_fields': {
                        'query_hash': self._current_query_hash,
                        'latency_ms': latency_ms,
                        'error_type': type(e).__name__,
                        'filters_applied': filters.__dict__ if filters else None
                    }
                },
                exc_info=True
            )
            raise
    
    async def aretrieve(
        self, 
        query: str,
        filters: Optional[MetadataFilter] = None
    ) -> RetrievalResult:
        """
        Execute asynchronous retrieval.
        
        Args:
            query: User query string
            filters: Optional metadata filters to apply to search
            
        Returns:
            RetrievalResult containing retrieved documents and metadata
            
        Raises:
            RetrievalError: If retrieval fails
            ConfigurationError: If pipeline is not initialized
        """
        # For now, run the synchronous version in a thread pool
        # In a full async implementation, all components would have async methods
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.retrieve, query, filters)
    
    def _execute_retrieval_workflow(
        self,
        query: str,
        filters: Optional[MetadataFilter] = None
    ) -> RetrievalResult:
        """
        Execute the core retrieval workflow.
        
        Args:
            query: User query string
            filters: Optional metadata filters
            
        Returns:
            RetrievalResult containing retrieved documents and metadata
        """
        workflow_start = time.time()
        metadata = {
            'original_query': query,
            'filters_applied': filters.__dict__ if filters else {},
            'rewrite_attempts': 0,
            'compression_applied': False,
            'workflow_steps': []
        }
        
        # Reset query rewriter attempts for new query
        if self.query_rewriter:
            self.query_rewriter.reset_attempts()
        
        current_query = query
        best_result = None
        best_relevance_score = 0.0
        
        # Main retrieval loop (allows for query rewriting)
        max_iterations = self.config.rewriter_config.max_attempts + 1
        
        for iteration in range(max_iterations):
            try:
                # Step 1: Process query
                step_start = time.time()
                processed_query = self.query_processor.process(current_query)
                step_time = (time.time() - step_start) * 1000
                
                metadata['workflow_steps'].append({
                    'step': 'query_processing',
                    'iteration': iteration,
                    'time_ms': step_time,
                    'query_truncated': processed_query.truncated
                })
                
                # Step 2: Vector search
                step_start = time.time()
                search_result = self.vector_searcher.search(
                    processed_query.embedding,
                    filters
                )
                step_time = (time.time() - step_start) * 1000
                
                metadata['workflow_steps'].append({
                    'step': 'vector_search',
                    'iteration': iteration,
                    'time_ms': step_time,
                    'documents_found': len(search_result.documents),
                    'search_type': search_result.search_metadata.get('search_type')
                })
                
                if not search_result.documents:
                    self.logger.warning(f"No documents found for query (iteration {iteration})")
                    # If no documents found, return empty result
                    if iteration == 0:
                        return self._create_empty_result(query, metadata, workflow_start)
                    else:
                        # Use best result from previous iteration
                        break
                
                # Step 3: Contextual compression
                step_start = time.time()
                compression_result = self.context_compressor.compress(
                    current_query,
                    search_result.documents
                )
                step_time = (time.time() - step_start) * 1000
                
                metadata['compression_applied'] = self.config.compressor_config.enabled
                metadata['workflow_steps'].append({
                    'step': 'contextual_compression',
                    'iteration': iteration,
                    'time_ms': step_time,
                    'input_documents': len(search_result.documents),
                    'output_documents': len(compression_result.documents),
                    'compression_ratio': compression_result.compression_ratio
                })
                
                # Calculate average relevance score
                avg_relevance = (
                    sum(compression_result.relevance_scores) / len(compression_result.relevance_scores)
                    if compression_result.relevance_scores else 0.0
                )
                
                # Step 4: Check if we should rewrite the query
                should_rewrite = (
                    iteration < max_iterations - 1 and  # Not the last iteration
                    self.query_rewriter.should_rewrite(avg_relevance) and
                    len(compression_result.documents) > 0  # Only rewrite if we have some results
                )
                
                if should_rewrite:
                    # Step 5: Query rewriting
                    step_start = time.time()
                    rewrite_result = self.query_rewriter.rewrite(
                        current_query,
                        f"Previous search returned {len(compression_result.documents)} documents "
                        f"with average relevance {avg_relevance:.3f}"
                    )
                    step_time = (time.time() - step_start) * 1000
                    
                    metadata['rewrite_attempts'] += 1
                    metadata['workflow_steps'].append({
                        'step': 'query_rewriting',
                        'iteration': iteration,
                        'time_ms': step_time,
                        'attempt_number': rewrite_result.attempt_number,
                        'improvement_reason': rewrite_result.improvement_reason
                    })
                    
                    # Update current query for next iteration
                    current_query = rewrite_result.rewritten_query
                    
                    self.logger.info(
                        f"Query rewritten (iteration {iteration}): "
                        f"relevance {avg_relevance:.3f} < threshold {self.config.rewriter_config.relevance_threshold:.3f}"
                    )
                    
                    # Store this result as potential fallback
                    if avg_relevance > best_relevance_score:
                        best_result = (compression_result, search_result.scores, metadata.copy())
                        best_relevance_score = avg_relevance
                    
                    continue  # Try again with rewritten query
                
                else:
                    # No rewrite needed or max attempts reached - use current results
                    final_documents = compression_result.documents
                    final_scores = search_result.scores[:len(final_documents)]
                    break
                    
            except Exception as e:
                self.logger.error(f"Error in retrieval workflow iteration {iteration}: {e}")
                if iteration == 0:
                    # If first iteration fails, re-raise the error
                    raise
                else:
                    # Use best result from previous iterations
                    self.logger.warning(f"Using best result from previous iteration due to error: {e}")
                    break
        
        # Use best result if we have one, otherwise use current results
        if best_result is not None and (not 'final_documents' in locals() or len(final_documents) == 0):
            compression_result, final_scores, metadata = best_result
            final_documents = compression_result.documents
            self.logger.info("Using best result from previous iteration")
        
        # Step 6: Document formatting
        step_start = time.time()
        formatted_context = self.document_formatter.format(
            final_documents,
            final_scores
        )
        step_time = (time.time() - step_start) * 1000
        
        metadata['workflow_steps'].append({
            'step': 'document_formatting',
            'time_ms': step_time,
            'output_length': len(formatted_context.text),
            'documents_formatted': formatted_context.document_count,
            'truncated': formatted_context.truncated
        })
        
        # Step 7: Optional evaluation (placeholder for future implementation)
        evaluation_scores = None
        if self.config.enable_evaluation and self.evaluator:
            # TODO: Implement RAGAS evaluation
            pass
        
        # Create final result
        total_workflow_time = (time.time() - workflow_start) * 1000
        metadata['total_workflow_time_ms'] = total_workflow_time
        
        result = RetrievalResult(
            query=query,
            documents=final_documents,
            formatted_context=formatted_context.text,
            scores=final_scores,
            metadata=metadata,
            evaluation_scores=evaluation_scores
        )
        
        self.logger.info(
            f"Retrieval workflow completed in {total_workflow_time:.2f}ms",
            extra={
                'extra_fields': {
                    'query_hash': self._current_query_hash,
                    'total_time_ms': total_workflow_time,
                    'final_documents': len(final_documents),
                    'rewrite_attempts': metadata['rewrite_attempts'],
                    'compression_applied': metadata['compression_applied'],
                    'workflow_steps': len(metadata['workflow_steps'])
                }
            }
        )
        
        return result
    
    def _create_empty_result(
        self,
        query: str,
        metadata: Dict[str, Any],
        workflow_start: float
    ) -> RetrievalResult:
        """
        Create an empty result when no documents are found.
        
        Args:
            query: Original query
            metadata: Workflow metadata
            workflow_start: Workflow start time
            
        Returns:
            Empty RetrievalResult
        """
        total_time = (time.time() - workflow_start) * 1000
        metadata['total_workflow_time_ms'] = total_time
        
        return RetrievalResult(
            query=query,
            documents=[],
            formatted_context="",
            scores=[],
            metadata=metadata
        )
    
    def _execute_with_retry(
        self, 
        func,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with retry logic for transient failures.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            RetrievalError: If all retries are exhausted
        """
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):  # +1 for initial attempt
            try:
                return func(*args, **kwargs)
                
            except (ConnectionError, RetrievalError) as e:
                last_exception = e
                
                if attempt < self.config.max_retries:
                    wait_time = self.config.retry_delay_seconds * (2 ** attempt)  # Exponential backoff
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed, retrying in {wait_time:.1f}s: {e}",
                        extra={
                            'extra_fields': {
                                'attempt': attempt + 1,
                                'max_retries': self.config.max_retries,
                                'wait_time_seconds': wait_time,
                                'error_type': type(e).__name__
                            }
                        }
                    )
                    time.sleep(wait_time)
                else:
                    self.logger.error(
                        f"All {self.config.max_retries + 1} attempts failed",
                        extra={
                            'extra_fields': {
                                'total_attempts': self.config.max_retries + 1,
                                'final_error_type': type(e).__name__
                            }
                        }
                    )
                    break
            
            except Exception as e:
                # Non-retrieval errors should not be retried
                self.logger.error(f"Non-retrieval error, not retrying: {e}")
                raise
        
        # If we get here, all retries were exhausted
        raise RetrievalError(
            f"Retrieval failed after {self.config.max_retries + 1} attempts",
            details={
                "max_retries": self.config.max_retries,
                "last_error": str(last_exception),
                "last_error_type": type(last_exception).__name__ if last_exception else None
            }
        ) from last_exception
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive pipeline statistics and health information.
        
        Returns:
            Dictionary containing pipeline statistics
        """
        stats = {
            "initialized": self._initialized,
            "configuration": {
                "max_retries": self.config.max_retries,
                "retry_delay_seconds": self.config.retry_delay_seconds,
                "enable_evaluation": self.config.enable_evaluation
            },
            "components": {}
        }
        
        if self._initialized:
            # Get component statistics
            if self.vector_searcher:
                try:
                    stats["components"]["vector_searcher"] = self.vector_searcher.get_index_stats()
                except Exception as e:
                    stats["components"]["vector_searcher"] = {"error": str(e)}
            
            if self.context_compressor:
                stats["components"]["context_compressor"] = self.context_compressor.get_compression_stats()
            
            if self.query_rewriter:
                stats["components"]["query_rewriter"] = self.query_rewriter.get_rewriter_stats()
            
            if self.document_formatter:
                stats["components"]["document_formatter"] = self.document_formatter.get_formatting_stats()
            
            if self.cache:
                stats["components"]["cache"] = self.cache.get_stats()
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of the pipeline and all components.
        
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
            ("query_processor", self.query_processor),
            ("vector_searcher", self.vector_searcher),
            ("context_compressor", self.context_compressor),
            ("query_rewriter", self.query_rewriter),
            ("document_formatter", self.document_formatter),
            ("cache", self.cache)
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
                    health["components"][component_name] = "healthy"
                    
            except Exception as e:
                health["components"][component_name] = f"error: {str(e)}"
                health["errors"].append(f"{component_name}: {str(e)}")
                health["status"] = "unhealthy"
        
        return health
    
    def clear_cache(self) -> None:
        """Clear the result cache."""
        if self.cache:
            self.cache.clear()
            self.logger.info("Pipeline cache cleared")
    
    def __enter__(self):
        """Context manager entry."""
        if not self._initialized:
            self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Cleanup if needed
        pass