"""
Query Rewriter for the retrieval pipeline.

This module implements query rewriting using LLM-based reformulation to improve
retrieval results when initial queries produce low-relevance documents. It helps
transform vague or unclear queries into more specific, searchable versions.
"""

from dataclasses import dataclass
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ..config import RetrievalSettings
from ..exceptions import RetrievalError, ConfigurationError
from ..logging import RetrievalLoggerMixin, log_retrieval_operation


@dataclass
class RewriterConfig:
    """Configuration for query rewriting."""
    
    max_attempts: int = 2
    relevance_threshold: float = 0.5  # Trigger rewrite below this score
    rewrite_prompt: Optional[str] = None  # Custom rewrite prompt
    llm_model: str = "gpt-3.5-turbo"  # LLM model for rewriting
    temperature: float = 0.3  # Some creativity but mostly deterministic
    max_tokens: Optional[int] = 150  # Reasonable length for rewritten queries


@dataclass
class RewriteResult:
    """Result of query rewriting containing original and rewritten queries."""
    
    original_query: str
    rewritten_query: str
    attempt_number: int
    improvement_reason: str
    processing_time_ms: float = 0.0


class QueryRewriter(RetrievalLoggerMixin):
    """
    Rewrites queries for improved retrieval results.
    
    The QueryRewriter uses an LLM to reformulate queries that produce low-relevance
    results. It analyzes the original query and context to generate clearer, more
    specific versions that are likely to yield better search results.
    """
    
    DEFAULT_REWRITE_PROMPT = """
    You are an expert at improving search queries for e-commerce product searches.
    
    The original query below produced search results with low relevance scores, 
    indicating the query might be too vague, unclear, or not well-suited for 
    product search.
    
    Original Query: "{original_query}"
    
    Context (if available): {context}
    
    Please rewrite this query to be:
    1. More specific and clear
    2. Better suited for finding relevant products
    3. Focused on searchable product attributes (name, features, category, etc.)
    4. Concise but descriptive
    
    Provide ONLY the improved query, nothing else. Do not include explanations 
    or additional text.
    
    Improved Query:"""
    
    def __init__(self, config: RewriterConfig, openai_api_key: str):
        """
        Initialize the QueryRewriter with configuration.
        
        Args:
            config: RewriterConfig containing rewriting settings
            openai_api_key: OpenAI API key for LLM access
            
        Raises:
            ConfigurationError: If API key is missing or LLM initialization fails
        """
        super().__init__()
        self.config = config
        self.llm: Optional[ChatOpenAI] = None
        self.rewrite_chain = None
        self._attempt_count = 0
        
        if not openai_api_key:
            raise ConfigurationError(
                "OpenAI API key is required for query rewriting. "
                "Please set OPENAI_API_KEY environment variable."
            )
        
        try:
            # Initialize the LLM
            self.llm = ChatOpenAI(
                model=config.llm_model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                openai_api_key=openai_api_key
            )
            
            # Test the LLM connection
            test_response = self.llm.invoke("Test connection")
            if not test_response or not hasattr(test_response, 'content'):
                raise ConfigurationError("LLM connection test failed")
            
            # Initialize the rewrite chain
            self._initialize_chain()
            
            self.logger.info(
                f"Successfully initialized QueryRewriter with model: {config.llm_model}"
            )
            
        except Exception as e:
            if "api" in str(e).lower() or "key" in str(e).lower():
                raise ConfigurationError(f"Invalid OpenAI API key or API error: {e}")
            else:
                raise ConfigurationError(f"Failed to initialize LLM for query rewriting: {e}")
    
    @classmethod
    def from_settings(cls, settings: RetrievalSettings) -> 'QueryRewriter':
        """
        Create QueryRewriter from RetrievalSettings.
        
        Args:
            settings: RetrievalSettings instance
            
        Returns:
            QueryRewriter instance
        """
        config = RewriterConfig(
            max_attempts=settings.max_rewrite_attempts,
            relevance_threshold=settings.rewrite_threshold,
            rewrite_prompt=settings.rewrite_prompt,
            llm_model="gpt-3.5-turbo",  # Use cost-effective model for rewriting
            temperature=0.3,  # Balance creativity with consistency
            max_tokens=150  # Reasonable length for rewritten queries
        )
        
        return cls(config, settings.openai_api_key)
    
    def _initialize_chain(self) -> None:
        """
        Initialize the LLM chain for query rewriting.
        
        This method sets up a LangChain chain with the configured
        rewrite prompt and LLM model for query reformulation.
        
        Raises:
            ConfigurationError: If LLM is not initialized
        """
        if self.llm is None:
            raise ConfigurationError("LLM not initialized for query rewriting")
        
        try:
            # Use custom prompt if provided, otherwise use default
            prompt_template = self.config.rewrite_prompt or self.DEFAULT_REWRITE_PROMPT
            
            # Create the prompt template
            prompt = PromptTemplate(
                input_variables=["original_query", "context"],
                template=prompt_template
            )
            
            # Create the chain: prompt -> llm -> output parser
            output_parser = StrOutputParser()
            self.rewrite_chain = prompt | self.llm | output_parser
            
            self.logger.info("LLM rewrite chain initialized successfully")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize LLM rewrite chain: {e}")
    
    def should_rewrite(self, relevance_score: float) -> bool:
        """
        Determine if query should be rewritten based on relevance score.
        
        Args:
            relevance_score: Average relevance score of retrieved documents (0.0-1.0)
            
        Returns:
            bool: True if query should be rewritten, False otherwise
        """
        # Don't rewrite if we've already reached max attempts
        if self._attempt_count >= self.config.max_attempts:
            self.logger.info(
                f"Max rewrite attempts ({self.config.max_attempts}) reached, skipping rewrite"
            )
            return False
        
        # Rewrite if relevance score is below threshold
        should_rewrite = relevance_score < self.config.relevance_threshold
        
        if should_rewrite:
            self.logger.info(
                f"Relevance score {relevance_score:.3f} below threshold "
                f"{self.config.relevance_threshold:.3f}, triggering rewrite"
            )
        else:
            self.logger.info(
                f"Relevance score {relevance_score:.3f} above threshold "
                f"{self.config.relevance_threshold:.3f}, no rewrite needed"
            )
        
        return should_rewrite
    
    @log_retrieval_operation("query_rewriting")
    def rewrite(
        self, 
        query: str, 
        context: Optional[str] = None
    ) -> RewriteResult:
        """
        Generate improved version of query.
        
        Args:
            query: Original query to rewrite
            context: Optional context about why rewrite is needed or previous results
            
        Returns:
            RewriteResult containing original and rewritten queries
            
        Raises:
            RetrievalError: If rewriting fails
        """
        import time
        start_time = time.time()
        
        # Increment attempt counter
        self._attempt_count += 1
        
        if self.rewrite_chain is None:
            self.logger.warning("Rewrite chain not initialized, initializing now")
            self._initialize_chain()
        
        try:
            # Generate the rewritten query
            rewritten_query = self._generate_rewrite(query, context or "")
            
            # Determine improvement reason
            improvement_reason = self._analyze_improvement(query, rewritten_query)
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Log rewrite metrics
            self.logger.info(
                f"Query rewritten successfully (attempt {self._attempt_count})",
                extra={
                    'extra_fields': {
                        'original_query_length': len(query),
                        'rewritten_query_length': len(rewritten_query),
                        'attempt_number': self._attempt_count,
                        'processing_time_ms': processing_time_ms,
                        'improvement_reason': improvement_reason,
                        'original_query_hash': hash(query) % 10000,
                        'rewritten_query_hash': hash(rewritten_query) % 10000
                    }
                }
            )
            
            return RewriteResult(
                original_query=query,
                rewritten_query=rewritten_query,
                attempt_number=self._attempt_count,
                improvement_reason=improvement_reason,
                processing_time_ms=processing_time_ms
            )
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            self.logger.error(
                f"Query rewriting failed after {processing_time_ms:.2f}ms: {e}",
                extra={
                    'extra_fields': {
                        'attempt_number': self._attempt_count,
                        'processing_time_ms': processing_time_ms,
                        'error': str(e),
                        'original_query_hash': hash(query) % 10000
                    }
                },
                exc_info=True
            )
            
            # Return original query on failure to ensure system continues working
            return RewriteResult(
                original_query=query,
                rewritten_query=query,  # Fallback to original
                attempt_number=self._attempt_count,
                improvement_reason="Rewrite failed, using original query",
                processing_time_ms=processing_time_ms
            )
    
    def _generate_rewrite(self, query: str, context: str) -> str:
        """
        Use LLM to rewrite the query.
        
        Args:
            query: Original query to rewrite
            context: Context about why rewrite is needed
            
        Returns:
            str: Rewritten query
            
        Raises:
            RetrievalError: If LLM rewrite fails
        """
        if self.rewrite_chain is None:
            raise RetrievalError("LLM rewrite chain not initialized")
        
        try:
            # Use the LLM chain to generate rewrite
            rewritten = self.rewrite_chain.invoke({
                "original_query": query,
                "context": context
            })
            
            # Clean up the response
            rewritten = rewritten.strip()
            
            # Validate the rewrite
            if not rewritten or len(rewritten) < 3:
                self.logger.warning(
                    f"LLM produced very short rewrite: '{rewritten}', using original query"
                )
                return query
            
            # If the rewrite is identical to original, that's okay but log it
            if rewritten.lower() == query.lower():
                self.logger.info("LLM rewrite is identical to original query")
            
            return rewritten
            
        except Exception as e:
            raise RetrievalError(
                f"Failed to generate query rewrite: {e}",
                details={
                    "original_query": query,
                    "context": context,
                    "attempt_number": self._attempt_count
                }
            ) from e
    
    def _analyze_improvement(self, original: str, rewritten: str) -> str:
        """
        Analyze what improvements were made in the rewrite.
        
        Args:
            original: Original query
            rewritten: Rewritten query
            
        Returns:
            str: Description of improvements made
        """
        # Simple heuristic analysis of improvements
        improvements = []
        
        # Check if query became more specific (longer)
        if len(rewritten) > len(original) * 1.2:
            improvements.append("added specificity")
        
        # Check if query became more concise (shorter but not too short)
        elif len(rewritten) < len(original) * 0.8 and len(rewritten) > len(original) * 0.5:
            improvements.append("improved conciseness")
        
        # Check for common product-related terms
        product_terms = ["phone", "smartphone", "mobile", "device", "product", "brand", "model", "price", "rating"]
        original_terms = set(original.lower().split())
        rewritten_terms = set(rewritten.lower().split())
        
        added_product_terms = [term for term in product_terms if term in rewritten_terms and term not in original_terms]
        if added_product_terms:
            improvements.append(f"added product terms: {', '.join(added_product_terms)}")
        
        # Check if question words were removed (making it more search-friendly)
        question_words = ["what", "how", "why", "when", "where", "which", "who"]
        removed_questions = [word for word in question_words if word in original.lower() and word not in rewritten.lower()]
        if removed_questions:
            improvements.append("converted from question to search terms")
        
        # Default improvement reason
        if not improvements:
            improvements.append("general query optimization")
        
        return "; ".join(improvements)
    
    def reset_attempts(self) -> None:
        """
        Reset the attempt counter.
        
        This should be called when starting a new query to reset the
        rewrite attempt counter.
        """
        self._attempt_count = 0
        self.logger.debug("Reset rewrite attempt counter")
    
    def get_attempt_count(self) -> int:
        """
        Get the current attempt count.
        
        Returns:
            int: Number of rewrite attempts made for current query
        """
        return self._attempt_count
    
    def get_rewriter_stats(self) -> dict:
        """
        Get rewriter statistics and configuration.
        
        Returns:
            Dictionary with rewriter statistics and settings
        """
        return {
            "max_attempts": self.config.max_attempts,
            "relevance_threshold": self.config.relevance_threshold,
            "llm_model": self.config.llm_model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "current_attempts": self._attempt_count,
            "has_custom_prompt": self.config.rewrite_prompt is not None,
            "rewriter_initialized": self.rewrite_chain is not None
        }