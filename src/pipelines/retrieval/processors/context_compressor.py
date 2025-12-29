"""
Context Compressor for the retrieval pipeline.

This module implements contextual compression using LLM-based filtering to evaluate
document relevance and filter out irrelevant documents from retrieval results.
It integrates with LangChain's document compression framework.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ..config import RetrievalSettings
from ..exceptions import RetrievalError, ConfigurationError
from ..logging import RetrievalLoggerMixin, log_retrieval_operation


@dataclass
class CompressorConfig:
    """Configuration for contextual compression."""
    
    enabled: bool = True # Turn on/off
    relevance_prompt: Optional[str] = None # Your own evaluation prompt
    llm_model: str = "gpt-3.5-turbo" # Which LLM to use
    temperature: float = 0.0 # Deterministic responses
    max_tokens: Optional[int] = None
    relevance_threshold: float = 0.5


@dataclass
class CompressionResult:
    """Result of contextual compression containing filtered documents and metrics."""
    
    documents: List[Document]
    filtered_count: int
    relevance_scores: List[float]
    compression_ratio: float
    processing_time_ms: float


class ContextCompressor(RetrievalLoggerMixin):
    """
    Filters documents for relevance using LLM-based evaluation.
    
    The ContextCompressor uses LangChain's LLMChainFilter to evaluate each document's
    relevance to the query and filters out irrelevant documents. This helps ensure
    that only truly relevant context is passed to the inference pipeline.
    """
    
    DEFAULT_RELEVANCE_PROMPT = """
    Given the following question and document, determine if the document is relevant 
    to answering the question. Consider the document relevant if it contains information 
    that could help answer the question, even if it doesn't contain the complete answer.
    
    Question: {question}
    
    Document: {context}
    
    Is this document relevant? Answer with 'Yes' if relevant, 'No' if not relevant.
    """
    
    def __init__(self, config: CompressorConfig, openai_api_key: str):
        """
        Initialize the ContextCompressor with configuration.
        
        Args:
            config: CompressorConfig containing compression settings
            openai_api_key: OpenAI API key for LLM access
            
        Raises:
            ConfigurationError: If API key is missing or LLM initialization fails
        """
        super().__init__()
        self.config = config
        self.llm: Optional[ChatOpenAI] = None
        self.relevance_chain = None
        
        if not config.enabled:
            self.logger.info("Context compression is disabled")
            return
        
        if not openai_api_key:
            raise ConfigurationError(
                "OpenAI API key is required for context compression. "
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
            
            self.logger.info(
                f"Successfully initialized ContextCompressor with model: {config.llm_model}"
            )
            
        except Exception as e:
            if "api" in str(e).lower() or "key" in str(e).lower():
                raise ConfigurationError(f"Invalid OpenAI API key or API error: {e}")
            else:
                raise ConfigurationError(f"Failed to initialize LLM for compression: {e}")
    
    @classmethod
    def from_settings(cls, settings: RetrievalSettings) -> 'ContextCompressor':
        """
        Create ContextCompressor from RetrievalSettings.
        
        Args:
            settings: RetrievalSettings instance
            
        Returns:
            ContextCompressor instance
        """
        config = CompressorConfig(
            enabled=settings.compression_enabled,
            relevance_prompt=settings.relevance_prompt,
            llm_model="gpt-3.5-turbo",  # Use a cost-effective model for filtering
            temperature=0.0,  # Deterministic responses for filtering
            max_tokens=10  # Short responses for yes/no filtering
        )
        
        return cls(config, settings.openai_api_key)
    
    def initialize(self) -> None:
        """
        Initialize the LLM chain for document relevance evaluation.
        
        This method sets up a custom LangChain chain with the configured
        relevance prompt and LLM model for document filtering.
        
        Raises:
            ConfigurationError: If compression is enabled but LLM is not initialized
        """
        if not self.config.enabled:
            self.logger.info("Skipping compressor initialization - compression disabled")
            return
        
        if self.llm is None:
            raise ConfigurationError("LLM not initialized for context compression")
        
        try:
            # Use custom prompt if provided, otherwise use default
            prompt_template = self.config.relevance_prompt or self.DEFAULT_RELEVANCE_PROMPT
            
            # Create the prompt template
            prompt = PromptTemplate(
                input_variables=["question", "context"],
                template=prompt_template
            )
            
            # Create the chain: prompt -> llm -> output parser
            output_parser = StrOutputParser()
            self.relevance_chain = prompt | self.llm | output_parser
            
            self.logger.info("LLM relevance chain initialized successfully")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize LLM relevance chain: {e}")
    
    @log_retrieval_operation("context_compression")
    def compress(
        self, 
        query: str, 
        documents: List[Document]
    ) -> CompressionResult:
        """
        Filter documents based on relevance to query.
        
        Args:
            query: The user query to evaluate relevance against
            documents: List of documents to filter
            
        Returns:
            CompressionResult containing filtered documents and metrics
            
        Raises:
            RetrievalError: If compression fails
        """
        import time
        start_time = time.time()
        
        # If compression is disabled, return all documents
        if not self.config.enabled:
            self.logger.info("Context compression disabled, returning all documents")
            processing_time_ms = (time.time() - start_time) * 1000
            return CompressionResult(
                documents=documents,
                filtered_count=0,
                relevance_scores=[1.0] * len(documents),  # Assume all relevant
                compression_ratio=1.0,
                processing_time_ms=processing_time_ms
            )
        
        if not documents:
            self.logger.info("No documents to compress")
            processing_time_ms = (time.time() - start_time) * 1000
            return CompressionResult(
                documents=[],
                filtered_count=0,
                relevance_scores=[],
                compression_ratio=1.0,
                processing_time_ms=processing_time_ms
            )
        
        if self.relevance_chain is None:
            self.logger.warning("Relevance chain not initialized, initializing now")
            self.initialize()
        
        try:
            # Evaluate relevance for each document
            filtered_documents = []
            relevance_scores = []
            
            for doc in documents:
                try:
                    is_relevant, score = self._evaluate_relevance(query, doc)
                    relevance_scores.append(score)
                    
                    if is_relevant:
                        # Preserve all original metadata
                        filtered_doc = Document(
                            page_content=doc.page_content,
                            metadata=doc.metadata.copy()  # Preserve metadata
                        )
                        filtered_documents.append(filtered_doc)
                    
                except Exception as e:
                    # If evaluation fails for a document, include it to be safe
                    self.logger.warning(
                        f"Failed to evaluate document relevance, including document: {e}",
                        extra={
                            'extra_fields': {
                                'document_id': doc.metadata.get('id', 'unknown'),
                                'error': str(e)
                            }
                        }
                    )
                    filtered_documents.append(doc)
                    relevance_scores.append(0.5)  # Neutral score for failed evaluation
            
            filtered_count = len(documents) - len(filtered_documents)
            compression_ratio = len(filtered_documents) / len(documents) if documents else 1.0
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Log compression metrics
            self.logger.info(
                f"Compression completed: {len(documents)} -> {len(filtered_documents)} documents",
                extra={
                    'extra_fields': {
                        'input_documents': len(documents),
                        'output_documents': len(filtered_documents),
                        'filtered_count': filtered_count,
                        'compression_ratio': compression_ratio,
                        'processing_time_ms': processing_time_ms,
                        'avg_relevance_score': sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
                    }
                }
            )
            
            return CompressionResult(
                documents=filtered_documents,
                filtered_count=filtered_count,
                relevance_scores=relevance_scores,
                compression_ratio=compression_ratio,
                processing_time_ms=processing_time_ms
            )
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            self.logger.error(
                f"Context compression failed after {processing_time_ms:.2f}ms: {e}",
                extra={
                    'extra_fields': {
                        'input_documents': len(documents),
                        'processing_time_ms': processing_time_ms,
                        'error': str(e)
                    }
                },
                exc_info=True
            )
            
            # Return uncompressed documents on failure to ensure system continues working
            return CompressionResult(
                documents=documents,
                filtered_count=0,
                relevance_scores=[0.5] * len(documents),  # Neutral scores
                compression_ratio=1.0,
                processing_time_ms=processing_time_ms
            )
    
    def _evaluate_relevance(
        self, 
        query: str, 
        document: Document
    ) -> Tuple[bool, float]:
        """
        Evaluate if document is relevant to query using LLM.
        
        Args:
            query: The user query
            document: Document to evaluate
            
        Returns:
            Tuple of (is_relevant: bool, relevance_score: float)
            
        Raises:
            RetrievalError: If LLM evaluation fails
        """
        if self.relevance_chain is None:
            raise RetrievalError("LLM relevance chain not initialized")
        
        try:
            # Use the LLM chain to evaluate relevance
            response = self.relevance_chain.invoke({
                "question": query,
                "context": document.page_content
            })
            
            # Parse the response to determine relevance
            response_lower = response.lower().strip()
            
            # Look for clear yes/no indicators
            if "yes" in response_lower or "relevant" in response_lower:
                is_relevant = True
                relevance_score = 1.0
            elif "no" in response_lower or "not relevant" in response_lower or "irrelevant" in response_lower:
                is_relevant = False
                relevance_score = 0.0
            else:
                # If unclear, err on the side of inclusion
                self.logger.warning(
                    f"Unclear relevance response: '{response}', defaulting to relevant",
                    extra={
                        'extra_fields': {
                            'response': response,
                            'document_id': document.metadata.get('id', 'unknown')
                        }
                    }
                )
                is_relevant = True
                relevance_score = 0.5  # Neutral score for unclear responses
            
            return is_relevant, relevance_score
            
        except Exception as e:
            raise RetrievalError(
                f"Failed to evaluate document relevance: {e}",
                details={
                    "query": query,
                    "document_id": document.metadata.get('id', 'unknown'),
                    "document_length": len(document.page_content)
                }
            ) from e
    
    def batch_compress(
        self,
        query: str,
        document_batches: List[List[Document]]
    ) -> List[CompressionResult]:
        """
        Compress multiple batches of documents.
        
        Args:
            query: The user query
            document_batches: List of document batches to compress
            
        Returns:
            List of CompressionResult objects, one per batch
        """
        results = []
        
        for i, batch in enumerate(document_batches):
            self.logger.info(
                f"Compressing batch {i+1}/{len(document_batches)} ({len(batch)} documents)"
            )
            
            result = self.compress(query, batch)
            results.append(result)
        
        return results
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """
        Get compression statistics and configuration.
        
        Returns:
            Dictionary with compression statistics and settings
        """
        return {
            "enabled": self.config.enabled,
            "llm_model": self.config.llm_model if self.config.enabled else None,
            "temperature": self.config.temperature,
            "relevance_threshold": self.config.relevance_threshold,
            "has_custom_prompt": self.config.relevance_prompt is not None,
            "compressor_initialized": self.relevance_chain is not None
        }