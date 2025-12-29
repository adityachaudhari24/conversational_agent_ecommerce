"""
Document Formatter for the retrieval pipeline.

This module formats retrieved documents into structured context strings for the
inference pipeline. It provides configurable templates, handles missing metadata
gracefully, and supports context length limits.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document

from ..config import RetrievalSettings
from ..exceptions import RetrievalError
from ..logging import RetrievalLoggerMixin, log_retrieval_operation


@dataclass
class FormatterConfig:
    """Configuration for document formatting."""
    
    template: Optional[str] = None  # Custom format template
    delimiter: str = "\n\n---\n\n"
    include_scores: bool = False
    max_context_length: int = 4000
    truncate_on_limit: bool = True


@dataclass
class FormattedContext:
    """Result of document formatting containing formatted text and metadata."""
    
    text: str
    document_count: int
    truncated: bool = False
    total_length: int = 0
    template_used: str = ""


class DocumentFormatter(RetrievalLoggerMixin):
    """
    Formats documents into structured context string.
    
    The DocumentFormatter takes a list of retrieved documents and formats them
    into a structured string suitable for the inference pipeline. It handles
    missing metadata gracefully, supports configurable templates, and can
    truncate output to stay within context limits.
    """
    
    DEFAULT_TEMPLATE = """Title: {product_name}
Price: {price}
Rating: {rating}
Review: {review_text}"""
    
    def __init__(self, config: FormatterConfig):
        """
        Initialize the DocumentFormatter with configuration.
        
        Args:
            config: FormatterConfig containing formatting settings
        """
        super().__init__()
        self.config = config
        
        # Use custom template if provided, otherwise use default
        self.template = config.template or self.DEFAULT_TEMPLATE
        
        self.logger.info(
            f"DocumentFormatter initialized with template length: {len(self.template)} chars",
            extra={
                'extra_fields': {
                    'template_length': len(self.template),
                    'delimiter_length': len(config.delimiter),
                    'max_context_length': config.max_context_length,
                    'include_scores': config.include_scores,
                    'truncate_on_limit': config.truncate_on_limit
                }
            }
        )
    
    @classmethod
    def from_settings(cls, settings: RetrievalSettings) -> 'DocumentFormatter':
        """
        Create DocumentFormatter from RetrievalSettings.
        
        Args:
            settings: RetrievalSettings instance
            
        Returns:
            DocumentFormatter instance
        """
        config = FormatterConfig(
            template=getattr(settings, 'format_template', None),
            delimiter=getattr(settings, 'format_delimiter', "\n\n---\n\n"),
            include_scores=getattr(settings, 'include_scores_in_format', False),
            max_context_length=getattr(settings, 'max_context_length', 4000),
            truncate_on_limit=getattr(settings, 'truncate_on_limit', True)
        )
        
        return cls(config)
    
    @log_retrieval_operation("document_formatting")
    def format(
        self, 
        documents: List[Document],
        scores: Optional[List[float]] = None
    ) -> FormattedContext:
        """
        Format documents into context string.
        
        Args:
            documents: List of documents to format
            scores: Optional similarity scores for each document
            
        Returns:
            FormattedContext containing formatted text and metadata
            
        Raises:
            RetrievalError: If formatting fails
        """
        if not documents:
            self.logger.info("No documents to format, returning empty context")
            return FormattedContext(
                text="",
                document_count=0,
                truncated=False,
                total_length=0,
                template_used=self.template
            )
        
        # Validate scores length if provided
        if scores is not None and len(scores) != len(documents):
            self.logger.warning(
                f"Scores length ({len(scores)}) doesn't match documents length ({len(documents)}), ignoring scores"
            )
            scores = None
        
        try:
            formatted_parts = []
            total_length = 0
            truncated = False
            
            for i, document in enumerate(documents):
                # Get score for this document if available
                score = scores[i] if scores is not None else None
                
                # Format the single document
                formatted_doc = self._format_single(document, score)
                
                # Check if adding this document would exceed the limit
                delimiter_length = len(self.config.delimiter) if formatted_parts else 0
                new_length = total_length + delimiter_length + len(formatted_doc)
                
                if (self.config.truncate_on_limit and 
                    new_length > self.config.max_context_length and 
                    formatted_parts):  # Don't truncate if this is the first document
                    
                    self.logger.info(
                        f"Truncating at document {i+1}/{len(documents)} to stay within {self.config.max_context_length} chars",
                        extra={
                            'extra_fields': {
                                'documents_included': len(formatted_parts),
                                'documents_total': len(documents),
                                'current_length': total_length,
                                'would_be_length': new_length,
                                'max_length': self.config.max_context_length
                            }
                        }
                    )
                    truncated = True
                    break
                
                formatted_parts.append(formatted_doc)
                total_length = new_length
            
            # Join all formatted documents with delimiter
            formatted_text = self.config.delimiter.join(formatted_parts)
            
            # Log formatting metrics
            self.logger.info(
                f"Formatted {len(formatted_parts)} documents into {len(formatted_text)} characters",
                extra={
                    'extra_fields': {
                        'input_documents': len(documents),
                        'output_documents': len(formatted_parts),
                        'output_length': len(formatted_text),
                        'truncated': truncated,
                        'delimiter_count': len(formatted_parts) - 1 if formatted_parts else 0,
                        'avg_doc_length': len(formatted_text) / len(formatted_parts) if formatted_parts else 0
                    }
                }
            )
            
            return FormattedContext(
                text=formatted_text,
                document_count=len(formatted_parts),
                truncated=truncated,
                total_length=len(formatted_text),
                template_used=self.template
            )
            
        except Exception as e:
            self.logger.error(
                f"Document formatting failed: {e}",
                extra={
                    'extra_fields': {
                        'input_documents': len(documents),
                        'error': str(e),
                        'template_length': len(self.template)
                    }
                },
                exc_info=True
            )
            
            raise RetrievalError(
                f"Failed to format documents: {e}",
                details={
                    "document_count": len(documents),
                    "template": self.template,
                    "config": {
                        "max_context_length": self.config.max_context_length,
                        "include_scores": self.config.include_scores,
                        "truncate_on_limit": self.config.truncate_on_limit
                    }
                }
            ) from e
    
    def _format_single(
        self, 
        document: Document, 
        score: Optional[float] = None
    ) -> str:
        """
        Format a single document using the configured template.
        
        Args:
            document: Document to format
            score: Optional similarity score for the document
            
        Returns:
            str: Formatted document string
        """
        try:
            # Extract metadata values with fallbacks
            format_values = {
                'product_name': self._get_metadata_value(document, 'product_name'),
                'price': self._get_metadata_value(document, 'price'),
                'rating': self._get_metadata_value(document, 'rating'),
                'review_text': document.page_content or "N/A"
            }
            
            # Add score if requested and available
            if self.config.include_scores and score is not None:
                format_values['score'] = f"{score:.3f}"
            
            # Format using the template
            formatted = self.template.format(**format_values)
            
            return formatted
            
        except KeyError as e:
            # Handle missing template variables
            missing_key = str(e).strip("'\"")
            self.logger.warning(
                f"Template variable '{missing_key}' not found in document metadata",
                extra={
                    'extra_fields': {
                        'missing_key': missing_key,
                        'available_keys': list(document.metadata.keys()),
                        'document_id': document.metadata.get('id', 'unknown')
                    }
                }
            )
            
            # Return a fallback format
            return f"Document (ID: {document.metadata.get('id', 'unknown')}): {document.page_content[:200]}..."
            
        except Exception as e:
            self.logger.error(
                f"Failed to format single document: {e}",
                extra={
                    'extra_fields': {
                        'document_id': document.metadata.get('id', 'unknown'),
                        'metadata_keys': list(document.metadata.keys()),
                        'content_length': len(document.page_content) if document.page_content else 0,
                        'error': str(e)
                    }
                }
            )
            
            # Return minimal fallback
            return f"Error formatting document: {str(e)}"
    
    def _get_metadata_value(
        self, 
        document: Document, 
        key: str
    ) -> str:
        """
        Get metadata value with N/A fallback for missing fields.
        
        Args:
            document: Document to extract metadata from
            key: Metadata key to retrieve
            
        Returns:
            str: Metadata value or "N/A" if missing
        """
        if not document.metadata:
            return "N/A"
        
        value = document.metadata.get(key)
        
        if value is None:
            return "N/A"
        
        # Convert to string and handle empty values
        str_value = str(value).strip()
        if not str_value:
            return "N/A"
        
        return str_value
    
    def format_with_custom_template(
        self,
        documents: List[Document],
        template: str,
        scores: Optional[List[float]] = None
    ) -> FormattedContext:
        """
        Format documents using a custom template (one-time use).
        
        Args:
            documents: List of documents to format
            template: Custom template string to use
            scores: Optional similarity scores for each document
            
        Returns:
            FormattedContext containing formatted text and metadata
        """
        # Temporarily store the original template
        original_template = self.template
        
        try:
            # Use the custom template
            self.template = template
            
            # Format with the custom template
            result = self.format(documents, scores)
            
            # Update the template_used field
            result.template_used = template
            
            return result
            
        finally:
            # Restore the original template
            self.template = original_template
    
    def get_template_variables(self) -> List[str]:
        """
        Extract template variables from the current template.
        
        Returns:
            List of template variable names
        """
        import re
        
        # Find all {variable} patterns in the template
        variables = re.findall(r'\{(\w+)\}', self.template)
        
        return list(set(variables))  # Remove duplicates
    
    def validate_template(self, template: str) -> Dict[str, Any]:
        """
        Validate a template string and return validation results.
        
        Args:
            template: Template string to validate
            
        Returns:
            Dict containing validation results
        """
        try:
            # Extract variables from template
            variables = re.findall(r'\{(\w+)\}', template)
            
            # Test format with dummy data
            test_data = {var: f"test_{var}" for var in variables}
            test_result = template.format(**test_data)
            
            return {
                "valid": True,
                "variables": list(set(variables)),
                "variable_count": len(set(variables)),
                "test_length": len(test_result),
                "error": None
            }
            
        except Exception as e:
            return {
                "valid": False,
                "variables": [],
                "variable_count": 0,
                "test_length": 0,
                "error": str(e)
            }
    
    def get_formatting_stats(self) -> Dict[str, Any]:
        """
        Get formatting statistics and configuration.
        
        Returns:
            Dictionary with formatting statistics and settings
        """
        template_validation = self.validate_template(self.template)
        
        return {
            "template_length": len(self.template),
            "template_variables": template_validation["variables"],
            "delimiter": self.config.delimiter,
            "delimiter_length": len(self.config.delimiter),
            "max_context_length": self.config.max_context_length,
            "include_scores": self.config.include_scores,
            "truncate_on_limit": self.config.truncate_on_limit,
            "template_valid": template_validation["valid"],
            "template_error": template_validation.get("error")
        }