"""
Text Processor for transforming CSV data into LangChain Documents.

This module provides functionality to convert raw DataFrame rows into
structured Document objects suitable for embedding and vector storage.
Handles data cleaning, validation, and metadata sanitization.
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import pandas as pd
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class ProcessorConfig:
    """Configuration for the TextProcessor."""
    
    content_field: str = "review_text"
    metadata_fields: Optional[List[str]] = None
    
    def __post_init__(self):
        """Set default metadata fields if not provided."""
        if self.metadata_fields is None:
            self.metadata_fields = [
                "product_name", 
                "description", 
                "price", 
                "rating", 
                "review_title"
            ]


class TextProcessor:
    """
    Transforms DataFrame rows into LangChain Document objects.
    
    The TextProcessor is responsible for:
    - Converting DataFrame rows to Document objects
    - Sanitizing metadata by replacing NaN values with defaults
    - Validating content to ensure it's not empty or whitespace-only
    - Tracking skipped records for validation reporting
    - Providing detailed validation reports
    """
    
    def __init__(self, config: ProcessorConfig):
        """
        Initialize the TextProcessor with configuration.
        
        Args:
            config: ProcessorConfig containing field mappings and settings
        """
        self.config = config
        self.skipped_records: List[Dict[str, Any]] = []
        logger.info(f"Initialized TextProcessor with content field: {config.content_field}")
    
    def process(self, df: pd.DataFrame) -> List[Document]:
        """
        Convert DataFrame to list of Document objects.
        
        Args:
            df: DataFrame containing product review data
            
        Returns:
            List[Document]: List of valid Document objects ready for embedding
        """
        logger.info(f"Processing {len(df)} rows into Document objects")
        
        # Reset skipped records for this processing run
        self.skipped_records = []
        documents = []
        
        for index, row in df.iterrows():
            try:
                # Check if content is valid before creating document
                content = str(row.get(self.config.content_field, "")).strip()
                
                if not self._is_valid_content(content):
                    self._record_skipped(index, row, "Empty or whitespace-only content")
                    continue
                
                # Create document if content is valid
                document = self._create_document(row)
                documents.append(document)
                
            except Exception as e:
                logger.warning(f"Failed to process row {index}: {e}")
                self._record_skipped(index, row, f"Processing error: {e}")
                continue
        
        logger.info(f"Successfully processed {len(documents)} documents, skipped {len(self.skipped_records)} records")
        return documents
    
    def _create_document(self, row: pd.Series) -> Document:
        """
        Create a single Document from a DataFrame row.
        
        Builds page_content by prepending product name, price, and rating
        to the review text so the embedding captures product identity.
        
        Args:
            row: DataFrame row containing product data
            
        Returns:
            Document: LangChain Document with content and metadata
        """
        # Extract review text
        review_text = str(row.get(self.config.content_field, "")).strip()
        
        # Build enriched content with product context for better embeddings
        content = self._build_enriched_content(row, review_text)
        
        # Sanitize metadata
        metadata = self._sanitize_metadata(row)
        
        # Create and return Document
        document = Document(
            page_content=content,
            metadata=metadata
        )
        
        return document
    
    def _build_enriched_content(self, row: pd.Series, review_text: str) -> str:
        """
        Build enriched page_content by prepending product metadata to review text.
        
        This ensures the embedding captures product identity (name, price, rating)
        alongside the review, improving retrieval for product-specific queries.
        
        Args:
            row: DataFrame row containing product data
            review_text: The review text content
            
        Returns:
            str: Enriched content string
        """
        parts = []
        
        product_name = row.get("product_name")
        if pd.notna(product_name) and str(product_name).strip():
            parts.append(f"Product: {str(product_name).strip()}")
        
        price = row.get("price")
        if pd.notna(price):
            parts.append(f"Price: ${price}")
        
        rating = row.get("rating")
        if pd.notna(rating):
            parts.append(f"Rating: {rating}/5")
        
        parts.append(f"Review: {review_text}")
        
        return " | ".join(parts)
    
    def _sanitize_metadata(self, row: pd.Series) -> Dict[str, Any]:
        """
        Replace NaN values with appropriate defaults in metadata.
        
        Args:
            row: DataFrame row to extract metadata from
            
        Returns:
            Dict[str, Any]: Sanitized metadata dictionary
        """
        metadata = {}
        
        for field in self.config.metadata_fields:
            value = row.get(field)
            
            # Handle NaN/None values
            if pd.isna(value):
                # Use appropriate defaults based on field type
                if field in ["price", "rating"]:
                    # Numeric fields get "N/A" as string for consistency
                    metadata[field] = "N/A"
                else:
                    # Text fields get empty string
                    metadata[field] = ""
            else:
                # Convert to string and strip whitespace
                metadata[field] = str(value).strip()
        
        return metadata
    
    def _is_valid_content(self, content: str) -> bool:
        """
        Check if content is non-empty and non-whitespace.
        
        Args:
            content: Content string to validate
            
        Returns:
            bool: True if content is valid, False otherwise
        """
        # Content is invalid if it's empty or contains only whitespace
        return bool(content and content.strip())
    
    def _record_skipped(self, index: int, row: pd.Series, reason: str) -> None:
        """
        Record a skipped record for validation reporting.
        
        Args:
            index: DataFrame index of the skipped row
            row: The skipped DataFrame row
            reason: Reason why the record was skipped
        """
        skipped_record = {
            "index": index,
            "reason": reason,
            "data": {
                field: row.get(field) 
                for field in [self.config.content_field] + self.config.metadata_fields
                if field in row.index
            }
        }
        self.skipped_records.append(skipped_record)
        logger.debug(f"Skipped record at index {index}: {reason}")
    
    def get_validation_report(self) -> Dict[str, Any]:
        """
        Return summary of skipped records and validation statistics.
        
        Returns:
            Dict[str, Any]: Validation report with statistics and details
        """
        # Count reasons for skipping
        reason_counts = {}
        for record in self.skipped_records:
            reason = record["reason"]
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        report = {
            "total_skipped": len(self.skipped_records),
            "skip_reasons": reason_counts,
            "skipped_records": self.skipped_records
        }
        
        return report