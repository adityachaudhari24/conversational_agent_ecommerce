"""
Metadata Extractor for automatic filter extraction from queries.

This module provides functionality to extract structured metadata filters
(product names, price ranges, ratings) from natural language queries using LLMs.
"""

import json
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

from openai import OpenAI, APITimeoutError

from ..search.vector_searcher import MetadataFilter
from ..logging import RetrievalLoggerMixin


@dataclass
class ExtractorConfig:
    """Configuration for metadata extraction."""
    
    llm_model: str = "gpt-3.5-turbo"
    temperature: float = 0.0
    max_tokens: int = 200
    enabled: bool = True
    timeout_seconds: int = 3


class MetadataExtractor(RetrievalLoggerMixin):
    """
    Extracts structured metadata filters from natural language queries.
    
    Uses an LLM to identify product names, price ranges, and rating requirements
    from user queries and converts them into MetadataFilter objects for database filtering.
    """
    
    def __init__(self, config: ExtractorConfig, api_key: str):
        """
        Initialize the MetadataExtractor.
        
        Args:
            config: ExtractorConfig containing extraction settings
            api_key: OpenAI API key for LLM calls
        """
        super().__init__()
        self.config = config
        self.client = OpenAI(api_key=api_key)
        
        # Extraction prompt template
        self.extraction_prompt = """Extract structured metadata from the following query.
Return a JSON object with these fields (use null if not present):
- product_name_pattern: string (product name or brand mentioned)
- min_price: number (minimum price in dollars)
- max_price: number (maximum price in dollars)
- min_rating: number (minimum rating, 0-5 scale)

Examples:
Query: "iPhone 12 price"
Output: {{"product_name_pattern": "iPhone 12", "min_price": null, "max_price": null, "min_rating": null}}

Query: "phones under $300"
Output: {{"product_name_pattern": null, "min_price": null, "max_price": 300, "min_rating": null}}

Query: "highly rated Samsung phones over $400"
Output: {{"product_name_pattern": "Samsung", "min_price": 400, "max_price": null, "min_rating": 4.0}}

Query: "{query}"
Output:"""
    
    def extract(self, query: str) -> Optional[MetadataFilter]:
        """
        Extract metadata filters from a query.
        
        Args:
            query: User query string
            
        Returns:
            MetadataFilter object if extraction succeeds, None otherwise
        """
        if not self.config.enabled:
            return None
        
        content = None
        try:
            # Call LLM to extract metadata with timeout
            prompt = self.extraction_prompt.format(query=query)
            
            response = self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": "You are a metadata extraction assistant. Extract structured data from queries and return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout_seconds
            )
            
            # Parse response
            content = response.choices[0].message.content.strip()
            self.logger.debug(f"LLM response content: {content}")
            
            # Try to extract JSON if wrapped in markdown code blocks
            if content.startswith("```"):
                # Remove markdown code blocks
                lines = content.split("\n")
                content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
                content = content.replace("```json", "").replace("```", "").strip()
            
            metadata_dict = json.loads(content)
            
            # Convert to MetadataFilter
            filters = MetadataFilter(
                product_name_pattern=metadata_dict.get("product_name_pattern"),
                min_price=metadata_dict.get("min_price"),
                max_price=metadata_dict.get("max_price"),
                min_rating=metadata_dict.get("min_rating")
            )
            
            # Normalize product name pattern for case-insensitive matching
            if filters.product_name_pattern:
                filters.product_name_pattern = filters.product_name_pattern.lower().strip()
            
            # Only return filters if at least one field is set
            if any([
                filters.product_name_pattern,
                filters.min_price is not None,
                filters.max_price is not None,
                filters.min_rating is not None
            ]):
                return filters
            
            return None
            
        except APITimeoutError:
            self.logger.warning(
                f"Metadata extraction timed out after {self.config.timeout_seconds}s"
            )
            return None
        except TimeoutError:
            self.logger.warning(
                f"Metadata extraction timed out after {self.config.timeout_seconds}s"
            )
            return None
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse extraction response: {e}. Content: {content}")
            return None
        except Exception as e:
            self.logger.warning(f"Metadata extraction failed: {e}. Content: {content}")
            return None
