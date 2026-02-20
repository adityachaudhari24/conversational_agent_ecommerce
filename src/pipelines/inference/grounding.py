"""Grounding strategies to prevent LLM hallucination.

This module provides utilities to ensure LLM responses are strictly grounded
in retrieved context, preventing the model from making up product information
or recommendations.
"""

from typing import Optional


class GroundingStrategy:
    """Strategies for ensuring LLM responses are grounded in context."""
    
    # Strict grounding system prompt
    STRICT_GROUNDING_PROMPT = """You are a helpful e-commerce assistant specializing in phone products.

CRITICAL RULES - YOU MUST FOLLOW THESE STRICTLY:
1. ONLY recommend or mention products that are explicitly provided in the CONTEXT section below
2. NEVER make up product names, prices, ratings, or specifications
3. When recommending products, ALWAYS cite specific details from the context (name, price, rating)
4. Do NOT use your general knowledge about phones - ONLY use the provided context

CONTEXT:
{context}

IMPORTANT: 
- If the CONTEXT above contains relevant products, provide helpful recommendations based on that information
- If the CONTEXT is empty or doesn't contain relevant products, say: "I don't have information about products matching your query in my database."
- Do NOT say "I don't have information" if the context actually contains relevant product details

Remember: Use the context when available, admit when you don't have information."""

    FALLBACK_NO_CONTEXT_PROMPT = """You are a helpful e-commerce assistant.

The user asked: {query}

However, no relevant products were found in the database for this query.

Respond politely by:
1. Acknowledging their question
2. Explaining that you don't have matching products in the current database
3. Suggesting they rephrase their query or ask about different products
4. DO NOT recommend any specific products or make up information

Keep your response brief and helpful."""

    @staticmethod
    def build_grounded_system_prompt(
        context: Optional[str],
        query: str,
        strict_mode: bool = True
    ) -> str:
        """Build a system prompt that enforces grounding.
        
        Args:
            context: Retrieved context from RAG pipeline
            query: User query
            strict_mode: If True, use strict grounding rules
            
        Returns:
            System prompt string with grounding instructions
        """
        if not context or not context.strip():
            # No context available - use fallback prompt
            return GroundingStrategy.FALLBACK_NO_CONTEXT_PROMPT.format(query=query)
        
        if strict_mode:
            return GroundingStrategy.STRICT_GROUNDING_PROMPT.format(context=context)
        else:
            # Less strict mode - still emphasize context but allow some flexibility
            return f"""You are a helpful e-commerce assistant specializing in phone products.

Use the provided context to answer questions accurately. Prioritize information from the context.

CONTEXT:
{context}

Instructions:
- Base your recommendations primarily on the context provided
- If recommending products, include price and rating from the context
- Be conversational and helpful
- If the context doesn't fully answer the question, acknowledge the limitation
- Prefer saying "I don't have that information" over guessing"""

    @staticmethod
    def validate_response_grounding(
        response: str,
        context: Optional[str],
        strict_mode: bool = True
    ) -> tuple[bool, Optional[str]]:
        """Validate that a response appears to be grounded in context.
        
        This is a heuristic check - not foolproof but catches obvious issues.
        
        Args:
            response: Generated LLM response
            context: Context that was provided
            strict_mode: If True, apply stricter validation
            
        Returns:
            Tuple of (is_valid, warning_message)
        """
        if not context or not context.strip():
            # No context was provided
            if strict_mode:
                # Check if response makes specific product claims
                suspicious_patterns = [
                    "i recommend",
                    "you should buy",
                    "the best phone",
                    "top rated",
                    "price is",
                    "costs $",
                    "rating of"
                ]
                response_lower = response.lower()
                
                for pattern in suspicious_patterns:
                    if pattern in response_lower:
                        return False, f"Response makes specific claims without context: '{pattern}'"
            
            return True, None
        
        # Context exists - basic validation
        # This is a simple heuristic and can be enhanced
        return True, None

    @staticmethod
    def add_citation_instruction(base_prompt: str) -> str:
        """Add citation requirements to a prompt.
        
        Args:
            base_prompt: Base system prompt
            
        Returns:
            Enhanced prompt with citation instructions
        """
        citation_instruction = """

CITATION REQUIREMENT:
When mentioning specific products, always include:
- Product name (exactly as shown in context)
- Price (if available)
- Rating (if available)

Example format: "I recommend the [Product Name] priced at $[price] with a [rating] rating."
"""
        return base_prompt + citation_instruction

    @staticmethod
    def create_retrieval_quality_check(
        context: Optional[str],
        min_context_length: int = 50
    ) -> tuple[bool, Optional[str]]:
        """Check if retrieved context is sufficient for grounded responses.
        
        Args:
            context: Retrieved context
            min_context_length: Minimum acceptable context length
            
        Returns:
            Tuple of (is_sufficient, message)
        """
        if not context:
            return False, "No context retrieved"
        
        if len(context.strip()) < min_context_length:
            return False, f"Context too short ({len(context)} chars)"
        
        # Check if context contains product-like information
        # This is a heuristic - adjust based on your data format
        has_product_indicators = any(
            indicator in context.lower()
            for indicator in ["price", "rating", "review", "$", "product"]
        )
        
        if not has_product_indicators:
            return False, "Context doesn't appear to contain product information"
        
        return True, None


class GroundingConfig:
    """Configuration for grounding behavior."""
    
    def __init__(
        self,
        strict_mode: bool = True,
        require_context: bool = True,
        min_context_length: int = 50,
        enable_validation: bool = True,
        fallback_message: Optional[str] = None
    ):
        """Initialize grounding configuration.
        
        Args:
            strict_mode: Use strict grounding rules
            require_context: Refuse to answer without context
            min_context_length: Minimum context length required
            enable_validation: Enable response validation
            fallback_message: Custom message when no context available
        """
        self.strict_mode = strict_mode
        self.require_context = require_context
        self.min_context_length = min_context_length
        self.enable_validation = enable_validation
        self.fallback_message = fallback_message or (
            "I don't have relevant product information in my database to answer your question. "
            "Could you try rephrasing or asking about different products?"
        )
