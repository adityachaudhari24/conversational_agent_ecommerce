#!/usr/bin/env python3
"""Simple example demonstrating RAG grounding configuration.

This example shows how to configure and use grounding to prevent
LLM hallucination in product recommendations.
"""

from src.pipelines.inference.generation.generator import ResponseGenerator
from src.pipelines.inference.grounding import GroundingConfig, GroundingStrategy
from src.pipelines.inference.config import GeneratorConfig
from src.pipelines.inference.llm.client import LLMClient, LLMConfig


def example_strict_grounding():
    """Example: Maximum grounding (recommended for e-commerce)."""
    
    print("="*80)
    print("EXAMPLE 1: Strict Grounding (Recommended)")
    print("="*80)
    
    # Configure strict grounding
    grounding_config = GroundingConfig(
        strict_mode=True,           # Use strict grounding prompts
        require_context=True,       # Require context for responses
        min_context_length=50,      # Minimum 50 chars of context
        enable_validation=True      # Validate responses
    )
    
    print("\nConfiguration:")
    print(f"  - Strict Mode: {grounding_config.strict_mode}")
    print(f"  - Require Context: {grounding_config.require_context}")
    print(f"  - Min Context Length: {grounding_config.min_context_length}")
    
    # Example: Query with good context
    print("\n" + "-"*80)
    print("Scenario A: Query with retrieved context")
    print("-"*80)
    
    query = "What's the best phone under $500?"
    context = """
    Product: Samsung Galaxy A54
    Price: $449.99
    Rating: 4.5/5
    Reviews: Great camera, long battery life
    
    Product: Google Pixel 7a
    Price: $499.00
    Rating: 4.6/5
    Reviews: Excellent camera, clean Android
    """
    
    # Build grounded prompt
    system_prompt = GroundingStrategy.build_grounded_system_prompt(
        context=context,
        query=query,
        strict_mode=True
    )
    
    print(f"\nQuery: {query}")
    print(f"Context Length: {len(context)} chars")
    print(f"\nSystem Prompt Preview:")
    print(system_prompt[:300] + "...")
    print("\n✓ LLM will ONLY use the provided context")
    print("✓ Will cite Samsung Galaxy A54 and Google Pixel 7a")
    print("✓ Will include actual prices and ratings")
    
    # Example: Query without context
    print("\n" + "-"*80)
    print("Scenario B: Query without context")
    print("-"*80)
    
    query = "What's the best phone under $500?"
    context = None  # No context retrieved
    
    # Check if context is sufficient
    is_sufficient, msg = GroundingStrategy.create_retrieval_quality_check(context)
    
    print(f"\nQuery: {query}")
    print(f"Context: None")
    print(f"Context Sufficient: {is_sufficient}")
    print(f"Reason: {msg}")
    
    if not is_sufficient:
        print(f"\n✓ Will return fallback message:")
        print(f"  '{grounding_config.fallback_message}'")
        print("\n✓ Will NOT make up product recommendations")


def example_flexible_grounding():
    """Example: Flexible grounding (for general questions)."""
    
    print("\n\n" + "="*80)
    print("EXAMPLE 2: Flexible Grounding (For General Questions)")
    print("="*80)
    
    # Configure flexible grounding
    grounding_config = GroundingConfig(
        strict_mode=False,          # Less strict
        require_context=False,      # Allow responses without context
        min_context_length=0,       # No minimum
        enable_validation=False     # No validation
    )
    
    print("\nConfiguration:")
    print(f"  - Strict Mode: {grounding_config.strict_mode}")
    print(f"  - Require Context: {grounding_config.require_context}")
    print(f"  - Min Context Length: {grounding_config.min_context_length}")
    
    print("\n⚠️  Warning: This configuration allows more flexibility")
    print("   but increases risk of hallucination for product queries.")
    print("\n✓ Use for: General questions, FAQs, non-product queries")
    print("✗ Don't use for: Product recommendations, pricing, specifications")


def example_context_validation():
    """Example: Context quality validation."""
    
    print("\n\n" + "="*80)
    print("EXAMPLE 3: Context Quality Validation")
    print("="*80)
    
    test_contexts = [
        ("Empty context", ""),
        ("Too short", "Product: iPhone"),
        ("No product info", "This is some random text without product information"),
        ("Good context", "Product: iPhone 15\nPrice: $799\nRating: 4.8/5\nReviews: Excellent camera..."),
    ]
    
    print("\nValidating different context qualities:\n")
    
    for name, context in test_contexts:
        is_sufficient, msg = GroundingStrategy.create_retrieval_quality_check(
            context,
            min_context_length=50
        )
        
        status = "✓ PASS" if is_sufficient else "✗ FAIL"
        print(f"{status} - {name}")
        print(f"     Length: {len(context)} chars")
        if msg:
            print(f"     Reason: {msg}")
        print()


def example_custom_fallback():
    """Example: Custom fallback messages."""
    
    print("\n" + "="*80)
    print("EXAMPLE 4: Custom Fallback Messages")
    print("="*80)
    
    # Default fallback
    default_config = GroundingConfig()
    print("\nDefault Fallback:")
    print(f"  '{default_config.fallback_message}'")
    
    # Custom fallback
    custom_config = GroundingConfig(
        fallback_message=(
            "Sorry, I couldn't find matching products in our catalog. "
            "Please browse our website or contact customer support at 1-800-PHONES."
        )
    )
    print("\nCustom Fallback:")
    print(f"  '{custom_config.fallback_message}'")
    
    print("\n✓ Customize fallback messages for your brand voice")


def main():
    """Run all examples."""
    
    print("\n" + "="*80)
    print("RAG GROUNDING EXAMPLES")
    print("="*80)
    print("\nThese examples demonstrate how to configure grounding to prevent")
    print("LLM hallucination in your e-commerce conversational agent.\n")
    
    # Run examples
    example_strict_grounding()
    example_flexible_grounding()
    example_context_validation()
    example_custom_fallback()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
For e-commerce product recommendations, use:

    grounding_config = GroundingConfig(
        strict_mode=True,
        require_context=True,
        min_context_length=50
    )

This ensures:
✓ LLM only uses retrieved product data
✓ No hallucinated products or prices
✓ Safe fallback when data unavailable
✓ Accurate, trustworthy responses

See docs/GROUNDING_GUIDE.md for full documentation.
""")


if __name__ == "__main__":
    main()
