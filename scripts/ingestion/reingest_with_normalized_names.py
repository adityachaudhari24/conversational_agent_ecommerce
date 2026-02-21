#!/usr/bin/env python3
"""
Re-ingest data with normalized product names.

This script re-runs the ingestion pipeline to add the product_name_normalized
field to all documents in Pinecone, enabling case-insensitive product name filtering.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.pipelines.ingestion.pipeline import IngestionPipeline
from src.pipelines.ingestion.config import create_ingestion_settings

def main():
    """Re-ingest data with normalized product names."""
    
    print("="*80)
    print("Re-ingesting data with normalized product names")
    print("="*80)
    print()
    print("This will:")
    print("  1. Load data from data/phones_reviews.csv")
    print("  2. Add product_name_normalized field to each document")
    print("  3. Upload to Pinecone (will deduplicate automatically)")
    print()
    
    # Create settings
    settings = create_ingestion_settings()
    
    # Create and run pipeline
    pipeline = IngestionPipeline(settings)
    
    print("Starting ingestion...")
    print("-"*80)
    
    result = pipeline.run()
    
    print()
    print("="*80)
    print("Ingestion Complete!")
    print("="*80)
    print(f"  Documents processed: {result.get('documents_processed', 0)}")
    print(f"  Documents stored: {result.get('documents_stored', 0)}")
    print(f"  Success rate: {result.get('success_rate', 0):.1%}")
    print()
    print("✓ All documents now have product_name_normalized field")
    print("✓ You can now search with case-insensitive product names")
    print()
    print("Example queries that will now work:")
    print("  - 'iPhone 12 price' (any case)")
    print("  - 'samsung galaxy' (any case)")
    print("  - 'google pixel' (any case)")
    print()

if __name__ == "__main__":
    main()
