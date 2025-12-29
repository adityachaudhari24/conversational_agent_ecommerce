# Pipeline Visualization Scripts

This directory contains demonstration scripts that show how the data ingestion pipeline transforms data at each stage.

## Scripts Overview

### 1. `visualize_pipeline.py` - Basic Pipeline Flow
Shows the first 5 CSV rows going through the complete pipeline:
- CSV Loading → DataFrame
- Text Processing → Documents  
- Text Chunking → Optimized chunks

```bash
python scripts/visualize_pipeline.py
```

**Best for:** Understanding the basic data transformation flow

### 2. `chunking_demo.py` - Text Chunking Deep Dive
Demonstrates how long reviews get split into smaller chunks:
- Creates a sample long review (1,766 characters)
- Shows how it gets split into 5 chunks
- Preserves metadata across all chunks

```bash
python scripts/chunking_demo.py
```

**Best for:** Understanding how chunking works with long text

### 3. `complete_ingestion_pipeline_demo.py` - End-to-End Visualization
Shows the complete transformation including mock embeddings:
- CSV → Documents → Chunks → Vectors → Vector DB format
- Shows what the final Pinecone entries would look like
- Demonstrates search capabilities

```bash
python scripts/complete_pipeline_demo.py
```

**Best for:** Understanding the complete end-to-end data flow

## What You'll Learn

### Data Flow Understanding
- **Input:** CSV rows with product reviews
- **Processing:** Text cleaning, validation, chunking
- **Output:** Vector embeddings ready for semantic search

### Key Insights
- Only `review_text` gets embedded (becomes searchable vectors)
- All product info (`product_name`, `price`, `rating`, etc.) stored as metadata
- Long reviews get split into optimal-sized chunks
- Each chunk becomes a separate searchable vector
- Metadata is preserved across all chunks

### Search Capabilities
- **Semantic Search:** Find reviews by meaning/sentiment
- **Metadata Filtering:** Filter by product, price, rating
- **Hybrid Search:** Combine semantic similarity with filters

## Example Output

Each script shows:
- ✅ Step-by-step transformation
- 📊 Data statistics and analysis  
- 🎯 Final vector database format
- 🔍 Search capabilities explanation

## Requirements

All scripts use the existing pipeline components and require:
- Rich library (for beautiful terminal output)
- Existing pipeline modules (already implemented)
- Sample CSV data (`data/phones_reviews.csv`)

Run from the project root directory to ensure proper imports.