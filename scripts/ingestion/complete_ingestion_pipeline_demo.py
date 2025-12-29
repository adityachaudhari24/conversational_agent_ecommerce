#!/usr/bin/env python3
"""
Complete Pipeline Demonstration

This script shows the complete end-to-end pipeline transformation:
CSV → Documents → Chunks → Mock Embeddings → Vector Store Format

This gives you the complete picture of what data flows through each stage
and what the final vector database entries would look like.

Run with: python scripts/complete_pipeline_demo.py
"""

import sys
from pathlib import Path
import pandas as pd
import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.json import JSON

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipelines.ingestion.loaders.document_loader import DocumentLoader, LoaderConfig
from pipelines.ingestion.processors.text_processor import TextProcessor, ProcessorConfig
from pipelines.ingestion.processors.text_chunker import TextChunker, ChunkerConfig

console = Console()

def mock_embedding_generation(documents):
    """
    Mock embedding generation to show what the final vector format would look like.
    In reality, this would call OpenAI API to generate 3072-dimensional vectors.
    """
    mock_vectors = []
    
    for i, doc in enumerate(documents):
        # Create a mock 3072-dimensional vector (normally from OpenAI)
        # Using a pattern for demo - real embeddings would be different
        mock_vector = [0.1 + (i * 0.01) + (j * 0.0001) for j in range(3072)]
        
        # Format as it would be stored in Pinecone
        vector_entry = {
            "id": f"doc_{i+1}",
            "values": mock_vector[:10] + ["..."] + [f"({len(mock_vector)} total dimensions)"],  # Show first 10 for demo
            "metadata": {
                **doc.metadata,
                "content_preview": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            }
        }
        
        mock_vectors.append(vector_entry)
    
    return mock_vectors

def main():
    """Demonstrate the complete pipeline."""
    console.print(Panel.fit(
        "[bold blue]Complete Data Ingestion Pipeline Demo[/bold blue]\n"
        "End-to-end transformation: CSV → Documents → Chunks → Vectors → Vector DB",
        border_style="blue"
    ))
    
    try:
        # Stage 1: Load sample data
        console.print("\n[bold]🔄 STAGE 1: Loading CSV Data[/bold]")
        
        loader_config = LoaderConfig(file_path=Path("data/phones_reviews.csv"))
        loader = DocumentLoader(loader_config)
        df = loader.load()
        
        # Take first 3 rows for demo
        sample_df = df.head(3)
        console.print(f"✅ Loaded {len(sample_df)} sample rows")
        
        # Stage 2: Process to Documents
        console.print("\n[bold]🔄 STAGE 2: Converting to Documents[/bold]")
        
        processor_config = ProcessorConfig()
        processor = TextProcessor(processor_config)
        documents = processor.process(sample_df)
        
        console.print(f"✅ Created {len(documents)} documents")
        
        # Stage 3: Chunk Documents
        console.print("\n[bold]🔄 STAGE 3: Chunking Documents[/bold]")
        
        chunker_config = ChunkerConfig(chunk_size=300, chunk_overlap=50)  # Smaller for demo
        chunker = TextChunker(chunker_config)
        chunks = chunker.chunk_documents(documents)
        
        console.print(f"✅ Created {len(chunks)} chunks")
        
        # Stage 4: Mock Embedding Generation
        console.print("\n[bold]🔄 STAGE 4: Generating Embeddings (Mocked)[/bold]")
        
        vectors = mock_embedding_generation(chunks)
        console.print(f"✅ Generated {len(vectors)} vector embeddings")
        
        # Show the complete transformation
        console.print("\n[bold]📊 COMPLETE DATA TRANSFORMATION RESULTS[/bold]")
        
        # Summary table
        summary_table = Table(show_header=True, header_style="bold green")
        summary_table.add_column("Stage", style="cyan")
        summary_table.add_column("Input", style="yellow")
        summary_table.add_column("Output", style="green")
        summary_table.add_column("Data Structure", style="white")
        
        summary_table.add_row(
            "CSV Loading", 
            "Raw CSV file", 
            f"{len(sample_df)} rows", 
            "pandas DataFrame"
        )
        summary_table.add_row(
            "Text Processing", 
            f"{len(sample_df)} DataFrame rows", 
            f"{len(documents)} Documents", 
            "LangChain Document objects"
        )
        summary_table.add_row(
            "Text Chunking", 
            f"{len(documents)} Documents", 
            f"{len(chunks)} Chunks", 
            "Smaller Document objects"
        )
        summary_table.add_row(
            "Embedding Generation", 
            f"{len(chunks)} Chunks", 
            f"{len(vectors)} Vectors", 
            "3072-dimensional embeddings"
        )
        summary_table.add_row(
            "Vector Storage", 
            f"{len(vectors)} Vectors", 
            "Pinecone Index", 
            "Searchable vector database"
        )
        
        console.print(summary_table)
        
        # Show sample final vector entries
        console.print(f"\n[bold]🎯 FINAL VECTOR DATABASE ENTRIES[/bold]")
        console.print("This is what gets stored in Pinecone for semantic search:")
        
        for i, vector in enumerate(vectors[:2]):  # Show first 2 vectors
            console.print(f"\n[bold yellow]Vector Entry {i+1}:[/bold yellow]")
            
            # Show the vector structure
            vector_display = {
                "id": vector["id"],
                "values": vector["values"],  # Mock vector values
                "metadata": vector["metadata"]
            }
            
            console.print(JSON.from_data(vector_display, indent=2))
        
        if len(vectors) > 2:
            console.print(f"\n[dim]... and {len(vectors) - 2} more vector entries[/dim]")
        
        # Show search capabilities
        console.print(f"\n[bold]🔍 SEARCH CAPABILITIES[/bold]")
        
        search_table = Table(show_header=True, header_style="bold magenta")
        search_table.add_column("Search Type", style="cyan")
        search_table.add_column("How It Works", style="white")
        search_table.add_column("Example Query", style="green")
        
        search_table.add_row(
            "Semantic Search",
            "Vector similarity on review content",
            '"battery life issues"'
        )
        search_table.add_row(
            "Metadata Filtering",
            "Filter by product attributes",
            'product_name="iPhone" AND rating>=4.0'
        )
        search_table.add_row(
            "Hybrid Search",
            "Combine semantic + metadata filters",
            '"camera quality" + price<500'
        )
        
        console.print(search_table)
        
        # Final insights
        console.print(Panel(
            "[bold green]✅ Complete Pipeline Demonstration Finished![/bold green]\n\n"
            "[bold]Key Insights:[/bold]\n"
            f"• {len(sample_df)} CSV rows → {len(vectors)} searchable vectors\n"
            f"• Each vector has 3,072 dimensions (OpenAI text-embedding-3-large)\n"
            f"• All product metadata preserved for filtering\n"
            f"• Review content becomes semantically searchable\n"
            f"• Ready for real-time similarity search in production\n\n"
            "[bold]Next Steps:[/bold]\n"
            "• Run EmbeddingGenerator with real OpenAI API\n"
            "• Store vectors in Pinecone vector database\n"
            "• Build retrieval system for semantic search\n"
            "• Create conversational interface for users",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(f"[bold red]❌ Error: {e}[/bold red]")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())