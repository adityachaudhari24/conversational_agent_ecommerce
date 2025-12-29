#!/usr/bin/env python3
"""
Pipeline Visualization Script

This script demonstrates the complete data transformation pipeline by processing
the first 5 rows from the CSV file and showing the output at each stage:
1. CSV Loading
2. Text Processing (DataFrame → Documents)
3. Text Chunking (Document splitting)

Run with: python scripts/visualize_pipeline.py
"""

import sys
import os
from pathlib import Path
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.syntax import Syntax

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipelines.ingestion.loaders.document_loader import DocumentLoader, LoaderConfig
from pipelines.ingestion.processors.text_processor import TextProcessor, ProcessorConfig
from pipelines.ingestion.processors.text_chunker import TextChunker, ChunkerConfig

console = Console()

def print_stage_header(stage_num: int, title: str, description: str):
    """Print a formatted stage header."""
    console.print(f"\n{'='*80}")
    console.print(f"STAGE {stage_num}: {title}", style="bold blue")
    console.print(f"{'='*80}")
    console.print(description, style="italic")
    console.print()

def print_csv_data(df: pd.DataFrame):
    """Print CSV data in a nice table format."""
    console.print("[bold]Raw CSV Data (First 5 rows):[/bold]")
    
    table = Table(show_header=True, header_style="bold magenta")
    
    # Add columns
    for col in df.columns:
        table.add_column(col, style="cyan", max_width=30)
    
    # Add rows
    for _, row in df.head(5).iterrows():
        table.add_row(*[str(row[col])[:100] + "..." if len(str(row[col])) > 100 else str(row[col]) for col in df.columns])
    
    console.print(table)

def print_documents(documents, title: str):
    """Print documents in a detailed format."""
    console.print(f"\n[bold]{title}:[/bold]")
    console.print(f"Total Documents: {len(documents)}")
    
    for i, doc in enumerate(documents):
        console.print(f"\n[bold yellow]Document {i+1}:[/bold yellow]")
        
        # Create a panel for the document content
        content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        console.print(Panel(
            content_preview,
            title="Page Content (What gets embedded)",
            title_align="left",
            border_style="green"
        ))
        
        # Create a table for metadata
        metadata_table = Table(show_header=True, header_style="bold blue", title="Metadata (Stored with vector)")
        metadata_table.add_column("Field", style="cyan")
        metadata_table.add_column("Value", style="white")
        
        for key, value in doc.metadata.items():
            value_str = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
            metadata_table.add_row(key, value_str)
        
        console.print(metadata_table)
        
        if i >= 2:  # Show only first 3 documents to avoid too much output
            remaining = len(documents) - 3
            if remaining > 0:
                console.print(f"\n[dim]... and {remaining} more documents[/dim]")
            break

def print_chunking_analysis(original_docs, chunked_docs):
    """Print analysis of chunking results."""
    console.print(f"\n[bold]Chunking Analysis:[/bold]")
    
    analysis_table = Table(show_header=True, header_style="bold magenta")
    analysis_table.add_column("Metric", style="cyan")
    analysis_table.add_column("Before Chunking", style="yellow")
    analysis_table.add_column("After Chunking", style="green")
    
    analysis_table.add_row("Total Documents", str(len(original_docs)), str(len(chunked_docs)))
    
    # Calculate average lengths
    avg_length_before = sum(len(doc.page_content) for doc in original_docs) / len(original_docs) if original_docs else 0
    avg_length_after = sum(len(doc.page_content) for doc in chunked_docs) / len(chunked_docs) if chunked_docs else 0
    
    analysis_table.add_row("Average Content Length", f"{avg_length_before:.0f} chars", f"{avg_length_after:.0f} chars")
    
    # Find documents that were chunked
    chunked_count = len(chunked_docs) - len(original_docs)
    analysis_table.add_row("Documents Split", "0", str(max(0, chunked_count)))
    
    console.print(analysis_table)

def main():
    """Main function to demonstrate the pipeline."""
    console.print(Panel.fit(
        "[bold blue]Data Ingestion Pipeline Visualization[/bold blue]\n"
        "This script shows how CSV data transforms through each pipeline stage",
        border_style="blue"
    ))
    
    try:
        # Stage 1: Load CSV Data
        print_stage_header(1, "CSV DATA LOADING", "Loading raw data from CSV file")
        
        loader_config = LoaderConfig(
            file_path=Path("data/phones_reviews.csv")
        )
        loader = DocumentLoader(loader_config)
        df = loader.load()
        
        console.print(f"✅ Loaded {len(df)} total rows from CSV")
        print_csv_data(df)
        
        # Stage 2: Text Processing (DataFrame → Documents)
        print_stage_header(2, "TEXT PROCESSING", "Converting CSV rows to LangChain Document objects")
        
        processor_config = ProcessorConfig()
        processor = TextProcessor(processor_config)
        
        # Process only first 5 rows for visualization
        sample_df = df.head(5)
        documents = processor.process(sample_df)
        
        console.print(f"✅ Processed {len(sample_df)} rows into {len(documents)} documents")
        
        # Show validation report
        report = processor.get_validation_report()
        if report['total_skipped'] > 0:
            console.print(f"⚠️  Skipped {report['total_skipped']} records:")
            for reason, count in report['skip_reasons'].items():
                console.print(f"   - {reason}: {count}")
        
        print_documents(documents, "Processed Documents")
        
        # Stage 3: Text Chunking
        print_stage_header(3, "TEXT CHUNKING", "Splitting large documents into smaller chunks for optimal embedding")
        
        chunker_config = ChunkerConfig(
            chunk_size=500,  # Smaller for demo to show chunking
            chunk_overlap=100
        )
        chunker = TextChunker(chunker_config)
        
        chunked_documents = chunker.chunk_documents(documents)
        
        console.print(f"✅ Chunked {len(documents)} documents into {len(chunked_documents)} chunks")
        
        print_chunking_analysis(documents, chunked_documents)
        print_documents(chunked_documents, "Final Chunked Documents (Ready for Embedding)")
        
        # Stage 4: Summary
        print_stage_header(4, "PIPELINE SUMMARY", "What happens next in the pipeline")
        
        summary_table = Table(show_header=True, header_style="bold green")
        summary_table.add_column("Stage", style="cyan")
        summary_table.add_column("Input", style="yellow")
        summary_table.add_column("Output", style="green")
        summary_table.add_column("Purpose", style="white")
        
        summary_table.add_row(
            "1. Loading", 
            "CSV file", 
            f"{len(df)} DataFrame rows", 
            "Read raw data"
        )
        summary_table.add_row(
            "2. Processing", 
            f"{len(sample_df)} DataFrame rows", 
            f"{len(documents)} Documents", 
            "Structure data for embedding"
        )
        summary_table.add_row(
            "3. Chunking", 
            f"{len(documents)} Documents", 
            f"{len(chunked_documents)} Chunks", 
            "Optimize for embedding size"
        )
        summary_table.add_row(
            "4. Embedding", 
            f"{len(chunked_documents)} Chunks", 
            f"{len(chunked_documents)} Vectors", 
            "Convert text to embeddings"
        )
        summary_table.add_row(
            "5. Storage", 
            f"{len(chunked_documents)} Vectors", 
            "Pinecone Index", 
            "Store for similarity search"
        )
        
        console.print(summary_table)
        
        console.print(Panel(
            "[bold green]✅ Pipeline visualization complete![/bold green]\n\n"
            "[bold]Key Insights:[/bold]\n"
            f"• Each review becomes a searchable vector embedding\n"
            f"• Product info (name, price, rating) stored as metadata\n"
            f"• Long reviews get split into {chunker_config.chunk_size}-char chunks\n"
            f"• Metadata preserved across all chunks\n"
            f"• Ready for semantic search + metadata filtering",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(f"[bold red]❌ Error: {e}[/bold red]")
        console.print(f"[dim]Make sure you're running from the project root directory[/dim]")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())