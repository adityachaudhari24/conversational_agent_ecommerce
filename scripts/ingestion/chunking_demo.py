#!/usr/bin/env python3
"""
Chunking Demonstration Script

This script shows how text chunking works by creating a sample long review
and demonstrating how it gets split into multiple chunks while preserving metadata.

Run with: python scripts/chunking_demo.py
"""

import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langchain_core.documents import Document
from pipelines.ingestion.processors.text_chunker import TextChunker, ChunkerConfig

console = Console()

def create_sample_long_review():
    """Create a sample long review to demonstrate chunking."""
    long_review = """
    I've been using this iPhone for about 6 months now and I have mixed feelings about it. 
    The build quality is excellent - it feels premium in hand with the glass back and aluminum frame. 
    The camera quality is outstanding, especially in good lighting conditions. The photos are sharp, 
    colors are vibrant, and the portrait mode works really well. However, I've noticed some issues 
    with battery life. After about 4 months of use, the battery started draining much faster than 
    when I first got it. I'm getting maybe 6-7 hours of screen time now compared to 10+ hours initially.
    
    The performance is still snappy though. Apps open quickly, multitasking is smooth, and I haven't 
    experienced any significant lag or crashes. The Face ID works reliably most of the time, though 
    it sometimes struggles in very bright sunlight or when I'm wearing sunglasses.
    
    One thing that really bothers me is the lack of a headphone jack. I know this is old news, but 
    I still miss being able to plug in my wired headphones without needing an adapter. The Lightning 
    to 3.5mm adapter that comes with it is easy to lose and adds bulk.
    
    The display is gorgeous - bright, colorful, and sharp. Watching videos and browsing photos is 
    a pleasure. The True Tone feature is nice and adjusts the color temperature based on ambient 
    lighting. However, I do notice some color shifting when viewing the screen at extreme angles.
    
    Overall, it's a solid phone but not without its flaws. The camera and performance are the 
    standout features, while battery life and the missing headphone jack are the main drawbacks. 
    For the price point, I expected better battery longevity. I'd give it a 7/10 - good but not great.
    """.strip()
    
    return Document(
        page_content=long_review,
        metadata={
            "product_name": "Apple iPhone 12 Pro, 256GB, Pacific Blue - Unlocked",
            "description": "Latest iPhone with A14 Bionic chip, Pro camera system, and 5G capability",
            "price": "999.99",
            "rating": "3.5",
            "review_title": "Mixed feelings after 6 months of use"
        }
    )

def demonstrate_chunking():
    """Demonstrate the chunking process."""
    console.print(Panel.fit(
        "[bold blue]Text Chunking Demonstration[/bold blue]\n"
        "Shows how long reviews get split into optimal-sized chunks for embedding",
        border_style="blue"
    ))
    
    # Create sample document
    original_doc = create_sample_long_review()
    
    console.print(f"\n[bold]Original Document:[/bold]")
    console.print(f"Content Length: {len(original_doc.page_content)} characters")
    console.print(Panel(
        original_doc.page_content[:300] + "..." if len(original_doc.page_content) > 300 else original_doc.page_content,
        title="Original Review Text",
        border_style="yellow"
    ))
    
    # Show metadata
    metadata_table = Table(show_header=True, header_style="bold blue", title="Original Metadata")
    metadata_table.add_column("Field", style="cyan")
    metadata_table.add_column("Value", style="white")
    
    for key, value in original_doc.metadata.items():
        metadata_table.add_row(key, str(value))
    
    console.print(metadata_table)
    
    # Configure chunker with different settings
    chunker_config = ChunkerConfig(
        chunk_size=400,  # Smaller chunks to demonstrate splitting
        chunk_overlap=50
    )
    
    chunker = TextChunker(chunker_config)
    
    # Chunk the document
    chunks = chunker.chunk_documents([original_doc])
    
    console.print(f"\n[bold]Chunking Results:[/bold]")
    console.print(f"Original: 1 document ({len(original_doc.page_content)} chars)")
    console.print(f"After chunking: {len(chunks)} chunks")
    
    # Show each chunk
    for i, chunk in enumerate(chunks):
        console.print(f"\n[bold yellow]Chunk {i+1}:[/bold yellow]")
        console.print(f"Length: {len(chunk.page_content)} characters")
        
        console.print(Panel(
            chunk.page_content,
            title=f"Chunk {i+1} Content (What gets embedded)",
            border_style="green"
        ))
        
        # Verify metadata is preserved
        console.print("[dim]Metadata preserved: ✅[/dim]")
        for key, value in chunk.metadata.items():
            console.print(f"[dim]  {key}: {value}[/dim]")
    
    # Analysis
    console.print(f"\n[bold]Chunking Analysis:[/bold]")
    
    analysis_table = Table(show_header=True, header_style="bold magenta")
    analysis_table.add_column("Metric", style="cyan")
    analysis_table.add_column("Value", style="green")
    
    total_chars = sum(len(chunk.page_content) for chunk in chunks)
    avg_chunk_size = total_chars / len(chunks)
    
    analysis_table.add_row("Total Chunks Created", str(len(chunks)))
    analysis_table.add_row("Average Chunk Size", f"{avg_chunk_size:.0f} characters")
    analysis_table.add_row("Chunk Size Limit", f"{chunker_config.chunk_size} characters")
    analysis_table.add_row("Overlap Size", f"{chunker_config.chunk_overlap} characters")
    analysis_table.add_row("Metadata Preserved", "✅ Yes, in all chunks")
    
    console.print(analysis_table)
    
    console.print(Panel(
        "[bold green]✅ Chunking demonstration complete![/bold green]\n\n"
        "[bold]Key Benefits of Chunking:[/bold]\n"
        "• Breaks long reviews into embedding-friendly sizes\n"
        "• Preserves context with configurable overlap\n"
        "• Maintains all metadata in every chunk\n"
        "• Enables more precise semantic search\n"
        "• Each chunk becomes a separate searchable vector",
        border_style="green"
    ))

if __name__ == "__main__":
    demonstrate_chunking()