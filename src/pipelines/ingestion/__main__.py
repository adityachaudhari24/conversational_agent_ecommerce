"""
Entry point for running the ingestion pipeline as a module.

Allows the pipeline to be executed with:
    python -m src.pipelines.ingestion
"""

from .cli import main

if __name__ == "__main__":
    exit(main())