# Technology Stack & Build System

## Package Management
- **Primary**: `uv` - Modern Python package manager for fast, reliable dependency management. To install any library you can use command "uv pip install <package-name>" in terminal
- **Python Version**: 3.11+ recommended for optimal performance

## Core Technologies

### Backend
- **FastAPI**: High-performance async web framework for API endpoints
- **Pydantic**: Data validation and settings management
- **SQLAlchemy**: Database ORM for data persistence
- **Alembic**: Database migration management

### Frontend
- **Streamlit**: Interactive web application framework for chat interface
- **Streamlit-Chat**: Enhanced chat components for conversational UI

### RAG & ML Stack
- **LangChain**: Framework for LLM application development
- **LangGraph**: Framework for LLM application development
- **ChromaDB/Pinecone**: Vector database for embeddings storage
- **OpenAI/Anthropic**: LLM providers for text generation
- **Sentence-Transformers**: Embedding models for semantic search
- **FAISS**: Efficient similarity search library

### Data Processing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **BeautifulSoup4**: Web scraping and HTML parsing
- **PyPDF2/PDFPlumber**: PDF document processing

### Testing & Quality
- **Pytest**: Testing framework
- **Pytest-asyncio**: Async testing support
- **Black**: Code formatting
- **Ruff**: Fast Python linter
- **MyPy**: Static type checking

## Common Commands

### Environment Setup
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

### Development
```bash
# Run FastAPI backend (development)
uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Run Streamlit frontend
uv run streamlit run src/frontend/app.py --server.port 8501

# Run data ingestion pipeline
uv run python -m src.pipelines.ingestion

# Run tests
uv run pytest tests/ -v

# Code formatting and linting
uv run black src/ tests/
uv run ruff check src/ tests/
uv run mypy src/
```

### Production
```bash
# Install production dependencies only
uv pip install -r requirements.txt --no-dev

# Run with Gunicorn
uv run gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## Configuration Management
- Environment variables via `.env` files
- Pydantic Settings for configuration validation
- Separate configs for development, testing, and production environments