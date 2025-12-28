# Project Structure & Organization

## Directory Layout

```
conversational_agent_ecommerce/
├── src/                          # Main source code
│   ├── api/                      # FastAPI backend
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI app entry point
│   │   ├── routes/              # API route handlers
│   │   ├── middleware/          # Custom middleware
│   │   └── dependencies.py     # Dependency injection
│   ├── frontend/                # Streamlit frontend
│   │   ├── __init__.py
│   │   ├── app.py              # Main Streamlit app
│   │   ├── components/         # Reusable UI components
│   │   └── utils/              # Frontend utilities
│   ├── pipelines/              # RAG pipeline modules
│   │   ├── __init__.py
│   │   ├── ingestion/          # Data ingestion pipeline
│   │   ├── retrieval/          # Data retrieval pipeline
│   │   └── inference/          # Data inference pipeline
│   ├── core/                   # Core business logic
│   │   ├── __init__.py
│   │   ├── models/             # Pydantic models & schemas
│   │   ├── services/           # Business logic services
│   │   └── database/           # Database models & connections
│   ├── utils/                  # Shared utilities
│   │   ├── __init__.py
│   │   ├── logging.py          # Logging configuration
│   │   ├── config.py           # Configuration management
│   │   └── exceptions.py       # Custom exceptions
│   └── __init__.py
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   ├── e2e/                    # End-to-end tests
│   ├── fixtures/               # Test fixtures
│   └── conftest.py            # Pytest configuration
├── data/                       # Data storage
│   ├── raw/                    # Raw scraped data
│   ├── processed/              # Processed data
│   └── embeddings/             # Vector embeddings
├── docs/                       # Documentation
│   ├── api/                    # API documentation
│   ├── architecture/           # Architecture diagrams
│   └── deployment/             # Deployment guides
├── scripts/                    # Utility scripts
│   ├── setup.py               # Environment setup
│   ├── data_ingestion.py      # Data processing scripts
│   └── deploy.py              # Deployment scripts
├── config/                     # Configuration files
│   ├── development.env
│   ├── production.env
│   └── logging.yaml
├── requirements.txt            # Python dependencies
├── pyproject.toml             # Project configuration
├── .env.example               # Environment template
├── docker-compose.yml         # Local development setup
├── Dockerfile                 # Container configuration
└── README.md                  # Project documentation
```

## Module Organization Principles

### Pipeline Architecture
Each pipeline (ingestion, retrieval, inference) follows a consistent structure:
- `pipeline.py`: Main pipeline orchestrator
- `processors/`: Individual processing components
- `models/`: Pipeline-specific data models
- `config.py`: Pipeline configuration
- `__init__.py`: Module exports

### API Structure
- **Routes**: Organized by domain (chat, health, admin)
- **Dependencies**: Shared dependencies for authentication, database, etc.
- **Middleware**: Cross-cutting concerns (logging, CORS, rate limiting)

### Frontend Components
- **Pages**: Main application pages
- **Components**: Reusable UI elements
- **Utils**: Frontend-specific utilities and helpers

## Naming Conventions

### Files & Directories
- Use snake_case for Python files and directories
- Use descriptive names that indicate purpose
- Group related functionality in subdirectories

### Python Code
- Classes: PascalCase (e.g., `DataProcessor`, `ChatService`)
- Functions/Variables: snake_case (e.g., `process_data`, `user_query`)
- Constants: UPPER_SNAKE_CASE (e.g., `MAX_TOKENS`, `DEFAULT_MODEL`)
- Private methods: Leading underscore (e.g., `_validate_input`)

### Configuration
- Environment variables: UPPER_SNAKE_CASE with prefixes
- Config sections: lowercase with underscores
- API endpoints: kebab-case in URLs

## Import Organization
1. Standard library imports
2. Third-party imports
3. Local application imports
4. Relative imports (if necessary)

## Testing Structure
- Mirror source structure in tests/
- Use descriptive test names: `test_should_return_valid_response_when_query_is_valid`
- Group tests by functionality, not by file structure