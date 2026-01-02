"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .dependencies import set_inference_pipeline
from .routes import chat_router, sessions_router, health_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup
    logger.info("Starting E-Commerce Chat API...")
    
    try:
        # Initialize inference pipeline
        from src.pipelines.retrieval.pipeline import RetrievalPipeline
        from src.pipelines.inference.pipeline import InferencePipeline
        
        logger.info("Initializing retrieval pipeline...")
        retrieval_pipeline = RetrievalPipeline.from_config_file()
        retrieval_pipeline.initialize()
        
        logger.info("Initializing inference pipeline...")
        inference_pipeline = InferencePipeline.from_config_file(retrieval_pipeline)
        inference_pipeline.initialize()
        
        set_inference_pipeline(inference_pipeline)
        logger.info("Pipelines initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize pipelines: {e}")
        logger.warning("API will start without inference capability")
        set_inference_pipeline(None)
    
    yield
    
    # Shutdown
    logger.info("Shutting down E-Commerce Chat API...")


# Create FastAPI app
app = FastAPI(
    title="E-Commerce Chat API",
    description="REST API for conversational e-commerce assistant with streaming support",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router, prefix="/api")
app.include_router(sessions_router, prefix="/api")
app.include_router(chat_router, prefix="/api")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "E-Commerce Chat API",
        "version": "1.0.0",
        "docs": "/docs"
    }
