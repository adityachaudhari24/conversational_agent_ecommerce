"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .dependencies import set_inference_pipeline, set_conversation_store
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
        # Initialize retrieval pipeline
        from src.pipelines.retrieval.pipeline import RetrievalPipeline
        from src.pipelines.inference.pipeline import InferencePipeline, InferenceConfig
        from src.pipelines.inference.config import create_settings_from_yaml
        from src.pipelines.inference.conversation.store import ConversationStore
        from src.pipelines.inference.config import (
            LLMConfig,
            ConversationConfig,
            GeneratorConfig,
            WorkflowConfig
        )
        
        logger.info("Initializing retrieval pipeline...")
        retrieval_pipeline = RetrievalPipeline.from_config_file()
        retrieval_pipeline.initialize()
        
        logger.info("Loading inference configuration...")
        settings = create_settings_from_yaml()
        
        # Create component configurations
        llm_config = LLMConfig(
            provider=settings.llm.provider,
            model_name=settings.llm.model_name,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
            api_key=settings.get_api_key()
        )
        
        conversation_config = ConversationConfig(
            max_history_length=settings.conversation.max_history_length,
            storage_dir=settings.conversation.storage_dir
        )
        
        generator_config = GeneratorConfig(
            system_prompt=settings.generator.system_prompt,
            max_context_tokens=settings.generator.max_context_tokens
        )
        
        workflow_config = WorkflowConfig(
            product_keywords=settings.workflow.product_keywords,
            tool_keywords=settings.workflow.tool_keywords
        )
        
        # Create main pipeline configuration
        pipeline_config = InferenceConfig(
            llm_config=llm_config,
            conversation_config=conversation_config,
            generator_config=generator_config,
            workflow_config=workflow_config,
            enable_streaming=settings.enable_streaming,
            max_retries=settings.max_retries,
            timeout_seconds=settings.timeout_seconds
        )
        
        # Create ConversationStore instance
        logger.info("Initializing conversation store...")
        conversation_store = ConversationStore(
            storage_dir=conversation_config.storage_dir,
            max_history_length=conversation_config.max_history_length
        )
        set_conversation_store(conversation_store)
        
        # Create inference pipeline with conversation store
        logger.info("Initializing inference pipeline...")
        inference_pipeline = InferencePipeline(pipeline_config, retrieval_pipeline, conversation_store)
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
