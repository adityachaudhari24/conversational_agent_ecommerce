"""Health check endpoint."""

import time
from fastapi import APIRouter, Depends

from ..models.schemas import HealthResponse
from ..dependencies import get_inference_pipeline

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(
    inference_pipeline=Depends(get_inference_pipeline)
) -> HealthResponse:
    """Check API and dependency health.
    
    Returns health status of the API and its dependencies.
    """
    start_time = time.time()
    services = {"api": "up"}
    overall_status = "healthy"
    
    # Check inference pipeline
    if inference_pipeline is not None and inference_pipeline._initialized:
        services["inference_pipeline"] = "up"
    else:
        services["inference_pipeline"] = "down"
        overall_status = "unhealthy"
    
    response_time_ms = (time.time() - start_time) * 1000
    
    return HealthResponse(
        status=overall_status,
        services=services,
        response_time_ms=round(response_time_ms, 2)
    )
