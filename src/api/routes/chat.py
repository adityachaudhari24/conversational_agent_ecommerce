"""Chat endpoints with streaming support."""

import json
import logging
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from ..models.schemas import ChatRequest, ChatResponse
from ..dependencies import get_inference_pipeline, get_session_store
from ..services.session_store import SessionStore

router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    inference_pipeline=Depends(get_inference_pipeline),
    session_store: SessionStore = Depends(get_session_store)
) -> ChatResponse:
    """Send a message and get a response (non-streaming)."""
    logger.info(f"Chat request: session={request.session_id}, query={request.query[:50]}...")
    
    # Verify session exists
    session = session_store.get_session(request.session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "SESSION_NOT_FOUND", "message": f"Session {request.session_id} not found"}
        )
    
    # Check pipeline availability
    if inference_pipeline is None or not inference_pipeline._initialized:
        raise HTTPException(
            status_code=503,
            detail={"error_code": "PIPELINE_UNAVAILABLE", "message": "Inference pipeline is not available"}
        )
    
    try:
        # Generate response using inference pipeline
        result = await inference_pipeline.agenerate(
            query=request.query,
            session_id=request.session_id
        )
        
        # Save messages to session store
        session_store.add_message(request.session_id, "user", request.query)
        session_store.add_message(request.session_id, "assistant", result.response)
        
        logger.info(f"Chat response generated: session={request.session_id}, length={len(result.response)}")
        
        return ChatResponse(
            response=result.response,
            session_id=request.session_id,
            metadata=result.metadata
        )
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error_code": "INFERENCE_ERROR", "message": str(e)}
        )


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    inference_pipeline=Depends(get_inference_pipeline),
    session_store: SessionStore = Depends(get_session_store)
):
    """Stream a response using Server-Sent Events (SSE)."""
    logger.info(f"Stream request: session={request.session_id}, query={request.query[:50]}...")
    
    # Verify session exists
    session = session_store.get_session(request.session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "SESSION_NOT_FOUND", "message": f"Session {request.session_id} not found"}
        )
    
    # Check pipeline availability
    if inference_pipeline is None or not inference_pipeline._initialized:
        raise HTTPException(
            status_code=503,
            detail={"error_code": "PIPELINE_UNAVAILABLE", "message": "Inference pipeline is not available"}
        )
    
    async def generate_stream():
        """Generate SSE stream."""
        complete_response = ""
        
        try:
            # Save user message first
            session_store.add_message(request.session_id, "user", request.query)
            
            # Stream response chunks
            async for chunk in inference_pipeline.stream(
                query=request.query,
                session_id=request.session_id
            ):
                complete_response += chunk
                # SSE format: data: <json>\n\n
                event_data = json.dumps({"chunk": chunk, "done": False})
                yield f"data: {event_data}\n\n"
            
            # Save complete assistant response
            session_store.add_message(request.session_id, "assistant", complete_response)
            
            # Send completion event
            completion_data = json.dumps({
                "chunk": "",
                "done": True,
                "session_id": request.session_id,
                "total_length": len(complete_response)
            })
            yield f"data: {completion_data}\n\n"
            
            logger.info(f"Stream completed: session={request.session_id}, length={len(complete_response)}")
            
        except Exception as e:
            logger.error(f"Stream error: {str(e)}")
            # Send error event with partial response
            error_data = json.dumps({
                "error": str(e),
                "error_code": "STREAMING_ERROR",
                "partial_response": complete_response,
                "done": True
            })
            yield f"data: {error_data}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )
