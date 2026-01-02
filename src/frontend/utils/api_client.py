"""API client for communicating with FastAPI backend."""

import json
import httpx
from typing import Iterator, Optional, Dict, Any, List

from src.frontend.config import settings


class APIClient:
    """Client for FastAPI backend communication."""
    
    def __init__(self, base_url: Optional[str] = None):
        """Initialize API client.
        
        Args:
            base_url: Backend API base URL (defaults to settings)
        """
        self.base_url = base_url or settings.api_base_url
        self.timeout = settings.request_timeout
    
    def create_session(self) -> Dict[str, Any]:
        """Create a new chat session.
        
        Returns:
            Session data dict
            
        Raises:
            Exception: If request fails
        """
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(f"{self.base_url}/api/sessions")
                response.raise_for_status()
                return response.json()
        except httpx.ConnectError:
            raise Exception("Cannot connect to backend API. Please ensure the API server is running.")
        except httpx.TimeoutException:
            raise Exception("Request timed out. The API server may be overloaded.")
        except httpx.HTTPStatusError as e:
            raise Exception(f"API error ({e.response.status_code}): {e.response.text}")
        except Exception as e:
            raise Exception(f"Unexpected error creating session: {str(e)}")
    
    def get_sessions(self) -> List[Dict[str, Any]]:
        """Get all sessions.
        
        Returns:
            List of session data dicts
        """
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(f"{self.base_url}/api/sessions")
                response.raise_for_status()
                data = response.json()
                return data.get("sessions", [])
        except httpx.ConnectError:
            raise Exception("Cannot connect to backend API. Please ensure the API server is running.")
        except httpx.TimeoutException:
            raise Exception("Request timed out. The API server may be overloaded.")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return []  # No sessions found
            raise Exception(f"API error ({e.response.status_code}): {e.response.text}")
        except Exception as e:
            raise Exception(f"Unexpected error loading sessions: {str(e)}")
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific session with messages.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data dict or None if not found
        """
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(f"{self.base_url}/api/sessions/{session_id}")
                response.raise_for_status()
                return response.json()
        except httpx.ConnectError:
            raise Exception("Cannot connect to backend API. Please ensure the API server is running.")
        except httpx.TimeoutException:
            raise Exception("Request timed out. The API server may be overloaded.")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise Exception(f"API error ({e.response.status_code}): {e.response.text}")
        except Exception as e:
            raise Exception(f"Unexpected error loading session: {str(e)}")
    
    def send_message(self, query: str, session_id: str) -> Dict[str, Any]:
        """Send a message and get response (non-streaming).
        
        Args:
            query: User message
            session_id: Session identifier
            
        Returns:
            Response data dict
        """
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/api/chat",
                    json={"query": query, "session_id": session_id}
                )
                response.raise_for_status()
                return response.json()
        except httpx.ConnectError:
            raise Exception("Cannot connect to backend API. Please ensure the API server is running.")
        except httpx.TimeoutException:
            raise Exception("Request timed out. The response may be taking longer than expected.")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 422:
                raise Exception("Invalid request. Please check your message and try again.")
            elif e.response.status_code == 404:
                raise Exception("Session not found. Please start a new chat.")
            else:
                raise Exception(f"API error ({e.response.status_code}): {e.response.text}")
        except Exception as e:
            raise Exception(f"Unexpected error sending message: {str(e)}")
    
    def stream_message(self, query: str, session_id: str) -> Iterator[str]:
        """Stream a response via SSE.
        
        Args:
            query: User message
            session_id: Session identifier
            
        Yields:
            Response chunks as they arrive
        """
        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream(
                    "POST",
                    f"{self.base_url}/api/chat/stream",
                    json={"query": query, "session_id": session_id}
                ) as response:
                    response.raise_for_status()
                    
                    for line in response.iter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]  # Remove "data: " prefix
                            try:
                                data = json.loads(data_str)
                                
                                # Check for error
                                if "error" in data:
                                    raise Exception(data.get("error", "Unknown streaming error"))
                                
                                # Check if done
                                if data.get("done", False):
                                    break
                                
                                # Yield chunk
                                chunk = data.get("chunk", "")
                                if chunk:
                                    yield chunk
                                    
                            except json.JSONDecodeError:
                                continue
        except httpx.ConnectError:
            raise Exception("Cannot connect to backend API. Please ensure the API server is running.")
        except httpx.TimeoutException:
            raise Exception("Streaming timed out. The response may be taking longer than expected.")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 422:
                raise Exception("Invalid request. Please check your message and try again.")
            elif e.response.status_code == 404:
                raise Exception("Session not found. Please start a new chat.")
            else:
                raise Exception(f"API error ({e.response.status_code}): {e.response.text}")
        except Exception as e:
            raise Exception(f"Unexpected error during streaming: {str(e)}")
    
    def check_health(self) -> Dict[str, Any]:
        """Check API health.
        
        Returns:
            Health status dict
        """
        with httpx.Client(timeout=10) as client:
            try:
                response = client.get(f"{self.base_url}/api/health")
                response.raise_for_status()
                return response.json()
            except Exception as e:
                return {"status": "unhealthy", "error": str(e)}
