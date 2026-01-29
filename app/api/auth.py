"""
API Key Authentication Middleware
Validates x-api-key header for all protected endpoints
"""

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from app.config import settings


# API Key header scheme
api_key_header = APIKeyHeader(name=settings.api_key_header, auto_error=False)


async def validate_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    Validate API key from request header
    
    Args:
        api_key: The API key from x-api-key header
        
    Returns:
        The validated API key
        
    Raises:
        HTTPException: If API key is missing or invalid
    """
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "status": "error",
                "message": "Missing API key. Please provide x-api-key header."
            }
        )
    
    if api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "status": "error",
                "message": "Invalid API key or malformed request"
            }
        )
    
    return api_key
