"""
Pydantic schemas for request/response validation
Ensures strict typing for API contracts
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict
from app.config import settings


# Supported language type
LanguageType = Literal["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

# Classification type
ClassificationType = Literal["AI_GENERATED", "HUMAN"]


class VoiceDetectionRequest(BaseModel):
    """Request schema for voice detection endpoint"""
    
    language: LanguageType = Field(
        ...,
        description="Language of the audio: Tamil, English, Hindi, Malayalam, or Telugu"
    )
    audioFormat: Literal["mp3"] = Field(
        ...,
        description="Audio format, must be 'mp3'"
    )
    audioBase64: str = Field(
        ...,
        min_length=100,
        description="Base64-encoded MP3 audio data"
    )
    
    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        """Ensure language is in supported list"""
        if v not in settings.supported_languages:
            raise ValueError(
                f"Unsupported language: {v}. "
                f"Supported: {', '.join(settings.supported_languages)}"
            )
        return v
    
    @field_validator("audioBase64")
    @classmethod
    def validate_base64(cls, v: str) -> str:
        """Basic validation for base64 string"""
        # Remove potential data URL prefix
        if v.startswith("data:"):
            v = v.split(",", 1)[-1]
        
        # Check for valid base64 characters
        import re
        if not re.match(r'^[A-Za-z0-9+/=]+$', v):
            raise ValueError("Invalid base64 encoding")
        
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "language": "Tamil",
                "audioFormat": "mp3",
                "audioBase64": "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMAAAAAAA..."
            }
        }
    )


class VoiceDetectionResponse(BaseModel):
    """Successful response schema for voice detection"""
    
    status: Literal["success"] = "success"
    language: LanguageType
    classification: ClassificationType = Field(
        ...,
        description="AI_GENERATED or HUMAN"
    )
    confidenceScore: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0"
    )
    explanation: str = Field(
        ...,
        description="Short reason for the classification decision"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "language": "Tamil",
                "classification": "AI_GENERATED",
                "confidenceScore": 0.91,
                "explanation": "Unnatural pitch consistency and robotic speech patterns detected"
            }
        }
    )


class ErrorResponse(BaseModel):
    """Error response schema"""
    
    status: Literal["error"] = "error"
    message: str = Field(
        ...,
        description="Error description"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "error",
                "message": "Invalid API key or malformed request"
            }
        }
    )


class HealthResponse(BaseModel):
    """Health check response"""
    
    status: str = "healthy"
    version: str
    supported_languages: list[str]
