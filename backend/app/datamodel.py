"""Type definitions for data interfaces."""
from enum import Enum
from typing import Optional

from pydantic import BaseModel

from .common import test_qr_dataurl


class HealthResponse(BaseModel):
    status: str
    status_unserious: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "status": "200 OK",
            }
        }


class HealthRequest(BaseModel):
    serious: bool = True


class JobStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    CONSUMED = "CONSUMED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class ImagePayload(BaseModel):
    image_data: str


class JobRequest(BaseModel):
    prompt: str
    image: ImagePayload

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "a Shiba Inu drinking an Americano and eating pancakes",
                "image": ImagePayload(image_data=test_qr_dataurl),
            }
        }


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus

    class Config:
        json_schema_extra = {
            "example": {"job_id": "_test", "status": JobStatus.COMPLETE}
        }
