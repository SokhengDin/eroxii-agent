import asyncio
import base64
from typing import Optional

from fastapi import APIRouter, File, UploadFile, HTTPException
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.core.logger import logger

router = APIRouter(prefix="/api/v1/detect", tags=["ALPR Detection"])


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionResult(BaseModel):
    license_plate : str
    vehicle_type  : str
    confidence    : Optional[float] = None
    bounding_box  : Optional[BoundingBox] = None


_SYSTEM = (
    "You are a license plate detection system. Analyze the image and extract:\n"
    "- license_plate: the plate number string, empty string if not found\n"
    "- vehicle_type: one of CAR, MOTORCYCLE, TRUCK, BUS, VAN, OTHER\n"
    "- confidence: your confidence score between 0.0 and 1.0\n"
    "- bounding_box: normalized coords (0.0–1.0) of the plate region, null if undetermined"
)


@router.post("", response_model=DetectionResult)
async def detect_license_plate(file: UploadFile = File(...)):
    """
    Detect license plate from an uploaded image.
    Returns plate text, vehicle type, confidence score, and bounding box (if determinable).
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    logger.info(f"[detect] Received image: {file.filename} ({len(image_bytes)} bytes)")

    b64  = base64.b64encode(image_bytes).decode("utf-8")
    mime = file.content_type or "image/jpeg"

    from app.agent.ocr_agent import _get_llm

    try:
        llm      = _get_llm().with_structured_output(DetectionResult)
        messages = [
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=[
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": "Detect the license plate in this image."},
            ]),
        ]

        result: DetectionResult = await asyncio.wait_for(
            llm.ainvoke(messages),
            timeout=30.0,
        )
        logger.info(f"[detect] plate={result.license_plate!r} type={result.vehicle_type!r} confidence={result.confidence}")
        return result

    except asyncio.TimeoutError:
        logger.error("[detect] Timeout waiting for LLM")
        raise HTTPException(status_code=504, detail="Detection timed out.")
    except Exception as e:
        logger.error(f"[detect] Error: {e}")
        raise HTTPException(status_code=500, detail="Detection failed.")
