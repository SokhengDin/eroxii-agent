import asyncio
import base64
import json
import re

from fastapi import APIRouter, File, UploadFile, HTTPException
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.core.logger import logger

router = APIRouter(prefix="/api/v1/detect", tags=["ALPR Detection"])


class DetectionResult(BaseModel):
    license_plate : str
    vehicle_type  : str


_SYSTEM = (
    "You are a license plate OCR system. "
    "Look at the image and read the license plate number and identify the vehicle type. "
    "Reply with ONLY this JSON — no markdown, no explanation:\n"
    '{"license_plate": "ABC123", "vehicle_type": "CAR"}\n'
    "vehicle_type must be one of: CAR, MOTORCYCLE, TRUCK, BUS, VAN, OTHER. "
    "Use empty string for license_plate if you cannot read it."
)


def _parse(raw: str) -> DetectionResult:
    cleaned = re.sub(r'```[a-z]*\n?|\n?```', '', raw).strip()
    match   = re.search(r'\{[^{}]*\}', cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            return DetectionResult(
                license_plate = data.get("license_plate", "").strip(),
                vehicle_type  = data.get("vehicle_type", "UNKNOWN").strip().upper(),
            )
        except json.JSONDecodeError:
            pass
    # last resort: regex grab plate-like token
    plate = re.search(r'\b([A-Z0-9]{4,10})\b', cleaned.upper())
    return DetectionResult(
        license_plate = plate.group(1) if plate else "",
        vehicle_type  = "UNKNOWN",
    )


@router.post("", response_model=DetectionResult)
async def detect_license_plate(file: UploadFile = File(...)):
    """Detect license plate text and vehicle type from an uploaded image."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    logger.info(f"[detect] {file.filename} ({len(image_bytes)} bytes)")

    b64  = base64.b64encode(image_bytes).decode("utf-8")
    mime = file.content_type or "image/jpeg"

    from app.agent.ocr_agent import _get_llm

    try:
        llm      = _get_llm()
        response = await asyncio.wait_for(
            llm.ainvoke([
                SystemMessage(content=_SYSTEM),
                HumanMessage(content=[
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": 'Read the license plate. Reply with ONLY JSON: {"license_plate": "...", "vehicle_type": "..."}'},
                ]),
            ]),
            timeout=30.0,
        )
        raw    = response.content if hasattr(response, "content") else str(response)
        logger.info(f"[detect] raw={repr(raw)}")
        result = _parse(raw)
        logger.info(f"[detect] plate={result.license_plate!r} type={result.vehicle_type!r}")
        return result

    except asyncio.TimeoutError:
        logger.error("[detect] Timeout")
        raise HTTPException(status_code=504, detail="Detection timed out.")
    except Exception as e:
        logger.error(f"[detect] Error: {e}")
        raise HTTPException(status_code=500, detail="Detection failed.")
