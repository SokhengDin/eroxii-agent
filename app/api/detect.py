import asyncio
import base64
import json
import re
from typing import Optional

from fastapi import APIRouter, File, UploadFile, HTTPException
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.core.logger import logger

router = APIRouter(prefix="/api/v1/detect", tags=["ALPR Detection"])


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class DetectionResult(BaseModel):
    license_plate : str
    vehicle_type  : str
    confidence    : Optional[float] = None
    bounding_box  : Optional[BoundingBox] = None


_SYSTEM = (
    "You are a license plate detection system. Analyze the image.\n"
    "Detect all objects and return a JSON array using this format:\n"
    '[{"bbox_2d": [x1, y1, x2, y2], "label": "<label>"}]\n\n'
    "Labels to use:\n"
    "- 'license_plate' for the plate region\n"
    "- 'CAR', 'MOTORCYCLE', 'TRUCK', 'BUS', 'VAN', 'OTHER' for the vehicle\n"
    "Also include a 'text' field on the license_plate object with the plate number string.\n"
    "Example: [{\"bbox_2d\": [100, 400, 600, 500], \"label\": \"license_plate\", \"text\": \"ABC123\", \"confidence\": 0.95}, "
    "{\"bbox_2d\": [0, 0, 800, 600], \"label\": \"CAR\"}]\n"
    "Return ONLY the JSON array, no markdown, no explanation."
)


def _parse_qwen_output(raw: str) -> DetectionResult:
    # strip markdown fences
    cleaned = re.sub(r'```[a-z]*\n?|\n?```', '', raw).strip()

    # find JSON array
    match = re.search(r'\[.*\]', cleaned, re.DOTALL)
    if not match:
        logger.warning(f"[detect] No JSON array in output: {repr(raw)}")
        return DetectionResult(license_plate="", vehicle_type="UNKNOWN")

    items: list = json.loads(match.group(0))

    plate_item   = next((i for i in items if i.get("label") == "license_plate"), None)
    vehicle_item = next((i for i in items if i.get("label") not in ("license_plate",)), None)

    license_plate = ""
    bbox          = None
    confidence    = None
    vehicle_type  = "UNKNOWN"

    if plate_item:
        license_plate = plate_item.get("text", "").strip()
        confidence    = plate_item.get("confidence")
        coords        = plate_item.get("bbox_2d")
        if coords and len(coords) == 4:
            bbox = BoundingBox(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3])

    if vehicle_item:
        vehicle_type = vehicle_item.get("label", "UNKNOWN").upper()

    return DetectionResult(
        license_plate = license_plate,
        vehicle_type  = vehicle_type,
        confidence    = float(confidence) if confidence is not None else None,
        bounding_box  = bbox,
    )


@router.post("", response_model=DetectionResult)
async def detect_license_plate(file: UploadFile = File(...)):
    """
    Detect license plate from an uploaded image.
    Returns plate text, vehicle type, confidence score, and bounding box (pixel coords).
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
        llm      = _get_llm()
        messages = [
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=[
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": "Detect the license plate and vehicle. Return JSON array only."},
            ]),
        ]

        response = await asyncio.wait_for(llm.ainvoke(messages), timeout=30.0)
        raw      = response.content if hasattr(response, "content") else str(response)
        logger.info(f"[detect] Raw output: {repr(raw)}")

        result = _parse_qwen_output(raw)
        logger.info(f"[detect] plate={result.license_plate!r} type={result.vehicle_type!r} bbox={result.bounding_box}")
        return result

    except asyncio.TimeoutError:
        logger.error("[detect] Timeout waiting for LLM")
        raise HTTPException(status_code=504, detail="Detection timed out.")
    except json.JSONDecodeError as e:
        logger.error(f"[detect] JSON parse error: {e}")
        raise HTTPException(status_code=422, detail="Could not parse detection result.")
    except Exception as e:
        logger.error(f"[detect] Error: {e}")
        raise HTTPException(status_code=500, detail="Detection failed.")
