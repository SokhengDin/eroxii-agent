import asyncio
import base64
import json
import re
from typing import Optional

from fastapi import APIRouter, File, UploadFile, HTTPException
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


_DETECTION_PROMPT = (
    "You are a license plate detection system. Analyze the image and extract:\n"
    "1. The license plate number\n"
    "2. The vehicle type (CAR, MOTORCYCLE, TRUCK, BUS, VAN, OTHER)\n"
    "3. The bounding box of the license plate as normalized coordinates [0.0–1.0] relative to image size\n"
    "4. Your confidence score [0.0–1.0]\n\n"
    "Reply ONLY with this exact JSON and nothing else:\n"
    '{"license_plate": "ABC123", "vehicle_type": "CAR", "confidence": 0.95, '
    '"bounding_box": {"x1": 0.2, "y1": 0.7, "x2": 0.8, "y2": 0.9}}\n\n'
    "If no plate is found use empty string for license_plate. "
    "If bounding box cannot be determined set it to null. "
    "DO NOT include any text outside the JSON."
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

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    mime = file.content_type or "image/jpeg"

    from app.agent.ocr_agent import _get_llm
    from langchain.agents import create_agent
    from langchain_core.messages import HumanMessage

    try:
        extractor = create_agent(
            _get_llm(),
            tools         = [],
            system_prompt = _DETECTION_PROMPT,
        )

        result = await asyncio.wait_for(
            extractor.ainvoke({
                "messages": [HumanMessage(content=[
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": "Detect the license plate and return JSON only."},
                ])]
            }),
            timeout=30.0,
        )

        messages = result.get("messages", [])
        raw      = messages[-1].content if messages else ""
        logger.info(f"[detect] Raw output: {repr(raw)}")

        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if not match:
            logger.warning(f"[detect] No JSON in output: {repr(raw)}")
            return DetectionResult(license_plate="", vehicle_type="UNKNOWN")

        data = json.loads(match.group())

        bbox = None
        if data.get("bounding_box"):
            b = data["bounding_box"]
            try:
                bbox = BoundingBox(
                    x1=float(b["x1"]),
                    y1=float(b["y1"]),
                    x2=float(b["x2"]),
                    y2=float(b["y2"]),
                )
            except Exception:
                bbox = None

        return DetectionResult(
            license_plate = data.get("license_plate", "").strip(),
            vehicle_type  = data.get("vehicle_type", "UNKNOWN").strip(),
            confidence    = float(data["confidence"]) if data.get("confidence") is not None else None,
            bounding_box  = bbox,
        )

    except asyncio.TimeoutError:
        logger.error("[detect] Timeout waiting for LLM")
        raise HTTPException(status_code=504, detail="Detection timed out.")
    except json.JSONDecodeError as e:
        logger.error(f"[detect] JSON parse error: {e}")
        raise HTTPException(status_code=422, detail="Could not parse detection result.")
    except Exception as e:
        logger.error(f"[detect] Error: {e}")
        raise HTTPException(status_code=500, detail="Detection failed.")
