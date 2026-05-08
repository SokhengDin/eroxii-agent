import asyncio
import base64
import json
import re

from fastapi import APIRouter, File, UploadFile, HTTPException
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from app.core.logger import logger

router = APIRouter(prefix="/api/v1/detect", tags=["ALPR Detection"])


class DetectionResult(BaseModel):
    license_plate : str
    vehicle_type  : str


@router.post("", response_model=DetectionResult)
async def detect_license_plate(file: UploadFile = File(...)):
    """Detect license plate text and vehicle type from an uploaded image."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    logger.info(f"[detect] {file.filename} ({len(image_bytes)} bytes)")

    from app.agent.ocr_agent import _get_llm
    from langchain.agents import create_agent

    b64  = base64.b64encode(image_bytes).decode("utf-8")
    mime = file.content_type or "image/jpeg"

    try:
        extractor = create_agent(
            _get_llm(),
            tools         = [],
            system_prompt = (
                "You are a vehicle license plate recognition assistant. "
                "Analyze the image carefully. It may be a vehicle photo or a parking ticket/receipt. "
                "Read the actual characters on the license plate visible in the image. "
                "Reply with ONLY a JSON object — no markdown, no code block, no explanation:\n"
                '{"license_plate": "<actual plate text>", "vehicle_type": "<type>"}\n'
                "vehicle_type must be one of: CAR, MOTORCYCLE, TRUCK, BUS, VAN, OTHER. "
                "NEVER use placeholder or example values — only real text read from the image. "
                "If the plate is unreadable, use empty string."
            ),
        )

        result = await asyncio.wait_for(
            extractor.ainvoke({"messages": [HumanMessage(content=[
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": "What is the license plate number shown in this image? Reply with ONLY JSON, no markdown."},
            ])]}),
            timeout=30.0,
        )

        messages = result.get("messages", [])
        raw      = messages[-1].content if messages else ""
        logger.info(f"[detect] raw={repr(raw)}")

        cleaned    = re.sub(r'```[a-z]*\n?|\n?```', '', raw).strip()
        json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if not json_match:
            logger.warning(f"[detect] No JSON found in: {repr(raw)}")
            return DetectionResult(license_plate="", vehicle_type="UNKNOWN")

        data = json.loads(json_match.group(0))
        det  = DetectionResult(
            license_plate = data.get("license_plate", "").strip(),
            vehicle_type  = data.get("vehicle_type", "UNKNOWN").strip().upper(),
        )
        logger.info(f"[detect] plate={det.license_plate!r} type={det.vehicle_type!r}")
        return det

    except asyncio.TimeoutError:
        logger.error("[detect] Timeout")
        raise HTTPException(status_code=504, detail="Detection timed out.")
    except json.JSONDecodeError as e:
        logger.error(f"[detect] JSON parse error: {e}")
        raise HTTPException(status_code=422, detail="Could not parse detection result.")
    except Exception as e:
        logger.error(f"[detect] Error: {e}")
        raise HTTPException(status_code=500, detail="Detection failed.")
