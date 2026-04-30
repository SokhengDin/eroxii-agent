import asyncio
import base64
import json
import re
import torch
import httpx

from datetime import datetime
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from transformers import pipeline

from app.core.config import settings
from app.core.logger import logger

_llm: ChatHuggingFace | None = None


def _get_llm() -> ChatHuggingFace:
    global _llm
    if _llm is not None:
        logger.debug("LLM singleton reused")
        return _llm

    device = 0 if torch.cuda.is_available() else -1
    logger.info(f"Loading LLM {settings.HF_MODEL_ID} on {'cuda' if device == 0 else 'cpu'}")
    pipe = pipeline(
        "text-generation",
        model          = settings.HF_MODEL_ID,
        device         = device,
        dtype          = torch.float16 if device == 0 else torch.float32,
        max_new_tokens = 512,
        do_sample      = False,
    )
    _llm = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe))
    logger.info("LLM loaded successfully")
    return _llm


def _format_reply(user: dict) -> str:
    expired_raw = user.get("subscription_expired_at", "")
    try:
        expired = datetime.fromisoformat(expired_raw).strftime("%Y-%m-%d %H:%M")
    except Exception:
        expired = expired_raw

    return (
        f"Name        : {user.get('first_name', '')} {user.get('last_name', '')}\n"
        f"Type        : {user.get('user_type', '')}\n"
        f"Phone       : {user.get('phone_number', '')}\n"
        f"Payment Due : {'Yes' if user.get('payment_needed') else 'No'}\n"
        f"Vehicle     : {user.get('vehicle_type', '')} — {user.get('license_plate', '')}\n"
        f"Plate Type  : {user.get('plate_type', '')}\n"
        f"Subscription: {'Active' if user.get('has_subscription') else 'None'}\n"
        f"Expires     : {expired}"
    )


async def _search_plate(license_plate: str) -> str:
    logger.info(f"Searching backend for plate: {license_plate}")
    async with httpx.AsyncClient(timeout=15) as http:
        resp = await http.get(
            f"{settings.API_BASE_URL}/api/v1/users/members/search",
            params={
                "search_term" : license_plate,
                "user_type"   : "ALL",
                "skip"        : 0,
                "limit"       : 1,
            },
        )

    logger.info(f"Backend response: {resp.status_code}")
    if resp.status_code != 200:
        return f"Backend error {resp.status_code} for plate: {license_plate}"

    users = resp.json().get("users", [])
    logger.info(f"Found {len(users)} user(s) for plate: {license_plate}")

    if not users:
        return f"Plate {license_plate} not registered in the system."

    replies = [f"Plate: {license_plate} — {len(users)} result(s) found\n"]
    for u in users:
        replies.append(_format_reply(u))
        replies.append("")
    return "\n".join(replies).strip()


def _llm_extract_plate(ocr_text: str) -> dict:
    logger.info(f"Sending OCR text to LLM for extraction, text length={len(ocr_text)}")
    llm = _get_llm()
    messages = [
        SystemMessage(content=(
            "You are a license plate extractor. "
            "Given raw OCR text from a vehicle image or parking receipt, "
            "extract the license plate number and vehicle type. "
            'Reply ONLY with a valid JSON object: {"license_plate": "ABC123", "vehicle_type": "CAR"} '
            "If you cannot find a plate, use empty string."
        )),
        HumanMessage(content=f"OCR text:\n{ocr_text}"),
    ]
    response = llm.invoke(messages)
    raw = response.content if hasattr(response, "content") else str(response)
    logger.info(f"LLM extraction output: {repr(raw)}")
    return raw


async def process_vehicle_image(image_bytes: bytes) -> str:
    logger.info(f"process_vehicle_image called, image size={len(image_bytes)} bytes")
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    logger.info("Calling ocr_image MCP tool directly")
    client = MultiServerMCPClient(
        {
            "ocr": {
                "url"       : settings.OCR_MCP_URL,
                "transport" : "streamable_http",
            },
        }
    )
    tools = await client.get_tools()
    logger.info(f"OCR tools: {[t.name for t in tools]}")

    ocr_tool = next((t for t in tools if t.name == "ocr_image"), None)
    if ocr_tool is None:
        logger.error("ocr_image tool not found on MCP server")
        return "OCR service unavailable."

    logger.info("Invoking ocr_image tool")
    try:
        ocr_result = await ocr_tool.ainvoke({"image_base64": b64})
    except* Exception as eg:
        for exc in eg.exceptions:
            logger.error(f"OCR MCP sub-error: {type(exc).__name__}: {exc}")
        raise
    logger.info(f"OCR tool result: {repr(ocr_result)}")

    ocr_text = ocr_result.get("raw_text", "") if isinstance(ocr_result, dict) else str(ocr_result)
    logger.info(f"OCR raw text: {repr(ocr_text)}")

    if not ocr_text.strip():
        return "No text detected in the image."
    
    logger.info("Running LLM extraction in thread executor")

    loop    = asyncio.get_event_loop()
    raw_llm = await loop.run_in_executor(None, _llm_extract_plate, ocr_text)

    match   = re.search(r'\{.*?\}', raw_llm, re.DOTALL)
    if not match:
        logger.warning(f"No JSON in LLM output: {repr(raw_llm)}")
        return "Could not extract vehicle info from image."

    try:
        extracted = json.loads(match.group())
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e} — raw: {repr(raw_llm)}")
        return "Could not parse extraction result."

    license_plate = extracted.get("license_plate", "").strip()
    logger.info(f"Extracted plate={license_plate!r} type={extracted.get('vehicle_type')!r}")

    if not license_plate:
        return "No license plate detected in the image."

    return await _search_plate(license_plate)


async def process_text_message(text: str) -> str:
    logger.info(f"process_text_message: {repr(text)}")
    loop = asyncio.get_event_loop()

    def _invoke():
        llm         = _get_llm()
        messages    = [
            SystemMessage(content=(
                "You are an ALPR parking assistant. "
                "Answer questions about vehicle registration, parking subscriptions, and related topics concisely."
            )),
            HumanMessage(content=text),
        ]
        response = llm.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)

    reply = await loop.run_in_executor(None, _invoke)
    logger.info(f"Text reply: {repr(reply)}")
    return reply
