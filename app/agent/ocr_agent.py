import asyncio
import base64
import json
import re
from typing import Any

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langchain.messages import RemoveMessage
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

from app.core.config import settings
from app.core.logger import logger

_llm: ChatOllama | None = None
_checkpointer    = InMemorySaver()
_agent           = None

TOKEN_LIMIT = 100000

_PLATE_RE   = re.compile(r'\b([A-Z]{1,3}[\s-]?\d{4,6}|\d{4,6}[A-Z]{0,2})\b')


def _get_llm() -> ChatOllama:
    global _llm
    if _llm is not None:
        return _llm
    logger.info(f"Loading ChatOllama model={settings.OLLAMA_MODEL}")
    _llm = ChatOllama(
        model       = settings.OLLAMA_MODEL,
        base_url    = settings.OLLAMA_BASE_URL,
        temperature = 0,
        reasoning   = False
    )
    return _llm


@before_model
def _trim_on_token_limit(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:  # noqa: ARG001
    messages = state["messages"]
    if len(messages) <= 1:
        return None

    llm = _get_llm()
    try:
        total = llm.get_num_tokens_from_messages(messages)
    except Exception:
        total = sum(len(str(m.content)) // 4 for m in messages)

    if total <= TOKEN_LIMIT:
        return None

    logger.info(f"History {total} tokens > {TOKEN_LIMIT}, trimming...")

    keep = list(messages)
    while len(keep) > 1:
        try:
            current = llm.get_num_tokens_from_messages(keep)
        except Exception:
            current = sum(len(str(m.content)) // 4 for m in keep)
        if current <= TOKEN_LIMIT:
            break
        keep.pop(1)

    logger.info(f"Trimmed to {len(keep)} messages")
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *keep,
        ]
    }


def _unwrap_tool_reply(raw) -> str:
    if isinstance(raw, list):
        return next((c["text"] for c in raw if isinstance(c, dict) and "text" in c), str(raw))
    if isinstance(raw, dict):
        return raw.get("raw_text") or raw.get("text") or str(raw)
    return str(raw)


async def _get_mcp_tools(server_key: str, url: str, timeout: float = 10.0) -> list:
    client = MultiServerMCPClient({
        server_key: {"url": url, "transport": "streamable_http"}
    })
    tools = await asyncio.wait_for(client.get_tools(), timeout=timeout)
    logger.info(f"Loaded {len(tools)} tool(s) from {server_key}: {[t.name for t in tools]}")
    return tools


async def _ocr_fallback(b64: str) -> str | None:
    """Call OCR MCP server, extract plate from raw text. Returns plate string or None."""
    logger.info("Falling back to OCR MCP server")
    try:
        tools    = await _get_mcp_tools("ocr", settings.OCR_MCP_URL)
        ocr_tool = next((t for t in tools if t.name == "ocr_image"), None)
        if ocr_tool is None:
            logger.warning("ocr_image tool not found on OCR MCP server")
            return None

        raw = await ocr_tool.ainvoke({"image_base64": b64, "task": "ocr"})
        text = _unwrap_tool_reply(raw)
        logger.info(f"OCR MCP raw text: {repr(text)}")

        match = _PLATE_RE.search(text.upper())
        if not match:
            logger.warning(f"No plate found in OCR text: {repr(text)}")
            return None

        plate = re.sub(r'[\s-]', '', match.group(1))
        logger.info(f"OCR fallback extracted plate: {plate!r}")
        return plate
    except Exception as e:
        logger.error(f"OCR fallback failed: {e}")
        return None


async def _search_plate(license_plate: str) -> str:
    tools       = await _get_mcp_tools("alpr", settings.ALPR_MCP_URL)
    search_tool = next((t for t in tools if t.name == "search_member_by_plate"), None)
    if search_tool is None:
        return "ALPR search tool unavailable."

    raw_reply = await search_tool.ainvoke({"license_plate": license_plate})
    logger.info(f"Search reply: {repr(raw_reply)}")
    return _unwrap_tool_reply(raw_reply)


async def _get_agent():
    global _agent
    if _agent is not None:
        return _agent
    tools  = await _get_mcp_tools("alpr", settings.ALPR_MCP_URL)
    _agent = create_agent(
        _get_llm(),
        tools         = tools,
        system_prompt = (
            "You are an ALPR parking assistant. "
            "Use search_member_by_plate when the user asks about a vehicle or license plate. "
            "Answer questions about vehicle registration, parking subscriptions, and related topics concisely. "
            "Remember previous messages in this conversation. "
            "IMPORTANT: Reply in plain text only. No markdown, no bold, no bullets, no asterisks, no headers. "
            "When returning tool results, copy them exactly as-is without rephrasing."
        ),
        middleware    = [_trim_on_token_limit],
        checkpointer  = _checkpointer,
    )
    logger.info("Conversational ALPR agent created")
    return _agent


async def process_vehicle_image(image_bytes: bytes, chat_id: int) -> str:
    logger.info(f"process_vehicle_image: {len(image_bytes)} bytes, chat_id={chat_id}")
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    # Step 1: Qwen vision extraction
    license_plate: str | None = None
    try:
        extractor = create_agent(
            _get_llm(),
            tools         = [],
            system_prompt = (
                "You are a vehicle license plate recognition assistant. "
                "Analyze the image. It may be a raw vehicle photo or a parking ticket/receipt. "
                "Your ONLY job is to extract the license plate number and vehicle type. "
                "You MUST reply with ONLY this exact JSON structure and nothing else — no explanation, no markdown, no extra text:\n"
                '{"license_plate": "ABC123", "vehicle_type": "CAR"}\n'
                "vehicle_type must be one of: CAR, MOTORCYCLE, TRUCK, BUS, VAN, OTHER. "
                "If you cannot find the license plate, use empty string. "
                "DO NOT include any text outside the JSON object."
            ),
        )

        result   = await asyncio.wait_for(extractor.ainvoke({
            "messages": [
                HumanMessage(content=[
                    {
                        "type"      : "image_url",
                        "image_url" : {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                    {"type": "text", "text": 'Reply with ONLY valid JSON: {"license_plate": "...", "vehicle_type": "..."}'},
                ])
            ]
        }), timeout=30.0)

        messages = result.get("messages", [])
        raw      = messages[-1].content if messages else ""
        logger.info(f"Extractor output: {repr(raw)}")

        json_match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if json_match:
            extracted     = json.loads(json_match.group())
            license_plate = extracted.get("license_plate", "").strip() or None
            logger.info(f"Qwen extracted plate={license_plate!r} type={extracted.get('vehicle_type')!r}")

    except Exception as e:
        logger.error(f"Qwen extraction failed: {e}")

    # Step 2: fallback to OCR MCP if Qwen failed
    if not license_plate:
        logger.info("Qwen did not extract plate, trying OCR MCP fallback")
        license_plate = await _ocr_fallback(b64)

    if not license_plate:
        return "No license plate detected in the image."

    reply = await _search_plate(license_plate)
    return "<code>" + reply.strip() + "</code>"


async def process_text_message(text: str, chat_id: int) -> str:
    logger.info(f"process_text_message chat_id={chat_id}: {repr(text)}")
    agent  = await _get_agent()
    config: RunnableConfig = {"configurable": {"thread_id": str(chat_id)}}
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=text)]},
        config,
    )
    reply = result["messages"][-1].content
    logger.info(f"Reply: {repr(reply)}")
    return "<code>" + reply.strip() + "</code>"
