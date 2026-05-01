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
_checkpointer           = InMemorySaver()
_agent                  = None

TOKEN_LIMIT = 100000
_PLATE_RE   = re.compile(r'\b([A-Z]{1,3}[\s-]?\d{4,6}|\d{4,6}[A-Z]{0,2})\b')

_SYSTEM_PROMPT = (
    "You are Rex, eRoxii's sharp and friendly parking assistant. "
    "You are helpful, concise, and occasionally witty — but always professional. "
    "When a plate is found, present the info clearly. When it's not found, say so directly. "
    "You remember the conversation and can answer follow-up questions.\n\n"
    "TOOLS:\n"
    "- search_member_by_plate(license_plate): Member info + subscription for an exact plate.\n"
    "- search_plate_full_info(search_term, limit): Search by plate OR name — full session + captures.\n"
    "- search_parking_records(search_term, start_date, end_date, limit): Free-text search with optional ISO 8601 date range.\n"
    "- get_latest_records(limit, vehicle_type): Recent records across all plates. vehicle_type: CAR, MOTORCYCLE, TRUCK, BUS, VAN.\n"
    "- read_resource(uri): Read any resource by URI when the plate is already known.\n\n"
    "RESOURCE URIs (pass to read_resource):\n"
    "- alpr://member/{license_plate} — member + subscription info\n"
    "- alpr://plate/{license_plate}/full-info — full user + vehicle + latest captures\n"
    "- alpr://plate/{license_plate}/latest-detection — live detection with is_valid flag\n"
    "- alpr://plate/{license_plate}/captures — entry/exit capture list\n"
    "- alpr://plate/{license_plate}/session-history — parking session history\n"
    "- alpr://latest-records — recent records across all plates\n\n"
    "Decision guide:\n"
    "- Exact plate, need member/subscription → read_resource('alpr://member/{plate}')\n"
    "- Exact plate, need full session + photos → read_resource('alpr://plate/{plate}/full-info')\n"
    "- Exact plate, need captures only → read_resource('alpr://plate/{plate}/captures')\n"
    "- Exact plate, need session history → read_resource('alpr://plate/{plate}/session-history')\n"
    "- Exact plate, need live status → read_resource('alpr://plate/{plate}/latest-detection')\n"
    "- Name or partial plate → search_plate_full_info\n"
    "- Date range query → search_parking_records\n"
    "- Recent activity all plates → get_latest_records or read_resource('alpr://latest-records')\n\n"
    "IMPORTANT: Reply in plain text only. No markdown, no bold, no bullets, no asterisks, no headers. "
    "Copy tool results exactly as-is without rephrasing or summarizing."
)


def _get_llm() -> ChatOllama:
    global _llm
    if _llm is not None:
        return _llm
    logger.info(f"Loading ChatOllama model={settings.OLLAMA_MODEL}")
    _llm = ChatOllama(
        model       = settings.OLLAMA_MODEL,
        base_url    = settings.OLLAMA_BASE_URL,
        temperature = 0,
        reasoning   = False,
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
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *keep]}


def _unwrap_tool_reply(raw) -> str:
    if isinstance(raw, list):
        return next((c["text"] for c in raw if isinstance(c, dict) and "text" in c), str(raw))
    if isinstance(raw, dict):
        return raw.get("raw_text") or raw.get("text") or str(raw)
    return str(raw)


def _alpr_client_config() -> dict:
    return {
        "alpr": {
            "url"      : settings.ALPR_MCP_URL,
            "transport": "streamable_http",
        }
    }


async def _load_alpr_tools(timeout: float = 10.0) -> list:
    """Load ALPR tools via a fresh stateless client call."""
    client = MultiServerMCPClient(_alpr_client_config())
    tools  = await asyncio.wait_for(client.get_tools(), timeout=timeout)
    logger.info(f"Loaded {len(tools)} ALPR tool(s): {[t.name for t in tools]}")
    return tools


async def _get_agent(force_reload: bool = False):
    global _agent
    if _agent is not None and not force_reload:
        return _agent
    tools  = await _load_alpr_tools()
    _agent = create_agent(
        _get_llm(),
        tools         = tools,
        system_prompt = _SYSTEM_PROMPT,
        middleware    = [_trim_on_token_limit],
        checkpointer  = _checkpointer,
    )
    logger.info("Conversational ALPR agent created")
    return _agent


async def _ocr_fallback(b64: str) -> str | None:
    logger.info("Falling back to OCR MCP server")
    try:
        client   = MultiServerMCPClient({"ocr": {"url": settings.OCR_MCP_URL, "transport": "streamable_http"}})
        tools    = await asyncio.wait_for(client.get_tools(), timeout=10.0)
        ocr_tool = next((t for t in tools if t.name == "ocr_image"), None)
        if ocr_tool is None:
            logger.warning("ocr_image tool not found on OCR MCP server")
            return None

        raw  = await ocr_tool.ainvoke({"image_base64": b64, "task": "ocr"})
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
    try:
        tools       = await _load_alpr_tools()
        search_tool = next((t for t in tools if t.name == "search_member_by_plate"), None)
        if search_tool is None:
            return "ALPR search tool unavailable."
        raw_reply = await search_tool.ainvoke({"license_plate": license_plate})
        logger.info(f"Search reply: {repr(raw_reply)}")
        return _unwrap_tool_reply(raw_reply)
    except Exception as e:
        logger.error(f"_search_plate failed: {e}")
        return f"Search failed: {e}"


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
        result = await asyncio.wait_for(extractor.ainvoke({
            "messages": [HumanMessage(content=[
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": 'Reply with ONLY valid JSON: {"license_plate": "...", "vehicle_type": "..."}'},
            ])]
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

    # Step 2: OCR MCP fallback
    if not license_plate:
        logger.info("Qwen did not extract plate, trying OCR MCP fallback")
        license_plate = await _ocr_fallback(b64)

    # Step 3: no plate — let agent freely describe the image
    if not license_plate:
        logger.info("No plate found, routing image to agent for free description")
        try:
            agent  = await _get_agent()
            config: RunnableConfig = {"configurable": {"thread_id": str(chat_id)}}
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content=[
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": "No license plate was detected in this image. Describe what you see freely and helpfully."},
                ])]},
                config,
            )
            return result["messages"][-1].content.strip()
        except Exception as e:
            logger.error(f"Free description failed: {e}")
            # Reload agent with fresh tools on next call
            await _get_agent(force_reload=True)
            return "Could not analyze the image. Please try again."

    reply = await _search_plate(license_plate)
    return "<code>" + reply.strip() + "</code>"


async def process_text_message(text: str, chat_id: int) -> str:
    logger.info(f"process_text_message chat_id={chat_id}: {repr(text)}")
    try:
        agent  = await _get_agent()
        config: RunnableConfig = {"configurable": {"thread_id": str(chat_id)}}
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=text)]},
            config,
        )
        reply = result["messages"][-1].content
        logger.info(f"Reply: {repr(reply)}")
        return "<code>" + reply.strip() + "</code>"
    except Exception as e:
        logger.error(f"process_text_message failed: {e}")
        # Force reload agent on next call in case tools are stale
        await _get_agent(force_reload=True)
        return "Something went wrong. Please try again."
