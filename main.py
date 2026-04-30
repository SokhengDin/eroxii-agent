import asyncio

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.middleware.response_middleware import ResponseMiddleware
from app.core.config import settings
from bot.telegram_bot import build_telegram_app
from app import logger


async def _run_alpr_mcp():
    from app.mcp.alpr_server import mcp
    logger.info("Starting ALPR MCP server on :8003")
    await mcp.run_async(transport="streamable-http", host="0.0.0.0", port=8003)


async def _run_ocr_mcp():
    from app.mcp.ocr_server import run_server
    logger.info("Starting OCR MCP server on :8001")
    await asyncio.get_event_loop().run_in_executor(None, run_server)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up ...")

    from app.agent.ocr_agent import _get_llm
    _get_llm()
    logger.info("Qwen LLM initialized")

    alpr_mcp_task = asyncio.create_task(_run_alpr_mcp())
    logger.info("ALPR MCP server task started")

    ocr_mcp_task = asyncio.create_task(_run_ocr_mcp())
    logger.info("OCR MCP server task started")

    telegram_app = build_telegram_app()
    await telegram_app.initialize()
    await telegram_app.start()
    await telegram_app.updater.start_polling(drop_pending_updates=True)
    logger.info("Telegram bot polling started")

    yield

    await telegram_app.updater.stop()
    await telegram_app.stop()
    await telegram_app.shutdown()
    alpr_mcp_task.cancel()
    ocr_mcp_task.cancel()
    logger.info("Shutting down ...")


app = FastAPI(
    lifespan  = lifespan,
    title     = "eRoxii ALPR Agent",
    docs_url  = None if settings.ENV == "prod" else "/docs",
    redoc_url = None if settings.ENV == "prod" else "/redoc",
)

app.add_middleware(ResponseMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)
