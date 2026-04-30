import traceback

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from app.agent.ocr_agent import process_vehicle_image, process_text_message
from app.core.config import settings
from app.core.logger import logger


def _is_allowed(chat_id: int, thread_id: int | None) -> bool:
    if settings.TELEGRAM_ALLOWED_CHAT_IDS and chat_id not in settings.TELEGRAM_ALLOWED_CHAT_IDS:
        return False
    if settings.TELEGRAM_ALLOWED_THREAD_IDS and thread_id not in settings.TELEGRAM_ALLOWED_THREAD_IDS:
        return False
    return True


def _log_exc(e: BaseException, label: str):
    if isinstance(e, ExceptionGroup):
        for sub in e.exceptions:
            _log_exc(sub, label)
    else:
        logger.error(f"{label}: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())


async def _cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id   = update.effective_chat.id
    thread_id = update.message.message_thread_id if update.message else None
    if not _is_allowed(chat_id, thread_id):
        return
    await update.message.reply_text(
        "ALPR Agent ready. Send a photo of a vehicle or parking receipt to look up its license plate."
    )


async def _handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id   = update.effective_chat.id
    thread_id = update.message.message_thread_id if update.message else None

    if not _is_allowed(chat_id, thread_id):
        logger.warning(f"Rejected photo from chat_id={chat_id} thread_id={thread_id}")
        return

    photo       = update.message.photo[-1]
    file        = await context.bot.get_file(photo.file_id)
    image_bytes = bytes(await file.download_as_bytearray())

    # await update.message.reply_text("Analyzing image...", message_thread_id=thread_id)

    try:
        response = await process_vehicle_image(image_bytes, chat_id)
        await update.message.reply_text(response, message_thread_id=thread_id, parse_mode='HTML')
    except* Exception as eg:
        _log_exc(eg, "Photo handler error")
        await update.message.reply_text("Failed to process the image.", message_thread_id=thread_id)


async def _handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id   = update.effective_chat.id
    thread_id = update.message.message_thread_id if update.message else None

    if not _is_allowed(chat_id, thread_id):
        logger.warning(f"Rejected text from chat_id={chat_id} thread_id={thread_id}")
        return

    try:
        response = await process_text_message(update.message.text, chat_id)
        await update.message.reply_text(response, message_thread_id=thread_id, parse_mode='HTML')
    except* Exception as eg:
        _log_exc(eg, "Text handler error")
        await update.message.reply_text("Failed to process your message.", message_thread_id=thread_id)


def build_telegram_app() -> Application:
    app = (
        Application.builder()
        .token(settings.TELEGRAM_BOT_TOKEN)
        .build()
    )
    app.add_handler(CommandHandler("start", _cmd_start))
    app.add_handler(MessageHandler(filters.PHOTO, _handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_text))
    return app
