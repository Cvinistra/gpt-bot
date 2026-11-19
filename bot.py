import asyncio
import logging
import os

from aiohttp import web
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage

from handlers import router
from providers.gemini import close_clients

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


async def health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "service": "telegram-bot"})


async def start_health_server() -> web.AppRunner:
    # Render Web Service требует, чтобы приложение слушало PORT на 0.0.0.0.
    port = int(os.getenv("PORT", "10000"))
    app = web.Application()
    app.router.add_get("/", health)
    app.router.add_get("/health", health)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host="0.0.0.0", port=port)
    await site.start()
    logger.info("Health web server is listening on 0.0.0.0:%s", port)
    return runner


async def main():
    token = os.getenv("BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Не задан BOT_TOKEN в переменных окружения.")

    bot = Bot(
        token=token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)

    web_runner = None
    try:
        web_runner = await start_health_server()
        logger.info("Gemini Telegram bot is starting")
        await dp.start_polling(bot)
    finally:
        if web_runner is not None:
            await web_runner.cleanup()
        await close_clients()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
