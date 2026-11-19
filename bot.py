import asyncio
import logging
import os

from aiohttp import web
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import Update

from handlers import router
from providers.gemini import close_clients

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "").strip()


async def health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "service": "telegram-bot"})


async def telegram_webhook(request: web.Request) -> web.Response:
    # Проверяем секретный заголовок Telegram, чтобы посторонние не отправляли Update.
    if WEBHOOK_SECRET:
        supplied = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
        if supplied != WEBHOOK_SECRET:
            raise web.HTTPUnauthorized(text="Invalid webhook secret")

    try:
        data = await request.json()
        update = Update.model_validate(data, context={"bot": request.app["bot"]})
        await request.app["dispatcher"].feed_update(request.app["bot"], update)
    except Exception:
        logger.exception("Failed to process Telegram webhook update")
        # Возвращаем 200, чтобы Telegram не зацикливал повторную доставку
        # при внутренней ошибке обработчика. Ошибка остаётся в логах Render.
    return web.Response(text="OK")


async def on_startup(app: web.Application) -> None:
    bot: Bot = app["bot"]
    external_url = os.getenv("RENDER_EXTERNAL_URL", "").rstrip("/")
    if not external_url:
        raise RuntimeError(
            "Render не передал RENDER_EXTERNAL_URL. Проверь, что сервис создан как Web Service."
        )
    webhook_url = f"{external_url}/webhook"
    await bot.set_webhook(
        url=webhook_url,
        secret_token=WEBHOOK_SECRET or None,
        allowed_updates=app["dispatcher"].resolve_used_update_types(),
    )
    logger.info("Telegram webhook configured: %s/webhook", external_url)


async def on_cleanup(app: web.Application) -> None:
    # Не вызываем delete_webhook при остановке: во время перекатки Render
    # старый экземпляр мог бы удалить webhook, уже установленный новым.
    await close_clients()
    await app["bot"].session.close()


async def main() -> None:
    if not BOT_TOKEN:
        raise RuntimeError("Не задан BOT_TOKEN в переменных окружения.")

    bot = Bot(
        token=BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dispatcher = Dispatcher()
    dispatcher.include_router(router)

    app = web.Application()
    app["bot"] = bot
    app["dispatcher"] = dispatcher
    app.router.add_get("/", health)
    app.router.add_get("/health", health)
    app.router.add_post("/webhook", telegram_webhook)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    port = int(os.getenv("PORT", "10000"))
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    try:
        await site.start()
        logger.info("Web Service listening on 0.0.0.0:%s", port)
        # Ждём остановки процесса Render.
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
