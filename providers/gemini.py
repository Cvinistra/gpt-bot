import asyncio
import os
import logging
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()

# История только в оперативной памяти процесса:
# ключ = (Telegram user_id, номер выбранного API).
_histories: dict[tuple[int, int], list[types.Content]] = {}
_clients: dict[int, genai.Client] = {}
_locks: dict[tuple[int, int], asyncio.Lock] = {}

MAX_HISTORY_MESSAGES = 20  # максимум 10 пар user/model на пользователя и API


def _key_for(number: int) -> str:
    return os.getenv(f"GEMINI_API_KEY_{number}", "").strip()


def available_keys() -> list[int]:
    return [number for number in (1, 2, 3) if _key_for(number)]


def _client_for(number: int) -> genai.Client:
    key = _key_for(number)
    if not key:
        raise GeminiUnavailableError("API key is not configured")
    if number not in _clients:
        # Ограничиваем сетевое ожидание, чтобы бот не зависал бесконечно.
        _clients[number] = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(timeout=15000),
        )
    return _clients[number]


class GeminiQuotaError(Exception):
    pass


class GeminiTemporaryError(Exception):
    pass


class GeminiUnavailableError(Exception):
    pass


def _error_details(exc: Exception) -> tuple[int | None, str]:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    message = str(exc).lower()
    if code is None:
        # SDK-исключения могут содержать HTTP-код только в тексте.
        for candidate in (400, 401, 403, 404, 408, 429, 500, 502, 503, 504):
            if str(candidate) in message:
                code = candidate
                break
    try:
        code = int(code) if code is not None else None
    except (TypeError, ValueError):
        code = None
    return code, message


def _translate_error(exc: Exception) -> Exception:
    code, message = _error_details(exc)
    if (
        "resource_exhausted" in message
        or "quota" in message
        or "billing" in message
        or "limit: 0" in message
    ):
        return GeminiQuotaError(str(exc))
    if code in (429, 503, 502, 504, 500, 408):
        return GeminiTemporaryError(str(exc))
    if code in (401, 403, 404, 400) or any(
        phrase in message for phrase in
        ("api key not valid", "invalid api key", "permission denied", "not found")
    ):
        return GeminiUnavailableError(str(exc))
    if code in (408, 500, 502, 503, 504) or isinstance(exc, (TimeoutError, ConnectionError)):
        return GeminiTemporaryError(str(exc))
    return GeminiUnavailableError(str(exc))


async def _ask_with_key(user_id: int, key_number: int, prompt: str) -> str:
    client = _client_for(key_number)
    history_key = (user_id, key_number)
    lock = _locks.setdefault(history_key, asyncio.Lock())

    async with lock:
        contents = list(_histories.get(history_key, []))
        contents.append(types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        ))
        try:
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=MODEL,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=(
                            "Ты полезный русскоязычный помощник. "
                            "Отвечай понятно и по делу. Если пользователь пишет на другом языке, "
                            "можешь отвечать на его языке."
                        ),
                    ),
                ),
                timeout=20,
            )
            answer = (response.text or "").strip()
            if not answer:
                raise GeminiTemporaryError("Gemini вернул пустой ответ")

            updated = contents + [types.Content(
                role="model",
                parts=[types.Part.from_text(text=answer)],
            )]
            _histories[history_key] = updated[-MAX_HISTORY_MESSAGES:]
            return answer
        except asyncio.TimeoutError as exc:
            raise GeminiTemporaryError("Превышено время ожидания ответа Gemini") from exc
        except (GeminiQuotaError, GeminiTemporaryError, GeminiUnavailableError):
            raise
        except Exception as exc:
            logger.warning("Gemini API %s request failed: %s", key_number, exc)
            raise _translate_error(exc) from exc


async def ask_gemini(user_id: int, preferred_key: int, prompt: str) -> tuple[str, int]:
    """Сначала использует выбранный ключ, затем автоматически пробует остальные."""
    keys = available_keys()
    if not keys:
        raise GeminiUnavailableError("Не настроен ни один Gemini API key")

    ordered = ([preferred_key] if preferred_key in keys else []) + [
        number for number in keys if number != preferred_key
    ]
    errors: list[Exception] = []

    for number in ordered:
        try:
            return await _ask_with_key(user_id, number, prompt), number
        except (GeminiQuotaError, GeminiTemporaryError, GeminiUnavailableError) as exc:
            errors.append(exc)
            logger.warning("Gemini API %s failed; trying next configured key", number)

    for error_type in (GeminiQuotaError, GeminiTemporaryError, GeminiUnavailableError):
        for exc in errors:
            if isinstance(exc, error_type):
                raise error_type(str(exc))
    raise GeminiUnavailableError("Все настроенные Gemini API недоступны")


def clear_user_history(user_id: int) -> None:
    for history_key in list(_histories):
        if history_key[0] == user_id:
            _histories.pop(history_key, None)


async def close_clients() -> None:
    for client in _clients.values():
        try:
            await client.aio.aclose()
        except Exception:
            logger.exception("Could not close Gemini client")
