"""
OpenRouter LLM client using the OpenAI-compatible SDK.
Supports configurable models, retry logic, and structured output.
"""

import asyncio
import json

from openai import AsyncOpenAI, APIError, APIConnectionError, RateLimitError

from app.config import get_settings
from app.core.exceptions import LLMError
from app.core.logging import get_logger

logger = get_logger(__name__)

# Retry configuration
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0  # seconds


class LLMClient:
    """Async client for OpenRouter LLM API calls."""

    def __init__(self) -> None:
        settings = get_settings()
        self._model = settings.OPENROUTER_MODEL
        self._temperature = settings.LLM_TEMPERATURE
        self._max_tokens = settings.LLM_MAX_TOKENS
        self._client = AsyncOpenAI(
            base_url=settings.OPENROUTER_BASE_URL,
            api_key=settings.OPENROUTER_API_KEY,
            default_headers={
                "HTTP-Referer": "https://mentoria.onrender.com",
                "X-Title": "MentorIA",
            },
        )

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send a chat completion request to OpenRouter.

        Args:
            system_prompt: The system/persona instructions.
            user_prompt: The user query with context.
            model: Override the default model.
            temperature: Override the default temperature.
            max_tokens: Override the default max tokens.

        Returns:
            The LLM's response text.

        Raises:
            LLMError: If all retries fail.
        """
        target_model = model or self._model
        target_temp = temperature if temperature is not None else self._temperature
        target_max = max_tokens or self._max_tokens

        logger.info("Calling LLM: model=%s, temp=%.2f, max_tokens=%d", target_model, target_temp, target_max)

        last_error: Exception | None = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await self._client.chat.completions.create(
                    model=target_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=target_temp,
                    max_tokens=target_max,
                )

                content = response.choices[0].message.content
                if not content:
                    raise LLMError("LLM returned an empty response")

                logger.info(
                    "LLM response received: %d characters (model: %s)",
                    len(content),
                    response.model,
                )
                return content.strip()

            except RateLimitError as exc:
                last_error = exc
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Rate limited (attempt %d/%d). Retrying in %.1fs...",
                    attempt, MAX_RETRIES, delay,
                )
                await asyncio.sleep(delay)

            except APIConnectionError as exc:
                last_error = exc
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Connection error (attempt %d/%d). Retrying in %.1fs...",
                    attempt, MAX_RETRIES, delay,
                )
                await asyncio.sleep(delay)

            except APIError as exc:
                last_error = exc
                logger.error("API error from OpenRouter: %s", exc)
                raise LLMError(
                    f"OpenRouter API error: {exc.message}",
                    detail=str(exc),
                ) from exc

            except Exception as exc:
                last_error = exc
                logger.error("Unexpected LLM error: %s", exc)
                raise LLMError(
                    "Unexpected error during LLM call",
                    detail=str(exc),
                ) from exc

        raise LLMError(
            f"LLM call failed after {MAX_RETRIES} retries",
            detail=str(last_error),
        )

    async def stream(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        """Stream chat completion responses from OpenRouter.

        Args:
            system_prompt: The system/persona instructions.
            user_prompt: The user query with context.
            model: Override the default model.
            temperature: Override the default temperature.
            max_tokens: Override the default max tokens.

        Yields:
            Text chunks from the LLM.
        """
        target_model = model or self._model
        target_temp = temperature if temperature is not None else self._temperature
        target_max = max_tokens or self._max_tokens

        logger.info("Streaming from LLM: model=%s", target_model)

        try:
            response = await self._client.chat.completions.create(
                model=target_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=target_temp,
                max_tokens=target_max,
                stream=True,
            )

            async for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    yield content

        except Exception as exc:
            logger.error("LLM streaming failed: %s", exc)
            raise LLMError("Error during LLM streaming", detail=str(exc)) from exc

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
    ) -> dict:
        """Generate a response and parse it as JSON.

        Args:
            system_prompt: The system/persona instructions.
            user_prompt: The user query with context.
            model: Override the default model.

        Returns:
            Parsed JSON dictionary from the LLM response.

        Raises:
            LLMError: If the response cannot be parsed as JSON.
        """
        raw = await self.generate(system_prompt, user_prompt, model=model)

        # Strip markdown code fences if the LLM wraps the JSON
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last lines (code fences)
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse LLM JSON response: %s", exc)
            logger.debug("Raw LLM response:\n%s", raw[:2000])
            raise LLMError(
                "LLM response is not valid JSON",
                detail=f"Parse error: {exc}. Raw response (first 500 chars): {raw[:500]}",
            ) from exc
