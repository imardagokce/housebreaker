import asyncio
import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar

import httpx
import yaml
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)
DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"


class FetcherError(Exception):
    """Base exception for API fetcher failures."""


class FetcherConfigurationError(FetcherError):
    """Raised when a fetcher cannot be configured."""


class FetcherHTTPError(FetcherError):
    """Raised when an API returns an unsuccessful HTTP status."""


class FetcherAPIError(FetcherError):
    """Raised when an API reports an application-level error."""


class FetcherResponseError(FetcherError):
    """Raised when an API response is invalid or incomplete."""


def load_rate_limit(api_name: str, settings_path: str | Path | None = None) -> int:
    path = Path(settings_path) if settings_path is not None else DEFAULT_SETTINGS_PATH
    try:
        with path.open(encoding="utf-8") as settings_file:
            settings = yaml.safe_load(settings_file)
    except (OSError, yaml.YAMLError) as exc:
        logger.error("Unable to load API rate limits from %s.", path)
        raise FetcherConfigurationError(
            f"Unable to load API rate limits from {path}."
        ) from exc

    rate_limits = settings.get("api_rate_limits") if isinstance(settings, Mapping) else None
    api_settings = rate_limits.get(api_name) if isinstance(rate_limits, Mapping) else None
    requests_per_minute = (
        api_settings.get("requests_per_minute")
        if isinstance(api_settings, Mapping)
        else None
    )
    if (
        isinstance(requests_per_minute, bool)
        or not isinstance(requests_per_minute, int)
        or requests_per_minute < 1
    ):
        logger.error("The configured requests-per-minute limit for %s is invalid.", api_name)
        raise FetcherConfigurationError(
            f"Configure api_rate_limits.{api_name}.requests_per_minute "
            "as a positive integer."
        )
    return requests_per_minute


class AsyncRateLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        if requests_per_minute < 1:
            raise ValueError("requests_per_minute must be a positive integer.")
        self._minimum_interval = 60.0 / requests_per_minute
        self._next_request_time = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait_time = self._next_request_time - now
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self._next_request_time = time.monotonic() + self._minimum_interval


async def get_json(
    client: httpx.AsyncClient,
    rate_limiter: AsyncRateLimiter,
    endpoint: str,
    params: Mapping[str, str | int | float] | None = None,
    headers: Mapping[str, str] | None = None,
) -> Any:
    await rate_limiter.acquire()
    try:
        response = await client.get(endpoint, params=params, headers=headers)
    except httpx.TimeoutException as exc:
        logger.error("The request to %s timed out.", endpoint)
        raise FetcherHTTPError(f"The request to {endpoint} timed out.") from exc
    except httpx.RequestError as exc:
        logger.error("The request to %s failed (%s).", endpoint, type(exc).__name__)
        raise FetcherHTTPError(
            f"The request to {endpoint} failed ({type(exc).__name__})."
        ) from exc

    if response.is_error:
        logger.error("The request to %s returned HTTP %s.", endpoint, response.status_code)
        raise FetcherHTTPError(
            f"The request to {endpoint} returned HTTP {response.status_code}."
        )

    try:
        return response.json()
    except ValueError as exc:
        logger.error("The response from %s was not valid JSON.", endpoint)
        raise FetcherResponseError(
            f"The response from {endpoint} was not valid JSON."
        ) from exc


def validate_model(
    model: type[ModelT], payload: Any, endpoint: str
) -> ModelT:
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors(
                include_input=False, include_context=False, include_url=False
            )
        )
        logger.error("The response from %s failed validation: %s", endpoint, details)
        raise FetcherResponseError(
            f"The response from {endpoint} failed validation: {details}"
        ) from exc


def validate_list(
    model: type[ModelT], payload: Any, endpoint: str
) -> list[ModelT]:
    if not isinstance(payload, list):
        logger.error("The response from %s was not a list as expected.", endpoint)
        raise FetcherResponseError(
            f"The response from {endpoint} was not a list as expected."
        )
    try:
        return [model.model_validate(item) for item in payload]
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors(
                include_input=False, include_context=False, include_url=False
            )
        )
        logger.error("The response from %s failed validation: %s", endpoint, details)
        raise FetcherResponseError(
            f"The response from {endpoint} failed validation: {details}"
        ) from exc


def get_api_response(payload: Any, endpoint: str) -> list[dict[str, Any]]:
    envelope = validate_model(FootballAPIEnvelope, payload, endpoint)
    if envelope.errors:
        logger.error("The API reported an error for %s: %s", endpoint, envelope.errors)
        raise FetcherAPIError(f"The API reported an error for {endpoint}: {envelope.errors}")
    return envelope.response


class FootballAPIEnvelope(BaseModel):
    errors: Any = None
    response: list[dict[str, Any]]
