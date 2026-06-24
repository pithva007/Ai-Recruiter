# utils/llm_client.py
# LLM client — Gemini backend via google-genai SDK.
# All pipeline stages must import and use call_llm() from this module.
# Do not duplicate this pattern elsewhere.
#
# Required env var: GEMINI_API_KEY  (loaded from .env via python-dotenv)
# Model: gemini-2.5-flash (primary reasoning model for all agent stages)

import json
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from tenacity import RetryError

# Load .env from the project root (two levels up: utils/ → root)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# Client picks up GEMINI_API_KEY from the environment automatically.
client = genai.Client()

# Primary model — override with GEMINI_MODEL env var
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Retry config
MAX_ATTEMPTS   = 6
BASE_WAIT_SEC  = 10   # seconds between first retries
MAX_WAIT_SEC   = 90   # ceiling


def _parse_retry_delay(exc: BaseException) -> float:
    """Extract retryDelay seconds from the API error response if present."""
    import re
    msg = str(exc)
    m = re.search(r"retryDelay.*?(\d+)s", msg)
    if m:
        return float(m.group(1)) + 2   # add 2s buffer
    return 0.0


def _is_transient(exc: BaseException) -> bool:
    """
    True for 429 (rate limit), 503 (server overloaded), and network errors.
    Unwraps RetryError from the SDK's internal tenacity and cause chains.
    """
    # Direct Gemini API error — check status code
    if isinstance(exc, (genai_errors.ServerError, genai_errors.ClientError)):
        code = getattr(exc, "code", 0) or 0
        return code in (429, 503)
    # Network-level errors (SSL drop, connection reset, timeout)
    if any(
        name in type(exc).__name__
        for name in ("ConnectError", "TimeoutError", "ConnectionError", "ReadError")
    ):
        return True
    # SDK wraps its own retries in tenacity RetryError — unwrap it
    if isinstance(exc, RetryError):
        inner = exc.last_attempt.exception()
        if inner and inner is not exc:
            return _is_transient(inner)
    # Generic cause chain unwrap
    cause = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
    if cause and cause is not exc:
        return _is_transient(cause)
    # Fallback: string match
    msg = str(exc)
    return (
        "429" in msg or "503" in msg
        or "UNAVAILABLE" in msg or "RESOURCE_EXHAUSTED" in msg
        or "SSL" in msg or "EOF" in msg or "timed out" in msg.lower()
    )


def _call_once(system_prompt: str, user_content: str, temperature: float) -> str:
    """Single API call — returns raw text. Raises on any error."""
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=4096,
        ),
    )
    return response.text.strip()


def call_llm(system_prompt: str, user_content: str, temperature: float = 0.0):
    """
    Call the Gemini LLM with a system prompt and user content.
    Returns a parsed JSON dict.

    Retry policy:
    - Up to 6 attempts on 429/503 with exponential backoff (10s → 90s)
    - Does NOT retry on auth/validation errors (400/401/403/404)
    - On JSON parse failure, makes one explicit JSON-only retry call
    """
    last_exc = None
    wait = BASE_WAIT_SEC
    raw = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            raw = _call_once(system_prompt, user_content, temperature)
            last_exc = None
            break   # success
        except Exception as exc:
            last_exc = exc
            if _is_transient(exc) and attempt < MAX_ATTEMPTS:
                # Respect retryDelay hint from the API if present
                api_wait = _parse_retry_delay(exc)
                actual_wait = max(wait, api_wait)
                print(
                    f"[llm_client] Attempt {attempt}/{MAX_ATTEMPTS} failed "
                    f"({type(exc).__name__}: {str(exc)[:100]}). "
                    f"Retrying in {actual_wait}s ..."
                )
                time.sleep(actual_wait)
                wait = min(wait * 2, MAX_WAIT_SEC)
            else:
                raise   # non-transient OR final attempt — propagate immediately

    if last_exc is not None:
        raise last_exc  # should not reach here, but safety net

    # Strip markdown fences if the model wraps its output
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # One explicit JSON-only retry
        try:
            raw2 = _call_once(
                system_prompt,
                (
                    f"{user_content}\n\n"
                    "Your previous response was not valid JSON. "
                    "Respond only with valid JSON. "
                    "No markdown. No explanation. No code blocks. "
                    "Just the raw JSON object."
                ),
                0.0,
            )
            raw2 = raw2.replace("```json", "").replace("```", "").strip()
            return json.loads(raw2)
        except Exception:
            raise ValueError(
                f"LLM returned non-JSON after retry. Raw output (first 300 chars):\n{raw[:300]}"
            )
