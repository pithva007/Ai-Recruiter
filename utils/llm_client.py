# utils/llm_client.py
# LLM client — Gemini backend via google-genai SDK.
# All pipeline stages must import and use call_llm() from this module.
# Do not duplicate this pattern elsewhere.
#
# Required env var: GEMINI_API_KEY  (loaded from .env via python-dotenv)
# Model: gemini-2.0-flash (primary reasoning model for all agent stages)

import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

# Load .env from the project root (two levels up from this file: utils/ → root)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# Client picks up GEMINI_API_KEY from the environment automatically.
client = genai.Client()

# Primary model — fast, capable, cost-effective for production pipelines.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_llm(system_prompt: str, user_content: str, temperature: float = 0.0) -> dict:
    """
    Call the Gemini LLM with a system prompt and user content.
    Returns a parsed JSON dict.

    Behaviour:
    - temperature=0.0 for all scoring/extraction tasks (per CLAUDE.md)
    - temperature=0.3 for interview question generation (caller sets this)
    - On JSON parse failure, retries once with an explicit JSON-only instruction
    - Tenacity retries up to 3× on any transient exception (network, rate-limit)
    """
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=4096,
        ),
    )

    raw = response.text.strip()
    # Strip markdown code fences if the model wraps the JSON
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # One explicit retry: ask for clean JSON only
        retry_response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=(
                f"{user_content}\n\n"
                "Your previous response was not valid JSON. "
                "Respond only with valid JSON. "
                "No markdown. No explanation. No code blocks. "
                "Just the raw JSON object."
            ),
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.0,
                max_output_tokens=4096,
            ),
        )
        cleaned = retry_response.text.strip()
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)
