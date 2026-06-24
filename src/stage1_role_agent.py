# src/stage1_role_agent.py
# Stage 1: Role Understanding Agent
#
# Input:  data/raw/job_description.txt  (raw JD text)
# Output: data/processed/jd_schema.json (structured requirement schema)
#
# References:
#   - System prompt:  CLAUDE.md  § "System Prompt: Stage 1 — Role Understanding Agent"
#   - Output schema:  AGENT.md   § "Output Format Contract" / Stage 1 JSON structure
#   - LLM client:     SKILLS.md  § "LLM Client Pattern" → utils/llm_client.py
#
# Required env var: GEMINI_API_KEY

import json
import os
import sys

# Allow imports from the project root regardless of working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.llm_client import call_llm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_JD_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "job_description.txt")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "jd_schema.json")

# ---------------------------------------------------------------------------
# System prompt — copied verbatim from CLAUDE.md § Stage 1
# ---------------------------------------------------------------------------
STAGE1_SYSTEM_PROMPT = """You are a Role Understanding Agent. Your job is to deeply analyze a job description and produce a structured requirement schema. You are NOT extracting keywords. You are reasoning about what a great recruiter would understand this role to actually need.

Output exactly this JSON structure:
{
  "role_title": "",
  "seniority_level": "<junior | mid | senior | lead | principal>",
  "explicit_requirements": [
    {"skill": "", "importance": "<must_have | nice_to_have>", "context": ""}
  ],
  "implicit_requirements": [
    {"requirement": "", "reasoning": ""}
  ],
  "domain_context": "",
  "culture_signals": [],
  "red_flags_for_candidates": [],
  "ideal_candidate_summary": ""
}

Implicit requirements are things the JD does not say directly but that a smart recruiter would know this role needs. Example: a JD for a 'founding engineer' implicitly requires comfort with ambiguity, ability to work without process, and willingness to do non-engineering tasks.

If you cannot determine a field from the JD, output null. Do not invent or infer beyond what is stated."""


# ---------------------------------------------------------------------------
# Required top-level fields for the JD schema (from AGENT.md)
# ---------------------------------------------------------------------------
REQUIRED_FIELDS = [
    "role_title",
    "seniority_level",
    "explicit_requirements",
    "implicit_requirements",
    "domain_context",
    "culture_signals",
    "red_flags_for_candidates",
    "ideal_candidate_summary",
]


def validate_jd_schema(raw: dict) -> dict:
    """
    Ensure every required field is present in the returned dict.
    Missing fields are set to null (None) — never raise a KeyError.
    List fields that are None are normalised to an empty list.
    """
    list_fields = {"explicit_requirements", "implicit_requirements",
                   "culture_signals", "red_flags_for_candidates"}

    validated = {}
    for field in REQUIRED_FIELDS:
        value = raw.get(field, None)
        # Normalise absent list fields to [] rather than None
        if value is None and field in list_fields:
            value = []
        validated[field] = value

    return validated


def run_stage1(jd_text: str) -> dict:
    """
    Stage 1 core logic.

    Args:
        jd_text: Raw job description text.

    Returns:
        Validated JD schema dict (matches AGENT.md schema contract).
    """
    user_content = (
        "Analyze the following job description and produce the structured "
        "requirement schema as specified.\n\n"
        f"JOB DESCRIPTION:\n{jd_text}"
    )

    raw_result = call_llm(
        system_prompt=STAGE1_SYSTEM_PROMPT,
        user_content=user_content,
        temperature=0.0,
    )

    return validate_jd_schema(raw_result)


def save_jd_schema(schema: dict, output_path: str) -> None:
    """Persist the validated JD schema to disk as pretty-printed JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)


def main() -> None:
    """
    Entry point for Stage 1.
    Reads data/raw/job_description.txt → runs the Role Understanding Agent
    → saves to data/processed/jd_schema.json.
    """
    # --- Read raw JD ---
    jd_path = os.path.abspath(RAW_JD_PATH)
    if not os.path.exists(jd_path):
        print(f"[Stage 1 ERROR] Job description file not found: {jd_path}")
        sys.exit(1)

    with open(jd_path, "r", encoding="utf-8") as f:
        jd_text = f.read().strip()

    if not jd_text:
        print("[Stage 1 ERROR] job_description.txt is empty.")
        sys.exit(1)

    print("[Stage 1] Running Role Understanding Agent...")

    # --- Run stage ---
    schema = run_stage1(jd_text)

    # --- Save output ---
    output_path = os.path.abspath(OUTPUT_PATH)
    save_jd_schema(schema, output_path)

    # --- Console confirmation (as specified) ---
    role_title = schema.get("role_title") or "unknown"
    seniority = schema.get("seniority_level") or "unknown"
    print(
        f"[Stage 1 Complete] JD schema saved to data/processed/jd_schema.json"
        f" | role_title={role_title!r}"
        f" | seniority_level={seniority!r}"
    )


if __name__ == "__main__":
    main()
