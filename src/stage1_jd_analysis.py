# src/stage1_jd_analysis.py
# Phase A — JD Analysis Agent
#
# Input:  data/raw/job_description.docx
# Output: data/processed/jd_features.json
#
# Runs ONCE offline. LLM permitted. No time limit.
# Reads the actual challenge JD (job_description.docx) via python-docx,
# sends it to the LLM using the Phase A system prompt from CLAUDE.md,
# and saves the structured JD features for use by precompute.py and rank.py.
#
# Required env var: GEMINI_API_KEY

import json
import os
import sys

import docx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.llm_client import call_llm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.join(os.path.dirname(__file__), "..")
JD_DOCX_PATH   = os.path.join(ROOT, "data", "raw", "job_description.docx")
OUTPUT_PATH    = os.path.join(ROOT, "data", "processed", "jd_features.json")

# ---------------------------------------------------------------------------
# System prompt — Phase A from CLAUDE.md
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a Senior Technical Recruiter with deep expertise in AI/ML hiring.

Your job is to analyze the provided job description and produce a structured requirements
schema that will be used by an automated ranking system.

Output exactly this JSON structure:

{
  "role_title": "",
  "company": "",
  "seniority_level": "<junior | mid | senior | lead | principal>",
  "experience_range": {"min_years": 0, "max_years": 0, "ideal_years": 0},
  "location": {"primary_cities": [], "country": "", "work_mode": "<hybrid | onsite | remote>"},
  "must_have_skills": [
    {"skill": "", "context": "", "importance": "must_have"}
  ],
  "nice_to_have_skills": [
    {"skill": "", "context": "", "importance": "nice_to_have"}
  ],
  "explicit_disqualifiers": [
    {"pattern": "", "reason": ""}
  ],
  "services_company_names": [],
  "implicit_requirements": [
    {"requirement": "", "reasoning": ""}
  ],
  "culture_signals": [],
  "ideal_candidate_summary": "",
  "jd_trap_warning": ""
}

The jd_trap_warning field should describe any intentional traps or anti-patterns the JD
warns about (e.g., keyword-only matching without career history validation).

Extract the services company blacklist from any explicit company names mentioned as
disqualifiers in the JD.

If you cannot determine a field from the JD, output null. Do not invent or infer beyond
what is stated."""

# ---------------------------------------------------------------------------
# Required output fields
# ---------------------------------------------------------------------------
REQUIRED_FIELDS = [
    "role_title", "company", "seniority_level", "experience_range", "location",
    "must_have_skills", "nice_to_have_skills", "explicit_disqualifiers",
    "services_company_names", "implicit_requirements", "culture_signals",
    "ideal_candidate_summary", "jd_trap_warning",
]
LIST_FIELDS = {
    "must_have_skills", "nice_to_have_skills", "explicit_disqualifiers",
    "services_company_names", "implicit_requirements", "culture_signals",
}


def extract_jd_text(docx_path: str) -> str:
    """Extract all paragraph text from a .docx file."""
    doc = docx.Document(docx_path)
    lines = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n".join(lines)


def validate_jd_features(raw: dict) -> dict:
    """Ensure all required fields are present; default missing lists to []."""
    out = {}
    for field in REQUIRED_FIELDS:
        val = raw.get(field, None)
        if val is None and field in LIST_FIELDS:
            val = []
        out[field] = val
    return out


def run_jd_analysis(jd_text: str) -> dict:
    user_content = (
        "Analyze the following job description and produce the structured "
        "requirements schema as specified.\n\n"
        f"JOB DESCRIPTION:\n{jd_text}"
    )
    raw = call_llm(
        system_prompt=SYSTEM_PROMPT,
        user_content=user_content,
        temperature=0.0,
    )
    return validate_jd_features(raw)


def main() -> None:
    jd_path = os.path.abspath(JD_DOCX_PATH)
    if not os.path.exists(jd_path):
        print(f"[Stage 1 ERROR] JD file not found: {jd_path}")
        sys.exit(1)

    print("[Stage 1] Extracting text from job_description.docx ...")
    jd_text = extract_jd_text(jd_path)
    if not jd_text.strip():
        print("[Stage 1 ERROR] job_description.docx is empty.")
        sys.exit(1)

    print("[Stage 1] Running JD Analysis Agent ...")
    features = run_jd_analysis(jd_text)

    out_path = os.path.abspath(OUTPUT_PATH)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2, ensure_ascii=False)

    role  = features.get("role_title") or "unknown"
    level = features.get("seniority_level") or "unknown"
    must  = len(features.get("must_have_skills") or [])
    print(
        f"[Stage 1 Complete] JD features saved → data/processed/jd_features.json"
        f" | role={role!r} | level={level!r} | must_have_skills={must}"
    )


if __name__ == "__main__":
    main()
