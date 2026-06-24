# src/stage6_scoring_engine.py
# Stage 6: LLM Hiring Intelligence Engine (Scoring)
#
# Input:
#   data/processed/retrieval_results.json  — Stage 5 output; use top hybrid_rank <= 30
#   data/raw/candidates.jsonl              — real candidate data (profile, skills, career)
#   data/processed/evidence/              — Stage 3 evidence items per candidate
#   data/processed/jd_features.json       — Stage 1 JD schema
#
# Output per candidate:
#   data/processed/scores/{candidate_id}_scores.json
#
# LLM permitted here. One call per candidate.
# Free-tier RPM respected via inter-call throttling.
#
# Usage:
#   python src/stage6_scoring_engine.py
#   python src/stage6_scoring_engine.py --top-k 10   # dev: score only top 10
#   python src/stage6_scoring_engine.py --force       # re-score even if file exists

import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.llm_client import call_llm
from utils.json_validator import CandidateScore
from pydantic import ValidationError

ROOT              = os.path.join(os.path.dirname(__file__), "..")
DEFAULT_RETRIEVAL = os.path.join(ROOT, "data", "processed", "retrieval_results.json")
DEFAULT_JSONL     = os.path.join(ROOT, "data", "raw",       "candidates.jsonl")
DEFAULT_EV_DIR    = os.path.join(ROOT, "data", "processed", "evidence")
DEFAULT_JD        = os.path.join(ROOT, "data", "processed", "jd_features.json")
DEFAULT_OUT_DIR   = os.path.join(ROOT, "data", "processed", "scores")

TOP_K             = 30          # process candidates with hybrid_rank <= TOP_K
LLM_WAIT          = 7           # seconds between LLM calls (free-tier RPM)

# ---------------------------------------------------------------------------
# System prompt: Stage 6 — Hiring Intelligence Engine (Scoring)
# From the original specification referenced in CLAUDE.md
# ---------------------------------------------------------------------------
SCORING_SYSTEM_PROMPT = """You are a Hiring Intelligence Engine. Given a job requirement schema (from Stage 1), a candidate structured profile (from Stage 2), and a list of evidence items (from Stage 3), produce a scoring assessment for this candidate.

Scoring rules:
- fit_score (0-100): How well does the candidate's evidence match the JD's explicit and implicit requirements? Only score on evidence you can see. Missing evidence = lower score, not assumed score.
- impact_score (0-100): Sum of quantified impact signals. A candidate with zero quantified impact signals scores maximum 40 on this dimension.
- potential_score (0-100): Apply this formula — (career_velocity * 0.4) + (complexity_growth * 0.3) + (self_learning_signals * 0.3). Normalize each sub-factor to 0-100 before weighting. complexity_growth = your assessment (0-100) of whether project complexity increased across the candidate's career chronologically.
- risk_score (0-100): Higher = more risk. Score based on: skill gaps vs JD must-haves, very short tenures, no collaboration evidence, domain mismatch, overqualification signals.

Output exactly:
{
  "candidate_id": "",
  "fit_score": <int>,
  "impact_score": <int>,
  "potential_score": <int>,
  "risk_score": <int>,
  "fit_reasoning": "",
  "impact_reasoning": "",
  "potential_reasoning": "",
  "risk_reasoning": "",
  "green_flags": [],
  "yellow_flags": [],
  "skill_gaps": [],
  "confidence_level": "<high | medium | low>"
}

If you cannot determine a score from available evidence, set it to 50 and set confidence_level to low. Do not invent evidence. Do not infer beyond what is stated."""

# ---------------------------------------------------------------------------
# System prompt: Interview Question Generation
# From CLAUDE.md / original specification
# ---------------------------------------------------------------------------
INTERVIEW_SYSTEM_PROMPT = """You are a Technical Recruiter preparing interview questions. Given a candidate's profile, their evidence items, their skill gaps, and the job requirements, generate exactly 3 interview questions tailored to THIS specific candidate.

Rules:
- Question 1: Probe their strongest evidence claim. Ask them to go deeper on their most impressive achievement.
- Question 2: Probe their biggest skill gap identified in scoring. Design a question that reveals whether the gap is real or just missing from their resume.
- Question 3: Probe a behavioral signal — how they handle a situation relevant to this role's implicit requirements.

Output as a JSON array of 3 strings. No preamble. No explanation. Just the 3 questions."""


# ---------------------------------------------------------------------------
# Load top-K candidates from retrieval results
# ---------------------------------------------------------------------------
def load_top_candidates(retrieval_path: str, top_k: int) -> list[dict]:
    with open(retrieval_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    candidates = [c for c in data["candidates"] if c["hybrid_rank"] <= top_k]
    candidates.sort(key=lambda c: c["hybrid_rank"])
    return candidates


# ---------------------------------------------------------------------------
# Stream JSONL to collect real candidate records for top-K ids
# ---------------------------------------------------------------------------
def load_candidate_records(jsonl_path: str, target_ids: set) -> dict[str, dict]:
    records = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
                cid = c.get("candidate_id", "")
                if cid in target_ids:
                    records[cid] = c
                    if len(records) == len(target_ids):
                        break
            except json.JSONDecodeError:
                continue
    return records


# ---------------------------------------------------------------------------
# Build a condensed candidate profile dict to send to the LLM
# ---------------------------------------------------------------------------
def build_candidate_profile(raw: dict) -> dict:
    """
    Condense the real candidate record into a profile summary for the LLM.
    Includes: profile fields, career_history, skills, education, redrob_signals.
    """
    profile  = raw.get("profile", {})
    signals  = raw.get("redrob_signals", {})
    career   = raw.get("career_history", [])

    return {
        "candidate_id":      raw.get("candidate_id", ""),
        "candidate_name":    profile.get("anonymized_name", ""),
        "current_title":     profile.get("current_title", ""),
        "current_company":   profile.get("current_company", ""),
        "years_experience":  profile.get("years_of_experience", 0),
        "location":          profile.get("location", ""),
        "country":           profile.get("country", ""),
        "summary":           profile.get("summary", ""),
        "career_history": [
            {
                "title":            r.get("title", ""),
                "company":          r.get("company", ""),
                "duration_months":  r.get("duration_months", 0),
                "industry":         r.get("industry", ""),
                "is_current":       r.get("is_current", False),
                "description":      r.get("description", ""),
            }
            for r in career
        ],
        "skills": [
            {
                "name":             s.get("name", ""),
                "proficiency":      s.get("proficiency", ""),
                "endorsements":     s.get("endorsements", 0),
                "duration_months":  s.get("duration_months", 0),
            }
            for s in raw.get("skills", [])
        ],
        "education": [
            {
                "degree":         e.get("degree", ""),
                "field_of_study": e.get("field_of_study", ""),
                "institution":    e.get("institution", ""),
                "tier":           e.get("tier", "unknown"),
                "end_year":       e.get("end_year", ""),
            }
            for e in raw.get("education", [])
        ],
        "certifications": raw.get("certifications", []),
        "redrob_signals": {
            "open_to_work":              signals.get("open_to_work_flag", False),
            "last_active_date":          signals.get("last_active_date", ""),
            "recruiter_response_rate":   signals.get("recruiter_response_rate", 0),
            "notice_period_days":        signals.get("notice_period_days", 90),
            "github_activity_score":     signals.get("github_activity_score", -1),
            "skill_assessment_scores":   signals.get("skill_assessment_scores", {}),
            "interview_completion_rate": signals.get("interview_completion_rate", 0.5),
        },
    }


# ---------------------------------------------------------------------------
# Load evidence items for a candidate
# ---------------------------------------------------------------------------
def load_evidence(ev_dir: str, candidate_id: str) -> list[dict]:
    path = os.path.join(ev_dir, f"{candidate_id}_evidence.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d.get("evidence", [])


# ---------------------------------------------------------------------------
# Score one candidate — LLM call (temperature 0.0)
# ---------------------------------------------------------------------------
def score_candidate(
    jd_schema:         dict,
    candidate_profile: dict,
    evidence_items:    list,
) -> CandidateScore:
    """
    Send JD schema + candidate profile + evidence to the LLM scoring engine.
    Returns a validated CandidateScore.
    """
    user_content = (
        f"JD Requirements Schema:\n{json.dumps(jd_schema, indent=2)}\n\n"
        f"Candidate Profile:\n{json.dumps(candidate_profile, indent=2)}\n\n"
        f"Evidence Items:\n{json.dumps(evidence_items, indent=2)}\n\n"
        "Score this candidate on all four dimensions following the scoring rules."
    )

    raw = call_llm(
        system_prompt=SCORING_SYSTEM_PROMPT,
        user_content=user_content,
        temperature=0.0,
    )

    # Normalise: LLM may return a list accidentally — take first element
    if isinstance(raw, list):
        raw = raw[0] if raw else {}

    # Inject candidate_id if LLM omitted it
    if not raw.get("candidate_id"):
        raw["candidate_id"] = candidate_profile.get("candidate_id", "")

    # Validate and clamp scores via CandidateScore pydantic model
    try:
        scored = CandidateScore(**raw)
    except (ValidationError, Exception) as e:
        # Fallback: build minimal valid record with default scores
        print(f"\n  [Stage 6 WARNING] Validation failed, using defaults: {str(e)[:80]}")
        scored = CandidateScore(
            candidate_id    = candidate_profile.get("candidate_id", ""),
            fit_score       = raw.get("fit_score", 50),
            impact_score    = raw.get("impact_score", 50),
            potential_score = raw.get("potential_score", 50),
            risk_score      = raw.get("risk_score", 50),
            confidence_level= "low",
            green_flags     = raw.get("green_flags", []) or [],
            yellow_flags    = raw.get("yellow_flags", []) or [],
            skill_gaps      = raw.get("skill_gaps", []) or [],
        )

    scored.compute_composite()
    return scored


# ---------------------------------------------------------------------------
# Generate interview questions — LLM call (temperature 0.3)
# ---------------------------------------------------------------------------
def generate_interview_questions(
    candidate_profile: dict,
    evidence_items:    list,
    skill_gaps:        list,
    jd_schema:         dict,
) -> list[str]:
    """
    Generate 3 tailored interview questions for this candidate.
    Returns a list of 3 strings. Falls back to defaults on failure.
    """
    user_content = (
        f"Candidate: {candidate_profile.get('candidate_name', 'Candidate')}\n"
        f"Title: {candidate_profile.get('current_title', '')}\n"
        f"Years: {candidate_profile.get('years_experience', 0)}\n\n"
        f"Evidence Items:\n{json.dumps(evidence_items[:10], indent=2)}\n\n"
        f"Skill Gaps:\n{json.dumps(skill_gaps, indent=2)}\n\n"
        f"JD Must-Have Skills:\n"
        f"{json.dumps([s['skill'] for s in (jd_schema.get('must_have_skills') or [])], indent=2)}"
    )

    try:
        raw = call_llm(
            system_prompt=INTERVIEW_SYSTEM_PROMPT,
            user_content=user_content,
            temperature=0.3,
        )
        if isinstance(raw, list) and len(raw) >= 3:
            return [str(q).strip() for q in raw[:3]]
        if isinstance(raw, dict):
            questions = raw.get("questions") or raw.get("interview_questions") or []
            if len(questions) >= 3:
                return [str(q).strip() for q in questions[:3]]
    except Exception as e:
        print(f"\n  [Stage 6 WARNING] Interview questions LLM failed: {type(e).__name__}")

    # Rule-based fallback — always 3 specific questions
    title  = candidate_profile.get("current_title", "your role")
    gaps   = skill_gaps[:1] if skill_gaps else ["production deployment experience"]
    return [
        f"Walk me through your most impactful project as a {title} — what was the measurable outcome?",
        f"We noticed {gaps[0]} may be a gap. Can you describe any experience you have there, even informally?",
        "Tell me about a time you had to ship something quickly without full certainty — how did you decide what to cut?",
    ]


# ---------------------------------------------------------------------------
# Save score file
# ---------------------------------------------------------------------------
def save_score(score_data: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    cid = score_data["candidate_id"]
    path = os.path.join(out_dir, f"{cid}_scores.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(score_data, f, indent=2, ensure_ascii=False)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 6: LLM Hiring Intelligence Engine — score top candidates."
    )
    parser.add_argument("--retrieval", default=DEFAULT_RETRIEVAL)
    parser.add_argument("--jsonl",     default=DEFAULT_JSONL)
    parser.add_argument("--evidence",  default=DEFAULT_EV_DIR)
    parser.add_argument("--jd",        default=DEFAULT_JD)
    parser.add_argument("--out",       default=DEFAULT_OUT_DIR)
    parser.add_argument("--top-k",     type=int, default=TOP_K,
                        help=f"Score candidates with hybrid_rank <= this (default {TOP_K})")
    parser.add_argument("--force",     action="store_true",
                        help="Re-score even if scores file already exists")
    args = parser.parse_args()

    for path, label in [
        (args.retrieval, "retrieval_results.json"),
        (args.jsonl,     "candidates.jsonl"),
        (args.evidence,  "evidence dir"),
        (args.jd,        "jd_features.json"),
    ]:
        if not os.path.exists(path):
            print(f"[Stage 6 ERROR] {label} not found: {path}")
            sys.exit(1)

    # Load JD schema
    with open(args.jd, "r", encoding="utf-8") as f:
        jd_schema = json.load(f)

    # Load top-K candidates from retrieval
    top_candidates = load_top_candidates(args.retrieval, args.top_k)
    total = len(top_candidates)
    print(f"[Stage 6] Scoring top {total} candidates (hybrid_rank <= {args.top_k})")
    print(f"[Stage 6] Loading candidate records from JSONL ...")

    target_ids = {c["candidate_id"] for c in top_candidates}
    candidate_records = load_candidate_records(args.jsonl, target_ids)
    print(f"[Stage 6] Found {len(candidate_records)}/{total} candidates in JSONL\n")

    success = 0
    failed  = 0

    for i, ret_row in enumerate(top_candidates, start=1):
        cid = ret_row["candidate_id"]

        # Resume support
        out_path = os.path.join(args.out, f"{cid}_scores.json")
        if os.path.exists(out_path) and not args.force:
            with open(out_path) as f:
                existing = json.load(f)
            cname = existing.get("candidate_name", cid)
            print(f"[Stage 6] {cid} ({cname}): already scored — skipping")
            success += 1
            continue

        raw_candidate = candidate_records.get(cid)
        if raw_candidate is None:
            print(f"[Stage 6 WARNING] {cid} not found in JSONL — skipping")
            failed += 1
            continue

        candidate_profile = build_candidate_profile(raw_candidate)
        evidence_items    = load_evidence(args.evidence, cid)
        cname             = candidate_profile.get("candidate_name", cid)

        # Throttle between LLM calls
        if i > 1:
            time.sleep(LLM_WAIT)

        try:
            # --- Score call (temperature 0.0) ---
            scored = score_candidate(jd_schema, candidate_profile, evidence_items)

            # Second throttle before interview questions call
            time.sleep(LLM_WAIT)

            # --- Interview questions call (temperature 0.3) ---
            questions = generate_interview_questions(
                candidate_profile = candidate_profile,
                evidence_items    = evidence_items,
                skill_gaps        = scored.skill_gaps,
                jd_schema         = jd_schema,
            )

            # Build full score document
            score_doc = {
                "candidate_id":      cid,
                "candidate_name":    cname,
                "hybrid_rank":       ret_row["hybrid_rank"],
                "fit_score":         scored.fit_score,
                "impact_score":      scored.impact_score,
                "potential_score":   scored.potential_score,
                "risk_score":        scored.risk_score,
                "composite_score":   scored.composite_score,
                "confidence_level":  scored.confidence_level,
                "fit_reasoning":     scored.fit_reasoning,
                "impact_reasoning":  scored.impact_reasoning,
                "potential_reasoning": scored.potential_reasoning,
                "risk_reasoning":    scored.risk_reasoning,
                "green_flags":       scored.green_flags,
                "yellow_flags":      scored.yellow_flags,
                "skill_gaps":        scored.skill_gaps,
                "interview_questions": questions,
            }

            save_score(score_doc, args.out)
            success += 1

            print(
                f"[Stage 6] Scored {cname}: "
                f"composite={scored.composite_score}, "
                f"fit={scored.fit_score}, "
                f"impact={scored.impact_score}, "
                f"potential={scored.potential_score}, "
                f"risk={scored.risk_score}"
            )

        except Exception as e:
            print(
                f"\n[Stage 6 ERROR] {cname} ({cid}) failed: "
                f"{type(e).__name__}: {str(e)[:100]}"
            )
            traceback.print_exc()
            failed += 1
            # Write sentinel so we know this was attempted
            save_score(
                {
                    "candidate_id":    cid,
                    "candidate_name":  cname,
                    "error":           str(e),
                    "composite_score": None,
                },
                args.out,
            )

    print(
        f"\n[Stage 6 Complete] Scored {success}/{total} candidates "
        f"{f'  ({failed} failed)' if failed else ''}"
    )
    print(f"[Stage 6] Score files → {args.out}")


if __name__ == "__main__":
    main()
