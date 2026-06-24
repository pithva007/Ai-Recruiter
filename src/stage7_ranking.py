# src/stage7_ranking.py
# Stage 7a: Explainable Ranking + Stage 7b: Dark Horse Discovery
#
# Input:
#   data/processed/scores/          — Stage 6 score files (one per candidate)
#   data/processed/retrieval_results.json — hybrid_rank per candidate
#   data/processed/evidence/        — evidence items (for dark horse analysis)
#   data/processed/jd_features.json — JD schema (for dark horse context)
#
# Output:
#   outputs/ranked_candidates.csv   — full 19-column CSV per SKILLS.md schema
#   outputs/ranking_summary.json    — summary stats
#
# LLM permitted: rationale generation (temp 0.0) + dark horse analysis (temp 0.0)
# Free-tier RPM respected via inter-call throttling.
#
# Usage:
#   python src/stage7_ranking.py
#   python src/stage7_ranking.py --force   # regenerate even if rationale exists

import argparse
import csv
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.llm_client import call_llm

ROOT              = os.path.join(os.path.dirname(__file__), "..")
DEFAULT_SCORES    = os.path.join(ROOT, "data", "processed", "scores")
DEFAULT_RETRIEVAL = os.path.join(ROOT, "data", "processed", "retrieval_results.json")
DEFAULT_EV_DIR    = os.path.join(ROOT, "data", "processed", "evidence")
DEFAULT_JD        = os.path.join(ROOT, "data", "processed", "jd_features.json")
DEFAULT_CSV       = os.path.join(ROOT, "outputs", "ranked_candidates.csv")
DEFAULT_SUMMARY   = os.path.join(ROOT, "outputs", "ranking_summary.json")

# SKILLS.md output CSV column order — exact
CSV_COLUMNS = [
    "rank", "candidate_id", "candidate_name", "composite_score",
    "fit_score", "impact_score", "potential_score", "risk_score",
    "confidence_level", "green_flags", "yellow_flags", "skill_gaps",
    "dark_horse", "dark_horse_reason", "transferable_skills_map",
    "interview_q1", "interview_q2", "interview_q3", "llm_rationale",
]

# Dark horse threshold from AGENT.md (exact)
DARK_HORSE_HYBRID_RANK_THRESHOLD = 15   # hybrid_rank > 15
DARK_HORSE_SCORE_THRESHOLD       = 75   # impact_score OR potential_score >= 75
DARK_HORSE_FIT_THRESHOLD         = 50   # fit_score >= 50

LLM_WAIT = 7  # seconds between LLM calls (free-tier RPM)

# ---------------------------------------------------------------------------
# System Prompt: Stage 7a — LLM Rationale (per task specification)
# ---------------------------------------------------------------------------
RATIONALE_SYSTEM_PROMPT = (
    "You are a recruiter writing a one-paragraph rationale (max 100 words) for why "
    "a candidate received this ranking. Be specific, reference their evidence. "
    "Do not be generic. Do not use the word 'candidate'. Use their name.\n\n"
    "Output a single JSON object: {\"rationale\": \"<text, max 100 words>\"}\n\n"
    "If you cannot find evidence for the rationale, write a honest brief summary. "
    "Do not invent or infer beyond what is stated."
)

# ---------------------------------------------------------------------------
# System Prompt: Stage 7b — Dark Horse Discovery (from original specification)
# ---------------------------------------------------------------------------
DARK_HORSE_SYSTEM_PROMPT = """You are a Dark Horse Discovery Agent. Given a candidate who ranked below position 15 in vector similarity search but has high impact or potential scores, determine if they are a true dark horse.

A dark horse is a candidate who would be missed by a traditional ATS but who a great recruiter would shortlist.

Analyze:
1. Does this candidate have transferable skills that map to JD requirements even if they never used the exact terminology?
2. Is there a non-obvious connection between their domain and the target role?
3. Does their growth trajectory suggest they would ramp quickly in this role?

Output:
{
  "is_dark_horse": <true | false>,
  "dark_horse_reason": "",
  "transferable_skills_map": [
    {"candidate_skill": "", "maps_to_jd_requirement": "", "mapping_reasoning": ""}
  ],
  "confidence": "<high | medium | low>"
}

If not a dark horse, set is_dark_horse to false and all other fields to null. Do not invent or infer beyond what is stated."""


# ---------------------------------------------------------------------------
# Load all score files
# ---------------------------------------------------------------------------
def load_scores(scores_dir: str) -> list[dict]:
    records = []
    for fname in sorted(os.listdir(scores_dir)):
        if not fname.endswith("_scores.json"):
            continue
        path = os.path.join(scores_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        if d.get("error") or d.get("composite_score") is None:
            continue   # skip error sentinels
        records.append(d)
    return records


# ---------------------------------------------------------------------------
# Load hybrid_rank lookup from retrieval results
# ---------------------------------------------------------------------------
def load_hybrid_ranks(retrieval_path: str) -> dict[str, int]:
    with open(retrieval_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {c["candidate_id"]: c["hybrid_rank"] for c in data["candidates"]}


# ---------------------------------------------------------------------------
# Load evidence items for a candidate
# ---------------------------------------------------------------------------
def load_evidence(ev_dir: str, cid: str) -> list[dict]:
    path = os.path.join(ev_dir, f"{cid}_evidence.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d.get("evidence", [])


# ---------------------------------------------------------------------------
# Generate LLM rationale (Stage 7a) — temperature 0.0
# ---------------------------------------------------------------------------
def generate_rationale(score: dict, rank: int, evidence: list, call_idx: int) -> str:
    """
    Generate a ≤100-word rationale string for this candidate's ranking.
    Falls back to rule-based string if LLM fails.
    """
    if call_idx > 0:
        time.sleep(LLM_WAIT)

    name     = score.get("candidate_name", score["candidate_id"])
    green    = "; ".join(score.get("green_flags", [])[:3])
    gaps     = "; ".join(score.get("skill_gaps", [])[:2])
    top_ev   = "; ".join(
        [e["claim"][:60] for e in evidence[:3] if e.get("confidence") in ("high", "medium")]
    )

    user_content = (
        f"Candidate name: {name}\n"
        f"Rank: {rank}\n"
        f"composite_score={score.get('composite_score')}  "
        f"fit={score.get('fit_score')}  impact={score.get('impact_score')}  "
        f"potential={score.get('potential_score')}  risk={score.get('risk_score')}\n"
        f"Green flags: {green or 'none'}\n"
        f"Skill gaps: {gaps or 'none'}\n"
        f"Top evidence: {top_ev or 'not available'}\n\n"
        f"Write a max-100-word rationale for why {name} ranked #{rank}."
    )

    try:
        raw = call_llm(
            system_prompt=RATIONALE_SYSTEM_PROMPT,
            user_content=user_content,
            temperature=0.0,
        )
        if isinstance(raw, dict):
            text = raw.get("rationale", "")
        elif isinstance(raw, str):
            text = raw
        else:
            text = ""
        if text.strip():
            # Enforce 100-word cap
            words = text.strip().split()
            if len(words) > 100:
                text = " ".join(words[:100])
            return text.strip()
    except Exception as e:
        print(f"\n  [Stage 7a] LLM rationale failed for {name}: {type(e).__name__}")

    # Rule-based fallback
    return (
        f"{name} ranked #{rank} with composite {score.get('composite_score')}. "
        f"Strengths: {green or 'strong ML background'}. "
        f"{'Gaps: ' + gaps + '.' if gaps else 'No significant skill gaps identified.'}"
    )[:500]


# ---------------------------------------------------------------------------
# Dark horse analysis (Stage 7b) — temperature 0.0
# ---------------------------------------------------------------------------
def analyse_dark_horse(score: dict, evidence: list, jd_features: dict, call_idx: int) -> dict:
    """
    Call Stage 7b LLM to confirm/deny dark horse status.
    Returns the raw LLM output dict.
    """
    if call_idx > 0:
        time.sleep(LLM_WAIT)

    name     = score.get("candidate_name", score["candidate_id"])
    jd_reqs  = [s.get("skill", "") for s in (jd_features.get("must_have_skills") or [])]

    user_content = (
        f"Candidate: {name}\n"
        f"hybrid_rank (from retrieval): {score.get('_hybrid_rank', '?')}\n"
        f"impact_score: {score.get('impact_score')}  potential_score: {score.get('potential_score')}\n"
        f"fit_score: {score.get('fit_score')}  risk_score: {score.get('risk_score')}\n\n"
        f"Green flags: {'; '.join(score.get('green_flags', []))}\n"
        f"Skill gaps: {'; '.join(score.get('skill_gaps', []))}\n\n"
        f"Evidence (top 8):\n"
        + "\n".join(
            f"  [{e.get('evidence_type')}] {e.get('claim', '')[:80]}"
            for e in evidence[:8]
        )
        + f"\n\nJD must-have skills: {', '.join(jd_reqs[:5])}"
    )

    try:
        raw = call_llm(
            system_prompt=DARK_HORSE_SYSTEM_PROMPT,
            user_content=user_content,
            temperature=0.0,
        )
        if isinstance(raw, list):
            raw = raw[0] if raw else {}
        return raw if isinstance(raw, dict) else {}
    except Exception as e:
        print(f"\n  [Stage 7b] Dark horse LLM failed for {name}: {type(e).__name__}")
        return {"is_dark_horse": False, "dark_horse_reason": None, "transferable_skills_map": None}


# ---------------------------------------------------------------------------
# Serialise list field to pipe-separated string for CSV
# ---------------------------------------------------------------------------
def pipe(lst) -> str:
    if not lst:
        return ""
    if isinstance(lst, list):
        # Handle list of dicts (transferable_skills_map)
        parts = []
        for item in lst:
            if isinstance(item, dict):
                parts.append(
                    f"{item.get('candidate_skill','?')} → {item.get('maps_to_jd_requirement','?')}"
                )
            else:
                parts.append(str(item))
        return " | ".join(parts)
    return str(lst)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 7a + 7b: Explainable Ranking and Dark Horse Discovery."
    )
    parser.add_argument("--scores",    default=DEFAULT_SCORES)
    parser.add_argument("--retrieval", default=DEFAULT_RETRIEVAL)
    parser.add_argument("--evidence",  default=DEFAULT_EV_DIR)
    parser.add_argument("--jd",        default=DEFAULT_JD)
    parser.add_argument("--csv",       default=DEFAULT_CSV)
    parser.add_argument("--summary",   default=DEFAULT_SUMMARY)
    parser.add_argument("--force",     action="store_true",
                        help="Regenerate rationales even if already present")
    args = parser.parse_args()

    for path, label in [
        (args.scores,    "scores dir"),
        (args.retrieval, "retrieval_results.json"),
        (args.evidence,  "evidence dir"),
        (args.jd,        "jd_features.json"),
    ]:
        if not os.path.exists(path):
            print(f"[Stage 7 ERROR] {label} not found: {path}")
            sys.exit(1)

    print("[Stage 7] Loading inputs ...")
    scores        = load_scores(args.scores)
    hybrid_ranks  = load_hybrid_ranks(args.retrieval)
    with open(args.jd, "r", encoding="utf-8") as f:
        jd_features = json.load(f)

    if not scores:
        print("[Stage 7 ERROR] No valid score files found.")
        sys.exit(1)

    # Attach hybrid_rank to each score record for dark horse logic
    for s in scores:
        s["_hybrid_rank"] = hybrid_ranks.get(s["candidate_id"], 999)

    # -----------------------------------------------------------------------
    # PART A — Sort by composite_score descending, assign rank
    # -----------------------------------------------------------------------
    scores.sort(key=lambda s: (-(s.get("composite_score") or 0), s["candidate_id"]))
    n = len(scores)
    print(f"[Stage 7a] Ranking {n} scored candidates ...")

    # -----------------------------------------------------------------------
    # PART B — Identify dark horse candidates (AGENT.md exact definition)
    # -----------------------------------------------------------------------
    dark_horse_ids: set[str] = set()
    for s in scores:
        hr  = s["_hybrid_rank"]
        imp = s.get("impact_score", 0) or 0
        pot = s.get("potential_score", 0) or 0
        fit = s.get("fit_score", 0) or 0
        if hr > DARK_HORSE_HYBRID_RANK_THRESHOLD and (imp >= DARK_HORSE_SCORE_THRESHOLD or
                pot >= DARK_HORSE_SCORE_THRESHOLD) and fit >= DARK_HORSE_FIT_THRESHOLD:
            dark_horse_ids.add(s["candidate_id"])

    print(f"[Stage 7b] Dark horse candidates identified: {len(dark_horse_ids)}")
    if dark_horse_ids:
        print(f"  IDs: {sorted(dark_horse_ids)}")

    # -----------------------------------------------------------------------
    # Build output rows with LLM calls
    # -----------------------------------------------------------------------
    output_rows = []
    llm_call_idx = 0

    for rank_num, score in enumerate(scores, start=1):
        cid   = score["candidate_id"]
        cname = score.get("candidate_name", cid)
        evidence = load_evidence(args.evidence, cid)

        # --- Stage 7a: Generate LLM rationale ---
        existing_rationale = score.get("_rationale", "")
        if existing_rationale and not args.force:
            rationale = existing_rationale
        else:
            rationale = generate_rationale(score, rank_num, evidence, llm_call_idx)
            llm_call_idx += 1

        # --- Stage 7b: Dark horse analysis (LLM) ---
        dark_horse_flag   = False
        dark_horse_reason = ""
        transferable_map  = []

        if cid in dark_horse_ids:
            dh_result = analyse_dark_horse(score, evidence, jd_features, llm_call_idx)
            llm_call_idx += 1

            if dh_result.get("is_dark_horse", False):
                dark_horse_flag   = True
                dark_horse_reason = dh_result.get("dark_horse_reason") or ""
                raw_map           = dh_result.get("transferable_skills_map") or []
                transferable_map  = raw_map if isinstance(raw_map, list) else []

        # --- Build questions ---
        questions = score.get("interview_questions", ["", "", ""])
        q1 = questions[0] if len(questions) > 0 else ""
        q2 = questions[1] if len(questions) > 1 else ""
        q3 = questions[2] if len(questions) > 2 else ""

        # --- Assemble CSV row (SKILLS.md column order) ---
        row = {
            "rank":                 rank_num,
            "candidate_id":         cid,
            "candidate_name":       cname,
            "composite_score":      score.get("composite_score", 0),
            "fit_score":            score.get("fit_score", 0),
            "impact_score":         score.get("impact_score", 0),
            "potential_score":      score.get("potential_score", 0),
            "risk_score":           score.get("risk_score", 0),
            "confidence_level":     score.get("confidence_level", "medium"),
            "green_flags":          pipe(score.get("green_flags", [])),
            "yellow_flags":         pipe(score.get("yellow_flags", [])),
            "skill_gaps":           pipe(score.get("skill_gaps", [])),
            "dark_horse":           dark_horse_flag,
            "dark_horse_reason":    dark_horse_reason,
            "transferable_skills_map": pipe(transferable_map),
            "interview_q1":         q1,
            "interview_q2":         q2,
            "interview_q3":         q3,
            "llm_rationale":        rationale,
        }
        output_rows.append(row)

        print(
            f"[Stage 7] Rank {rank_num:2d}  {cname:<22}  "
            f"composite={score.get('composite_score'):<6}  "
            f"{'🌟 DARK HORSE' if dark_horse_flag else ''}"
        )

    # -----------------------------------------------------------------------
    # PART C — Save outputs
    # -----------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)

    # Save ranked_candidates.csv
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in output_rows:
            writer.writerow(row)

    # Save ranking_summary.json
    composites         = [r["composite_score"] for r in output_rows]
    dark_horse_in_top  = [r["candidate_id"] for r in output_rows if r["dark_horse"]]
    top10_ids          = [r["candidate_id"] for r in output_rows[:10]]

    summary = {
        "total_candidates_scored": n,
        "top_10_candidate_ids":    top10_ids,
        "dark_horse_count":        len(dark_horse_in_top),
        "dark_horse_candidate_ids": dark_horse_in_top,
        "highest_composite_score": max(composites),
        "lowest_composite_score":  min(composites),
        "average_composite_score": round(sum(composites) / len(composites), 2),
    }
    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    top_name   = output_rows[0]["candidate_name"]
    top_score  = output_rows[0]["composite_score"]
    dh_count   = len(dark_horse_in_top)

    print(
        f"\n[Stage 7 Complete] Ranked {n} candidates. "
        f"Top candidate: {top_name} ({top_score}). "
        f"Dark horses found: {dh_count}"
    )
    print(f"[Stage 7] CSV    → {args.csv}")
    print(f"[Stage 7] Summary → {args.summary}")


if __name__ == "__main__":
    main()
