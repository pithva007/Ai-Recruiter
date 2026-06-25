# src/reason.py
# Phase C — Reasoning Generation for Top 100
#
# Input:  outputs/ranked_top100_raw.csv  (from rank.py — no reasoning column)
#         data/raw/candidates.jsonl      (full candidate profiles for rich prompts)
#         data/processed/features.pkl   (optional snapshot fallback for safe_fallback())
# Output: outputs/submission.csv         (final submission with reasoning)
#
# LLM is permitted here — only 100 candidates, runs offline after ranking.
# Temperature: 0.3 for ranks 1-30, 0.5 for ranks 31-100 (more variation for lower ranks).
# Falls back to rule-based reasoning on LLM failure or hallucination (max 2 retries).
#
# Usage:
#   python src/reason.py                          # LLM mode
#   python src/reason.py --no-llm                 # rule-based only, instant
#   python src/reason.py --raw outputs/ranked_top100_raw.csv --out outputs/submission.csv
#
# After this runs, validate with:
#   python validate_submission.py outputs/submission.csv

import argparse
import csv
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.llm_client import call_llm

ROOT             = os.path.join(os.path.dirname(__file__), "..")
DEFAULT_RAW      = os.path.join(ROOT, "outputs", "ranked_top100_raw.csv")
DEFAULT_CANDS    = os.path.join(ROOT, "data",    "raw",       "candidates.jsonl")
DEFAULT_FEATURES = os.path.join(ROOT, "data",    "processed", "features.pkl")
DEFAULT_OUT      = os.path.join(ROOT, "outputs", "submission.csv")

# Inter-call throttle: respect free-tier RPM limits
LLM_INTER_CALL_WAIT = 7   # seconds between LLM calls


# ---------------------------------------------------------------------------
# THE REASONING SYSTEM PROMPT — used EXACTLY as specified
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a senior technical recruiter writing 1-2 sentence reasoning for why a candidate is ranked at a specific position for this job opening.

The job: Senior AI Engineer — Founding Team at Redrob AI (Series A)
Requirements: 5-9 years experience, production retrieval/search/ranking systems, embeddings, vector databases, Python. Product company experience preferred. Located in or willing to relocate to Pune/Noida/Hyderabad/Mumbai/Delhi NCR. JD explicitly rejects pure-services careers, pure research with no production, and LangChain-only AI experience.

Rules you must follow:
1. Write exactly 1-2 sentences. No more.
2. Reference at least 2 specific facts from the profile: use the actual years_of_experience, actual current_title, actual company name, actual skill names with duration, actual redrob signal values (last_active_date, notice_period_days, recruiter_response_rate).
3. Connect the reasoning to the JD: mention retrieval/search/ranking or explain why this candidate does or does not have it.
4. For ranks 1-20: lead with the strongest positive signal, then note one concern if any exists.
5. For ranks 21-60: balance positives and concerns equally.
6. For ranks 61-100: lead with the concern or gap, end with why they made the list at all.
7. NEVER mention a skill, company, or technology not present in the provided profile data.
8. NEVER use these phrases: "strong candidate", "great fit", "impressive background", "solid experience", "excellent skills". These are generic and penalized.
9. DO acknowledge notice period if > 60 days, location if outside preferred cities, or if last_active_date was > 60 days ago.
10. Vary your sentence structure — no two reasoning entries should start the same way.

Output: exactly 1-2 sentences of plain text. No quotes, no prefixes, no JSON."""


# ---------------------------------------------------------------------------
# Build user prompt — rich profile data for the LLM
# ---------------------------------------------------------------------------
def build_user_prompt(candidate: dict, rank: int, score: float) -> str:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    skills  = candidate.get("skills", [])
    career  = candidate.get("career_history", [])

    # Extract top 5 most relevant skills (by JD relevance, then duration)
    RELEVANT_SKILL_KEYWORDS = {
        "embedding", "vector", "retrieval", "search", "ranking", "recommendation",
        "faiss", "pinecone", "qdrant", "milvus", "elasticsearch", "transformer",
        "bert", "llm", "fine-tun", "rag", "python", "pytorch", "tensorflow",
        "lora", "qlora", "xgboost", "learning to rank", "ndcg"
    }
    relevant_skills = [
        s for s in skills
        if any(kw in s.get("name", "").lower() for kw in RELEVANT_SKILL_KEYWORDS)
    ]
    relevant_skills.sort(
        key=lambda s: (s.get("duration_months", 0), s.get("endorsements", 0)),
        reverse=True,
    )
    top_skills = relevant_skills[:5] if relevant_skills else skills[:5]

    # Format career history (last 2 roles only, descriptions truncated to 150 chars)
    recent_career = sorted(career, key=lambda j: j.get("start_date", ""), reverse=True)[:2]
    career_text = "\n".join([
        f"- {j.get('title')} at {j.get('company')} "
        f"({j.get('duration_months', 0)} months, "
        f"{'current' if j.get('is_current') else 'past'}): "
        f"{str(j.get('description', ''))[:150]}"
        for j in recent_career
    ])

    # Format skills
    skills_text = ", ".join([
        f"{s.get('name')} ({s.get('duration_months', 0)}mo, {s.get('endorsements', 0)} endorsements)"
        for s in top_skills
    ])

    # Format key signals
    last_active       = signals.get("last_active_date", "unknown")
    notice            = signals.get("notice_period_days", "unknown")
    response_rate     = signals.get("recruiter_response_rate", "unknown")
    open_to_work      = signals.get("open_to_work_flag", False)
    location          = profile.get("location", "unknown")
    willing_relocate  = signals.get("willing_to_relocate", False)

    return f"""CANDIDATE PROFILE:
Name: {profile.get('anonymized_name', 'Unknown')}
Rank: {rank} | Score: {score:.4f}
Current title: {profile.get('current_title', 'unknown')} at {profile.get('current_company', 'unknown')} ({profile.get('current_company_size', 'unknown')})
Total experience: {profile.get('years_of_experience', 'unknown')} years
Location: {location} | Willing to relocate: {willing_relocate}
Industry: {profile.get('current_industry', 'unknown')}
Summary (first 300 chars): {str(profile.get('summary', ''))[:300]}

RECENT CAREER:
{career_text}

RELEVANT SKILLS:
{skills_text}

KEY BEHAVIORAL SIGNALS:
Last active: {last_active}
Open to work: {open_to_work}
Notice period: {notice} days
Recruiter response rate: {response_rate}

Write 1-2 sentences explaining rank {rank} for this candidate."""


# ---------------------------------------------------------------------------
# Hallucination guard
# ---------------------------------------------------------------------------
def check_hallucination(reasoning: str, candidate: dict) -> bool:
    """
    Returns True if reasoning mentions a company/tech not in the candidate profile.
    Checks for the most common hallucination: prestigious companies invented by the LLM.
    """
    valid_terms: set = set()

    for job in candidate.get("career_history", []):
        valid_terms.add(job.get("company", "").lower())
        valid_terms.add(job.get("title", "").lower())
    for skill in candidate.get("skills", []):
        valid_terms.add(skill.get("name", "").lower())

    profile = candidate.get("profile", {})
    valid_terms.add(str(profile.get("years_of_experience", "")))
    valid_terms.add(profile.get("current_company", "").lower())
    valid_terms.add(profile.get("current_title", "").lower())

    # Remove empty strings
    valid_terms.discard("")

    # Check for common hallucination patterns — prestigious companies not in profile
    hallucination_signals = ["google", "meta", "openai", "microsoft", "amazon", "apple"]
    for company in hallucination_signals:
        if company in reasoning.lower() and company not in valid_terms:
            return True   # Hallucinated a prestigious company
    return False


# ---------------------------------------------------------------------------
# Safe fallback — uses ONLY verified profile fields (no LLM)
# ---------------------------------------------------------------------------
def safe_fallback(candidate: dict, rank: int) -> str:
    """
    Template-based reasoning using only verified fields.
    Used when LLM fails after 2 retries or hallucinates persistently.
    """
    profile  = candidate.get("profile", {})
    signals  = candidate.get("redrob_signals", {})
    name     = profile.get("anonymized_name", "Candidate")
    title    = profile.get("current_title", "professional")
    company  = profile.get("current_company", "current employer")
    yoe      = profile.get("years_of_experience", "?")
    notice   = signals.get("notice_period_days", "?")
    location = profile.get("location", "unknown location")

    if rank <= 30:
        return (
            f"{name} brings {yoe} years of experience as {title} at {company}; "
            f"ranked {rank} based on career-history alignment with retrieval/ranking "
            f"systems and platform availability signals."
        )
    else:
        return (
            f"{name} ({title} at {company}, {yoe} years, {location}) has adjacent "
            f"experience but gaps in the specific production retrieval and search-system "
            f"background the JD requires; included at rank {rank} based on partial "
            f"skill-signal overlap."
        )


# ---------------------------------------------------------------------------
# Rule-based fallback (features.pkl snapshot — used when no full candidate)
# ---------------------------------------------------------------------------
def _rule_based_reasoning(rec: dict, rank_num: int) -> str:
    """
    Build reasoning purely from features.pkl snapshot fields.
    References actual data — never invents anything.
    """
    title   = rec.get("current_title", "Engineer")
    yoe     = rec.get("years_experience", 0)
    rr      = rec.get("recruiter_rr", 0)
    notice  = rec.get("notice_days", 90)
    open_w  = rec.get("open_to_work", False)
    skills  = rec.get("top_skills", [])

    parts = [f"{title} with {yoe} yrs exp"]
    if skills:
        parts.append(f"skills: {', '.join(skills[:3])}")
    parts.append(f"response rate {rr:.2f}")

    if notice <= 30:
        parts.append(f"notice {notice}d")
    elif notice > 90:
        parts.append(f"notice {notice}d is a concern")

    if not open_w and rank_num > 50:
        parts.append("not flagged open-to-work")

    return "; ".join(parts) + "."


# ---------------------------------------------------------------------------
# LLM reasoning call — plain text output, with hallucination guard + retries
# ---------------------------------------------------------------------------
def generate_reasoning_llm(
    candidate: dict,
    rank: int,
    score: float,
    features_rec: dict,
) -> tuple[str, bool]:
    """
    Calls LLM with the judge-facing system prompt and rich user prompt.
    Temperature: 0.3 for ranks 1-30, 0.5 for ranks 31-100.
    Runs hallucination guard. Max 2 retries on hallucination.
    Returns (reasoning_text, used_llm).
    """
    temperature = 0.3 if rank <= 30 else 0.5
    user_prompt = build_user_prompt(candidate, rank, score)

    for attempt in range(3):   # initial + up to 2 retries
        try:
            # call_llm returns parsed JSON dict by default.
            # Our new SYSTEM_PROMPT asks for plain text — so we call _call_once directly.
            from utils.llm_client import _call_once
            raw_text = _call_once(SYSTEM_PROMPT, user_prompt, temperature)
            # Strip any accidental JSON wrapper, quotes, markdown
            raw_text = (
                raw_text
                .replace("```json", "")
                .replace("```", "")
                .strip()
                .strip('"')
                .strip("'")
            )
            if not raw_text:
                break   # empty — fall through to safe_fallback

            # Hallucination guard
            if check_hallucination(raw_text, candidate):
                print(f"    [HALLUCINATION detected attempt {attempt + 1}] regenerating...")
                if attempt < 2:
                    time.sleep(2)
                    continue   # retry
                else:
                    # Exhausted retries — use safe_fallback
                    return safe_fallback(candidate, rank), False

            return raw_text, True

        except Exception as e:
            print(f"\n  [LLM ERROR attempt {attempt + 1}] {type(e).__name__}: {str(e)[:120]}")
            if attempt < 2:
                time.sleep(10)
            else:
                break   # fall through to safe_fallback

    # Final safety net
    if candidate:
        return safe_fallback(candidate, rank), False
    return _rule_based_reasoning(features_rec, rank), False


# ---------------------------------------------------------------------------
# Load candidates.jsonl → {candidate_id: dict}
# ---------------------------------------------------------------------------
def load_candidates_lookup(candidates_path: str) -> dict:
    """
    Stream candidates.jsonl and build an ID→candidate dict.
    Only loads the 100 IDs we actually need to save memory.
    But since we need it as a lookup, we do a single pass and load all.
    """
    import re
    CAND_ID_PAT = re.compile(r"^CAND_[0-9]{7}$")
    lookup = {}
    print(f"[reason.py] Loading candidate profiles from {candidates_path} ...")

    try:
        import gzip
        opener = gzip.open if candidates_path.endswith(".gz") else open
        with opener(candidates_path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    c = json.loads(line)
                    cid = c.get("candidate_id", "")
                    if CAND_ID_PAT.match(cid):
                        lookup[cid] = c
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"[reason.py WARNING] Could not load candidates.jsonl: {e}")

    print(f"[reason.py] Loaded {len(lookup):,} candidate profiles.")
    return lookup


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def run_reasoning(
    raw_csv: str,
    candidates_path: str,
    features_path: str,
    output_path: str,
    use_llm: bool = True,
) -> None:
    # Load raw ranking CSV
    with open(raw_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"[reason.py] Loaded {len(rows)} ranked candidates from {raw_csv}")

    # Load full candidate profiles for rich prompts
    candidates_lookup: dict = {}
    if use_llm and os.path.exists(candidates_path):
        candidates_lookup = load_candidates_lookup(candidates_path)
    elif use_llm:
        print(f"[reason.py WARNING] candidates.jsonl not found at {candidates_path}. "
              f"LLM prompts will be degraded.")

    # Load features.pkl snapshot (fallback data source for rule-based)
    features: dict = {}
    if os.path.exists(features_path):
        with open(features_path, "rb") as f:
            features = pickle.load(f)
        print(f"[reason.py] Loaded {len(features):,} feature snapshots from {features_path}")
    else:
        print(f"[reason.py WARNING] features.pkl not found at {features_path}. "
              f"Rule-based fallbacks will use minimal data.")

    mode_label = "LLM + hallucination guard + safe fallback" if use_llm else "rule-based only (instant)"
    print(f"[reason.py] Mode: {mode_label}")
    if use_llm:
        print(f"[reason.py] Temperature: 0.3 for ranks 1-30, 0.5 for ranks 31-100")
        print(f"[reason.py] Inter-call wait: {LLM_INTER_CALL_WAIT}s (RPM compliance)")
    print(f"[reason.py] Generating reasoning for {len(rows)} candidates ...\n")

    final_rows   = []
    llm_count    = 0
    fallback_count = 0
    total_chars  = 0

    for i, row in enumerate(rows, start=1):
        cid      = row["candidate_id"]
        rank_num = int(row["rank"])
        score    = float(row["score"])

        # Get full candidate profile (for rich LLM prompt + hallucination guard)
        candidate    = candidates_lookup.get(cid, {})
        features_rec = features.get(cid, {"candidate_id": cid})

        # Preview for console
        title_preview = (
            candidate.get("profile", {}).get("current_title", "?")
            if candidate else features_rec.get("current_title", "?")
        )
        print(
            f"  [{i:3d}/100] {cid}  rank={rank_num:3d}  score={score:.4f}  "
            f"{str(title_preview)[:28]:<28}",
            end="",
            flush=True,
        )

        # Throttle between LLM calls (not before the very first one)
        if use_llm and i > 1:
            time.sleep(LLM_INTER_CALL_WAIT)

        if use_llm:
            reasoning, used_llm = generate_reasoning_llm(
                candidate=candidate,
                rank=rank_num,
                score=score,
                features_rec=features_rec,
            )
        else:
            # Rule-based only mode
            if candidate:
                reasoning = safe_fallback(candidate, rank_num)
                used_llm = False
            else:
                reasoning = _rule_based_reasoning(features_rec, rank_num)
                used_llm = False

        if used_llm:
            llm_count += 1
            print("  [llm]")
        else:
            fallback_count += 1
            print("  [fallback]")

        total_chars += len(reasoning)
        final_rows.append({
            "candidate_id": cid,
            "rank":         rank_num,
            "score":        score,
            "reasoning":    reasoning,
        })

    # Write final submission CSV (reasoning is CSV-safe via csv.writer quoting)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in final_rows:
            writer.writerow([
                r["candidate_id"],
                r["rank"],
                r["score"],    # preserve exact score from raw CSV — do not re-round
                r["reasoning"],
            ])

    avg_len = total_chars // max(len(final_rows), 1)

    # Completion summary — exact format specified
    print(f"\n[reason.py] Complete. Generated reasoning for {len(final_rows)} candidates.")
    print(f"[reason.py] LLM calls: {llm_count} successful, {fallback_count} fallbacks used")
    print(f"[reason.py] Estimated avg reasoning length: {avg_len} characters")
    print(f"[reason.py] Submission saved → {output_path}")
    print(f"[reason.py] Next: python validate_submission.py {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reasoning for top 100 ranked candidates (Phase C)."
    )
    parser.add_argument("--raw",        default=DEFAULT_RAW,      help="Raw ranking CSV from rank.py")
    parser.add_argument("--candidates", default=DEFAULT_CANDS,    help="candidates.jsonl (full profiles)")
    parser.add_argument("--features",   default=DEFAULT_FEATURES, help="Pre-computed features.pkl (snapshot)")
    parser.add_argument("--out",        default=DEFAULT_OUT,      help="Output submission CSV path")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM calls — use rule-based reasoning only (instant, no quota needed)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.raw):
        print(f"[reason.py ERROR] raw ranking CSV not found: {args.raw}")
        sys.exit(1)

    run_reasoning(
        raw_csv=args.raw,
        candidates_path=args.candidates,
        features_path=args.features,
        output_path=args.out,
        use_llm=not args.no_llm,
    )


if __name__ == "__main__":
    main()
