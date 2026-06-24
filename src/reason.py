# src/reason.py
# Phase C — Reasoning Generation for Top 100
#
# Input:  outputs/ranked_top100_raw.csv  (from rank.py — no reasoning column)
#         data/processed/features.pkl    (candidate snapshot data)
# Output: outputs/submission.csv         (final submission with reasoning)
#
# LLM is permitted here — only 100 candidates, runs offline after ranking.
# Free-tier RPM limit (10 RPM for gemini-2.5-flash-lite) is respected via
# inter-call throttling. Falls back to rule-based reasoning on LLM failure.
#
# Usage:
#   python src/reason.py                          # LLM mode (throttled)
#   python src/reason.py --no-llm                 # rule-based only, instant
#   python src/reason.py --out outputs/team_id.csv
#
# After this runs, validate with:
#   python validate_submission.py outputs/submission.csv

import argparse
import csv
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.llm_client import call_llm

ROOT             = os.path.join(os.path.dirname(__file__), "..")
DEFAULT_RAW      = os.path.join(ROOT, "outputs", "ranked_top100_raw.csv")
DEFAULT_FEATURES = os.path.join(ROOT, "data",    "processed", "features.pkl")
DEFAULT_OUT      = os.path.join(ROOT, "outputs", "submission.csv")

# Free-tier RPM: 10 req/min for gemini-2.5-flash-lite → wait 7s between calls
LLM_INTER_CALL_WAIT = 7   # seconds

# ---------------------------------------------------------------------------
# Phase C system prompt — from CLAUDE.md
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a technical recruiter writing a candidate assessment summary for a Senior AI Engineer role.

The ranking system has already scored and ranked this candidate. Your job is to write a concise,
honest 1-2 sentence reasoning that explains WHY this candidate ranked where they did.

Rules for your reasoning:
1. Reference only information that actually exists in the candidate data provided.
2. Do NOT invent skills, companies, or achievements not in the data.
3. Do NOT write the same reasoning for every candidate — be specific to this individual.
4. Reference at least one of: their actual current_title, their actual years_of_experience,
   their actual relevant skills (by name), their recruiter_response_rate, or their notice_period_days.
5. If the candidate ranks in the top 20, highlight their strongest signal.
6. If the candidate ranks 50-100, acknowledge the limiting factors.
7. Maximum 30 words. No bullet points. Plain sentence format.

Output a single JSON object:
{
  "candidate_id": "<exact candidate_id from input>",
  "reasoning": "<1-2 sentences, max 30 words>"
}

If you cannot write honest specific reasoning from the provided data, output:
{
  "candidate_id": "<id>",
  "reasoning": "Adjacent background with some relevant skills; included based on behavioral engagement signals."
}

Do not invent or infer beyond what is stated."""


# ---------------------------------------------------------------------------
# Rule-based fallback — no LLM, always works
# ---------------------------------------------------------------------------
def _rule_based_reasoning(rec: dict, rank_num: int) -> str:
    """
    Build reasoning purely from candidate snapshot fields.
    References actual data — never invents anything.
    """
    title    = rec.get("current_title", "Engineer")
    yoe      = rec.get("years_experience", 0)
    rr       = rec.get("recruiter_rr", 0)
    notice   = rec.get("notice_days", 90)
    open_w   = rec.get("open_to_work", False)
    skills   = rec.get("top_skills", [])

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
# LLM reasoning (with rule-based fallback)
# ---------------------------------------------------------------------------
def generate_reasoning(rec: dict, rank_num: int, use_llm: bool = True) -> tuple[str, bool]:
    """
    Returns (reasoning_text, used_llm).
    Falls back to rule-based if use_llm=False or LLM call fails.
    """
    if not use_llm:
        return _rule_based_reasoning(rec, rank_num), False

    user_content = (
        f"Generate reasoning for this candidate at rank {rank_num}.\n\n"
        f"candidate_id: {rec['candidate_id']}\n"
        f"current_title: {rec.get('current_title', 'unknown')}\n"
        f"years_of_experience: {rec.get('years_experience', 0)}\n"
        f"location: {rec.get('location', '')} ({rec.get('country', '')})\n"
        f"open_to_work: {rec.get('open_to_work', False)}\n"
        f"notice_period_days: {rec.get('notice_days', 'unknown')}\n"
        f"recruiter_response_rate: {rec.get('recruiter_rr', 0):.2f}\n"
        f"top_skills: {', '.join(rec.get('top_skills', [])) or 'none listed'}\n"
        f"score: {rec.get('score', 0):.4f}\n"
    )
    try:
        result = call_llm(
            system_prompt=SYSTEM_PROMPT,
            user_content=user_content,
            temperature=0.0,
        )
        text = str(result.get("reasoning", "")).strip()
        if text:
            return text, True
        return _rule_based_reasoning(rec, rank_num), False
    except Exception as e:
        print(f"\n  [Reason FALLBACK] {type(e).__name__} — using rule-based for {rec['candidate_id']}")
        return _rule_based_reasoning(rec, rank_num), False


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def run_reasoning(
    raw_csv: str,
    features_path: str,
    output_path: str,
    use_llm: bool = True,
) -> None:
    # Load raw ranking
    with open(raw_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"[Reason] Loaded {len(rows)} ranked candidates from {raw_csv}")

    # Load feature snapshots
    with open(features_path, "rb") as f:
        features: dict = pickle.load(f)

    mode_label = "LLM + rule-based fallback" if use_llm else "rule-based only (fast)"
    print(f"[Reason] Mode: {mode_label}")
    if use_llm:
        print(f"[Reason] Inter-call wait: {LLM_INTER_CALL_WAIT}s (free-tier RPM compliance)")
    print(f"[Reason] Generating reasoning for {len(rows)} candidates ...\n")

    final_rows = []
    llm_count  = 0
    rule_count = 0

    for i, row in enumerate(rows, start=1):
        cid      = row["candidate_id"]
        rank_num = int(row["rank"])
        score    = float(row["score"])
        rec      = features.get(cid, {"candidate_id": cid})

        title_preview = rec.get("current_title", "?")[:25]
        print(
            f"  [{i:3d}/100] {cid}  score={score:.4f}  "
            f"{title_preview:<25}",
            end="",
            flush=True,
        )

        # Throttle before every LLM call after the first
        if use_llm and i > 1:
            time.sleep(LLM_INTER_CALL_WAIT)

        reasoning, used_llm = generate_reasoning(rec, rank_num, use_llm=use_llm)

        if used_llm:
            llm_count += 1
            print("  [llm]")
        else:
            rule_count += 1
            print("  [rule]")

        final_rows.append({
            "candidate_id": cid,
            "rank":         rank_num,
            "score":        score,
            "reasoning":    reasoning,
        })

    # Write final submission CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in final_rows:
            writer.writerow([
                r["candidate_id"],
                r["rank"],
                r["score"],    # preserve exact score from raw CSV — do not re-round
                r["reasoning"],
            ])

    print(f"\n[Reason] LLM-generated: {llm_count}  |  Rule-based: {rule_count}")
    print(f"[Reason Complete] Final submission saved → {output_path}")
    print(f"[Reason] Next step:  python validate_submission.py {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reasoning for top 100 ranked candidates."
    )
    parser.add_argument("--raw",      default=DEFAULT_RAW,      help="Raw ranking CSV from rank.py")
    parser.add_argument("--features", default=DEFAULT_FEATURES, help="Pre-computed features.pkl")
    parser.add_argument("--out",      default=DEFAULT_OUT,      help="Output submission CSV path")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM calls — use rule-based reasoning only (instant, no quota needed)",
    )
    args = parser.parse_args()

    for path, label in [(args.raw, "raw_ranking"), (args.features, "features")]:
        if not os.path.exists(path):
            print(f"[Reason ERROR] {label} file not found: {path}")
            sys.exit(1)

    run_reasoning(
        raw_csv=args.raw,
        features_path=args.features,
        output_path=args.out,
        use_llm=not args.no_llm,
    )


if __name__ == "__main__":
    main()
