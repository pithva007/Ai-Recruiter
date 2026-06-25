# src/rank.py
# Phase B — Ranking  (THE SUBMISSION-CRITICAL SCRIPT)
#
# Input:  data/processed/features.pkl   (preferred — from precompute.py)
#      OR data/raw/candidates.jsonl     (fallback — stream-scores on the fly)
# Output: outputs/ranked_top100_raw.csv (top 100, NO reasoning column)
#         reasoning is added separately by reason.py
#
# HARD CONSTRAINTS (Docker sandbox):
#   - ZERO network calls (no LLM, no API, no HTTP)
#   - ZERO imports that trigger network activity
#   - Must complete in < 5 minutes wall-clock on CPU with 16 GB RAM
#   - No GPU
#
# Reproduce command (single command, works in Docker):
#   python src/rank.py \
#       --candidates ./data/raw/candidates.jsonl \
#       --features   ./data/processed/features.pkl \
#       --out        ./outputs/submission.csv

import argparse
import csv
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

# ── stdlib only ─────────────────────────────────────────────────────────────
# NO network-capable imports below this line.
# NO: requests, httpx, aiohttp, openai, anthropic, google.generativeai, genai
# NO: sentence_transformers, faiss, torch, tensorflow (would trigger downloads)
# ────────────────────────────────────────────────────────────────────────────

ROOT             = Path(__file__).parent.parent
DEFAULT_FEATURES = ROOT / "data" / "processed" / "features.pkl"
DEFAULT_CANDS    = ROOT / "data" / "raw" / "candidates.jsonl"
DEFAULT_OUTPUT   = ROOT / "outputs" / "ranked_top100_raw.csv"

CAND_ID_PATTERN  = re.compile(r"^CAND_[0-9]{7}$")
TOP_N            = 100


# ---------------------------------------------------------------------------
# Fast path: load pre-computed features.pkl
# ---------------------------------------------------------------------------
def rank_from_features(features_path: Path) -> list:
    """
    Load pre-computed score dict from features.pkl.
    Filter to valid IDs, sort, and return top 100 candidates.
    """
    with open(features_path, "rb") as f:
        features: dict = pickle.load(f)
    print(f"[rank.py] Loaded {len(features):,} pre-computed candidate records.")

    # Filter to valid IDs
    valid = [
        rec for cid, rec in features.items()
        if CAND_ID_PATTERN.match(str(cid))
    ]
    
    # Sort: score descending, then candidate_id ascending for ties
    ranked = sorted(valid, key=lambda r: (-r["score"], r["candidate_id"]))
    return ranked[:TOP_N]


# ---------------------------------------------------------------------------
# Slow path: stream candidates.jsonl and score each line
# Used when features.pkl is absent (fresh Docker container without precompute).
# ---------------------------------------------------------------------------
def rank_streamed_top100(candidates_path: Path) -> list:
    """
    Stream candidates.jsonl line by line, score each candidate,
    and keep only the top 100 candidates in memory at any time.
    Do NOT accumulate all candidates (or scored features) in RAM.
    """
    print(f"[rank.py] Streaming {candidates_path} (this takes ~15s for 100K candidates) ...")

    # Import scoring only when needed — still stdlib + pure Python, no network
    sys.path.insert(0, str(ROOT))
    from utils.feature_engineering import compute_final_score, is_honeypot

    top100 = []
    count       = 0
    honeypots   = 0
    t_stream    = time.time()
    PRINT_EVERY = 10_000

    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue

            cid = candidate.get("candidate_id", "")
            if not CAND_ID_PATTERN.match(cid):
                continue

            hp    = is_honeypot(candidate)
            score = 0.0 if hp else compute_final_score(candidate)
            if hp:
                honeypots += 1

            profile = candidate.get("profile", {})
            rec = {
                "candidate_id":    cid,
                "score":           score,
                "honeypot":        hp,
                "current_title":   profile.get("current_title", ""),
                "years_experience":profile.get("years_of_experience", 0),
                "location":        profile.get("location", ""),
                "country":         profile.get("country", ""),
                "recruiter_rr":    candidate.get("redrob_signals", {}).get("recruiter_response_rate", 0),
                "notice_days":     candidate.get("redrob_signals", {}).get("notice_period_days", 90),
                "open_to_work":    candidate.get("redrob_signals", {}).get("open_to_work_flag", False),
                "top_skills":      [
                    s["name"] for s in candidate.get("skills", [])
                    if s.get("duration_months", 0) > 0
                ][:5],
            }

            # Maintain exactly top 100 sorted elements in RAM
            key = (-score, cid)
            if len(top100) < TOP_N:
                top100.append(rec)
                top100.sort(key=lambda r: (-r["score"], r["candidate_id"]))
            else:
                worst_rec = top100[-1]
                worst_key = (-worst_rec["score"], worst_rec["candidate_id"])
                if key < worst_key:
                    top100[-1] = rec
                    top100.sort(key=lambda r: (-r["score"], r["candidate_id"]))

            count += 1

            if count % PRINT_EVERY == 0:
                elapsed_so_far = time.time() - t_stream
                rate = count / elapsed_so_far
                eta  = (100_000 - count) / rate if rate > 0 else 0
                print(
                    f"  [{count:,}/100,000] "
                    f"elapsed={elapsed_so_far:.1f}s  "
                    f"rate={rate:.0f}/s  "
                    f"eta={eta:.0f}s  "
                    f"honeypots={honeypots}"
                )

    stream_elapsed = time.time() - t_stream
    print(
        f"[rank.py] Scored {count:,} candidates in {stream_elapsed:.1f}s "
        f"({count / stream_elapsed:.0f} candidates/sec)  "
        f"honeypots={honeypots}"
    )
    return top100


# ---------------------------------------------------------------------------
# Save top 100 candidates to CSV
# ---------------------------------------------------------------------------
def save_csv(candidates: list, output_path: Path) -> None:
    """
    Write ranked candidates list to a CSV with empty reasoning column.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank_num, rec in enumerate(candidates, start=1):
            writer.writerow([
                rec["candidate_id"],
                rank_num,
                round(rec["score"], 6),   # 6dp avoids tie-breaking rounding issues
                "",                        # placeholder — filled by reason.py
            ])

    print(
        f"[rank.py] Top {len(candidates)} sorted in "
        f"{candidates[0]['score']:.4f} – {candidates[-1]['score']:.4f} score range."
    )
    print(f"[rank.py] Honeypots in top {len(candidates)}: {sum(1 for r in candidates if r.get('honeypot'))}")
    print(f"[rank.py] Raw ranking saved → {output_path}")


# ---------------------------------------------------------------------------
# Main — argument parsing + timing instrumentation (CHECK 2)
# ---------------------------------------------------------------------------
def main() -> None:
    # ── CHECK 2: Timing starts at the very top of main() ──
    start = time.time()

    parser = argparse.ArgumentParser(
        description=(
            "Rank top 100 candidates. "
            "Uses features.pkl (fast) if present; falls back to streaming candidates.jsonl."
        )
    )
    # ── CHECK 4: Argparse support for required reproduce command flags ──
    parser.add_argument(
        "--candidates",
        default=str(DEFAULT_CANDS),
        help="Path to candidates.jsonl. Default: data/raw/candidates.jsonl",
    )
    parser.add_argument(
        "--features",
        default=str(DEFAULT_FEATURES),
        help="Path to pre-computed features.pkl. Default: data/processed/features.pkl",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUTPUT),
        help="Output CSV path. Default: outputs/ranked_top100_raw.csv",
    )
    args = parser.parse_args()

    # ── CHECK 3: Announce path choice exactly as required ──
    features_path = Path(args.features) if hasattr(args, 'features') else Path("data/processed/features.pkl")
    candidates_path = Path(args.candidates) if hasattr(args, 'candidates') else Path("data/raw/candidates.jsonl")

    if features_path.exists():
        print(f"[rank.py] Loading pre-computed features from {features_path}")
        candidates = rank_from_features(features_path)
    else:
        print(f"[rank.py] features.pkl not found, computing scores from {candidates_path}")
        candidates = rank_streamed_top100(candidates_path)

    # ── CHECK 2: Timing instrumentation and 4-minute warning before saving CSV ──
    elapsed = time.time() - start
    print(f"[rank.py] Completed in {elapsed:.2f}s for {len(candidates)} candidates")
    if elapsed > 240:  # warn at 4 minutes, budget is 5
        print(f"[WARNING] Approaching 5-minute compute limit: {elapsed:.1f}s elapsed")

    # Save outputs
    output_path = Path(args.out)
    save_csv(candidates, output_path)


if __name__ == "__main__":
    main()
