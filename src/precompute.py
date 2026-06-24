# src/precompute.py
# Phase A — Offline Feature Pre-computation
#
# Input:  data/raw/candidates.jsonl       (100K candidates)
#         data/processed/jd_features.json (from stage1_jd_analysis.py)
# Output: data/processed/features.pkl    (scored feature vectors for all 100K)
#
# Runs ONCE offline. No LLM calls. No network. No time limit.
# Computes deterministic feature vectors for every candidate.
# The output features.pkl is loaded by rank.py during the <5-min ranking step.
#
# Usage:
#   python src/precompute.py
#   python src/precompute.py --candidates data/raw/candidates.jsonl \
#                            --jd data/processed/jd_features.json \
#                            --out data/processed/features.pkl

import argparse
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.feature_engineering import compute_final_score, is_honeypot

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.join(os.path.dirname(__file__), "..")
DEFAULT_CANDIDATES = os.path.join(ROOT, "data", "raw",       "candidates.jsonl")
DEFAULT_JD         = os.path.join(ROOT, "data", "processed", "jd_features.json")
DEFAULT_OUTPUT     = os.path.join(ROOT, "data", "processed", "features.pkl")

BATCH_PRINT = 5000  # print progress every N candidates


def load_candidates(path: str):
    """Stream candidates from .jsonl one line at a time."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def precompute(candidates_path: str, jd_path: str, output_path: str) -> None:
    # Load JD features
    if not os.path.exists(jd_path):
        print(f"[Precompute ERROR] JD features not found: {jd_path}")
        print("  Run: python src/stage1_jd_analysis.py  first.")
        sys.exit(1)

    with open(jd_path, "r", encoding="utf-8") as f:
        jd_features = json.load(f)

    print(f"[Precompute] Loaded JD features: {jd_features.get('role_title', 'unknown')}")
    print(f"[Precompute] Reading candidates from: {candidates_path}")
    print(f"[Precompute] Output: {output_path}\n")

    results = {}           # candidate_id → {score, honeypot, snapshot}
    t_start = time.time()
    count = 0
    honeypot_count = 0

    for candidate in load_candidates(candidates_path):
        cid = candidate.get("candidate_id", f"UNKNOWN_{count}")

        hp = is_honeypot(candidate)
        if hp:
            honeypot_count += 1
            score = 0.0
        else:
            score = compute_final_score(candidate)

        # Store minimal snapshot for reasoning generation later
        profile = candidate.get("profile", {})
        results[cid] = {
            "candidate_id": cid,
            "score": score,
            "honeypot": hp,
            # Snapshot fields used by reason.py (no need to reload full jsonl)
            "current_title":    profile.get("current_title", ""),
            "years_experience": profile.get("years_of_experience", 0),
            "location":         profile.get("location", ""),
            "country":          profile.get("country", ""),
            "recruiter_rr":     candidate.get("redrob_signals", {}).get("recruiter_response_rate", 0),
            "notice_days":      candidate.get("redrob_signals", {}).get("notice_period_days", 90),
            "open_to_work":     candidate.get("redrob_signals", {}).get("open_to_work_flag", False),
            "top_skills":       [
                s["name"] for s in candidate.get("skills", [])
                if s.get("duration_months", 0) > 0
            ][:5],
        }

        count += 1
        if count % BATCH_PRINT == 0:
            elapsed = time.time() - t_start
            rate = count / elapsed
            eta = (100000 - count) / rate if rate > 0 else 0
            print(
                f"  [{count:,}/100,000] "
                f"elapsed={elapsed:.1f}s  rate={rate:.0f}/s  eta={eta:.0f}s  "
                f"honeypots={honeypot_count}"
            )

    elapsed = time.time() - t_start
    print(
        f"\n[Precompute] Scored {count:,} candidates in {elapsed:.1f}s  "
        f"({count/elapsed:.0f} candidates/sec)"
    )
    print(f"[Precompute] Honeypots detected: {honeypot_count}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[Precompute Complete] features.pkl saved ({size_mb:.1f} MB)  →  {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute candidate feature scores.")
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument("--jd",         default=DEFAULT_JD)
    parser.add_argument("--out",        default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    for path, label in [(args.candidates, "candidates"), (args.jd, "jd_features")]:
        if not os.path.exists(path):
            print(f"[Precompute ERROR] {label} file not found: {path}")
            sys.exit(1)

    precompute(args.candidates, args.jd, args.out)


if __name__ == "__main__":
    main()
