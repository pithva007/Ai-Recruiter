# src/rank.py
# Phase B — Ranking  (THE SUBMISSION-CRITICAL SCRIPT)
#
# Input:  data/processed/features.pkl   (from precompute.py)
# Output: outputs/ranked_top100_raw.csv (top 100, no reasoning yet)
#
# HARD CONSTRAINTS — enforced here:
#   - Zero network calls
#   - Zero LLM calls
#   - Must complete in < 5 minutes on CPU with 16 GB RAM
#   - candidate_id must match CAND_[0-9]{7}
#
# Usage:
#   python src/rank.py
#   python src/rank.py --features data/processed/features.pkl \
#                      --out outputs/ranked_top100_raw.csv
#
# The single reproduce command is:
#   python src/rank.py --features data/processed/features.pkl --out outputs/{team_id}_raw.csv

import argparse
import csv
import os
import pickle
import re
import sys
import time

ROOT = os.path.join(os.path.dirname(__file__), "..")
DEFAULT_FEATURES = os.path.join(ROOT, "data",    "processed", "features.pkl")
DEFAULT_OUTPUT   = os.path.join(ROOT, "outputs", "ranked_top100_raw.csv")

CAND_ID_PATTERN  = re.compile(r"^CAND_[0-9]{7}$")
TOP_N            = 100


def rank(features_path: str, output_path: str) -> list[dict]:
    t0 = time.time()

    print(f"[Rank] Loading features from: {features_path}")
    with open(features_path, "rb") as f:
        features: dict = pickle.load(f)

    print(f"[Rank] Loaded {len(features):,} candidate records.")

    # Filter to valid candidate IDs only (guards against any data corruption)
    valid = {
        cid: rec for cid, rec in features.items()
        if CAND_ID_PATTERN.match(str(cid))
    }
    invalid_count = len(features) - len(valid)
    if invalid_count:
        print(f"[Rank WARNING] Skipped {invalid_count} records with invalid candidate_id format.")

    # Sort: score descending (full precision), then candidate_id ascending (tie-break per spec)
    ranked = sorted(valid.values(), key=lambda r: (-r["score"], r["candidate_id"]))

    # Take top 100
    top100 = ranked[:TOP_N]

    # Write raw CSV (no reasoning column yet — reason.py adds it)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank_num, rec in enumerate(top100, start=1):
            writer.writerow([
                rec["candidate_id"],
                rank_num,
                round(rec["score"], 6),   # 6dp avoids rounding ties that 4dp creates
                "",   # placeholder — filled by reason.py
            ])

    elapsed = time.time() - t0
    print(f"\n[Rank] Top 100 ranked in {elapsed:.2f}s")
    print(f"[Rank] Score range: {top100[-1]['score']:.4f} – {top100[0]['score']:.4f}")
    print(f"[Rank] Honeypots in top 100: {sum(1 for r in top100 if r.get('honeypot'))}")
    print(f"[Rank Complete] Raw ranking saved → {output_path}")

    return top100


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank top 100 candidates from pre-computed features.")
    parser.add_argument("--features", default=DEFAULT_FEATURES)
    parser.add_argument("--out",      default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not os.path.exists(args.features):
        print(f"[Rank ERROR] features.pkl not found: {args.features}")
        print("  Run: python src/precompute.py  first.")
        sys.exit(1)

    rank(args.features, args.out)


if __name__ == "__main__":
    main()
