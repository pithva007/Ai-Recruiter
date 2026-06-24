# src/stage3_evidence_extraction.py
# Stage 3: Evidence Extraction Layer
#
# Input:
#   outputs/ranked_top100_raw.csv          — which candidates to process
#   data/raw/candidates.jsonl              — full candidate records (raw text)
#   data/processed/features.pkl            — pre-computed profile snapshots
#
# Output per candidate:
#   data/processed/evidence/{candidate_id}_evidence.json
#   Schema: {"evidence": [...EvidenceItem + weight], "entities": [...str]}
#
# Runs offline. LLM permitted (one call per candidate).
# Processes ONLY the top-100 ranked candidates — not all 100K.
# Free-tier RPM respected via inter-call throttling.
#
# Usage:
#   python src/stage3_evidence_extraction.py
#   python src/stage3_evidence_extraction.py --candidates outputs/ranked_top100_raw.csv
#   python src/stage3_evidence_extraction.py --limit 10   # dev: first N candidates only

import argparse
import json
import os
import pickle
import re
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.llm_client import call_llm
from utils.json_validator import EvidenceItem
from pydantic import ValidationError

ROOT             = os.path.join(os.path.dirname(__file__), "..")
DEFAULT_RANKED   = os.path.join(ROOT, "outputs",  "ranked_top100_raw.csv")
DEFAULT_JSONL    = os.path.join(ROOT, "data",      "raw",       "candidates.jsonl")
DEFAULT_FEATURES = os.path.join(ROOT, "data",      "processed", "features.pkl")
DEFAULT_OUT_DIR  = os.path.join(ROOT, "data",      "processed", "evidence")

# Free-tier RPM limit — 10 RPM for gemini-2.5-flash-lite → 7s between calls
LLM_INTER_CALL_WAIT = 7  # seconds

# ---------------------------------------------------------------------------
# Stage 3 system prompt — from CLAUDE.md § Stage 3
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an Evidence Extraction Agent. Given a candidate profile, extract all pieces of evidence and classify them by type. Evidence types:
- technical: tools, languages, frameworks, architectures the candidate demonstrably used
- impact: measurable outcomes the candidate produced (numbers, scale, revenue, users)
- leadership: evidence of leading people, mentoring, managing, or influencing without authority
- learning: evidence of self-directed growth (new certifications, OSS, side projects, hackathons, technology pivots)
- behavioral: writing style, how they describe collaboration, how they handle failure or ambiguity

For EACH piece of evidence output:
{
  "claim": "",
  "evidence_type": "",
  "confidence": "<high | medium | low>",
  "source_text": "",
  "quantified": <true | false>
}

Few-shot examples:

Example 1 (high confidence, quantified):
Source text: "Built a recommendation engine that increased user retention by 23%"
Output: {"claim": "Built recommendation engine improving retention by 23%", "evidence_type": "impact", "confidence": "high", "source_text": "Built a recommendation engine that increased user retention by 23%", "quantified": true}

Example 2 (low confidence, null source):
Source text: "Worked on various ML projects at a leading tech company"
Output: {"claim": "ML project experience", "evidence_type": "technical", "confidence": "low", "source_text": null, "quantified": false}

Output a JSON array of all evidence items. If you cannot find evidence for a type, output an empty array for that type. Do not invent or infer beyond what is stated."""


# ---------------------------------------------------------------------------
# Load ranked candidate IDs from CSV
# ---------------------------------------------------------------------------
def load_ranked_ids(ranked_csv: str) -> list[str]:
    import csv
    with open(ranked_csv, "r", encoding="utf-8") as f:
        return [row["candidate_id"] for row in csv.DictReader(f)]


# ---------------------------------------------------------------------------
# Build index: candidate_id → byte offset in JSONL (for fast random access)
# ---------------------------------------------------------------------------
def build_jsonl_index(jsonl_path: str, target_ids: set) -> dict[str, dict]:
    """
    Stream through candidates.jsonl once and collect full records for target_ids.
    Returns {candidate_id: full_candidate_dict}.
    More memory-efficient than loading all 100K — we only keep the 100 we need.
    """
    records = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                cand = json.loads(line)
                cid = cand.get("candidate_id", "")
                if cid in target_ids:
                    records[cid] = cand
                    if len(records) == len(target_ids):
                        break  # found all — stop early
            except json.JSONDecodeError:
                continue
    return records


# ---------------------------------------------------------------------------
# Build raw profile text from the real candidate schema
# ---------------------------------------------------------------------------
def build_raw_profile_text(candidate: dict) -> str:
    """
    Construct a rich raw text representation from the real candidate schema.
    Combines profile summary, career history descriptions, skills, education.
    This is what gets sent to the LLM as the 'Original Profile Text'.
    """
    parts = []
    profile = candidate.get("profile", {})

    # Headline + summary
    if profile.get("headline"):
        parts.append(f"Headline: {profile['headline']}")
    if profile.get("summary"):
        parts.append(f"Summary: {profile['summary']}")

    # Career history
    for role in candidate.get("career_history", []):
        title   = role.get("title", "")
        company = role.get("company", "")
        dur     = role.get("duration_months", 0)
        desc    = role.get("description", "")
        end     = "present" if role.get("is_current") else role.get("end_date", "")
        line    = f"Role: {title} at {company} ({dur} months, ending {end})"
        if desc:
            line += f"\n  {desc}"
        parts.append(line)

    # Skills
    skill_lines = []
    for s in candidate.get("skills", []):
        name  = s.get("name", "")
        prof  = s.get("proficiency", "")
        dur   = s.get("duration_months", 0)
        end_s = s.get("endorsements", 0)
        skill_lines.append(f"{name} ({prof}, {dur}mo, {end_s} endorsements)")
    if skill_lines:
        parts.append("Skills: " + "; ".join(skill_lines))

    # Certifications
    certs = candidate.get("certifications", [])
    if certs:
        cert_strs = [f"{c.get('name','')} ({c.get('issuer','')}, {c.get('year','')})" for c in certs]
        parts.append("Certifications: " + "; ".join(cert_strs))

    # Education
    for edu in candidate.get("education", []):
        parts.append(
            f"Education: {edu.get('degree','')} in {edu.get('field_of_study','')} "
            f"at {edu.get('institution','')} ({edu.get('end_year','')})"
        )

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Entity extraction from evidence items
# ---------------------------------------------------------------------------
def extract_entities(evidence_items: list[dict], candidate: dict) -> list[str]:
    """
    Build a flat list of unique entity strings for Stage 4 graph building.
    Combines: skill names, tools/frameworks mentioned in claims, company names.
    """
    entities = set()

    # From skills[]
    for s in candidate.get("skills", []):
        name = s.get("name", "").strip()
        if name:
            entities.add(name)

    # From career history companies and titles
    for role in candidate.get("career_history", []):
        company = role.get("company", "").strip()
        title   = role.get("title", "").strip()
        if company:
            entities.add(company)
        if title:
            entities.add(title)

    # From evidence claims — extract capitalized tokens and known tool patterns
    tech_pattern = re.compile(
        r'\b(PyTorch|TensorFlow|FAISS|Pinecone|Weaviate|Qdrant|Milvus|'
        r'OpenSearch|Elasticsearch|Spark|Airflow|Kafka|Docker|Kubernetes|'
        r'AWS|GCP|Azure|Python|SQL|dbt|Snowflake|MLflow|LoRA|QLoRA|PEFT|'
        r'RAG|BERT|GPT|LLM|NLP|XGBoost|scikit-learn|Hugging\s*Face|'
        r'BentoML|Triton|ONNX|FastAPI|Flask|Redis|PostgreSQL)\b',
        re.IGNORECASE,
    )
    for item in evidence_items:
        claim = item.get("claim", "") or ""
        for match in tech_pattern.findall(claim):
            entities.add(match.strip())

    return sorted(entities)


# ---------------------------------------------------------------------------
# Core: process one candidate
# ---------------------------------------------------------------------------
def process_candidate(
    candidate: dict,
    profile_snapshot: dict,
    call_index: int,
) -> dict:
    """
    Run evidence extraction for one candidate.
    Returns the output dict ready to be written to disk.
    """
    cid  = candidate["candidate_id"]
    profile = candidate.get("profile", {})

    # Build the two inputs for the LLM
    profile_summary = {
        "candidate_id":       cid,
        "current_title":      profile.get("current_title", ""),
        "years_experience":   profile.get("years_of_experience", 0),
        "current_company":    profile.get("current_company", ""),
        "current_industry":   profile.get("current_industry", ""),
        "location":           profile.get("location", ""),
        "summary":            profile.get("summary", ""),
        "top_skills":         profile_snapshot.get("top_skills", []),
        "score":              profile_snapshot.get("score", 0),
    }

    raw_profile_text = build_raw_profile_text(candidate)

    user_content = (
        f"Candidate Profile Summary: {json.dumps(profile_summary)}\n\n"
        f"Original Profile Text:\n{raw_profile_text}\n\n"
        "Extract all evidence items from the Original Profile Text. "
        "Use the Candidate Profile Summary only for context."
    )

    # Throttle after first call
    if call_index > 0:
        time.sleep(LLM_INTER_CALL_WAIT)

    raw_result = call_llm(
        system_prompt=SYSTEM_PROMPT,
        user_content=user_content,
        temperature=0.0,
    )

    # LLM returns a list directly for array outputs (correct behavior)
    # or a dict wrapping the list
    if isinstance(raw_result, list):
        raw_items = raw_result
    elif isinstance(raw_result, dict):
        raw_items = (
            raw_result.get("evidence_items")
            or raw_result.get("evidence")
            or raw_result.get("items")
            or raw_result.get("results")
            or []
        )
        # Single item returned as dict
        if not raw_items and "claim" in raw_result:
            raw_items = [raw_result]
    else:
        raw_items = []

    # Validate each item with EvidenceItem pydantic model + add weight
    validated = []
    skip_count = 0
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            skip_count += 1
            continue
        try:
            item = EvidenceItem(**raw_item)
            item_dict = item.model_dump()
            item_dict["weight"] = 1.5 if item.quantified else 1.0
            validated.append(item_dict)
        except (ValidationError, Exception) as e:
            skip_count += 1
            print(f"\n[Stage 3] Skipped invalid evidence item for {cid}: {str(e)[:80]}")

    n_quantified = sum(1 for it in validated if it["quantified"])
    entities     = extract_entities(validated, candidate)

    output = {
        "candidate_id": cid,
        "evidence":     validated,
        "entities":     entities,
    }

    print(
        f"[Stage 3] {cid}: extracted {len(validated)} evidence items "
        f"({n_quantified} quantified, {len(entities)} entities)"
        + (f"  [skipped {skip_count}]" if skip_count else "")
    )

    return output


# ---------------------------------------------------------------------------
# Save to disk
# ---------------------------------------------------------------------------
def save_evidence(output: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    cid      = output["candidate_id"]
    filepath = os.path.join(out_dir, f"{cid}_evidence.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    return filepath


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 3: Extract evidence items for ranked candidates."
    )
    parser.add_argument("--candidates", default=DEFAULT_RANKED,
                        help="Ranked CSV from rank.py (default: outputs/ranked_top100_raw.csv)")
    parser.add_argument("--jsonl",      default=DEFAULT_JSONL,
                        help="Full candidates JSONL (default: data/raw/candidates.jsonl)")
    parser.add_argument("--features",   default=DEFAULT_FEATURES,
                        help="Pre-computed features.pkl (default: data/processed/features.pkl)")
    parser.add_argument("--out",        default=DEFAULT_OUT_DIR,
                        help="Output directory (default: data/processed/evidence/)")
    parser.add_argument("--limit",      type=int, default=None,
                        help="Process only first N candidates (for dev/testing)")
    args = parser.parse_args()

    # Validate inputs
    for path, label in [
        (args.candidates, "ranked CSV"),
        (args.jsonl,      "candidates JSONL"),
        (args.features,   "features.pkl"),
    ]:
        if not os.path.exists(path):
            print(f"[Stage 3 ERROR] {label} not found: {path}")
            sys.exit(1)

    # Load ranked IDs
    ranked_ids = load_ranked_ids(args.candidates)
    if args.limit:
        ranked_ids = ranked_ids[: args.limit]
    print(f"[Stage 3] Processing {len(ranked_ids)} candidates from {args.candidates}")

    # Load feature snapshots (lightweight — no need to reload full JSONL)
    with open(args.features, "rb") as f:
        import pickle
        features: dict = pickle.load(f)

    # Stream JSONL once to collect only the records we need
    print(f"[Stage 3] Indexing candidates from JSONL ...")
    target_set  = set(ranked_ids)
    candidate_records = build_jsonl_index(args.jsonl, target_set)
    found = len(candidate_records)
    print(f"[Stage 3] Found {found}/{len(ranked_ids)} candidates in JSONL\n")

    # Process each candidate
    success = 0
    failed  = 0

    for call_idx, cid in enumerate(ranked_ids):
        candidate = candidate_records.get(cid)
        if candidate is None:
            print(f"[Stage 3 WARNING] {cid} not found in JSONL — skipping")
            failed += 1
            continue

        snapshot = features.get(cid, {})

        # Skip if already processed (resume support)
        out_path = os.path.join(args.out, f"{cid}_evidence.json")
        if os.path.exists(out_path):
            print(f"[Stage 3] {cid}: already exists — skipping")
            success += 1
            # Don't count toward call_idx throttle
            call_idx -= 1
            continue

        try:
            output = process_candidate(candidate, snapshot, call_index=call_idx)
            save_evidence(output, args.out)
            success += 1
        except Exception as e:
            print(f"\n[Stage 3 ERROR] {cid} failed: {type(e).__name__}: {str(e)[:120]}")
            traceback.print_exc()
            # Write error sentinel so we know this candidate was attempted
            save_evidence(
                {"candidate_id": cid, "evidence": [], "entities": [], "error": str(e)},
                args.out,
            )
            failed += 1

    print(
        f"\n[Stage 3 Complete] {success}/{len(ranked_ids)} candidates processed successfully"
        f"{f'  ({failed} failed)' if failed else ''}"
    )
    print(f"[Stage 3] Evidence files saved to: {args.out}")


if __name__ == "__main__":
    main()
