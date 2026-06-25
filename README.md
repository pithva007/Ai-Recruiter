# AI Recruiter — Redrob Hackathon Submission

Ranks the top 100 candidates from a 100,000-candidate pool for the role of **Senior AI Engineer — Founding Team at Redrob AI (Series A)**.

---

## Quick Start (Reproduce Submission CSV)

```bash
git clone https://github.com/pithva007/Ai-Recruiter.git
cd Ai-Recruiter
pip install -r requirements.txt

# Fast path: rank from pre-computed features (< 1 second)
python src/rank.py \
  --candidates ./data/raw/candidates.jsonl \
  --features ./data/processed/features.pkl \
  --out ./outputs/submission.csv

python validate_submission.py ./outputs/submission.csv
```

---

## Architecture

```
candidates.jsonl (100K) + job_description.docx
        │
        ▼ Stage 1 — JD Analysis  (LLM once, offline)
        │   → data/processed/jd_features.json
        │
        ▼ Stage 2 — Precompute  (CPU, no LLM, ~14s)
        │   → data/processed/features.pkl  (17 MB, all 100K candidates scored)
        │
        ▼ rank.py — Sort top 100  (<0.3s, no network, no LLM)  ← SUBMISSION CORE
        │   → outputs/ranked_top100_raw.csv
        │
        ▼ reason.py — LLM reasoning for top 100  (offline, ~12 min)
        │   → outputs/submission.csv  ← SUBMIT THIS
        │
        ▼ Stages 3–7 — Optional LLM enrichment pipeline  (offline)
        │   Evidence → Graph → Retrieval → Scoring → Ranking
        │   → outputs/ranked_candidates.csv  (30 deeply-scored candidates)
        │
        ▼ Stage 8 — Streamlit Dashboard
        │   streamlit run app.py
        │
        ▼ Stage 9 — PDF Report
            → outputs/shortlist_report.pdf
```

---

## Two-Step Submission Process

```
rank.py   →  ranked CSV (no reasoning, no network, < 5 min on CPU)
reason.py →  adds LLM-generated reasoning to top-100 (requires GEMINI_API_KEY, run offline)
```

The final `submission.csv` was produced by running `reason.py` after `rank.py`.
`rank.py` alone satisfies all compute constraints. `reason.py` is offline pre-processing.

---

## Pre-computation (only needed if features.pkl is missing)

```bash
cp .env.example .env   # add your GEMINI_API_KEY
python src/stage1_jd_analysis.py   # one-time JD analysis (needs Gemini, ~30s)
python src/precompute.py           # scores all 100K candidates (~14 seconds, no network)
```

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env: GEMINI_API_KEY=your_key_here
```

**Environment variables (`.env`):**
```
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-flash-lite-latest
LOG_LEVEL=INFO
```

---

## Compute Constraints Compliance

| Constraint | Limit | Our System |
|---|---|---|
| Runtime | ≤ 5 min | **< 1 second** (from features.pkl) |
| Memory | ≤ 16 GB | **~2 GB** (features.pkl is 17 MB) |
| CPU only | Yes | **No GPU used** at any stage |
| No network | Yes | **rank.py has zero network calls** |
| Streaming fallback | Required | streams candidates.jsonl if pkl missing (~15s) |

---

## Key Design Decisions

1. **Career history dominates**: production retrieval/ranking at product companies is weighted 0.30 in the final formula — the single largest component
2. **Services penalty**: pure TCS/Infosys/Wipro careers get a 0.05× multiplier (JD explicitly rejects pure-services profiles); mixed careers get 0.40–0.75×
3. **Skills trust scoring**: zero-duration skills get near-zero weight (0.05–0.10×) regardless of declared proficiency — catches keyword stuffers
4. **Behavioral multiplier**: 8 availability signals multiply the base score multiplicatively — an inactive candidate (last active > 180 days) loses 60% of their score
5. **Honeypot detection**: 472 impossible-profile candidates scored near-zero (impossible timelines, mass zero-duration expert skills, copy-paste descriptions)

**Scoring formula:**
```
final_score = (career×0.30 + skill×0.20 + retrieval×0.30 + fit×0.20)
              × behavioral_multiplier
              × services_penalty
```

---

## File Structure

```
ai-recruiter/
├── run_pipeline.py              ← Single command to run all 9 stages
├── app.py                       ← Streamlit dashboard entry point
├── validate_submission.py       ← Official submission validator
├── submission_metadata.yaml     ← Hackathon submission metadata
├── AGENT.md                     ← Scoring contracts, JD requirements, honeypot spec
├── CLAUDE.md                    ← LLM prompts, anti-hallucination rules
├── SKILLS.md                    ← Technical reference, formulas, skill lists
│
├── src/
│   ├── stage1_jd_analysis.py    ← Phase A: JD → jd_features.json  (LLM, once)
│   ├── precompute.py            ← Phase A: 100K candidates → features.pkl
│   ├── rank.py                  ← Phase B: features.pkl → ranked CSV  (<1s, no network)
│   ├── reason.py                ← Phase C: top-100 → reasoning  (LLM, offline)
│   ├── stage3_evidence_extraction.py
│   ├── stage4_graph_builder.py
│   ├── stage5_hybrid_retrieval.py
│   ├── stage6_scoring_engine.py
│   ├── stage7_ranking.py
│   └── stage8_dashboard.py
│
├── utils/
│   ├── llm_client.py            ← Gemini API wrapper (retry, rate-limit aware)
│   ├── feature_engineering.py  ← All scoring functions (deterministic, no network)
│   ├── embedding_client.py      ← sentence-transformers (local, offline)
│   ├── json_validator.py        ← Pydantic models for all pipeline I/O
│   └── report_generator.py     ← reportlab PDF generator
│
├── data/
│   ├── raw/                     ← Input files + symlinks to challenge bundle
│   │   ├── candidates.jsonl     ← Symlink → challenge bundle (465 MB, not committed)
│   │   └── job_description.docx
│   └── processed/               ← Generated: features.pkl, evidence/, scores/, etc.
│       ├── features.pkl         ← 17 MB pre-computed scores (run precompute.py once)
│       └── jd_features.json
│
└── outputs/                     ← Final submission artifacts
    ├── submission.csv           ← SUBMIT THIS (100 rows, validated)
    ├── ranked_top100_raw.csv    ← rank.py output (no reasoning)
    ├── ranked_candidates.csv    ← 30 LLM-scored candidates
    └── shortlist_report.pdf     ← 17-page PDF shortlist report
```

---

## Validate

```bash
python validate_submission.py outputs/submission.csv
# → Submission is valid.
```

---

## Dashboard

```bash
streamlit run app.py
```

Features: 3-panel layout, hiring decision simulator with weight sliders, radar charts,
dark horse spotlight, bias audit.

---

## Sandbox

[Link to HuggingFace Spaces demo]

Accepts up to 100 candidates as input, produces ranked CSV. Deploy with:
```bash
# Deploy app.py to HuggingFace Spaces (Streamlit SDK)
# Set GEMINI_API_KEY as a Space secret
streamlit run app.py
```
