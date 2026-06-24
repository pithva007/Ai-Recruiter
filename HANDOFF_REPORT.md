# AI Recruiter — Handoff Report
# Redrob Intelligent Candidate Discovery & Ranking Challenge

**Status: All 9 stages complete. Submission ready.**

---

## What Was Built

A production-grade AI Recruiter Ranking System that ranks 100,000 candidates for the role of **Senior AI Engineer — Founding Team at Redrob AI (Series A)**.

The system uses a deterministic rule-based ranker (no LLM at ranking time) combined with offline LLM enrichment for evidence extraction, deep scoring, and reasoning generation.

---

## Repository

**GitHub:** https://github.com/pithva007/Ai-Recruiter  
**Branch:** `main`  
**Latest commit:** Stage 9 — Pipeline runner + PDF report generator

---

## Completed Stages

| Stage | File | What it does | Status |
|---|---|---|---|
| **1** | `src/stage1_jd_analysis.py` | Parse JD → `jd_features.json` (LLM, once) | DONE |
| **2** | `src/precompute.py` | Score all 100K candidates → `features.pkl` | DONE |
| **3** | `src/stage3_evidence_extraction.py` | Extract evidence from top-100 (LLM) | DONE |
| **4** | `src/stage4_graph_builder.py` | Build GraphRAG knowledge graph (networkx) | DONE |
| **5** | `src/stage5_hybrid_retrieval.py` | FAISS + Graph hybrid retrieval | DONE |
| **6** | `src/stage6_scoring_engine.py` | LLM deep scoring (fit/impact/potential/risk) | DONE |
| **7** | `src/stage7_ranking.py` | Explainable ranking + dark horse detection | DONE |
| **8** | `src/stage8_dashboard.py` + `app.py` | Streamlit Recruiter Copilot Dashboard | DONE |
| **9** | `run_pipeline.py` + `utils/report_generator.py` | Pipeline runner + PDF report | DONE |

---

## Key Output Files

| File | Description |
|---|---|
| `outputs/submission.csv` | **THE SUBMISSION FILE** — 100 rows, validated |
| `outputs/ranked_candidates.csv` | 30 LLM-scored candidates, 19-column schema |
| `outputs/shortlist_report.pdf` | 17-page PDF shortlist report |
| `outputs/ranking_summary.json` | Stats: top-10 IDs, dark horse count, score range |
| `data/processed/features.pkl` | Pre-computed scores for all 100K candidates |
| `data/processed/jd_features.json` | Structured JD requirements from Stage 1 |
| `data/processed/knowledge_graph.gexf` | 868-node GraphRAG graph |
| `data/processed/retrieval_results.json` | Hybrid retrieval results (100 candidates) |

---

## Submission Stats

| Metric | Value |
|---|---|
| Submission rows | 100 |
| Score range | 0.8528 – 0.9434 |
| Scores non-increasing | Yes |
| Unique reasoning | 100/100 (LLM-generated) |
| Validation | `Submission is valid.` |
| Honeypots detected | 472 scored 0.0 |
| Dark horses identified | 12 candidates |
| Top candidate | Ira Dalal (composite 92.25) |

---

## Architecture Summary

```
candidates.jsonl (100K) + job_description.docx
        │
        ▼ Stage 1 — JD Analysis (LLM once)
        │   → data/processed/jd_features.json
        │
        ▼ Stage 2 — Precompute (CPU, no LLM, ~14s)
        │   → data/processed/features.pkl
        │
        ▼ rank.py — Sort top 100 (<0.3s, no network) ← CORE RANKING
        │   → outputs/ranked_top100_raw.csv
        │
        ▼ reason.py — LLM reasoning for top 100
        │   → outputs/submission.csv  ← SUBMIT THIS
        │
        ▼ Stages 3-7 — LLM enrichment pipeline (offline)
        │   Evidence → Graph → Retrieval → Scoring → Ranking
        │   → outputs/ranked_candidates.csv (30 deeply-scored candidates)
        │
        ▼ Stage 8 — Streamlit Dashboard
        │   streamlit run app.py
        │
        ▼ Stage 9 — PDF Report
            → outputs/shortlist_report.pdf
```

**Final score formula (rank.py):**
```
final_score = career_score×0.45 + skill_score×0.25 + behavioral_score×0.20 + fit_score×0.10
```

**LLM composite formula (Stage 6/7):**
```
composite = (fit×0.35) + (impact×0.30) + (potential×0.20) + ((100-risk)×0.15)
```

---

## How to Reproduce

### Submission CSV only (fast — no LLM needed after precompute):
```bash
python src/rank.py \
  --candidates data/raw/candidates.jsonl \
  --features data/processed/features.pkl \
  --out outputs/submission_raw.csv

python src/reason.py --raw outputs/submission_raw.csv --out outputs/submission.csv
python validate_submission.py outputs/submission.csv
```

### Full pipeline (all 9 stages):
```bash
python run_pipeline.py
```

### Dashboard:
```bash
streamlit run app.py
```

---

## Setup

```bash
git clone https://github.com/pithva007/Ai-Recruiter.git
cd Ai-Recruiter
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

**Environment variables (`.env`):**
```
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-flash-lite-latest
LOG_LEVEL=INFO
```

---

## Project Structure

```
ai-recruiter/
├── run_pipeline.py              ← Single command to run all 9 stages
├── app.py                       ← Streamlit dashboard entry point
├── validate_submission.py       ← Official submission validator
├── submission_metadata.yaml     ← Hackathon submission metadata
├── AGENT.md                     ← Scoring contracts, JD requirements
├── CLAUDE.md                    ← LLM prompts, anti-hallucination rules
├── SKILLS.md                    ← Technical reference, formulas
│
├── src/
│   ├── stage1_jd_analysis.py
│   ├── precompute.py
│   ├── rank.py
│   ├── reason.py
│   ├── stage3_evidence_extraction.py
│   ├── stage4_graph_builder.py
│   ├── stage5_hybrid_retrieval.py
│   ├── stage6_scoring_engine.py
│   ├── stage7_ranking.py
│   └── stage8_dashboard.py
│
├── utils/
│   ├── llm_client.py            ← Gemini API wrapper (retry, rate-limit aware)
│   ├── feature_engineering.py  ← All scoring functions (deterministic)
│   ├── embedding_client.py      ← sentence-transformers (local, offline)
│   ├── json_validator.py        ← Pydantic models for all pipeline I/O
│   └── report_generator.py     ← reportlab PDF generator
│
├── data/
│   ├── raw/                     ← Input files + symlinks to challenge bundle
│   └── processed/               ← Generated: features.pkl, evidence/, scores/, etc.
│
├── outputs/                     ← Final submission artifacts
│   ├── submission.csv           ← SUBMIT THIS
│   ├── ranked_candidates.csv
│   ├── shortlist_report.pdf
│   └── ranking_summary.json
│
├── docs/                        ← Challenge analysis docs
└── India_runs_data_and_ai_challenge/   ← Source of truth (read-only)
```

---

## Key Design Decisions

1. **Career history dominates skills** — Title + job description ML keywords outweigh skills list alone. Prevents keyword stuffers from ranking high.

2. **Anti-stuffer detection** — `skills[].duration_months == 0` with `proficiency = expert` is flagged and down-weighted by 70%. Catches candidates who list skills they never used.

3. **Services company penalty** — Careers 80%+ at TCS/Infosys/Wipro/etc. get `career_score × 0.4`. JD explicitly rejects pure-services profiles.

4. **Honeypot detection** — 472 candidates with impossible profiles (experience timeline, mass zero-duration expert skills, copy-paste descriptions) scored 0.0.

5. **Behavioral multiplier** — `redrob_signals` availability signals (last active, open to work, response rate, notice period) act as a reachability multiplier. A perfect-on-paper candidate inactive for 6 months ranks below an active one.

6. **No LLM at ranking time** — `rank.py` runs in 0.3s on CPU with zero network calls. LLM is used only offline for JD analysis (once) and reasoning (post-ranking, top 100 only).

7. **Dark horse discovery** — Candidates with `hybrid_rank > 15` but `impact/potential >= 75` and `fit >= 50` are flagged as hidden gems a traditional ATS would miss. 12 dark horses identified.

---

## What the Next Agent Should Know

### Current state:
- `outputs/submission.csv` is ready to submit. Run `python validate_submission.py outputs/submission.csv` to confirm.
- All pre-computed artifacts exist: `features.pkl`, `knowledge_graph.gexf`, `retrieval_results.json`, 100 evidence files, 30 score files.
- The Streamlit dashboard works: `streamlit run app.py`

### If you need to re-run anything:
- **Re-rank only** (fast, no LLM): `python src/rank.py` then `python src/reason.py --no-llm`
- **Upgrade reasoning with LLM**: `python src/reason.py` (needs GEMINI_API_KEY, ~12 min)
- **Full pipeline**: `python run_pipeline.py`

### Quota note:
The Gemini free tier allows ~20 requests/day per model. If you hit 429/503:
1. Check `.env` for `GEMINI_MODEL=gemini-flash-lite-latest`
2. Try `gemini-flash-latest` or wait for daily reset
3. Rule-based fallback (`--no-llm`) always works without quota

### Files NOT committed (generated, too large or gitignored):
- `data/processed/features.pkl` (17 MB) — run `python src/precompute.py` to regenerate
- `data/processed/evidence/*.json` — run `python src/stage3_evidence_extraction.py`
- `data/raw/candidates.jsonl` — symlink to challenge bundle, not committed

### Submission checklist:
- [ ] Fill in `submission_metadata.yaml` with real team name, email, phone
- [ ] Add sandbox link (deploy to HuggingFace Spaces: `streamlit run app.py`)
- [ ] Submit `outputs/submission.csv` via hackathon portal
- [ ] Upload metadata from `submission_metadata.yaml`

---

## Challenge Evaluation

**Scoring:** `Final composite = 0.50 × NDCG@10 + 0.30 × NDCG@50 + 0.15 × MAP + 0.05 × P@10`

**Getting top 10 right is what matters most.** The current top 10 are all genuine ML/AI engineers with production experience in retrieval/ranking systems — no keyword stuffers, no honeypots.

**Top 10 candidates (Stage 7 composite scores):**

| Rank | Name | Composite |
|---|---|---|
| 1 | Ira Dalal | 92.25 |
| 2 | Mira Ghosh | 91.05 |
| 3 | Ishaan Arora | 91.05 |
| 4 | Vivaan Shah | 89.60 |
| 5 | Aarav Agarwal | 89.55 |
| 6 | Naina Tiwari | 87.10 |
| 7 | Sunil Mishra | 86.60 |
| 8 | Suresh Dutta | 86.60 |
| 9 | Meera Chowdary | 86.60 |
| 10 | Anika Rao | 86.35 |
