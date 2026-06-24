# AI Recruiter — Intelligent Candidate Discovery & Ranking

Redrob Hackathon: Intelligent Candidate Discovery & Ranking Challenge.

Ranks the top 100 candidates from a 100,000-candidate pool for the role of **Senior AI Engineer — Founding Team at Redrob AI**.

---

## Setup

```bash
git clone https://github.com/pithva007/Ai-Recruiter.git
cd Ai-Recruiter
pip install -r requirements.txt
cp .env.example .env
# Add your GEMINI_API_KEY to .env
```

---

## Reproduce the Submission (single command)

```bash
python src/rank.py \
  --candidates India_runs_data_and_ai_challenge/candidates.jsonl \
  --features data/processed/features.pkl \
  --out outputs/submission_raw.csv

python src/reason.py \
  --raw outputs/submission_raw.csv \
  --out outputs/submission.csv

python validate_submission.py outputs/submission.csv
```

**Runtime:** rank.py completes in ~0.3 seconds on CPU. No GPU, no network calls.

---

## Full Pipeline (Stages 1–8)

| Stage | Script | What it does | LLM? | Time |
|---|---|---|---|---|
| 1 | `src/stage1_jd_analysis.py` | Parse JD → structured requirements | Yes (once) | ~30s |
| 2 | `src/precompute.py` | Score all 100K candidates | No | ~14s |
| 3 | `src/rank.py` | Sort → top 100 CSV | No | **0.3s** |
| 4 | `src/reason.py` | LLM reasoning for top 100 | Yes | ~12min |
| 5 | `src/stage3_evidence_extraction.py` | Extract evidence from top 100 | Yes | ~15min |
| 6 | `src/stage4_graph_builder.py` | Build GraphRAG knowledge graph | No | ~2s |
| 7 | `src/stage5_hybrid_retrieval.py` | FAISS + graph hybrid retrieval | No | ~10s |
| 8 | `src/stage6_scoring_engine.py` | LLM deep scoring (fit/impact/potential/risk) | Yes | ~10min |
| 9 | `src/stage7_ranking.py` | Explainable ranking + dark horse detection | Yes | ~10min |
| 10 | `app.py` | Streamlit dashboard | No | instant |

---

## Architecture

```
candidates.jsonl (100K)
        │
        ▼
precompute.py ──── deterministic scoring (title + skills + behavioral + fit)
        │
        ▼
rank.py ──────── sort → top 100 (< 5 min, CPU, no network) ← SUBMISSION CORE
        │
        ▼
reason.py ──── LLM reasoning per candidate → submission.csv ← SUBMIT THIS
```

**Scoring formula:**
```
final_score = career_score×0.45 + skill_score×0.25 + behavioral_score×0.20 + fit_score×0.10
```

**Key design decisions:**
- Career history title + description dominate over skills keywords (anti-stuffer)
- 472 honeypot candidates detected and scored 0.0
- Services company penalty (TCS/Infosys/Wipro/etc.) applied to pure-services careers
- Behavioral signals from redrob_signals used as availability multiplier
- skill.duration_months anti-stuffer: expert skill with 0 months = flagged and down-weighted

---

## Dashboard

```bash
streamlit run app.py
```

Features: 3-panel layout, hiring decision simulator with weight sliders, radar charts, dark horse spotlight, bias audit.

---

## Validate

```bash
python validate_submission.py outputs/submission.csv
# → Submission is valid.
```

---

## Pre-computed Artifacts

`data/processed/features.pkl` must exist to run `rank.py`. Generate it once:

```bash
python src/stage1_jd_analysis.py   # requires GEMINI_API_KEY, runs once
python src/precompute.py           # no network, ~14s
```
