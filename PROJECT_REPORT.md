# AI Recruiter — Comprehensive Project Report
# Redrob Intelligent Candidate Discovery & Ranking Challenge

**Team:** pithva007  
**GitHub:** https://github.com/pithva007/Ai-Recruiter  
**Submission:** `outputs/submission.csv` — 100 rows, validated  
**Submission status:** `Submission is valid.`  
**Top candidate:** CAND_0018499 (score 1.0000)  
**Score range:** 0.4740 – 1.0000  
**Honeypots detected:** 472 / 100,000  

---

## 1. Problem Statement

The challenge: rank the **top 100 candidates** from a pool of **100,000** for the role of  
**Senior AI Engineer — Founding Team at Redrob AI (Series A)**.

The key insight from the JD: this is not a keyword-matching problem. A candidate whose  
skills list contains every AI keyword but whose career shows Marketing Manager →  
Operations Manager is NOT a fit. Career history and actual deployed systems dominate  
over skill keywords.

**Evaluation metric:**
```
Final composite = 0.50 × NDCG@10 + 0.30 × NDCG@50 + 0.15 × MAP + 0.05 × P@10
```
Getting the **top 10 right is worth 5× more** than getting the bottom 90 right.

**Hard constraints (Docker sandbox):**
- Ranking step: < 5 minutes wall-clock
- CPU only, no GPU
- 16 GB RAM
- ZERO network calls during ranking

---

## 2. Architecture Overview

The system has 9 stages organized into 4 phases:

```
Phase A — Offline Pre-computation (no time limit, LLM permitted)
  Stage 1:  JD Analysis          → data/processed/jd_features.json
  Stage 2:  Precompute           → data/processed/features.pkl  (100K scored, 15s)

Phase B — Ranking (< 5 min, CPU, ZERO network)
  Stage 3:  Rank                 → outputs/ranked_top100_raw.csv  (0.3s)

Phase C — Post-Ranking Enrichment (offline, LLM permitted)
  Stage 4:  Reason               → outputs/submission.csv  ← SUBMIT THIS
  Stage 5:  Evidence Extraction  → data/processed/evidence/  (100 files)
  Stage 6:  Graph Builder        → data/processed/knowledge_graph.gexf
  Stage 7:  Hybrid Retrieval     → data/processed/retrieval_results.json
  Stage 8:  LLM Scoring Engine   → data/processed/scores/  (30 files)
  Stage 9:  Explainable Ranking  → outputs/ranked_candidates.csv + shortlist_report.pdf

Phase D — Dashboard
  Stage 10: Streamlit Dashboard  → streamlit run app.py
```

The **submission-critical path** is Phases A+B: precompute → rank → reason.  
Phases C and D enrich the top candidates for human review but are not scored.

---

## 3. Core Scoring Engine (utils/feature_engineering.py)

All scoring is **deterministic — zero LLM, zero network, zero randomness**.

### 3.1 Final Score Formula

```python
final_score = (
    career_score    * 0.30 +   # title + career history ML evidence
    skill_score     * 0.20 +   # depth-weighted AI skills (anti-stuffer)
    retrieval_score * 0.30 +   # production retrieval/ranking/search evidence
    fit_score       * 0.20     # experience band + education tier
) * availability_multiplier    # 8-signal behavioral gate
  * services_penalty           # near-disqualification for pure services
```

**Score range:** [0.0, 1.0] — clamped and rounded to 6dp.

### 3.2 Component 1 — Career Score (weight 0.30)

Measures genuine ML/AI engineering background.

**Inputs:**
- `profile.current_title` → title score via lookup table
- `career_history[].title` → ML/AI role identification per role
- `career_history[].description` → ML keyword evidence (retrieval, ranking, embedding, FAISS...)
- `career_history[].company` → services company blacklist check
- `career_history[].duration_months` → tenure weighting

**Title classification:**
```
1.00 : ML Engineer, AI Engineer, NLP Engineer, Search Engineer, Ranking Engineer
0.85 : Data Scientist, MLOps Engineer, AI Researcher, Research Scientist
0.70 : Data Engineer, Analytics Engineer, Backend Engineer (ML context)
0.50 : Software Developer, Data Analyst, DevOps, Cloud Engineer
0.25 : Business Analyst, Product Manager, Tech Lead, Scrum Master
0.00 : Marketing Manager, HR Manager, Operations Manager, Accountant
0.40 : (default for unknown titles)
```

**Services company penalty (Fix 1 — critical):**
```python
if services_ratio >= 0.80 and no_non_services_role: return 0.05  # near-disqualified
if services_ratio >= 0.80 and has_escape_role:       return 0.40  # significant penalty
if services_ratio >= 0.50:                            return 0.75  # moderate penalty
return 1.0  # no penalty
```

Services blacklist: TCS, Tata Consultancy, Infosys, Wipro, Accenture, Cognizant, Capgemini,  
HCL Technologies, Tech Mahindra, Mphasis, Hexaware, L&T Infotech, LTIMindtree, and 15+ more.

### 3.3 Component 2 — Skill Trust Score (weight 0.20)

Resists keyword stuffing by trusting only skills with demonstrated usage.

**Per-skill scoring:**
```python
if duration == 0 and proficiency in ['expert', 'advanced']:
    skill_weight = 0.05   # claims expertise, zero usage — worst signal
elif duration == 0:
    skill_weight = 0.10   # keyword stuffer — any proficiency
else:
    duration_trust    = min(duration / 24.0, 1.0)
    endorsement_trust = min(endorsements / 20.0, 1.0)
    skill_weight = 0.40 + (0.30 * duration_trust) + (0.30 * endorsement_trust)
```

**JD relevance multiplier:**
- Must-have skills (FAISS, embeddings, retrieval, ranking, PyTorch, LLM...): `2.0×`
- Nice-to-have (LoRA, QLoRA, XGBoost, MLflow, LangChain...): `1.0×`
- Supporting (Spark, Kafka, Docker, Kubernetes...): `0.5×`
- Trap/irrelevant (SEO, Photoshop, SAP, Angular...): `0.0×` (skipped)

Platform-verified `skill_assessment_scores` from redrob_signals override self-reported proficiency  
(70% trust on verified, 30% on self-reported).

### 3.4 Component 3 — Production Retrieval Experience (weight 0.30)

This is the JD's #1 explicit requirement and carries the highest weight.

```python
RETRIEVAL_KEYWORDS = {
    "retrieval", "search", "ranking", "recommendation", "vector",
    "embedding", "similarity", "index", "faiss", "elasticsearch",
    "recommend", "ranker", "rerank", "recall", "precision", "ndcg", ...
}

per role:
  3+ keywords + NOT services + duration >= 6 months → +0.40  (strong: shipped retrieval)
  1-2 keywords + NOT services + duration >= 6 months → +0.15  (weak: some retrieval work)
  3+ keywords + IS services                          → +0.10  (partial: services retrieval)

total capped at 1.0
```

### 3.5 Component 4 — Fit Score (weight 0.20)

```python
experience_band:
  5.5–9.0 years  → 1.00  (ideal band from JD)
  9.0–11.0 years → 0.95
  5.0–5.5 years  → 0.88
  4.0–5.0 years  → 0.72
  < 4.0 years    → 0.35
  > 11.0 years   → 0.85

education_tier:
  tier_1 → 1.0, tier_2 → 0.8, tier_3 → 0.6, tier_4 → 0.4, unknown → 0.5

fit_score = (exp_mult * 0.70) + (edu_tier * 0.30)
```

### 3.6 Availability Multiplier (8 signals — multiplicative)

This is applied as a **multiplier on the final score**, not an additive component.  
A perfect-on-paper candidate who is unreachable or inactive scores near zero.

```python
Signal 1 — last_active_date (reference: 2026-06-01):
  > 180 days inactive: × 0.40  (effectively gone)
  > 90 days:           × 0.65  (likely passive)
  > 30 days:           × 0.85  (slightly passive)

Signal 2 — open_to_work_flag:
  False: × 0.70

Signal 3 — recruiter_response_rate:
  < 0.10: × 0.50   (very hard to reach)
  < 0.25: × 0.75
  > 0.70: × 1.10   (bonus: responsive)

Signal 4 — notice_period_days:
  ≤ 15:  × 1.05   (near-immediate start)
  ≤ 30:  × 1.00   (JD says can buy out 30 days)
  ≤ 60:  × 0.90
  ≤ 90:  × 0.75
  > 90:  × 0.55   (significant barrier)

Signal 5 — interview_completion_rate:
  < 0.40: × 0.60  (ghosts interviews)
  < 0.60: × 0.80

Signal 6 — location + willing_to_relocate:
  Outside India:             × 0.20  (no visa sponsorship)
  Wrong city, won't relocate: × 0.50
  Wrong city, will relocate:  × 0.90

Signal 7 — github_activity_score:
  > 60: × 1.08  (active OSS — JD explicitly values this)
  > 30: × 1.03

Signal 8 — verified_email AND verified_phone both False:
  × 0.80  (unverified identity)

Cap at 1.15 (bonuses ≤ 15%)
```

### 3.7 Honeypot Detection

472 honeypot candidates detected and scored 0.0 (prevented from appearing in top 100).

```python
Rule 1: 5+ expert/advanced skills with duration_months == 0 → honeypot
Rule 2: profile.years_of_experience > 2 AND career_months < 6 → honeypot
```

### 3.8 Additional Penalties

**Title-chaser penalty** (≥3 roles, most jobs < 18 months):
```python
short_tenure_ratio >= 0.70: × 0.60
short_tenure_ratio >= 0.50: × 0.80
```

**Experience band multiplier** ensures candidates outside 5.5–9 year band are penalized  
proportionally — both under-experienced and over-experienced candidates score lower.

---

## 4. The Ranking Pipeline (src/rank.py)

```bash
python src/rank.py \
  --candidates ./data/raw/candidates.jsonl \
  --features   ./data/processed/features.pkl \
  --out        ./outputs/submission.csv
```

**Fast path (features.pkl exists):** 0.34 seconds  
**Slow path (streaming candidates.jsonl):** 14.7 seconds  

Both paths verified correct and identical rankings.

The script has **zero network calls**. All imports are Python stdlib:  
`argparse, csv, json, os, pickle, re, sys, time, pathlib`

The `feature_engineering` module is imported lazily inside the slow-path function body  
to prevent accidental network calls at module load time.

---

## 5. LLM Usage (Gemini API)

LLM is used in **exactly two places**, both offline and post-ranking:

### 5.1 Stage 1 — JD Analysis (runs ONCE)

```
Model:       gemini-flash-lite-latest
Temperature: 0.0
Calls:       1 total
Input:       job_description.docx text (~9,500 chars)
Output:      data/processed/jd_features.json
```

Extracts: role_title, seniority_level, must_have_skills, nice_to_have_skills,  
explicit_disqualifiers, services_company_names, implicit_requirements,  
ideal_candidate_summary, jd_trap_warning.

### 5.2 Stage 4 — Reasoning Generation (top 100 only)

```
Model:       gemini-flash-lite-latest
Temperature: 0.0
Calls:       100 (one per candidate)
Rate limit:  7s inter-call wait (free-tier: 10 RPM)
Input:       candidate snapshot (title, yoe, skills, response_rate, notice)
Output:      reasoning column in submission.csv
```

**Anti-hallucination rules enforced:**
- Never mention skills not in candidate's actual skills list
- Never invent impact numbers
- Never claim ML experience not evidenced in career history
- Never write identical reasoning for two candidates
- Reference actual `current_title`, `years_experience`, real skill names

**Fallback:** `--no-llm` mode uses deterministic rule-based reasoning with actual candidate  
fields (title, years, top skills, response rate). Always produces 100 unique, factual strings.

### 5.3 Stage 6 — LLM Deep Scoring (top 30 only, optional enrichment)

```
Model:       gemini-flash-lite-latest
Temperature: 0.0 (scoring), 0.3 (interview questions)
Calls:       30 × 2 = 60 (scoring + interview questions per candidate)
Input:       JD schema + candidate profile + evidence items
Output:      data/processed/scores/{cid}_scores.json
```

Four LLM-scored dimensions: fit_score, impact_score, potential_score, risk_score (all 0-100).

```python
composite = (fit * 0.35) + (impact * 0.30) + (potential * 0.20) + ((100 - risk) * 0.15)
```

This is for the **enriched dashboard** (Stage 9), not the submission CSV.

### 5.4 Stage 7b — Dark Horse Discovery (top 30 only, optional)

```
Model:       gemini-flash-lite-latest
Calls:       up to 30
Criteria:    hybrid_rank > 15 AND (impact >= 75 OR potential >= 75) AND fit >= 50
Output:      dark_horse flag + transferable_skills_map in ranked_candidates.csv
```

12 dark horse candidates identified — candidates a traditional ATS would miss.

---

## 6. Evidence Extraction Pipeline (Stages 5-9)

These stages enrich the top 100 candidates for human review.  
They do NOT affect the submission CSV.

### Stage 5 — Evidence Extraction

- Input: `data/raw/candidates.jsonl` + `outputs/ranked_top100_raw.csv`
- LLM: yes (1 call per candidate, top 100)
- Output: `data/processed/evidence/{cid}_evidence.json` (100 files)
- Per candidate: 9.3 evidence items avg, 305 quantified items total

Evidence types: technical, impact, leadership, learning, behavioral  
Quantified evidence (contains numbers) weighted 1.5× unquantified evidence.

### Stage 6 — GraphRAG Knowledge Graph (Stage 4 in pipeline)

- Input: evidence files + jd_features.json
- LLM: no (pure networkx computation)
- Output: `data/processed/knowledge_graph.gexf` (868 nodes, 3,297 edges)

Node types: CANDIDATE (100), SKILL (407), TOOL (180), IMPACT_KEYWORD (154),  
DOMAIN (7), JD_REQUIREMENT (20)

### Stage 7 — Hybrid Retrieval (Stage 5 in pipeline)

- Input: knowledge graph + evidence files
- LLM: no (FAISS + graph)
- Output: `data/processed/retrieval_results.json`

```python
hybrid_score = (faiss_similarity_normalized * 0.6) + (graph_score * 0.4)
```

Uses `sentence-transformers/all-mpnet-base-v2` (768-dim, L2-normalised)  
for JD → candidate semantic similarity. FAISS `IndexFlatIP` (inner product = cosine after normalisation).

### Stage 8 — LLM Scoring Engine (Stage 6 in pipeline)

- 30 candidates deep-scored
- Pydantic validation on all LLM outputs (scores clamped to [0,100])
- Impact score cap: max 40 if zero quantified evidence

### Stage 9 — Explainable Ranking + Dark Horse (Stage 7 in pipeline)

- Sorts by LLM composite score
- Generates ≤100 word rationales per candidate
- Identifies 12 dark horses
- Output: `outputs/ranked_candidates.csv` (30 candidates, 19 columns)

### Stage 10 — PDF Report

- 17-page reportlab PDF: cover, executive summary, top-10 candidate pages, dark horses, methodology
- Output: `outputs/shortlist_report.pdf`

---

## 7. Data Sources & File Layout

```
India_runs_data_and_ai_challenge/    ← SOURCE OF TRUTH — read-only
  candidates.jsonl                   ← 100,000 candidates (465 MB)
  candidate_schema.json
  job_description.docx
  validate_submission.py

data/raw/
  candidates.jsonl → symlink (no duplication of 465 MB)
  job_description.docx
  sample_candidates.json

data/processed/
  jd_features.json          (8 KB)   ← Stage 1
  features.pkl               (17 MB)  ← Stage 2 (all 100K scored)
  evidence/                  (100 JSONs) ← Stage 5
  knowledge_graph.gexf       (1 MB)   ← Stage 6
  retrieval_results.json     (19 KB)  ← Stage 7
  scores/                    (30 JSONs) ← Stage 8

outputs/
  submission.csv             ← SUBMIT THIS (100 rows, validated)
  ranked_top100_raw.csv      ← Phase B output (no reasoning)
  ranked_candidates.csv      ← Stage 9 output (30 deep-scored candidates)
  ranking_summary.json
  shortlist_report.pdf       (17 pages)
```

---

## 8. Candidate Schema (100,000 candidates)

Each candidate has 6 top-level sections:

| Section | Fields |
|---|---|
| `profile` | anonymized_name, headline, summary, years_of_experience, current_title, current_company, current_company_size, current_industry, location, country |
| `career_history[]` | company, title, start_date, end_date, duration_months, is_current, industry, company_size, description |
| `education[]` | institution, degree, field_of_study, start_year, end_year, grade, tier (tier_1/2/3/4/unknown) |
| `skills[]` | name, proficiency (beginner/intermediate/advanced/expert), endorsements, duration_months |
| `certifications[]` | name, issuer, year |
| `redrob_signals` | 23 behavioral signals (see Section 3.6) |

**Key sentinel values:**
- `github_activity_score = -1` → no GitHub linked (treated as neutral)
- `offer_acceptance_rate = -1` → no offer history (treated as neutral)
- `skill_assessment_scores = {}` → no platform assessments

---

## 9. Results & Validation

### Submission CSV Quality

| Metric | Value |
|---|---|
| Rows | 100 |
| Score range | 0.4740 – 1.0000 |
| Scores non-increasing | Yes |
| Unique candidate IDs | 100 |
| All IDs valid (CAND_XXXXXXX) | Yes |
| Empty reasoning | 0 |
| Unique reasoning strings | 100 |
| Honeypots in top 100 | 0 |
| Services-penalized in top 100 | 0 |
| Validator result | `Submission is valid.` |

### Top 10 Ranked Candidates

| Rank | Candidate ID | Score | Title |
|---|---|---|---|
| 1 | CAND_0018499 | 1.0000 | Senior Machine Learning Engineer |
| 2 | CAND_0081846 | 0.9086 | Lead AI Engineer |
| 3 | CAND_0077337 | 0.8872 | Staff Machine Learning Engineer |
| 4 | CAND_0046525 | 0.8858 | Senior Machine Learning Engineer |
| 5 | CAND_0050454 | 0.8840 | AI Engineer |
| 6 | CAND_0064326 | 0.8836 | Search Engineer |
| 7 | CAND_0068811 | 0.8391 | Applied ML Engineer |
| 8 | CAND_0042506 | 0.8129 | Search Engineer |
| 9 | CAND_0005509 | 0.8121 | Data Scientist |
| 10 | CAND_0068932 | 0.7598 | ML Engineer |

### Scale & Runtime

| Operation | Time | Count |
|---|---|---|
| Precompute all 100K | 15.2s | 100,000 candidates |
| Rank (fast path, pkl) | 0.34s | 100,000 → top 100 |
| Rank (slow path, jsonl) | 14.7s | 100,000 → top 100 |
| Evidence extraction (LLM) | ~12 min | 100 candidates |
| Reasoning generation (LLM) | ~12 min | 100 candidates |
| LLM deep scoring | ~10 min | 30 candidates |

---

## 10. Hackathon Traps & How We Beat Them

### Trap 1: Keyword stuffers

**The trap:** A candidate with 50 AI skills listed but career history of  
Marketing Manager → Operations Manager → HR Manager.

**Our defence:** Career history dominates (30% weight). Skills need  
`duration_months > 0` to count. Expert skills with 0 months get `0.05` weight  
instead of full credit. The retrieval experience component (30% weight) requires  
actual career descriptions with retrieval/ranking keywords — not just skill list claims.

### Trap 2: Services company candidates

**The trap:** Candidates from TCS/Infosys/Wipro with impressive-sounding AI titles  
and skill lists but no product deployment experience.

**Our defence:** Pure services careers get `0.05×` multiplier on the final score —  
near-disqualification. This pushes them from rank ~70 to below rank 5,000.

### Trap 3: Honeypot candidates

**The trap:** ~472 candidates with logically impossible profiles (9 years experience,  
0 months career history; 5+ expert skills with 0 months duration each).

**Our defence:** Two honeypot rules detect and score them 0.0, preventing any  
from appearing in top 100.

### Trap 4: Inactive/unreachable candidates

**The trap:** Perfect-on-paper candidates who haven't logged in for 6 months and  
have a 5% recruiter response rate — they look great but can't be hired.

**Our defence:** The 8-signal multiplicative availability multiplier. Such a  
candidate gets: `0.40 × 0.50 = 0.20×` applied to their final score. A candidate  
scoring 0.85 on skills+career becomes 0.17 after the multiplier.

---

## 11. Reproduce Commands

### Full pipeline (all stages):

```bash
# Prerequisites
pip install -r requirements.txt
cp .env.example .env  # add GEMINI_API_KEY

# Phase A: Pre-computation (run once, ~15s total)
python src/stage1_jd_analysis.py        # JD → jd_features.json (LLM, once)
python src/precompute.py                 # 100K candidates → features.pkl (~15s)

# Phase B: Ranking (the submission-critical step, ~0.3s)
python src/rank.py \
  --candidates ./data/raw/candidates.jsonl \
  --features   ./data/processed/features.pkl \
  --out        ./outputs/ranked_top100_raw.csv

# Phase C: Reasoning (LLM, ~12 min for 100 candidates)
python src/reason.py \
  --raw outputs/ranked_top100_raw.csv \
  --out outputs/submission.csv

# Validate
python validate_submission.py outputs/submission.csv
# → Submission is valid.
```

### Dashboard:

```bash
streamlit run app.py
```

### Full pipeline in one command:

```bash
python run_pipeline.py
```

---

## 12. File Index

| File | Purpose |
|---|---|
| `src/rank.py` | Phase B: sort features.pkl → top 100 CSV (0.3s, zero network) |
| `src/precompute.py` | Phase A: score 100K candidates → features.pkl |
| `src/stage1_jd_analysis.py` | Phase A: parse JD → jd_features.json (LLM once) |
| `src/reason.py` | Phase C: generate LLM reasoning for top 100 |
| `src/stage3_evidence_extraction.py` | Extract evidence items (LLM, top 100) |
| `src/stage4_graph_builder.py` | Build GraphRAG knowledge graph |
| `src/stage5_hybrid_retrieval.py` | FAISS + graph hybrid retrieval |
| `src/stage6_scoring_engine.py` | LLM deep scoring (top 30) |
| `src/stage7_ranking.py` | Explainable ranking + dark horse detection |
| `src/stage8_dashboard.py` | Streamlit dashboard logic |
| `src/stage9_pdf_report.py` | PDF shortlist report |
| `app.py` | Streamlit entry point |
| `utils/feature_engineering.py` | All scoring functions (deterministic, no network) |
| `utils/llm_client.py` | Gemini API wrapper with retry/rate-limit handling |
| `utils/embedding_client.py` | sentence-transformers wrapper (local, offline) |
| `utils/json_validator.py` | Pydantic models for all pipeline I/O |
| `utils/report_generator.py` | reportlab PDF generator |
| `run_pipeline.py` | Single-command full pipeline orchestrator |
| `validate_submission.py` | Official submission validator |
| `AGENT.md` | Scoring contracts, JD requirements, honeypot detection |
| `CLAUDE.md` | LLM prompts, anti-hallucination rules |
| `SKILLS.md` | Technical reference, formulas, skill lists |

---

## 13. Key Design Decisions

**Why career (30%) + retrieval (30%) instead of skills alone?**  
The JD explicitly warns: "The right answer is not find candidates whose skills section  
contains the most AI keywords." A candidate who built a recommendation system at  
Flipkart without using the word "RAG" outranks a candidate who lists RAG/FAISS/Pinecone  
as skills with 0 months duration. Career history and actual deployed systems are ground truth.

**Why a fixed reference date (2026-06-01) for availability?**  
Using `date.today()` would cause score drift across days — the same candidate would  
rank differently depending on when precompute.py was run. Fixed reference = reproducible  
rankings in the Docker sandbox.

**Why services penalty applied twice (in career_score AND final_score)?**  
Career score applies it to the career component only. The final score penalty applies  
to the whole score, ensuring pure-services candidates cannot compensate with high  
skill or retrieval scores. Double application is intentional.

**Why 0.05 (not 0.0) for pure-services candidates?**  
A score of exactly 0.0 would be indistinguishable from honeypots. 0.05 keeps them  
rankable but far outside the top 100 (rank ~5,000+).

**Why separate precompute.py + rank.py instead of one script?**  
The challenge requires the ranking step to run in < 5 minutes in a Docker container  
with no pre-existing files. precompute.py runs offline once; rank.py loads the pkl and  
sorts in 0.3 seconds. In a fresh Docker without pkl, rank.py falls back to streaming  
candidates.jsonl and scoring on the fly (~15s).

---

*Report generated: June 2026*  
*All scores verified on real 100,000-candidate hackathon dataset.*
