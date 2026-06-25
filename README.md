<div align="center">

# 🧠 AI Recruiter

### Intelligent Candidate Discovery & Ranking System

**Redrob AI Hackathon 2025 — Track 01: Intelligent Candidate Discovery**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Gemini](https://img.shields.io/badge/Gemini%20AI-Primary%20LLM-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev)
[![Groq](https://img.shields.io/badge/Groq-Failover%20LLM-F55036?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS-0071C5?style=for-the-badge)](https://faiss.ai)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

> **Ranking 100,000 candidates for a Senior AI Engineer role in < 1 second.**  
> Zero GPU. Zero network during ranking. Pure deterministic intelligence.

---

[Quick Start](#-quick-start) · [Architecture](#-architecture) · [Scoring Engine](#-scoring-engine) · [Setup](#-setup) · [Dashboard](#-dashboard) · [Validation](#-validation)

</div>

---

## ⚡ Quick Start (Reproduce Submission)

```bash
git clone https://github.com/pithva007/Ai-Recruiter.git
cd Ai-Recruiter

pip install -r requirements.txt
cp .env.example .env          # Add your GEMINI_API_KEY and GROQ_API_KEY

# ── Fast path: rank from pre-computed features (< 1 second, ZERO network) ──
python src/rank.py \
  --candidates ./data/raw/candidates.jsonl \
  --features   ./data/processed/features.pkl \
  --out        ./outputs/submission.csv

# ── Validate submission ──
python validate_submission.py ./outputs/submission.csv
# → Submission is valid. ✅
```

> **One-command full pipeline:**
>
> ```bash
> python run_pipeline.py   # runs all 9 stages end-to-end with timing
> ```

---

## 🏗️ Architecture

```
╔══════════════════════════════════════════════════════════════════════════════╗
║               AI RECRUITER — 9-STAGE INTELLIGENT PIPELINE                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

  INPUT
  ┌────────────────────────────┐    ┌────────────────────────────────────────┐
  │  job_description.docx      │    │  candidates.jsonl  (100,000 profiles)  │
  │  (Senior AI Engineer JD)   │    │  465 MB · 23 Redrob signals each       │
  └──────────────┬─────────────┘    └────────────────────┬───────────────────┘
                 │                                        │
  ╔══════════════▼═══════════════╗                        │
  ║  PHASE A — OFFLINE PRE-COMP  ║◄───────────────────────┘
  ║  (No time limit, LLM OK)     ║
  ╚══════════════╤═══════════════╝
                 │
        ┌────────▼────────┐
        │   STAGE  1      │  LLM: Gemini → Groq failover
        │   JD Analysis   │  Extracts: must-haves, nice-to-haves,
        │                 │  implicit requirements, ideal profile
        └────────┬────────┘
                 │ jd_features.json
        ┌────────▼────────┐
        │   STAGE  2      │  CPU only · No network · ~14 seconds
        │   Precompute    │  Scores ALL 100K candidates deterministically
        │   100K Features │  ─────────────────────────────────────────
        │                 │  • career_score    (title + history + services)
        │                 │  • skill_score     (trust-weighted depth)
        │                 │  • retrieval_score (production IR experience)
        │                 │  • fit_score       (location + YoE + notice)
        │                 │  • behavioral mult (8 Redrob availability signals)
        │                 │  • honeypot flags  (7 impossibility signals)
        └────────┬────────┘
                 │ features.pkl  (17 MB)
                 │
  ╔══════════════▼═══════════════╗
  ║  PHASE B — RANKING CORE      ║  ⚡ < 1 SECOND · ZERO NETWORK · CPU ONLY
  ║  (Compute-constrained)       ║
  ╚══════════════╤═══════════════╝
                 │
        ┌────────▼────────┐
        │   rank.py        │  Loads features.pkl → sort by final_score
        │   Sort Top 100   │  Streaming fallback if pkl missing (~15s)
        └────────┬────────┘
                 │ ranked_top100_raw.csv
                 │
  ╔══════════════▼═══════════════╗
  ║  PHASE C — POST-RANKING      ║
  ║  (Offline, LLM OK, top-100)  ║
  ╚══════════════╤═══════════════╝
                 │
        ┌────────▼────────┐
        │   reason.py      │  LLM-generated reasoning per candidate
        │   Reasoning      │  Anti-hallucination guards enforced
        │   Generation     │  Gemini → Groq seamless failover
        └────────┬────────┘
                 │ submission.csv  ◄─── SUBMIT THIS
                 │
  ╔══════════════▼═══════════════╗
  ║  PHASE D — DEEP ENRICHMENT   ║  Optional · LLM-augmented · Offline
  ╚══════════════╤═══════════════╝
                 │
        ┌────────▼────────┐
        │   STAGE  3      │  LLM per candidate · Pydantic-validated
        │   Evidence      │  Extracts: claims, entities, confidence
        │   Extraction    │
        └────────┬────────┘
                 │ evidence/{cid}_evidence.json
        ┌────────▼────────┐
        │   STAGE  4      │  networkx · GEXF export
        │   GraphRAG      │  Nodes: JD_REQ, SKILL, COMPANY, CANDIDATE
        │   Knowledge     │  Edges: HAS_SKILL, WORKED_AT, MATCHES_REQ
        │   Graph Builder │
        └────────┬────────┘
                 │ knowledge_graph.gexf
        ┌────────▼────────┐
        │   STAGE  5      │  sentence-transformers · FAISS IndexFlatIP
        │   Hybrid        │  FAISS vector search      (60% weight)
        │   Retrieval     │  + GraphRAG entity match   (40% weight)
        │                 │  → Top-30 candidate pool
        └────────┬────────┘
                 │ retrieval_results.json
        ┌────────▼────────┐
        │   STAGE  6      │  LLM per candidate (temp 0.0)
        │   LLM Scoring   │  4-dimensional intelligence scoring:
        │   Engine        │  • fit_score      (evidence vs JD match)
        │                 │  • impact_score   (quantified achievements)
        │                 │  • potential_score (velocity + growth + learning)
        │                 │  • risk_score     (gaps, tenure, domain mismatch)
        │                 │  + 3 tailored interview questions generated
        └────────┬────────┘
                 │ scores/{cid}_scores.json
        ┌────────▼────────┐
        │   STAGE  7a/7b  │  7a: LLM rationale per candidate (≤100 words)
        │   Explainable   │  7b: Dark Horse Discovery Agent
        │   Ranking +     │      (missed by ATS → shortlisted by AI)
        │   Dark Horse    │  → ranked_candidates.csv (19 columns)
        └────────┬────────┘
                 │
  ╔══════════════▼═══════════════╗
  ║  PHASE E — OUTPUTS           ║
  ╚══════════════╤═══════════════╝
                 │
     ┌───────────┼────────────┐
     ▼           ▼            ▼
 ┌───────┐  ┌────────┐  ┌──────────────┐
 │  PDF  │  │  CSV   │  │  Streamlit   │
 │Report │  │ Output │  │  Dashboard   │
 │ (17pg)│  │ Submit │  │   app.py     │
 └───────┘  └────────┘  └──────────────┘
```

---

## 🔢 Scoring Engine

The core of our system is a **deterministic, zero-network scoring formula** applied to all 100K candidates.

### Final Score Formula (v4)

```
final_score = (
    career_score    × 0.30   ←  title + career arc + product company history
    skill_score     × 0.20   ←  trust-weighted AI/ML skill depth
    retrieval_score × 0.30   ←  production IR/ranking/embedding experience
    fit_score       × 0.20   ←  location + notice period + YoE + education
)
× behavioral_multiplier        ←  8 Redrob availability signals [0.4 – 1.15]
× services_penalty             ←  5-tier blacklist penalty [0.05 – 1.00]
× title_relevance_gate         ←  hard gate for irrelevant titles [0.05 / 1.0]
× must_have_coverage_gate      ←  NDCG@10 calibration: skill group coverage [0.25 – 1.0]
× honeypot_suspicion           ←  soft borderline-fake penalty [0.50 – 1.00]
```

> **Honeypots** — confirmed impossible profiles → score clamped to `0.0`

---

### 🕵️ Honeypot Detection (7 Signals)

Our honeypot detection exceeds the competition with **7 hard disqualification signals**:

| #   | Signal                                     | Threshold                                                  |
| --- | ------------------------------------------ | ---------------------------------------------------------- |
| 1   | Claimed YoE inflated vs actual career span | YoE > career_months/12 + 2yr **and** YoE > 5               |
| 2   | Expert skills declared but never used      | ≥ **3** expert/advanced skills with `duration_months == 0` |
| 3   | Impossible total expert skill count        | ≥ **12** expert-level skills total                         |
| 4   | Impossible tenure at young startups        | Sarvam AI > 38mo · Krutrim > 30mo                          |
| 5   | Copy-paste career descriptions             | ≥ 3 identical non-empty descriptions                       |
| 6   | Ghost completeness                         | `completeness > 85` but ALL descriptions empty             |
| 7   | Overlapping fake tenures                   | Total career months > 2.5× declared YoE                    |

**Soft suspicion scoring** — borderline profiles receive a `0.50–0.85×` multiplier without hard disqualification.

---

### ⚖️ Services Company Penalty (5-Tier)

Candidates whose careers are dominated by pure IT services firms receive tiered penalties:

| Career Composition                       | Multiplier | Rationale                              |
| ---------------------------------------- | ---------- | -------------------------------------- |
| ≥ 80% services, **no** product role ever | **0.05×**  | Pure services — near disqualification  |
| ≥ 80% services, **with** ≥1 product role | **0.40×**  | Escaped services — significant penalty |
| 70–79% services                          | **0.45×**  | Heavy services — notable penalty       |
| 50–69% services                          | **0.70×**  | Mixed — moderate penalty               |
| 25–49% services                          | **0.85×**  | Services-leaning — light penalty       |
| < 25% services                           | **1.00×**  | No penalty                             |

Services blacklist includes: `TCS · Infosys · Wipro · Accenture · Cognizant · Capgemini · HCL · Mindtree · Tech Mahindra · IBM GBS · Mphasis · Hexaware · NIIT · Cyient · LTIMindtree · Persistent Systems`

---

### 🎯 Must-Have Skill Coverage Gate

Directly calibrated to maximize **NDCG@10** (50% of evaluation weight):

| Must-Have Groups Covered (of 8) | Gate Multiplier                |
| ------------------------------- | ------------------------------ |
| ≥ 5 groups                      | **1.00×** — clearly qualified  |
| 3–4 groups                      | **0.85×** — borderline         |
| 1–2 groups                      | **0.50×** — weak coverage      |
| 0 groups                        | **0.25×** — no relevant skills |

**The 8 JD Must-Have Skill Groups:** Embeddings · Vector Databases · Ranking/IR · Evaluation Metrics · Python · NLP/Transformers · ML Core · Search Systems

---

### 🤖 LLM Failover: Gemini → Groq

Our pipeline features **seamless, zero-delay failover** between LLM providers:

```
LLM Request
     │
     ▼ Attempt 1 (immediate)
 ┌─────────┐   success → return result
 │  Gemini  │
 │  Flash   │   fail (rate limit / error)
 └────┬────┘          │
      │                ▼ Attempt 2 (immediate, no delay)
      │           ┌─────────┐   success → return result
      │           │  Groq    │
      │           │  Llama   │   fail → exponential backoff retry loop
      │           └─────────┘
      └─────────────────────────────────────────────────► result
```

> **No perceptible delay.** Groq triggers instantly on Gemini failure — not after backoff.

---

## 🏆 Competitive Edge

What makes our system different from traditional ATS approaches:

| Capability                 | Our System                                          | Traditional ATS      |
| -------------------------- | --------------------------------------------------- | -------------------- |
| **Semantic Understanding** | FAISS + sentence-transformers                       | Keyword matching     |
| **Career Context**         | 8-stage pipeline with LLM evidence extraction       | Resume field parsing |
| **Honeypot Detection**     | 7-signal system with soft suspicion scoring         | None                 |
| **LLM Scoring**            | 4-dimensional: fit + impact + potential + risk      | Not applicable       |
| **Dark Horse Discovery**   | Stage 7b agent identifies missed top candidates     | Not applicable       |
| **Reasoning**              | Per-candidate, anti-hallucination guarded, grounded | Template strings     |
| **Interview Questions**    | 3 tailored questions per candidate (Stage 6)        | Not applicable       |
| **Knowledge Graph**        | GraphRAG with networkx GEXF                         | Not applicable       |
| **Ranking Speed**          | **< 1 second** (100K candidates)                    | Minutes to hours     |

---

## ⚙️ Setup

### 1. Environment

```bash
git clone https://github.com/pithva007/Ai-Recruiter.git
cd Ai-Recruiter
pip install -r requirements.txt
cp .env.example .env
```

### 2. Environment Variables (`.env`)

```env
# Primary LLM — Gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash-lite

# Failover LLM — Groq (instant failover if Gemini hits rate limits)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama3-70b-8192

# Logging
LOG_LEVEL=INFO
```

### 3. Pre-computation (one-time setup)

```bash
# Step 1 — Analyze the JD with LLM (~30s, needs GEMINI_API_KEY)
python src/stage1_jd_analysis.py

# Step 2 — Score all 100K candidates (~14s, pure CPU, no network)
python src/precompute.py

# Done. features.pkl is now ready for instant ranking.
```

---

## 📁 Project Structure

```
ai-recruiter/
│
├── 🚀 run_pipeline.py              ← Single command: runs all 9 stages with timing
├── 📊 app.py                       ← Streamlit dashboard entry point
├── ✅ validate_submission.py       ← Official submission validator
├── 📋 submission_metadata.yaml     ← Hackathon submission metadata
│
├── 📖 AGENT.md                     ← Scoring contracts, JD requirements, honeypot spec
├── 📖 CLAUDE.md                    ← LLM prompts, anti-hallucination rules
├── 📖 SKILLS.md                    ← Technical reference: formulas, skill lists, FAISS patterns
│
├── src/
│   ├── stage1_jd_analysis.py       ← [LLM] JD → structured requirements schema
│   ├── precompute.py               ← [CPU] 100K candidates → features.pkl
│   ├── rank.py                     ← [< 1s, no network] features.pkl → ranked CSV
│   ├── reason.py                   ← [LLM] top-100 → submission.csv with reasoning
│   ├── stage3_evidence_extraction.py  ← [LLM] Per-candidate evidence items (Pydantic)
│   ├── stage4_graph_builder.py     ← [CPU] GraphRAG knowledge graph (networkx)
│   ├── stage5_hybrid_retrieval.py  ← [CPU] FAISS + Graph hybrid retrieval
│   ├── stage6_scoring_engine.py    ← [LLM] 4D scoring + interview questions
│   └── stage7_ranking.py           ← [LLM] Rationale generation + Dark Horse Discovery
│
├── utils/
│   ├── feature_engineering.py      ← ⭐ All scoring functions (v4, 100% deterministic)
│   ├── llm_client.py               ← Gemini → Groq failover client
│   ├── embedding_client.py         ← sentence-transformers (offline, no API)
│   ├── json_validator.py           ← Pydantic models for all pipeline I/O
│   └── report_generator.py         ← reportlab PDF shortlist generator
│
├── data/
│   ├── raw/                        ← Input files + symlinks to challenge bundle
│   │   ├── candidates.jsonl        ← Symlink → 465 MB pool (not committed)
│   │   └── job_description.docx   ← JD source
│   └── processed/                  ← Generated artifacts
│       ├── features.pkl            ← 17 MB pre-scored 100K candidates
│       ├── jd_features.json        ← Stage 1 output
│       ├── evidence/               ← Per-candidate evidence items
│       ├── scores/                 ← Per-candidate LLM scores
│       └── knowledge_graph.gexf   ← GraphRAG graph
│
└── outputs/                        ← Final submission artifacts
    ├── submission.csv              ← ✅ SUBMIT THIS (100 rows, validated)
    ├── ranked_top100_raw.csv       ← rank.py output (no reasoning)
    ├── ranked_candidates.csv       ← 30 deeply-scored candidates (19 columns)
    └── shortlist_report.pdf        ← 17-page PDF shortlist report
```

---

## 📏 Compute Constraints Compliance

| Constraint    | Limit       | Our System                                       |
| ------------- | ----------- | ------------------------------------------------ |
| ⏱️ Runtime    | ≤ 5 min     | **< 1 second** (from `features.pkl`)             |
| 🧠 Memory     | ≤ 16 GB RAM | **~2 GB** (`features.pkl` = 17 MB)               |
| 💻 CPU Only   | Required    | **Zero GPU** at any stage                        |
| 🌐 No Network | Required    | **`rank.py` has zero network calls**             |
| 🔄 Streaming  | Recommended | Streams `candidates.jsonl` if pkl missing (~15s) |

---

## 📊 Dashboard

```bash
streamlit run app.py
```

**Features:**

- 🎛️ **Hiring Decision Simulator** — interactive weight sliders per scoring dimension
- 📡 **Radar Charts** — per-candidate skill fingerprint visualization
- 🌟 **Dark Horse Spotlight** — candidates missed by traditional ATS
- ⚖️ **Bias Audit Panel** — services company and location distribution stats
- 🔍 **Candidate Deep-Dive** — evidence items, interview questions, LLM rationale

---

## ✅ Validation

```bash
python validate_submission.py outputs/submission.csv
# → Submission is valid. ✅
```

The validator checks:

- Exactly 100 rows with `candidate_id, rank, score, reasoning` columns
- Ranks 1–100, each appearing exactly once
- Scores non-increasing: `score[rank_i] >= score[rank_{i+1}]`
- All `candidate_id` values match `CAND_[0-9]{7}` and exist in the pool

---

## 📐 Evaluation Metric Alignment

Our scoring weights are calibrated against the official evaluation formula:

```
Final composite = 0.50 × NDCG@10      ← Highest weight — top 10 matter most
               + 0.30 × NDCG@50
               + 0.15 × MAP
               + 0.05 × P@10
```

**Our NDCG@10 optimizations:**

- Must-have skill coverage gate (P2) — hard discriminator at the top
- Reduced title default score from 0.40 → 0.25 (P1-b)
- Honeypot suspicion multiplier (P0-b) — keeps borderline fakes out of top-10

---

## 🔬 Key Design Decisions

1. **Retrieval dominates** — `retrieval_score` carries `0.30` weight, equal to career. The JD's #1 explicit requirement is production search/ranking/embedding experience. We detect it via 30+ keyword patterns across career descriptions.

2. **Skills trust, not keyword count** — a skill with `duration_months = 0` and `proficiency = expert` is a stuffer flag. Zero-duration expert skills get a `0.05–0.10×` trust multiplier regardless of endorsements.

3. **Behavioral availability** — an inactive candidate (`last_active_date > 180 days`) loses up to 60% of their score. We hire people who respond to recruiters, not ghosts.

4. **Services penalty is brutal by design** — the JD explicitly rejects pure-services backgrounds. A pure TCS/Infosys career gets `0.05×` — effectively disqualified.

5. **Dark Horse recovery** — Stage 7b surfaces candidates who ranked below position 15 in vector search but have `impact_score >= 75` or `potential_score >= 75` with `fit_score >= 50`. Traditional ATS would miss them entirely.

6. **LLM never during ranking** — zero LLM calls during the 100K candidate scoring phase. LLMs are used only for JD analysis (once), evidence extraction (top-30), and reasoning generation (top-100).

---

<div align="center">

**Built for Redrob AI Hackathon 2026**  
_Intelligent Candidate Discovery — Track 01_

</div>
