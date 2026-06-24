# Challenge Analysis — Redrob Intelligent Candidate Discovery & Ranking Challenge

## Dataset Overview

| Item | Value |
|---|---|
| Total candidates | 100,000 (candidates.jsonl, one JSON object per line) |
| Sample provided | 6 candidates in sample_candidates.json |
| Job description | Senior AI Engineer — Founding Team at Redrob AI (Series A) |
| Submission format | CSV: candidate_id, rank, score, reasoning |
| Submission size | Exactly top 100 candidates |
| Evaluation metric | NDCG@10 (50%) + NDCG@50 (30%) + MAP (15%) + P@10 (5%) |
| Compute constraint | 5 minutes, CPU only, 16 GB RAM, NO network calls during ranking |
| Honeypot count | ~80 candidates with impossible profiles — disqualified if >10% in top 100 |

---

## File-by-File Analysis

### candidate_schema.json
```json
{
  "file_name": "candidate_schema.json",
  "purpose": "JSON Schema (draft-07) defining the exact structure of every candidate in candidates.jsonl",
  "important_findings": [
    "candidate_id pattern: CAND_[0-9]{7} — 7-digit zero-padded integer",
    "6 required top-level sections: candidate_id, profile, career_history, education, skills, redrob_signals",
    "redrob_signals has 23 required sub-fields — this is the behavioral layer",
    "skills items have: name, proficiency (beginner/intermediate/advanced/expert), endorsements (int), duration_months (int)",
    "education items have a 'tier' field: tier_1 through tier_4 or 'unknown' — institution prestige signal",
    "career_history items have duration_months (pre-computed), company_size enum, industry string",
    "github_activity_score and offer_acceptance_rate can be -1 (no data) — must be handled as null/absent",
    "expected_salary_range_inr_lpa has min and max in INR Lakhs Per Annum",
    "profile.current_company_size is an enum: 1-10, 11-50, 51-200, 201-500, 501-1000, 1001-5000, 5001-10000, 10001+"
  ],
  "architecture_impact": [
    "All scoring must operate on structured fields — no free-text-only parsing",
    "The 23 redrob_signals are first-class scoring inputs, not secondary",
    "skill duration_months enables genuine depth vs keyword-stuffing detection",
    "education tier enables prestige scoring without raw institution lookup"
  ],
  "implementation_impact": [
    "Parse candidates.jsonl line-by-line (not as a JSON array)",
    "Handle -1 sentinel values for github_activity_score and offer_acceptance_rate",
    "Compute weighted skill scores using proficiency + duration_months + endorsements",
    "Normalize all numeric signals before scoring"
  ]
}
```

### candidates.jsonl
```json
{
  "file_name": "candidates.jsonl",
  "purpose": "The full 100,000-candidate pool. One JSON object per line. This is the input to the ranker.",
  "important_findings": [
    "100,000 candidates, one per line",
    "IDs run CAND_0000001 through CAND_0100000 (likely, with some honeypots mixed in)",
    "Profiles are synthetic but structurally realistic — names, career histories, skill sets",
    "The dataset is intentionally noisy: marketing managers with AI keyword skills, accountants, operations managers, etc.",
    "Many non-ML candidates appear with AI skill keywords (keyword stuffers / traps)",
    "~80 honeypot candidates have logically impossible profiles"
  ],
  "architecture_impact": [
    "The ranker MUST differentiate between true ML/AI engineers and keyword stuffers",
    "Title + career history is more reliable than skills list alone",
    "Must process 100K candidates efficiently within 5-minute CPU budget",
    "Cannot use LLM API calls per candidate — must be rule-based or pre-computed"
  ],
  "implementation_impact": [
    "Load with: gzip.open for .jsonl.gz or open() for .jsonl",
    "Build feature vectors for all 100K candidates in memory or via streaming",
    "Score all candidates, sort descending, take top 100",
    "Runtime constraint: entire pipeline must complete in <5 minutes on CPU"
  ]
}
```

### sample_candidates.json
```json
{
  "file_name": "sample_candidates.json",
  "purpose": "6 sample candidates showing full schema structure with realistic data. Use for development and testing.",
  "important_findings": [
    "CAND_0000001: Backend Engineer (Ira Vora) — data engineering background, transitioning to ML. Has Spark/Airflow/Kafka skills + some ML skills (NLP, Fine-tuning LLMs, Milvus). Tier-3 education. Low github score (9.2).",
    "CAND_0000002: Operations Manager (Saanvi Sethi) — 12.5 years, non-ML background. Has some AI-adjacent keywords but no real depth. No github. Last active 6 months ago.",
    "CAND_0000003: Customer Support (Yash Agarwal) — 1.1 years, essentially irrelevant. Low profile completeness (31.9). Not open to work.",
    "CAND_0000004: Marketing Manager (Anil Bose) — 3.8 years. Non-ML. Has Airflow skill but mostly business/marketing.",
    "CAND_0000005: Accountant (Aisha Sethi) — 11 years, accounting/HR background. Has Image Classification skill (advanced) but it's a stuffer.",
    "CAND_0000006: Business Analyst (Rajesh Desai) — 6 years, consulting background (pure services companies).",
    "Skills with high endorsements but low duration_months are keyword stuffers",
    "Candidates 2-6 are exactly the 'noise' the ranker must reject"
  ],
  "architecture_impact": [
    "The noise ratio is very high — most of 100K are non-fits",
    "Must not rank by skill presence alone — must use career_history title + description",
    "Behavioral signals (last_active_date, recruiter_response_rate) are critical differentiators"
  ],
  "implementation_impact": [
    "Use as dev/test sample before running against full 100K",
    "Validate that your ranker correctly deprioritizes CAND_0000002 through CAND_0000006"
  ]
}
```

### sample_submission.csv
```json
{
  "file_name": "sample_submission.csv",
  "purpose": "Format reference for the submission CSV. NOT a high-quality ranking — only shows the required format.",
  "important_findings": [
    "4 columns exactly: candidate_id, rank, score, reasoning",
    "Ranks 1-100, each appearing exactly once",
    "Scores are floats, non-increasing (rank 1 has highest score)",
    "Score at rank 1 = 0.9920, rank 100 = 0.2000 in the sample",
    "Score step in sample = 0.008 per rank (this is artificial — your scores should reflect actual model output)",
    "Reasoning is 1-2 sentences: title + years + AI skill count + response rate",
    "The sample reasoning format: '{Title} with {N} yrs; {M} AI core skills; response rate {R}'",
    "Notable: sample ranks non-ML titles (HR Manager, Marketing Manager, Accountant) highly — this is NOT a good ranking, just a format example",
    "Candidate IDs in sample are from the real 100K pool (CAND_0004989, etc.)"
  ],
  "architecture_impact": [
    "score column must be a float, non-increasing — enforce in output generation",
    "reasoning must be candidate-specific — identical strings are penalized at Stage 4",
    "score range is 0.0 to 1.0 (implied by sample)"
  ],
  "implementation_impact": [
    "Output exactly 4 columns in this order: candidate_id,rank,score,reasoning",
    "No trailing spaces in reasoning that could cause column mismatch",
    "Use csv.writer or equivalent to handle commas in reasoning",
    "Validate with validate_submission.py before submitting"
  ]
}
```

### submission_spec.docx
```json
{
  "file_name": "submission_spec.docx",
  "purpose": "Complete rules for submission format, compute constraints, evaluation metrics, and the multi-stage evaluation pipeline.",
  "important_findings": [
    "CRITICAL: No LLM API calls during ranking step — ranking must be CPU-only, offline, <5 min",
    "Evaluation: Final composite = 0.50 × NDCG@10 + 0.30 × NDCG@50 + 0.15 × MAP + 0.05 × P@10",
    "Tiebreak: Higher P@5 wins, then P@10, then earlier submission timestamp",
    "3 submissions maximum — last valid submission counts",
    "Honeypot filter at Stage 3: >10% honeypots in top 100 = disqualification",
    "Stage 4 manual review: 10 random rows sampled, reasoning quality evaluated",
    "Reasoning penalized for: empty, all-identical, templated, hallucinated skills, contradicts rank",
    "Each candidate_id in submission must exist in candidates.jsonl",
    "Tie-breaking for equal scores: candidate_id ascending",
    "The validator checks: score non-increasing, unique ranks 1-100, unique candidate IDs, CAND pattern"
  ],
  "architecture_impact": [
    "Architecture CANNOT rely on LLM per-candidate at ranking time",
    "Pre-computation of embeddings/features is allowed (outside the 5-min window)",
    "Ranking step must be a fast scoring function over pre-computed features",
    "The top-10 candidates are 5x more important than ranks 11-50 (NDCG@10 = 50% of score)"
  ],
  "implementation_impact": [
    "Pre-compute: skill embeddings, career history embeddings, feature vectors — offline",
    "Ranking step: weighted scoring function over pre-computed features + redrob_signals",
    "Output validation: run validate_submission.py before every submission",
    "Reasoning must be generated per-candidate and reference actual profile data"
  ]
}
```

### validate_submission.py
```json
{
  "file_name": "validate_submission.py",
  "purpose": "Official submission validator. Run locally before submitting.",
  "important_findings": [
    "Header must be EXACTLY: candidate_id,rank,score,reasoning",
    "Exactly 100 data rows required (rows 2-101)",
    "candidate_id must match pattern: CAND_[0-9]{7}",
    "No duplicate candidate_ids",
    "No duplicate ranks",
    "All ranks 1-100 must appear exactly once",
    "Scores must be non-increasing by rank (s[rank_i] >= s[rank_{i+1}])",
    "Tie-break rule: equal scores require candidate_id ascending",
    "File must be UTF-8 encoded, .csv extension",
    "Filename must be participant_id.csv"
  ],
  "architecture_impact": [
    "Score must be computed as a continuous float — not just rank order",
    "If scoring produces ties, secondary sort by candidate_id ascending"
  ],
  "implementation_impact": [
    "Run: python validate_submission.py your_team_id.csv",
    "Generate scores as normalized floats in [0, 1]",
    "Sort by score descending, then candidate_id ascending for tie-breaking",
    "Assign ranks 1..100 sequentially after sorting"
  ]
}
```

### submission_metadata_template.yaml
```json
{
  "file_name": "submission_metadata_template.yaml",
  "purpose": "Template for team metadata required at submission time.",
  "important_findings": [
    "Sandbox link is REQUIRED — must be a hosted environment that can run the ranker",
    "reproduce_command must be a single command that produces submission.csv from candidates.jsonl",
    "compute.has_network_during_ranking must be false",
    "compute.uses_gpu_for_inference must be false",
    "AI tools declaration is required but transparent — not penalized",
    "github_repo is required"
  ],
  "architecture_impact": [
    "Must build a Streamlit or HuggingFace Spaces demo for the sandbox",
    "The reproduce_command must be a single Python script: python rank.py --candidates ... --out ..."
  ],
  "implementation_impact": [
    "Create rank.py as the main entry point for ranking",
    "Stage 8 dashboard doubles as the sandbox demo",
    "Document pre-computation steps separately from the ranking step"
  ]
}
```

---

## Candidate Schema Summary

### Top-Level Structure
```
candidate_id         CAND_XXXXXXX (7-digit)
profile              Object — static identity and current role
career_history       Array[1-10] — work history entries
education            Array[0-5] — education entries
skills               Array — skills with proficiency + endorsements + duration
certifications       Array (optional) — certs with name, issuer, year
languages            Array (optional) — languages with proficiency
redrob_signals       Object — 23 behavioral platform signals
```

### Profile Fields (All Used for Ranking)
| Field | Type | Ranking Use |
|---|---|---|
| headline | string | Weak title signal — often job-seeker crafted |
| summary | string | Semantic content — evidence of ML thinking |
| years_of_experience | float | Experience range filter (5-9 yrs ideal) |
| current_title | string | PRIMARY title signal |
| current_company | string | Company type signal (product vs services) |
| current_company_size | enum | Startup/scale-up preference |
| current_industry | string | Domain relevance |
| location / country | string | India preferred; Pune/Noida/Hyderabad/Mumbai/Delhi NCR |

### Career History Fields
| Field | Type | Ranking Use |
|---|---|---|
| title | string | Role progression signal |
| company | string | Product company vs services company |
| industry | string | Domain relevance per role |
| company_size | enum | Company stage signal |
| duration_months | int | Tenure stability signal |
| description | string | Evidence of ML/retrieval/ranking work |
| start_date / end_date | date | Chronology, recency |
| is_current | bool | Current role identification |

### Skills Fields
| Field | Type | Ranking Use |
|---|---|---|
| name | string | Core AI skill matching |
| proficiency | enum | beginner/intermediate/advanced/expert |
| endorsements | int | Social validation — trust multiplier |
| duration_months | int | CRITICAL: detects keyword stuffers (high skill + 0 duration = stuffer) |

### Education Fields
| Field | Type | Ranking Use |
|---|---|---|
| tier | enum | tier_1 > tier_2 > tier_3 > tier_4 > unknown |
| degree | string | CS/Engineering/ML preferred |
| field_of_study | string | Relevance to ML/AI |
| end_year | int | Recency |

### Redrob Signals (All 23)
| # | Signal | Range | Ranking Weight |
|---|---|---|---|
| 1 | profile_completeness_score | 0-100 | Medium — incomplete profiles are risky |
| 2 | signup_date | date | Low — recency of platform join |
| 3 | last_active_date | date | HIGH — inactive candidates are unavailable |
| 4 | open_to_work_flag | bool | HIGH — direct availability signal |
| 5 | profile_views_received_30d | int≥0 | Medium — market desirability |
| 6 | applications_submitted_30d | int≥0 | Medium — active job seeker signal |
| 7 | recruiter_response_rate | 0.0-1.0 | HIGH — reachability |
| 8 | avg_response_time_hours | float≥0 | Medium — lower is better |
| 9 | skill_assessment_scores | dict | HIGH — verified skill depth |
| 10 | connection_count | int≥0 | Low — network size |
| 11 | endorsements_received | int≥0 | Low — social proof |
| 12 | notice_period_days | 0-180 | HIGH — JD wants sub-30 days |
| 13 | expected_salary_range_inr_lpa | {min, max} | Medium — budget fit |
| 14 | preferred_work_mode | enum | Medium — Hybrid/onsite preferred by JD |
| 15 | willing_to_relocate | bool | Medium — Pune/Noida preferred |
| 16 | github_activity_score | -1 to 100 | HIGH — OSS/coding activity |
| 17 | search_appearance_30d | int≥0 | Low — platform visibility |
| 18 | saved_by_recruiters_30d | int≥0 | Medium — other-recruiter interest |
| 19 | interview_completion_rate | 0.0-1.0 | Medium — reliability |
| 20 | offer_acceptance_rate | -1 to 1.0 | Medium — intent signal (-1 = no history) |
| 21 | verified_email | bool | Low — trust signal |
| 22 | verified_phone | bool | Low — trust signal |
| 23 | linkedin_connected | bool | Low — profile completeness |

---

## Job Description Structure

**Role:** Senior AI Engineer — Founding Team  
**Company:** Redrob AI (Series A, AI-native talent intelligence)  
**Location:** Pune/Noida, India (Hybrid) — open to Hyderabad, Mumbai, Delhi NCR  
**Experience:** 5-9 years (flexible; judgment matters more than exact years)  

### Must-Have Skills (Hard Requirements)
1. Production embeddings-based retrieval (sentence-transformers, OpenAI embeddings, BGE, E5 — any)
2. Production vector database / hybrid search (Pinecone, Weaviate, Qdrant, Milvus, FAISS, ES/OpenSearch)
3. Strong Python
4. Evaluation frameworks for ranking systems (NDCG, MRR, MAP, offline-to-online, A/B testing)

### Nice-to-Have Skills
- LLM fine-tuning (LoRA, QLoRA, PEFT)
- Learning-to-rank (XGBoost-based or neural)
- HR-tech / recruiting / marketplace product experience
- Distributed systems / large-scale inference
- Open-source contributions in AI/ML

### Explicit Disqualifiers
- Pure research background (no production deployment)
- AI experience = only LangChain + OpenAI in last 12 months (without pre-LLM ML production experience)
- Senior engineers who haven't written production code in 18 months
- Entire career at TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini/Accenture — pure services
- Primary expertise in CV/speech/robotics without significant NLP/IR
- 5+ years of only closed-source proprietary work without external validation

### Implicit Requirements (Read Between the Lines)
- Shipped at least one end-to-end ranking/search/recommendation system at meaningful scale
- Product company experience (at least some — not exclusively services)
- Active on job market / Redrob platform
- Short notice period (sub-30 days ideal; buyable up to 30 days)
- Located in India (Pune/Noida/Hyderabad/Mumbai/Delhi NCR) or willing to relocate
- Writes clearly (async-first company)
- Comfortable with ambiguity and fast iteration
- 6-8 years total, 4-5 in applied ML/AI at product companies (ideal band)

---

## Submission Requirements

### File Format
- Filename: `{participant_id}.csv`
- Encoding: UTF-8
- Columns (exact order): `candidate_id,rank,score,reasoning`
- Rows: Exactly 100 data rows + 1 header row = 101 total rows

### Validation Rules (from validate_submission.py)
1. Header = exactly `candidate_id,rank,score,reasoning`
2. Exactly 100 non-empty data rows
3. `candidate_id` matches `CAND_[0-9]{7}`
4. No duplicate candidate_ids
5. No duplicate ranks
6. All ranks 1-100 present exactly once
7. Scores non-increasing (score[rank_i] ≥ score[rank_{i+1}])
8. Equal scores: candidate_id ascending (tie-break)

### Scoring Formula
```
composite = 0.50 × NDCG@10 + 0.30 × NDCG@50 + 0.15 × MAP + 0.05 × P@10
```
NDCG@10 is 5x more important than P@10 — **the top 10 slots matter most.**

---

## Redrob Signals Summary

The 23 behavioral signals function as a **multiplier layer** on top of skill/career matching. The JD explicitly states: "A perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% response rate is, for hiring purposes, not actually available."

### High-Impact Signals (use as primary multipliers)
- `open_to_work_flag` — false = candidate is not signaling availability
- `last_active_date` — >90 days inactive = strong down-weight
- `recruiter_response_rate` — <0.2 = not reachable
- `notice_period_days` — >90 days = hard constraint for this JD
- `github_activity_score` — >0 = technical credibility signal; -1 = neutral
- `skill_assessment_scores` — verified skills override self-reported proficiency

### Medium-Impact Signals
- `profile_completeness_score` — <50 is a red flag
- `interview_completion_rate` — <0.5 = reliability concern
- `avg_response_time_hours` — >168 hours (1 week) = very slow
- `applications_submitted_30d` — > 0 = actively searching
- `willing_to_relocate` + `preferred_work_mode` — location/mode fit

### Low-Impact Signals (use as tie-breakers)
- `verified_email`, `verified_phone`, `linkedin_connected`
- `connection_count`, `endorsements_received`
- `saved_by_recruiters_30d`, `search_appearance_30d`

---

## Evaluation Strategy

### Why NDCG@10 dominates (50% weight)
The top 10 candidates are the ones a recruiter will actually look at first. Getting the top 3-5 right matters enormously. Your system should optimize heavily for precision at the top.

### Architecture Recommendation
**Do NOT use an LLM per candidate at ranking time.** 100,000 candidates × any LLM latency = impossible in 5 minutes on CPU.

**Recommended approach:**
1. Pre-compute feature vectors for all 100K candidates (offline, can take longer)
2. At ranking time: apply a fast weighted scoring formula over pre-computed features + live redrob_signals
3. Sort all 100K by score descending, take top 100
4. Generate reasoning for the top 100 only (this is where LLM can help — only 100 rows)

### The "Trap" Warning (from JD)
> "The right answer is not 'find candidates whose skills section contains the most AI keywords.'"

A candidate who has "RAG", "FAISS", "LLM" listed as skills but has been an "HR Manager" for 10 years is NOT a fit. Career history title and description must dominate over skills keywords.

### Honeypot Detection
Look for logical impossibilities:
- `years_of_experience` at a company > company founding age
- "expert" proficiency on 10+ skills with `duration_months = 0`
- Career history that doesn't add up temporally
- Profile descriptions that are clearly copy-pasted boilerplate

---

## Architecture Recommendations

### Phase 1: Offline Pre-computation
1. Load all 100K candidates from candidates.jsonl
2. For each candidate, compute:
   - `ai_core_skill_count`: count of skills from the JD's required AI skill list (embeddings, vector DB, Python, ranking eval, etc.)
   - `ai_skill_depth_score`: weighted by proficiency × endorsements × duration_months
   - `career_title_score`: current_title relevance to ML/AI/data engineering
   - `career_history_score`: presence of ML/retrieval/ranking in job descriptions
   - `services_company_penalty`: ratio of career at TCS/Infosys/Wipro/etc.
   - `education_tier_score`: based on institution tier field
   - `experience_fit_score`: years_of_experience in [5,9] band
3. Save feature vectors to disk

### Phase 2: Ranking (must complete in <5 minutes)
1. Load pre-computed feature vectors
2. Compute behavioral_multiplier from redrob_signals
3. Compute final_score = skill_career_score × behavioral_multiplier × location_fit
4. Sort descending, take top 100
5. Output CSV

### Phase 3: Reasoning Generation (top 100 only)
- For top 100 candidates, generate a 1-2 sentence reasoning
- Reference actual profile data (title, years, specific skills, response rate)
- This is where a light LLM call is appropriate (only 100 rows)
