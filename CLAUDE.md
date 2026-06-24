# CLAUDE.md — LLM Prompting Strategy
# Challenge: Redrob Intelligent Candidate Discovery & Ranking

---

## LLM Use Policy — Read First

**LLM calls are permitted ONLY in two phases:**

1. **Phase A (offline pre-computation):** JD analysis — runs ONCE
2. **Phase C (post-ranking reasoning):** Generate reasoning for top 100 candidates only

**LLM is STRICTLY FORBIDDEN during Phase B (ranking of 100K candidates).**

The ranking step must complete in <5 minutes on CPU with no network access.
100,000 candidates × any LLM latency = impossible in 5 minutes.

---

## Model Selection

- **Primary reasoning model:** `gemini-2.5-flash`
- **Embeddings (optional offline pre-compute):** `sentence-transformers/all-mpnet-base-v2` (local)
- **Temperature:** `0.0` for JD analysis and reasoning. Never use temperature > 0.1 for structured output.
- **Never** use a model that requires a network call during the ranking step.

---

## LLM Client Pattern (use exactly — from utils/llm_client.py)

```python
from utils.llm_client import call_llm

# All LLM calls go through call_llm().
# Returns a parsed dict. Handles JSON retry automatically.
# Raises after 3 tenacity retries on network/rate errors.
result = call_llm(
    system_prompt=SYSTEM_PROMPT,
    user_content=user_message,
    temperature=0.0
)
```

---

## Prompt Engineering Rules

1. Every prompt must state: the model's role, exact output format, and the anti-hallucination constraint.
2. Every prompt must end with: `"If you cannot determine a field from the provided text, output null. Do not invent or infer beyond what is stated."`
3. All structured outputs must be valid JSON. `call_llm()` handles retry on parse failure.
4. Never send more than one candidate per LLM call.
5. For reasoning generation: reference only actual candidate fields — never invent claims.

---

## Challenge Alignment Rules

These rules override all general prompt engineering practices:

- **Rule C1:** No LLM call may process more than one candidate at a time.
- **Rule C2:** The ranking scoring formula is deterministic (rule-based). LLM is never used to assign scores.
- **Rule C3:** Reasoning must reference actual `profile.current_title`, `profile.years_of_experience`, actual skill names from `skills[]`, and actual `redrob_signals.recruiter_response_rate`.
- **Rule C4:** Reasoning must NOT hallucinate skills, companies, or impact numbers not present in the candidate's actual data.
- **Rule C5:** Reasoning for two different candidates must not be identical.

---

## System Prompt: Phase A — JD Analysis Agent

This prompt runs ONCE against the actual JD text to produce a structured requirements document.

```
You are a Senior Technical Recruiter with deep expertise in AI/ML hiring.

Your job is to analyze the provided job description and produce a structured requirements
schema that will be used by an automated ranking system.

Output exactly this JSON structure:

{
  "role_title": "",
  "company": "",
  "seniority_level": "<junior | mid | senior | lead | principal>",
  "experience_range": {"min_years": <int>, "max_years": <int>, "ideal_years": <int>},
  "location": {"primary_cities": [], "country": "", "work_mode": "<hybrid | onsite | remote>"},
  "must_have_skills": [
    {"skill": "", "context": "", "importance": "must_have"}
  ],
  "nice_to_have_skills": [
    {"skill": "", "context": "", "importance": "nice_to_have"}
  ],
  "explicit_disqualifiers": [
    {"pattern": "", "reason": ""}
  ],
  "services_company_names": [],
  "implicit_requirements": [
    {"requirement": "", "reasoning": ""}
  ],
  "culture_signals": [],
  "ideal_candidate_summary": "",
  "jd_trap_warning": ""
}

The jd_trap_warning field should describe any intentional traps or anti-patterns the JD warns about
(e.g., keyword-only matching without career history validation).

Extract the services company blacklist from any explicit company names mentioned as disqualifiers in the JD.

If you cannot determine a field from the JD, output null. Do not invent or infer beyond what is stated.
```

---

## System Prompt: Phase C — Reasoning Generation for Top 100

This prompt generates the `reasoning` column for the final submission CSV.
Run per candidate. References ACTUAL candidate data fields — no hallucination.

```
You are a technical recruiter writing a candidate assessment summary for a Senior AI Engineer role.

The ranking system has already scored and ranked this candidate. Your job is to write a concise,
honest 1-2 sentence reasoning that explains WHY this candidate ranked where they did.

Rules for your reasoning:
1. Reference only information that actually exists in the candidate data provided.
2. Do NOT invent skills, companies, or achievements not in the data.
3. Do NOT write the same reasoning for every candidate — be specific to this individual.
4. Reference at least one of: their actual current_title, their actual years_of_experience,
   their actual relevant skills (by name), their recruiter_response_rate, or their notice_period_days.
5. If the candidate ranks in the top 20, the reasoning should highlight their strongest signal.
6. If the candidate ranks 50-100, the reasoning should acknowledge the limiting factors.
7. Maximum 30 words. No bullet points. Plain sentence format.

Output a single JSON object:
{
  "candidate_id": "<exact candidate_id from input>",
  "reasoning": "<1-2 sentences, max 30 words>"
}

If you cannot write honest, specific reasoning from the provided data, output:
{
  "candidate_id": "<id>",
  "reasoning": "Adjacent background with some relevant skills; included based on behavioral engagement signals."
}

Do not invent or infer beyond what is stated.
```

---

## Prompt: Validation Check (use before submitting)

This is NOT a prompt to send to an LLM. It is a checklist to run before generating the final CSV.

```
Pre-submission validation checklist:
[ ] Exactly 100 data rows in output CSV
[ ] Header is exactly: candidate_id,rank,score,reasoning
[ ] All ranks 1-100 present exactly once, no duplicates
[ ] All candidate_ids match CAND_[0-9]{7}
[ ] All candidate_ids exist in candidates.jsonl
[ ] Scores are non-increasing (score[i] >= score[i+1] for all i)
[ ] No identical reasoning strings across rows
[ ] No reasoning references skills not in the candidate's skills[] list
[ ] run: python India_runs_data_and_ai_challenge/validate_submission.py {file}.csv
[ ] Zero validation errors before submitting
```

---

## Output Validation Rules

Every row in the submission CSV must satisfy:

```python
# Validated by validate_submission.py — these are the actual checks:
assert len(data_rows) == 100
assert header == ['candidate_id', 'rank', 'score', 'reasoning']
assert all(re.match(r'^CAND_[0-9]{7}$', row['candidate_id']) for row in data_rows)
assert len(set(row['candidate_id'] for row in data_rows)) == 100  # no duplicates
assert len(set(row['rank'] for row in data_rows)) == 100          # no duplicate ranks
assert set(row['rank'] for row in data_rows) == set(range(1, 101))  # ranks 1-100
assert all(data_rows[i]['score'] >= data_rows[i+1]['score']
           for i in range(99))  # non-increasing scores
# Tie-break: equal scores → candidate_id ascending
```

---

## Field-Level Confidence Rules

When generating reasoning, apply these confidence levels:

| Field source | Confidence | Use in reasoning |
|---|---|---|
| `redrob_signals.skill_assessment_scores[skill]` | Highest — platform-verified | Yes, cite as "verified [skill] score" |
| `career_history[].description` contains exact text | High | Yes, cite directly |
| `skills[].name` with `duration_months > 6` | High | Yes, cite as demonstrated skill |
| `profile.current_title` | High | Always cite |
| `skills[].name` with `duration_months == 0` | Low — possible stuffer | Omit from reasoning |
| `profile.summary` only (no career corroboration) | Low | Qualify with "self-reported" |
| Inferred from company popularity | Zero — do not use | Never cite |

---

## Reasoning Quality Anti-Patterns (Stage 4 Penalty Triggers)

The challenge manual review samples 10 random rows and penalizes these:

| Anti-pattern | Example (bad) | Fix |
|---|---|---|
| Empty reasoning | `""` | Always write something specific |
| All-identical reasoning | Same sentence for 100 rows | Must be candidate-specific |
| Templated name-only | `"John is a good fit."` | Reference actual skills/experience |
| Hallucinated skills | "Expert in RAG" (not in their skills[]) | Only cite skills that exist |
| Contradicts rank | Top-10 reasoning says "limited ML experience" | Reasoning must match rank direction |
| Too long | > 2 sentences | Keep to 1-2 sentences, max 30 words |

---

## Sample Good Reasoning Strings

```
# Rank 1 (strong fit):
"ML Engineer at product company with 7 years; ships retrieval systems; response rate 0.82, notice period 15 days."

# Rank 5 (strong fit, minor concern):
"6 years applied ML with FAISS and embedding deployment; strong engagement signals; notice period 60 days is the main friction."

# Rank 30 (mid-tier):
"Data engineer with adjacent ML skills (Spark, Airflow, some NLP); career transition in progress; moderate recruiter responsiveness."

# Rank 80 (borderline):
"Marketing background with AI keywords in skills; no ML career history found; included on engagement signals and recent activity."

# Rank 100 (bottom of top 100):
"Adjacent skills only; non-ML career history; included as final filler given experience and platform engagement."
```
