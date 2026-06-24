# SKILLS.md — Technical Implementation Reference

## Project Structure

```
ai-recruiter/
├── AGENT.md                    # Agent identity, reasoning contracts, scoring formulas
├── CLAUDE.md                   # LLM prompting strategy and all system prompts
├── SKILLS.md                   # This file — technical reference
├── README.md                   # Project overview and setup instructions
│
├── main.py                     # Entry point — orchestrates the full 8-stage pipeline
├── config.py                   # All configuration: model names, weights, thresholds
├── requirements.txt            # Pinned dependencies
│
├── agents/
│   ├── __init__.py
│   ├── role_understanding.py   # Stage 1: JD → structured requirement schema
│   ├── candidate_understanding.py  # Stage 2: raw profile → normalized candidate schema
│   ├── evidence_extraction.py  # Stage 3: candidate schema → evidence item list
│   ├── graph_builder.py        # Stage 4: evidence items → knowledge graph (NetworkX)
│   ├── hybrid_retrieval.py     # Stage 5: FAISS vector search + graph traversal
│   ├── scoring_engine.py       # Stage 6: LLM scoring — fit, impact, potential, risk
│   ├── ranking.py              # Stage 7a: composite score → ranked list
│   ├── dark_horse.py           # Stage 7b: dark horse detection + transferable skills map
│   └── interview_questions.py  # Generates 3 tailored questions per candidate
│
├── dashboard/
│   ├── app.py                  # Streamlit dashboard entry point (Stage 8)
│   ├── components/
│   │   ├── leaderboard.py      # Ranked candidate table with score breakdown
│   │   ├── candidate_card.py   # Detailed view: flags, gaps, questions, rationale
│   │   ├── dark_horse_panel.py # Dark horse spotlight section
│   │   └── export.py           # PDF/CSV export of ranked results
│   └── styles/
│       └── theme.css           # Custom Streamlit CSS
│
├── data/
│   ├── sample_jd.json          # Sample job description for testing
│   ├── sample_candidates/      # Sample candidate profiles (JSON)
│   └── outputs/                # Pipeline outputs stored here (gitignored)
│
├── embeddings/
│   ├── __init__.py
│   ├── embedder.py             # Unified embedding interface (OpenAI or local)
│   └── faiss_index.py          # FAISS index build, save, load, query
│
├── graph/
│   ├── __init__.py
│   ├── knowledge_graph.py      # NetworkX graph schema and operations
│   └── graph_queries.py        # Graph traversal queries for Stage 5
│
├── llm/
│   ├── __init__.py
│   ├── client.py               # LLM client wrapper (Anthropic + OpenAI)
│   ├── prompts.py              # All prompt builders — pulls system prompts from CLAUDE.md logic
│   └── retry.py                # JSON parse retry logic and error handling
│
├── schemas/
│   ├── __init__.py
│   ├── jd_schema.py            # Pydantic model: JD requirement schema
│   ├── candidate_schema.py     # Pydantic model: normalized candidate profile
│   ├── evidence_schema.py      # Pydantic model: evidence item
│   ├── scoring_schema.py       # Pydantic model: scoring output
│   └── output_schema.py        # Pydantic model: final ranked output per candidate
│
└── tests/
    ├── test_evidence_extraction.py
    ├── test_scoring_engine.py
    ├── test_dark_horse.py
    └── test_pipeline_integration.py
```

---

## Core Dependencies

| Package | Version | Purpose |
|---|---|---|
| `anthropic` | `>=0.25.0` | Claude API client |
| `openai` | `>=1.30.0` | GPT-4o + embeddings client |
| `faiss-cpu` | `>=1.8.0` | Vector similarity search |
| `networkx` | `>=3.3` | Knowledge graph (GraphRAG) |
| `sentence-transformers` | `>=3.0.0` | Local embedding fallback |
| `pydantic` | `>=2.7.0` | Schema validation for all pipeline I/O |
| `streamlit` | `>=1.35.0` | Recruiter Copilot Dashboard (Stage 8) |
| `numpy` | `>=1.26.0` | Numerical operations for scoring |
| `pandas` | `>=2.2.0` | Data manipulation for ranking output |
| `python-dotenv` | `>=1.0.0` | Environment variable management |
| `tenacity` | `>=8.3.0` | Retry logic for LLM calls |
| `reportlab` | `>=4.2.0` | PDF export of ranked results |
| `pytest` | `>=8.2.0` | Test framework |

---

## Data Schemas

### Candidate Input Schema (raw)
```json
{
  "candidate_id": "string (required, unique)",
  "candidate_name": "string (required)",
  "raw_profile": "string (required — full resume text or structured profile text)",
  "source": "<resume | linkedin | manual>"
}
```

### Job Description Input Schema (raw)
```json
{
  "jd_id": "string (required, unique)",
  "raw_jd": "string (required — full job description text)",
  "company_name": "string (optional)",
  "role_title": "string (optional — pre-fill, will be validated by Stage 1)"
}
```

### Evidence Item Schema
```json
{
  "claim": "string",
  "evidence_type": "technical | impact | leadership | learning | behavioral",
  "confidence": "high | medium | low",
  "source_text": "string or null",
  "quantified": "boolean"
}
```

### Final Ranked Output Schema (one object per candidate)
```json
{
  "rank": "int",
  "candidate_id": "string",
  "candidate_name": "string",
  "composite_score": "float (0-100)",
  "fit_score": "int (0-100)",
  "impact_score": "int (0-100)",
  "potential_score": "int (0-100)",
  "risk_score": "int (0-100)",
  "confidence_level": "high | medium | low",
  "green_flags": ["string"],
  "yellow_flags": ["string"],
  "skill_gaps": ["string"],
  "dark_horse": "boolean",
  "dark_horse_reason": "string or null",
  "transferable_skills_map": [
    {
      "candidate_skill": "string",
      "maps_to_jd_requirement": "string",
      "mapping_reasoning": "string"
    }
  ],
  "interview_questions": ["string", "string", "string"],
  "llm_rationale": "string (max 100 words)"
}
```

---

## Scoring Implementation Rules

### Composite Score Formula
```python
composite_score = (
    (fit_score    * 0.35) +
    (impact_score * 0.30) +
    (potential_score * 0.20) +
    ((100 - risk_score) * 0.15)
)
```

### Potential Score Formula
```python
# Normalize each sub-factor to 0-100 before weighting
potential_score = (
    (career_velocity_normalized * 0.4) +
    (complexity_growth * 0.3) +
    (self_learning_signals_normalized * 0.3)
)
```

### Quantified Evidence Weight
```python
# Applied during evidence aggregation in Stage 3 / Stage 6
effective_weight = 1.5 if evidence_item["quantified"] else 1.0
```

### Impact Score Cap
```python
# Enforced in scoring_engine.py
if quantified_impact_count == 0:
    impact_score = min(impact_score, 40)
```

---

## GraphRAG Knowledge Graph Schema

### Node Types
| Node Type | Attributes | Example |
|---|---|---|
| `Candidate` | id, name, career_arc, velocity | `{id: "C001", name: "Alice"}` |
| `Skill` | name, category, normalized_name | `{name: "PyTorch", category: "technical"}` |
| `Company` | name, domain, size_signal | `{name: "Stripe", domain: "fintech"}` |
| `Achievement` | claim, quantified, impact_value | `{claim: "23% retention increase"}` |
| `Role` | title, seniority, company_id | `{title: "ML Engineer", seniority: "senior"}` |

### Edge Types
| Edge | From | To | Attributes |
|---|---|---|---|
| `HAS_SKILL` | Candidate | Skill | confidence, years |
| `WORKED_AT` | Candidate | Company | start, end, role_id |
| `ACHIEVED` | Candidate | Achievement | role_id, year |
| `REQUIRES` | JD | Skill | importance |
| `SIMILAR_TO` | Skill | Skill | similarity_score |

---

## FAISS Index Configuration

```python
# embeddings/faiss_index.py
EMBEDDING_DIM = 1536          # text-embedding-3-small
INDEX_TYPE = "IndexFlatIP"    # Inner product (cosine after normalization)
TOP_K_RETRIEVAL = 25          # Retrieve top 25 before re-ranking
# Dark horse threshold: candidates ranked > 15 in FAISS are dark horse candidates
DARK_HORSE_RANK_THRESHOLD = 15
```

---

## LLM Client Configuration

```python
# config.py
PRIMARY_MODEL = "claude-3-5-sonnet-20241022"   # or "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"
TEMPERATURE_SCORING = 0.0
TEMPERATURE_INTERVIEW_QS = 0.3
TEMPERATURE_RATIONALE = 0.0
MAX_RETRIES = 2
JSON_RETRY_SUFFIX = "Respond only with valid JSON. No markdown. No explanation."
```

---

## Environment Variables Required

```
ANTHROPIC_API_KEY=       # Required if using Claude
OPENAI_API_KEY=          # Required if using GPT-4o or OpenAI embeddings
LLM_PROVIDER=            # "anthropic" | "openai"
EMBEDDING_PROVIDER=      # "openai" | "local"
LOG_LEVEL=               # "DEBUG" | "INFO" | "WARNING"
```

---

## Pipeline Execution Contract

Each stage function signature follows this pattern:

```python
def run_stage_N(input: StageNInput) -> StageNOutput:
    """
    Input contract:  StageNInput (Pydantic model)
    Output contract: StageNOutput (Pydantic model)
    Never raises silently — all errors must propagate with stage context.
    Never mutates input.
    """
```

Stages are orchestrated sequentially in `main.py`. No stage may be skipped. No stage output may be passed to a non-adjacent stage without going through the intermediate stage first.

---

## Testing Strategy

- **Unit tests**: Each agent module tested in isolation with mocked LLM responses.
- **Schema tests**: Every Pydantic model tested with valid and invalid inputs.
- **Integration test**: Full pipeline run on `sample_jd.json` + 5 sample candidates, asserts output schema compliance.
- **Anti-hallucination test**: Feed a candidate profile with zero quantified claims, assert `impact_score <= 40`.
- **Dark horse test**: Feed a candidate with `vector_rank=20`, `impact_score=80`, `fit_score=60`, assert `dark_horse=true`.
