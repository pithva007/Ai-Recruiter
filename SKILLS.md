# SKILLS.md — Technical Implementation Reference

## Project Structure

```
ai-recruiter/
├── AGENT.md
├── CLAUDE.md
├── SKILLS.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── job_description.txt
│   │   └── candidates.csv          (or candidates.json)
│   └── processed/
│       ├── jd_schema.json
│       ├── candidate_profiles/
│       │   └── {candidate_id}_profile.json
│       ├── evidence/
│       │   └── {candidate_id}_evidence.json
│       └── scores/
│           └── {candidate_id}_scores.json
├── src/
│   ├── stage1_role_agent.py
│   ├── stage2_candidate_agent.py
│   ├── stage3_evidence_extraction.py
│   ├── stage4_graph_builder.py
│   ├── stage5_hybrid_retrieval.py
│   ├── stage6_scoring_engine.py
│   ├── stage7_ranking.py
│   ├── stage7b_dark_horse.py
│   └── stage8_dashboard.py
├── utils/
│   ├── llm_client.py
│   ├── embedding_client.py
│   └── json_validator.py
├── outputs/
│   ├── ranked_candidates.csv
│   └── shortlist_report.pdf
└── app.py                          (Streamlit entry point)
```

---

## Required Libraries

```
# requirements.txt
openai>=1.0.0
anthropic>=0.20.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4
networkx>=3.0
pandas>=2.0.0
numpy>=1.24.0
streamlit>=1.30.0
plotly>=5.18.0
pydantic>=2.0.0
python-dotenv>=1.0.0
tenacity>=8.2.0
reportlab>=4.0.0
```

---

## LLM Client Pattern (use this exact pattern in every stage)

```python
# utils/llm_client.py
import json
from tenacity import retry, stop_after_attempt, wait_exponential
import anthropic

client = anthropic.Anthropic()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_llm(system_prompt: str, user_content: str, temperature: float = 0.0) -> dict:
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4096,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}]
    )
    raw = response.content[0].text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        retry_response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4096,
            temperature=0.0,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "Your response was not valid JSON. Respond only with valid JSON. No markdown. No explanation. No code blocks. Just the raw JSON object."}
            ]
        )
        return json.loads(retry_response.content[0].text.strip())
```

---

## Embedding Client Pattern

```python
# utils/embedding_client.py
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-mpnet-base-v2')

def get_embedding(text: str) -> np.ndarray:
    return model.encode(text, normalize_embeddings=True)

def get_batch_embeddings(texts: list[str]) -> np.ndarray:
    return model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True)
```

---

## JSON Validator Pattern

```python
# utils/json_validator.py
from pydantic import BaseModel, validator
from typing import Optional, List

class EvidenceItem(BaseModel):
    claim: str
    evidence_type: str
    confidence: str
    source_text: Optional[str]
    quantified: bool

class CandidateScore(BaseModel):
    candidate_id: str
    fit_score: int
    impact_score: int
    potential_score: int
    risk_score: int
    composite_score: Optional[float] = None

    @validator('fit_score', 'impact_score', 'potential_score', 'risk_score')
    def score_range(cls, v):
        assert 0 <= v <= 100, f"Score must be 0-100, got {v}"
        return v

    def compute_composite(self):
        self.composite_score = round(
            (self.fit_score    * 0.35) +
            (self.impact_score * 0.30) +
            (self.potential_score * 0.20) +
            ((100 - self.risk_score) * 0.15), 2
        )
        return self.composite_score
```

---

## Graph Builder Pattern

```python
# Stage 4 uses networkx.
# Node types: CANDIDATE, SKILL, TOOL, DOMAIN, COMPANY_TYPE, IMPACT_KEYWORD
# Edge types: HAS_SKILL, USED_TOOL, WORKS_IN_DOMAIN, WORKED_AT_TYPE, ACHIEVED

import networkx as nx

def build_candidate_graph(candidate_id: str, evidence_items: list) -> nx.Graph:
    G = nx.Graph()
    G.add_node(candidate_id, node_type="CANDIDATE")
    for item in evidence_items:
        entity = item["claim"]
        G.add_node(entity, node_type=item["evidence_type"].upper())
        G.add_edge(candidate_id, entity, confidence=item["confidence"])
    return G

def find_graph_matches(jd_entities: list, candidate_graphs: dict, top_k: int = 20) -> list:
    scores = []
    for cid, G in candidate_graphs.items():
        shared = sum(1 for e in jd_entities if G.has_node(e))
        neighbor_overlap = sum(
            1 for e in jd_entities
            for n in G.neighbors(cid)
            if e.lower() in n.lower()
        )
        graph_score = (shared * 2 + neighbor_overlap) / (len(jd_entities) + 1)
        scores.append((cid, graph_score))
    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
```

---

## FAISS Index Pattern

```python
import faiss
import numpy as np

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index

def search_faiss(index, query_embedding: np.ndarray, top_k: int = 20):
    scores, indices = index.search(
        query_embedding.reshape(1, -1).astype(np.float32), top_k
    )
    return scores[0], indices[0]
```

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

### Composite Score Formula (canonical — use only this)
```python
composite_score = (
    (fit_score       * 0.35) +
    (impact_score    * 0.30) +
    (potential_score * 0.20) +
    ((100 - risk_score) * 0.15)
)
```

### Potential Score Formula
```python
# Normalize each sub-factor to 0-100 before weighting
potential_score = (
    (career_velocity_normalized    * 0.4) +
    (complexity_growth             * 0.3) +
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
| Node Type      | Attributes                       | Example                                  |
|----------------|----------------------------------|------------------------------------------|
| `Candidate`    | id, name, career_arc, velocity   | `{id: "C001", name: "Alice"}`            |
| `Skill`        | name, category, normalized_name  | `{name: "PyTorch", category: "technical"}`|
| `Company`      | name, domain, size_signal        | `{name: "Stripe", domain: "fintech"}`    |
| `Achievement`  | claim, quantified, impact_value  | `{claim: "23% retention increase"}`      |
| `Role`         | title, seniority, company_id     | `{title: "ML Engineer", seniority: "senior"}` |

### Edge Types
| Edge             | From      | To          | Attributes             |
|------------------|-----------|-------------|------------------------|
| `HAS_SKILL`      | Candidate | Skill       | confidence, years      |
| `WORKED_AT`      | Candidate | Company     | start, end, role_id    |
| `ACHIEVED`       | Candidate | Achievement | role_id, year          |
| `REQUIRES`       | JD        | Skill       | importance             |
| `SIMILAR_TO`     | Skill     | Skill       | similarity_score       |

---

## FAISS Index Configuration

```python
# embeddings/faiss_index.py
EMBEDDING_DIM = 768               # all-mpnet-base-v2
INDEX_TYPE = "IndexFlatIP"        # Inner product (cosine after normalization)
TOP_K_RETRIEVAL = 25              # Retrieve top 25 before re-ranking
DARK_HORSE_RANK_THRESHOLD = 15    # Candidates ranked > 15 are dark horse candidates
```

---

## LLM Client Configuration

```python
# config.py
PRIMARY_MODEL       = "claude-3-5-sonnet-20241022"
EMBEDDING_MODEL     = "all-mpnet-base-v2"
TEMPERATURE_SCORING      = 0.0
TEMPERATURE_INTERVIEW_QS = 0.3
TEMPERATURE_RATIONALE    = 0.0
MAX_RETRIES              = 3
JSON_RETRY_SUFFIX        = "Respond only with valid JSON. No markdown. No explanation."
```

---

## Environment Variables Required

```
ANTHROPIC_API_KEY=      # Required — Claude API key
LOG_LEVEL=              # "DEBUG" | "INFO" | "WARNING"
```

---

## Hiring Decision Simulator Weight Schema
The dashboard allows the recruiter to override weights. Use this schema:
```json
{
  "fit_weight":       0.35,
  "impact_weight":    0.30,
  "potential_weight": 0.20,
  "risk_weight":      0.15
}
```
- fit_weight range: 0.10 to 0.60
- impact_weight range: 0.10 to 0.50
- potential_weight range: 0.05 to 0.40
- risk_weight range: 0.05 to 0.30

Weights must always sum to 1.0. Normalize after each slider change.

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

Stages are orchestrated sequentially in `app.py`. No stage may be skipped. No stage output may be passed to a non-adjacent stage without going through the intermediate stage first.

---

## Testing Strategy

- **Unit tests**: Each agent module tested in isolation with mocked LLM responses.
- **Schema tests**: Every Pydantic model tested with valid and invalid inputs.
- **Integration test**: Full pipeline run on `job_description.txt` + sample candidates, asserts output schema compliance.
- **Anti-hallucination test**: Feed a candidate profile with zero quantified claims, assert `impact_score <= 40`.
- **Dark horse test**: Feed a candidate with `vector_rank=20`, `impact_score=80`, `fit_score=60`, assert `dark_horse=true`.
- **Weight normalization test**: Change one slider value, assert all weights still sum to 1.0.
