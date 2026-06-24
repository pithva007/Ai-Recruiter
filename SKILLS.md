# SKILLS.md — Technical Implementation Reference
# Challenge: Redrob Intelligent Candidate Discovery & Ranking
# Ground truth: India_runs_data_and_ai_challenge/ files

---

## Project Structure

```
ai-recruiter/
├── AGENT.md                          # Scoring contracts, JD requirements, honeypot detection
├── CLAUDE.md                         # LLM prompts, reasoning generation, validation rules
├── SKILLS.md                         # This file — implementation reference
├── requirements.txt                  # Pinned dependencies
├── .env                              # GEMINI_API_KEY, GEMINI_MODEL
├── submission_metadata.yaml          # Fill before submitting
│
├── India_runs_data_and_ai_challenge/ # Challenge bundle (read-only)
│   ├── candidates.jsonl              # 100K candidate pool — PRIMARY INPUT
│   ├── candidate_schema.json         # Schema reference
│   ├── sample_candidates.json        # 6 sample candidates for dev/test
│   ├── sample_submission.csv         # Format reference only (NOT a good ranking)
│   ├── job_description.docx          # The actual JD
│   ├── redrob_signals_doc.docx       # Signal definitions
│   ├── submission_spec.docx          # Rules and evaluation
│   ├── README.docx                   # Getting started
│   ├── submission_metadata_template.yaml
│   └── validate_submission.py        # Run before every submission
│
├── src/
│   ├── stage1_jd_analysis.py         # Phase A: Parse JD → jd_features.json (LLM, once)
│   ├── precompute.py                 # Phase A: Score all 100K → features.pkl (offline)
│   ├── rank.py                       # Phase B: Load features → top 100 CSV (< 5 min, no net)
│   ├── reason.py                     # Phase C: LLM reasoning for top 100 → final CSV
│   └── stage8_dashboard.py           # Streamlit demo / sandbox
│
├── utils/
│   ├── llm_client.py                 # Gemini client wrapper
│   ├── embedding_client.py           # Local sentence-transformers (offline only)
│   ├── json_validator.py             # Pydantic models for schema validation
│   └── feature_engineering.py       # All scoring functions (deterministic)
│
├── data/
│   ├── raw/
│   │   └── job_description.txt       # Copy of JD text for Stage 1
│   └── processed/
│       ├── jd_features.json          # Stage 1 output
│       └── features.pkl              # Precompute output (all 100K feature vectors)
│
└── outputs/
    ├── ranked_top100_raw.csv         # Phase B output (no reasoning)
    ├── {team_id}.csv                 # Final submission (with reasoning, validated)
    └── submission_metadata.yaml      # Filled-in metadata
```


---

## Required Libraries

```
# requirements.txt
google-genai>=1.0.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4
networkx>=3.0
pandas>=2.0.0
numpy>=1.24.0
streamlit>=1.35.0
plotly>=5.18.0
pydantic>=2.7.0
python-dotenv>=1.0.0
tenacity>=8.2.0
reportlab>=4.0.0
python-docx>=1.1.0
scikit-learn>=1.4.0
```

---

## Challenge Dataset Reference

### Primary Input
```
File:     India_runs_data_and_ai_challenge/candidates.jsonl
Format:   One JSON object per line (JSONL — NOT a JSON array)
Count:    100,000 candidates
Load:     import json
          with open('candidates.jsonl') as f:
              candidates = [json.loads(line) for line in f if line.strip()]
Alt load: import gzip, json
          with gzip.open('candidates.jsonl.gz', 'rt') as f:
              candidates = [json.loads(line) for line in f if line.strip()]
```

### Candidate ID Format
```python
import re
CAND_ID_PATTERN = re.compile(r'^CAND_[0-9]{7}$')
# Valid examples: CAND_0000001, CAND_0042871, CAND_0100000
# The validator rejects any ID not matching this pattern
```

### Sentinel Values (must handle)
```python
github_activity_score  = -1   # means: no GitHub linked → treat as 0.5 (neutral)
offer_acceptance_rate  = -1   # means: no offer history → treat as 0.5 (neutral)
skill_assessment_scores = {}  # empty dict = no platform assessments taken
```


---

## Candidate Schema Mapping

### Full field path → scoring component
```
profile.years_of_experience        → fit_score (experience band)
profile.current_title              → career_score (title classification)
profile.current_company            → career_score (services blacklist)
profile.current_company_size       → career_score (product/startup proxy)
profile.current_industry           → career_score (domain relevance)
profile.location                   → fit_score (location fit)
profile.country                    → fit_score (India preference)
profile.summary                    → skill_score (low-confidence evidence)
profile.headline                   → career_score (weak signal)

career_history[i].title            → career_score (ML/AI role detection)
career_history[i].company          → career_score (services penalty)
career_history[i].industry         → career_score (domain relevance)
career_history[i].company_size     → career_score (company stage)
career_history[i].duration_months  → career_score (tenure weighting)
career_history[i].description      → career_score (retrieval/ranking evidence)
career_history[i].start_date       → honeypot detection
career_history[i].end_date         → honeypot detection
career_history[i].is_current       → career_score (current role weight)

skills[i].name                     → skill_score (AI core skill match)
skills[i].proficiency              → skill_score (depth weight)
skills[i].endorsements             → skill_score (trust multiplier)
skills[i].duration_months          → skill_score (anti-stuffer signal)

education[i].tier                  → fit_score (prestige signal)
education[i].degree                → fit_score (CS/ML relevance)
education[i].field_of_study        → fit_score (ML/AI field bonus)

certifications[i].name             → skill_score (AI/ML cert bonus)
certifications[i].issuer           → skill_score (trusted issuer weight)
certifications[i].year             → skill_score (recency weight)

redrob_signals.profile_completeness_score   → behavioral_score
redrob_signals.last_active_date             → behavioral_score (HIGH)
redrob_signals.open_to_work_flag            → behavioral_score (HIGH)
redrob_signals.recruiter_response_rate      → behavioral_score (HIGH)
redrob_signals.avg_response_time_hours      → behavioral_score
redrob_signals.skill_assessment_scores      → skill_score (HIGHEST trust)
redrob_signals.notice_period_days           → fit_score (HIGH)
redrob_signals.github_activity_score        → skill_score (HIGH, -1=neutral)
redrob_signals.interview_completion_rate    → behavioral_score
redrob_signals.applications_submitted_30d   → behavioral_score
redrob_signals.preferred_work_mode          → fit_score
redrob_signals.willing_to_relocate          → fit_score
redrob_signals.offer_acceptance_rate        → behavioral_score (-1=neutral)
redrob_signals.saved_by_recruiters_30d      → behavioral_score (tie-breaker)
redrob_signals.verified_email               → behavioral_score (tie-breaker)
redrob_signals.verified_phone               → behavioral_score (tie-breaker)
redrob_signals.linkedin_connected           → behavioral_score (tie-breaker)
redrob_signals.connection_count             → behavioral_score (tie-breaker)
redrob_signals.endorsements_received        → behavioral_score (tie-breaker)
redrob_signals.profile_views_received_30d   → behavioral_score (tie-breaker)
redrob_signals.search_appearance_30d        → behavioral_score (tie-breaker)
```


---

## Job Description Mapping

### AI Core Skill Lists (use in skill_score computation)

```python
# Must-have domain skills → weight 1.0
SKILLS_MUST_HAVE = {
    'embeddings', 'sentence-transformers', 'sentence transformers',
    'vector search', 'vector database', 'vector db',
    'faiss', 'pinecone', 'weaviate', 'qdrant', 'milvus',
    'opensearch', 'elasticsearch', 'hybrid search',
    'retrieval', 'information retrieval',
    'ranking', 'ranking systems', 'learning to rank', 'ltr',
    'ndcg', 'map', 'mrr', 'mean reciprocal rank',
    'nlp', 'natural language processing',
    'llm', 'large language model', 'language model',
    'pytorch', 'tensorflow',
    'machine learning', 'ml', 'deep learning',
    'recommendation systems', 'recommender systems',
    'a/b testing', 'ab testing', 'experimentation',
    'bge', 'e5', 'openai embeddings', 'text embeddings',
}

# Nice-to-have skills → weight 0.6
SKILLS_NICE_TO_HAVE = {
    'lora', 'qlora', 'peft', 'fine-tuning', 'fine tuning', 'finetuning',
    'rag', 'retrieval augmented generation', 'retrieval-augmented generation',
    'xgboost', 'gradient boosting',
    'mlflow', 'weights & biases', 'wandb', 'experiment tracking',
    'bentoml', 'triton', 'onnx', 'model serving', 'model deployment',
    'hugging face', 'huggingface', 'transformers',
    'feature store', 'feast',
    'distributed training', 'data parallel', 'model parallel',
    'spark ml', 'pyspark ml',
}

# Supporting technical skills → weight 0.35
SKILLS_SUPPORTING = {
    'python', 'spark', 'pyspark', 'apache spark',
    'airflow', 'apache airflow',
    'kafka', 'apache kafka',
    'dbt', 'snowflake', 'databricks',
    'scikit-learn', 'sklearn',
    'data pipelines', 'feature engineering', 'etl',
    'docker', 'kubernetes', 'k8s',
    'aws', 'aws sagemaker', 'gcp', 'gcp vertex ai', 'azure ml', 'azure',
    'fastapi', 'flask', 'rest api', 'api design',
    'postgresql', 'redis', 'cassandra',
    'git', 'github', 'version control',
    'sql', 'nosql',
    'milvus', 'redis search', 'pgvector',
}

# Trap/noise skills — DO NOT count toward skill_score
SKILLS_TRAP = {
    'seo', 'content writing', 'copywriting', 'photoshop', 'adobe',
    'cad', 'solidworks', 'ansys', 'creo', 'autocad',
    'six sigma', 'sap', 'erp', 'salesforce', 'crm',
    'accounting', 'tally', 'quickbooks',
    'marketing', 'digital marketing', 'google analytics',
    'project management', 'pmp', 'prince2',
    'powerpoint', 'excel', 'word',  # unless in ML context
    'angular', 'react', 'vue', 'typescript', 'javascript', 'node.js',
    'redux', 'webpack', 'graphql', 'tailwind',  # frontend-only
    'manual testing', 'selenium', 'qa testing',
}
```


---

## Services Company Blacklist

```python
SERVICES_BLACKLIST = {
    'tcs', 'tata consultancy services',
    'infosys',
    'wipro',
    'accenture',
    'cognizant', 'cognizant technology solutions', 'cts',
    'capgemini',
    'hcl', 'hcl technologies',
    'mindtree',
    'tech mahindra',
    'ibm global services', 'ibm gbs',
    'mphasis',
    'hexaware',
    'niit technologies',
    'cyient',
    'ltimindtree', 'larsen & toubro infotech', 'lti',
    'persistent systems',
    'zensar', 'birlasoft', 'mastech', 'igate', 'patni',
}

def compute_services_penalty(career_history: list) -> float:
    """
    Returns a multiplier [0.4, 1.0] to apply to career_score.
    1.0  = no services penalty (mostly product companies)
    0.65 = mixed career (50-80% services)
    0.40 = pure services (80%+ at blacklisted companies)
    """
    total_months = sum(r.get('duration_months', 0) for r in career_history)
    if total_months == 0:
        return 1.0

    services_months = sum(
        r.get('duration_months', 0)
        for r in career_history
        if any(bl in r.get('company', '').lower() for bl in SERVICES_BLACKLIST)
    )

    services_ratio = services_months / total_months
    if services_ratio >= 0.80:
        return 0.40
    elif services_ratio >= 0.50:
        return 0.65
    else:
        return 1.0
```

---

## Title Classification Map

```python
TITLE_SCORES = {
    # Score 1.0 — perfect fit
    1.0: [
        'ml engineer', 'machine learning engineer', 'ai engineer',
        'senior ml engineer', 'senior machine learning engineer',
        'senior ai engineer', 'staff ml engineer', 'principal ml engineer',
        'nlp engineer', 'search engineer', 'ranking engineer',
        'recommendation engineer', 'applied scientist',
        'research engineer', 'applied ml engineer',
        'machine learning scientist', 'ai scientist',
    ],
    # Score 0.85 — strong fit
    0.85: [
        'data scientist', 'senior data scientist', 'lead data scientist',
        'mlops engineer', 'ml platform engineer', 'ml infrastructure engineer',
        'ai researcher', 'research scientist',
        'deep learning engineer', 'computer vision engineer',
        'conversational ai engineer', 'speech engineer',
    ],
    # Score 0.70 — adjacent fit (dark horse territory)
    0.70: [
        'data engineer', 'senior data engineer', 'lead data engineer',
        'analytics engineer', 'ml data engineer',
        'backend engineer', 'software engineer',  # check descriptions
        'platform engineer', 'infrastructure engineer',
        'full stack engineer',  # check if ML context
    ],
    # Score 0.50 — weak signal
    0.50: [
        'software developer', 'developer', 'engineer',
        'data analyst', 'senior data analyst',
        'devops engineer', 'cloud engineer', 'sre',
        'product engineer', 'growth engineer',
    ],
    # Score 0.25 — likely noise
    0.25: [
        'business analyst', 'product manager', 'product owner',
        'project manager', 'program manager', 'scrum master',
        'technical lead', 'tech lead',  # check if hands-on coding
    ],
    # Score 0.0 — not a fit
    0.0: [
        'marketing manager', 'hr manager', 'operations manager',
        'content writer', 'graphic designer', 'sales executive',
        'accountant', 'customer support', 'civil engineer',
        'mechanical engineer', 'chemical engineer',
        'finance manager', 'legal counsel',
    ],
}

def get_title_score(title: str) -> float:
    title_lower = title.lower().strip()
    for score, titles in TITLE_SCORES.items():
        for t in titles:
            if t in title_lower:
                return score
    # Default: unknown title gets 0.4
    return 0.40
```


---

## Feature Engineering Specification

### Career Score Features
```python
def compute_career_score(profile: dict, career_history: list) -> float:
    """
    Measures genuine ML/AI engineering background from actual career evidence.
    Returns float [0.0, 1.0].
    """
    import math

    # 1. Current title score
    current_title_score = get_title_score(profile.get('current_title', ''))

    # 2. Career history ML evidence score
    #    Weight each role by duration and title relevance
    total_months = sum(r.get('duration_months', 1) for r in career_history)
    if total_months == 0:
        total_months = 1

    history_score = 0.0
    ml_keywords = {
        'retrieval', 'ranking', 'embedding', 'embeddings', 'vector',
        'recommendation', 'search', 'nlp', 'natural language', 'llm',
        'machine learning', 'deep learning', 'neural', 'model', 'inference',
        'feature', 'pipeline', 'faiss', 'transformer', 'bert', 'gpt',
        'fine-tun', 'training', 'evaluation', 'a/b test',
    }
    for role in career_history:
        role_title_score = get_title_score(role.get('title', ''))
        desc = role.get('description', '').lower()
        desc_evidence = sum(1 for kw in ml_keywords if kw in desc)
        desc_score = min(desc_evidence / 5.0, 1.0)  # cap at 5 matches
        combined_role_score = 0.6 * role_title_score + 0.4 * desc_score
        weight = role.get('duration_months', 1) / total_months
        history_score += combined_role_score * weight

    # 3. Services penalty
    services_penalty = compute_services_penalty(career_history)

    # 4. Product company bonus (any role at non-services company with ML title)
    has_product_ml = any(
        get_title_score(r.get('title', '')) >= 0.70
        and not any(bl in r.get('company', '').lower() for bl in SERVICES_BLACKLIST)
        for r in career_history
    )
    product_bonus = 1.10 if has_product_ml else 1.0

    raw_score = (0.40 * current_title_score + 0.60 * history_score)
    return min(raw_score * services_penalty * product_bonus, 1.0)
```

### Skill Score Features
```python
import math

PROFICIENCY_WEIGHTS = {'beginner': 0.25, 'intermediate': 0.50, 'advanced': 0.75, 'expert': 1.0}

def compute_skill_score(skills: list, skill_assessment_scores: dict) -> float:
    """
    Depth-weighted AI skill score. Resists keyword stuffing.
    Returns float [0.0, 1.0].
    """
    if not skills:
        return 0.0

    total_weighted = 0.0
    max_possible = 0.0

    for skill in skills:
        name = skill.get('name', '').lower().strip()
        proficiency = skill.get('proficiency', 'beginner')
        endorsements = skill.get('endorsements', 0)
        duration = skill.get('duration_months', 0)

        # Determine skill category weight
        if any(kw in name for kw in SKILLS_MUST_HAVE):
            category_weight = 1.0
        elif any(kw in name for kw in SKILLS_NICE_TO_HAVE):
            category_weight = 0.6
        elif any(kw in name for kw in SKILLS_SUPPORTING):
            category_weight = 0.35
        elif any(kw in name for kw in SKILLS_TRAP):
            continue  # skip entirely
        else:
            category_weight = 0.2  # unknown skill

        # Proficiency score
        prof_score = PROFICIENCY_WEIGHTS.get(proficiency, 0.25)

        # Platform-verified override
        if name in {k.lower() for k in skill_assessment_scores}:
            verified_key = next(k for k in skill_assessment_scores if k.lower() == name)
            verified_score = skill_assessment_scores[verified_key] / 100.0
            prof_score = 0.3 * prof_score + 0.7 * verified_score

        # Depth multiplier (anti-stuffer)
        depth = min(duration / 24.0, 1.0) if duration > 0 else 0.0
        stuffer_flag = (proficiency in ('advanced', 'expert') and duration == 0)
        stuffer_penalty = 0.3 if stuffer_flag else 1.0

        # Endorsement trust
        trust = math.log1p(endorsements) / math.log1p(100)

        skill_val = prof_score * (0.5 + 0.3 * depth + 0.2 * trust) * stuffer_penalty
        max_possible += category_weight
        total_weighted += category_weight * skill_val

    if max_possible == 0:
        return 0.0
    return min(total_weighted / max(max_possible, 1.0), 1.0)
```


### Behavioral Score Features
```python
from datetime import date

TODAY = date.today()

def compute_behavioral_score(signals: dict) -> float:
    """
    Availability + reachability multiplier from redrob_signals.
    Returns float [0.10, 1.0].
    """
    # Activity recency (30%)
    last_active = signals.get('last_active_date', '')
    try:
        la_date = date.fromisoformat(str(last_active))
        days = (TODAY - la_date).days
    except Exception:
        days = 365
    if   days <=  30: activity = 1.00
    elif days <=  60: activity = 0.85
    elif days <=  90: activity = 0.70
    elif days <= 180: activity = 0.50
    else:             activity = 0.25

    # Open to work (20%)
    otw = 1.0 if signals.get('open_to_work_flag', False) else 0.60

    # Recruiter response rate (25%)
    rr = float(signals.get('recruiter_response_rate', 0.3))

    # Interview completion rate (15%)
    ir = float(signals.get('interview_completion_rate', 0.5))

    # Active applications (10%)
    apps = min(int(signals.get('applications_submitted_30d', 0)) / 5.0, 1.0)

    raw = (activity * 0.30) + (otw * 0.20) + (rr * 0.25) + (ir * 0.15) + (apps * 0.10)
    return max(0.10, raw)


def compute_fit_score(profile: dict, education: list, signals: dict) -> float:
    """
    Logistics fit: location, experience years, notice period, work mode, education tier.
    Returns float [0.0, 1.0].
    """
    # Experience years (35%)
    yoe = float(profile.get('years_of_experience', 0))
    if   5.0 <= yoe <= 9.0:   exp = 1.00
    elif 3.0 <= yoe <  5.0:   exp = 0.75
    elif 9.0 <  yoe <= 12.0:  exp = 0.80
    elif yoe > 12.0:           exp = 0.65
    else:                      exp = 0.40

    # Notice period (25%)
    np_days = int(signals.get('notice_period_days', 90))
    if   np_days <=  15: notice = 1.00
    elif np_days <=  30: notice = 0.95
    elif np_days <=  60: notice = 0.80
    elif np_days <=  90: notice = 0.65
    elif np_days <= 120: notice = 0.45
    else:                notice = 0.30

    # Location fit (20%)
    target_cities = ['pune', 'noida', 'delhi', 'ncr', 'gurgaon', 'gurugram',
                     'mumbai', 'hyderabad', 'bangalore', 'bengaluru', 'new delhi']
    loc = profile.get('location', '').lower()
    country = profile.get('country', '').lower()
    relocate = bool(signals.get('willing_to_relocate', False))
    in_target = any(c in loc for c in target_cities)
    in_india = country in ('india', 'in')
    if in_target:             location = 1.00
    elif in_india and relocate: location = 0.90
    elif in_india:            location = 0.75
    elif relocate:            location = 0.65
    else:                     location = 0.40

    # Work mode (10%)
    mode = signals.get('preferred_work_mode', 'flexible')
    wm = {'hybrid': 1.0, 'onsite': 0.90, 'flexible': 0.85, 'remote': 0.60}.get(mode, 0.75)

    # Education tier (10%)
    tier_map = {'tier_1': 1.0, 'tier_2': 0.8, 'tier_3': 0.6, 'tier_4': 0.4, 'unknown': 0.5}
    best = max((tier_map.get(e.get('tier', 'unknown'), 0.5) for e in education), default=0.5)

    return (exp * 0.35) + (notice * 0.25) + (location * 0.20) + (wm * 0.10) + (best * 0.10)
```


---

## Honeypot Detection Specification

```python
def is_honeypot(candidate: dict) -> bool:
    """
    Returns True if candidate has an impossible/fabricated profile.
    Honeypots score 0.0 and must not appear in top 100.
    ~80 honeypots exist in the 100K pool.
    """
    profile = candidate.get('profile', {})
    career = candidate.get('career_history', [])
    skills = candidate.get('skills', [])

    # Rule 1: Impossible experience timeline
    total_career_months = sum(r.get('duration_months', 0) for r in career)
    claimed_years = float(profile.get('years_of_experience', 0))
    career_years_from_history = total_career_months / 12.0
    # Allow 2-year gap for education/breaks
    if claimed_years > career_years_from_history + 2.0 and claimed_years > 5:
        return True

    # Rule 2: Mass expert skills with zero duration
    expert_zero_duration = sum(
        1 for s in skills
        if s.get('proficiency') in ('advanced', 'expert')
        and s.get('duration_months', 0) == 0
    )
    if expert_zero_duration >= 7:  # 7+ expert skills with 0 months each
        return True

    # Rule 3: Identical description copy-pasted across roles
    descs = [r.get('description', '').strip() for r in career if r.get('description', '').strip()]
    if len(descs) >= 3:
        unique_descs = set(descs)
        if len(unique_descs) == 1:  # all same description
            return True

    # Rule 4: Profile completeness > 90 but empty career descriptions
    completeness = candidate.get('redrob_signals', {}).get('profile_completeness_score', 0)
    all_descs_empty = all(
        not r.get('description', '').strip() for r in career
    )
    if completeness > 85 and all_descs_empty and len(career) > 1:
        return True

    return False
```

---

## Final Score Assembly

```python
def compute_final_score(candidate: dict) -> float:
    """
    Assemble the four components into a single [0.0, 1.0] score.
    Used in rank.py for all 100K candidates.
    NO network calls. NO LLM calls. Pure computation.
    """
    if is_honeypot(candidate):
        return 0.0

    profile    = candidate.get('profile', {})
    career     = candidate.get('career_history', [])
    education  = candidate.get('education', [])
    skills     = candidate.get('skills', [])
    signals    = candidate.get('redrob_signals', {})
    assessments = signals.get('skill_assessment_scores', {})

    # GitHub score — handle -1 sentinel
    gh_raw = float(signals.get('github_activity_score', -1))
    gh_score = 0.50 if gh_raw == -1 else gh_raw / 100.0

    career_s     = compute_career_score(profile, career)
    skill_s      = compute_skill_score(skills, assessments)
    skill_s      = 0.80 * skill_s + 0.20 * gh_score  # blend github in
    behavioral_s = compute_behavioral_score(signals)
    fit_s        = compute_fit_score(profile, education, signals)

    raw = (
        career_s     * 0.45 +
        skill_s      * 0.25 +
        behavioral_s * 0.20 +
        fit_s        * 0.10
    )
    return min(max(raw, 0.0), 1.0)
```

---

## Submission Format Contract

```python
# From validate_submission.py — enforced exactly
REQUIRED_HEADER  = ['candidate_id', 'rank', 'score', 'reasoning']
EXPECTED_ROWS    = 100
CAND_ID_PATTERN  = r'^CAND_[0-9]{7}$'

# Output generation pattern
def write_submission(ranked: list, filepath: str):
    """
    ranked: list of dicts sorted by score descending, then candidate_id ascending.
    Each dict: {candidate_id, score, reasoning}
    """
    import csv
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(REQUIRED_HEADER)
        for rank_num, row in enumerate(ranked[:100], start=1):
            writer.writerow([
                row['candidate_id'],
                rank_num,
                round(row['score'], 4),
                row['reasoning'],
            ])
```

### Tie-breaking
```python
# Sort: score descending, then candidate_id ascending (validator enforces this for ties)
ranked = sorted(
    all_scored,
    key=lambda x: (-x['score'], x['candidate_id'])
)
```

---

## Environment Variables

```
GEMINI_API_KEY=     # Required — Gemini API key (aistudio.google.com/apikey)
GEMINI_MODEL=       # Optional — defaults to gemini-2.5-flash
LOG_LEVEL=          # DEBUG | INFO | WARNING
```

---

## LLM Client Configuration

```python
# config.py
PRIMARY_MODEL            = 'gemini-2.5-flash'   # override with GEMINI_MODEL env var
TEMPERATURE_JD_ANALYSIS  = 0.0
TEMPERATURE_REASONING    = 0.0
MAX_RETRIES              = 3
JSON_RETRY_SUFFIX        = 'Respond only with valid JSON. No markdown. No explanation.'
```

---

## Pipeline Execution Summary

| Phase | Script | LLM? | Time limit | Network? |
|---|---|---|---|---|
| A1 — JD Analysis | `src/stage1_jd_analysis.py` | Yes | Unlimited | Yes |
| A2 — Pre-compute | `src/precompute.py` | No | Unlimited | No |
| B — Ranking | `src/rank.py` | **NO** | **< 5 min** | **NO** |
| C — Reasoning | `src/reason.py` | Yes (top 100 only) | Unlimited | Yes |
| Validate | `validate_submission.py` | No | Seconds | No |
| Demo | `src/stage8_dashboard.py` | Optional | N/A | Optional |

**Reproduce command (single command from repo root):**
```bash
python src/rank.py \
  --candidates India_runs_data_and_ai_challenge/candidates.jsonl \
  --features data/processed/features.pkl \
  --out outputs/{team_id}.csv
```

---

## Validation Checklist (run before every submission)

```bash
# Step 1: Validate format
python India_runs_data_and_ai_challenge/validate_submission.py outputs/{team_id}.csv

# Step 2: Manual checks
python -c "
import csv
with open('outputs/{team_id}.csv') as f:
    rows = list(csv.DictReader(f))
print(f'Rows: {len(rows)}')                          # must be 100
print(f'Unique IDs: {len(set(r[\"candidate_id\"] for r in rows))}')  # must be 100
scores = [float(r[\"score\"]) for r in rows]
print(f'Score range: {min(scores):.4f} - {max(scores):.4f}')
print(f'Non-increasing: {all(scores[i]>=scores[i+1] for i in range(99))}')  # must be True
print(f'Empty reasoning: {sum(1 for r in rows if not r[\"reasoning\"].strip())}')  # must be 0
unique_reasoning = len(set(r[\"reasoning\"] for r in rows))
print(f'Unique reasoning strings: {unique_reasoning}')  # should be 100
"
```
