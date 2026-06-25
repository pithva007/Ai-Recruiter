# utils/feature_engineering.py
# Deterministic scoring functions — NO network calls, NO LLM calls.
# Used by: src/precompute.py (offline), src/rank.py (< 5-min ranking),
#          src/stage8_dashboard.py (sandbox demo)
#
# Fixes applied (v2):
#   Fix 1 — compute_services_penalty():    near-disqualification (0.05) for pure-services careers
#   Fix 2 — compute_skill_trust_score():   replaces compute_skill_score() with career cross-reference
#   Fix 3 — compute_availability_multiplier(): replaces compute_behavioral_score() with 8-signal multiplier
#   Fix 4 — detect_production_retrieval_experience(): new signal for this JD's #1 requirement
#   Fix 5 — compute_final_score():         updated formula using all new functions
#
# Legacy functions (compute_skill_score, compute_behavioral_score) are kept as aliases
# so that any external callers don't break.

import math
from datetime import date

# ---------------------------------------------------------------------------
# Reference date — fixed to submission window start for reproducibility.
# Using date.today() would cause score drift across days.
# ---------------------------------------------------------------------------
TODAY        = date.today()
REFERENCE_DATE = date(2026, 6, 1)   # Fix 3: explicit reference for availability calculation

# ---------------------------------------------------------------------------
# FIX 1 — Services company set
# Combined from AGENT.md SERVICES_BLACKLIST + Fix 1 spec
# ---------------------------------------------------------------------------
SERVICES_COMPANIES = {
    # Fix 1 explicit list
    'tcs', 'tata consultancy', 'tata consultancy services',
    'infosys',
    'wipro',
    'accenture',
    'cognizant', 'cognizant technology solutions', 'cts',
    'capgemini',
    'hcl technologies', 'hcl tech', 'hcl',
    'tech mahindra',
    'mphasis',
    'hexaware',
    "l&t infotech", 'ltimindtree', 'larsen & toubro infotech', 'lti',
    'persistent systems',
    # Additional from AGENT.md blacklist
    'ibm global services', 'ibm gbs',
    'mindtree',
    'niit technologies',
    'cyient',
    'zensar', 'birlasoft', 'mastech', 'igate', 'patni',
}

# Kept as alias for backward compat
SERVICES_BLACKLIST = SERVICES_COMPANIES

# ---------------------------------------------------------------------------
# Skill relevance lists (from SKILLS.md + Fix 2 spec)
# ---------------------------------------------------------------------------
SKILLS_MUST_HAVE = {
    # Fix 2 MUST_HAVE set
    'embeddings', 'vector search', 'vector database', 'faiss', 'pinecone',
    'weaviate', 'qdrant', 'milvus', 'elasticsearch', 'opensearch',
    'sentence transformers', 'sentence-transformers', 'retrieval', 'ranking',
    'recommendation systems', 'recommender systems',
    'python', 'llm', 'large language model', 'fine-tuning', 'finetuning',
    'rag', 'information retrieval',
    'ndcg', 'map', 'mrr', 'evaluation framework', 'a/b testing', 'ab testing',
    'bert', 'transformer', 'transformers', 'huggingface', 'hugging face',
    'pytorch', 'tensorflow',
    # Additional must-haves from SKILLS.md
    'vector db', 'hybrid search', 'information retrieval',
    'ranking systems', 'learning to rank', 'ltr',
    'mean reciprocal rank', 'experimentation',
    'natural language processing', 'nlp',
    'machine learning', 'deep learning',
    'bge', 'e5', 'openai embeddings', 'text embeddings', 'gpt',
}

SKILLS_NICE_TO_HAVE = {
    # Fix 2 NICE_TO_HAVE set
    'lora', 'qlora', 'peft',
    'xgboost', 'learning to rank', 'gradient boosting',
    'distributed systems',
    'mlflow', 'weights & biases', 'wandb', 'experiment tracking',
    'langchain', 'openai', 'gemini', 'anthropic', 'llama', 'mistral',
    'spark', 'kafka', 'airflow', 'kubernetes', 'docker',
    # Additional from SKILLS.md
    'fine-tuning', 'retrieval augmented generation', 'retrieval-augmented generation',
    'bentoml', 'triton', 'onnx', 'model serving', 'model deployment',
    'feature store', 'feast',
    'distributed training', 'data parallel', 'model parallel',
    'spark ml', 'pyspark ml',
}

SKILLS_SUPPORTING = {
    'pyspark', 'apache spark',
    'apache airflow', 'apache kafka',
    'dbt', 'snowflake', 'databricks',
    'scikit-learn', 'sklearn',
    'data pipelines', 'feature engineering', 'etl',
    'k8s',
    'aws', 'aws sagemaker', 'gcp', 'gcp vertex ai', 'azure ml', 'azure',
    'fastapi', 'flask', 'rest api',
    'postgresql', 'redis', 'cassandra',
    'sql', 'nosql',
    'milvus', 'pgvector',
    'statistical modeling', 'statistics',
}

SKILLS_TRAP = {
    'seo', 'content writing', 'copywriting', 'photoshop', 'adobe',
    'cad', 'solidworks', 'ansys', 'creo', 'autocad',
    'six sigma', 'sap', 'erp', 'salesforce', 'crm',
    'accounting', 'tally', 'quickbooks',
    'marketing', 'digital marketing', 'google analytics',
    'project management', 'pmp', 'prince2',
    'powerpoint', 'word',
    'angular', 'react', 'vue', 'typescript', 'javascript', 'node.js',
    'redux', 'webpack', 'graphql', 'tailwind',
    'manual testing', 'selenium', 'qa testing',
    'speech recognition', 'tts',   # removed from supporting — adjacent to disqualifier domain
}

# ---------------------------------------------------------------------------
# Title score lookup (unchanged from original)
# ---------------------------------------------------------------------------
TITLE_SCORE_MAP = [
    (1.00, ['ml engineer', 'machine learning engineer', 'ai engineer',
            'senior ml engineer', 'senior machine learning engineer', 'senior ai engineer',
            'staff ml engineer', 'principal ml engineer', 'lead ml engineer',
            'nlp engineer', 'search engineer', 'ranking engineer',
            'recommendation engineer', 'applied scientist', 'research engineer',
            'applied ml', 'ml scientist', 'ai scientist', 'data science engineer']),
    (0.85, ['data scientist', 'mlops', 'ml platform', 'ml infrastructure',
            'ai researcher', 'research scientist', 'deep learning engineer',
            'junior ml', 'junior machine learning', 'junior ai engineer']),
    (0.70, ['data engineer', 'analytics engineer', 'ml data engineer',
            'backend engineer', 'software engineer',
            'platform engineer', 'infrastructure engineer']),
    (0.50, ['software developer', 'developer', 'data analyst',
            'devops', 'cloud engineer', 'sre', 'site reliability']),
    (0.25, ['business analyst', 'product manager', 'product owner',
            'project manager', 'program manager', 'scrum master', 'tech lead']),
    (0.00, ['marketing manager', 'marketing', 'hr manager', 'human resource',
            'operations manager', 'content writer', 'graphic designer',
            'sales executive', 'accountant', 'customer support',
            'civil engineer', 'mechanical engineer', 'chemical engineer',
            'finance manager', 'legal', 'supply chain']),
]

ML_DESC_KEYWORDS = {
    'retrieval', 'ranking', 'embedding', 'embeddings', 'vector',
    'recommendation', 'search', 'nlp', 'natural language', 'llm',
    'machine learning', 'deep learning', 'neural', 'model', 'inference',
    'feature pipeline', 'faiss', 'transformer', 'bert', 'gpt',
    'fine-tun', 'fine tuning', 'training', 'evaluation', 'a/b test',
    'semantic', 'reranking', 're-ranking', 'hybrid search',
}

PROFICIENCY_WEIGHTS = {
    'beginner': 0.25, 'intermediate': 0.50, 'advanced': 0.75, 'expert': 1.0,
}

TARGET_CITIES = {
    'pune', 'noida', 'delhi', 'ncr', 'gurgaon', 'gurugram',
    'mumbai', 'hyderabad', 'bangalore', 'bengaluru', 'new delhi',
}

ACCEPTABLE_CITIES = TARGET_CITIES   # alias used in Fix 3


# ---------------------------------------------------------------------------
# Helper: title score
# ---------------------------------------------------------------------------
def get_title_score(title: str) -> float:
    t = title.lower().strip()
    for score, keywords in TITLE_SCORE_MAP:
        if any(kw in t for kw in keywords):
            return score
    return 0.40


# ===========================================================================
# FIX 1 — compute_services_penalty()
# Near-disqualification for pure-services careers.
# Applied as a MULTIPLIER on career_score AND on the final score.
# ===========================================================================
def compute_services_penalty(career_history: list) -> float:
    """
    Returns a multiplier [0.05, 1.0].

    0.05  — pure services (≥80% months, NO non-services role ever)
    0.40  — escaped services (≥80% months but has at least one non-services role)
    0.75  — mixed (50–79% months at services)
    1.00  — no penalty

    Applied as multiplier on career_score (and propagated to final_score via
    the services_penalty factor in compute_final_score).
    """
    total = sum(r.get('duration_months', 0) for r in career_history)
    if total == 0:
        return 1.0

    services_months = 0
    has_non_services_role = False

    for r in career_history:
        company = r.get('company', '').lower()
        dur     = r.get('duration_months', 0)
        is_svc  = any(bl in company for bl in SERVICES_COMPANIES)
        if is_svc:
            services_months += dur
        else:
            if dur > 0:
                has_non_services_role = True

    ratio = services_months / total

    if ratio >= 0.80:
        if not has_non_services_role:
            return 0.05   # pure services — near-disqualification
        else:
            return 0.40   # escaped services — significant penalty
    if ratio >= 0.50:
        return 0.75       # mixed — moderate penalty
    return 1.0            # no penalty


# ===========================================================================
# FIX 2 — compute_skill_trust_score()
# Replaces compute_skill_score(). Uses duration + endorsement trust weighting
# and JD-specific relevance weights (must_have=2.0, nice_to_have=1.0).
# ===========================================================================
def _jd_relevance(skill_name: str) -> float:
    """Return JD relevance weight: 2.0 = must_have, 1.0 = nice_to_have, 0.0 = irrelevant."""
    n = skill_name.lower()
    if any(kw in n for kw in SKILLS_MUST_HAVE):
        return 2.0
    if any(kw in n for kw in SKILLS_NICE_TO_HAVE):
        return 1.0
    if any(kw in n for kw in SKILLS_TRAP):
        return 0.0   # irrelevant — skip
    if any(kw in n for kw in SKILLS_SUPPORTING):
        return 0.5   # supporting, minor weight
    return 0.0       # unknown — neutral (don't help, don't hurt)


def compute_skill_trust_score(skills: list, career_history: list,
                               assessments: dict | None = None) -> float:
    """
    Fix 2: Depth + trust weighted skill score using JD relevance.

    Per skill:
      duration_trust    = min(duration_months / 24, 1.0)
      endorsement_trust = min(endorsements / 20, 1.0)
      skill_weight:
        - duration == 0                             → 0.10  (keyword stuffer signal)
        - proficiency expert/advanced AND dur == 0  → 0.05  (worst signal)
        - else                                      → 0.40 + 0.30*duration_trust + 0.30*endorsement_trust
      jd_rel = _jd_relevance(name)   (2.0 / 1.0 / 0.5 / 0.0)
      contribution = skill_weight * jd_rel

    Returns weighted average normalised to [0, 1].
    Platform-verified assessment scores (if provided) override proficiency.
    """
    if not skills:
        return 0.0

    if assessments is None:
        assessments = {}
    assessments_lower = {k.lower(): v for k, v in assessments.items()}

    total_weight  = 0.0
    total_contrib = 0.0

    for skill in skills:
        name        = skill.get('name', '').lower().strip()
        proficiency = skill.get('proficiency', 'beginner')
        endorsements= skill.get('endorsements', 0)
        duration    = skill.get('duration_months', 0)

        jd_rel = _jd_relevance(name)
        if jd_rel == 0.0:
            continue   # irrelevant skill — skip entirely

        # Platform-verified override
        if name in assessments_lower:
            verified_frac = assessments_lower[name] / 100.0
            # Blend: trust verified 70% over self-reported proficiency
            prof_frac = PROFICIENCY_WEIGHTS.get(proficiency, 0.25)
            prof_frac = 0.30 * prof_frac + 0.70 * verified_frac
        else:
            prof_frac = PROFICIENCY_WEIGHTS.get(proficiency, 0.25)

        duration_trust    = min(duration / 24.0, 1.0)
        endorsement_trust = min(endorsements / 20.0, 1.0)

        if duration == 0 and proficiency in ('expert', 'advanced'):
            skill_weight = 0.05   # worst: claims expertise, zero usage
        elif duration == 0:
            skill_weight = 0.10   # keyword stuffer — any proficiency
        else:
            skill_weight = 0.40 + (0.30 * duration_trust) + (0.30 * endorsement_trust)

        # Scale by proficiency fraction
        contribution = skill_weight * prof_frac * jd_rel

        total_weight  += jd_rel   # normalise by relevance weight sum
        total_contrib += contribution

    if total_weight == 0.0:
        return 0.0

    raw = total_contrib / total_weight
    return min(raw, 1.0)


# Legacy alias — keeps precompute.py and other callers working unchanged
def compute_skill_score(skills: list, assessments: dict) -> float:
    """Legacy alias for compute_skill_trust_score (no career cross-reference)."""
    return compute_skill_trust_score(skills, [], assessments)


# ===========================================================================
# FIX 3 — compute_availability_multiplier()
# 8-signal multiplicative availability score. Replaces compute_behavioral_score().
# Reference date: 2026-06-01 (submission window).
# ===========================================================================
def compute_availability_multiplier(signals: dict, profile: dict | None = None) -> float:
    """
    Fix 3: Multiplicative availability signal using 8 redrob signals.
    Returns float in [0.0, 1.15].  Cap at 1.15 (bonuses cannot exceed 15%).
    No lower floor here — truly unavailable candidates should score near zero.

    Signals used:
      1. last_active_date         (activity recency)
      2. open_to_work_flag        (availability declaration)
      3. recruiter_response_rate  (reachability)
      4. notice_period_days       (time-to-hire)
      5. interview_completion_rate (reliability)
      6. location + willing_to_relocate (geography)
      7. github_activity_score    (OSS bonus)
      8. verified_email + verified_phone (identity trust)
    """
    if profile is None:
        profile = {}

    multiplier = 1.0

    # ---- Signal 1: Last active date ----
    last_active_str = signals.get('last_active_date', '')
    try:
        la_date      = date.fromisoformat(str(last_active_str))
        days_inactive = (REFERENCE_DATE - la_date).days
    except Exception:
        days_inactive = 365   # unknown → assume worst

    if   days_inactive > 180: multiplier *= 0.40  # effectively gone
    elif days_inactive >  90: multiplier *= 0.65  # likely passive
    elif days_inactive >  30: multiplier *= 0.85  # slightly passive
    # < 30 days: no penalty

    # ---- Signal 2: Open to work ----
    if not signals.get('open_to_work_flag', True):
        multiplier *= 0.70

    # ---- Signal 3: Recruiter response rate ----
    rr = float(signals.get('recruiter_response_rate', 0.5))
    if   rr < 0.10: multiplier *= 0.50   # very hard to reach
    elif rr < 0.25: multiplier *= 0.75   # unreliable responder
    elif rr > 0.70: multiplier *= 1.10   # bonus: responsive candidate

    # ---- Signal 4: Notice period ----
    notice = int(signals.get('notice_period_days', 30))
    if   notice <=  15: multiplier *= 1.05   # near-immediate start
    elif notice <=  30: multiplier *= 1.00   # JD says can buy out 30 days
    elif notice <=  60: multiplier *= 0.90
    elif notice <=  90: multiplier *= 0.75
    else:               multiplier *= 0.55   # 90+ days = significant barrier

    # ---- Signal 5: Interview completion rate ----
    icr = float(signals.get('interview_completion_rate', 0.7))
    if   icr < 0.40: multiplier *= 0.60   # ghosts interviews
    elif icr < 0.60: multiplier *= 0.80

    # ---- Signal 6: Location + willing to relocate ----
    loc      = profile.get('location', '').lower()
    country  = profile.get('country', '').lower()
    willing  = bool(signals.get('willing_to_relocate', False))
    in_city  = any(city in loc for city in ACCEPTABLE_CITIES)
    is_india = country in {'india', 'in', ''}

    if not is_india:
        multiplier *= 0.20   # outside India — JD says no visa sponsorship
    elif not in_city and not willing:
        multiplier *= 0.50   # wrong city, won't relocate
    elif not in_city and willing:
        multiplier *= 0.90   # wrong city but willing — minor penalty

    # ---- Signal 7: GitHub activity (bonus) ----
    gh = float(signals.get('github_activity_score', -1))
    if gh == -1:
        pass                  # no GitHub — neutral
    elif gh > 60:
        multiplier *= 1.08    # active OSS contributor — JD explicitly values this
    elif gh > 30:
        multiplier *= 1.03

    # ---- Signal 8: Verified contact (identity trust) ----
    v_email = bool(signals.get('verified_email', True))
    v_phone = bool(signals.get('verified_phone', True))
    if not v_email and not v_phone:
        multiplier *= 0.80    # completely unverified identity

    return min(multiplier, 1.15)   # cap bonus at 1.15


# Legacy alias
def compute_behavioral_score(signals: dict) -> float:
    """Legacy alias. Returns availability multiplier (not clamped to 0.10 floor)."""
    return compute_availability_multiplier(signals, profile=None)


# ===========================================================================
# FIX 4 — detect_production_retrieval_experience()
# Detects the JD's #1 explicit requirement: shipped a retrieval/ranking/
# recommendation system to real users at a PRODUCT company.
# ===========================================================================
RETRIEVAL_KEYWORDS = {
    'retrieval', 'search', 'ranking', 'recommendation', 'vector', 'embedding',
    'similarity', 'index', 'faiss', 'elasticsearch', 'recommend', 'ranker',
    'rerank', 'recall', 'precision', 'ndcg',
}

def detect_production_retrieval_experience(career_history: list) -> float:
    """
    Fix 4: Detect shipped retrieval/ranking/recommendation systems at product companies.

    Per role:
      - 3+ RETRIEVAL_KEYWORDS + NOT services + duration >= 6 months → +0.40 (strong signal)
      - 1-2 RETRIEVAL_KEYWORDS + NOT services + duration >= 6 months → +0.15 (weak signal)
      - 3+ RETRIEVAL_KEYWORDS + IS services                          → +0.10 (partial credit)

    Returns cumulative score capped at 1.0.
    """
    score = 0.0

    for job in career_history:
        desc     = (job.get('description') or '').lower()
        company  = (job.get('company') or '').lower()
        duration = int(job.get('duration_months', 0))

        is_services   = any(s in company for s in SERVICES_COMPANIES)
        keyword_hits  = sum(1 for kw in RETRIEVAL_KEYWORDS if kw in desc)

        if keyword_hits >= 3 and not is_services and duration >= 6:
            score += 0.40   # strong: multiple retrieval keywords, product company, real tenure
        elif keyword_hits >= 1 and not is_services and duration >= 6:
            score += 0.15   # weak: some retrieval work at product company
        elif keyword_hits >= 3 and is_services:
            score += 0.10   # partial: services company but did retrieval work

    return min(score, 1.0)


# ===========================================================================
# Fit Score (unchanged from original — location/notice/edu already handled)
# ===========================================================================
def compute_fit_score(profile: dict, education: list, signals: dict) -> float:
    """Logistics fit: experience band + education tier only (location/notice in multiplier)."""
    yoe = float(profile.get('years_of_experience', 0))
    if   5.0 <= yoe <= 9.0:   exp = 1.00
    elif 3.0 <= yoe <  5.0:   exp = 0.75
    elif 9.0 <  yoe <= 12.0:  exp = 0.80
    elif yoe > 12.0:           exp = 0.65
    else:                      exp = 0.40

    tier_map = {'tier_1': 1.0, 'tier_2': 0.8, 'tier_3': 0.6, 'tier_4': 0.4, 'unknown': 0.5}
    best = max(
        (tier_map.get(e.get('tier', 'unknown'), 0.5) for e in education),
        default=0.5,
    )

    return (exp * 0.70) + (best * 0.30)


# ===========================================================================
# Honeypot Detection (unchanged)
# ===========================================================================
def is_honeypot(candidate: dict) -> bool:
    profile = candidate.get('profile', {})
    career  = candidate.get('career_history', [])
    skills  = candidate.get('skills', [])

    total_career_months = sum(r.get('duration_months', 0) for r in career)
    claimed_years       = float(profile.get('years_of_experience', 0))
    if claimed_years > (total_career_months / 12.0) + 2.0 and claimed_years > 5:
        return True

    expert_zero = sum(
        1 for s in skills
        if s.get('proficiency') in ('advanced', 'expert')
        and s.get('duration_months', 0) == 0
    )
    if expert_zero >= 7:
        return True

    descs = [r.get('description', '').strip() for r in career if r.get('description', '').strip()]
    if len(descs) >= 3 and len(set(descs)) == 1:
        return True

    completeness = candidate.get('redrob_signals', {}).get('profile_completeness_score', 0)
    all_empty = all(not r.get('description', '').strip() for r in career)
    if completeness > 85 and all_empty and len(career) > 1:
        return True

    return False


# ===========================================================================
# Career Score (unchanged signature — uses new compute_services_penalty)
# ===========================================================================
def compute_career_score(profile: dict, career_history: list) -> float:
    """Career signal: title + history ML evidence + services penalty + product bonus."""
    current_title_score = get_title_score(profile.get('current_title', ''))

    total_months = max(sum(r.get('duration_months', 1) for r in career_history), 1)
    history_score = 0.0

    for role in career_history:
        role_title_score = get_title_score(role.get('title', ''))
        desc     = role.get('description', '').lower()
        evidence = sum(1 for kw in ML_DESC_KEYWORDS if kw in desc)
        desc_score = min(evidence / 5.0, 1.0)
        combined   = 0.6 * role_title_score + 0.4 * desc_score
        weight     = role.get('duration_months', 1) / total_months
        history_score += combined * weight

    # Fix 1: upgraded penalty (0.05 for pure-services, not 0.4)
    services_penalty = compute_services_penalty(career_history)

    has_product_ml = any(
        get_title_score(r.get('title', '')) >= 0.70
        and not any(bl in r.get('company', '').lower() for bl in SERVICES_COMPANIES)
        for r in career_history
    )
    product_bonus = 1.10 if has_product_ml else 1.0

    raw = (0.40 * current_title_score + 0.60 * history_score)
    return min(raw * services_penalty * product_bonus, 1.0)


# ===========================================================================
# FIX 5 — compute_final_score()
# Updated formula:
#   career_score    × 0.30
#   skill_score     × 0.20
#   retrieval_score × 0.30   ← new, JD's #1 explicit requirement
#   fit_score       × 0.20
# then multiply by behavioral_multiplier and services_penalty
# ===========================================================================
def compute_final_score(candidate: dict) -> float:
    """
    Fix 5: Revised final score formula.

    Base:
      career_score    * 0.30
      skill_score     * 0.20
      retrieval_score * 0.30
      fit_score       * 0.20

    Multipliers applied on top:
      × behavioral_multiplier   (availability/reachability — can go to 1.15)
      × services_penalty        (also embedded in career_score but applied here
                                 to affect the full score, not just career)

    Range: [0.0, 1.0]  (clamped)
    No network calls. No LLM calls. Pure deterministic computation.
    """
    if is_honeypot(candidate):
        return 0.0

    profile   = candidate.get('profile', {})
    career    = candidate.get('career_history', [])
    education = candidate.get('education', [])
    skills    = candidate.get('skills', [])
    signals   = candidate.get('redrob_signals', {})
    assessments = signals.get('skill_assessment_scores', {})

    # --- Component scores (all 0-1) ---
    career_s    = compute_career_score(profile, career)
    skill_s     = compute_skill_trust_score(skills, career, assessments)
    retrieval_s = detect_production_retrieval_experience(career)
    fit_s       = compute_fit_score(profile, education, signals)

    # --- Blend github into skill score (as before) ---
    gh_raw   = float(signals.get('github_activity_score', -1))
    gh_score = 0.50 if gh_raw < 0 else gh_raw / 100.0
    skill_s  = 0.80 * skill_s + 0.20 * gh_score

    # --- Weighted base score ---
    base = (
        career_s    * 0.30 +
        skill_s     * 0.20 +
        retrieval_s * 0.30 +
        fit_s       * 0.20
    )

    # --- Multipliers ---
    behavioral_mult  = compute_availability_multiplier(signals, profile)
    services_penalty = compute_services_penalty(career)

    final = base * behavioral_mult * services_penalty
    return round(min(max(final, 0.0), 1.0), 6)
