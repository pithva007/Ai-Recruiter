# utils/feature_engineering.py
# Deterministic scoring functions — NO network calls, NO LLM calls.
# Used by: src/precompute.py (offline), src/rank.py (< 5-min ranking),
#          src/stage8_dashboard.py (sandbox demo)
#
# All scoring weights and skill lists are grounded in:
#   AGENT.md   — scoring architecture and formulas
#   SKILLS.md  — skill lists, title map, services blacklist
#   docs/challenge_findings.md — signal weights

import math
from datetime import date

# ---------------------------------------------------------------------------
# Reference date (used for recency calculations)
# ---------------------------------------------------------------------------
TODAY = date.today()

# ---------------------------------------------------------------------------
# Services company blacklist (from AGENT.md + JD explicit disqualifiers)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# AI skill lists (from SKILLS.md — grounded in JD must-haves/nice-to-haves)
# ---------------------------------------------------------------------------
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
    'machine learning', 'deep learning',
    'recommendation systems', 'recommender systems',
    'a/b testing', 'ab testing', 'experimentation',
    'bge', 'e5', 'openai embeddings', 'text embeddings',
    'transformer', 'bert', 'gpt',
}

SKILLS_NICE_TO_HAVE = {
    'lora', 'qlora', 'peft', 'fine-tuning', 'fine tuning', 'finetuning',
    'rag', 'retrieval augmented generation', 'retrieval-augmented generation',
    'xgboost', 'gradient boosting',
    'mlflow', 'weights & biases', 'wandb', 'experiment tracking',
    'bentoml', 'triton', 'onnx', 'model serving', 'model deployment',
    'hugging face', 'huggingface', 'transformers',
    'feature store', 'feast',
    'distributed training',
    'spark ml', 'pyspark ml',
    'image classification', 'object detection',  # CV but acceptable
}

SKILLS_SUPPORTING = {
    'python', 'spark', 'pyspark', 'apache spark',
    'airflow', 'apache airflow',
    'kafka', 'apache kafka',
    'dbt', 'snowflake', 'databricks',
    'scikit-learn', 'sklearn',
    'data pipelines', 'feature engineering', 'etl',
    'docker', 'kubernetes', 'k8s',
    'aws', 'aws sagemaker', 'gcp', 'gcp vertex ai', 'azure ml', 'azure',
    'fastapi', 'flask', 'rest api',
    'postgresql', 'redis', 'cassandra',
    'sql', 'nosql',
    'milvus', 'pgvector',
    'statistical modeling', 'statistics',
    'speech recognition', 'tts',       # adjacent, minor weight
    'gans', 'nlp',
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
}

# ---------------------------------------------------------------------------
# Title score lookup (from SKILLS.md)
# ---------------------------------------------------------------------------
TITLE_SCORE_MAP = [
    # (score, keyword_list)  — checked in order; first match wins
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
            'backend engineer', 'software engineer',   # check descriptions
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

# Keywords indicating ML/retrieval work in job descriptions
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


# ---------------------------------------------------------------------------
# Helper: title score
# ---------------------------------------------------------------------------
def get_title_score(title: str) -> float:
    t = title.lower().strip()
    for score, keywords in TITLE_SCORE_MAP:
        if any(kw in t for kw in keywords):
            return score
    return 0.40  # unknown title default


# ---------------------------------------------------------------------------
# Helper: services penalty
# ---------------------------------------------------------------------------
def compute_services_penalty(career_history: list) -> float:
    total = sum(r.get('duration_months', 0) for r in career_history)
    if total == 0:
        return 1.0
    services = sum(
        r.get('duration_months', 0)
        for r in career_history
        if any(bl in r.get('company', '').lower() for bl in SERVICES_BLACKLIST)
    )
    ratio = services / total
    if ratio >= 0.80:  return 0.40
    if ratio >= 0.50:  return 0.65
    return 1.0


# ---------------------------------------------------------------------------
# Component 1: Career Score
# ---------------------------------------------------------------------------
def compute_career_score(profile: dict, career_history: list) -> float:
    current_title_score = get_title_score(profile.get('current_title', ''))

    total_months = max(sum(r.get('duration_months', 1) for r in career_history), 1)
    history_score = 0.0

    for role in career_history:
        role_title_score = get_title_score(role.get('title', ''))
        desc = role.get('description', '').lower()
        evidence = sum(1 for kw in ML_DESC_KEYWORDS if kw in desc)
        desc_score = min(evidence / 5.0, 1.0)
        combined = 0.6 * role_title_score + 0.4 * desc_score
        weight = role.get('duration_months', 1) / total_months
        history_score += combined * weight

    services_penalty = compute_services_penalty(career_history)

    has_product_ml = any(
        get_title_score(r.get('title', '')) >= 0.70
        and not any(bl in r.get('company', '').lower() for bl in SERVICES_BLACKLIST)
        for r in career_history
    )
    product_bonus = 1.10 if has_product_ml else 1.0

    raw = (0.40 * current_title_score + 0.60 * history_score)
    return min(raw * services_penalty * product_bonus, 1.0)


# ---------------------------------------------------------------------------
# Component 2: Skill Score
# ---------------------------------------------------------------------------
def compute_skill_score(skills: list, assessments: dict) -> float:
    if not skills:
        return 0.0

    total_weighted = 0.0
    max_possible = 0.0

    for skill in skills:
        name = skill.get('name', '').lower().strip()
        proficiency = skill.get('proficiency', 'beginner')
        endorsements = skill.get('endorsements', 0)
        duration = skill.get('duration_months', 0)

        # Category weight
        if any(kw in name for kw in SKILLS_MUST_HAVE):
            cat_w = 1.0
        elif any(kw in name for kw in SKILLS_NICE_TO_HAVE):
            cat_w = 0.6
        elif any(kw in name for kw in SKILLS_SUPPORTING):
            cat_w = 0.35
        elif any(kw in name for kw in SKILLS_TRAP):
            continue   # skip trap skills entirely
        else:
            cat_w = 0.20

        prof_score = PROFICIENCY_WEIGHTS.get(proficiency, 0.25)

        # Platform-verified override
        assessments_lower = {k.lower(): v for k, v in assessments.items()}
        if name in assessments_lower:
            verified = assessments_lower[name] / 100.0
            prof_score = 0.3 * prof_score + 0.7 * verified

        # Anti-stuffer: expert with zero duration
        stuffer = (proficiency in ('advanced', 'expert') and duration == 0)
        stuffer_penalty = 0.3 if stuffer else 1.0

        depth = min(duration / 24.0, 1.0) if duration > 0 else 0.0
        trust = math.log1p(endorsements) / math.log1p(100)

        skill_val = prof_score * (0.5 + 0.3 * depth + 0.2 * trust) * stuffer_penalty
        max_possible += cat_w
        total_weighted += cat_w * skill_val

    if max_possible == 0:
        return 0.0
    return min(total_weighted / max(max_possible, 1.0), 1.0)


# ---------------------------------------------------------------------------
# Component 3: Behavioral Score
# ---------------------------------------------------------------------------
def compute_behavioral_score(signals: dict) -> float:
    # Activity recency
    last_active_str = signals.get('last_active_date', '')
    try:
        la_date = date.fromisoformat(str(last_active_str))
        days = (TODAY - la_date).days
    except Exception:
        days = 365
    if   days <=  30: activity = 1.00
    elif days <=  60: activity = 0.85
    elif days <=  90: activity = 0.70
    elif days <= 180: activity = 0.50
    else:             activity = 0.25

    otw  = 1.0 if signals.get('open_to_work_flag', False) else 0.60
    rr   = float(signals.get('recruiter_response_rate', 0.3))
    ir   = float(signals.get('interview_completion_rate', 0.5))
    apps = min(int(signals.get('applications_submitted_30d', 0)) / 5.0, 1.0)

    raw = (activity * 0.30) + (otw * 0.20) + (rr * 0.25) + (ir * 0.15) + (apps * 0.10)
    return max(0.10, raw)


# ---------------------------------------------------------------------------
# Component 4: Fit Score
# ---------------------------------------------------------------------------
def compute_fit_score(profile: dict, education: list, signals: dict) -> float:
    yoe = float(profile.get('years_of_experience', 0))
    if   5.0 <= yoe <= 9.0:   exp = 1.00
    elif 3.0 <= yoe <  5.0:   exp = 0.75
    elif 9.0 <  yoe <= 12.0:  exp = 0.80
    elif yoe > 12.0:           exp = 0.65
    else:                      exp = 0.40

    np_days = int(signals.get('notice_period_days', 90))
    if   np_days <=  15: notice = 1.00
    elif np_days <=  30: notice = 0.95
    elif np_days <=  60: notice = 0.80
    elif np_days <=  90: notice = 0.65
    elif np_days <= 120: notice = 0.45
    else:                notice = 0.30

    loc     = profile.get('location', '').lower()
    country = profile.get('country', '').lower()
    relocate = bool(signals.get('willing_to_relocate', False))
    mode    = signals.get('preferred_work_mode', 'flexible')

    in_target = any(c in loc for c in TARGET_CITIES)
    in_india  = country in ('india', 'in')
    if in_target:              location = 1.00
    elif in_india and relocate: location = 0.90
    elif in_india:              location = 0.75
    elif relocate:              location = 0.65
    else:                       location = 0.40

    wm = {'hybrid': 1.0, 'onsite': 0.90, 'flexible': 0.85, 'remote': 0.60}.get(mode, 0.75)

    tier_map = {'tier_1': 1.0, 'tier_2': 0.8, 'tier_3': 0.6, 'tier_4': 0.4, 'unknown': 0.5}
    best_tier = max(
        (tier_map.get(e.get('tier', 'unknown'), 0.5) for e in education),
        default=0.5,
    )

    return (exp * 0.35) + (notice * 0.25) + (location * 0.20) + (wm * 0.10) + (best_tier * 0.10)


# ---------------------------------------------------------------------------
# Honeypot Detection
# ---------------------------------------------------------------------------
def is_honeypot(candidate: dict) -> bool:
    profile = candidate.get('profile', {})
    career  = candidate.get('career_history', [])
    skills  = candidate.get('skills', [])

    # Rule 1: Impossible experience timeline
    total_career_months = sum(r.get('duration_months', 0) for r in career)
    claimed_years = float(profile.get('years_of_experience', 0))
    if claimed_years > (total_career_months / 12.0) + 2.0 and claimed_years > 5:
        return True

    # Rule 2: Mass expert skills with zero duration
    expert_zero = sum(
        1 for s in skills
        if s.get('proficiency') in ('advanced', 'expert')
        and s.get('duration_months', 0) == 0
    )
    if expert_zero >= 7:
        return True

    # Rule 3: All descriptions are identical copy-paste
    descs = [r.get('description', '').strip() for r in career if r.get('description', '').strip()]
    if len(descs) >= 3 and len(set(descs)) == 1:
        return True

    # Rule 4: High completeness score but all empty career descriptions
    completeness = candidate.get('redrob_signals', {}).get('profile_completeness_score', 0)
    all_empty = all(not r.get('description', '').strip() for r in career)
    if completeness > 85 and all_empty and len(career) > 1:
        return True

    return False


# ---------------------------------------------------------------------------
# Final Score Assembly
# ---------------------------------------------------------------------------
def compute_final_score(candidate: dict) -> float:
    """
    Assemble all four components into a single [0.0, 1.0] score.
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

    # GitHub: -1 means no GitHub linked → neutral 0.5
    gh_raw   = float(signals.get('github_activity_score', -1))
    gh_score = 0.50 if gh_raw < 0 else gh_raw / 100.0

    career_s   = compute_career_score(profile, career)
    skill_s    = compute_skill_score(skills, assessments)
    skill_s    = 0.80 * skill_s + 0.20 * gh_score   # blend github credibility in
    behav_s    = compute_behavioral_score(signals)
    fit_s      = compute_fit_score(profile, education, signals)

    raw = (
        career_s * 0.45 +
        skill_s  * 0.25 +
        behav_s  * 0.20 +
        fit_s    * 0.10
    )
    return min(max(raw, 0.0), 1.0)
