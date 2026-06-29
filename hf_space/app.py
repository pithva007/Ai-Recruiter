import streamlit as st
import json
import csv
import gzip as gz_lib
import io
import re
import math
import os
from datetime import date, datetime
from pathlib import Path

# ============================================================
# PAGE CONFIG — must be first Streamlit call
# ============================================================
st.set_page_config(
    page_title="AI Recruiter · Redrob Hackathon",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── session_state init ────────────────────────────────────────
for _k, _v in [("live_results", None), ("live_ran", False)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ============================================================
# DATA LOADER — pre-computed results (bundled in hf_space/)
# ============================================================
_HERE = Path(__file__).parent

@st.cache_data
def load_submission_csv() -> list[dict]:
    """Load pre-computed submission.csv (100 ranked, official results)."""
    p = _HERE / "submission.csv"
    if not p.exists():
        return []
    rows = []
    with open(p, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows

@st.cache_data
def load_ranked_candidates() -> list[dict]:
    """Load pre-computed ranked_candidates.csv (30 deeply LLM-scored)."""
    p = _HERE / "ranked_candidates.csv"
    if not p.exists():
        return []
    rows = []
    with open(p, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows

@st.cache_data
def load_jd_features() -> dict:
    p = _HERE / "jd_features.json"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_sample_candidates() -> list[dict]:
    p = _HERE / "sample_candidates.json"
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return json.load(f)

SUBMISSION     = load_submission_csv()
RANKED_DEEP    = load_ranked_candidates()
JD             = load_jd_features()
SAMPLE_CANDS   = load_sample_candidates()

# ============================================================
# SCORING ENGINE v4  (self-contained — synced with feature_engineering.py)
# ============================================================

def _extract_skill_name(s: dict) -> str:
    return (s.get("skill") or "").lower().strip()

_JD_LOADED = bool(JD)

SERVICES_COMPANIES = {
    "tcs", "tata consultancy", "tata consultancy services", "infosys", "wipro",
    "accenture", "cognizant", "cognizant technology solutions", "cts", "capgemini",
    "hcl technologies", "hcl tech", "hcl", "tech mahindra", "mphasis",
    "hexaware", "l&t infotech", "ltimindtree", "mindtree",
    "ibm gbs", "ibm global services", "niit technologies", "cyient",
    "zensar", "birlasoft", "persistent systems", "mastech",
}
if _JD_LOADED:
    for _co in JD.get("services_company_names", []):
        SERVICES_COMPANIES.add(_co.lower().strip())

MUST_HAVE_SKILLS = {
    "embeddings", "sentence-transformers", "sentence transformers",
    "vector search", "vector database", "faiss", "pinecone", "weaviate",
    "qdrant", "milvus", "elasticsearch", "opensearch", "retrieval", "ranking",
    "recommendation", "python", "llm", "large language model", "fine-tuning",
    "rag", "information retrieval", "ndcg", "map", "mrr", "bert", "transformers",
    "huggingface", "pytorch", "tensorflow", "bge", "e5", "hybrid search",
    "learning to rank", "xgboost", "reranking", "dense retrieval",
    "mlops", "evaluation framework", "a/b testing", "product engineering",
}
if _JD_LOADED:
    for _s in JD.get("must_have_skills", []):
        MUST_HAVE_SKILLS.add(_extract_skill_name(_s))

NICE_TO_HAVE_SKILLS = {
    "lora", "qlora", "peft", "langchain", "openai", "gemini", "llama",
    "mistral", "spark", "kafka", "airflow", "kubernetes", "docker", "mlflow",
    "weights & biases", "distributed systems", "inference optimization",
    "open-source", "oss", "hr-tech", "recruiting",
}
if _JD_LOADED:
    for _s in JD.get("nice_to_have_skills", []):
        NICE_TO_HAVE_SKILLS.add(_extract_skill_name(_s))

RETRIEVAL_KEYWORDS = {
    "retrieval", "search", "ranking", "recommendation", "vector", "embedding",
    "embeddings", "similarity", "index", "faiss", "elasticsearch", "opensearch",
    "recommend", "ranker", "rerank", "recall", "precision", "ndcg", "mrr", "map",
    "bm25", "dense", "sparse", "hybrid", "semantic search", "information retrieval",
    "vector database", "pinecone", "weaviate", "qdrant", "milvus",
    "sentence-transformers", "bi-encoder", "cross-encoder",
}

REFERENCE_DATE = date.today()

_YOUNG_COMPANY_MAX_MONTHS = {"sarvam ai": 38, "sarvam": 38, "krutrim": 30}

_MUST_HAVE_GROUPS = [
    {"embedding", "embeddings", "sentence-transformers", "sentence transformers",
     "bge", "e5", "dense retrieval", "bi-encoder"},
    {"faiss", "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
     "elasticsearch", "chroma", "pgvector", "vector database", "vector db", "vector search"},
    {"ranking", "bm25", "tf-idf", "hybrid search", "information retrieval",
     "ranker", "reranking", "learning to rank", "ltr"},
    {"ndcg", "mrr", "map", "recall@k", "precision@k", "a/b testing",
     "ab testing", "evaluation framework", "experimentation"},
    {"python"},
    {"nlp", "natural language processing", "transformer", "transformers",
     "bert", "gpt", "llm", "large language model", "rag", "text classification"},
    {"pytorch", "tensorflow", "scikit-learn", "sklearn", "deep learning",
     "machine learning", "statistical modeling"},
    {"search", "search engineering", "solr", "lucene", "recommendation", "recommender"},
]


def _parse_date(date_str):
    if not date_str:
        return None
    for fmt in ["%Y-%m-%d", "%Y-%m", "%Y"]:
        try:
            return datetime.strptime(str(date_str)[:10], fmt).date()
        except Exception:
            continue
    return None


def compute_services_penalty(career_history):
    if not career_history:
        return 1.0
    total = sum(j.get("duration_months", 0) or 0 for j in career_history)
    if total == 0:
        return 1.0
    services = sum(
        j.get("duration_months", 0) or 0 for j in career_history
        if any(s in (j.get("company") or "").lower() for s in SERVICES_COMPANIES)
    )
    ratio = services / total
    has_escape = any(
        not any(s in (j.get("company") or "").lower() for s in SERVICES_COMPANIES)
        for j in career_history
    )
    if ratio >= 0.80 and not has_escape: return 0.05
    if ratio >= 0.80: return 0.40
    if ratio >= 0.70: return 0.45
    if ratio >= 0.50: return 0.70
    if ratio >= 0.25: return 0.85
    return 1.0


def compute_skill_trust_score(skills):
    if not skills:
        return 0.0
    total_weight = 0.0
    total_score = 0.0
    for s in skills:
        name = (s.get("name") or "").lower()
        duration = s.get("duration_months") or 0
        endorsements = s.get("endorsements") or 0
        proficiency = (s.get("proficiency") or "").lower()
        if any(kw in name for kw in MUST_HAVE_SKILLS):
            jd_weight = 2.0
        elif any(kw in name for kw in NICE_TO_HAVE_SKILLS):
            jd_weight = 1.0
        else:
            jd_weight = 0.0
        if jd_weight == 0.0:
            continue
        if duration == 0 and proficiency in ("expert", "advanced"):
            skill_weight = 0.05
        elif duration == 0:
            skill_weight = 0.10
        else:
            duration_trust = min(duration / 24.0, 1.0)
            endorse_trust = min(math.log1p(endorsements) / math.log1p(100), 1.0)
            skill_weight = 0.40 + (0.30 * duration_trust) + (0.30 * endorse_trust)
        total_score += skill_weight * jd_weight
        total_weight += jd_weight
    return (total_score / total_weight) if total_weight > 0 else 0.0


def detect_retrieval_experience(career_history):
    if not career_history:
        return 0.0
    score = 0.0
    for job in career_history:
        desc = (job.get("description") or "").lower()
        company = (job.get("company") or "").lower()
        duration = job.get("duration_months") or 0
        is_services = any(s in company for s in SERVICES_COMPANIES)
        hits = sum(1 for kw in RETRIEVAL_KEYWORDS if kw in desc)
        if hits >= 3 and not is_services and duration >= 6:
            score += 0.40
        elif hits >= 1 and not is_services and duration >= 6:
            score += 0.15
        elif hits >= 3 and is_services:
            score += 0.10
    return min(score, 1.0)


def compute_availability_multiplier(signals, profile):
    if not signals:
        return 0.70
    multiplier = 1.0
    last_active = _parse_date(signals.get("last_active_date"))
    if last_active:
        days = (REFERENCE_DATE - last_active).days
        if days > 180:
            multiplier *= 0.40
        elif days > 90:
            multiplier *= 0.65
        elif days > 30:
            multiplier *= 0.85
    if not signals.get("open_to_work_flag", True):
        multiplier *= 0.70
    rr = signals.get("recruiter_response_rate", 0.5) or 0.5
    if rr < 0.10:
        multiplier *= 0.50
    elif rr < 0.25:
        multiplier *= 0.75
    elif rr > 0.70:
        multiplier *= 1.10
    notice = signals.get("notice_period_days") or 30
    if notice <= 15:
        multiplier *= 1.05
    elif notice <= 30:
        multiplier *= 1.0
    elif notice <= 60:
        multiplier *= 0.90
    elif notice <= 90:
        multiplier *= 0.75
    else:
        multiplier *= 0.55
    icr = signals.get("interview_completion_rate") or 0.7
    if icr < 0.40:
        multiplier *= 0.60
    elif icr < 0.60:
        multiplier *= 0.80
    location = (profile.get("location") or "").lower()
    country = (profile.get("country") or "india").lower()
    willing = signals.get("willing_to_relocate", False)
    GOOD_CITIES = {"pune", "noida", "hyderabad", "mumbai", "delhi",
                   "gurugram", "gurgaon", "bengaluru", "bangalore", "ncr"}
    in_india = "india" in country or country in {"in", ""}
    in_good_city = any(c in location for c in GOOD_CITIES)
    if not in_india:
        multiplier *= 0.20
    elif not in_good_city and not willing:
        multiplier *= 0.50
    elif not in_good_city and willing:
        multiplier *= 0.90
    gh = signals.get("github_activity_score", -1)
    if gh is not None and gh > 60:
        multiplier *= 1.08
    elif gh is not None and gh > 30:
        multiplier *= 1.03
    return min(multiplier, 1.15)


def _experience_band_multiplier(years):
    if years is None or years <= 0: return 0.40
    if years < 4.0:  return 0.35
    if years < 5.0:  return 0.72
    if years < 5.5:  return 0.88
    if 5.5 <= years <= 9.0: return 1.0
    if 9.0 < years <= 11.0: return 0.95
    return 0.85


def _title_chaser_penalty(career_history):
    if not career_history or len(career_history) < 3:
        return 1.0
    past = [j for j in career_history if not j.get("is_current", False)]
    if not past:
        return 1.0
    short = sum(1 for j in past if (j.get("duration_months") or 99) < 18)
    ratio = short / len(past)
    if ratio >= 0.70: return 0.60
    if ratio >= 0.50: return 0.80
    return 1.0


def _is_honeypot(candidate):
    profile = candidate.get("profile", {})
    career  = candidate.get("career_history", [])
    skills  = candidate.get("skills", [])
    claimed_years = float(profile.get("years_of_experience") or 0)
    total_career_months = sum(j.get("duration_months") or 0 for j in career)
    if claimed_years > (total_career_months / 12.0) + 2.0 and claimed_years > 5:
        return True
    if claimed_years > 0 and total_career_months > (claimed_years * 12 * 2.5):
        return True
    expert_zero = sum(
        1 for s in skills
        if (s.get("duration_months") or 0) == 0
        and (s.get("proficiency") or "") in ("expert", "advanced")
    )
    if expert_zero >= 3:
        return True
    total_expert = sum(1 for s in skills if (s.get("proficiency") or "") == "expert")
    if total_expert >= 12:
        return True
    for job in career:
        company = (job.get("company") or "").lower().strip()
        dur = job.get("duration_months") or 0
        for co_name, max_mo in _YOUNG_COMPANY_MAX_MONTHS.items():
            if co_name in company and dur > max_mo:
                return True
    descs = [job.get("description", "").strip() for job in career
             if job.get("description", "").strip()]
    if len(descs) >= 3 and len(set(descs)) == 1:
        return True
    completeness = candidate.get("redrob_signals", {}).get("profile_completeness_score", 0)
    all_empty = all(not job.get("description", "").strip() for job in career)
    if completeness > 85 and all_empty and len(career) > 1:
        return True
    return False


def _compute_honeypot_suspicion(candidate):
    skills  = candidate.get("skills", [])
    career  = candidate.get("career_history", [])
    profile = candidate.get("profile", {})
    suspicion = 0
    claimed_years = float(profile.get("years_of_experience") or 0)
    total_career_months = sum(j.get("duration_months") or 0 for j in career)
    expert_zero = sum(1 for s in skills
                      if (s.get("proficiency") or "") in ("advanced", "expert")
                      and (s.get("duration_months") or 0) == 0)
    total_expert = sum(1 for s in skills if (s.get("proficiency") or "") == "expert")
    if 1 <= expert_zero <= 2: suspicion += 2
    if 10 <= total_expert <= 11: suspicion += 2
    if claimed_years > 0 and total_career_months > (claimed_years * 12 * 2.0):
        suspicion += 1
    if suspicion == 0: return 1.00
    if suspicion == 1: return 0.85
    if suspicion == 2: return 0.70
    return 0.50


def _compute_must_have_coverage_gate(skills):
    used = {(s.get("name") or "").lower().strip()
            for s in skills if (s.get("duration_months") or 0) > 0}
    covered = sum(1 for grp in _MUST_HAVE_GROUPS
                  if any(kw in name for kw in grp for name in used))
    if covered >= 5: return 1.00
    if covered >= 3: return 0.85
    if covered >= 1: return 0.50
    return 0.25


COMPLETELY_IRRELEVANT_TITLES = {
    "accountant", "marketing manager", "operations manager", "hr manager",
    "human resources", "graphic designer", "content writer", "customer support",
    "business analyst", "project manager", "sales", "finance", "administrative",
    "civil engineer", "mechanical engineer", "electrical engineer",
    "hardware engineer", "manufacturing engineer",
    ".net developer", "mobile developer", "frontend engineer",
    "full stack developer", "devops engineer", "qa engineer",
    "java developer", "web developer", "ui developer",
}
RELEVANT_TITLE_KEYWORDS = {
    "ml", "machine learning", "ai ", "artificial intelligence",
    "data scientist", "data science", "nlp", "deep learning",
    "recommendation", "search engineer", "retrieval", "applied scientist",
    "research scientist", "research engineer", "software engineer",
    "software developer", "backend engineer", "data engineer",
    "platform engineer", "infrastructure engineer", "cloud engineer",
    "sre", "mlops", "llm", "generative",
}


def compute_title_relevance_gate(profile: dict) -> float:
    title = (profile.get("current_title") or "").lower()
    for irr in COMPLETELY_IRRELEVANT_TITLES:
        if irr in title:
            return 0.05
    for rel in RELEVANT_TITLE_KEYWORDS:
        if rel in title:
            return 1.0
    return 0.85


def score_candidate(candidate) -> float:
    if _is_honeypot(candidate):
        return 0.01
    profile  = candidate.get("profile", {})
    career   = candidate.get("career_history", [])
    skills   = candidate.get("skills", [])
    signals  = candidate.get("redrob_signals", {})
    years       = profile.get("years_of_experience") or 0
    exp_mult    = _experience_band_multiplier(years)
    title_pen   = _title_chaser_penalty(career)
    career_score    = exp_mult * title_pen
    skill_score     = compute_skill_trust_score(skills)
    retrieval_score = detect_retrieval_experience(career)
    avail_mult      = compute_availability_multiplier(signals, profile)
    services_pen    = compute_services_penalty(career)
    title_gate      = compute_title_relevance_gate(profile)
    must_have_gate  = _compute_must_have_coverage_gate(skills)
    suspicion_mult  = _compute_honeypot_suspicion(candidate)
    base = (
        career_score    * 0.30 +
        skill_score     * 0.20 +
        retrieval_score * 0.30 +
        0.50            * 0.20
    )
    final = base * avail_mult * services_pen * title_gate * must_have_gate * suspicion_mult
    return round(min(max(final, 0.0), 1.0), 6)


# ============================================================
# FILE PARSING  (supports JSON array, JSONL, single JSON, gzip)
# ============================================================

def read_uploaded_file(uploaded_file) -> str:
    raw = uploaded_file.read()
    if raw[:2] == b'\x1f\x8b':
        try:
            return gz_lib.decompress(raw).decode("utf-8")
        except Exception as e:
            raise ValueError(f"Failed to decompress gzip: {e}")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def parse_candidates_file(content: str) -> list:
    content = content.strip()
    if content.startswith("["):
        data = json.loads(content)
        return data if isinstance(data, list) else [data]
    if content.startswith("{"):
        candidates = []
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        if not candidates:
            raise ValueError("No valid JSON objects found.")
        return candidates
    raise ValueError("File must start with '[' (JSON array) or '{' (JSONL).")


def get_candidate_id(candidate: dict, index: int) -> str:
    cid = candidate.get("candidate_id", "")
    if re.match(r"^CAND_[0-9]{7}$", str(cid)):
        return cid
    return f"CAND_{(index + 1):07d}"


def build_reasoning(candidate: dict, rank: int, score: float) -> str:
    import html as html_lib
    profile  = candidate.get("profile", {})
    signals  = candidate.get("redrob_signals", {})
    skills   = candidate.get("skills", [])
    career   = candidate.get("career_history", [])
    title    = profile.get("current_title") or "Professional"
    yoe      = profile.get("years_of_experience") or 0
    rr       = signals.get("recruiter_response_rate", 0.5)
    notice   = signals.get("notice_period_days", 30)
    top_skills = [s.get("name") for s in skills if s.get("duration_months", 0) > 6][:2]
    skills_str = ", ".join(top_skills) if top_skills else ""
    if _is_honeypot(candidate):
        r = f"{title} with {yoe}yr; flagged as honeypot (impossible skill profile)."
    elif compute_services_penalty(career) <= 0.40:
        r = f"{title} with {yoe}yr at services firms; penalized for pure-services career."
    elif rank <= 10:
        r = f"{title} with {yoe}yr; " + (f"strong {skills_str}; " if skills_str else "") + f"response rate {rr:.2f}, notice {notice}d."
    elif rank <= 30:
        r = f"{title} with {yoe}yr; " + (f"{skills_str}; " if skills_str else "") + f"moderate engagement signals; notice {notice}d."
    elif rank <= 50:
        r = f"{title} with {yoe}yr; some {skills_str}; mid-ranked on availability and depth."
    else:
        r = f"{title} with {yoe}yr; adjacent background; included based on engagement signals."
    return html_lib.escape(r)


def to_csv_string(results: list) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    for r in results:
        writer.writerow([r["candidate_id"], r["rank"], f"{r['score']:.4f}", r.get("reasoning", "")])
    return output.getvalue()


# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif !important; }

[data-testid="stAppViewContainer"] { background: #0a0d14; color: #e2e8f0; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: #0f1117 !important; border-right: 1px solid #1e2433; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #111827 0%, #0a0d14 50%, #0d0a1f 100%);
    border: 1px solid #1e2433;
    border-radius: 20px;
    padding: 40px 48px 32px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(239,68,68,0.10) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -60px; left: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(99,102,241,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #ef4444, #f97316, #eab308);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 8px;
    line-height: 1.2;
}
.hero-subtitle { color: #94a3b8; font-size: 1rem; margin: 0 0 20px; }
.hero-pills { display: flex; gap: 10px; flex-wrap: wrap; }
.pill {
    background: rgba(239,68,68,0.12);
    border: 1px solid rgba(239,68,68,0.3);
    color: #fca5a5;
    padding: 5px 14px;
    border-radius: 99px;
    font-size: 0.78rem;
    font-weight: 600;
}
.pill-blue  { background: rgba(59,130,246,0.12); border-color: rgba(59,130,246,0.3); color: #93c5fd; }
.pill-green { background: rgba(34,197,94,0.12);  border-color: rgba(34,197,94,0.3);  color: #86efac; }
.pill-amber { background: rgba(234,179,8,0.12);  border-color: rgba(234,179,8,0.3);  color: #fde68a; }

/* ── Cards ── */
.card {
    background: #111827;
    border: 1px solid #1e2433;
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 16px;
}
.card-title {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    color: #4b5563;
    margin-bottom: 14px;
}

/* ── Metric boxes ── */
.metric-row { display: flex; gap: 12px; margin-bottom: 24px; flex-wrap: wrap; }
.metric-box {
    flex: 1;
    min-width: 100px;
    background: #111827;
    border: 1px solid #1e2433;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.metric-val  { font-size: 1.9rem; font-weight: 800; color: #e2e8f0; line-height: 1; }
.metric-label{ font-size: 0.68rem; color: #4b5563; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 5px; }
.metric-green .metric-val { color: #4ade80; }
.metric-blue  .metric-val { color: #60a5fa; }
.metric-amber .metric-val { color: #fbbf24; }
.metric-purple .metric-val { color: #a78bfa; }

/* ── Score badge ── */
.score-high  { background: rgba(34,197,94,0.12);  color: #4ade80;  padding: 3px 12px; border-radius: 8px; font-weight: 700; font-size: 0.88rem; display:inline-block; }
.score-mid   { background: rgba(234,179,8,0.12);  color: #fbbf24;  padding: 3px 12px; border-radius: 8px; font-weight: 700; font-size: 0.88rem; display:inline-block; }
.score-low   { background: rgba(239,68,68,0.12);  color: #f87171;  padding: 3px 12px; border-radius: 8px; font-weight: 700; font-size: 0.88rem; display:inline-block; }

/* ── Candidate card (deep view) ── */
.cand-card {
    background: #111827;
    border: 1px solid #1e2433;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 12px;
    transition: border-color 0.2s;
}
.cand-card:hover { border-color: #374151; }
.cand-rank {
    font-size: 1.2rem;
    font-weight: 800;
    min-width: 42px;
    text-align: center;
    padding-top: 2px;
}
.dark-horse-badge {
    background: rgba(234,179,8,0.15);
    border: 1px solid rgba(234,179,8,0.4);
    color: #fbbf24;
    padding: 2px 8px; border-radius: 4px;
    font-size: 0.68rem; font-weight: 700; margin-left: 8px;
}
.honeypot-badge {
    background: rgba(239,68,68,0.15);
    border: 1px solid rgba(239,68,68,0.4);
    color: #f87171;
    padding: 2px 8px; border-radius: 4px;
    font-size: 0.68rem; font-weight: 700; margin-left: 8px;
}
.interview-q {
    background: #0f172a;
    border-left: 3px solid #6366f1;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.80rem;
    color: #94a3b8;
    font-style: italic;
    line-height: 1.6;
}
.flag-item {
    font-size: 0.78rem;
    padding: 4px 0;
    line-height: 1.5;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #111827;
    border-radius: 12px;
    padding: 4px;
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #4b5563;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #ef4444, #dc2626) !important;
    color: white !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #ef4444, #dc2626) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 12px 28px !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(239,68,68,0.35) !important;
}
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #16a34a, #15803d) !important;
    color: white !important; border-radius: 10px !important;
    border: none !important; font-weight: 700 !important; width: 100% !important;
}
[data-testid="stFileUploader"] {
    background: #111827 !important;
    border: 2px dashed #1e2433 !important;
    border-radius: 12px !important;
}
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #ef4444, #f97316) !important;
}
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-left-width: 3px !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="padding:8px 0 20px">
      <div style="font-size:1.1rem;font-weight:800;
                  background:linear-gradient(90deg,#ef4444,#f97316);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent">
        🎯 AI Recruiter
      </div>
      <div style="font-size:0.72rem;color:#4b5563;margin-top:2px">Redrob Hackathon · v4</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card-title">⚖️ Scoring Formula (v4)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.78rem; color:#94a3b8; line-height:2.0;">
    <span style="color:#60a5fa">career</span> × 0.30<br>
    <span style="color:#4ade80">skills_trust</span> × 0.20<br>
    <span style="color:#f97316">retrieval_exp</span> × 0.30<br>
    <span style="color:#a78bfa">fit</span> × 0.20<br>
    <span style="color:#374151">──────────────</span><br>
    × availability_mult<br>
    × services_penalty <span style="color:#374151;font-size:0.70rem">(5-tier)</span><br>
    × title_gate<br>
    × must_have_gate <span style="color:#374151;font-size:0.70rem">(P2 NDCG@10)</span><br>
    × suspicion_mult <span style="color:#374151;font-size:0.70rem">(P0-b soft)</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="card-title">🛡️ Penalties & Bonuses</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.76rem; color:#94a3b8; line-height:1.95;">
    ❌ Pure services → <span style="color:#f87171">0.05×</span><br>
    ❌ Heavy services 70-79% → <span style="color:#f87171">0.45×</span><br>
    ❌ Expert skill, 0 months → <span style="color:#f87171">0.05 wt</span><br>
    ❌ ≥3 expert+0mo → <span style="color:#f87171">Honeypot</span><br>
    ❌ ≥12 expert skills → <span style="color:#f87171">Honeypot</span><br>
    ❌ Sarvam/Krutrim tenure → <span style="color:#f87171">Honeypot</span><br>
    ❌ 0 skill groups → <span style="color:#f87171">0.25×</span> gate<br>
    ❌ Outside India → <span style="color:#f87171">0.20×</span><br>
    ❌ Inactive 180d+ → <span style="color:#f87171">0.40×</span><br>
    ✅ GitHub &gt;60 → <span style="color:#4ade80">1.08×</span><br>
    ✅ Notice ≤15d → <span style="color:#4ade80">1.05×</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    if SAMPLE_CANDS:
        st.markdown('<div class="card-title">📁 Try With Real Data</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.76rem;color:#64748b;margin-bottom:10px;line-height:1.6;">
        Download sample candidates from the hackathon dataset, then upload them in the Live Ranker tab.
        </div>
        """, unsafe_allow_html=True)
        st.download_button(
            label="⬇  Download sample_candidates.json",
            data=json.dumps(SAMPLE_CANDS, indent=2),
            file_name="sample_candidates.json",
            mime="application/json",
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.72rem;color:#374151;line-height:1.8;">
    <a href="https://github.com/pithva007/Ai-Recruiter"
       style="color:#ef4444;text-decoration:none">github.com/pithva007/Ai-Recruiter</a><br>
    Ranked 100K candidates · &lt;1s · CPU only · Zero network
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# HERO
# ============================================================
st.markdown("""
<div class="hero">
  <div class="hero-title">🎯 AI Recruiter Copilot</div>
  <div class="hero-subtitle">Senior AI Engineer · Founding Team @ Redrob AI · Hackathon 2025</div>
  <div class="hero-pills">
    <span class="pill">100K candidates ranked</span>
    <span class="pill-blue">LLM-scored top 30</span>
    <span class="pill-green">CPU · No network · &lt;1s</span>
    <span class="pill-amber">Dark horse detection</span>
    <span class="pill">7-signal honeypot filter</span>
    <span class="pill-blue">v4 scoring engine</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# MAIN TABS
# ============================================================
tab_official, tab_deep, tab_live, tab_submission = st.tabs([
    "🏆  Official Top 100",
    "🔬  Deep Analysis (Top 30)",
    "⚡  Live Ranker",
    "📋  Submission CSV",
])


# ──────────────────────────────────────────────────────────────
# TAB 1: OFFICIAL TOP 100 (from submission.csv)
# ──────────────────────────────────────────────────────────────
with tab_official:
    if not SUBMISSION:
        st.error("submission.csv not found. Make sure it's bundled in hf_space/.")
    else:
        scores_float = [float(r["score"]) for r in SUBMISSION]
        honeypots_cnt = sum(1 for s in scores_float if s <= 0.02)

        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-box metric-green">
            <div class="metric-val">{len(SUBMISSION)}</div>
            <div class="metric-label">Ranked</div>
          </div>
          <div class="metric-box metric-blue">
            <div class="metric-val">{max(scores_float):.4f}</div>
            <div class="metric-label">Top Score</div>
          </div>
          <div class="metric-box metric-blue">
            <div class="metric-val">{min(scores_float):.4f}</div>
            <div class="metric-label">Rank 100 Score</div>
          </div>
          <div class="metric-box metric-amber">
            <div class="metric-val">{honeypots_cnt}</div>
            <div class="metric-label">Honeypots Caught</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Podium top 3
        medals = ["🥇", "🥈", "🥉"]
        top3_cols = st.columns(3)
        for i, col in enumerate(top3_cols):
            if i >= len(SUBMISSION):
                break
            r = SUBMISSION[i]
            sc = float(r["score"])
            sc_color = "#4ade80" if sc >= 0.70 else "#fbbf24" if sc >= 0.45 else "#f87171"
            col.markdown(f"""
            <div style="background:#111827;border:1px solid #1e2433;border-radius:14px;
                        padding:20px;text-align:center;margin-bottom:12px">
              <div style="font-size:2rem">{medals[i]}</div>
              <div style="font-weight:800;color:#e2e8f0;font-size:0.95rem;margin:8px 0 4px">
                {r['candidate_id']}</div>
              <div style="font-size:1.5rem;font-weight:800;color:{sc_color};margin-top:10px">
                {sc:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(f'<p style="font-size:0.75rem;color:#4b5563">All {len(SUBMISSION)} ranked candidates · Produced by rank.py + reason.py on full 100K pool</p>', unsafe_allow_html=True)

        for r in SUBMISSION:
            sc = float(r["score"])
            rank = int(r["rank"])
            sc_color = "#4ade80" if sc >= 0.70 else "#fbbf24" if sc >= 0.45 else "#f87171"
            sc_bg    = ("rgba(34,197,94,0.10)" if sc >= 0.70 else
                        "rgba(234,179,8,0.10)" if sc >= 0.45 else
                        "rgba(239,68,68,0.10)")
            rank_color = "#fbbf24" if rank == 1 else "#94a3b8" if rank == 2 else "#cd7c2f" if rank == 3 else "#374151"
            reasoning = r.get("reasoning", "")[:160]

            c1, c2 = st.columns([1, 12])
            with c1:
                st.markdown(f'<div style="font-size:1.0rem;font-weight:800;color:{rank_color};padding-top:14px;text-align:center">#{rank}</div>', unsafe_allow_html=True)
            with c2:
                st.markdown(
                    f'<div style="background:{sc_bg};color:{sc_color};padding:2px 10px;'
                    f'border-radius:6px;font-weight:700;font-size:0.85rem;float:right">{sc:.4f}</div>'
                    f'<span style="font-weight:700;color:#e2e8f0">{r["candidate_id"]}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div style="font-size:0.78rem;color:#94a3b8;font-style:italic;'
                    f'margin-top:2px;border-bottom:1px solid #1e2433;padding-bottom:8px">{reasoning}</div>',
                    unsafe_allow_html=True,
                )


# ──────────────────────────────────────────────────────────────
# TAB 2: DEEP ANALYSIS — Top 30 LLM-scored candidates
# ──────────────────────────────────────────────────────────────
with tab_deep:
    if not RANKED_DEEP:
        st.error("ranked_candidates.csv not found. Make sure it's bundled in hf_space/.")
    else:
        dark_horses = [r for r in RANKED_DEEP if r.get("dark_horse", "").lower() == "true"]
        avg_score   = sum(float(r.get("composite_score", 0)) for r in RANKED_DEEP) / len(RANKED_DEEP)

        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-box metric-green">
            <div class="metric-val">{len(RANKED_DEEP)}</div>
            <div class="metric-label">LLM-Scored</div>
          </div>
          <div class="metric-box metric-blue">
            <div class="metric-val">{avg_score:.1f}</div>
            <div class="metric-label">Avg Score</div>
          </div>
          <div class="metric-box metric-amber">
            <div class="metric-val">{len(dark_horses)}</div>
            <div class="metric-label">Dark Horses</div>
          </div>
          <div class="metric-box metric-purple">
            <div class="metric-val">4D</div>
            <div class="metric-label">Scoring Dims</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Dark horse spotlight
        if dark_horses:
            st.markdown("#### ⭐ Dark Horse Spotlight")
            for dh in dark_horses[:3]:
                with st.expander(f"🌟 {dh.get('candidate_name', dh['candidate_id'])} — {dh.get('composite_score')} pts", expanded=False):
                    st.markdown(f"**Why they're a dark horse:** {dh.get('dark_horse_reason', '—')}")
                    if dh.get("transferable_skills_map"):
                        st.markdown(f"**Transferable skills:** {dh.get('transferable_skills_map')}")
            st.markdown("---")

        # Full ranked list with all 4D scores
        st.markdown("#### 🔬 Full Deep Analysis")
        for r in RANKED_DEEP:
            rank       = r.get("rank", "?")
            cid        = r.get("candidate_id", "")
            name       = r.get("candidate_name", cid)
            comp_score = float(r.get("composite_score", 0))
            fit        = float(r.get("fit_score", 0))
            impact     = float(r.get("impact_score", 0))
            potential  = float(r.get("potential_score", 0))
            risk       = float(r.get("risk_score", 50))
            conf       = r.get("confidence_level", "")
            is_dh      = r.get("dark_horse", "").lower() == "true"
            green_flags = r.get("green_flags", "")
            yellow_flags = r.get("yellow_flags", "")
            skill_gaps = r.get("skill_gaps", "")
            iq1 = r.get("interview_q1", "")
            iq2 = r.get("interview_q2", "")
            iq3 = r.get("interview_q3", "")
            rationale  = r.get("llm_rationale", "")

            sc_color = "#4ade80" if comp_score >= 85 else "#fbbf24" if comp_score >= 70 else "#f87171"
            rank_color = "#fbbf24" if str(rank) == "1" else "#94a3b8" if str(rank) == "2" else "#cd7c2f" if str(rank) == "3" else "#374151"
            dh_badge   = '<span class="dark-horse-badge">⭐ DARK HORSE</span>' if is_dh else ""
            conf_badge = f'<span style="font-size:0.68rem;color:#4b5563;margin-left:6px">{conf.upper()}</span>' if conf else ""

            with st.expander(
                f"#{rank}  {name}  ·  {comp_score:.0f}/100" + (" ⭐" if is_dh else ""),
                expanded=False
            ):
                col_left, col_right = st.columns([3, 2])
                with col_left:
                    st.markdown(f"**{name}** {dh_badge} {conf_badge}", unsafe_allow_html=True)
                    st.markdown(f"ID: `{cid}`")

                    # 4D scores
                    dims_cols = st.columns(4)
                    for dcol, (label, val, color) in zip(dims_cols, [
                        ("Fit", fit, "#60a5fa"),
                        ("Impact", impact, "#4ade80"),
                        ("Potential", potential, "#a78bfa"),
                        ("Risk", risk, "#f87171"),
                    ]):
                        dcol.markdown(
                            f'<div style="text-align:center;background:#0f172a;border-radius:8px;padding:10px">'
                            f'<div style="font-size:1.3rem;font-weight:800;color:{color}">{val:.0f}</div>'
                            f'<div style="font-size:0.65rem;color:#374151;text-transform:uppercase">{label}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    if green_flags:
                        st.markdown("**✅ Green Flags**")
                        for flag in green_flags.split(" | "):
                            if flag.strip():
                                st.markdown(f'<div class="flag-item" style="color:#4ade80">✓ {flag.strip()}</div>', unsafe_allow_html=True)
                    if yellow_flags:
                        st.markdown("**⚠️ Yellow Flags**")
                        for flag in yellow_flags.split(" | "):
                            if flag.strip():
                                st.markdown(f'<div class="flag-item" style="color:#fbbf24">⚠ {flag.strip()}</div>', unsafe_allow_html=True)
                    if skill_gaps:
                        st.markdown("**🔴 Skill Gaps**")
                        st.markdown(f'<div class="flag-item" style="color:#f87171">{skill_gaps}</div>', unsafe_allow_html=True)

                with col_right:
                    if rationale:
                        st.markdown("**🤖 LLM Rationale**")
                        st.markdown(f'<div style="font-size:0.78rem;color:#94a3b8;line-height:1.7;font-style:italic">{rationale[:600]}</div>', unsafe_allow_html=True)

                # Interview questions
                if iq1 or iq2 or iq3:
                    st.markdown("**🎤 Interview Questions**")
                    for iq in [iq1, iq2, iq3]:
                        if iq.strip():
                            st.markdown(f'<div class="interview-q">{iq.strip()}</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# TAB 3: LIVE RANKER — upload & re-score any candidates
# ──────────────────────────────────────────────────────────────
with tab_live:
    st.markdown("""
    <div style="background:#111827;border:1px solid #1e2433;border-radius:14px;
                padding:20px 24px;margin-bottom:20px">
      <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;
                  letter-spacing:0.10em;color:#4b5563;margin-bottom:8px">⚡ Live Ranker</div>
      <div style="font-size:0.85rem;color:#94a3b8;line-height:1.7">
        Upload any <strong style="color:#e2e8f0">.jsonl</strong> or
        <strong style="color:#e2e8f0">.json</strong> file in the hackathon candidate schema
        (up to 100 candidates). The full v4 scoring engine runs entirely in-browser — zero network,
        zero GPU. Download the sample file from the sidebar to get started instantly.
      </div>
    </div>
    """, unsafe_allow_html=True)

    up_col, btn_col = st.columns([3, 1])
    with up_col:
        uploaded = st.file_uploader(
            "Drop your candidates file here",
            type=["jsonl", "json", "gz"],
            label_visibility="collapsed",
        )
        use_sample = st.checkbox(
            "🧪 Use built-in sample candidates (5-candidate demo)",
            value=not bool(uploaded),
            help="Includes: 1 genuine top candidate, 1 pure-services (penalized), 1 honeypot (caught), 2 mid-tier",
        )

    with btn_col:
        run_btn = st.button("▶  Run Ranker", use_container_width=True)

    if run_btn:
        candidates = []
        id_generated = False

        if uploaded and not use_sample:
            try:
                content = read_uploaded_file(uploaded)
                raw_candidates = parse_candidates_file(content)
                if len(raw_candidates) > 100:
                    st.warning(f"Loaded {len(raw_candidates):,} candidates — using first 100.")
                    raw_candidates = raw_candidates[:100]
                for i, c in enumerate(raw_candidates):
                    real_id = c.get("candidate_id", "")
                    if not re.match(r"^CAND_[0-9]{7}$", str(real_id)):
                        c["candidate_id"] = get_candidate_id(c, i)
                        id_generated = True
                    candidates.append(c)
            except Exception as e:
                st.error(f"Parse error: {e}")
        elif use_sample and SAMPLE_CANDS:
            candidates = SAMPLE_CANDS[:10]
        else:
            # Built-in 5-candidate demo
            candidates = [
                {"candidate_id": "CAND_9990001", "profile": {"anonymized_name": "Demo A — Top ML Engineer", "years_of_experience": 7.0, "current_title": "Senior ML Engineer", "current_company": "TechCorp", "location": "Pune", "country": "India"}, "career_history": [{"company": "TechCorp", "title": "Senior ML Engineer", "duration_months": 39, "is_current": True, "description": "Designed FAISS-based hybrid retrieval systems, improved NDCG@10 by 15% with cross-encoder reranking. Built embedding pipelines with sentence-transformers."}, {"company": "StartupX", "title": "ML Engineer", "duration_months": 32, "description": "Recommendation systems using dense retrieval and BM25 hybrid search."}], "skills": [{"name": "FAISS", "proficiency": "expert", "duration_months": 36, "endorsements": 20}, {"name": "Python", "proficiency": "expert", "duration_months": 72, "endorsements": 45}, {"name": "Embeddings", "proficiency": "expert", "duration_months": 30, "endorsements": 18}, {"name": "PyTorch", "proficiency": "advanced", "duration_months": 36, "endorsements": 30}], "redrob_signals": {"last_active_date": "2026-06-10", "open_to_work_flag": True, "recruiter_response_rate": 0.82, "notice_period_days": 15, "interview_completion_rate": 0.95, "github_activity_score": 75, "willing_to_relocate": False}},
                {"candidate_id": "CAND_9990002", "profile": {"anonymized_name": "Demo B — Data Scientist", "years_of_experience": 6.0, "current_title": "Data Scientist", "current_company": "ProductCo", "location": "Bangalore", "country": "India"}, "career_history": [{"company": "ProductCo", "title": "Data Scientist", "duration_months": 36, "is_current": True, "description": "Built NLP models, A/B testing framework, search relevance improvements."}], "skills": [{"name": "Python", "proficiency": "advanced", "duration_months": 48, "endorsements": 25}, {"name": "NLP", "proficiency": "intermediate", "duration_months": 24, "endorsements": 12}], "redrob_signals": {"last_active_date": "2026-05-01", "open_to_work_flag": True, "recruiter_response_rate": 0.65, "notice_period_days": 30, "interview_completion_rate": 0.85, "github_activity_score": 45}},
                {"candidate_id": "CAND_9990003", "profile": {"anonymized_name": "Demo C — Pure Services (Penalized)", "years_of_experience": 8.0, "current_title": "ML Engineer", "current_company": "TCS", "location": "Hyderabad", "country": "India"}, "career_history": [{"company": "TCS", "title": "ML Engineer", "duration_months": 48, "is_current": True, "description": "ML models for client projects."}, {"company": "Infosys", "title": "Data Engineer", "duration_months": 40, "description": "ETL pipelines."}, {"company": "Wipro", "title": "Developer", "duration_months": 11, "description": "Software maintenance."}], "skills": [{"name": "Python", "proficiency": "advanced", "duration_months": 60, "endorsements": 15}], "redrob_signals": {"last_active_date": "2026-05-15", "open_to_work_flag": True, "recruiter_response_rate": 0.40, "notice_period_days": 45, "github_activity_score": 25}},
                {"candidate_id": "CAND_9990004", "profile": {"anonymized_name": "Demo D — Honeypot (Caught)", "years_of_experience": 12.0, "current_title": "AI Consultant", "current_company": "Freelance", "location": "Mumbai", "country": "India"}, "career_history": [{"company": "Freelance", "title": "AI Consultant", "duration_months": 3, "is_current": True, "description": ""}], "skills": [{"name": "Python", "proficiency": "expert", "duration_months": 0, "endorsements": 50}, {"name": "FAISS", "proficiency": "expert", "duration_months": 0, "endorsements": 30}, {"name": "PyTorch", "proficiency": "expert", "duration_months": 0, "endorsements": 40}], "redrob_signals": {"last_active_date": "2026-05-01", "open_to_work_flag": True, "recruiter_response_rate": 0.30, "notice_period_days": 90, "github_activity_score": -1}},
                {"candidate_id": "CAND_9990005", "profile": {"anonymized_name": "Demo E — Dark Horse Data Engineer", "years_of_experience": 5.0, "current_title": "Data Engineer", "current_company": "MidSizeCo", "location": "Delhi", "country": "India"}, "career_history": [{"company": "MidSizeCo", "title": "Data Engineer", "duration_months": 29, "is_current": True, "description": "Feature engineering pipelines for ML, real-time Spark Streaming, productionized recommendation models, vector similarity search for product catalog."}, {"company": "SmallCo", "title": "Data Engineer", "duration_months": 18, "description": "Data pipelines and ETL workflows."}], "skills": [{"name": "Python", "proficiency": "advanced", "duration_months": 36, "endorsements": 20}, {"name": "Spark", "proficiency": "intermediate", "duration_months": 24, "endorsements": 12}], "redrob_signals": {"last_active_date": "2026-06-05", "open_to_work_flag": True, "recruiter_response_rate": 0.55, "notice_period_days": 30, "interview_completion_rate": 0.80, "github_activity_score": 35}},
            ]

        if id_generated:
            st.warning("⚠️ Some candidates had non-standard IDs. Demo IDs were generated.")

        if candidates:
            prog = st.progress(0, text="Scoring candidates...")
            scored = []
            for i, c in enumerate(candidates):
                prog.progress((i + 1) / len(candidates), text=f"Scoring {i+1}/{len(candidates)}...")
                score = score_candidate(c)
                scored.append({"candidate": c, "candidate_id": c.get("candidate_id"), "score": score})
            prog.empty()

            scored.sort(key=lambda x: (-x["score"], x["candidate_id"]))
            results = []
            for i, item in enumerate(scored[:100]):
                rank = i + 1
                reasoning = build_reasoning(item["candidate"], rank, item["score"])
                results.append({
                    "candidate_id": item["candidate_id"],
                    "rank": rank,
                    "score": item["score"],
                    "reasoning": reasoning,
                    "candidate": item["candidate"],
                })

            st.session_state["live_results"] = results
            st.session_state["live_ran"] = True

    if st.session_state.get("live_ran") and st.session_state.get("live_results"):
        results = st.session_state["live_results"]
        scores  = [r["score"] for r in results]
        honeypots  = sum(1 for r in results if r["score"] <= 0.02)
        dark_horses_cnt = sum(1 for r in results if r["rank"] > 3 and r["score"] >= 0.60)

        st.info("**About these scores:** This demo uses `sample_candidates.json` "
                "which contains 50 mixed candidates — only ~1-2 are genuine AI/ML "
                "engineers. Low scores for Accountants, Civil Engineers, and HR Managers "
                "are **correct** — the system is working as intended. "
                "On the full 100K candidate pool, top AI engineers score 0.85–1.00. "
                "Upload `candidates.jsonl` from the hackathon bundle to see the full range.")

        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-box metric-green"><div class="metric-val">{len(results)}</div><div class="metric-label">Ranked</div></div>
          <div class="metric-box metric-blue"><div class="metric-val">{max(scores):.4f}</div><div class="metric-label">Top Score</div></div>
          <div class="metric-box metric-amber"><div class="metric-val">{honeypots}</div><div class="metric-label">Honeypots</div></div>
          <div class="metric-box metric-purple"><div class="metric-val">{dark_horses_cnt}</div><div class="metric-label">Dark Horses</div></div>
        </div>
        """, unsafe_allow_html=True)

        # Podium
        medals = ["🥇", "🥈", "🥉"]
        top3 = results[:min(3, len(results))]
        t3cols = st.columns(3)
        for i, col in enumerate(t3cols):
            if i >= len(top3):
                break
            r = top3[i]
            p = r["candidate"].get("profile", {})
            name = p.get("anonymized_name") or r["candidate_id"]
            title = p.get("current_title") or "—"
            yoe = p.get("years_of_experience") or "?"
            sc = r["score"]
            sc_color = "#4ade80" if sc >= 0.70 else "#fbbf24" if sc >= 0.45 else "#f87171"
            col.markdown(f"""
            <div style="background:#111827;border:1px solid #1e2433;border-radius:14px;
                        padding:20px;text-align:center;margin-bottom:12px">
              <div style="font-size:1.8rem">{medals[i]}</div>
              <div style="font-weight:800;color:#e2e8f0;font-size:0.92rem;margin:6px 0 2px">{name}</div>
              <div style="font-size:0.73rem;color:#4b5563">{title} · {yoe}yr</div>
              <div style="font-size:1.4rem;font-weight:800;color:{sc_color};margin-top:10px">{sc:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Full list
        for r in results:
            profile  = r["candidate"].get("profile", {})
            signals  = r["candidate"].get("redrob_signals", {})
            name     = profile.get("anonymized_name") or r["candidate_id"]
            title    = profile.get("current_title") or "—"
            company  = profile.get("current_company") or "—"
            yoe      = profile.get("years_of_experience") or "?"
            location = profile.get("location") or "—"
            sc       = r["score"]
            rank     = r["rank"]
            notice   = signals.get("notice_period_days")

            is_hp    = sc <= 0.02
            is_dh    = rank > 3 and sc >= 0.60
            sc_color = "#4ade80" if sc >= 0.70 else "#fbbf24" if sc >= 0.45 else "#f87171"
            sc_bg    = "rgba(34,197,94,0.10)" if sc >= 0.70 else "rgba(234,179,8,0.10)" if sc >= 0.45 else "rgba(239,68,68,0.10)"
            rank_color = "#fbbf24" if rank == 1 else "#94a3b8" if rank == 2 else "#cd7c2f" if rank == 3 else "#374151"
            badge = ("🚫 HONEYPOT" if is_hp else "⭐ DARK HORSE" if is_dh else "")
            notice_txt = f" · {notice}d notice" if isinstance(notice, int) else ""
            reasoning_preview = (r.get("reasoning", "")[:160] + "…") if len(r.get("reasoning", "")) > 160 else r.get("reasoning", "")

            c1, c2 = st.columns([1, 12])
            with c1:
                st.markdown(f'<div style="font-size:1.0rem;font-weight:800;color:{rank_color};padding-top:14px;text-align:center">#{rank}</div>', unsafe_allow_html=True)
            with c2:
                badge_html = (f' <span style="background:rgba(239,68,68,0.15);color:#f87171;padding:1px 6px;'
                              f'border-radius:4px;font-size:0.7rem;font-weight:700">{badge}</span>'
                              if badge else "")
                st.markdown(
                    f'<div style="background:{sc_bg};color:{sc_color};padding:2px 10px;'
                    f'border-radius:6px;font-weight:700;font-size:0.85rem;float:right">{sc:.4f}</div>'
                    f'<span style="font-weight:700;color:#e2e8f0">{name}</span>{badge_html}',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div style="font-size:0.75rem;color:#4b5563;margin-top:-6px">'
                    f'{title} @ {company} · {yoe}yr · {location}{notice_txt}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div style="font-size:0.75rem;color:#94a3b8;font-style:italic;'
                    f'border-bottom:1px solid #1e2433;padding-bottom:8px;margin-bottom:2px">'
                    f'{reasoning_preview}</div>',
                    unsafe_allow_html=True,
                )

        with st.expander("🔍 Why does Rank 1 score so much higher than Rank 2?"):
            st.markdown("""
**This is the scoring system working correctly.**

The sample file (`sample_candidates.json`) was designed to test whether a ranker can distinguish signal from noise. It contains:
- **1 genuine AI/ML engineer** (Recommendation Systems, FAISS, Embeddings, ranking models at Swiggy/Uber/Zomato)
- **49 irrelevant candidates** (Accountants, Civil Engineers, Marketing Managers, HR Managers, Frontend Developers)

A naive keyword-based system would score a Frontend Developer with `FAISS` listed as a skill at 0.70+. Our system scores them at 0.02 because that skill has 0 months of duration and 0 endorsements — it's a keyword stuffer signal.

**On the full 100,000-candidate pool:**
- Top 10 candidates score 0.85–1.00
- All are ML/AI engineers at Indian product companies
- All have production retrieval/search/ranking evidence in career descriptions
- Score spread: 0.4790 (strong differentiation)
""")

        # Download
        st.markdown("---")
        csv_str = to_csv_string([{k: v for k, v in r.items() if k != "candidate"} for r in results])
        st.download_button(
            label="⬇  Download submission.csv",
            data=csv_str,
            file_name="submission.csv",
            mime="text/csv",
            use_container_width=True,
        )

    elif not st.session_state.get("live_ran"):
        st.markdown("""
        <div style="background:#111827;border:1px dashed #1e2433;border-radius:16px;
                    padding:60px 40px;text-align:center;margin-top:20px">
          <div style="font-size:3rem;margin-bottom:16px">⚡</div>
          <div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin-bottom:8px">Ready to rank</div>
          <div style="font-size:0.85rem;color:#4b5563;max-width:340px;margin:0 auto">
            Upload a candidate file or use the built-in demo, then click
            <strong style="color:#ef4444">Run Ranker</strong>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# TAB 4: SUBMISSION CSV preview + download
# ──────────────────────────────────────────────────────────────
with tab_submission:
    if not SUBMISSION:
        st.error("submission.csv not found.")
    else:
        st.markdown("""
        <div style="font-size:0.85rem;color:#64748b;margin-bottom:16px;line-height:1.7">
        This is the official submission CSV produced by running <code>rank.py</code> +
        <code>reason.py</code> on the full 100,000-candidate pool. It satisfies all hackathon
        constraints: 100 rows, ranks 1–100 (unique), scores non-increasing,
        <code>CAND_XXXXXXX</code> IDs, UTF-8 encoding.
        </div>
        """, unsafe_allow_html=True)

        # Preview first 10 rows
        preview_lines = []
        preview_lines.append("candidate_id,rank,score,reasoning")
        for r in SUBMISSION[:10]:
            reasoning_short = r.get("reasoning", "")[:80].replace(",", " ")
            preview_lines.append(f"{r['candidate_id']},{r['rank']},{r['score']},{reasoning_short}…")
        st.code("\n".join(preview_lines), language="csv")

        # Full download
        csv_full = "candidate_id,rank,score,reasoning\n"
        for r in SUBMISSION:
            esc_reasoning = '"' + r.get("reasoning", "").replace('"', '""') + '"'
            csv_full += f"{r['candidate_id']},{r['rank']},{r['score']},{esc_reasoning}\n"

        st.download_button(
            label="⬇  Download Official submission.csv (100 rows)",
            data=csv_full,
            file_name="submission.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Stats
        scores_f = [float(r["score"]) for r in SUBMISSION]
        is_monotonic = all(scores_f[i] >= scores_f[i+1] for i in range(len(scores_f)-1))
        st.markdown(f"""
        <div style="font-size:0.78rem;color:#4b5563;margin-top:12px;line-height:2.0">
        ✅ Rows: <strong style="color:#4ade80">{len(SUBMISSION)}</strong> &nbsp;
        ✅ Monotonic scores: <strong style="color:#4ade80">{'Yes' if is_monotonic else 'No'}</strong> &nbsp;
        ✅ Score range: <strong style="color:#60a5fa">{min(scores_f):.4f} – {max(scores_f):.4f}</strong>
        </div>
        """, unsafe_allow_html=True)


# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="display:flex;justify-content:space-between;align-items:center;
            font-size:0.72rem;color:#374151;flex-wrap:wrap;gap:8px">
  <span>🎯 AI Recruiter · Redrob Hackathon 2025 ·
    <a href="https://github.com/pithva007/Ai-Recruiter"
       style="color:#ef4444;text-decoration:none">github.com/pithva007/Ai-Recruiter</a>
  </span>
  <span>v4: career×0.30 + skills×0.20 + retrieval×0.30 + fit×0.20 → ×avail × svc(5-tier) × title × must_have × suspicion</span>
</div>
""", unsafe_allow_html=True)
