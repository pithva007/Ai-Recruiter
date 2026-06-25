import streamlit as st
import json
import csv
import gzip as gz_lib
import io
import re
import math
from datetime import date, datetime

# ── session_state init ─────────────────────────────────────
if "results" not in st.session_state:
    st.session_state["results"] = None
if "ran" not in st.session_state:
    st.session_state["ran"] = False

# ============================================================
# SCORING ENGINE  (self-contained — no external imports)
# ============================================================

SERVICES_COMPANIES = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl technologies", "hcl tech",
    "tech mahindra", "mphasis", "hexaware", "l&t infotech", "ltimindtree"
}

MUST_HAVE_SKILLS = {
    "embeddings", "vector search", "vector database", "faiss", "pinecone",
    "weaviate", "qdrant", "milvus", "elasticsearch", "opensearch",
    "sentence transformers", "retrieval", "ranking", "recommendation",
    "python", "llm", "fine-tuning", "rag", "information retrieval",
    "ndcg", "map", "mrr", "bert", "transformers", "huggingface",
    "pytorch", "tensorflow", "bge", "e5", "hybrid search",
    "learning to rank", "xgboost", "reranking", "dense retrieval"
}

NICE_TO_HAVE_SKILLS = {
    "lora", "qlora", "peft", "langchain", "openai", "gemini",
    "llama", "mistral", "spark", "kafka", "airflow", "kubernetes",
    "docker", "mlflow", "weights & biases", "a/b testing",
    "distributed systems", "inference optimization"
}

RETRIEVAL_KEYWORDS = {
    "retrieval", "search", "ranking", "recommendation", "vector",
    "embedding", "similarity", "index", "faiss", "elasticsearch",
    "recommend", "ranker", "rerank", "recall", "precision", "ndcg",
    "bm25", "dense", "sparse", "hybrid", "semantic search"
}

REFERENCE_DATE = date(2026, 6, 1)

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
        j.get("duration_months", 0) or 0
        for j in career_history
        if any(s in (j.get("company") or "").lower() for s in SERVICES_COMPANIES)
    )
    ratio = services / total
    has_escape = any(
        not any(s in (j.get("company") or "").lower() for s in SERVICES_COMPANIES)
        for j in career_history
    )
    if ratio >= 0.80 and not has_escape:
        return 0.05
    if ratio >= 0.80:
        return 0.40
    if ratio >= 0.50:
        return 0.75
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
        if duration == 0 and proficiency in ["expert", "advanced"]:
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
    if years is None or years <= 0:
        return 0.40
    if years < 4.0:
        return 0.35
    if years < 5.0:
        return 0.72
    if years < 5.5:
        return 0.88
    if 5.5 <= years <= 9.0:
        return 1.0
    if 9.0 < years <= 11.0:
        return 0.95
    return 0.85


def _title_chaser_penalty(career_history):
    if not career_history or len(career_history) < 3:
        return 1.0
    past = [j for j in career_history if not j.get("is_current", False)]
    if not past:
        return 1.0
    short = sum(1 for j in past if (j.get("duration_months") or 99) < 18)
    ratio = short / len(past)
    if ratio >= 0.70:
        return 0.60
    if ratio >= 0.50:
        return 0.80
    return 1.0


def _is_honeypot(candidate):
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    zero_dur_expert = sum(
        1 for s in skills
        if (s.get("duration_months") or 0) == 0
        and (s.get("proficiency") or "") in ["expert", "advanced"]
    )
    if zero_dur_expert >= 5:
        return True
    stated_yoe = profile.get("years_of_experience") or 0
    career_months = sum(j.get("duration_months") or 0 for j in career)
    if stated_yoe > 2 and career_months < 6:
        return True
    return False


COMPLETELY_IRRELEVANT_TITLES = {
    # Non-technical roles
    "accountant", "marketing manager", "operations manager",
    "hr manager", "human resources", "graphic designer",
    "content writer", "customer support", "business analyst",
    "project manager", "sales", "finance", "administrative",
    
    # Wrong engineering domains  
    "civil engineer", "mechanical engineer", "electrical engineer",
    "hardware engineer", "manufacturing engineer",
    
    # Non-ML tech roles that lack any ML path
    ".net developer", "mobile developer", "frontend engineer",
    "full stack developer", "devops engineer", "qa engineer",
    "java developer", "web developer", "ui developer",
}

def compute_title_relevance_gate(profile: dict) -> float:
    """
    Hard gate for completely irrelevant titles.
    Returns a multiplier: 0.05 for irrelevant, 1.0 otherwise.
    
    This prevents location/availability bonuses from inflating
    candidates who are fundamentally wrong for the role.
    """
    title = (profile.get("current_title") or "").lower()
    
    # Check for completely irrelevant title keywords
    for irrelevant in COMPLETELY_IRRELEVANT_TITLES:
        if irrelevant in title:
            return 0.05  # near-disqualification
    
    # Check for ML/AI/data relevant title keywords — these get full score
    RELEVANT_TITLE_KEYWORDS = {
        "ml", "machine learning", "ai ", "artificial intelligence",
        "data scientist", "data science", "nlp", "deep learning",
        "recommendation", "search engineer", "retrieval",
        "applied scientist", "research scientist", "research engineer",
        "software engineer", "software developer", "backend engineer",
        "data engineer", "platform engineer", "infrastructure engineer",
        "cloud engineer", "sre", "mlops", "llm", "generative"
    }
    
    for relevant in RELEVANT_TITLE_KEYWORDS:
        if relevant in title:
            return 1.0
    
    # Unknown title — neutral, don't penalize
    return 0.85


def score_candidate(candidate):
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    if _is_honeypot(candidate):
        return 0.01
    years = profile.get("years_of_experience") or 0
    exp_mult = _experience_band_multiplier(years)
    title_pen = _title_chaser_penalty(career)
    services_pen = compute_services_penalty(career)
    skill_score = compute_skill_trust_score(skills)
    retrieval_score = detect_retrieval_experience(career)
    avail_mult = compute_availability_multiplier(signals, profile)
    career_score = exp_mult * title_pen
    base = (
        career_score    * 0.30 +
        skill_score     * 0.20 +
        retrieval_score * 0.30 +
        0.50            * 0.20
    )
    title_gate = compute_title_relevance_gate(profile)
    final = base * avail_mult * services_pen * title_gate
    return round(min(max(final, 0.0), 1.0), 6)


# ============================================================
# FORMAT DETECTION & FILE PARSING  (self-contained)
# ============================================================

def read_uploaded_file(uploaded_file) -> str:
    """Read uploaded file, handling gzip automatically."""
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
    """
    Handles THREE formats:
    1. JSON array:  [{...}, {...}]           (sample_candidates.json)
    2. JSONL:       {...}\n{...}\n{...}      (candidates.jsonl)
    3. Single JSON: {...}                    (single candidate)
    """
    content = content.strip()

    if content.startswith("["):
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON array: {e}")

    if content.startswith("{"):
        candidates = []
        errors = []
        for i, line in enumerate(content.split("\n"), 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                candidates.append(obj)
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: {e}")

        if not candidates and errors:
            raise ValueError(f"No valid JSON objects found. First error: {errors[0]}")

        return candidates

    raise ValueError(
        f"Unrecognized format. File must start with '[' (JSON array) "
        f"or '{{' (JSONL). Got: {content[:30]!r}"
    )


def get_candidate_id(candidate: dict, index: int) -> str:
    """Get candidate ID, generating one if missing or invalid."""
    cid = candidate.get("candidate_id", "")
    if re.match(r"^CAND_[0-9]{7}$", str(cid)):
        return cid
    return f"CAND_{(index + 1):07d}"


# ============================================================
# RANKING ENGINE  (deterministic, CPU-only)
# ============================================================

def build_reasoning(candidate: dict, rank: int, score: float) -> str:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])

    title = profile.get("current_title") or "Professional"
    yoe = profile.get("years_of_experience") or 0
    rr = signals.get("recruiter_response_rate", 0.5)
    notice = signals.get("notice_period_days", 30)

    top_skills = [
        s.get("name") for s in skills
        if s.get("duration_months", 0) > 6
    ][:2]
    skills_str = ", ".join(top_skills) if top_skills else ""

    if _is_honeypot(candidate):
        reasoning = f"{title} with {yoe}yr; flagged as honeypot (impossible skill profile)."
    elif compute_services_penalty(career) <= 0.40:
        reasoning = f"{title} with {yoe}yr at services firms; penalized for pure-services career."
    elif rank <= 10:
        base = f"{title} with {yoe}yr; "
        if skills_str:
            base += f"strong {skills_str}; "
        base += f"response rate {rr:.2f}, notice {notice}d."
        reasoning = base
    elif rank <= 30:
        ref = f"{skills_str}; " if skills_str else ""
        reasoning = f"{title} with {yoe}yr; {ref}moderate engagement signals; notice {notice}d."
    elif rank <= 50:
        reasoning = f"{title} with {yoe}yr; some {skills_str}; mid-ranked on availability and depth."
    else:
        adj = f"adjacent background with {skills_str}; " if skills_str else "adjacent background; "
        reasoning = f"{title} with {yoe}yr; {adj}included based on engagement signals."

    # Strip any HTML chars that could break rendering
    import html as html_lib
    reasoning = html_lib.escape(reasoning)
    return reasoning


def rank_candidates(candidates: list) -> list:
    scored = []
    for c in candidates:
        score = score_candidate(c)
        cid = get_candidate_id(c, len(scored))
        scored.append({"candidate": c, "candidate_id": cid, "score": score})

    scored.sort(key=lambda x: (-x["score"], x["candidate_id"]))

    results = []
    for i, item in enumerate(scored[:100]):
        rank = i + 1
        candidate = item["candidate"]
        reasoning = build_reasoning(candidate, rank, item["score"])
        results.append({
            "candidate_id": item["candidate_id"],
            "rank": rank,
            "score": item["score"],
            "reasoning": reasoning,
            "candidate": candidate
        })
    return results


def to_csv_string(results: list) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    for r in results:
        writer.writerow([r["candidate_id"], r["rank"], f"{r['score']:.4f}", r["reasoning"]])
    return output.getvalue()


def _get_builtin_candidates() -> list:
    return [
        {
            "candidate_id": "CAND_9990001",
            "profile": {
                "anonymized_name": "Demo A",
                "headline": "Senior ML Engineer | Search & Retrieval | 7yr",
                "summary": "ML engineer with 7 years building production retrieval and ranking systems. Strong background in embeddings, vector search, and deep learning.",
                "location": "Pune",
                "country": "India",
                "years_of_experience": 7.0,
                "current_title": "Senior ML Engineer",
                "current_company": "TechCorp",
                "current_company_size": "501-1000",
                "current_industry": "Technology"
            },
            "career_history": [
                {
                    "company": "TechCorp",
                    "title": "Senior ML Engineer",
                    "start_date": "2023-03-01",
                    "end_date": None,
                    "duration_months": 39,
                    "is_current": True,
                    "industry": "Technology",
                    "company_size": "501-1000",
                    "description": "Designed and deployed production retrieval systems using FAISS and dense embeddings. Built hybrid search pipelines combining BM25 and vector similarity. Improved NDCG@10 by 15% through reranking with cross-encoders."
                },
                {
                    "company": "StartupX",
                    "title": "ML Engineer",
                    "start_date": "2020-06-01",
                    "end_date": "2023-02-01",
                    "duration_months": 32,
                    "is_current": False,
                    "industry": "Technology",
                    "company_size": "51-200",
                    "description": "Built recommendation systems for content discovery. Implemented embedding-based similarity search using sentence-transformers. Worked on ranking models with learning-to-rank techniques."
                },
                {
                    "company": "DataCorp",
                    "title": "Data Scientist",
                    "start_date": "2019-01-01",
                    "end_date": "2020-05-01",
                    "duration_months": 16,
                    "is_current": False,
                    "industry": "Technology",
                    "company_size": "1001-5000",
                    "description": "Developed ML models for customer analytics. Built data pipelines and feature engineering workflows."
                }
            ],
            "skills": [
                {"name": "Python", "proficiency": "expert", "duration_months": 72, "endorsements": 45},
                {"name": "FAISS", "proficiency": "expert", "duration_months": 36, "endorsements": 20},
                {"name": "PyTorch", "proficiency": "advanced", "duration_months": 36, "endorsements": 30},
                {"name": "Embeddings", "proficiency": "expert", "duration_months": 30, "endorsements": 18},
                {"name": "Transformers", "proficiency": "advanced", "duration_months": 24, "endorsements": 15},
                {"name": "RAG", "proficiency": "intermediate", "duration_months": 12, "endorsements": 8},
                {"name": "Machine Learning", "proficiency": "expert", "duration_months": 60, "endorsements": 40},
                {"name": "Elasticsearch", "proficiency": "advanced", "duration_months": 18, "endorsements": 10}
            ],
            "redrob_signals": {
                "last_active_date": "2026-05-25",
                "open_to_work_flag": True,
                "recruiter_response_rate": 0.82,
                "notice_period_days": 15,
                "interview_completion_rate": 0.95,
                "github_activity_score": 75,
                "willing_to_relocate": False,
                "preferred_work_mode": "hybrid",
                "applications_submitted_30d": 3,
                "profile_completeness_score": 95,
                "skill_assessment_scores": {"Python": 92, "Machine Learning": 88, "PyTorch": 85}
            }
        },
        {
            "candidate_id": "CAND_9990002",
            "profile": {
                "anonymized_name": "Demo B",
                "headline": "Data Scientist | NLP | 6yr Experience",
                "summary": "Data scientist with strong NLP background and experience deploying ML models in production.",
                "location": "Bangalore",
                "country": "India",
                "years_of_experience": 6.0,
                "current_title": "Data Scientist",
                "current_company": "ProductCo",
                "current_company_size": "1001-5000",
                "current_industry": "Technology"
            },
            "career_history": [
                {
                    "company": "ProductCo",
                    "title": "Data Scientist",
                    "start_date": "2023-06-01",
                    "end_date": None,
                    "duration_months": 36,
                    "is_current": True,
                    "industry": "Technology",
                    "company_size": "1001-5000",
                    "description": "Built NLP models for text classification and entity extraction. Deployed ML models to production using A/B testing framework. Worked on search relevance and ranking improvements."
                },
                {
                    "company": "OtherCo",
                    "title": "Data Analyst",
                    "start_date": "2021-01-01",
                    "end_date": "2023-05-01",
                    "duration_months": 28,
                    "is_current": False,
                    "industry": "Finance",
                    "company_size": "501-1000",
                    "description": "Developed dashboards and analytical models. Built feature pipelines for ML models."
                },
                {
                    "company": "StartCo",
                    "title": "Junior Data Analyst",
                    "start_date": "2020-01-01",
                    "end_date": "2020-12-01",
                    "duration_months": 11,
                    "is_current": False,
                    "industry": "Technology",
                    "company_size": "11-50",
                    "description": "Data cleaning and exploratory analysis."
                }
            ],
            "skills": [
                {"name": "Python", "proficiency": "advanced", "duration_months": 48, "endorsements": 25},
                {"name": "Machine Learning", "proficiency": "advanced", "duration_months": 36, "endorsements": 20},
                {"name": "NLP", "proficiency": "intermediate", "duration_months": 24, "endorsements": 12},
                {"name": "TensorFlow", "proficiency": "intermediate", "duration_months": 18, "endorsements": 10},
                {"name": "SQL", "proficiency": "expert", "duration_months": 60, "endorsements": 30}
            ],
            "redrob_signals": {
                "last_active_date": "2026-04-01",
                "open_to_work_flag": True,
                "recruiter_response_rate": 0.65,
                "notice_period_days": 30,
                "interview_completion_rate": 0.85,
                "github_activity_score": 45,
                "willing_to_relocate": True,
                "preferred_work_mode": "hybrid",
                "applications_submitted_30d": 2,
                "profile_completeness_score": 88,
                "skill_assessment_scores": {"Python": 85, "Machine Learning": 78}
            }
        },
        {
            "candidate_id": "CAND_9990003",
            "profile": {
                "anonymized_name": "Demo C",
                "headline": "ML Engineer | 8yr | Deep Learning & AI",
                "summary": "Experienced ML engineer with strong background in deep learning and AI solutions.",
                "location": "Hyderabad",
                "country": "India",
                "years_of_experience": 8.0,
                "current_title": "ML Engineer",
                "current_company": "TCS",
                "current_company_size": "10001+",
                "current_industry": "IT Services"
            },
            "career_history": [
                {
                    "company": "TCS",
                    "title": "ML Engineer",
                    "start_date": "2022-06-01",
                    "end_date": None,
                    "duration_months": 48,
                    "is_current": True,
                    "industry": "IT Services",
                    "company_size": "10001+",
                    "description": "Built ML models for client projects using TensorFlow and PyTorch. Managed data pipelines and model deployment."
                },
                {
                    "company": "Infosys",
                    "title": "Data Engineer",
                    "start_date": "2019-01-01",
                    "end_date": "2022-05-01",
                    "duration_months": 40,
                    "is_current": False,
                    "industry": "IT Services",
                    "company_size": "10001+",
                    "description": "Developed ETL pipelines and data infrastructure. Worked with Spark and cloud platforms."
                },
                {
                    "company": "Wipro",
                    "title": "Junior Developer",
                    "start_date": "2018-01-01",
                    "end_date": "2018-12-01",
                    "duration_months": 11,
                    "is_current": False,
                    "industry": "IT Services",
                    "company_size": "10001+",
                    "description": "Software development and maintenance."
                }
            ],
            "skills": [
                {"name": "Python", "proficiency": "advanced", "duration_months": 60, "endorsements": 15},
                {"name": "Machine Learning", "proficiency": "advanced", "duration_months": 36, "endorsements": 10},
                {"name": "Deep Learning", "proficiency": "intermediate", "duration_months": 24, "endorsements": 8},
                {"name": "TensorFlow", "proficiency": "intermediate", "duration_months": 18, "endorsements": 5},
                {"name": "Docker", "proficiency": "intermediate", "duration_months": 24, "endorsements": 6}
            ],
            "redrob_signals": {
                "last_active_date": "2026-05-15",
                "open_to_work_flag": True,
                "recruiter_response_rate": 0.40,
                "notice_period_days": 45,
                "interview_completion_rate": 0.70,
                "github_activity_score": 25,
                "willing_to_relocate": True,
                "preferred_work_mode": "hybrid",
                "applications_submitted_30d": 1,
                "profile_completeness_score": 75,
                "skill_assessment_scores": {"Python": 80, "Machine Learning": 70}
            }
        },
        {
            "candidate_id": "CAND_9990004",
            "profile": {
                "anonymized_name": "Demo D",
                "headline": "AI/ML Expert | 12yr | All Major Frameworks",
                "summary": "Expert in all AI/ML technologies. Built numerous production systems.",
                "location": "Mumbai",
                "country": "India",
                "years_of_experience": 12.0,
                "current_title": "AI Consultant",
                "current_company": "Freelance",
                "current_company_size": "1-10",
                "current_industry": "Technology"
            },
            "career_history": [
                {
                    "company": "Freelance",
                    "title": "AI Consultant",
                    "start_date": "2026-03-01",
                    "end_date": None,
                    "duration_months": 3,
                    "is_current": True,
                    "industry": "Technology",
                    "company_size": "1-10",
                    "description": ""
                }
            ],
            "skills": [
                {"name": "Python", "proficiency": "expert", "duration_months": 0, "endorsements": 50},
                {"name": "FAISS", "proficiency": "expert", "duration_months": 0, "endorsements": 30},
                {"name": "PyTorch", "proficiency": "expert", "duration_months": 0, "endorsements": 40},
                {"name": "TensorFlow", "proficiency": "expert", "duration_months": 0, "endorsements": 35},
                {"name": "NLP", "proficiency": "expert", "duration_months": 0, "endorsements": 25},
                {"name": "LLM", "proficiency": "expert", "duration_months": 0, "endorsements": 20},
                {"name": "RAG", "proficiency": "expert", "duration_months": 0, "endorsements": 15},
                {"name": "Embeddings", "proficiency": "expert", "duration_months": 0, "endorsements": 20},
                {"name": "Transformers", "proficiency": "expert", "duration_months": 0, "endorsements": 18}
            ],
            "redrob_signals": {
                "last_active_date": "2026-05-01",
                "open_to_work_flag": True,
                "recruiter_response_rate": 0.30,
                "notice_period_days": 90,
                "interview_completion_rate": 0.50,
                "github_activity_score": -1,
                "willing_to_relocate": False,
                "preferred_work_mode": "remote",
                "applications_submitted_30d": 0,
                "profile_completeness_score": 95,
                "skill_assessment_scores": {}
            }
        },
        {
            "candidate_id": "CAND_9990005",
            "profile": {
                "anonymized_name": "Demo E",
                "headline": "Data Engineer | Spark, Airflow, Python",
                "summary": "Data engineer with 5 years building data infrastructure. Transitioning toward ML engineering.",
                "location": "Delhi",
                "country": "India",
                "years_of_experience": 5.0,
                "current_title": "Data Engineer",
                "current_company": "MidSizeCo",
                "current_company_size": "501-1000",
                "current_industry": "Technology"
            },
            "career_history": [
                {
                    "company": "MidSizeCo",
                    "title": "Data Engineer",
                    "start_date": "2024-01-01",
                    "end_date": None,
                    "duration_months": 29,
                    "is_current": True,
                    "industry": "Technology",
                    "company_size": "501-1000",
                    "description": "Built feature engineering pipelines for ML teams. Implemented real-time data processing with Spark Streaming. Worked closely with data scientists to productionize models."
                },
                {
                    "company": "SmallCo",
                    "title": "Data Engineer",
                    "start_date": "2022-06-01",
                    "end_date": "2023-12-01",
                    "duration_months": 18,
                    "is_current": False,
                    "industry": "Technology",
                    "company_size": "51-200",
                    "description": "Developed data pipelines and ETL workflows. Built dashboards and reporting infrastructure."
                },
                {
                    "company": "StartCo",
                    "title": "Junior Data Analyst",
                    "start_date": "2021-06-01",
                    "end_date": "2022-05-01",
                    "duration_months": 11,
                    "is_current": False,
                    "industry": "Technology",
                    "company_size": "11-50",
                    "description": "Data analysis and reporting."
                }
            ],
            "skills": [
                {"name": "Python", "proficiency": "advanced", "duration_months": 36, "endorsements": 20},
                {"name": "Spark", "proficiency": "intermediate", "duration_months": 24, "endorsements": 12},
                {"name": "SQL", "proficiency": "expert", "duration_months": 48, "endorsements": 25},
                {"name": "Airflow", "proficiency": "intermediate", "duration_months": 18, "endorsements": 8}
            ],
            "redrob_signals": {
                "last_active_date": "2026-05-20",
                "open_to_work_flag": True,
                "recruiter_response_rate": 0.55,
                "notice_period_days": 30,
                "interview_completion_rate": 0.80,
                "github_activity_score": 35,
                "willing_to_relocate": True,
                "preferred_work_mode": "hybrid",
                "applications_submitted_30d": 2,
                "profile_completeness_score": 82,
                "skill_assessment_scores": {"Python": 78, "SQL": 85}
            }
        }
    ]


# ============================================================
# PAGE CONFIG & CUSTOM CSS
# ============================================================

st.set_page_config(
    page_title="AI Recruiter · Redrob Hackathon",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
/* ── Global ── */
[data-testid="stAppViewContainer"] {
    background: #0f1117;
    color: #e2e8f0;
}
[data-testid="stHeader"] { background: transparent; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #1a1f35 0%, #0f1117 50%, #1a1235 100%);
    border: 1px solid #2d3748;
    border-radius: 16px;
    padding: 36px 40px 28px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(239,68,68,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.1rem;
    font-weight: 800;
    background: linear-gradient(90deg, #ef4444, #f97316, #eab308);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 6px;
    line-height: 1.2;
}
.hero-subtitle {
    color: #94a3b8;
    font-size: 0.95rem;
    margin: 0 0 18px;
}
.hero-pills {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}
.pill {
    background: rgba(239,68,68,0.12);
    border: 1px solid rgba(239,68,68,0.3);
    color: #fca5a5;
    padding: 4px 12px;
    border-radius: 99px;
    font-size: 0.78rem;
    font-weight: 600;
}
.pill-blue {
    background: rgba(59,130,246,0.12);
    border-color: rgba(59,130,246,0.3);
    color: #93c5fd;
}
.pill-green {
    background: rgba(34,197,94,0.12);
    border-color: rgba(34,197,94,0.3);
    color: #86efac;
}

/* ── Cards ── */
.card {
    background: #1a1f2e;
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.card-title {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
    margin-bottom: 12px;
}

/* ── Metric boxes ── */
.metric-row {
    display: flex;
    gap: 12px;
    margin-bottom: 20px;
}
.metric-box {
    flex: 1;
    background: #1a1f2e;
    border: 1px solid #2d3748;
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
}
.metric-val {
    font-size: 1.8rem;
    font-weight: 800;
    color: #e2e8f0;
    line-height: 1;
}
.metric-label {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 4px;
}
.metric-green .metric-val { color: #4ade80; }
.metric-blue .metric-val  { color: #60a5fa; }
.metric-amber .metric-val { color: #fbbf24; }

/* ── Score badge ── */
.score-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 6px;
    font-weight: 700;
    font-size: 0.85rem;
}
.score-high   { background: rgba(34,197,94,0.15);  color: #4ade80; }
.score-mid    { background: rgba(234,179,8,0.15);  color: #fbbf24; }
.score-low    { background: rgba(239,68,68,0.15);  color: #f87171; }

/* ── Rank table rows ── */
.rank-row {
    display: flex;
    align-items: flex-start;
    gap: 16px;
    padding: 14px 0;
    border-bottom: 1px solid #1e2433;
}
.rank-num {
    min-width: 36px;
    text-align: center;
    font-size: 1rem;
    font-weight: 800;
    color: #64748b;
    padding-top: 2px;
}
.rank-1  .rank-num { color: #fbbf24; }
.rank-2  .rank-num { color: #94a3b8; }
.rank-3  .rank-num { color: #cd7c2f; }
.rank-name { font-weight: 700; color: #e2e8f0; font-size: 0.95rem; }
.rank-meta { font-size: 0.78rem; color: #64748b; margin-top: 2px; }
.rank-reasoning {
    font-size: 0.78rem;
    color: #94a3b8;
    margin-top: 6px;
    line-height: 1.5;
    font-style: italic;
}
.dark-horse-badge {
    display: inline-block;
    background: rgba(234,179,8,0.15);
    border: 1px solid rgba(234,179,8,0.4);
    color: #fbbf24;
    padding: 1px 8px;
    border-radius: 4px;
    font-size: 0.68rem;
    font-weight: 700;
    margin-left: 8px;
    vertical-align: middle;
}
.honeypot-badge {
    display: inline-block;
    background: rgba(239,68,68,0.15);
    border: 1px solid rgba(239,68,68,0.4);
    color: #f87171;
    padding: 1px 8px;
    border-radius: 4px;
    font-size: 0.68rem;
    font-weight: 700;
    margin-left: 8px;
    vertical-align: middle;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: #1a1f2e !important;
    border: 2px dashed #2d3748 !important;
    border-radius: 12px !important;
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
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(239,68,68,0.35) !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #16a34a, #15803d) !important;
    color: white !important;
    border-radius: 10px !important;
    border: none !important;
    font-weight: 700 !important;
    width: 100% !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #1a1f2e;
    border-radius: 10px;
    padding: 4px;
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #64748b;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: #ef4444 !important;
    color: white !important;
}

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #ef4444, #f97316) !important;
}

/* ── Info / warning boxes ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-left-width: 3px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">🎯 AI Recruiter Copilot</div>
  <div class="hero-subtitle">Senior AI Engineer · Founding Team @ Redrob AI · Hackathon Demo</div>
  <div class="hero-pills">
    <span class="pill">100K candidate pool</span>
    <span class="pill-blue">Evidence-based scoring</span>
    <span class="pill-green">CPU · No network · &lt;5 min</span>
    <span class="pill">Dark horse detection</span>
    <span class="pill-blue">Honeypot filtering</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Layout ──────────────────────────────────────────────────
left, right = st.columns([1, 2], gap="large")

with left:
    st.markdown('<div class="card-title">📁 Upload Candidates</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.82rem; color:#64748b; margin-bottom:12px; line-height:1.6;">
    Upload a <strong style="color:#94a3b8">.jsonl</strong> or
    <strong style="color:#94a3b8">.json</strong> file from the hackathon bundle
    (up to 100 candidates). Accepts both JSONL and JSON array formats.
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop file here",
        type=["jsonl", "json", "gz"],
        label_visibility="collapsed"
    )

    use_sample = st.checkbox(
        "🧪 Use built-in 5-candidate demo",
        value=not bool(uploaded),
        help="Includes 1 services candidate (penalized), 1 honeypot (caught), and 3 genuine ML engineers"
    )

    st.markdown("---")

    # Scoring formula explainer
    st.markdown('<div class="card-title">⚖️ Scoring Formula</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.80rem; color:#94a3b8; line-height:1.9;">
    <span style="color:#60a5fa">career</span> × 0.30<br>
    <span style="color:#4ade80">skills_trust</span> × 0.20<br>
    <span style="color:#f97316">retrieval_exp</span> × 0.30<br>
    <span style="color:#a78bfa">fit</span> × 0.20<br>
    <span style="color:#64748b">─────────────────</span><br>
    × availability_multiplier<br>
    × services_penalty
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Key rules
    st.markdown('<div class="card-title">🛡️ Anti-Stuffer Rules</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.78rem; color:#94a3b8; line-height:1.9;">
    ❌ Pure services career → <span style="color:#f87171">0.05×</span><br>
    ❌ Expert skill, 0 months → <span style="color:#f87171">0.05 weight</span><br>
    ❌ Outside India → <span style="color:#f87171">0.20×</span><br>
    ❌ Inactive 180d+ → <span style="color:#f87171">0.40×</span><br>
    ❌ Response rate &lt;10% → <span style="color:#f87171">0.50×</span><br>
    ✅ GitHub score &gt;60 → <span style="color:#4ade80">1.08× bonus</span><br>
    ✅ Notice ≤15 days → <span style="color:#4ade80">1.05× bonus</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    run_btn = st.button("▶  Run Ranker", use_container_width=True)

# ── Right panel ───────────────────────────────────────────────
with right:
    if run_btn:
        candidates = []
        id_generated = False

        if uploaded and not use_sample:
            try:
                content = read_uploaded_file(uploaded)
                raw_candidates = parse_candidates_file(content)

                if len(raw_candidates) > 100:
                    st.warning(f"Loaded {len(raw_candidates)} candidates — using first 100 for demo.")
                    raw_candidates = raw_candidates[:100]

                for i, c in enumerate(raw_candidates):
                    real_id = c.get("candidate_id", "")
                    generated = not re.match(r"^CAND_[0-9]{7}$", str(real_id))
                    if generated:
                        c["candidate_id"] = get_candidate_id(c, i)
                        id_generated = True
                    candidates.append(c)

            except ValueError as e:
                st.error(f"Parse error: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
        else:
            candidates = _get_builtin_candidates()

        if id_generated:
            st.warning(
                "⚠️ Some candidates had missing/non-standard IDs. "
                "Demo IDs were generated. Real submission uses official CAND_XXXXXXX IDs."
            )

        if candidates:
            prog = st.progress(0, text="Scoring candidates...")

            scored_results = []
            for i, c in enumerate(candidates):
                prog.progress((i + 1) / len(candidates),
                              text=f"Scoring {i+1}/{len(candidates)}...")
                score = score_candidate(c)
                scored_results.append({"candidate": c, "candidate_id": c.get("candidate_id"), "score": score})

            prog.empty()

            scored_results.sort(key=lambda x: (-x["score"], x["candidate_id"]))
            results = []
            for i, item in enumerate(scored_results[:100]):
                rank = i + 1
                reasoning = build_reasoning(item["candidate"], rank, item["score"])
                results.append({
                    "candidate_id": item["candidate_id"],
                    "rank": rank,
                    "score": item["score"],
                    "reasoning": reasoning,
                    "candidate": item["candidate"]
                })

            st.session_state["results"] = results
            st.session_state["ran"] = True
        elif not (uploaded and not use_sample):
            pass
        else:
            st.error("No valid candidates to rank.")

    if st.session_state.get("ran") and st.session_state.get("results"):
        results = st.session_state["results"]
        scores = [r["score"] for r in results]

        honeypots = sum(1 for r in results if r["score"] <= 0.02)
        dark_horses = sum(1 for r in results if r["rank"] > 5 and r["score"] >= 0.70)

        # ── Metric row ──
        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-box metric-green">
            <div class="metric-val">{len(results)}</div>
            <div class="metric-label">Ranked</div>
          </div>
          <div class="metric-box metric-blue">
            <div class="metric-val">{max(scores):.3f}</div>
            <div class="metric-label">Top Score</div>
          </div>
          <div class="metric-box metric-amber">
            <div class="metric-val">{honeypots}</div>
            <div class="metric-label">Honeypots</div>
          </div>
          <div class="metric-box">
            <div class="metric-val" style="color:#a78bfa">{dark_horses}</div>
            <div class="metric-label">Dark Horses</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Tabs ──
        tab1, tab2, tab3 = st.tabs(["🏆  Rankings", "📋  Submission CSV", "🔍  Score Breakdown"])

        with tab1:
            # ── Sort results by rank ──
            sorted_results = sorted(results, key=lambda r: r["rank"])
            
            # ── Top 3 podium ──
            if len(sorted_results) >= 3:
                medals = ["🥇", "🥈", "🥉"]
                podium_cols = st.columns(3)
                for i, col in enumerate(podium_cols):
                    r = sorted_results[i]
                    profile = r["candidate"].get("profile", {})
                    name = profile.get("anonymized_name") or r["candidate_id"]
                    title = profile.get("current_title") or "—"
                    company = profile.get("current_company") or "—"
                    yoe = profile.get("years_of_experience") or "?"
                    score = r["score"]
                    
                    score_color = "#4ade80" if score >= 0.70 else "#fbbf24" if score >= 0.45 else "#f87171"
                    
                    col.markdown(f"""
        <div style="
            background:#1a1f2e;
            border:1px solid #2d3748;
            border-radius:12px;
            padding:16px;
            text-align:center;
            margin-bottom:8px;
        ">
          <div style="font-size:1.8rem">{medals[i]}</div>
          <div style="font-weight:800;color:#e2e8f0;font-size:0.95rem;margin:6px 0 2px">{name}</div>
          <div style="font-size:0.75rem;color:#64748b">{title} @ {company}</div>
          <div style="font-size:0.75rem;color:#64748b">{yoe}yr</div>
          <div style="font-size:1.3rem;font-weight:800;color:{score_color};margin-top:8px">{score:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # ── Full ranked list using native Streamlit ──
            st.markdown(
                '<p style="font-size:0.75rem;color:#64748b;margin-bottom:8px;">'
                f'All {len(sorted_results)} candidates ranked · '
                'Scroll to see full list</p>',
                unsafe_allow_html=True
            )
            
            for r in sorted_results:
                profile = r["candidate"].get("profile", {})
                signals = r["candidate"].get("redrob_signals", {})
                
                name = profile.get("anonymized_name") or r["candidate_id"]
                title = profile.get("current_title") or "—"
                company = profile.get("current_company") or "—"
                yoe = profile.get("years_of_experience") or "?"
                location = profile.get("location") or "—"
                score = r["score"]
                rank = r["rank"]
                notice = signals.get("notice_period_days")
                
                # Determine badges
                is_honeypot = score <= 0.02
                is_dark_horse = rank > 5 and score >= 0.70
                
                # Score color
                if score >= 0.70:
                    score_color = "#4ade80"
                    score_bg = "rgba(34,197,94,0.12)"
                elif score >= 0.45:
                    score_color = "#fbbf24"
                    score_bg = "rgba(234,179,8,0.12)"
                else:
                    score_color = "#f87171"
                    score_bg = "rgba(239,68,68,0.12)"
                
                # Rank color
                if rank == 1:
                    rank_color = "#fbbf24"
                elif rank == 2:
                    rank_color = "#94a3b8"
                elif rank == 3:
                    rank_color = "#cd7c2f"
                else:
                    rank_color = "#475569"
                
                # Build badge text (safe — no HTML interpolation of user data)
                badges = ""
                if is_honeypot:
                    badges = "🚫 HONEYPOT"
                elif is_dark_horse:
                    badges = "⭐ DARK HORSE"
                
                # Notice flag
                notice_text = f" · {notice}d notice" if isinstance(notice, int) else ""
                
                # Reasoning — clean any problematic chars
                reasoning = r.get("reasoning", "")
                # Truncate to 150 chars for display
                reasoning_preview = (reasoning[:150] + "…") if len(reasoning) > 150 else reasoning
                
                with st.container():
                    row_col1, row_col2 = st.columns([1, 11])
                    
                    with row_col1:
                        st.markdown(
                            f'<div style="font-size:1.1rem;font-weight:800;color:{rank_color};'
                            f'padding-top:14px;text-align:center">#{rank}</div>',
                            unsafe_allow_html=True
                        )
                    
                    with row_col2:
                        # Name line
                        name_line = f"**{name}**"
                        if badges:
                            name_line += f" `{badges}`"
                        
                        score_display = (
                            f'<span style="background:{score_bg};color:{score_color};'
                            f'padding:2px 10px;border-radius:6px;font-weight:700;'
                            f'font-size:0.85rem;float:right">{score:.4f}</span>'
                        )
                        
                        st.markdown(
                            f'{score_display}<span style="font-weight:700;color:#e2e8f0">'
                            f'{name}</span>'
                            + (f' <span style="background:rgba(234,179,8,0.15);color:#fbbf24;'
                               f'padding:1px 6px;border-radius:4px;font-size:0.7rem;font-weight:700">'
                               f'{badges}</span>' if badges else ''),
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            f'<div style="font-size:0.78rem;color:#64748b;margin-top:-8px">'
                            f'{title} @ {company} · {yoe}yr · {location}{notice_text}</div>',
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            f'<div style="font-size:0.78rem;color:#94a3b8;'
                            f'font-style:italic;margin-bottom:8px;border-bottom:1px solid #1e2433;'
                            f'padding-bottom:10px">{reasoning_preview}</div>',
                            unsafe_allow_html=True
                        )

        with tab2:
            csv_str = to_csv_string([
                {k: v for k, v in r.items() if k != "candidate"}
                for r in results
            ])

            st.markdown("""
            <div style="font-size:0.82rem; color:#64748b; margin-bottom:12px;">
            This CSV matches the official submission format exactly:
            <code>candidate_id, rank, score, reasoning</code>
            — monotonic scores, unique ranks 1-100, CAND_XXXXXXX IDs.
            </div>
            """, unsafe_allow_html=True)

            st.code(
                "\n".join(csv_str.split("\n")[:6]),
                language="csv"
            )

            st.download_button(
                label="⬇  Download submission.csv",
                data=csv_str,
                file_name="submission.csv",
                mime="text/csv",
                use_container_width=True
            )

        with tab3:
            import pandas as pd
            breakdown_data = []
            for r in results[:20]:
                profile = r["candidate"].get("profile", {})
                signals = r["candidate"].get("redrob_signals", {})
                career = r["candidate"].get("career_history", [])
                skills_list = r["candidate"].get("skills", [])

                import html as html_lib
                name = html_lib.escape(
                    profile.get("anonymized_name") or r["candidate_id"]
                )
                yoe = profile.get("years_of_experience") or 0
                notice = signals.get("notice_period_days") or "?"
                last_active = signals.get("last_active_date") or "?"

                retrieval = detect_retrieval_experience(career)
                skill_trust = compute_skill_trust_score(skills_list)
                avail = compute_availability_multiplier(signals, profile)
                svc = compute_services_penalty(career)
                exp_mult = _experience_band_multiplier(yoe)

                breakdown_data.append({
                    "Rank": r["rank"],
                    "Name": name[:20],
                    "Final": f"{r['score']:.4f}",
                    "Exp×": f"{exp_mult:.2f}",
                    "Skill": f"{skill_trust:.2f}",
                    "Retrieval": f"{retrieval:.2f}",
                    "Avail×": f"{avail:.2f}",
                    "Svc×": f"{svc:.2f}",
                    "Notice": notice,
                    "Last Active": str(last_active)[:10]
                })

            df = pd.DataFrame(breakdown_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption("Showing top 20 candidates. Each multiplier shown individually so you can see exactly why each candidate ranked where they did.")

    elif not st.session_state.get("ran"):
        st.markdown("""
        <div style="
            background: #1a1f2e;
            border: 1px dashed #2d3748;
            border-radius: 16px;
            padding: 60px 40px;
            text-align: center;
            margin-top: 20px;
        ">
          <div style="font-size:3rem; margin-bottom:16px;">🎯</div>
          <div style="font-size:1.1rem; font-weight:700; color:#e2e8f0; margin-bottom:8px;">
            Ready to rank
          </div>
          <div style="font-size:0.85rem; color:#64748b; max-width:340px; margin:0 auto;">
            Upload a JSONL or JSON candidate file, or use the built-in 5-candidate
            demo, then click <strong style="color:#ef4444">Run Ranker</strong>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ──────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center;
            font-size:0.75rem; color:#475569; flex-wrap:wrap; gap:8px;">
  <span>🎯 AI Recruiter · Redrob Hackathon ·
    <a href="https://github.com/pithva007/Ai-Recruiter"
       style="color:#ef4444; text-decoration:none">
      github.com/pithva007/Ai-Recruiter
    </a>
  </span>
  <span>career×0.30 + skills×0.20 + retrieval×0.30 + fit×0.20 → ×availability × services_penalty</span>
</div>
""", unsafe_allow_html=True)
