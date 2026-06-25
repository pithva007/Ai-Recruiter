import streamlit as st
import json
import csv
import io
import re
import math
from datetime import date, datetime
from pathlib import Path

st.set_page_config(
    page_title="AI Recruiter — Redrob Hackathon Demo",
    page_icon="🎯",
    layout="wide"
)

# ============================================================
# SCORING ENGINE (self-contained, no external imports)
# ============================================================

SERVICES_COMPANIES = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl technologies", "hcl tech",
    "tech mahindra", "mphasis", "hexaware", "l&t infotech",
    "ltimindtree"
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
    """Detect impossible profiles."""
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


def score_candidate(candidate):
    """Compute final score for one candidate. Returns float 0-1."""
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

    final = base * avail_mult * services_pen
    return round(min(max(final, 0.0), 1.0), 6)


def build_reasoning(candidate, rank, score):
    """Build reasoning string without LLM — uses only real profile fields."""
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    name = profile.get("anonymized_name") or "Candidate"
    title = profile.get("current_title") or "engineer"
    company = profile.get("current_company") or "their current employer"
    yoe = profile.get("years_of_experience") or "?"
    notice = signals.get("notice_period_days")
    location = profile.get("location") or "unknown location"
    willing = signals.get("willing_to_relocate", False)

    RELEVANT = {"embedding", "vector", "retrieval", "search", "ranking",
                "faiss", "transformer", "llm", "rag", "nlp", "recommendation"}
    rel_skills = sorted(
        [s for s in skills if any(kw in (s.get("name") or "").lower() for kw in RELEVANT)],
        key=lambda s: s.get("duration_months") or 0, reverse=True
    )
    top_skill = f"{rel_skills[0]['name']} ({rel_skills[0].get('duration_months', 0)}mo)" if rel_skills else None

    has_retrieval = any(
        any(kw in (j.get("description") or "").lower() for kw in RETRIEVAL_KEYWORDS)
        for j in career
    )

    GOOD_CITIES = {"pune", "noida", "hyderabad", "mumbai", "delhi",
                   "gurugram", "gurgaon", "bengaluru", "bangalore"}
    in_good = any(c in (location or "").lower() for c in GOOD_CITIES)

    notice_flag = f"{notice}-day notice period" if isinstance(notice, int) and notice > 45 else None
    location_flag = f"location ({location})" if not in_good and not willing else None

    if rank <= 20:
        skill_part = f"with {top_skill} expertise" if top_skill else "with applied ML background"
        retrieval_part = " and demonstrated production retrieval experience" if has_retrieval else ""
        concern = notice_flag or location_flag or "no critical gaps identified"
        return f"{name} ({yoe} years, {title} at {company}) {skill_part}{retrieval_part}; {concern} is the primary consideration at rank {rank}."
    elif rank <= 60:
        gap = "limited production retrieval depth" if not has_retrieval else (notice_flag or "partial JD alignment")
        skill_part = f"{top_skill} background" if top_skill else "general ML background"
        return f"{name} ({title} at {company}, {yoe} years) brings {skill_part} but {gap}; ranked {rank} on partial overlap with the search-and-ranking requirements."
    else:
        gap = "no production retrieval or ranking system evidence" if not has_retrieval else (
            (notice_flag and location_flag and f"{notice_flag} and {location_flag}") or
            notice_flag or location_flag or "adjacent skill profile only"
        )
        return f"{name} ({yoe} years, {title}) shows {gap or 'limited JD alignment'}; included at rank {rank} based on partial signal overlap but below the threshold for confident recommendation."


def rank_candidates(candidates):
    """Score and rank. Returns list of dicts with all submission fields."""
    scored = []
    for c in candidates:
        cid = c.get("candidate_id", "")
        if not re.match(r"^CAND_[0-9]{7}$", cid):
            continue
        score = score_candidate(c)
        scored.append({"candidate": c, "candidate_id": cid, "score": score})

    scored.sort(key=lambda x: (-x["score"], x["candidate_id"]))

    results = []
    for i, item in enumerate(scored[:100]):
        rank = i + 1
        reasoning = build_reasoning(item["candidate"], rank, item["score"])
        results.append({
            "candidate_id": item["candidate_id"],
            "rank": rank,
            "score": item["score"],
            "reasoning": reasoning
        })
    return results


def to_csv_string(results):
    """Convert results to CSV string matching submission spec exactly."""
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=["candidate_id", "rank", "score", "reasoning"],
        quoting=csv.QUOTE_ALL
    )
    writer.writeheader()
    writer.writerows(results)
    return output.getvalue()


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("🎯 AI Recruiter — Redrob Hackathon Demo")
st.caption("Senior AI Engineer — Founding Team @ Redrob AI | Ranking system by Team pithva-ai-recruiter")

st.markdown("""
**How this works:** Upload a JSONL file with up to 100 candidate profiles
(same schema as `candidates.jsonl` from the hackathon bundle).
The ranker scores each candidate on career trajectory, skill trust,
production retrieval experience, and behavioral availability — then
outputs a submission-ready CSV.
""")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📁 Input")

    uploaded = st.file_uploader(
        "Upload candidates JSONL (≤100 candidates)",
        type=["jsonl", "json"],
        help="Each line must be a valid JSON candidate object with candidate_id field"
    )

    use_sample = st.checkbox(
        "Use built-in sample (5 synthetic candidates)",
        value=not bool(uploaded)
    )

    if st.button("▶ Run Ranker", type="primary", use_container_width=True):
        candidates = []

        if uploaded and not use_sample:
            try:
                content = uploaded.read().decode("utf-8")
                for line in content.strip().split("\n"):
                    if line.strip():
                        candidates.append(json.loads(line))
                if len(candidates) > 100:
                    st.warning(f"Loaded {len(candidates)} candidates — truncating to first 100 for demo.")
                    candidates = candidates[:100]
            except Exception as e:
                st.error(f"Failed to parse JSONL: {e}")
                candidates = []
        else:
            # Built-in 5 synthetic candidates for demo
            candidates = [
                {
                    "candidate_id": "CAND_9990001",
                    "profile": {
                        "anonymized_name": "Demo Candidate A",
                        "current_title": "Senior ML Engineer",
                        "current_company": "Zomato",
                        "current_company_size": "1001-5000",
                        "years_of_experience": 7.0,
                        "location": "Noida",
                        "country": "India",
                        "current_industry": "Food Tech",
                        "summary": "7 years in applied ML, built retrieval and ranking systems at scale."
                    },
                    "career_history": [
                        {
                            "company": "Zomato",
                            "title": "Senior ML Engineer",
                            "duration_months": 30,
                            "is_current": True,
                            "description": "Built hybrid retrieval system (BM25 + dense vector) for restaurant search. Designed FAISS index serving 50M queries/day. Implemented NDCG-based offline evaluation framework."
                        }
                    ],
                    "skills": [
                        {"name": "FAISS", "proficiency": "advanced", "duration_months": 30, "endorsements": 15},
                        {"name": "Embeddings", "proficiency": "advanced", "duration_months": 36, "endorsements": 22},
                        {"name": "Python", "proficiency": "expert", "duration_months": 84, "endorsements": 40},
                        {"name": "Ranking", "proficiency": "advanced", "duration_months": 30, "endorsements": 18}
                    ],
                    "redrob_signals": {
                        "last_active_date": "2026-05-28",
                        "open_to_work_flag": True,
                        "recruiter_response_rate": 0.75,
                        "notice_period_days": 30,
                        "interview_completion_rate": 0.90,
                        "willing_to_relocate": False,
                        "github_activity_score": 72.0
                    }
                },
                {
                    "candidate_id": "CAND_9990002",
                    "profile": {
                        "anonymized_name": "Demo Candidate B",
                        "current_title": "Backend Engineer",
                        "current_company": "TCS",
                        "current_company_size": "10001+",
                        "years_of_experience": 6.0,
                        "location": "Chennai",
                        "country": "India",
                        "current_industry": "IT Services",
                        "summary": "6 years in backend at TCS"
                    },
                    "career_history": [
                        {
                            "company": "TCS",
                            "title": "Backend Engineer",
                            "duration_months": 72,
                            "is_current": True,
                            "description": "Built REST APIs and maintained enterprise applications."
                        }
                    ],
                    "skills": [
                        {"name": "Java", "proficiency": "advanced", "duration_months": 60, "endorsements": 10},
                        {"name": "Spring Boot", "proficiency": "advanced", "duration_months": 48, "endorsements": 8}
                    ],
                    "redrob_signals": {
                        "last_active_date": "2026-04-01",
                        "open_to_work_flag": True,
                        "recruiter_response_rate": 0.50,
                        "notice_period_days": 90,
                        "interview_completion_rate": 0.70,
                        "willing_to_relocate": False,
                        "github_activity_score": -1
                    }
                },
                {
                    "candidate_id": "CAND_9990003",
                    "profile": {
                        "anonymized_name": "Demo Candidate C",
                        "current_title": "AI Engineer",
                        "current_company": "Razorpay",
                        "current_company_size": "1001-5000",
                        "years_of_experience": 5.5,
                        "location": "Bangalore",
                        "country": "India",
                        "current_industry": "Fintech",
                        "summary": "5.5yr applied ML, recommendation systems and search."
                    },
                    "career_history": [
                        {
                            "company": "Razorpay",
                            "title": "AI Engineer",
                            "duration_months": 24,
                            "is_current": True,
                            "description": "Built semantic search for merchant discovery using sentence-transformers and Elasticsearch. Designed offline NDCG evaluation pipeline."
                        }
                    ],
                    "skills": [
                        {"name": "Elasticsearch", "proficiency": "advanced", "duration_months": 24, "endorsements": 12},
                        {"name": "Sentence Transformers", "proficiency": "advanced", "duration_months": 24, "endorsements": 9},
                        {"name": "Python", "proficiency": "expert", "duration_months": 66, "endorsements": 30},
                        {"name": "NDCG", "proficiency": "intermediate", "duration_months": 18, "endorsements": 5}
                    ],
                    "redrob_signals": {
                        "last_active_date": "2026-06-01",
                        "open_to_work_flag": True,
                        "recruiter_response_rate": 0.80,
                        "notice_period_days": 45,
                        "interview_completion_rate": 0.85,
                        "willing_to_relocate": True,
                        "github_activity_score": 55.0
                    }
                },
                {
                    "candidate_id": "CAND_9990004",
                    "profile": {
                        "anonymized_name": "Demo Candidate D (Honeypot)",
                        "current_title": "ML Engineer",
                        "current_company": "FakeAI Corp",
                        "current_company_size": "11-50",
                        "years_of_experience": 9.0,
                        "location": "Mumbai",
                        "country": "India",
                        "current_industry": "AI",
                        "summary": "Expert in everything AI."
                    },
                    "career_history": [],
                    "skills": [
                        {"name": "FAISS", "proficiency": "expert", "duration_months": 0, "endorsements": 0},
                        {"name": "Pinecone", "proficiency": "expert", "duration_months": 0, "endorsements": 0},
                        {"name": "LLM Fine-tuning", "proficiency": "expert", "duration_months": 0, "endorsements": 0},
                        {"name": "RAG", "proficiency": "expert", "duration_months": 0, "endorsements": 0},
                        {"name": "Kubernetes", "proficiency": "expert", "duration_months": 0, "endorsements": 0}
                    ],
                    "redrob_signals": {
                        "last_active_date": "2026-06-01",
                        "open_to_work_flag": True,
                        "recruiter_response_rate": 0.90,
                        "notice_period_days": 0,
                        "interview_completion_rate": 1.0,
                        "willing_to_relocate": True,
                        "github_activity_score": 99.0
                    }
                },
                {
                    "candidate_id": "CAND_9990005",
                    "profile": {
                        "anonymized_name": "Demo Candidate E",
                        "current_title": "Data Scientist",
                        "current_company": "Meesho",
                        "current_company_size": "1001-5000",
                        "years_of_experience": 4.2,
                        "location": "Bangalore",
                        "country": "India",
                        "current_industry": "E-commerce",
                        "summary": "4 years data science, recommendation and ranking."
                    },
                    "career_history": [
                        {
                            "company": "Meesho",
                            "title": "Data Scientist",
                            "duration_months": 20,
                            "is_current": True,
                            "description": "Built product recommendation system using collaborative filtering and embedding models. A/B tested ranking algorithms."
                        }
                    ],
                    "skills": [
                        {"name": "Recommendation Systems", "proficiency": "intermediate", "duration_months": 20, "endorsements": 7},
                        {"name": "Python", "proficiency": "advanced", "duration_months": 48, "endorsements": 20},
                        {"name": "Embeddings", "proficiency": "intermediate", "duration_months": 15, "endorsements": 4}
                    ],
                    "redrob_signals": {
                        "last_active_date": "2026-05-15",
                        "open_to_work_flag": True,
                        "recruiter_response_rate": 0.60,
                        "notice_period_days": 30,
                        "interview_completion_rate": 0.75,
                        "willing_to_relocate": True,
                        "github_activity_score": 28.0
                    }
                }
            ]

        if candidates:
            with st.spinner(f"Ranking {len(candidates)} candidates..."):
                results = rank_candidates(candidates)

            st.session_state["results"] = results
            st.session_state["ran"] = True
        else:
            st.error("No valid candidates to rank.")

with col2:
    st.subheader("📊 Results")

    if st.session_state.get("ran") and st.session_state.get("results"):
        results = st.session_state["results"]

        scores = [r["score"] for r in results]
        m1, m2, m3 = st.columns(3)
        m1.metric("Candidates Ranked", len(results))
        m2.metric("Top Score", f"{max(scores):.4f}")
        m3.metric("Score Spread", f"{max(scores)-min(scores):.4f}")

        import pandas as pd
        df = pd.DataFrame([{
            "Rank": r["rank"],
            "Candidate ID": r["candidate_id"],
            "Score": f"{r['score']:.4f}",
            "Reasoning (preview)": r["reasoning"][:80] + "..."
        } for r in results])

        st.dataframe(df, use_container_width=True, hide_index=True)

        csv_str = to_csv_string(results)
        st.download_button(
            label="⬇ Download submission.csv",
            data=csv_str,
            file_name="submission.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.info(
            "✅ Output format matches submission spec: "
            "candidate_id, rank, score, reasoning — "
            "monotonic scores, unique ranks 1-100, CAND_XXXXXXX IDs."
        )

        with st.expander("View full reasoning — Top 5"):
            for r in results[:5]:
                st.markdown(f"**Rank {r['rank']} ({r['candidate_id']}):** {r['reasoning']}")
    else:
        st.info("Upload a JSONL file (or use the built-in sample) and click **Run Ranker** to see results.")

st.markdown("---")
st.caption(
    "AI Recruiter · Redrob Hackathon Submission · "
    "github.com/pithva007/Ai-Recruiter · "
    "Scoring: career×0.30 + skills×0.20 + retrieval×0.30 + fit×0.20 — "
    "then × availability_multiplier × services_penalty"
)
