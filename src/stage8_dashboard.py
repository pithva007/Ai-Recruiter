# src/stage8_dashboard.py
# Phase D — Streamlit Dashboard / Sandbox Demo
#
# Serves as the required sandbox link (HuggingFace Spaces / Streamlit Cloud).
# Accepts a small candidate sample (≤100 candidates) via file upload,
# runs the ranking pipeline end-to-end, and displays the top results.
#
# Usage:
#   streamlit run src/stage8_dashboard.py
#
# Deploy to Streamlit Cloud or HuggingFace Spaces for the sandbox_link field
# in submission_metadata.yaml.

import csv
import io
import json
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.feature_engineering import compute_final_score, is_honeypot

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Redrob AI Ranker",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 Redrob Intelligent Candidate Ranker")
st.caption("Senior AI Engineer — Founding Team @ Redrob AI")

# ---------------------------------------------------------------------------
# Sidebar — controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Controls")
    top_n = st.slider("Top N candidates to show", min_value=5, max_value=100, value=20)
    st.divider()
    st.markdown("**Score Weights**")
    w_career   = st.slider("Career weight",       0.10, 0.70, 0.45, 0.05)
    w_skill    = st.slider("Skill weight",         0.10, 0.50, 0.25, 0.05)
    w_behavior = st.slider("Behavioral weight",    0.05, 0.40, 0.20, 0.05)
    w_fit      = st.slider("Fit weight",           0.05, 0.30, 0.10, 0.05)
    total_w = w_career + w_skill + w_behavior + w_fit
    if abs(total_w - 1.0) > 0.01:
        # Normalize
        w_career   /= total_w
        w_skill    /= total_w
        w_behavior /= total_w
        w_fit      /= total_w
        st.warning(f"Weights normalized to sum to 1.0")

# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------
st.subheader("1. Upload Candidate Sample")
uploaded = st.file_uploader(
    "Upload candidates (.jsonl or .json — max 100 candidates for sandbox)",
    type=["jsonl", "json"],
    help="Upload a small sample of candidates in the challenge JSONL format.",
)

# Also allow using the built-in sample
use_sample = st.checkbox("Use built-in sample_candidates.json (6 candidates)", value=False)

candidates = []

if use_sample:
    sample_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "sample_candidates.json")
    if os.path.exists(sample_path):
        with open(sample_path, "r", encoding="utf-8") as f:
            candidates = json.load(f)
        st.success(f"Loaded {len(candidates)} sample candidates.")
    else:
        st.error("sample_candidates.json not found in data/raw/. Run data organization step first.")

elif uploaded is not None:
    content = uploaded.read().decode("utf-8")
    try:
        if uploaded.name.endswith(".jsonl"):
            candidates = [json.loads(line) for line in content.splitlines() if line.strip()]
        else:
            data = json.loads(content)
            candidates = data if isinstance(data, list) else data.get("candidates", [])
        candidates = candidates[:100]  # sandbox cap
        st.success(f"Loaded {len(candidates)} candidates from {uploaded.name}.")
    except Exception as e:
        st.error(f"Failed to parse file: {e}")

# ---------------------------------------------------------------------------
# Run ranking
# ---------------------------------------------------------------------------
if candidates:
    st.subheader("2. Run Ranking")
    if st.button("🚀 Rank Candidates", type="primary"):
        with st.spinner("Scoring candidates ..."):
            results = []
            for c in candidates:
                hp    = is_honeypot(c)
                score = 0.0 if hp else compute_final_score(c)
                p     = c.get("profile", {})
                sig   = c.get("redrob_signals", {})
                results.append({
                    "candidate_id":  c.get("candidate_id", ""),
                    "name":          p.get("anonymized_name", ""),
                    "title":         p.get("current_title", ""),
                    "years_exp":     p.get("years_of_experience", 0),
                    "location":      p.get("location", ""),
                    "score":         round(score, 4),
                    "honeypot":      hp,
                    "open_to_work":  sig.get("open_to_work_flag", False),
                    "response_rate": sig.get("recruiter_response_rate", 0),
                    "notice_days":   sig.get("notice_period_days", 0),
                    "github":        sig.get("github_activity_score", -1),
                    "skill_count":   len(c.get("skills", [])),
                })

        # Sort
        ranked = sorted(results, key=lambda r: (-r["score"], r["candidate_id"]))[:top_n]

        st.subheader(f"3. Top {min(top_n, len(ranked))} Results")

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Candidates Scored", len(results))
        col2.metric("Honeypots Detected", sum(1 for r in results if r["honeypot"]))
        col3.metric("Top Score", f"{ranked[0]['score']:.4f}" if ranked else "—")
        col4.metric("Open to Work (top 10)", sum(1 for r in ranked[:10] if r["open_to_work"]))

        st.divider()

        # Results table
        import pandas as pd
        df = pd.DataFrame(ranked)
        df.insert(0, "rank", range(1, len(df) + 1))
        df["score"] = df["score"].map("{:.4f}".format)
        df["response_rate"] = df["response_rate"].map("{:.2f}".format)
        df["open_to_work"] = df["open_to_work"].map(lambda x: "✅" if x else "❌")
        df["honeypot"] = df["honeypot"].map(lambda x: "⚠️ YES" if x else "—")

        st.dataframe(
            df[["rank", "candidate_id", "name", "title", "years_exp",
                "location", "score", "open_to_work", "response_rate",
                "notice_days", "github", "honeypot"]],
            use_container_width=True,
            hide_index=True,
        )

        # Download as CSV
        csv_buf = io.StringIO()
        writer = csv.writer(csv_buf)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, r in enumerate(ranked, 1):
            reasoning = (
                f"{r['title']} with {r['years_exp']} yrs; "
                f"response rate {float(r['response_rate']):.2f}."
            )
            writer.writerow([r["candidate_id"], i, r["score"], reasoning])

        st.download_button(
            label="⬇️  Download submission CSV",
            data=csv_buf.getvalue(),
            file_name="submission_sample.csv",
            mime="text/csv",
        )

else:
    st.info("Upload a candidate file or check 'Use built-in sample' above to get started.")

st.divider()
st.caption("Redrob Hackathon — AI Recruiter Ranking System | Phase D: Sandbox Demo")
