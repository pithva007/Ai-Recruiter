# src/stage8_dashboard.py
# Stage 8: Recruiter Copilot Dashboard
#
# All data is pre-computed by Stages 1-7. NO LLM calls from this module.
# Launch via:  streamlit run app.py
#
# Panel layout (width ratios):
#   Left sidebar  (1) — Hiring Decision Simulator + weight sliders
#   Center        (3) — Ranked Candidates table
#   Right         (2) — Candidate detail view

import json
import os
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = os.path.join(os.path.dirname(__file__), "..")
CSV_PATH     = os.path.join(ROOT, "outputs", "ranked_candidates.csv")
SCORES_DIR   = os.path.join(ROOT, "data",    "processed", "scores")
SUMMARY_PATH = os.path.join(ROOT, "outputs", "ranking_summary.json")

# ---------------------------------------------------------------------------
# Default weights from SKILLS.md weight schema
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = {
    "fit_weight":       0.35,
    "impact_weight":    0.30,
    "potential_weight": 0.20,
    "risk_weight":      0.15,
}
WEIGHT_RANGES = {
    "fit_weight":       (0.10, 0.60),
    "impact_weight":    (0.10, 0.50),
    "potential_weight": (0.05, 0.40),
    "risk_weight":      (0.05, 0.30),
}
WEIGHT_LABELS = {
    "fit_weight":       "Fit",
    "impact_weight":    "Impact",
    "potential_weight": "Potential",
    "risk_weight":      "Risk",
}

# ---------------------------------------------------------------------------
# Data loaders (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_ranked_csv() -> pd.DataFrame:
    """Load outputs/ranked_candidates.csv. Returns empty DataFrame if missing."""
    if not os.path.exists(CSV_PATH):
        return pd.DataFrame()
    df = pd.read_csv(CSV_PATH, dtype=str)
    for col in ["rank", "fit_score", "impact_score", "potential_score",
                "risk_score", "composite_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data
def load_score_file(candidate_id: str) -> dict:
    """Load data/processed/scores/{candidate_id}_scores.json."""
    path = os.path.join(SCORES_DIR, f"{candidate_id}_scores.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_summary() -> dict:
    if not os.path.exists(SUMMARY_PATH):
        return {}
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Weight normalisation
# ---------------------------------------------------------------------------

def normalise_weights(weights: dict, changed_key: str) -> dict:
    """
    After one slider changes, redistribute the residual proportionally
    among the other three weights while keeping them within their allowed ranges.
    Always ensures sum == 1.0.
    """
    changed_val = weights[changed_key]
    others = {k: v for k, v in weights.items() if k != changed_key}
    residual = 1.0 - changed_val
    other_sum = sum(others.values())

    if other_sum == 0:
        # Edge case: share equally
        share = residual / len(others)
        for k in others:
            others[k] = share
    else:
        scale = residual / other_sum
        for k in others:
            others[k] = round(others[k] * scale, 4)

    # Clamp to ranges
    for k, v in others.items():
        lo, hi = WEIGHT_RANGES[k]
        others[k] = max(lo, min(hi, v))

    # Fix any leftover rounding
    result = {changed_key: changed_val, **others}
    total = sum(result.values())
    diff  = round(1.0 - total, 6)
    if diff != 0:
        # Apply residual to the largest non-changed weight
        biggest = max(others, key=lambda k: others[k])
        result[biggest] = round(result[biggest] + diff, 6)

    return result


# ---------------------------------------------------------------------------
# Composite recalculation
# ---------------------------------------------------------------------------

def recalculate_composite(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """
    Re-compute composite_score for every row using current slider weights.
    Re-sort and re-assign ranks.
    """
    df = df.copy()
    fw = weights["fit_weight"]
    iw = weights["impact_weight"]
    pw = weights["potential_weight"]
    rw = weights["risk_weight"]

    df["composite_score"] = (
        df["fit_score"]       * fw +
        df["impact_score"]    * iw +
        df["potential_score"] * pw +
        (100 - df["risk_score"]) * rw
    ).round(2)

    df = df.sort_values(
        by=["composite_score", "candidate_id"],
        ascending=[False, True],
    ).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


# ---------------------------------------------------------------------------
# Colour helper for composite score
# ---------------------------------------------------------------------------

def score_color(score) -> str:
    try:
        s = float(score)
    except Exception:
        return ""
    if s >= 75:
        return "background-color: #d4edda; color: #155724"   # green
    elif s >= 50:
        return "background-color: #fff3cd; color: #856404"   # yellow
    else:
        return "background-color: #f8d7da; color: #721c24"   # red


def style_composite(val):
    return score_color(val)


# ---------------------------------------------------------------------------
# Radar chart
# ---------------------------------------------------------------------------

def make_radar_chart(fit: float, impact: float, potential: float, risk: float) -> go.Figure:
    categories = ["Fit", "Impact", "Potential", "Risk (inv)"]
    values     = [fit, impact, potential, 100 - risk]

    fig = go.Figure(go.Scatterpolar(
        r    = values + [values[0]],
        theta= categories + [categories[0]],
        fill = "toself",
        line = dict(color="#4c72b0", width=2),
        fillcolor="rgba(76, 114, 176, 0.25)",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont_size=9),
        ),
        showlegend=False,
        margin=dict(l=30, r=30, t=30, b=30),
        height=270,
    )
    return fig


# ---------------------------------------------------------------------------
# Split pipe-separated fields
# ---------------------------------------------------------------------------

def pipe_split(value: str) -> list[str]:
    if not value or str(value).strip() in ("", "nan"):
        return []
    return [s.strip() for s in str(value).split("|") if s.strip()]


# ---------------------------------------------------------------------------
# Main dashboard function — called by app.py
# ---------------------------------------------------------------------------

def run_dashboard() -> None:
    # ------------------------------------------------------------------
    # Page header
    # ------------------------------------------------------------------
    st.markdown(
        "<h1 style='margin-bottom:0'>🎯 AI Recruiter Copilot</h1>"
        "<p style='color:grey;margin-top:0'>Senior AI Engineer — Founding Team @ Redrob AI</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ------------------------------------------------------------------
    # Session state: weights and selected candidate
    # ------------------------------------------------------------------
    if "weights" not in st.session_state:
        st.session_state.weights = DEFAULT_WEIGHTS.copy()
    if "df" not in st.session_state:
        st.session_state.df = load_ranked_csv()
    if "selected_id" not in st.session_state:
        st.session_state.selected_id = None

    df_base = load_ranked_csv()
    if df_base.empty:
        st.error(
            "No ranked candidates found. Run Stage 7 first:\n"
            "`python src/stage7_ranking.py`"
        )
        return

    # ------------------------------------------------------------------
    # PANEL 1 — Left sidebar: Hiring Decision Simulator
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Hiring Decision Simulator")
        st.caption("Adjust weights to re-rank candidates without re-scoring.")

        weights = st.session_state.weights
        changed_key = None

        for key, label in WEIGHT_LABELS.items():
            lo, hi = WEIGHT_RANGES[key]
            new_val = st.slider(
                f"{label} weight",
                min_value=lo,
                max_value=hi,
                value=float(round(weights[key], 3)),
                step=0.01,
                key=f"slider_{key}",
            )
            if abs(new_val - weights[key]) > 0.001:
                weights[key] = round(new_val, 3)
                changed_key = key

        if changed_key:
            st.session_state.weights = normalise_weights(weights, changed_key)
            weights = st.session_state.weights
            st.rerun()

        total = round(sum(weights.values()), 3)
        color = "green" if abs(total - 1.0) < 0.01 else "red"
        st.markdown(
            f"<small>Weights sum: <b style='color:{color}'>{total:.3f}</b></small>",
            unsafe_allow_html=True,
        )

        col_r, col_c = st.columns(2)
        with col_r:
            if st.button("↺ Reset", use_container_width=True):
                st.session_state.weights = DEFAULT_WEIGHTS.copy()
                st.rerun()
        with col_c:
            if st.button("▶ Recalculate", use_container_width=True, type="primary"):
                st.session_state.df = recalculate_composite(
                    df_base, st.session_state.weights
                )
                st.session_state.selected_id = None
                st.rerun()

        st.divider()
        summary = load_summary()
        if summary:
            st.subheader("📊 Pipeline Summary")
            st.metric("Candidates scored", summary.get("total_candidates_scored", "—"))
            st.metric("Highest composite", summary.get("highest_composite_score", "—"))
            st.metric("Dark horses", summary.get("dark_horse_count", "—"))

    # ------------------------------------------------------------------
    # Working DataFrame (recalculated or original)
    # ------------------------------------------------------------------
    df = st.session_state.df if not st.session_state.df.empty else df_base

    # ------------------------------------------------------------------
    # PANELS 2 + 3 — two columns
    # ------------------------------------------------------------------
    col_center, col_right = st.columns([3, 2])

    # ------------------------------------------------------------------
    # PANEL 2 — Center: Ranked Candidates table
    # ------------------------------------------------------------------
    with col_center:
        st.subheader("🏆 Ranked Candidates")

        # Build display DataFrame
        display_cols = [
            "rank", "candidate_id", "candidate_name", "composite_score",
            "fit_score", "impact_score", "potential_score", "risk_score",
            "dark_horse",
        ]
        df_display = df[display_cols].copy()

        # Add star emoji to dark horse names
        df_display["candidate_name"] = df_display.apply(
            lambda r: f"⭐ {r['candidate_name']}"
            if str(r["dark_horse"]).strip().lower() == "true"
            else r["candidate_name"],
            axis=1,
        )
        df_display = df_display.drop(columns=["dark_horse"])
        df_display.columns = [
            "Rank", "ID", "Name", "Composite", "Fit", "Impact", "Potential", "Risk"
        ]

        # Style composite column
        styled = df_display.style.map(style_composite, subset=["Composite"])

        # Render with selection via a selectbox below the table
        st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True,
            height=min(420, 38 + len(df_display) * 35),
        )

        # Row selector
        name_options = df_display["Name"].tolist()
        selected_name = st.selectbox(
            "Select a candidate to view details →",
            options=["(none)"] + name_options,
            key="candidate_selector",
        )
        if selected_name != "(none)":
            clean_name = selected_name.replace("⭐ ", "")
            match = df[df["candidate_name"] == clean_name]
            if not match.empty:
                st.session_state.selected_id = match.iloc[0]["candidate_id"]

        # ------ Dark Horses expander ------
        dark_horses = df[df["dark_horse"].astype(str).str.lower() == "true"]
        with st.expander(f"⭐ Dark Horses ({len(dark_horses)})", expanded=False):
            if dark_horses.empty:
                st.write("No dark horse candidates in this run.")
            else:
                for _, dh in dark_horses.iterrows():
                    st.markdown(f"**{dh['candidate_name']}** — composite {dh['composite_score']}")
                    tsm = pipe_split(str(dh.get("transferable_skills_map", "")))
                    if tsm:
                        st.markdown("*Transferable skills:*")
                        for skill in tsm:
                            st.markdown(f"  • {skill}")
                    reason = str(dh.get("dark_horse_reason", "")).strip()
                    if reason and reason != "nan":
                        st.caption(reason[:200])
                    st.divider()

        # ------ Bias Audit expander ------
        with st.expander("🔎 Bias Audit", expanded=False):
            st.info(
                "Review these distributions to check for unintended systematic "
                "patterns in ranking."
            )

            score_cols = ["fit_score", "impact_score", "potential_score", "risk_score", "composite_score"]
            available = [c for c in score_cols if c in df.columns]

            if available:
                import plotly.express as px

                fig_dist = px.box(
                    df[available].apply(pd.to_numeric, errors="coerce"),
                    title="Score Distribution Across All Candidates",
                    labels={"value": "Score (0–100)", "variable": "Dimension"},
                    color="variable",
                )
                fig_dist.update_layout(height=300, margin=dict(t=40, b=20))
                st.plotly_chart(fig_dist, use_container_width=True)

            # Confidence level bar chart
            if "confidence_level" in df.columns:
                conf_counts = df["confidence_level"].value_counts().reset_index()
                conf_counts.columns = ["Confidence", "Count"]
                fig_conf = px.bar(
                    conf_counts,
                    x="Confidence", y="Count",
                    title="Scoring Confidence Level Distribution",
                    color="Confidence",
                    color_discrete_map={"high": "#2ecc71", "medium": "#f39c12", "low": "#e74c3c"},
                )
                fig_conf.update_layout(height=250, margin=dict(t=40, b=20), showlegend=False)
                st.plotly_chart(fig_conf, use_container_width=True)

    # ------------------------------------------------------------------
    # PANEL 3 — Right: Candidate Detail
    # ------------------------------------------------------------------
    with col_right:
        selected_id = st.session_state.selected_id

        if not selected_id:
            st.info("← Select a candidate from the table to view details.")
            st.markdown(
                "<br><br><br>"
                "<div style='text-align:center;color:grey;'>"
                "🎯 AI Recruiter Copilot<br>"
                "<small>Stages 1–7 complete · 100K candidates processed</small>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            row = df[df["candidate_id"] == selected_id]
            if row.empty:
                st.warning(f"Candidate {selected_id} not found in current view.")
            else:
                row = row.iloc[0]
                score_data = load_score_file(selected_id)

                # --- a. Header ---
                name       = row["candidate_name"]
                rank_num   = int(row["rank"])
                composite  = float(row["composite_score"])
                is_dh      = str(row.get("dark_horse", "")).lower() == "true"
                dh_badge   = " ⭐ Dark Horse" if is_dh else ""

                st.markdown(
                    f"<h3 style='margin-bottom:2px'>{name}{dh_badge}</h3>"
                    f"<small>Rank #{rank_num} · Composite {composite:.1f}</small>",
                    unsafe_allow_html=True,
                )

                # --- b. Score metric cards ---
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Fit",       int(row["fit_score"]))
                mc2.metric("Impact",    int(row["impact_score"]))
                mc3.metric("Potential", int(row["potential_score"]))
                mc4.metric("Risk",      int(row["risk_score"]))

                # --- c. Radar chart ---
                fig = make_radar_chart(
                    float(row["fit_score"]),
                    float(row["impact_score"]),
                    float(row["potential_score"]),
                    float(row["risk_score"]),
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

                # --- d. Green Flags ---
                green = pipe_split(str(row.get("green_flags", "")))
                if green:
                    st.markdown("**✅ Green Flags**")
                    for g in green:
                        st.markdown(f"• {g}")

                # --- e. Yellow Flags ---
                yellow = pipe_split(str(row.get("yellow_flags", "")))
                if yellow:
                    st.markdown("**⚠️ Yellow Flags**")
                    for y in yellow:
                        st.markdown(f"• {y}")

                # --- f. Skill Gaps ---
                gaps = pipe_split(str(row.get("skill_gaps", "")))
                if gaps:
                    st.markdown("**🔴 Skill Gaps**")
                    for gap in gaps:
                        st.markdown(f"• {gap}")
                else:
                    st.markdown("**🔴 Skill Gaps** — none identified")

                # --- g. Interview Questions ---
                q1 = str(row.get("interview_q1", "")).strip()
                q2 = str(row.get("interview_q2", "")).strip()
                q3 = str(row.get("interview_q3", "")).strip()
                questions = [q for q in [q1, q2, q3] if q and q != "nan"]
                if questions:
                    st.markdown("**💬 Interview Questions**")
                    for idx, q in enumerate(questions, 1):
                        st.markdown(f"{idx}. {q}")

                # --- h. LLM Rationale ---
                rationale = str(row.get("llm_rationale", "")).strip()
                if rationale and rationale != "nan":
                    st.info(f"**🤖 Ranking Rationale**\n\n{rationale}")

                # --- i. Why This Candidate? expander ---
                with st.expander("🔍 Why This Candidate? (Score Breakdown)", expanded=False):
                    if score_data:
                        for dim, label in [
                            ("fit_reasoning",        "Fit"),
                            ("impact_reasoning",     "Impact"),
                            ("potential_reasoning",  "Potential"),
                            ("risk_reasoning",       "Risk"),
                        ]:
                            text = score_data.get(dim, "")
                            if text and str(text).strip() not in ("", "None", "nan"):
                                st.markdown(f"**{label}:** {text}")
                    else:
                        st.caption("Score file not found. Run Stage 6 to generate detailed reasoning.")

                # Dark horse details
                if is_dh:
                    with st.expander("⭐ Dark Horse Analysis", expanded=False):
                        reason = str(row.get("dark_horse_reason", "")).strip()
                        if reason and reason != "nan":
                            st.markdown(f"**Why a dark horse:**\n{reason}")
                        tsm = pipe_split(str(row.get("transferable_skills_map", "")))
                        if tsm:
                            st.markdown("**Transferable skill mappings:**")
                            for skill in tsm:
                                st.markdown(f"• {skill}")
