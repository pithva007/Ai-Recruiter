# src/__init__.py
# Source package for the Redrob AI Recruiter Ranking System.
#
# Pipeline execution order:
#   Phase A (offline, LLM OK):
#     1. python src/stage1_jd_analysis.py   → data/processed/jd_features.json
#     2. python src/precompute.py            → data/processed/features.pkl
#
#   Phase B (< 5 min, CPU, NO network):
#     3. python src/rank.py                  → outputs/ranked_top100_raw.csv
#
#   Phase C (offline, LLM for top 100 only):
#     4. python src/reason.py               → outputs/submission.csv
#
#   Validate:
#     5. python validate_submission.py outputs/submission.csv
#
#   Dashboard / Sandbox:
#     6. streamlit run src/stage8_dashboard.py
