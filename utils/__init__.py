# utils/__init__.py
# Utility package for the Redrob AI Recruiter Ranking System.
#
# Modules:
#   llm_client          — Gemini API wrapper (call_llm)
#   feature_engineering — Deterministic scoring functions (compute_final_score, is_honeypot)
#   embedding_client    — Local sentence-transformers wrapper (offline pre-compute only)
#   json_validator      — Pydantic models for pipeline I/O validation
