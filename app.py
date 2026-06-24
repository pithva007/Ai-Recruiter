# app.py — Streamlit entry point for the AI Recruiter Copilot Dashboard
#
# Launch with:
#   streamlit run app.py
#
# This file:
#   1. Sets page config (wide layout, custom title)
#   2. Imports and runs src/stage8_dashboard.py
#   No LLM calls. All data is pre-computed by Stages 1-7.

import sys
import os

# Ensure project root is on the path so src/utils imports resolve
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

# Page config must be the first Streamlit call in the script
st.set_page_config(
    page_title="AI Recruiter Copilot",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "**AI Recruiter Copilot** — Redrob Intelligent Candidate Discovery & Ranking\n\n"
            "Stages 1–7 complete. Dashboard reads pre-computed outputs only. No LLM calls."
        )
    },
)

# Import and run the dashboard
from src.stage8_dashboard import run_dashboard

run_dashboard()
