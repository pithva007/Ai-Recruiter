---
title: AI Recruiter Redrob Demo
emoji: 🎯
colorFrom: red
colorTo: orange
sdk: streamlit
sdk_version: 1.30.0
app_file: app.py
pinned: false
---

# AI Recruiter — Redrob Hackathon Demo

Upload a JSONL file of up to 100 candidate profiles (same schema as
the hackathon bundle) and get a ranked submission CSV.

**Scoring formula:**
career×0.30 + skill_trust×0.20 + production_retrieval×0.30 + fit×0.20
Then multiplied by: availability_multiplier × services_penalty

**Compute:** CPU only, no network, <5 min, <16GB RAM.

**GitHub:** https://github.com/pithva007/Ai-Recruiter
