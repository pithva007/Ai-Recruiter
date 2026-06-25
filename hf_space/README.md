---
title: AI Recruiter Redrob Demo
emoji: 🎯
colorFrom: red
colorTo: orange
sdk: streamlit
sdk_version: 1.41.0
app_file: app.py
pinned: false
python_version: "3.11"
---

# AI Recruiter — Redrob Hackathon Demo

Upload a JSONL or JSON file of up to 100 candidate profiles (same schema as
the hackathon bundle) and get a ranked submission CSV instantly.

**Scoring formula (v4):**
```
(career×0.30 + skill_trust×0.20 + retrieval×0.30 + fit×0.20)
  × availability_multiplier
  × services_penalty        (5-tier: 0.05 – 1.00)
  × title_relevance_gate
  × must_have_coverage_gate (P2: NDCG@10 calibration)
  × honeypot_suspicion      (P0-b: soft borderline penalty)
```

**Honeypot detection:** 7 hard signals — including founding-date checks
for Sarvam AI / Krutrim, expert-skill cap (≥12), and tenure inflation.

**Compute:** CPU only, no network, <5 min, <16 GB RAM.

**GitHub:** https://github.com/pithva007/Ai-Recruiter
