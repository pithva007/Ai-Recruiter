# Gap Analysis — Old Architecture vs Challenge Requirements

## Summary

The original AGENT.md, CLAUDE.md, and SKILLS.md were written for a generic AI Recruiter system with no knowledge of the actual challenge. Every file needed a complete rewrite. This document records all changes made and the challenge evidence that drove each change.

---

## Critical Gaps (would have caused disqualification or zero score)

### Gap 1: LLM per-candidate at ranking time
```json
{
  "document": "AGENT.md, CLAUDE.md, SKILLS.md",
  "section": "Agent Pipeline Stages / LLM Client Pattern",
  "current_version": "Call call_llm() for each candidate through Stages 2-6",
  "challenge_requirement": "NO LLM API calls during ranking. 100K candidates in <5 minutes on CPU with no network.",
  "recommended_fix": "LLM use is ONLY permitted for: Stage 1 (JD analysis, once), reasoning generation for top 100 (offline, not during ranking). All 100K candidate scoring must be rule-based.",
  "priority": "critical"
}
```

### Gap 2: Wrong input data format
```json
{
  "document": "SKILLS.md, AGENT.md",
  "section": "Data Schemas / Candidate Input Schema",
  "current_version": "Input: data/raw/candidates.csv with candidate_id, candidate_name, resume_text",
  "challenge_requirement": "Input: India_runs_data_and_ai_challenge/candidates.jsonl — 100K candidates, one JSON object per line, with 6 required sections (profile, career_history, education, skills, redrob_signals)",
  "recommended_fix": "Replace all CSV-based candidate ingestion with JSONL parsing. The schema has 30+ fields across 6 sections — not a simple CSV.",
  "priority": "critical"
}
```

### Gap 3: Wrong output format
```json
{
  "document": "AGENT.md",
  "section": "Output Format Contract / Output CSV Schema",
  "current_version": "Output: 19-column CSV with fit_score, impact_score, potential_score, risk_score, confidence_level, green_flags, yellow_flags, skill_gaps, dark_horse, transferable_skills_map, interview_questions, llm_rationale",
  "challenge_requirement": "Output: 4-column CSV: candidate_id, rank, score, reasoning. Exactly 100 rows. ranks 1-100. score is non-increasing float.",
  "recommended_fix": "Rewrite entire output contract. The challenge output is minimal: 4 columns, 100 rows. Internal scoring can use any format but final output must match exactly.",
  "priority": "critical"
}
```

### Gap 4: Wrong composite score formula
```json
{
  "document": "AGENT.md",
  "section": "Composite Score Formula",
  "current_version": "composite_score = (fit_score * 0.35) + (impact_score * 0.30) + (potential_score * 0.20) + ((100 - risk_score) * 0.15)",
  "challenge_requirement": "Evaluation: 0.50 × NDCG@10 + 0.30 × NDCG@50 + 0.15 × MAP + 0.05 × P@10. Internal score must be a float 0.0-1.0, non-increasing. The weights on NDCG@10 mean top-10 precision is 5x more important than any other metric.",
  "recommended_fix": "The internal scoring formula is the designer's choice, but it must produce a single float 0.0-1.0 per candidate, sorted descending for ranking. Optimize internal weights to maximize NDCG@10 (top-10 precision).",
  "priority": "critical"
}
```

### Gap 5: Missing redrob_signals entirely
```json
{
  "document": "AGENT.md, CLAUDE.md, SKILLS.md",
  "section": "All sections",
  "current_version": "No mention of redrob_signals anywhere in original documents",
  "challenge_requirement": "23 behavioral signals are first-class scoring inputs. The JD explicitly says inactive/unresponsive candidates are 'not actually available.' Signals include: open_to_work_flag, last_active_date, recruiter_response_rate, notice_period_days, github_activity_score, skill_assessment_scores, interview_completion_rate.",
  "recommended_fix": "Add full signals catalog to AGENT.md and SKILLS.md. Add behavioral multiplier formula. Redrob signals should account for ~20-25% of final score as a multiplier layer.",
  "priority": "critical"
}
```

### Gap 6: Missing honeypot awareness
```json
{
  "document": "AGENT.md, CLAUDE.md, SKILLS.md",
  "current_version": "No honeypot detection anywhere",
  "challenge_requirement": "~80 honeypot candidates with impossible profiles. >10% in top 100 = Stage 3 disqualification.",
  "recommended_fix": "Add honeypot detection logic: check for impossible experience timelines, expert skills with 0 duration, copy-pasted descriptions, etc.",
  "priority": "critical"
}
```

---

## High Priority Gaps

### Gap 7: Missing services company penalty
```json
{
  "document": "AGENT.md",
  "section": "Anti-Hallucination Rules / Scoring Contracts",
  "current_version": "No mention of company type in scoring",
  "challenge_requirement": "JD explicitly rejects candidates from pure TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini careers. This is a major discriminating signal.",
  "recommended_fix": "Add services_company_penalty to scoring. Check career_history[].company against a blacklist. Penalty weight: -20 to -40% for pure services careers.",
  "priority": "high"
}
```

### Gap 8: Missing skill depth scoring (duration_months × proficiency)
```json
{
  "document": "AGENT.md, SKILLS.md",
  "section": "Evidence Extraction Rules / Scoring",
  "current_version": "Evidence extraction used LLM to assess claims. No mention of duration_months in skills.",
  "challenge_requirement": "skills[].duration_months is the key anti-stuffer signal. A skill listed as 'expert' with duration_months=0 is a red flag. Genuine skills have duration_months > 6.",
  "recommended_fix": "Add weighted skill scoring: proficiency_weight × log1p(duration_months) × log1p(endorsements). This is a deterministic computation, not LLM inference.",
  "priority": "high"
}
```

### Gap 9: Missing education tier field
```json
{
  "document": "SKILLS.md, AGENT.md",
  "section": "Data Schemas",
  "current_version": "No education tier field mentioned",
  "challenge_requirement": "education[].tier is a pre-computed field: tier_1 through tier_4 or unknown. Institution prestige scoring should use this, not raw institution name.",
  "recommended_fix": "Use education[].tier directly: tier_1=1.0, tier_2=0.8, tier_3=0.6, tier_4=0.4, unknown=0.5.",
  "priority": "high"
}
```

### Gap 10: Wrong dark horse definition
```json
{
  "document": "AGENT.md",
  "section": "Dark Horse Definition",
  "current_version": "Dark horse = vector similarity rank > 15 AND impact/potential >= 75",
  "challenge_requirement": "No explicit 'dark horse' concept in challenge. The equivalent is: candidates whose career titles are non-obvious (Data Engineer transitioning to ML) but whose career descriptions show deep ML/retrieval work. The challenge warns against over-ranking keyword-matching candidates — the 'dark horse' is someone who does the actual work without the ML title.",
  "recommended_fix": "Rename to 'transferable_fit_candidate'. Detect via: career history descriptions contain ML/retrieval keywords even when current_title is not ML. These should be ranked higher than candidates with ML titles but no production evidence.",
  "priority": "high"
}
```

### Gap 11: Wrong candidate ID format assumed
```json
{
  "document": "SKILLS.md, AGENT.md",
  "section": "Data Schemas",
  "current_version": "candidate_id: 'string (required, unique)' — no format constraint",
  "challenge_requirement": "candidate_id must match CAND_[0-9]{7} — exactly CAND_ followed by 7 digits. The validator rejects non-matching IDs.",
  "recommended_fix": "All ID generation and validation must enforce this pattern.",
  "priority": "high"
}
```

---

## Medium Priority Gaps

### Gap 12: Missing location fit scoring
```json
{
  "document": "AGENT.md, SKILLS.md",
  "current_version": "No location scoring",
  "challenge_requirement": "JD is Pune/Noida with hybrid work. India-based candidates with willing_to_relocate=true in target cities get a bonus. Non-India, non-relocatable candidates get a soft penalty.",
  "recommended_fix": "Add location_fit_score component using profile.location + profile.country + willing_to_relocate + preferred_work_mode.",
  "priority": "medium"
}
```

### Gap 13: Missing notice_period scoring
```json
{
  "document": "AGENT.md",
  "current_version": "notice_period not mentioned in scoring contracts",
  "challenge_requirement": "JD explicitly says notice period matters. Sub-30 days ideal. Up to 30 days buyable. >30 days = higher bar. >90 days = significant negative.",
  "recommended_fix": "Add notice_period_score component with tiered scoring.",
  "priority": "medium"
}
```

### Gap 14: Pipeline architecture fundamentally wrong
```json
{
  "document": "SKILLS.md",
  "section": "Project Structure",
  "current_version": "8-stage pipeline with LLM call at every stage: role_agent, candidate_agent, evidence_extraction, graph_builder, hybrid_retrieval, scoring_engine, ranking, dashboard",
  "challenge_requirement": "Correct architecture: (1) Offline pre-computation of features for 100K candidates, (2) Fast ranking scoring function < 5 min on CPU, (3) Reasoning generation for top 100 only using LLM. The 8-stage pipeline design is completely incompatible with the compute constraints.",
  "recommended_fix": "New architecture: precompute.py (offline, unlimited time) → rank.py (online, <5 min, no network) → reason.py (top 100 only, LLM OK) → validate.py → app.py (Streamlit dashboard).",
  "priority": "medium"
}
```

### Gap 15: Missing skill_assessment_scores usage
```json
{
  "document": "AGENT.md, CLAUDE.md",
  "current_version": "No mention of verified skill assessments",
  "challenge_requirement": "redrob_signals.skill_assessment_scores is a dict of platform-verified skill scores (0-100). These are the highest-confidence skill signals — they override self-reported proficiency.",
  "recommended_fix": "When skill_assessment_scores[skill_name] exists for a JD-relevant skill, use it as the primary proficiency signal instead of skills[].proficiency.",
  "priority": "medium"
}
```

---

## Low Priority Gaps

### Gap 16: Wrong embedding model purpose
```json
{
  "document": "SKILLS.md, CLAUDE.md",
  "section": "Embedding Client Pattern",
  "current_version": "all-mpnet-base-v2 used for candidate-JD matching in Stage 5",
  "challenge_requirement": "Embeddings can be used for: (a) pre-computing JD requirement embeddings, (b) computing career description embeddings to find semantic similarity to JD. But this is offline pre-computation, not per-candidate LLM calls.",
  "recommended_fix": "Embeddings are a valid pre-computation strategy. Keep sentence-transformers but use for offline batch embedding of career descriptions, not for live LLM calls.",
  "priority": "low"
}
```

### Gap 17: Score range mismatch
```json
{
  "document": "AGENT.md",
  "section": "Scoring Contracts",
  "current_version": "All scores are 0-100 integers",
  "challenge_requirement": "submission score column is a float 0.0-1.0 (implied by sample: 0.992, 0.984, etc.)",
  "recommended_fix": "Internal scores can use any range. Final submission score must be normalized to [0.0, 1.0].",
  "priority": "low"
}
```

---

## Changes Made to Project Files

### AGENT.md — Complete Rewrite
- Replaced generic agent identity with challenge-specific context
- Added actual candidate schema fields (all 6 sections)
- Added all 23 redrob_signals with scoring weights
- Added services company disqualifier list from JD
- Added honeypot detection contract
- Added JD requirement list (must-haves vs nice-to-haves vs disqualifiers)
- Replaced 8-stage LLM pipeline with challenge-compliant architecture
- Replaced 19-column output schema with 4-column submission format
- Replaced fit/impact/potential/risk with career_score/skill_score/behavioral_score/fit_score
- Added compute constraint: no LLM at ranking time

### CLAUDE.md — Complete Rewrite
- Updated model selection — retained gemini-2.5-flash but restricted to offline/pre-compute use only
- Removed per-candidate LLM calls
- Updated Stage 1 system prompt to use actual JD fields and disqualifier logic
- Replaced generic candidate understanding prompt with challenge-specific prompts
- Added reasoning generation prompt for top-100 post-ranking
- Added challenge alignment rules
- Added validation rules from validate_submission.py
- Removed scoring engine prompt (scoring is rule-based, not LLM)

### SKILLS.md — Complete Rewrite
- Updated project structure to match challenge architecture (precompute → rank → reason → validate → app)
- Added challenge dataset reference section
- Added full candidate schema mapping with all fields
- Added JD keyword lists (must-have, nice-to-have, disqualifiers, trap skills)
- Added redrob signal scoring formulas
- Added submission format contract from validate_submission.py
- Added services company blacklist
- Added honeypot detection patterns
- Replaced 8-stage pipeline contract with challenge-compliant 3-phase contract
- Added feature engineering specification
- Updated output schema to 4 columns
