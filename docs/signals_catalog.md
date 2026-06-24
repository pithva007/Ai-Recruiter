# Redrob Signals Catalog

## Overview

The 23 `redrob_signals` fields are behavioral platform signals — they measure **candidate availability and engagement**, not technical skill. They function as a **multiplier layer** over the career/skill scoring. A technically perfect candidate with zero availability signals should rank below a slightly less technical candidate who is clearly active and reachable.

---

## Full Signal Catalog

### Signal 1 — profile_completeness_score
```json
{
  "signal_name": "profile_completeness_score",
  "signal_type": "platform_quality",
  "range": "0–100",
  "ranking_importance": "medium",
  "extraction_strategy": "Use directly. Normalize: score / 100.",
  "confidence_strategy": "High confidence — computed by platform.",
  "score_mapping": {
    "fit_score": false,
    "impact_score": false,
    "potential_score": false,
    "risk_score": true,
    "behavioral_multiplier": true
  },
  "notes": "<50 is a yellow flag. >80 is a positive signal. Incomplete profiles suggest either low platform engagement or inability to present oneself clearly."
}
```

### Signal 2 — signup_date
```json
{
  "signal_name": "signup_date",
  "signal_type": "platform_activity",
  "range": "date string",
  "ranking_importance": "low",
  "extraction_strategy": "Compute days_since_signup = (today - signup_date). Use as context only.",
  "confidence_strategy": "Low importance — new signup may be very active; old signup may be inactive.",
  "score_mapping": {
    "behavioral_multiplier": false,
    "contextual_only": true
  },
  "notes": "Useful only in combination with last_active_date. Long tenure + recent activity = established user."
}
```

### Signal 3 — last_active_date
```json
{
  "signal_name": "last_active_date",
  "signal_type": "availability",
  "range": "date string",
  "ranking_importance": "high",
  "extraction_strategy": "Compute days_since_active = (today - last_active_date). Tier: 0-30d=1.0, 31-60d=0.85, 61-90d=0.70, 91-180d=0.50, >180d=0.25",
  "confidence_strategy": "High confidence — directly observable.",
  "score_mapping": {
    "behavioral_multiplier": true,
    "risk_score": true
  },
  "notes": ">90 days inactive is a strong down-weight. The JD explicitly states inactive candidates are 'for hiring purposes, not actually available.'"
}
```

### Signal 4 — open_to_work_flag
```json
{
  "signal_name": "open_to_work_flag",
  "signal_type": "availability",
  "range": "boolean",
  "ranking_importance": "high",
  "extraction_strategy": "Binary: true=1.0 multiplier, false=0.7 multiplier. Do NOT hard-exclude false — candidate may still be reachable.",
  "confidence_strategy": "High confidence — explicit candidate declaration.",
  "score_mapping": {
    "behavioral_multiplier": true,
    "fit_score": false
  },
  "notes": "false reduces but does not eliminate a candidate. Strong candidates who aren't flagged open_to_work can still be contacted."
}
```

### Signal 5 — profile_views_received_30d
```json
{
  "signal_name": "profile_views_received_30d",
  "signal_type": "market_demand",
  "range": "integer >= 0",
  "ranking_importance": "low-medium",
  "extraction_strategy": "Normalize using log1p(views) / log1p(max_views). Use as soft signal.",
  "confidence_strategy": "Medium — views reflect recruiter interest but can be gamed or noisy.",
  "score_mapping": {
    "behavioral_multiplier": false,
    "tie_breaker": true
  },
  "notes": "High views = other recruiters noticed. Useful as a tie-breaker."
}
```

### Signal 6 — applications_submitted_30d
```json
{
  "signal_name": "applications_submitted_30d",
  "signal_type": "availability",
  "range": "integer >= 0",
  "ranking_importance": "medium",
  "extraction_strategy": ">0 = actively searching. Score: 0=0.0, 1-3=0.5, 4-7=0.8, 8+=1.0 (normalized).",
  "confidence_strategy": "High confidence — platform-observed behavior.",
  "score_mapping": {
    "behavioral_multiplier": true
  },
  "notes": "Active job seekers are more likely to respond and be available quickly."
}
```

### Signal 7 — recruiter_response_rate
```json
{
  "signal_name": "recruiter_response_rate",
  "signal_type": "reachability",
  "range": "0.0–1.0",
  "ranking_importance": "high",
  "extraction_strategy": "Use directly as multiplier. Key thresholds: >0.7=excellent, 0.4-0.7=good, 0.2-0.4=fair, <0.2=poor.",
  "confidence_strategy": "High confidence — computed from historical platform interactions.",
  "score_mapping": {
    "behavioral_multiplier": true,
    "risk_score": true
  },
  "notes": "The single most important reachability signal. A candidate who doesn't respond to recruiters cannot be hired regardless of skills. Sample submission reasoning explicitly calls this out."
}
```

### Signal 8 — avg_response_time_hours
```json
{
  "signal_name": "avg_response_time_hours",
  "signal_type": "reachability",
  "range": "float >= 0",
  "ranking_importance": "medium",
  "extraction_strategy": "Invert and normalize: score = 1 - (hours / 336). Cap at 336 hours (2 weeks). Tier: <24h=1.0, 24-72h=0.8, 72-168h=0.6, 168-336h=0.4, >336h=0.2.",
  "confidence_strategy": "High confidence — platform-computed.",
  "score_mapping": {
    "behavioral_multiplier": true
  },
  "notes": "Use in combination with recruiter_response_rate, not in isolation."
}
```

### Signal 9 — skill_assessment_scores
```json
{
  "signal_name": "skill_assessment_scores",
  "signal_type": "verified_skills",
  "range": "dict[skill_name, 0-100]",
  "ranking_importance": "high",
  "extraction_strategy": "For each JD-relevant skill with an assessment score: use score/100 to override or augment self-reported proficiency. Average relevant skill assessment scores.",
  "confidence_strategy": "HIGHEST confidence — platform-verified, not self-reported.",
  "score_mapping": {
    "fit_score": true,
    "potential_score": true
  },
  "notes": "Empty dict = no assessments completed. Presence of relevant skill assessments is itself a positive signal (they invested time to verify skills). Use as a trust multiplier on the skills scoring."
}
```

### Signal 10 — connection_count
```json
{
  "signal_name": "connection_count",
  "signal_type": "network",
  "range": "integer >= 0",
  "ranking_importance": "low",
  "extraction_strategy": "Normalize: log1p(connections) / log1p(1000). Soft signal only.",
  "confidence_strategy": "Low — network size weakly correlates with professional quality.",
  "score_mapping": {
    "tie_breaker": true
  },
  "notes": "Low importance. Use as a very soft signal or ignore."
}
```

### Signal 11 — endorsements_received
```json
{
  "signal_name": "endorsements_received",
  "signal_type": "social_proof",
  "range": "integer >= 0",
  "ranking_importance": "low",
  "extraction_strategy": "Normalize: log1p(endorsements) / log1p(500). Soft signal.",
  "confidence_strategy": "Low — easily gamed, noisy.",
  "score_mapping": {
    "tie_breaker": true
  },
  "notes": "Weak signal. Cross-reference with endorsements on specific skills in skills[]."
}
```

### Signal 12 — notice_period_days
```json
{
  "signal_name": "notice_period_days",
  "signal_type": "availability",
  "range": "0–180",
  "ranking_importance": "high",
  "extraction_strategy": "Tier: 0-15d=1.0, 16-30d=0.95, 31-60d=0.80, 61-90d=0.65, 91-120d=0.45, >120d=0.30.",
  "confidence_strategy": "High — self-declared, but JD clearly states preference for sub-30 days.",
  "score_mapping": {
    "fit_score": true,
    "behavioral_multiplier": true
  },
  "notes": "JD explicitly says 'we'd love sub-30-day notice. We can buy out up to 30 days. 30+ day notice candidates are still in scope but the bar gets higher.' This is a meaningful scoring signal."
}
```

### Signal 13 — expected_salary_range_inr_lpa
```json
{
  "signal_name": "expected_salary_range_inr_lpa",
  "signal_type": "logistics",
  "range": "{min: float, max: float}",
  "ranking_importance": "medium",
  "extraction_strategy": "Use salary_mid = (min + max) / 2. A Series A company in India typically offers 30-80 LPA for this role. Fit: 20-100 LPA = reasonable. <10 LPA = possibly junior. >150 LPA = possibly overpriced.",
  "confidence_strategy": "Medium — self-declared, but indicates candidate's market perception.",
  "score_mapping": {
    "fit_score": false,
    "risk_score": true
  },
  "notes": "Very low or very high salary expectations are risk signals. Use as a soft modifier, not a hard filter."
}
```

### Signal 14 — preferred_work_mode
```json
{
  "signal_name": "preferred_work_mode",
  "signal_type": "logistics",
  "range": "onsite | hybrid | remote | flexible",
  "ranking_importance": "medium",
  "extraction_strategy": "JD is hybrid (Pune/Noida). Score: hybrid=1.0, onsite=0.9, flexible=0.85, remote=0.6.",
  "confidence_strategy": "High — explicit candidate preference.",
  "score_mapping": {
    "fit_score": true
  },
  "notes": "Remote-only candidates are a soft mismatch for this hybrid JD. Do not hard-exclude — many remote candidates are willing to adjust."
}
```

### Signal 15 — willing_to_relocate
```json
{
  "signal_name": "willing_to_relocate",
  "signal_type": "logistics",
  "range": "boolean",
  "ranking_importance": "medium",
  "extraction_strategy": "If not in India or not in target city AND willing_to_relocate=true: no penalty. If not in target city AND false: soft penalty.",
  "confidence_strategy": "High — explicit declaration.",
  "score_mapping": {
    "fit_score": true
  },
  "notes": "Combine with profile.location and profile.country for the full location fit picture."
}
```

### Signal 16 — github_activity_score
```json
{
  "signal_name": "github_activity_score",
  "signal_type": "technical_credibility",
  "range": "-1 to 100",
  "ranking_importance": "high",
  "extraction_strategy": "-1 = no GitHub = neutral (0.0 bonus, no penalty). 0-10 = minimal activity. 10-50 = moderate. 50-100 = highly active. Score: -1→0.5 (neutral), 0-10→0.55, 10-50→0.7, 50-100→1.0.",
  "confidence_strategy": "High — platform-computed from actual GitHub activity.",
  "score_mapping": {
    "potential_score": true,
    "fit_score": true
  },
  "notes": "JD mentions 'open-source contributions in AI/ML space' as a nice-to-have. High github score is a strong positive for this role specifically. -1 is neutral (no GitHub linked), not negative."
}
```

### Signal 17 — search_appearance_30d
```json
{
  "signal_name": "search_appearance_30d",
  "signal_type": "market_visibility",
  "range": "integer >= 0",
  "ranking_importance": "low",
  "extraction_strategy": "Normalize: log1p(appearances) / log1p(500). Tie-breaker only.",
  "confidence_strategy": "Medium — reflects platform search algorithm, not just candidate quality.",
  "score_mapping": {
    "tie_breaker": true
  },
  "notes": "High appearances mean the platform's own algorithm is surfacing them. Useful as a confidence signal."
}
```

### Signal 18 — saved_by_recruiters_30d
```json
{
  "signal_name": "saved_by_recruiters_30d",
  "signal_type": "market_demand",
  "range": "integer >= 0",
  "ranking_importance": "medium",
  "extraction_strategy": "Normalize: log1p(saves) / log1p(50). Moderate signal weight.",
  "confidence_strategy": "High — reflects actual recruiter behavior.",
  "score_mapping": {
    "behavioral_multiplier": true,
    "tie_breaker": true
  },
  "notes": "Other recruiters bookmarking this candidate is a strong market validation signal."
}
```

### Signal 19 — interview_completion_rate
```json
{
  "signal_name": "interview_completion_rate",
  "signal_type": "reliability",
  "range": "0.0–1.0",
  "ranking_importance": "medium",
  "extraction_strategy": "Use directly. Tier: >0.8=1.0, 0.6-0.8=0.85, 0.4-0.6=0.7, 0.2-0.4=0.5, <0.2=0.3.",
  "confidence_strategy": "High — platform-observed.",
  "score_mapping": {
    "behavioral_multiplier": true,
    "risk_score": true
  },
  "notes": "Candidates who ghost interviews waste recruiter time. Low completion rate is a reliability risk."
}
```

### Signal 20 — offer_acceptance_rate
```json
{
  "signal_name": "offer_acceptance_rate",
  "signal_type": "intent",
  "range": "-1 to 1.0",
  "ranking_importance": "medium",
  "extraction_strategy": "-1 = no prior offers = neutral (score: 0.5). 0.0-0.4 = declines most offers. 0.4-0.7 = selective. 0.7-1.0 = tends to accept. Use: -1→0.5, else offer_acceptance_rate directly.",
  "confidence_strategy": "High for candidates with history; uncertain for -1.",
  "score_mapping": {
    "behavioral_multiplier": true,
    "risk_score": true
  },
  "notes": "Very low acceptance rate may indicate a candidate who is using offers as leverage. -1 is neutral — no history to judge."
}
```

### Signal 21 — verified_email
```json
{
  "signal_name": "verified_email",
  "signal_type": "trust",
  "range": "boolean",
  "ranking_importance": "low",
  "extraction_strategy": "Binary soft bonus: true=1.0, false=0.9.",
  "confidence_strategy": "High — platform-verified.",
  "score_mapping": {
    "behavioral_multiplier": false,
    "tie_breaker": true
  },
  "notes": "Unverified email is a minor risk. Verified is a positive trust signal."
}
```

### Signal 22 — verified_phone
```json
{
  "signal_name": "verified_phone",
  "signal_type": "trust",
  "range": "boolean",
  "ranking_importance": "low",
  "extraction_strategy": "Binary soft bonus: true=1.0, false=0.95.",
  "confidence_strategy": "High — platform-verified.",
  "score_mapping": {
    "tie_breaker": true
  },
  "notes": "Phone verification is slightly more meaningful than email for recruiter reachability."
}
```

### Signal 23 — linkedin_connected
```json
{
  "signal_name": "linkedin_connected",
  "signal_type": "trust",
  "range": "boolean",
  "ranking_importance": "low",
  "extraction_strategy": "Binary soft bonus: true=1.0, false=0.95.",
  "confidence_strategy": "High — platform-verified.",
  "score_mapping": {
    "tie_breaker": true
  },
  "notes": "LinkedIn connection enables cross-platform profile verification."
}
```

---

## Signal-to-Score Mapping Summary

### Career/Fit Score (primary scoring component)
- `career_history[].title` — ML/AI keyword presence
- `career_history[].description` — retrieval/ranking/embedding evidence
- `profile.current_title` — title classification
- `profile.years_of_experience` — experience band fit
- `redrob_signals.skill_assessment_scores` — verified skill depth
- `redrob_signals.notice_period_days` — logistics fit
- `redrob_signals.preferred_work_mode` — logistics fit
- `redrob_signals.willing_to_relocate` + `profile.location` — location fit

### Skill Depth Score (anti-stuffer component)
- `skills[].name` × `skills[].proficiency` × `skills[].duration_months` × `skills[].endorsements`
- `redrob_signals.skill_assessment_scores` (overrides self-reported)
- `redrob_signals.github_activity_score`

### Potential Score (growth trajectory component)
- Career title progression across `career_history[]`
- `certifications[]` with AI/ML relevance
- `redrob_signals.github_activity_score`
- Education tier

### Behavioral Availability Score (multiplier layer)
- `redrob_signals.open_to_work_flag`
- `redrob_signals.last_active_date` → days_since_active
- `redrob_signals.recruiter_response_rate`
- `redrob_signals.avg_response_time_hours`
- `redrob_signals.interview_completion_rate`
- `redrob_signals.applications_submitted_30d`
- `redrob_signals.profile_completeness_score`

### Risk Score (penalty component)
- Services-company-only career
- No ML/AI in career history
- Skills with duration_months = 0 (keyword stuffers)
- `redrob_signals.notice_period_days` > 90
- `redrob_signals.recruiter_response_rate` < 0.2
- Honeypot signals (impossible profile data)

---

## Behavioral Multiplier Formula

```python
def compute_behavioral_multiplier(signals: dict) -> float:
    """
    Combines availability signals into a single multiplier [0.1, 1.0].
    Applied to the raw skill/career score.
    """
    score = 0.0
    
    # Recency of activity (30% of multiplier)
    days_since_active = (today - signals['last_active_date']).days
    if days_since_active <= 30:    activity_score = 1.0
    elif days_since_active <= 60:  activity_score = 0.85
    elif days_since_active <= 90:  activity_score = 0.70
    elif days_since_active <= 180: activity_score = 0.50
    else:                          activity_score = 0.25
    score += activity_score * 0.30
    
    # Open to work (20% of multiplier)
    score += (1.0 if signals['open_to_work_flag'] else 0.6) * 0.20
    
    # Recruiter response rate (25% of multiplier)
    score += signals['recruiter_response_rate'] * 0.25
    
    # Interview completion rate (15% of multiplier)
    score += signals['interview_completion_rate'] * 0.15
    
    # Active applications (10% of multiplier)
    app_score = min(signals['applications_submitted_30d'] / 5.0, 1.0)
    score += app_score * 0.10
    
    return max(0.1, score)  # Never go to zero — even inactive candidates may be reachable
```
