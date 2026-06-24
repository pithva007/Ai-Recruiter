# utils/json_validator.py
# Pydantic validation models for pipeline I/O contracts.
# Used to validate LLM outputs before writing to disk.

from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, field_validator, model_validator


# ---------------------------------------------------------------------------
# Evidence Item (Stage 3 output — one item per evidence claim)
# ---------------------------------------------------------------------------
VALID_EVIDENCE_TYPES = {"technical", "impact", "leadership", "learning", "behavioral"}
VALID_CONFIDENCE     = {"high", "medium", "low"}


class EvidenceItem(BaseModel):
    claim:         str
    evidence_type: str
    confidence:    str
    source_text:   Optional[str] = None
    quantified:    bool

    @field_validator("claim")
    @classmethod
    def claim_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("claim must not be empty")
        return v

    @field_validator("evidence_type")
    @classmethod
    def valid_evidence_type(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in VALID_EVIDENCE_TYPES:
            raise ValueError(
                f"evidence_type must be one of {VALID_EVIDENCE_TYPES}, got {v!r}"
            )
        return v

    @field_validator("confidence")
    @classmethod
    def valid_confidence(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in VALID_CONFIDENCE:
            raise ValueError(
                f"confidence must be one of {VALID_CONFIDENCE}, got {v!r}"
            )
        return v


# ---------------------------------------------------------------------------
# JD Features (Stage 1 output)
# ---------------------------------------------------------------------------
class ExperienceRange(BaseModel):
    min_years: Optional[int] = None
    max_years: Optional[int] = None
    ideal_years: Optional[int] = None


class Location(BaseModel):
    primary_cities: list[str] = []
    country: Optional[str] = None
    work_mode: Optional[str] = None


class SkillEntry(BaseModel):
    skill: str
    context: Optional[str] = None
    importance: str = "must_have"


class Disqualifier(BaseModel):
    pattern: str
    reason: Optional[str] = None


class ImplicitRequirement(BaseModel):
    requirement: str
    reasoning: Optional[str] = None


class JDFeatures(BaseModel):
    role_title: Optional[str] = None
    company: Optional[str] = None
    seniority_level: Optional[str] = None
    experience_range: Optional[ExperienceRange] = None
    location: Optional[Location] = None
    must_have_skills: list[SkillEntry] = []
    nice_to_have_skills: list[SkillEntry] = []
    explicit_disqualifiers: list[Disqualifier] = []
    services_company_names: list[str] = []
    implicit_requirements: list[ImplicitRequirement] = []
    culture_signals: list[str] = []
    ideal_candidate_summary: Optional[str] = None
    jd_trap_warning: Optional[str] = None


# ---------------------------------------------------------------------------
# Submission row validation
# ---------------------------------------------------------------------------
class SubmissionRow(BaseModel):
    candidate_id: str
    rank: int
    score: float
    reasoning: str

    @field_validator("candidate_id")
    @classmethod
    def validate_cand_id(cls, v: str) -> str:
        import re
        if not re.match(r"^CAND_[0-9]{7}$", v):
            raise ValueError(f"candidate_id must be CAND_XXXXXXX (7 digits), got: {v!r}")
        return v

    @field_validator("rank")
    @classmethod
    def validate_rank(cls, v: int) -> int:
        if not (1 <= v <= 100):
            raise ValueError(f"rank must be 1-100, got {v}")
        return v

    @field_validator("score")
    @classmethod
    def validate_score(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"score must be 0.0-1.0, got {v}")
        return v

    @field_validator("reasoning")
    @classmethod
    def validate_reasoning(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("reasoning must not be empty")
        return v.strip()


# ---------------------------------------------------------------------------
# Reasoning output (from LLM in reason.py)
# ---------------------------------------------------------------------------
class ReasoningOutput(BaseModel):
    candidate_id: str
    reasoning: str

    @field_validator("reasoning")
    @classmethod
    def check_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("reasoning is empty")
        return v.strip()
