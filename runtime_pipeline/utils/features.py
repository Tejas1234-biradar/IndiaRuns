"""
Runtime feature extraction for the XGBoost ranker.

Self-contained (no offline_pipeline imports) so the Docker runtime image stays lean.
"""

from __future__ import annotations

from datetime import datetime

# Must match offline_pipeline/feature_engineering/feature_schema.py
FEATURE_COLUMNS = [
    "years_of_experience",
    "num_previous_jobs",
    "faiss_distance_to_jd",
    "num_skills_listed",
    "max_assessment_score",
    "recruiter_response_rate",
    "interview_completion_rate",
    "github_activity_score",
    "days_since_active",
    "profile_views_received_30d",
    "avg_job_duration_months",
    "notice_period_days",
]

REFERENCE_DATE = datetime(2026, 6, 9)
SENTINEL = -1.0


def _days_since(date_str: str) -> int:
    if not date_str:
        return 365
    try:
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        return max(0, (REFERENCE_DATE - dt).days)
    except ValueError:
        return 365


def build_features_from_raw(raw: dict, faiss_score: float) -> dict:
    """Extract the 12 model features from a raw candidate JSON record."""
    profile = raw.get("profile", {})
    signals = raw.get("redrob_signals", {})
    skills = raw.get("skills", [])
    career = raw.get("career_history", [])

    years_exp = min(float(profile.get("years_of_experience", 0.0)), 40.0)
    num_jobs = len(career)
    num_skills = min(len(skills), 100)

    assessments = signals.get("skill_assessment_scores", {}) or {}
    max_assessment = (
        float(sum(assessments.values()) / len(assessments)) if assessments else 0.0
    )

    github = float(signals.get("github_activity_score", SENTINEL))
    recruiter = float(signals.get("recruiter_response_rate", SENTINEL))
    interview = float(signals.get("interview_completion_rate", SENTINEL))

    row = {
        "years_of_experience": years_exp,
        "num_previous_jobs": num_jobs,
        "faiss_distance_to_jd": float(faiss_score),
        "num_skills_listed": num_skills,
        "max_assessment_score": max_assessment,
        "recruiter_response_rate": recruiter,
        "interview_completion_rate": interview,
        "github_activity_score": github,
        "days_since_active": _days_since(signals.get("last_active_date", "")),
        "profile_views_received_30d": int(signals.get("profile_views_received_30d", 0)),
        "avg_job_duration_months": (years_exp * 12) / max(num_jobs, 1),
        "notice_period_days": int(signals.get("notice_period_days", 30)),
    }

    # Replace sentinel missing markers with NaN for imputation
    for col in ("github_activity_score", "recruiter_response_rate", "interview_completion_rate"):
        if row[col] == SENTINEL:
            row[col] = float("nan")

    return row


def apply_imputation(rows: list[dict], metadata: dict | None = None) -> list[dict]:
    """Apply schema imputation rules using optional metadata means/medians."""
    meta = (metadata or {}).get("features", {})
    imputed = [dict(r) for r in rows]

    def stat(col: str, kind: str, default: float) -> float:
        info = meta.get(col, {})
        if kind == "mean" and "mean" in info:
            return float(info["mean"])
        if kind == "median" and "mean" in info:  # metadata uses mean for numeric cols
            return float(info.get("mean", default))
        return default

    for row in imputed:
        if row.get("github_activity_score") != row.get("github_activity_score"):
            row["github_activity_score"] = 0.0
        if row.get("recruiter_response_rate") != row.get("recruiter_response_rate"):
            row["recruiter_response_rate"] = stat("recruiter_response_rate", "mean", 0.44)
        if row.get("interview_completion_rate") != row.get("interview_completion_rate"):
            row["interview_completion_rate"] = 0.0
        if row.get("avg_job_duration_months") != row.get("avg_job_duration_months"):
            row["avg_job_duration_months"] = stat("avg_job_duration_months", "mean", 28.0)

    return imputed
