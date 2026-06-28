"""
offline_pipeline/explainer/phrasing_engine.py

Task 3.6 - Deterministic Phrasing Engine
Compiles raw SHAP signals and raw numeric features into customized, 
non-templated natural language justifications for recruiters.
"""

import random

# Checklist [x]: Define a wide vocabulary matrix of descriptive grammar blocks
# Checklist [x]: Ensure explicit mentions of dynamic numerical values ({val})
GRAMMAR_MATRIX = {
    "Years of Experience": {
        "Positive Driver": [
            "an extensive background of {val} years",
            "a robust {val}-year professional tenure",
        ],
        "Negative Driver": [
            "a relatively brief timeline of {val} years",
            "limited practical exposure ({val} years)",
        ]
    },
    "Semantic Resume Match": {
        "Positive Driver": [
            "exceptional semantic alignment with the job description",
            "highly relevant core competencies matching the JD",
        ],
        "Negative Driver": [
            "low semantic overlap with the target role",
            "missing critical keywords from the job description",
        ]
    },
    "GitHub Activity Level": {
        "Positive Driver": [
            "a strong open-source footprint (score: {val})",
            "highly active code contributions (score: {val})",
        ],
        "Negative Driver": [
            "minimal visible GitHub activity (score: {val})",
            "a sparse open-source portfolio (score: {val})",
        ]
    },
    "Availability/Notice Period": {
        "Positive Driver": [
            "a highly favorable notice period of {val} days",
            "immediate availability ({val} days)",
        ],
        "Negative Driver": [
            "a restrictive {val}-day notice period",
            "delayed onboarding availability ({val} days)",
        ]
    },
    "Average Job Tenure": {
        "Positive Driver": [
            "demonstrated loyalty with {val} months average tenure",
            "stable career progression ({val} months/job)",
        ],
        "Negative Driver": [
            "a history of short stints ({val} months average)",
            "frequent job transitions averaging {val} months",
        ]
    },
    "Skills Listed": {
        "Positive Driver": [
            "a comprehensive technical stack of {val} skills",
            "broad technological proficiency ({val} listed skills)",
        ],
        "Negative Driver": [
            "a narrow recorded skill set ({val} skills)",
            "limited listed technical competencies ({val} skills)",
        ]
    }
}