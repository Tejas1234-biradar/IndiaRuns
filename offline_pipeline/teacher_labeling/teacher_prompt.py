import json
from pydantic import BaseModel, Field
from typing import List

# --- SCHEMA ENFORCEMENT ---
class TeacherEvaluationSchema(BaseModel):
    overall_score: float = Field(
        description="Weighted final evaluation from 0.0 to 10.0. If hard_reject is true, maximum score is 1.0.",
        ge=0.0, le=10.0
    )
    technical_fit: float = Field(
        description="Code quality, Python, retrieval, evaluation, and search.",
        ge=0.0, le=10.0
    )
    production_fit: float = Field(
        description="Experience shipping to real users, product mindset.",
        ge=0.0, le=10.0
    )
    career_fit: float = Field(
        description="Tenure stability, startup alignment, target years of experience.",
        ge=0.0, le=10.0
    )
    availability_fit: float = Field(
        description="Notice period, platform activity, location compatibility.",
        ge=0.0, le=10.0
    )
    credibility: float = Field(
        description="Consistency of profile details, lack of keyword stuffing or anomalies.",
        ge=0.0, le=10.0
    )
    hard_reject: bool = Field(description="True if any strict disqualifier is met.")
    evidence: List[str] = Field(description="List of key positive observations.")
    concerns: List[str] = Field(description="List of key negative observations or risks.")

# --- PROMPT TEMPLATES ---
SYSTEM_PROMPT = """You are the Talent Acquisition and AI Engineering Leadership team at Redrob AI, a Series A AI-native talent intelligence platform.
Your task is to evaluate a candidate's profile against our Job Description (JD) for the Senior AI Engineer (Founding Team) role.

Below is the Job Description summary and grading guidelines:

### Job Requirements
- Experience: Target range of 5-9 years (though high quality outside this range is acceptable).
- Hard Tech Skills:
  - Strong Python coding skills.
  - Production experience with embeddings-based retrieval systems (e.g. sentence-transformers, OpenAI embeddings, BGE, E5, etc.).
  - Production experience with vector databases or hybrid search (e.g. FAISS, Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, etc.).
  - Hands-on experience designing offline/online ranking evaluation frameworks (NDCG, MAP, MRR, A/B test setup).
- Nice-to-Haves: LLM fine-tuning (LoRA, QLoRA, PEFT), learning-to-rank models (XGBoost ranker), HR-tech exposure.
- Core Vibe: "Shipper" attitude. Scrappy, willing to build and deploy suboptimal systems fast and iterate, rather than a pure academic researcher.

### Disqualifiers (Must flag as hard_reject: true)
1. **Pure Research**: Worked entirely in academic labs or research-only roles without any production deployment.
2. **LangChain-Only wrappers**: Primary AI experience consists only of simple LangChain wrapper projects (under 12 months) without pre-LLM ML/systems foundation.
3. **No Coding**: Senior engineers who have moved entirely into management or architecture and haven't written code in the last 18 months.
4. **Title-Chasers**: Career history showing company switches every 1-1.5 years constantly.
5. **Consulting/Service Only**: Worked ONLY at outsourcing or IT consulting companies (e.g., TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, HCL, Tech Mahindra, etc.) for their entire career. (If they have at least one product company role, they are acceptable).
6. **Non-NLP/IR Specialization**: Primary experience is entirely Computer Vision, Speech, or Robotics without NLP or Search/IR exposure.

### Availability & Logistics guidelines
- Expected notice period: <30 days is preferred (we can buy out up to 30 days). Longer notice periods (90+ days) must be penalized.
- Platform availability: Check response rates, days since last active. Inactive profiles (>6 months) should have lower scores.

### Evaluation Tasks
1. Extract positive evidence (e.g. product company experience, shipped search engines, open source).
2. Extract concerns (e.g. long notice period, consulting background, title chasing).
3. Score each dimension on a scale of 0.0 to 10.0 based on the schema definitions.
4. Determine `hard_reject` based on the disqualifiers.
5. Calculate `overall_score` (0.0 to 10.0) as a weighted evaluation:
   - If `hard_reject` is true, maximum overall_score is 1.0.
   - Otherwise, weight technical/production/career/availability fit according to importance (suggested weights: 35% Tech Fit, 30% Production Fit, 15% Career Fit, 20% Availability/Logistics).
"""

def format_candidate_prompt(candidate):
    """
    Injects candidate profile into a clean string format for the LLM.
    """
    signals = candidate.get('redrob_signals', {})
    
    prompt = f"""
    CANDIDATE PROFILE:
    
    Title: {candidate.get('current_title', 'Unknown')}
    Experience: {candidate.get('years_of_experience', 0)} years
    Companies: {', '.join(candidate.get('career_companies', []))}
    Skills: {', '.join(candidate.get('skill_names', []))}
    
    BEHAVIORAL SIGNALS:
    Notice Period: {signals.get('notice_period_days', 'Unknown')} days
    Recruiter Response Rate: {signals.get('recruiter_resp_rate', 'N/A')}
    GitHub Activity: {signals.get('github_activity_score', 'N/A')}
    Days Since Active: {signals.get('days_inactive', 'N/A')}
    
    PROFILE SUMMARY/EMBEDDING TEXT:
    {candidate.get('embedding_text', '')}
    """
    return prompt