# LLM Teacher prompt templates and evaluation rubric matching job_description.docx

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
3. Score each dimension on a scale of 0.0 to 10.0:
   - `technical_fit`: Code quality, Python, retrieval, evaluation, and search.
   - `production_fit`: Experience shipping to real users, product mindset.
   - `career_fit`: Tenure stability, startup alignment, target years of experience.
   - `availability_fit`: Notice period, platform activity, location compatibility.
   - `credibility`: Consistency of profile details, lack of keyword stuffing or anomalies.
4. Determine `hard_reject` (true/false) based on the disqualifiers.
5. Calculate `overall_score` (0.0 to 10.0) as a weighted evaluation:
   - If `hard_reject` is true, maximum overall_score is 1.0.
   - Otherwise, weight technical/production/career/availability fit according to importance (suggested weights: 35% Tech Fit, 30% Production Fit, 15% Career Fit, 20% Availability/Logistics).

You MUST output your evaluation in the following JSON schema format exactly. Do not include markdown code block syntax or any additional text.

JSON Schema:
{
  "overall_score": float,  // 0.0 to 10.0
  "technical_fit": float,  // 0.0 to 10.0
  "production_fit": float, // 0.0 to 10.0
  "career_fit": float,     // 0.0 to 10.0
  "availability_fit": float, // 0.0 to 10.0
  "credibility": float,    // 0.0 to 10.0
  "hard_reject": boolean,  // true if any strict disqualifier is met
  "evidence": [string],    // List of key positive observations
  "concerns": [string]     // List of key negative observations / risks
}
"""

USER_PROMPT_TEMPLATE = """Please evaluate the candidate profile below.

### Candidate ID: {candidate_id}

### Profile Info:
Headline: {headline}
Summary: {summary}
Years of Experience: {years_of_experience}
Location: {location}, {country}

### Career History:
{career_history_summary}

### Skills & Assessments:
Skills: {skills_summary}
Redrob Skill Assessment Scores: {assessments_summary}

### Platform Activity Signals:
Notice Period: {notice_period_days} days
Recruiter Response Rate: {recruiter_response_rate:.2f}
Average Response Time: {avg_response_time_hours} hours
Last Active Date: {last_active_date}
Signup Date: {signup_date}
Open To Work: {open_to_work}
Willing To Relocate: {willing_to_relocate}
GitHub Activity Score: {github_activity_score}
Offer Acceptance Rate: {offer_acceptance_rate}
Connection Count: {connection_count}
Profile Views (30d): {profile_views}
Saved by Recruiters (30d): {saved_by_recruiters}
Interview Completion Rate: {interview_completion_rate:.2f}

Evaluate carefully and output ONLY the JSON object.
"""

def format_candidate_for_prompt(candidate):
    """
    Format a raw candidate JSON record from candidates.jsonl into user prompt variables.
    """
    cid = candidate.get("candidate_id")
    profile = candidate.get("profile", {})
    history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    
    # Format career history
    history_lines = []
    for job in history:
        company = job.get("company", "Unknown")
        title = job.get("title", "Unknown")
        duration = job.get("duration_months", 0)
        desc = job.get("description", "")
        industry = job.get("industry", "Unknown")
        company_size = job.get("company_size", "Unknown")
        history_lines.append(
            f"- Title: {title} | Company: {company} ({company_size}, {industry}) | Duration: {duration} months\n"
            f"  Description: {desc}"
        )
    career_history_summary = "\n".join(history_lines) if history_lines else "No work experience listed."
    
    # Format skills
    skills_summary = ", ".join([f"{s.get('name')} ({s.get('proficiency', '')}, dur: {s.get('duration_months', 0)}m)" for s in skills])
    
    # Format assessments
    assessments = signals.get("skill_assessment_scores", {})
    assessments_summary = ", ".join([f"{k}: {v}" for k, v in assessments.items()]) if assessments else "No Redrob platform assessments completed."
    
    # Format github and offer accept sentinel indicators
    github_act = signals.get("github_activity_score", -1)
    github_score_str = str(github_act) if github_act != -1 else "No GitHub account linked"
    
    offer_acc = signals.get("offer_acceptance_rate", -1.0)
    offer_acc_str = f"{offer_acc*100:.1f}%" if offer_acc != -1.0 else "No prior offers history"
    
    formatted_prompt = USER_PROMPT_TEMPLATE.format(
        candidate_id=cid,
        headline=profile.get("headline", ""),
        summary=profile.get("summary", ""),
        years_of_experience=profile.get("years_of_experience", 0),
        location=profile.get("location", ""),
        country=profile.get("country", ""),
        career_history_summary=career_history_summary,
        skills_summary=skills_summary,
        assessments_summary=assessments_summary,
        notice_period_days=signals.get("notice_period_days", 0),
        recruiter_response_rate=signals.get("recruiter_response_rate", 0.0),
        avg_response_time_hours=signals.get("avg_response_time_hours", 0.0),
        last_active_date=signals.get("last_active_date", ""),
        signup_date=signals.get("signup_date", ""),
        open_to_work="Yes" if signals.get("open_to_work_flag") else "No",
        willing_to_relocate="Yes" if signals.get("willing_to_relocate") else "No",
        github_activity_score=github_score_str,
        offer_acceptance_rate=offer_acc_str,
        connection_count=signals.get("connection_count", 0),
        profile_views=signals.get("profile_views_received_30d", 0),
        saved_by_recruiters=signals.get("saved_by_recruiters_30d", 0),
        interview_completion_rate=signals.get("interview_completion_rate", 0.0)
    )
    
    return formatted_prompt
