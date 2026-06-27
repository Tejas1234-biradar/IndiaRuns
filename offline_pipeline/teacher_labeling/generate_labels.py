import os
import json
import argparse
import time
import numpy as np
import pandas as pd
from datetime import datetime
from teacher_prompt import SYSTEM_PROMPT, format_candidate_for_prompt

# Reference date matching eda_signals.py
REF_DATE = datetime(2026, 6, 22)

CONSULTING_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini", "hcl",
    "tech mahindra", "mindtree", "l&t", "ltts", "deloitte", "ey", "pwc", "kpmg",
    "tata consultancy", "wipro technologies", "infosys limited"
}

def load_candidates(file_path):
    if file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    elif file_path.endswith(".jsonl") or file_path.endswith(".jsonl.gz"):
        import gzip
        open_fn = gzip.open if file_path.endswith(".gz") else open
        candidates = []
        with open_fn(file_path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    candidates.append(json.loads(line))
        return candidates
    else:
        raise ValueError("Unsupported file format. Must be .json, .jsonl, or .jsonl.gz")

def parse_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None

def compute_mock_evaluation(candidate):
    """
    Deterministically computes a high-fidelity mock grade mirroring the LLM Teacher's rubric.
    Used for offline pipelines, sandbox execution, and fallback modes.
    """
    cid = candidate.get("candidate_id")
    profile = candidate.get("profile", {})
    history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    
    # 1. Tech Fit (Base 5.0)
    tech_fit = 5.0
    has_retrieval = False
    has_vector = False
    has_python = False
    has_eval = False
    has_nice_to_haves = False
    has_cv_speech_robotics = False
    
    skill_names = [s.get("name", "").lower() for s in skills]
    for s in skill_names:
        if any(keyword in s for keyword in ["embedding", "retrieval", "search", "information retrieval", "nlp"]):
            has_retrieval = True
        if any(keyword in s for keyword in ["vector", "faiss", "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch"]):
            has_vector = True
        if "python" in s:
            has_python = True
        if any(keyword in s for keyword in ["ndcg", "mrr", "map", "eval", "metric", "ab test", "a/b test"]):
            has_eval = True
        if any(keyword in s for keyword in ["fine-tuning", "finetuning", "lora", "qlora", "peft", "xgboost", "lightgbm", "ranker", "learning to rank"]):
            has_nice_to_haves = True
        if any(keyword in s for keyword in ["computer vision", "speech recognition", "tts", "robotics", "speech-to-text", "image classification", "yolo"]):
            has_cv_speech_robotics = True

    if has_retrieval: tech_fit += 1.5
    if has_vector: tech_fit += 1.5
    if has_python: tech_fit += 1.0
    if has_eval: tech_fit += 1.0
    if has_nice_to_haves: tech_fit += 0.5
    if not (has_retrieval or has_vector): tech_fit -= 2.0
    tech_fit = max(0.0, min(10.0, tech_fit))
    
    # 2. Production Fit (Base 5.0)
    prod_fit = 5.0
    is_pure_consulting = True
    has_product_exp = False
    has_pure_research = True
    
    # Analyze jobs
    total_months = 0
    short_tenures = 0
    for job in history:
        comp = job.get("company", "").lower()
        title = job.get("title", "").lower()
        desc = job.get("description", "").lower()
        industry = job.get("industry", "").lower()
        duration = job.get("duration_months", 0)
        total_months += duration
        
        # Check consulting
        is_comp_consulting = any(c in comp for c in CONSULTING_COMPANIES) or "services" in industry or "consulting" in industry
        if not is_comp_consulting:
            is_pure_consulting = False
            has_product_exp = True
            
        # Check research
        is_job_research = "research" in title or "scientist" in title or "lab" in title or "thesis" in title or "academic" in title or "postdoc" in title
        is_job_prod = any(k in title or k in desc for k in ["shipped", "deployed", "production", "scale", "system", "backend", "software engineer"])
        if is_job_prod or not is_job_research:
            has_pure_research = False
            
        if duration < 12 and not job.get("is_current", False):
            short_tenures += 1
            
    if is_pure_consulting:
        prod_fit -= 3.0
    if has_product_exp:
        prod_fit += 2.0
    if has_pure_research:
        prod_fit -= 2.0
        
    # Check for shipper attitude vs pure coding
    desc_text = " ".join([j.get("description", "").lower() for j in history])
    if any(k in desc_text for k in ["production", "deploy", "scale", "aws", "docker", "pipeline"]):
        prod_fit += 1.0
        
    prod_fit = max(0.0, min(10.0, prod_fit))
    
    # 3. Career Fit (Base 6.0)
    career_fit = 6.0
    years_exp = profile.get("years_of_experience", 0)
    if 5.0 <= years_exp <= 9.0:
        career_fit += 1.5
    elif years_exp < 3.0 or years_exp > 12.0:
        career_fit -= 2.0
        
    # Job-hopping penalty
    avg_tenure = total_months / len(history) if history else 0
    if avg_tenure < 18:
        career_fit -= 2.0
        
    has_tier_1 = any(edu.get("tier") == "tier_1" for edu in candidate.get("education", []))
    if has_tier_1:
        career_fit += 1.0
        
    career_fit = max(0.0, min(10.0, career_fit))
    
    # 4. Availability Fit (Base 5.0)
    avail_fit = 5.0
    notice = signals.get("notice_period_days", 90)
    if notice < 30:
        avail_fit += 2.0
    elif notice <= 45:
        avail_fit += 0.5
    elif notice >= 90:
        avail_fit -= 2.0
        
    resp_rate = signals.get("recruiter_response_rate", 0.0)
    avail_fit += (resp_rate * 3.0)
    
    last_act_str = signals.get("last_active_date")
    last_act = parse_date(last_act_str)
    if last_act:
        days_inactive = (REF_DATE - last_act).days
        if days_inactive > 180: # 6 months
            avail_fit -= 3.0
        elif days_inactive < 30:
            avail_fit += 1.0
            
    if signals.get("open_to_work_flag"):
        avail_fit += 1.0
        
    country = profile.get("country", "").lower()
    if country != "india":
        # JD states open to Pune/Noida, India. Candidates outside India penalized unless very high fit.
        avail_fit -= 1.5
        
    avail_fit = max(0.0, min(10.0, avail_fit))
    
    # 5. Credibility (Base 8.0)
    credibility = 8.0
    
    # Anomalies
    salary_min = signals.get("expected_salary_range_inr_lpa", {}).get("min", 0)
    salary_max = signals.get("expected_salary_range_inr_lpa", {}).get("max", 0)
    
    is_salary_anom = salary_max < salary_min
    if is_salary_anom:
        credibility -= 3.0
        
    signup_d = parse_date(signals.get("signup_date"))
    if signup_d and last_act and last_act < signup_d:
        credibility -= 2.0
        
    # Skill durations gt total experience
    total_skill_months = max([sk.get("duration_months", 0) for sk in skills]) if skills else 0
    if total_skill_months > (years_exp * 12 + 6):
        credibility -= 2.0
        
    # Keyword stuffing check (lots of skills but short tenure or no descriptions)
    if len(skills) > 15 and len(history) <= 2:
        credibility -= 2.0
        
    credibility = max(0.0, min(10.0, credibility))
    
    # 6. Hard Reject Determine
    hard_reject = False
    evidence = []
    concerns = []
    
    # Strict disqualifiers
    if has_pure_research:
        hard_reject = True
        concerns.append("Disqualified: Candidate has worked exclusively in research or academic environments without production shipping experience.")
        
    if is_pure_consulting and len(history) >= 2:
        hard_reject = True
        concerns.append("Disqualified: Entire career history is in IT consulting or outsourcing services without product company experience.")
        
    if has_cv_speech_robotics and not (has_retrieval or has_vector):
        hard_reject = True
        concerns.append("Disqualified: Specializes in Computer Vision or Speech/Robotics without relevant Search/NLP experience.")
        
    if is_salary_anom:
        hard_reject = True
        concerns.append("Disqualified: Candidate expected salary max is less than min, flagging serious credibility anomaly.")
        
    if years_exp < 3.0:
        hard_reject = True
        concerns.append("Disqualified: Candidate experience is under 3 years, failing the senior level mandate.")
        
    # Positive evidence collection
    if has_retrieval and has_vector:
        evidence.append("Strong technical experience matching embeddings-based retrieval and vector databases.")
    if has_eval:
        evidence.append("Demonstrated knowledge of ranking evaluation metrics (NDCG, MAP, etc.).")
    if not is_pure_consulting:
        evidence.append("Has valuable product company engineering experience.")
    if avg_tenure >= 36:
        evidence.append("Great career stability with an average tenure over 3 years.")
    if notice < 30:
        evidence.append("Highly available with a short notice period under 30 days.")
        
    # Concerns collection
    if notice >= 90:
        concerns.append("Long notice period (90+ days) presents hiring logistics risks.")
    if short_tenures >= 2:
        concerns.append("Frequent job shifts suggest potential title-chasing behaviors.")
    if signals.get("github_activity_score") == -1:
        concerns.append("No active GitHub profile linked to the Redrob platform.")
    if last_act and (REF_DATE - last_act).days > 120:
        concerns.append("Inactive profile on Redrob for more than 4 months.")
        
    # Calculate overall score
    if hard_reject:
        overall_score = 0.5 + (0.5 * credibility / 10.0) # Scale between 0.5 and 1.0
    else:
        overall_score = (0.35 * tech_fit) + (0.30 * prod_fit) + (0.15 * career_fit) + (0.20 * avail_fit)
        # Apply slight penalty if credibility is low
        if credibility < 6.0:
            overall_score *= (credibility / 10.0 + 0.4)
            
    return {
        "overall_score": float(np.round(overall_score, 2)),
        "technical_fit": float(np.round(tech_fit, 2)),
        "production_fit": float(np.round(prod_fit, 2)),
        "career_fit": float(np.round(career_fit, 2)),
        "availability_fit": float(np.round(avail_fit, 2)),
        "credibility": float(np.round(credibility, 2)),
        "hard_reject": hard_reject,
        "evidence": evidence if evidence else ["Profile meets minimum baseline standards."],
        "concerns": concerns if concerns else ["No major concerns detected."]
    }

def run_live_llm_grading(candidate, api_provider="openai"):
    """
    Grades candidate using real API connection (OpenAI or Anthropic).
    """
    prompt = format_candidate_for_prompt(candidate)
    
    # We load dynamic client imports only when invoked to keep script lightweight.
    if api_provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        res_text = response.choices[0].message.content
    else:
        # Anthropic
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        res_text = response.content[0].text
        
    return json.loads(res_text)

def main():
    parser = argparse.ArgumentParser(description="LLM Teacher grading pipeline.")
    parser.add_argument("--input", type=str, default="sample_candidates.json", help="Path to input candidates file.")
    parser.add_argument("--output", type=str, default="labeled_candidates.json", help="Path to output file.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of candidates to process.")
    parser.add_argument("--live", action="store_true", help="Run live LLM queries instead of mock.")
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "anthropic"], help="LLM API provider.")
    args = parser.parse_args()
    
    print(f"Loading candidates from {args.input}...")
    candidates = load_candidates(args.input)
    
    if args.limit:
        candidates = candidates[:args.limit]
        print(f"Limited processing to first {args.limit} candidates.")
        
    labeled_data = []
    cache_file = "reports/labeling_cache.json"
    cache = {}
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                cache = json.load(f)
            print(f"Loaded {len(cache)} items from labeling cache.")
        except Exception:
            pass
            
    print(f"Starting grading of {len(candidates)} candidates...")
    start_time = time.time()
    
    for i, candidate in enumerate(candidates):
        cid = candidate.get("candidate_id")
        
        # Check cache first
        if cid in cache and not args.live:
            grade = cache[cid]
        else:
            if args.live:
                try:
                    # Respect rate-limit safety buffer
                    time.sleep(0.5)
                    grade = run_live_llm_grading(candidate, args.provider)
                    cache[cid] = grade
                except Exception as e:
                    print(f"Error calling LLM for {cid}: {e}. Falling back to mock grade.")
                    grade = compute_mock_evaluation(candidate)
            else:
                grade = compute_mock_evaluation(candidate)
                
        # Merge candidate info and grade results
        record = {
            "candidate_id": cid,
            "overall_score": grade.get("overall_score"),
            "technical_fit": grade.get("technical_fit"),
            "production_fit": grade.get("production_fit"),
            "career_fit": grade.get("career_fit"),
            "availability_fit": grade.get("availability_fit"),
            "credibility": grade.get("credibility"),
            "hard_reject": grade.get("hard_reject"),
            "evidence": grade.get("evidence", []),
            "concerns": grade.get("concerns", [])
        }
        
        labeled_data.append(record)
        
        if (i+1) % 10 == 0 or (i+1) == len(candidates):
            elapsed = time.time() - start_time
            print(f"Graded {i+1}/{len(candidates)} candidates. Elapsed time: {elapsed:.2f}s")
            
    # Save cache
    os.makedirs("reports", exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2)
        
    # Save output dataset
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(labeled_data, f, indent=2)
        
    print(f"Labeling run complete! Saved {len(labeled_data)} candidate grades to {args.output}")

if __name__ == "__main__":
    main()
