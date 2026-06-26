import json
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Reference date for computing recency
REF_DATE = datetime(2026, 6, 22)

def parse_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None

def analyze_candidates(file_path):
    print(f"Starting streaming analysis of {file_path}...")
    
    # Accumulators for numerical variables
    numerical_keys = [
        "profile_completeness_score",
        "profile_views_received_30d",
        "applications_submitted_30d",
        "recruiter_response_rate",
        "avg_response_time_hours",
        "connection_count",
        "endorsements_received",
        "notice_period_days",
        "search_appearance_30d",
        "saved_by_recruiters_30d",
        "interview_completion_rate",
        "github_activity_score",       # Sentinel: -1
        "offer_acceptance_rate",       # Sentinel: -1
        "salary_min",                  # From expected_salary_range_inr_lpa.min
        "salary_max",                  # From expected_salary_range_inr_lpa.max
        "years_of_experience",         # From profile.years_of_experience
        "num_skills",                  # Length of skills array
        "num_assessments",             # Length of skill_assessment_scores dict
        "avg_assessment_score",        # Mean of skill_assessment_scores dict values
        "max_assessment_score",        # Max of skill_assessment_scores dict values
        "num_jobs",                    # Length of career_history array
        "days_active",                 # last_active_date - signup_date
        "days_since_active"            # REF_DATE - last_active_date
    ]
    
    numerical_data = {k: [] for k in numerical_keys}
    
    # Sentinels / missingness trackers
    sentinel_stats = {
        "github_missing_count": 0,
        "offer_history_missing_count": 0,
        "no_assessments_count": 0,
        "no_jobs_count": 0,
    }
    
    # Accumulators for categorical variables
    categorical_keys = [
        "preferred_work_mode",
        "willing_to_relocate",
        "open_to_work_flag",
        "verified_email",
        "verified_phone",
        "linkedin_connected",
        "country"
    ]
    categorical_data = {k: {} for k in categorical_keys}
    
    # Anomaly / honeypot counts
    anomaly_counts = {
        "salary_max_lt_min": 0,
        "future_signup": 0,
        "future_active": 0,
        "active_before_signup": 0,
        "skill_duration_gt_experience": 0,
        "years_exp_anomaly": 0, # Total jobs duration sum drastically larger than years_of_experience
    }
    
    count = 0
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            
            data = json.loads(line)
            count += 1
            
            if count % 20000 == 0:
                print(f"Processed {count} candidates...")
            
            profile = data.get("profile", {})
            history = data.get("career_history", [])
            skills = data.get("skills", [])
            signals = data.get("redrob_signals", {})
            
            # --- Extract Categorical signals ---
            for cat_key in ["preferred_work_mode", "willing_to_relocate", "open_to_work_flag", "verified_email", "verified_phone", "linkedin_connected"]:
                val = signals.get(cat_key)
                val_str = str(val)
                categorical_data[cat_key][val_str] = categorical_data[cat_key].get(val_str, 0) + 1
            
            country_val = profile.get("country", "Unknown")
            categorical_data["country"][country_val] = categorical_data["country"].get(country_val, 0) + 1
            
            # --- Extract Numerical signals & parse dates ---
            # Basic profile & history features
            years_exp = profile.get("years_of_experience", 0)
            num_skills = len(skills)
            num_jobs = len(history)
            
            # Assess dict
            assess = signals.get("skill_assessment_scores", {})
            num_assess = len(assess)
            assess_scores = list(assess.values())
            avg_assess = np.mean(assess_scores) if num_assess > 0 else np.nan
            max_assess = np.max(assess_scores) if num_assess > 0 else np.nan
            
            if num_assess == 0:
                sentinel_stats["no_assessments_count"] += 1
            if num_jobs == 0:
                sentinel_stats["no_jobs_count"] += 1
                
            # Date signals
            signup_d = parse_date(signals.get("signup_date"))
            active_d = parse_date(signals.get("last_active_date"))
            
            days_active = np.nan
            days_since_active = np.nan
            
            if signup_d and active_d:
                days_active = (active_d - signup_d).days
                days_since_active = (REF_DATE - active_d).days
                
                # Check for date anomalies
                if signup_d > REF_DATE:
                    anomaly_counts["future_signup"] += 1
                if active_d > REF_DATE:
                    anomaly_counts["future_active"] += 1
                if active_d < signup_d:
                    anomaly_counts["active_before_signup"] += 1
                    
            # Salary
            salary_range = signals.get("expected_salary_range_inr_lpa", {})
            salary_min = salary_range.get("min", np.nan)
            salary_max = salary_range.get("max", np.nan)
            
            if salary_min is not None and salary_max is not None:
                if salary_max < salary_min:
                    anomaly_counts["salary_max_lt_min"] += 1
            
            # Github & Offer acceptance
            github_act = signals.get("github_activity_score", -1)
            offer_acc = signals.get("offer_acceptance_rate", -1.0)
            
            if github_act == -1:
                sentinel_stats["github_missing_count"] += 1
            if offer_acc == -1:
                sentinel_stats["offer_history_missing_count"] += 1
                
            # Experience and skill anomalies
            total_skill_months = 0
            for sk in skills:
                total_skill_months = max(total_skill_months, sk.get("duration_months", 0))
            if total_skill_months > (years_exp * 12 + 6): # Allow 6 months buffer
                anomaly_counts["skill_duration_gt_experience"] += 1
                
            # Total job duration check
            total_job_months = sum(j.get("duration_months", 0) for j in history)
            if total_job_months > (years_exp * 12 + 24): # Allow 2 years buffer
                anomaly_counts["years_exp_anomaly"] += 1
                
            # Pack all numerical metrics (use None or nan for sentinels during stat calculation)
            row_numerical = {
                "profile_completeness_score": signals.get("profile_completeness_score"),
                "profile_views_received_30d": signals.get("profile_views_received_30d"),
                "applications_submitted_30d": signals.get("applications_submitted_30d"),
                "recruiter_response_rate": signals.get("recruiter_response_rate"),
                "avg_response_time_hours": signals.get("avg_response_time_hours"),
                "connection_count": signals.get("connection_count"),
                "endorsements_received": signals.get("endorsements_received"),
                "notice_period_days": signals.get("notice_period_days"),
                "search_appearance_30d": signals.get("search_appearance_30d"),
                "saved_by_recruiters_30d": signals.get("saved_by_recruiters_30d"),
                "interview_completion_rate": signals.get("interview_completion_rate"),
                "github_activity_score": github_act if github_act != -1 else np.nan,
                "offer_acceptance_rate": offer_acc if offer_acc != -1.0 else np.nan,
                "salary_min": salary_min,
                "salary_max": salary_max,
                "years_of_experience": years_exp,
                "num_skills": num_skills,
                "num_assessments": num_assess,
                "avg_assessment_score": avg_assess,
                "max_assessment_score": max_assess,
                "num_jobs": num_jobs,
                "days_active": days_active,
                "days_since_active": days_since_active
            }
            
            for k in numerical_keys:
                numerical_data[k].append(row_numerical[k])
                
    # --- Compute statistics ---
    print("Computing summary statistics...")
    summary_rows = []
    
    df_num = pd.DataFrame(numerical_data)
    
    for col in numerical_keys:
        series = df_num[col]
        non_nan = series.dropna()
        
        count_valid = len(non_nan)
        count_missing = len(series) - count_valid
        
        # Calculate statistics
        if count_valid > 0:
            mean_val = non_nan.mean()
            std_val = non_nan.std()
            min_val = non_nan.min()
            p25 = non_nan.quantile(0.25)
            median_val = non_nan.median()
            p75 = non_nan.quantile(0.75)
            p95 = non_nan.quantile(0.95)
            max_val = non_nan.max()
            zeros_pct = (non_nan == 0).sum() / count_valid * 100
        else:
            mean_val = std_val = min_val = p25 = median_val = p75 = p95 = max_val = zeros_pct = np.nan
            
        summary_rows.append({
            "feature": col,
            "count_valid": count_valid,
            "count_missing_or_sentinel": count_missing,
            "pct_missing_or_sentinel": count_missing / count * 100,
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "p25": p25,
            "median": median_val,
            "p75": p75,
            "p95": p95,
            "max": max_val,
            "zeros_pct": zeros_pct
        })
        
    df_summary = pd.DataFrame(summary_rows)
    
    # Save reports folder structure
    os.makedirs("reports", exist_ok=True)
    
    df_summary.to_csv("reports/signal_summary.csv", index=False)
    
    # Save missingness
    missing_data = []
    for k, v in sentinel_stats.items():
        missing_data.append({"indicator": k, "count": v, "percentage": v / count * 100})
    pd.DataFrame(missing_data).to_csv("reports/missingness.csv", index=False)
    
    # Save correlation
    df_corr = df_num.corr()
    df_corr.to_csv("reports/correlations.csv")
    
    # Save categorical counts
    cat_rows = []
    for col, counts in categorical_data.items():
        for val, cnt in counts.items():
            cat_rows.append({"feature": col, "value": val, "count": cnt, "percentage": cnt / count * 100})
    df_cat = pd.DataFrame(cat_rows)
    df_cat.to_csv("reports/categorical_counts.csv", index=False)
    
    # Write markdown summary report
    print("Writing markdown summary...")
    with open("reports/eda_summary.md", "w", encoding="utf-8") as f:
        f.write("# Exploratory Data Analysis (EDA) of Redrob Behavioral Signals\n\n")
        f.write(f"Analyzed a total of **{count:,}** candidates.\n\n")
        
        f.write("## 1. Missingness & Sentinel Values\n")
        f.write("| Indicator | Count | Percentage |\n")
        f.write("| --- | --- | --- |\n")
        for k, v in sentinel_stats.items():
            f.write(f"| {k} | {v:,} | {v/count*100:.2f}% |\n")
        f.write("\n")
        
        f.write("## 2. Detected Anomalies / Potential Honeypots\n")
        f.write("| Anomaly Type | Count | Percentage |\n")
        f.write("| --- | --- | --- |\n")
        for k, v in anomaly_counts.items():
            f.write(f"| {k} | {v:,} | {v/count*100:.2f}% |\n")
        f.write("\n")
        
        f.write("## 3. Numeric Signals Summary Statistics\n")
        
        # Custom markdown table formatter to avoid 'tabulate' package dependency
        headers = list(df_summary.columns)
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for _, row in df_summary.iterrows():
            row_str = []
            for col in headers:
                val = row[col]
                if pd.isna(val):
                    row_str.append("")
                elif isinstance(val, float):
                    row_str.append(f"{val:.4f}")
                elif isinstance(val, (int, np.integer)):
                    row_str.append(f"{int(val):,}")
                else:
                    row_str.append(str(val))
            f.write("| " + " | ".join(row_str) + " |\n")
        f.write("\n\n")
        
        f.write("## 4. Key Categorical Signals\n")
        # Group by feature
        for feature, group in df_cat.groupby("feature"):
            f.write(f"### {feature}\n")
            f.write("| Value | Count | Percentage |\n")
            f.write("| --- | --- | --- |\n")
            for idx, row in group.iterrows():
                f.write(f"| {row['value']} | {row['count']:,} | {row['percentage']:.2f}% |\n")
            f.write("\n")
            
    print("EDA Complete. Reports saved to 'reports/' directory.")

if __name__ == "__main__":
    analyze_candidates("../../data/candidates.jsonl")
