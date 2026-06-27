
# GROQ CODE
import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv
from groq import Groq
from teacher_prompt import SYSTEM_PROMPT, format_candidate_prompt

# --- SYSTEM PATHS ---
INPUT_SAMPLE_PATH = "artifacts/teacher_sample.jsonl"
OUTPUT_LABELS_PATH = "artifacts/labeled_candidates.json"


# Load environment variables
load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# Groq Free Tier limit: 30 Requests Per Minute (RPM). 
# 60 seconds / 30 requests = 2.0 seconds per request. (Added 0.2s buffer).
DELAY_BETWEEN_REQUESTS = 2.2 

def main():
    print(f"Loading candidates from {INPUT_SAMPLE_PATH}...")
    all_candidates = []
    with open(INPUT_SAMPLE_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_candidates.append(json.loads(line))
    
    # --- CACHING & RESUME LOGIC ---
    labeled_data = []
    processed_ids = set()
    
    if os.path.exists(OUTPUT_LABELS_PATH):
        try:
            with open(OUTPUT_LABELS_PATH, 'r', encoding='utf-8') as f:
                labeled_data = json.load(f)
                processed_ids = {item["candidate_id"] for item in labeled_data}
                print(f"Resuming from cache. {len(processed_ids)} already processed.")
        except json.JSONDecodeError:
            print("Warning: Could not read existing labels. Starting fresh.")

    remaining = [c for c in all_candidates if c["candidate_id"] not in processed_ids]
    print(f"Starting grading for remaining {len(remaining)} candidates (Estimated time: {len(remaining) * DELAY_BETWEEN_REQUESTS / 60:.1f} minutes)...")

    for i, candidate in enumerate(remaining):
        cid = candidate.get("candidate_id", "UNKNOWN")
        prompt = format_candidate_prompt(candidate)
        
        try:
            print(f"[{i+1}/{len(remaining)}] Grading {cid} via Groq Llama 3...")
            
            # Satisfy Groq's requirement and give Llama 3 the exact schema it needs!
            groq_system_instruction = (
                SYSTEM_PROMPT + 
                "\n\nYou MUST respond strictly with a valid JSON object. Do not include markdown blocks. "
                "You must use EXACTLY the following JSON schema:\n"
                "{\n"
                '  "overall_score": float,\n'
                '  "technical_fit": float,\n'
                '  "production_fit": float,\n'
                '  "career_fit": float,\n'
                '  "availability_fit": float,\n'
                '  "credibility": float,\n'
                '  "hard_reject": boolean,\n'
                '  "evidence": ["string"],\n'
                '  "concerns": ["string"]\n'
                "}"
            )
            
            # Synchronous call to Groq
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": groq_system_instruction},
                    {"role": "user", "content": prompt}
                ],
                model="openai/gpt-oss-20b",
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(chat_completion.choices[0].message.content)
            
            record = {
                "candidate_id": cid,
                "overall_score": float(result.get("overall_score", 0.0)),
                "technical_fit": float(result.get("technical_fit", 0.0)),
                "production_fit": float(result.get("production_fit", 0.0)),
                "career_fit": float(result.get("career_fit", 0.0)),
                "availability_fit": float(result.get("availability_fit", 0.0)),
                "credibility": float(result.get("credibility", 0.0)),
                "hard_reject": bool(result.get("hard_reject", False)),
                "evidence": result.get("evidence", []),
                "concerns": result.get("concerns", [])
            }
            
            labeled_data.append(record)
            
            # SAVE AFTER EVERY CANDIDATE
            with open(OUTPUT_LABELS_PATH, 'w', encoding='utf-8') as f:
                json.dump(labeled_data, f, indent=2)
                
        except Exception as e:
            print(f"API Error for {cid}: {e}. Retrying this candidate next time.")
        
        # --- STRICT RATE LIMITING ---
        time.sleep(DELAY_BETWEEN_REQUESTS)

    print("Task 3.3 Complete. All candidates graded.")

if __name__ == "__main__":
    main()

"""
# GEMINI CODE
import os
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from teacher_prompt import SYSTEM_PROMPT, format_candidate_prompt, TeacherEvaluationSchema

# --- SYSTEM PATHS ---
INPUT_SAMPLE_PATH = "artifacts/teacher_sample.jsonl"
OUTPUT_LABELS_PATH = "artifacts/labeled_candidates.json"

# Load environment variables
load_dotenv()
client = genai.Client()

# Free tier limit: 5 Requests Per Minute (RPM). 
# 60 seconds / 5 requests = 12.0 seconds per request. (Added 0.1s buffer).
DELAY_BETWEEN_REQUESTS = 12.1 

def main():
    print(f"Loading candidates from {INPUT_SAMPLE_PATH}...")
    all_candidates = []
    with open(INPUT_SAMPLE_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_candidates.append(json.loads(line))
    
    # --- CACHING & RESUME LOGIC ---
    labeled_data = []
    processed_ids = set()
    
    if os.path.exists(OUTPUT_LABELS_PATH):
        try:
            with open(OUTPUT_LABELS_PATH, 'r', encoding='utf-8') as f:
                labeled_data = json.load(f)
                processed_ids = {item["candidate_id"] for item in labeled_data}
                print(f"Resuming from cache. {len(processed_ids)} already processed.")
        except json.JSONDecodeError:
            print("Warning: Could not read existing labels. Starting fresh.")

    # Filter out candidates we already scored
    remaining = [c for c in all_candidates if c["candidate_id"] not in processed_ids]
    print(f"Starting grading for remaining {len(remaining)} candidates (Estimated time: {len(remaining) * DELAY_BETWEEN_REQUESTS / 60:.1f} minutes)...")

    for i, candidate in enumerate(remaining):
        cid = candidate.get("candidate_id", "UNKNOWN")
        prompt = format_candidate_prompt(candidate)
        
        try:
            print(f"[{i+1}/{len(remaining)}] Grading {cid}...")
            
            # Synchronous call
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type='application/json',
                    response_schema=TeacherEvaluationSchema,
                    temperature=0.1
                ),
            )
            
            result = json.loads(response.text)
            
            record = {
                "candidate_id": cid,
                "overall_score": float(result.get("overall_score", 0.0)),
                "technical_fit": float(result.get("technical_fit", 0.0)),
                "production_fit": float(result.get("production_fit", 0.0)),
                "career_fit": float(result.get("career_fit", 0.0)),
                "availability_fit": float(result.get("availability_fit", 0.0)),
                "credibility": float(result.get("credibility", 0.0)),
                "hard_reject": bool(result.get("hard_reject", False)),
                "evidence": result.get("evidence", []),
                "concerns": result.get("concerns", [])
            }
            
            labeled_data.append(record)
            
            # SAVE AFTER EVERY CANDIDATE (No data loss if script is stopped)
            with open(OUTPUT_LABELS_PATH, 'w', encoding='utf-8') as f:
                json.dump(labeled_data, f, indent=2)
                
        except Exception as e:
            print(f"API Error for {cid}: {e}. Retrying this candidate next time.")
        
        # --- STRICT RATE LIMITING ---
        time.sleep(DELAY_BETWEEN_REQUESTS)

    print("Task 3.3 Complete. All candidates graded.")

if __name__ == "__main__":
    main()
    """
"""
## OPENROUTER CODE
import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv
from teacher_prompt import SYSTEM_PROMPT, format_candidate_prompt

# --- SYSTEM PATHS ---
INPUT_SAMPLE_PATH = "artifacts/teacher_sample.jsonl"
OUTPUT_LABELS_PATH = "artifacts/labeled_candidates.json"

# Load environment variables
load_dotenv()

# FIX: Safely retrieve your key from .env instead of using the raw string literal
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("Error: OPENROUTER_API_KEY is not set in your environment variables or .env file.")

# Initialize the OpenRouter client
client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
)

# OpenRouter Free tier limit handling buffer
DELAY_BETWEEN_REQUESTS = 2.5 

def main():
    print(f"Loading candidates from {INPUT_SAMPLE_PATH}...")
    all_candidates = []
    with open(INPUT_SAMPLE_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_candidates.append(json.loads(line))
    
    # --- CACHING & RESUME LOGIC ---
    labeled_data = []
    processed_ids = set()
    
    if os.path.exists(OUTPUT_LABELS_PATH):
        try:
            with open(OUTPUT_LABELS_PATH, 'r', encoding='utf-8') as f:
                labeled_data = json.load(f)
                processed_ids = {item["candidate_id"] for item in labeled_data}
                print(f"Resuming from cache. {len(processed_ids)} already processed.")
        except json.JSONDecodeError:
            print("Warning: Could not read existing labels. Starting fresh.")

    remaining = [c for c in all_candidates if c["candidate_id"] not in processed_ids]
    print(f"Starting grading for remaining {len(remaining)} candidates (Estimated time: {len(remaining) * DELAY_BETWEEN_REQUESTS / 60:.1f} minutes)...")

    for i, candidate in enumerate(remaining):
        cid = candidate.get("candidate_id", "UNKNOWN")
        prompt = format_candidate_prompt(candidate)
        
        try:
            print(f"[{i+1}/{len(remaining)}] Grading {cid} via OpenRouter Llama 3 Free...")
            
            # FIX: Swapped back to valid JSON structural dummy values (0.0, false)
            openrouter_system_instruction = (
                SYSTEM_PROMPT + 
                "\n\nYou MUST respond strictly with a valid JSON object. Do not include markdown blocks. "
                "Your output structure must match this example format exactly:\n"
                "{\n"
                '  "overall_score": 0.0,\n'
                '  "technical_fit": 0.0,\n'
                '  "production_fit": 0.0,\n'
                '  "career_fit": 0.0,\n'
                '  "availability_fit": 0.0,\n'
                '  "credibility": 0.0,\n'
                '  "hard_reject": false,\n'
                '  "evidence": ["string"],\n'
                '  "concerns": ["string"]\n'
                "}"
            )
            
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": openrouter_system_instruction},
                    {
                        "role": "user", 
                        "content": prompt + "\n\nCRITICAL INSTRUCTION: Return ONLY a valid JSON object. No pre-text, no post-text, no markdown wrap. Start immediately with the { character."
                    }
                ],
                model="google/gemma-4-31b-it:free",
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(chat_completion.choices[0].message.content)
            
            record = {
                "candidate_id": cid,
                "overall_score": float(result.get("overall_score", 0.0)),
                "technical_fit": float(result.get("technical_fit", 0.0)),
                "production_fit": float(result.get("production_fit", 0.0)),
                "career_fit": float(result.get("career_fit", 0.0)),
                "availability_fit": float(result.get("availability_fit", 0.0)),
                "credibility": float(result.get("credibility", 0.0)),
                "hard_reject": bool(result.get("hard_reject", False)),
                "evidence": result.get("evidence", []),
                "concerns": result.get("concerns", [])
            }
            
            labeled_data.append(record)
            
            # SAVE AFTER EVERY CANDIDATE
            with open(OUTPUT_LABELS_PATH, 'w', encoding='utf-8') as f:
                json.dump(labeled_data, f, indent=2)
                
        except Exception as e:
            print(f"API Error for {cid}: {e}. Retrying this candidate next time.")
        
        # --- RESILIENT RATE LIMITING ---
        time.sleep(DELAY_BETWEEN_REQUESTS)

    print("Task 3.3 Complete. All candidates graded.")

if __name__ == "__main__":
    main()
    """