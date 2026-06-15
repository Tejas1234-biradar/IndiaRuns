import os
import json
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai
from google.genai import types
import docx
from typing import List

load_dotenv()

class LogisticsContraints(BaseModel):
    preferred_locations: List[str] = Field(description="Target cities or regions explicitly allowed for hybrid/relocation.")
    availability_requirements: List[str] = Field(description="Requirements around platform activity or job market presence.")

class AlgorithmicRules(BaseModel):
    keyword_trap_mitigation_strategy: str = Field(description="Explicit guidance from JD on how to avoid keyword traps (e.g., 'checking titles vs skills).")
    behavioural_signals: List[str] = Field(description="List of behavioural red flags (e.g., low activity, low response rate) that require down-weighting.")

class JDExtractionSchema(BaseModel):
    role_name: str = Field(description = "The formal title of the role.")
    experience_range_years: List[int] = Field(description = "Min and max experience boundaries as intergers, e.g., [5, 9].")
    core_technical_requirements: List[str] = Field(description = "Explicity required tools, architectures, libraries, frameworks or math concepts.")
    hidden_text_markers: List[str] = Field(description = "Implicit mindset or culture requirements extracted from subtext, e.g., 'scrappy engineering mindset', 'willing to ship suboptimal code quickly', etc.")
    explicit_anti_patterns: List[str] = Field(description = "Strict candidate disqualifiers or types of profiles explicitly rejected by the JD, e.g., 'title chasers', 'pure corporate ladder climbers', etc.")
    optional_preferred_requirements: List[str] = Field(description="Nice-to-have skills, experiences, or traits mentioned that are not strict requirements.")
    logistics_constraints: LogisticsContraints = Field(description="Constraints related to location and availability.")
    algorithmic_rules: AlgorithmicRules = Field(description="Rules for algorithmic processing of the job description.")

    synthetic_ideal_candidate_embedding_string: str = Field(
        description=(
            "Write a highly dense, natural language paragraph describing the perfect candidate's "
            "skills, mindset, and experience. Do not use bullet points. Write it in the first-person "
            "as if it is a candidate's resume summary. This string will be converted into a mathematical "
            "vector to query a FAISS database, so prioritize semantic density and exact terminology."
        )
    )

def extract_text_from_doc(file_path: str) -> str:

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    doc = docx.Document(file_path)
    full_text = [paragraph.text for paragraph in doc.paragraphs]
    return "\n".join(full_text)

def decode_job_description(jd_text: str) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")  
    
    client = genai.Client(api_key=api_key)

    system_prompt = (
        "You are an expert technical recruiting parser built for high-scale talent indexing systems."
        "Your objective is to read raw, conversational Job Descriptions and extract specific requirements, "
        "hidden cultural subtext, anti-patterns, and explicitly stated algorithmic constraints."
    )

    user_prompt = f"Analyze this Job Description thoroughly and populate the requested schema properties:\n\n{jd_text}"

    print("Dispatching execution payload to Gemini 3.5 Flash...")

    response = client.models.generate_content(
        model='gemini-3.5-flash',
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type='application/json',
            response_schema=JDExtractionSchema,
            temperature=0.0
        ),
    )

    return json.loads(response.text)

if __name__ == "__main__":
    JD_FILE_PATH = "data/job_description.docx"
    OUTPUT_CONFIG_PATH = "offline_pipeline/jd_decoder/jd_structured_config.json"

    try:
        # Parse the JD document.
        print(f"Reading {JD_FILE_PATH}...")
        raw_jd_content = extract_text_from_doc(JD_FILE_PATH)

        # Run LLM decoder pipeline.
        structured_data = decode_job_description(raw_jd_content)

        print(f"Writing parsed configuration to {OUTPUT_CONFIG_PATH}...")
        with open(OUTPUT_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(structured_data, f, indent=4)

        print("JD decoding completed successfully. Result Preview:")
        print(json.dumps(structured_data, indent=2))

    except Exception as e:
        print(f"\nExecution failed: {str(e)}")