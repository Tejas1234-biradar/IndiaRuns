import json
import os

JD_CONFIG_PATH = "../../artifacts/jd_structured_config(2.5).json"

def load_and_prepare_query():
    print(f"Loading parsed JD configuration from {JD_CONFIG_PATH}...")

    if not os.path.exists(JD_CONFIG_PATH):
        raise FileNotFoundError(f"Configuration file missing at {JD_CONFIG_PATH}.")
    
    with open(JD_CONFIG_PATH, 'r', encoding = 'utf-8') as f:
        jd_data = json.load(f)
    
    # Use synthetic ideal candidate string to minimize the asymmetric search gap
    query_string = jd_data.get("synthetic_ideal_candidate_embedding_string")

    if not query_string:
        raise KeyError("synthetic_ideal_candidate_embedding_string not found in parsed JSON.")
    
    print(f"Extracted Query Payload ({len(query_string)} characters).")
    return query_string

if __name__ == "__main__":
    query = load_and_prepare_query()
