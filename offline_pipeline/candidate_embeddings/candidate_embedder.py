import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import time

def extract_semantic_text(candidate: dict) -> str:
  profile = candidate.get("profile", {})
  headline = profile.get("headline", "")
  summary = profile.get("summary", "")
    
  skills = [s.get("name", "") for s in candidate.get("skills", [])]
  skills_str = ", ".join(filter(None, skills))

  history = []
  for job in candidate.get("career_history", []):
    title = job.get("title", "")
    company = job.get("company", "")
    description = job.get("description","")
    history.append(f"Role: {title} at {company}. Responsibilities: {description}")

  history_str = " | ".join(history)
  return f"Headline: {headline}. Summary: {summary}. Skills: {skills_str}. History: {history_str}."

def process_and_embed(input_jsonl="candidates.jsonl", out_vectors="candidate_embeddings.npy", out_ids="candidate_ids.json"):
  print("Loading BAAI/bge-small-en-v1.5 onto GPU...")
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
    
  candidate_ids = []
  text_corpus = []

  print(f"Streaming data from {input_jsonl}...")
  start_time = time.time()

  with open(input_jsonl, 'r', encoding='utf-8') as f:
    for line in f:
      if not line.strip(): continue
      data = json.loads(line)

      # Maintain strict index alignment.
      candidate_ids.append(data["candidate_id"])
      text_corpus.append(extract_semantic_text(data))
            
  print(f"Parsed {len(candidate_ids)} candidates in {time.time() - start_time:.2f} seconds.")

  print("Starting GPU tensor generation.")
  embeddings = model.encode(
      text_corpus,
      batch_size=512,
      show_progress_bar=True,
      normalize_embeddings=True
  )

  print("Serializing outputs to disk...")
  np.save(out_vectors, embeddings)
  with open(out_ids, "w", encoding="utf-8") as f:
    json.dump(candidate_ids, f)

  print("Candidate embeddings complete. Saved vector shape: {embeddings.shape}")
  
if __name__ == "__main__":
  process_and_embed()
