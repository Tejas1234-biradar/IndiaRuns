# method 5 - finalized 

## **Phase 1: The Unconstrained Offline Pipeline (Pre-Computation)** 

_You run this on your own machine/cloud over a few days. You use GPUs, internet, and paid API calls._ 

## **1. The LLM JD Decoder (Cloud AI)** 

- Send the job_description.md to Claude 3.5 Sonnet or GPT-4o. Have it extract the exact core requirements, hidden subtext (e.g., "willing to ship scrappy code"), and explicit anti-patterns (e.g., "title chasers"). 

## **2. The Semantic Indexer (Neural AI)** 

- Use a local embedding model ( BAAI/bge-small-en-v1.5 ) to embed the concatenated text ( headline + summary + skills + career_history ) of all 100,000 candidates. 

- Save this directly into a **FAISS Vector Index file** ( faiss_index.bin ). This maps the "hidden gems" (candidates who didn't use exact keywords but have semantic equivalents). 

## **3. Feature Engineering & Honeypot Detection (Data Engineering)** 

- Scan the 100,000 profiles to flag the ~80 honeypots based on the redrob_signals constraints (e.g., impossible salary ranges, conflicting dates). Save their IDs into honeypot_ids.pkl . 

- Convert the complex candidate JSONs into a flat numerical matrix. Columns will include: years_of_experience , github_activity_score , recruiter_response_rate , interview_completion_rate , and faiss_distance_to_jd . 

## **4. The Teacher-Student Labeling (Learning to Rank)** 

- **The Teacher:** Use Claude/GPT-4 to act as an expert recruiter. Feed it a diverse sample of 3,000 candidates (including great matches, average matches, and keyword-stuffers). Have it score them from 0.0 to 10.0 based on the JD. 

- **The Student:** Train a lightweight **XGBoost Ranker** on this tabular data using the LLM's scores as the target variable. The model learns how to balance a high semantic text score against a low recruiter response rate. Save the trained model ( model.xgb ). 

## **Phase 2: The Constrained Runtime Pipeline (The Submission)** 

_You run this inside the Docker sandbox. It uses NO internet, NO GPUs, and must finish in under 5 minutes._ 

Your rank.py script executes the following four lightning-fast steps: 

**Step 1: The Honeypot Purge (0.5 seconds)** As the script reads candidates.jsonl.gz , it checks every ID against your pre-computed honeypot_ids.pkl . If it's a match, it is instantly assigned a score of 0.0 and dropped. **(Guarantees you beat the >10% honeypot disqualification rule).** 

**Step 2: FAISS Top-K Recall (1.5 seconds)** Instead of evaluating the remaining 99,000+ candidates, your script loads the faiss_index.bin and the vector for the JD. It executes a local similarity search to instantly retrieve the **Top 2,000** semantically closest candidates. _(You just eliminated 98% of the dataset in under 2 seconds, safely dodging the 5-minute timeout)._ 

**Step 3: XGBoost Re-Ranking (0.2 seconds)** For those Top 2,000 candidates, you load their pre-computed tabular features (experience, behavioral signals, semantic distance). You pass this 2,000-row matrix into your loaded model.xgb . The tree model natively applies its learned penalties: it mathematically crushes the keyword-stuffer who has a 5% 

recruiter_response_rate and boosts the Tier-5 hidden gem with an 85 

github_activity_score . Sort the output descending to get your final **Top 100** . 

**Step 4: Deterministic Dynamic Reasoning (0.1 seconds)** You must avoid the "templated strings" manual review flag. Do not use an LLM here. Instead, extract the **top 3 driving features** for each candidate from the XGBoost model (using SHAP values or basic feature weights). Use a Python function to stitch these specific data points together dynamically: 

_"Ranked #12 due to deep semantic alignment with core JD requirements, fortified by a top-percentile GitHub activity score (88) and 6.4 years of highly stable product-company tenure, cleanly offsetting a slight gap in exact framework keywords."_ 

**(Resolve ties by sorting candidate_id ascending to pass Stage 1 validation).** 

tasks 

To maximize velocity, the team is organized into highly specialized roles: 

- **M1: NLP & Retrieval Engineer** (Focus: Embeddings, JD Parsing, FAISS Indexing) 

- **M2: Data & Signal Engineer** (Focus: Parsing 100K JSONL, Feature Matrix, Honeypots) 

- **M3: Core Modeler** (Focus: XGBoost Ranker, LLM Teacher Labeling, SHAP Analysis) 

- **M4: Infrastructure & Sandbox Engineer** (Focus: Runtime Script, Docker/HF Sandbox, Validation) 

## **Sprint 1: Foundation, EDA, & Feature Extraction (Days 1–5)** 

**Goal:** Parse the raw 100,000-candidate dataset, identify the ~80 honeypots, extract dense text embeddings, and establish an engineering baseline. 

## **Task Matrix** 

## ● **M1 (NLP & Retrieval):** 

   - **Task 1.1:** Build a prompt workflow for Claude 3.5/GPT-4o to parse job_description.md and output structured target requirements, hidden text markers, and explicit anti-patterns. 

   - **Task 1.2:** Write a batch-tokenization script to embed all 100,000 candidate profiles using BAAI/bge-small-en-v1.5 over local GPUs. Serialized outputs must match candidate IDs. 

- **M2 (Data & Signals):** 

   - **Task 2.1:** Build a high-throughput stream-reader for the candidates.jsonl.gz dataset using ijson or orjson to parse nested structures efficiently. 

   - **Task 2.2:** Program a scanning script to detect all ~80 honeypots based on explicit redrob_signals violations (e.g., expected_salary_range_inr_lpa.max < .min, skill experience anomalies). Export these to honeypot_ids.pkl. 

- **M3 (Core Modeler):** 

   - **Task 3.1:** Perform descriptive Exploratory Data Analysis (EDA) on the 23 behavioral parameters from redrob_signals_doc.md to identify continuous variance, distributions, and missing data indicators (like -1 values). 

   - **Task 3.2:** Define the schema for the unified tabular feature matrix (e.g., tenure length, GitHub activity percentile, notice period penalties). 

- **M4 (Infrastructure):** 

   - **Task 4.1:** Establish the shared GitHub repository structure, implement strict linting, and create the baseline Dockerfile environment. 

   - **Task 4.2:** Integrate the official validate_submission.py script into a local continuous validation gate to verify output compliance instantly. 

## **Sprint 2: Teacher-Student Pipeline & Index Construction (Days 6–10)** 

**Goal:** Build the FAISS vector index, generate high-fidelity LLM teacher scores, and train the local student model. 

## **Task Matrix** 

- **M1 (NLP & Retrieval):** 

   - **Task 1.3:** Build the fixed faiss_index.bin using the generated candidate embeddings. Configure it for fast Euclidean ($L_2$) or Inner Product ($IP$) retrieval on CPU. 

   - **Task 1.4:** Generate the dense vector embedding for the extracted JD requirements. Test top-K retrieval accuracy against a synthetic test set. 

- **M2 (Data & Signals):** 

   - **Task 2.3:** Assemble the final tabular feature matrix for all 100,000 candidates, incorporating the FAISS similarity distance metrics calculated by M1. 

   - **Task 2.4:** Build a stratified sampling script to extract a highly diverse batch of 3,000 candidates (representing excellent fits, average matches, and obvious keyword stuffers) for LLM evaluation. 

- **M3 (Core Modeler):** 

   - **Task 3.3:** Build the automated batch-LLM prompt infrastructure to feed the 3,000 sample candidates to the LLM "Teacher". Collect deterministic 0.0 to 10.0 qualification scores. 

   - **Task 3.4:** Train an XGBoost or LightGBM Ranker (LambdaMART objective) using the generated tabular feature matrix against the LLM target scores. Save weights to model.xgb. 

- **M4 (Infrastructure):** 

   - **Task 4.3:** Build the feature serialization pipeline to pack look-up statistics into a highly compressed, fast-loading format (like a localized .parquet or .pkl file). 

   - **Task 4.4:** Configure the execution framework for the sandbox environment to handle static file asset routing natively within the target container. 

## **Sprint 3: Runtime Integration & Reasoner Engine (Days 11–15)** 

**Goal:** Create the runtime ranking file (rank.py), implement local model feature extraction, and build the dynamic text generator. 

## **Task Matrix** 

- **M1 (NLP & Retrieval):** 

   - **Task 1.5:** Implement the Phase 2 runtime retrieval step: verify the FAISS index slice properly isolates the top 2,000 candidates on CPU within a <2-second execution envelope. 

- **M2 (Data & Signals):** 

   - **Task 2.5:** Optimize the fast look-up data pipeline within rank.py to retrieve pre-computed tabular features for only the 2,000 candidates isolated by M1. 

- **M3 (Core Modeler):** 

   - **Task 3.5:** Extract local feature importance parameters from the trained XGBoost model using lightweight SHAP tree explainer evaluations or model weight approximations. 

   - **Task 3.6:** Program the deterministic reasoning generation engine. Write a rules-based matrix that maps the top 3 driving numeric features into custom, highly context-aware text justifications. 

## ● **M4 (Infrastructure):** 

- **Task 4.5:** Assemble the complete rank.py execution sequence: (1) Honeypot Purge, (2) FAISS Filtering, (3) XGBoost Scoring, (4) Dynamic Reasoning generation. 

- **Task 4.6:** Program the mandatory strict validation formatting layer within rank.py: sort ties explicitly by candidate_id ascending, output exactly 100 rows, and verify score monotonicity. 

## **Sprint 4: Sandbox Deployment, Testing, & Optimization (Days 16–20)** 

**Goal:** Deploy the runnable sandbox, stress test against unexpected inputs, resolve edge cases, and run full-scale verification. 

## **Task Matrix** 

## ● **M1 (NLP & Retrieval):** 

   - **Task 1.6:** Conduct sensitivity analysis on the retrieval step: verify that changing wording variants in sample JDs does not break index stability or skip critical candidates. 

- **M2 (Data & Signals):** 

   - **Task 2.6:** Review the top 100 ranked candidates manually to verify the absolute absence of honeypots and confirm that keyword-stuffers have been successfully pushed down by behavioral weights. 

- **M3 (Core Modeler):** 

   - **Task 3.7:** Calibrate the final XGBoost scoring parameters to ensure maximum variation across the top 100 candidates, preventing identical flat scores. 

   - **Task 3.8:** Run string-similarity checks (e.g., Levenshtein distance arrays) across generated candidate reasonings to ensure they pass manual review checks against cookie-cutter text patterns. 

- **M4 (Infrastructure):** 

   - **Task 4.7:** Deploy the live interactive sandbox environment on HuggingFace Spaces or Streamlit Cloud. Verify end-to-end functionality using an input slice of $\le 100$ sample candidates. 

   - **Task 4.8:** Execute localized execution budget profiles. Verify that total execution memory never spikes past 16 GB RAM and finishes processing in under 10 seconds total on standard CPU. Run final checks using validate_submission.py. 

git practices 

Based on the official **Stage 4 (Manual Review)** guidelines in the submission specification, a structured Git branching strategy is not just okay—it is highly recommended to protect your team from automatic elimination. 

The specification explicitly states that judges will audit your **"Git history authenticity (real iteration vs single dump)"** and that a **"flat git history with no iteration"** will result in immediate disqualification at Stage 4, even if you have a perfect ranking score. The organizers use Git logs as a primary filter to separate teams doing real human software engineering from "paste-and-pray" participants who let an LLM write the script and dumped it into a repository at the last minute. 

To ensure your 4-member team passes the manual Git audit seamlessly, follow these strict Git best practices: 

## **1. Execute a True Feature-Branch Workflow** 

Do not let the team push directly to the main or master branch. Instead, use a GitHub Flow-style branching strategy that aligns with your sprint tasks: 

- Create branches tied to specific features or checklist items (e.g., 

   - feature/m2-honeypot-scanner, feature/m1-faiss-indexing, feature/m3-xgb-student). 

- When a feature is complete, open a Pull Request (PR) to merge it into main. 

- **Crucial Rule: Do not squash-merge your PRs.** Squashing condenses all your incremental progress into a single flat commit, which destroys your iteration trail and makes your history look like a series of bulk code dumps. Use a standard merge commit to preserve the step-by-step history of how the code evolved. 

## **2. Enforce "Atomic Commits" (Commit Early, Commit Often)** 

A major red flag for judges is a repository with only 5 to 10 massive commits containing hundreds of lines of code changed at once. 

- Every team member should commit code multiple times a day. 

- Make **atomic commits** —meaning a commit should do exactly one small thing. For example, M2's progression for the Honeypot Scanner should look like: 

   1. feat(honeypots): add initial dictionary skeleton for profile validation rules 

   2. feat(honeypots): implement salary range max/min mathematical contradiction checks 

   3. feat(honeypots): implement timeline overlap logic for experience dates 

   4. test(honeypots): verify scanner isolates exactly the expected anomalous profiles 

- This chronological timeline provides undeniable proof of human iteration and development logic. 

## **3. Maintain Multi-Author Authenticity** 

Since you declared a 4-member team in your portal metadata, the Git history must show active code contributions from all 4 registered GitHub accounts/emails. 

- If 95% of the commits come from Member 1, and Members 2, 3, and 4 only have 1 or 2 cosmetic commits (like editing the README.md), the organizers will suspect that the work was fabricated or plagiarized. 

- Ensure every member is pushing their assigned checklist tasks from their own local machines using their authenticated Git profiles. This also protects you during the **Stage** 

**5 Defend-Your-Work Interview** , as everyone can point to their exact physical footprints in the code history. 

## **4. Use Semantic Commit Messages** 

Avoid lazy commit messages like _"update"_ , _"fix"_ , _"test"_ , or _"code working"_ —these look automated or rushed. Use professional, human engineering conventions (like Conventional Commits): 

- feat: for new features (e.g., feat(recall): integrate local FAISS top-k filtering slice) 

- fix: for bug fixes (e.g., fix(format): resolve candidate_id secondary sorting tiebreaker) 

- tune: or refactor: for adjustments (e.g., tune(xgb): adjust learning rate to combat keyword-stuffer weight inflation) 

- Clean, descriptive messages prove intent and show that you understand the underlying engineering trade-offs of your changes. 

## **5. Document the Evolution of Pre-computed Artifacts** 

The rules explicitly permit you to use pre-computed assets (like feature matrices, FAISS indexes, and model weights) to stay under the 5-minute CPU constraint. However, if those heavy binaries just appear out of nowhere in your repo with no backstory, it looks highly suspicious. 

- Keep your **offline scripts** (the code you used to query Claude/GPT-4 for teacher labels, the script that builds the FAISS index, and your XGBoost Jupyter training notebooks) directly inside an offline_pipeline/ directory in the repository. 

- Show Git history for _those_ scripts as well. When you adjust a prompt for the LLM Teacher, commit the prompt change. When you change hyper-parameters in your training script, commit those variations. This proves you didn't just hardcode or manipulate the final results manually. 

## **Comparison: Disqualification vs. Passing Git Trajectories** 

|**Metric**|❌**Automatic Elimination Trail (Stage 4)**|**Winning Human Engineering**<br>**Trail (Stage 4)**|
|---|---|---|
|**Commit**<br>**Volume**|3 to 7 total commits across the 20-day<br>timeline.|60+ total commits distributed<br>chronologically across the<br>timeline.|
|**Branch**<br>**Footprint**|Everything committed directly tomainor<br>squashed entirely into 1 flat layer.|Multiple feature/bug branches<br>tracking pull requests with clear<br>documentation.|



||**Author**<br>**Distribution**<br>1 person's Git signature dominates the<br>entire codebase.<br>Distinct commit histories for all<br>4 team members matching<br>their portal declarations.<br>**Pre-comp**<br>**Code**<br>Missing or unversioned offline tools; assets<br>just pop up statically.<br>Transparent iteration on prompt<br>files, modeling scripts, and<br>training files.<br>**Message**<br>**Quality**<br>"fixed bugs", "working version final",<br>"upload"<br>"refactor(data): optimize stream<br>reader memory footprint to<br>comfortably stay under 16GB".|
|---|---|



