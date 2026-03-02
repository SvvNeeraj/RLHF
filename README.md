# Reinforcement-Aligned Academic Intelligence System

Hybrid B.Tech academic tutor with role-based authentication, unified PDF retrieval (global FAISS), local LLM generation, and reward-driven response adaptation.

## What This Project Does

- Scans academic PDFs from local folders and builds one global vector index.
- Retrieves top relevant chunks per query using semantic + lexical + domain-aware ranking.
- Generates query-driven answers using local Qwen runtime (with CPU fallback support).
- Enforces guardrails for B.Tech-only academic scope.
- Collects user ratings and adapts future responses for repeated low-rated query patterns.
- Provides separate user and admin access in Streamlit UI.

## Supported Domains

- Computer Science Engineering (CSE)
- Information Technology (IT)
- Electronics and Communication Engineering (ECE)
- Electrical and Electronics Engineering (EEE)
- Artificial Intelligence and Machine Learning (AIML)
- Data Science (DS)
- Civil Engineering
- Mechanical Engineering
- Cyber Security

## Current Architecture

```mermaid
flowchart TD
    UI[Streamlit UI]
    API[FastAPI Backend]
    DB[(SQLite)]
    VS[(FAISS Global Index)]
    DOC[data/ PDFs]

    UI -->|POST /auth/register, /auth/login| API
    API --> AUTH[Token + Role Validation]
    AUTH --> DB

    UI -->|User Role: POST /chat| API
    API --> GRD[Guardrails + Intent Parsing]
    GRD --> RET[Hybrid Retrieval\nSemantic + Lexical + Domain Boost]
    RET --> VS
    RET --> PROMPT[Prompt Builder\nRequested Sections + RLHF Constraints]
    PROMPT --> LLM[Qwen Local Runtime\n4-bit attempt, CPU-safe fallback]
    LLM --> POST[Quality Gate + Response Modifier]
    POST --> API
    API --> DB
    API --> UI

    UI -->|User Role: POST /feedback| API
    API --> FB[Reward Model + Feedback Handler + Reward Tracker]
    FB --> DB
    FB --> OLRL[Online RLHF State\n(user_id + query_key)]
    OLRL --> PROMPT

    DOC --> IDX[Index Builder /rebuild-index]
    IDX --> VS

    UI -->|Admin Role: /analytics /reward-history /users /documents| API
    API --> DB
    API --> VS
```

## End-to-End Flow (Present Implementation)

1. User authenticates via `POST /auth/login` or `POST /auth/register`.
2. User sends query to `POST /chat` (token-protected, user role).
3. Guardrails validate B.Tech relevance and context adequacy.
4. Retriever fetches top-k chunks (`k=5`) from global FAISS.
5. Prompt is built using:
   - retrieved context,
   - requested section intent from query,
   - RLHF adaptation constraints (if low-rated history exists).
6. Local model generates response (`Qwen/Qwen2.5-0.5B-Instruct` primary candidate).
7. Quality gate validates structure and relevance; fallback synthesis is applied if needed.
8. Response + sources are stored in SQLite and returned to UI.
9. User rating (`POST /feedback`) updates:
   - reward logs,
   - preference profile,
   - query-level reward tracking,
   - online RLHF adaptation state.

## API Surface (Current)

### Auth
- `POST /auth/register`
- `POST /auth/login`
- `GET /auth/me`

### User Endpoints
- `POST /chat`
- `POST /feedback`

### Admin Endpoints
- `POST /rebuild-index`
- `GET /documents`
- `GET /analytics`
- `GET /reward-history`
- `GET /history`
- `GET /users`
- `POST /quality-gate`

### Utility
- `GET /health`

## Result Samples

### 1) Index Build Sample (Current Local Metadata)

From `vector_store/metadata.json`:

- `total_pdfs`: **144**
- `total_chunks`: **25988**
- status distribution:
  - `indexed`: **116**
  - `skipped_large_file`: **26**
  - `empty_chunks`: **1**
  - `skipped_error`: **1**

### 2) Chat Response Sample (Shape)

```json
{
  "chat_id": 428,
  "response": "Detailed academic answer...",
  "rejected": false,
  "sources": [
    {"file_name":"EEE (8).pdf","page_number":75,"score":0.4131,"subject":"EEE"}
  ],
  "response_improved": true,
  "improvement_reason": "low_reward_adaptation",
  "response_variant": "v2"
}
```

### 3) Feedback Response Sample (Shape)

```json
{
  "reward": -0.5,
  "preference_profile": {
    "preferred_mode": "detailed",
    "avg_reward": 0.12,
    "total_feedback": 37
  },
  "question_profile": {
    "query_key": "explain load flow studies...",
    "avg_reward": -0.2,
    "total_feedback": 6,
    "low_rating_count": 3
  }
}
```

### 4) Admin Dashboard Sample

- User DB table (user_id, role, created_at)
- Reward trend graphs:
  - Reward & rolling reward
  - Cumulative reward & low-rating rate
  - Rating distribution
- Rewarded query pattern table (weak topics / low-reward keys)

## Runtime and Performance Settings

Current defaults from `app/config.py`:

- `MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct`
- `MODEL_CANDIDATES=Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct`
- `EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`
- `RETRIEVAL_K=5`
- `MAX_NEW_TOKENS=700`
- `CHUNK_SIZE=500`
- `CHUNK_OVERLAP=50`
- `MAX_PDF_SIZE_MB=300`
- `MAX_PAGES_PER_PDF=120`

## Setup (Windows, CPU-only)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
python scripts/build_faiss.py
powershell -ExecutionPolicy Bypass -File scripts/run_api.ps1
powershell -ExecutionPolicy Bypass -File scripts/run_ui.ps1
```

## Authentication and Access

- Streamlit login page supports:
  - existing user login,
  - new user registration.
- User role:
  - chat and feedback only.
- Admin role:
  - analytics, history, users, documents, rebuild-index.

## RLHF Notes (Current vs True RLHF)

- Local backend implements online reward-aware adaptation:
  - query-level tracking,
  - low-rated response rewrite pressure,
  - overlap reduction with previously low-rated answers.
- True policy-parameter RLHF training is done externally in Colab.

### Colab Scripts

- `colab/lora_training_colab.py`
- `colab/true_rlhf_policy_gradient_colab.py`

### RLHF Data Export

```powershell
python scripts/export_rlhf_dataset.py
```

## Important Project Artifacts

- Global index: `vector_store/global_index/`
- Index metadata: `vector_store/metadata.json`
- RLHF online state: `data/rlhf/online_state.json`
- Feedback events: `data/rlhf/feedback_events.jsonl`
- Training examples: `data/rlhf/training_examples.jsonl`

## Troubleshooting

- If chat says index missing: run `python scripts/build_faiss.py` or admin `POST /rebuild-index`.
- If backend is not reachable: run `scripts/run_api.ps1`.
- If auth fails: verify default admin credentials in `app/config.py` and DB initialization.
