# MUST Academic Advising Chatbot — LangChain + Gemini + Supabase RAG

This project implements a Retrieval-Augmented Generation (RAG) chatbot for academic advising at Misr University for Science and Technology (MUST).

## Architecture

1. Raw academic JSON files are normalized into a unified document schema.
2. Normalized documents are chunked into retrieval units.
3. Each chunk is embedded with `gemini-embedding-001` using 1024 dimensions.
4. Embeddings and metadata are stored in Supabase/Postgres with pgvector.
5. A LangChain custom retriever embeds each user question, calls the Supabase `match_documents` RPC, and reranks candidates using academic intent signals such as course code, GPA, semester, major, electives, and prerequisites.
6. `gemini-2.5-flash` answers only from the retrieved context.
7. FastAPI exposes retrieval and chatbot endpoints.

## Project structure

```text
app/
  api.py          FastAPI application
  config.py       environment/configuration
  llm.py          LangChain Gemini chat model
  prompts.py      grounded academic-advising prompt
  rag_chain.py    LangChain RAG chain
  ranking.py      deterministic academic reranking rules
  retriever.py    Gemini + Supabase LangChain retriever

database/
  supabase_schema.sql

data/
  raw/
  processed/
  chunks/
pipeline/
  normalize.py
  chunking.py
  ingest.py
tests/
  test_rag_absolute.py
  test_rag_v2.py
```

## 1. Create and activate a virtual environment

### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in that terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

## 2. Configure environment variables

Copy `.env.example` to `.env`:

```powershell
Copy-Item .env.example .env
```

Fill in:

```env
GEMINI_API_KEY=...
SUPABASE_URL=https://YOUR_PROJECT.supabase.co
SUPABASE_KEY=YOUR_SERVER_SIDE_SECRET_KEY
LLM_MODEL=gemini-2.5-flash
EMBEDDING_MODEL=gemini-embedding-001
EMBEDDING_DIM=1024
```

Never commit `.env`. Do not expose the Supabase server-side key in frontend/mobile code.

## 3. Create the Supabase vector table and search function

Open Supabase Dashboard -> SQL Editor and run the full contents of:

```text
database/supabase_schema.sql
```

The SQL uses `vector(1024)`. If you change `EMBEDDING_DIM`, you must update both the table/function SQL and re-embed all stored documents.

## 4. Rebuild processed data

From the project root:

```powershell
python pipeline/normalize.py
python pipeline/chunking.py
```

Expected current dataset result:

```text
229 normalized documents
229 chunks
```

The pipeline now uses project-relative paths, so it works from VS Code/Windows and no longer depends on `/mnt/...` or `/home/...` paths.

## 5. Run the offline data-integrity tests

```powershell
python tests/test_rag_absolute.py --static-only
```

Expected result for the included data:

```text
8/8 checks passed
```

Some source courses have `credit_hours: null`. This is preserved intentionally; the chatbot must not invent missing academic facts.

## 6. Embed and upload all chunks to Supabase

```powershell
python pipeline/ingest.py
```

This script:

- reads `data/chunks/chunks.jsonl`
- creates document embeddings with `RETRIEVAL_DOCUMENT`
- stores 1024-dimensional vectors in `documents`
- upserts by `chunk_id`, so rerunning does not create duplicate chunks

Important: document embeddings and query embeddings must use the same embedding model and dimensionality.

## 7. Run the API

```powershell
uvicorn app.api:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

Health checks:

```text
GET /
GET /health
```

Retrieval test:

```http
POST /rag/search
Content-Type: application/json

{
  "question": "What is the prerequisite for AI.483?",
  "top_k": 3
}
```

Chatbot test:

```http
POST /chat
Content-Type: application/json

{
  "question": "لو المعدل التراكمي بتاعي 1.8 اقدر اسجل كام ساعة؟"
}
```

## 8. Run live RAG tests

Keep the FastAPI server running in one terminal, then open a second terminal in the same project and run:

```powershell
python tests/test_rag_absolute.py --live-only
```

Or run both static and live tests:

```powershell
python tests/test_rag_absolute.py
```

## Important fixes made

- Removed machine-specific absolute paths.
- Unified `/rag/search` response format with the included tests.
- Added `top_k` to the LangChain API search endpoint.
- Added both `/` and `/health` health endpoints.
- Removed the old circular dependency where the LangChain retriever imported clients from the legacy FastAPI module.
- Added centralized configuration and environment validation.
- Added a dedicated ranking module.
- Fixed English `IS` major detection without confusing the ordinary word `is`.
- Normalized course codes such as `CS371`, `CS-371`, `CS,371`, and `CS 371` to `CS.371` during normalization.
- Added the missing Supabase pgvector schema/RPC SQL.
- Made Gemini embedding configuration explicit and consistent between ingestion and querying.
- Preserved unknown academic values instead of fabricating them.
- Updated requirements with packages used by the tests.

## Common errors

### `Missing environment variable(s)`
Create `.env` from `.env.example` and add the three required credentials.

### `function match_documents(...) does not exist`
Run `database/supabase_schema.sql` in Supabase SQL Editor.

### vector dimension error
Your database vector dimension and `EMBEDDING_DIM` do not match. Keep both at 1024 for this project, then re-run ingestion.

### poor retrieval after changing embedding model
Delete/recreate or re-embed the existing vectors. Never compare query vectors produced by one embedding model with document vectors produced by another.

### `/rag/search` works but `/chat` fails
Check the Gemini API key/model access. Retrieval uses Gemini embeddings; answer generation additionally calls the chat model.

## Recommended development order

Do not start by debugging the LLM response. Verify each layer in this order:

1. `normalize.py`
2. `chunking.py`
3. static tests
4. Supabase schema
5. ingestion
6. `/rag/search`
7. live retrieval tests
8. `/chat`
9. frontend integration

If retrieval is wrong, changing the prompt will not fix the underlying RAG problem.
