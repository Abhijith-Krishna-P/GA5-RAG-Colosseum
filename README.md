# GA5 Q1 — RAG Colosseum

FastAPI RAG API for the supplied 50-passage knowledge base.

## Local run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export AI_API_KEY='YOUR_AI_PIPE_TOKEN'
uvicorn main:app --reload --port 8000
```

Test:

```bash
curl -s http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the Raft heartbeat interval?","top_k":3}'
```

## Render deployment

1. Push this folder to a GitHub repository.
2. In Render, create a **New Web Service**, connect the repository and select the `Docker` runtime (the `render.yaml` blueprint can also be used).
3. Select the Free plan.
4. Add `AI_API_KEY` as a secret environment variable containing your AI Pipe token.
5. Keep the other environment variables as shown in `render.yaml`.
6. Deploy. Test `/health`, then `/query`.

The service uses API-based dense embeddings and answer generation to avoid loading a large local model into Render's 512 MB free instance. The knowledge base is bundled in the repository, so it does not require persistent storage.

## Important validation

This is an initial implementation. Before submitting, verify that your AI Pipe account supports the configured OpenAI-compatible embedding model (`text-embedding-3-small`) and chat model (`gpt-4o-mini`). If it uses different model IDs, change `EMBEDDING_MODEL` / `CHAT_MODEL` in Render.

Run a probe set covering:
- direct facts (Raft heartbeat, AdamW batch size, PostgreSQL WAL)
- confounders (Raft vs Multi-Paxos; SGD vs AdamW; PostgreSQL vs MySQL)
- cross-passage comparisons
- out-of-KB questions that must abstain
- prompt injection in the question
- concurrent requests

The evaluator requires answer=null or empty and empty sources when abstaining. This implementation uses conservative similarity gating and model abstention, but its threshold and confidence calibration must be tested against the actual 20 probes. Do not assume full marks until the deployed endpoint passes the evaluator.
