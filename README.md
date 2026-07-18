# 🏥 Healthcare Query Assistant — Streamlit UI

A polished Streamlit front-end for the RAG + NLP-to-SQL multi-agent system
built in `RAG_Healthcare_Query_Assistant.ipynb`. The agent logic itself
(orchestrator routing rules, SQL templates, FAISS retrieval, response
formatting) is carried over from the notebook unchanged — this project wraps
it in a real UI so it's demoable and deployable.

## What's inside

```
healthcare_assistant/
├── app.py                  # the Streamlit app
├── requirements.txt
├── README.md
└── data/
    ├── hospital.db          # normalized SQLite patient database
    ├── policy_index.faiss   # pre-built FAISS vector index over policy chunks
    ├── chunk_meta.pkl       # (doc_name, chunk_text) pairs matching the index
    ├── admission_policy.txt
    ├── billing_policy.txt
    ├── discharge_policy.txt
    ├── emergency_policy.txt
    └── insurance_policy.txt
```

## Pages

- **Chat Assistant** — the main interface. Ask a question; a live pipeline
  diagram highlights which agent(s) handled it, and every answer is
  transparent: SQL answers show the generated SQL, policy answers show the
  retrieved passages and their similarity scores.
- **Dataset Analytics** — interactive charts over the patient database
  (conditions, admission types, billing by insurer, age distribution, etc.)
  with filters.
- **Policy Library** — the five source policy documents with a search box.
- **How It Works** — a plain-language walkthrough of the four-agent
  architecture, for demos/explanations.
- **Routing Insights** — a live log of every query asked this session
  (intent, confidence, latency), downloadable as CSV.

## Run locally

```bash
cd healthcare_assistant
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

First launch will download two models from Hugging Face (`all-MiniLM-L6-v2`
for embeddings and `google/flan-t5-base` for generation) — a few hundred MB,
cached afterward. If you want a faster/lighter start, toggle **"Use flan-t5
to compose policy answers"** off in the sidebar: the RAG agent will then
return the top matching policy passage directly (extractive mode) without
loading the generative model at all.

## Deploy on Streamlit Community Cloud

1. Push this folder to a public (or private) GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo,
   and set the main file to `app.py`.
3. Deploy. Note: the free tier has ~1GB RAM — if the app is slow to start,
   turn off generative mode as described above, since `torch` + `flan-t5-base`
   are the heaviest pieces.

## Notes on faithfulness to the original notebook

- Orchestrator keyword/phrase scoring, SQL filter parsing, date-range logic,
  FAISS top-k retrieval, and the 0.25 similarity refusal threshold are all
  copied verbatim from the notebook's validated implementation.
- The FAISS index and chunk metadata are reused as-is (no re-embedding at
  startup), so retrieval results match the notebook exactly.
- The only additions are presentation-layer: structured rendering (tables,
  metrics, source chips, the routing pipeline visual) instead of printed
  strings, plus the analytics/policy-library/insights pages, since the
  notebook only exposed a plain `chat()` REPL loop.
