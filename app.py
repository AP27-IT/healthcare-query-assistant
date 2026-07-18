"""
RAG-Based Healthcare Query Assistant — Streamlit UI
=====================================================
A multi-agent system: an Orchestrator classifies each query and routes it to
a template-driven NLP-to-SQL agent (structured patient-data questions) or a
FAISS + sentence-transformer RAG agent (hospital policy questions), then a
Response Formatter turns whichever agent's output into a clean reply.

This file is a presentation layer around that exact pipeline — the routing
rules, SQL templates, and retrieval logic below are the same ones developed
and validated in the source notebook.
"""

import os
import re
import time
import pickle
import sqlite3
import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------------------------------------------------
# Paths & page config
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "hospital.db")
FAISS_PATH = os.path.join(DATA_DIR, "policy_index.faiss")
CHUNK_META_PATH = os.path.join(DATA_DIR, "chunk_meta.pkl")

POLICY_FILES = {
    "admission_policy.txt": "Admission",
    "billing_policy.txt": "Billing",
    "discharge_policy.txt": "Discharge",
    "emergency_policy.txt": "Emergency Care",
    "insurance_policy.txt": "Insurance",
}
POLICY_ICONS = {
    "admission_policy.txt": "🛏️",
    "billing_policy.txt": "💳",
    "discharge_policy.txt": "🚪",
    "emergency_policy.txt": "🚑",
    "insurance_policy.txt": "🛡️",
}

st.set_page_config(
    page_title="Healthcare Query Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Styling — design tokens
# --------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root{
  --bg:#F4F7F6; --surface:#FFFFFF; --ink:#16262A; --ink-soft:#5B6C70;
  --teal:#0E4F52; --teal-dark:#0A3638; --teal-tint:#E4EFEE;
  --blue:#3E7CB1; --blue-tint:#E7F0F8;
  --amber:#C98A2E; --amber-tint:#FBF0DD;
  --purple:#8B5FA3; --purple-tint:#F1E9F4;
  --coral:#C9564B; --coral-tint:#FBEAE8;
  --border:#E1E7E6;
}

html, body, [class*="css"]  { font-family:'Inter', sans-serif; color:var(--ink); }
.stApp { background:var(--bg); }
h1,h2,h3,h4 { font-family:'Manrope', sans-serif; color:var(--teal-dark); letter-spacing:-0.01em; }
[data-testid="stSidebar"] { background:var(--teal-dark); }
[data-testid="stSidebar"] * { color:#EAF3F2 !important; }
[data-testid="stSidebar"] .stButton>button {
  background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.18);
  color:#EAF3F2 !important; border-radius:8px; text-align:left; font-size:0.85rem;
}
[data-testid="stSidebar"] .stButton>button:hover { background:rgba(255,255,255,0.18); border-color:rgba(255,255,255,0.35); }
[data-testid="stSidebar"] hr { border-color:rgba(255,255,255,0.15); }

/* Hero header */
.hero {
  background:linear-gradient(120deg, var(--teal-dark) 0%, var(--teal) 60%, #146B65 100%);
  border-radius:16px; padding:28px 32px; margin-bottom:22px; color:#fff;
  box-shadow:0 8px 24px rgba(10,54,56,0.18);
}
.hero-eyebrow { font-family:'JetBrains Mono', monospace; font-size:0.72rem; letter-spacing:0.14em;
  text-transform:uppercase; color:#9FD8CF; margin-bottom:6px; }
.hero h1 { color:#fff; font-size:1.9rem; margin:0 0 6px 0; }
.hero p { color:#D7ECE9; font-size:0.95rem; margin:0; max-width:720px; }
.hero-badges { margin-top:16px; display:flex; gap:8px; flex-wrap:wrap; }
.badge { display:inline-block; padding:4px 12px; border-radius:20px; font-size:0.72rem;
  font-family:'JetBrains Mono', monospace; background:rgba(255,255,255,0.12); color:#fff;
  border:1px solid rgba(255,255,255,0.25); }

/* Cards */
.card { background:var(--surface); border:1px solid var(--border); border-radius:14px;
  padding:18px 20px; box-shadow:0 1px 2px rgba(16,40,42,0.04); }
.kpi { background:var(--surface); border:1px solid var(--border); border-radius:14px;
  padding:16px 18px; text-align:left; }
.kpi .label { font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em; color:var(--ink-soft); font-family:'JetBrains Mono',monospace;}
.kpi .value { font-family:'Manrope',sans-serif; font-size:1.6rem; font-weight:800; color:var(--teal-dark); margin-top:2px;}

/* Intent badges */
.intent-tag { display:inline-flex; align-items:center; gap:6px; padding:3px 11px; border-radius:20px;
  font-size:0.72rem; font-family:'JetBrains Mono', monospace; font-weight:600; }
.intent-SQL { background:var(--blue-tint); color:var(--blue); }
.intent-POLICY { background:var(--amber-tint); color:var(--amber); }
.intent-AMBIGUOUS { background:var(--purple-tint); color:var(--purple); }
.intent-OUT_OF_DOMAIN { background:#EEF1F0; color:var(--ink-soft); }
.intent-ERROR { background:var(--coral-tint); color:var(--coral); }

/* Pipeline diagram */
.pipeline { display:flex; align-items:center; gap:6px; flex-wrap:wrap; margin:14px 0 6px 0; }
.pnode { flex:1; min-width:120px; background:var(--surface); border:1.5px solid var(--border); border-radius:10px;
  padding:10px 12px; text-align:center; font-size:0.78rem; font-weight:600; color:var(--ink-soft); transition:all .25s ease; }
.pnode .sub { display:block; font-weight:400; font-size:0.68rem; color:var(--ink-soft); font-family:'JetBrains Mono',monospace; margin-top:2px;}
.parrow { color:var(--border); font-size:1.1rem; }
.pnode.active-sql { border-color:var(--blue); background:var(--blue-tint); color:var(--blue); box-shadow:0 0 0 3px rgba(62,124,177,0.12);}
.pnode.active-policy { border-color:var(--amber); background:var(--amber-tint); color:var(--amber); box-shadow:0 0 0 3px rgba(201,138,46,0.12);}
.pnode.active-amb { border-color:var(--purple); background:var(--purple-tint); color:var(--purple); box-shadow:0 0 0 3px rgba(139,95,163,0.12);}
.pnode.active-always { border-color:var(--teal); background:var(--teal-tint); color:var(--teal-dark); }

/* Source chip */
.src-chip { display:inline-block; background:var(--teal-tint); color:var(--teal-dark); font-size:0.72rem;
  padding:2px 10px; border-radius:12px; margin-right:6px; font-family:'JetBrains Mono',monospace; }

.sql-block { background:#0F2C2E; color:#B9E7DF; font-family:'JetBrains Mono', monospace; font-size:0.78rem;
  padding:12px 14px; border-radius:10px; overflow-x:auto; white-space:pre-wrap; }

hr { border-color:var(--border); }
.small-note { color:var(--ink-soft); font-size:0.8rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Cached resource loaders
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Connecting to patient database…")
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn


@st.cache_resource(show_spinner="Loading policy vector index…")
def get_vector_store():
    import faiss
    index = faiss.read_index(FAISS_PATH)
    with open(CHUNK_META_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


@st.cache_resource(show_spinner="Loading sentence-embedding model (all-MiniLM-L6-v2)…")
def get_embed_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Loading generative model (flan-t5-base)…")
def get_gen_model():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tok, model


@st.cache_data(show_spinner=False)
def load_analytics_frame():
    conn = get_db()
    q = """
    SELECT p.age, p.gender, p.blood_type, a.medical_condition, a.admission_type,
           a.billing_amount, a.test_results, a.admission_date, a.discharge_date,
           h.hospital_name, ip.provider_name AS insurance_provider
    FROM admissions a
    JOIN patients p ON a.patient_id = p.patient_id
    JOIN hospitals h ON a.hospital_id = h.hospital_id
    JOIN insurance_providers ip ON a.provider_id = ip.provider_id
    """
    df = pd.read_sql_query(q, conn)
    df["admission_date"] = pd.to_datetime(df["admission_date"])
    df["discharge_date"] = pd.to_datetime(df["discharge_date"])
    df["stay_days"] = (df["discharge_date"] - df["admission_date"]).dt.days
    return df


@st.cache_data(show_spinner=False)
def load_policy_texts():
    texts = {}
    for fname in POLICY_FILES:
        with open(os.path.join(DATA_DIR, fname), "r", encoding="utf-8") as f:
            texts[fname] = f.read()
    return texts


# --------------------------------------------------------------------------
# 1. Orchestrator — intent classification
# --------------------------------------------------------------------------
SQL_KEYWORDS = [
    "how many", "count", "number of", "average", "list", "show", "which patients",
    "total", "highest", "lowest", "sum of", "patients with", "patients who",
    "billing amount", "test results", "admitted", "age", "gender", "blood type",
    "doctor", "hospital name",
]
POLICY_KEYWORDS = [
    "policy", "procedure", "require", "approval", "guideline", "allowed",
    "process for", "discharge process", "how do i", "am i eligible",
    "documentation needed", "prior authorization", "insurance approval",
    "protocol", "rules", "fee", "refund", "dispute",
]
DATA_COLUMN_HINTS = [
    "diabetic", "cancer", "obesity", "asthma", "hypertension", "arthritis",
    "abnormal", "normal", "inconclusive", "elective", "urgent", "emergency admission",
]
STRONG_POLICY_PHRASES = [
    "is prior insurance approval required", "what is the hospital discharge policy",
    "billing policy", "admission policy", "emergency policy", "insurance policy",
    "discharge policy", "what documents", "how long does",
]


def classify_intent(query: str):
    ql = query.lower()
    sql_score = sum(1 for kw in SQL_KEYWORDS if kw in ql)
    sql_score += sum(1 for kw in DATA_COLUMN_HINTS if kw in ql)
    policy_score = sum(1 for kw in POLICY_KEYWORDS if kw in ql)
    if any(p in ql for p in STRONG_POLICY_PHRASES):
        policy_score += 3

    if sql_score == 0 and policy_score == 0:
        return "OUT_OF_DOMAIN", 0.0
    if sql_score > policy_score:
        return "SQL", sql_score / (sql_score + policy_score)
    if policy_score > sql_score:
        return "POLICY", policy_score / (sql_score + policy_score)
    return "AMBIGUOUS", 0.5


# --------------------------------------------------------------------------
# 2. NLP-to-SQL agent
# --------------------------------------------------------------------------
CONDITIONS = ["cancer", "obesity", "diabetes", "asthma", "hypertension", "arthritis"]
ADMISSION_TYPES = ["elective", "urgent", "emergency"]
TEST_RESULTS = ["abnormal", "inconclusive", "normal"]
GENDERS = ["male", "female"]
BLOOD_TYPES = ["a+", "a-", "b+", "b-", "ab+", "ab-", "o+", "o-"]
INSURANCE = ["blue cross", "medicare", "aetna", "unitedhealthcare", "cigna"]

BASE_JOIN = """
FROM admissions a
JOIN patients p ON a.patient_id = p.patient_id
JOIN hospitals h ON a.hospital_id = h.hospital_id
JOIN doctors d ON a.doctor_id = d.doctor_id
JOIN insurance_providers ip ON a.provider_id = ip.provider_id
"""


def get_dataset_reference_date(conn):
    row = conn.execute("SELECT MAX(admission_date) FROM admissions").fetchone()
    return datetime.date.fromisoformat(row[0])


def parse_filters(q):
    ql = q.lower()
    filters = []
    for c in CONDITIONS:
        if c in ql or (c == "diabetes" and "diabetic" in ql):
            filters.append(f"LOWER(a.medical_condition) = '{c}'")
            break
    for t in ADMISSION_TYPES:
        if t in ql:
            filters.append(f"LOWER(a.admission_type) = '{t}'")
            break
    for tr in TEST_RESULTS:
        if re.search(rf"\b{tr}\b", ql):
            filters.append(f"LOWER(a.test_results) = '{tr}'")
            break
    for g in GENDERS:
        if re.search(rf"\b{g}\b", ql):
            filters.append(f"LOWER(p.gender) = '{g}'")
            break
    for bt in BLOOD_TYPES:
        if bt in ql:
            filters.append(f"UPPER(p.blood_type) = '{bt.upper()}'")
            break
    for ins in INSURANCE:
        if ins in ql:
            filters.append(f"LOWER(ip.provider_name) = '{ins}'")
            break
    m = re.search(r"(?:over|above|older than)\s+(\d+)", ql)
    if m:
        filters.append(f"p.age > {int(m.group(1))}")
    m = re.search(r"(?:under|below|younger than)\s+(\d+)", ql)
    if m:
        filters.append(f"p.age < {int(m.group(1))}")
    return filters


def parse_date_filter(q, ref_date):
    ql = q.lower()
    if "last month" in ql:
        first_of_this_month = ref_date.replace(day=1)
        last_month_end = first_of_this_month - datetime.timedelta(days=1)
        last_month_start = last_month_end.replace(day=1)
        return f"a.admission_date BETWEEN '{last_month_start}' AND '{last_month_end}'"
    m = re.search(r"\bin\s+(20\d{2})\b", ql)
    if m:
        y = m.group(1)
        return f"a.admission_date BETWEEN '{y}-01-01' AND '{y}-12-31'"
    return None


def build_sql(query, conn):
    ql = query.lower()
    filters = parse_filters(query)
    ref_date = get_dataset_reference_date(conn)
    date_filter = parse_date_filter(query, ref_date)
    if date_filter:
        filters.append(date_filter)
    where = f"WHERE {' AND '.join(filters)}" if filters else ""

    m = re.search(
        r"average\s+billing\s+amount\s+by\s+(insurance provider|hospital|gender|condition|admission type|doctor)",
        ql,
    )
    if m:
        group_map = {
            "insurance provider": ("ip.provider_name", "provider"),
            "hospital": ("h.hospital_name", "hospital"),
            "gender": ("p.gender", "gender"),
            "condition": ("a.medical_condition", "condition"),
            "admission type": ("a.admission_type", "admission_type"),
            "doctor": ("d.doctor_name", "doctor"),
        }
        col, label = group_map[m.group(1)]
        sql = f"SELECT {col} AS {label}, ROUND(AVG(a.billing_amount),2) AS avg_billing {BASE_JOIN} {where} GROUP BY {col} ORDER BY avg_billing DESC"
        return sql, "aggregate_table"

    if re.search(r"\bhow many\b|\bcount\b|\bnumber of\b", ql):
        sql = f"SELECT COUNT(*) AS patient_count {BASE_JOIN} {where}"
        return sql, "single_value"

    if re.search(r"\btotal billing\b|\bsum of billing\b", ql):
        sql = f"SELECT ROUND(SUM(a.billing_amount),2) AS total_billing {BASE_JOIN} {where}"
        return sql, "single_value"

    if re.search(r"\bhighest billing\b|\bmax(?:imum)? billing\b", ql):
        sql = f"SELECT p.name, a.billing_amount {BASE_JOIN} {where} ORDER BY a.billing_amount DESC LIMIT 10"
        return sql, "list"

    sql = (
        "SELECT p.name, p.age, p.gender, a.medical_condition, a.admission_type, "
        "a.test_results, a.billing_amount, h.hospital_name, ip.provider_name, a.admission_date "
        f"{BASE_JOIN} {where} ORDER BY a.admission_date DESC LIMIT 20"
    )
    return sql, "list"


def nl_to_sql_agent(query, conn):
    sql, kind = build_sql(query, conn)
    try:
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        return {"sql": sql, "kind": kind, "columns": cols, "rows": rows, "error": None}
    except Exception as e:
        return {"sql": sql, "kind": kind, "columns": [], "rows": [], "error": str(e)}


# --------------------------------------------------------------------------
# 3. RAG agent
# --------------------------------------------------------------------------
RETRIEVAL_TOP_K = 3
RETRIEVAL_MIN_SCORE = 0.25


def retrieve_policy_chunks(query, top_k=RETRIEVAL_TOP_K):
    index, chunks = get_vector_store()
    embed_model = get_embed_model()
    q_emb = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, idx = index.search(q_emb, top_k)
    results = []
    for score, i in zip(scores[0], idx[0]):
        doc_name, chunk = chunks[i]
        results.append({"doc": doc_name, "chunk": chunk, "score": float(score)})
    return results


def generate_answer(prompt, max_new_tokens=120):
    tok, model = get_gen_model()
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tok.decode(outputs[0], skip_special_tokens=True)


def rag_agent(query, use_generation=True):
    hits = retrieve_policy_chunks(query)
    if not hits or hits[0]["score"] < RETRIEVAL_MIN_SCORE:
        return {
            "answer": "I couldn't find anything in the hospital policy documents that "
                      "covers this question. Please rephrase or check with the policy office.",
            "sources": [],
            "hits": hits,
        }

    sources = sorted(set(h["doc"] for h in hits))
    if not use_generation:
        # Extractive fallback: return the top chunk directly, no generative model needed.
        return {"answer": hits[0]["chunk"], "sources": sources, "hits": hits}

    context = "\n\n".join(f"[{h['doc']}] {h['chunk']}" for h in hits)
    prompt = (
        "Answer the question using ONLY the hospital policy context below. "
        "Be concise and specific.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    try:
        output = generate_answer(prompt)
    except Exception:
        output = hits[0]["chunk"]
    return {"answer": output.strip(), "sources": sources, "hits": hits}


# --------------------------------------------------------------------------
# 4. Conversation manager / router — returns a structured result for the UI
# --------------------------------------------------------------------------
def ask(query: str, use_generation: bool = True) -> dict:
    conn = get_db()
    start = time.time()
    intent, confidence = classify_intent(query)
    result = {"query": query, "intent": intent, "confidence": confidence}

    try:
        if intent == "SQL":
            r = nl_to_sql_agent(query, conn)
            result.update(_pack_sql(r))
        elif intent == "POLICY":
            r = rag_agent(query, use_generation)
            result.update(_pack_policy(r))
        elif intent == "AMBIGUOUS":
            sql_r = nl_to_sql_agent(query, conn)
            rag_r = rag_agent(query, use_generation)
            result.update(_pack_sql(sql_r, prefix="sql_"))
            result.update(_pack_policy(rag_r, prefix="policy_"))
        else:
            result["text"] = (
                "I can only help with questions about patient records or hospital "
                "policies (admission, billing, discharge, insurance, emergency care). "
                "Could you rephrase your question around one of those topics?"
            )
    except Exception as e:
        result["intent"] = intent + "_ERROR"
        result["text"] = f"Something went wrong answering that ({e}). Please try rephrasing your question."

    result["latency"] = time.time() - start
    return result


def _pack_sql(r, prefix=""):
    out = {f"{prefix}sql": r["sql"], f"{prefix}sql_error": r["error"]}
    if r["error"]:
        out[f"{prefix}sql_kind"] = "error"
        return out
    df = pd.DataFrame(r["rows"], columns=r["columns"]) if r["columns"] else pd.DataFrame()
    if r["kind"] == "single_value" and len(df) == 1 and len(df.columns) == 1:
        out[f"{prefix}sql_kind"] = "metric"
        out[f"{prefix}sql_metric_label"] = df.columns[0].replace("_", " ").title()
        out[f"{prefix}sql_metric_value"] = df.iloc[0, 0]
    elif df.empty:
        out[f"{prefix}sql_kind"] = "empty"
    else:
        out[f"{prefix}sql_kind"] = "table"
        out[f"{prefix}sql_table"] = df
    return out


def _pack_policy(r, prefix=""):
    return {
        f"{prefix}text": r["answer"],
        f"{prefix}sources": r["sources"],
        f"{prefix}hits": r["hits"],
    }


# --------------------------------------------------------------------------
# UI helpers
# --------------------------------------------------------------------------
INTENT_LABELS = {
    "SQL": ("🗄️", "Patient Database"),
    "POLICY": ("📄", "Policy Documents"),
    "AMBIGUOUS": ("🔀", "Both Agents"),
    "OUT_OF_DOMAIN": ("🚫", "Out of Domain"),
}


def intent_badge(intent):
    base = intent.replace("_ERROR", "")
    css_class = "ERROR" if "_ERROR" in intent else base
    icon, label = INTENT_LABELS.get(base, ("⚠️", base))
    return f'<span class="intent-tag intent-{css_class}">{icon} {label}</span>'


def pipeline_diagram(active_intent=None):
    base = (active_intent or "").replace("_ERROR", "")
    sql_cls = "active-sql" if base in ("SQL", "AMBIGUOUS") else ""
    policy_cls = "active-policy" if base in ("POLICY", "AMBIGUOUS") else ""
    always_cls = "active-always" if active_intent else ""
    html = f"""
    <div class="pipeline">
      <div class="pnode {always_cls}">🧑 User Query</div>
      <div class="parrow">→</div>
      <div class="pnode {always_cls}">🧭 Orchestrator<span class="sub">intent classifier</span></div>
      <div class="parrow">→</div>
      <div class="pnode {sql_cls}">🗄️ NLP-to-SQL<span class="sub">SQLite</span></div>
      <div class="pnode {policy_cls}">📄 RAG Agent<span class="sub">FAISS + flan-t5</span></div>
      <div class="parrow">→</div>
      <div class="pnode {always_cls}">🧩 Formatter</div>
      <div class="parrow">→</div>
      <div class="pnode {always_cls}">💬 Answer</div>
    </div>
    """
    return html


def render_result(result: dict):
    intent = result["intent"]
    st.markdown(intent_badge(intent), unsafe_allow_html=True)

    base = intent.replace("_ERROR", "")

    if base == "SQL":
        _render_sql_block(result)
    elif base == "POLICY":
        _render_policy_block(result)
    elif base == "AMBIGUOUS":
        st.caption("This question touches both patient data and hospital policy.")
        st.markdown("**From the patient database:**")
        _render_sql_block(result, prefix="sql_")
        st.markdown("**From policy documents:**")
        _render_policy_block(result, prefix="policy_")
    else:
        st.info(result.get("text", ""))

    with st.expander("⏱ Query trace", expanded=False):
        c1, c2 = st.columns(2)
        c1.metric("Routing confidence", f"{result['confidence']*100:.0f}%")
        c2.metric("Latency", f"{result['latency']*1000:.0f} ms")


def _render_sql_block(result, prefix=""):
    kind = result.get(f"{prefix}sql_kind")
    if kind == "error":
        st.error(f"SQL execution failed: {result.get(f'{prefix}sql_error')}")
    elif kind == "metric":
        val = result.get(f"{prefix}sql_metric_value")
        label = result.get(f"{prefix}sql_metric_label", "Result")
        if isinstance(val, (int, float)) and "billing" in label.lower():
            st.metric(label, f"${val:,.2f}")
        else:
            st.metric(label, f"{val:,}" if isinstance(val, (int, float)) else val)
    elif kind == "empty":
        st.warning("No matching patient records were found for that query.")
    elif kind == "table":
        df = result.get(f"{prefix}sql_table")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(f"{len(df)} row(s) returned.")

    sql_text = result.get(f"{prefix}sql")
    if sql_text:
        with st.expander("🔍 View generated SQL"):
            st.markdown(f'<div class="sql-block">{sql_text.strip()}</div>', unsafe_allow_html=True)


def _render_policy_block(result, prefix=""):
    st.write(result.get(f"{prefix}text", ""))
    sources = result.get(f"{prefix}sources") or []
    if sources:
        chips = "".join(f'<span class="src-chip">{POLICY_ICONS.get(s,"📄")} {POLICY_FILES.get(s,s)}</span>' for s in sources)
        st.markdown(chips, unsafe_allow_html=True)
    hits = result.get(f"{prefix}hits") or []
    if hits:
        with st.expander("🔎 View retrieved passages & similarity scores"):
            for h in hits:
                st.markdown(
                    f"**{POLICY_FILES.get(h['doc'], h['doc'])}** · similarity `{h['score']:.3f}`"
                )
                st.caption(h["chunk"][:400] + ("…" if len(h["chunk"]) > 400 else ""))


# --------------------------------------------------------------------------
# Session state init
# --------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"assistant", ...}
if "routing_log" not in st.session_state:
    st.session_state.routing_log = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "use_generation" not in st.session_state:
    st.session_state.use_generation = True

SAMPLE_QUERIES = [
    "How many diabetic patients were admitted last month?",
    "Which patients have abnormal test results?",
    "What is the hospital discharge policy?",
    "Is prior insurance approval required for surgery?",
    "Show the average billing amount by insurance provider.",
    "What documents do I need to be admitted?",
]

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏥 Healthcare Query Assistant")
    st.caption("Multi-agent RAG + NLP-to-SQL system")
    page = st.radio(
        "Navigate",
        ["💬 Chat Assistant", "📊 Dataset Analytics", "📄 Policy Library", "🧠 How It Works", "📈 Routing Insights"],
        label_visibility="collapsed",
    )
    st.divider()

    try:
        _df_quick = load_analytics_frame()
        st.markdown("**Live database snapshot**")
        st.markdown(f"- {len(_df_quick):,} admission records")
        st.markdown(f"- {_df_quick['hospital_name'].nunique():,} hospitals")
        st.markdown(f"- ${_df_quick['billing_amount'].mean():,.0f} avg billing")
    except Exception:
        pass

    st.divider()
    st.markdown("**Generation mode**")
    st.session_state.use_generation = st.toggle(
        "Use flan-t5 to compose policy answers",
        value=st.session_state.use_generation,
        help="Off = extractive mode: returns the top matching policy passage directly, "
             "without loading the generative model. Faster & lighter on memory.",
    )

    st.divider()
    st.markdown("**Try a sample question**")
    for q in SAMPLE_QUERIES:
        if st.button(q, key=f"sample_{q}", use_container_width=True):
            st.session_state.pending_query = q

    st.divider()
    if st.button("🗑 Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.routing_log = []
        st.rerun()

# --------------------------------------------------------------------------
# PAGE: Chat Assistant
# --------------------------------------------------------------------------
if page == "💬 Chat Assistant":
    st.markdown(
        """
        <div class="hero">
          <div class="hero-eyebrow">Orchestrator · NLP-to-SQL · RAG · Formatter</div>
          <h1>Ask about patients or hospital policy</h1>
          <p>One assistant, two grounded agents. Structured questions are answered by
          querying the live patient database; policy questions are answered from the
          hospital's official documents via retrieval-augmented generation — never
          from the model's memory alone.</p>
          <div class="hero-badges">
            <span class="badge">SQLite · 55K+ admissions</span>
            <span class="badge">FAISS vector search</span>
            <span class="badge">all-MiniLM-L6-v2 embeddings</span>
            <span class="badge">flan-t5-base generation</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    last_intent = st.session_state.messages[-1]["data"]["intent"] if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant" else None
    st.markdown(pipeline_diagram(last_intent), unsafe_allow_html=True)
    st.write("")

    # Replay history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🏥"):
                render_result(msg["data"])

    # Handle a sample-question click
    query = st.chat_input("Ask about a patient record or a hospital policy…")
    if st.session_state.pending_query:
        query = st.session_state.pending_query
        st.session_state.pending_query = None

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)
        with st.chat_message("assistant", avatar="🏥"):
            with st.spinner("Routing query…"):
                result = ask(query, use_generation=st.session_state.use_generation)
            render_result(result)
        st.session_state.messages.append({"role": "assistant", "data": result})
        st.session_state.routing_log.append({
            "query": query,
            "intent": result["intent"],
            "confidence": round(result["confidence"], 2),
            "latency_ms": round(result["latency"] * 1000, 1),
        })
        st.rerun()

# --------------------------------------------------------------------------
# PAGE: Dataset Analytics
# --------------------------------------------------------------------------
elif page == "📊 Dataset Analytics":
    st.markdown(
        """
        <div class="hero">
          <div class="hero-eyebrow">Phase 1 · Data Preparation</div>
          <h1>Patient database at a glance</h1>
          <p>The normalized SQLite schema behind the NLP-to-SQL agent — patients,
          admissions, hospitals, doctors, and insurance providers — explored directly.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_analytics_frame()

    conditions = st.multiselect("Filter by medical condition", sorted(df["medical_condition"].unique()))
    adm_types = st.multiselect("Filter by admission type", sorted(df["admission_type"].unique()))
    fdf = df.copy()
    if conditions:
        fdf = fdf[fdf["medical_condition"].isin(conditions)]
    if adm_types:
        fdf = fdf[fdf["admission_type"].isin(adm_types)]

    k1, k2, k3, k4 = st.columns(4)
    for col, label, value in zip(
        [k1, k2, k3, k4],
        ["Admissions", "Avg. Billing", "Avg. Stay", "Hospitals"],
        [f"{len(fdf):,}", f"${fdf['billing_amount'].mean():,.0f}", f"{fdf['stay_days'].mean():.1f} days", f"{fdf['hospital_name'].nunique():,}"],
    ):
        col.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div></div>', unsafe_allow_html=True)

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        cond_counts = fdf["medical_condition"].value_counts().reset_index()
        cond_counts.columns = ["condition", "count"]
        fig = px.bar(cond_counts, x="count", y="condition", orientation="h",
                     color="count", color_continuous_scale=["#E4EFEE", "#0E4F52"])
        fig.update_layout(title="Admissions by medical condition", showlegend=False,
                           coloraxis_showscale=False, yaxis_title="", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        adm_counts = fdf["admission_type"].value_counts().reset_index()
        adm_counts.columns = ["type", "count"]
        fig = px.pie(adm_counts, names="type", values="count", hole=0.55,
                     color_discrete_sequence=["#0E4F52", "#3E7CB1", "#C98A2E"])
        fig.update_layout(title="Admission type distribution")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        bill_by_ins = fdf.groupby("insurance_provider")["billing_amount"].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(bill_by_ins, x="insurance_provider", y="billing_amount",
                     color_discrete_sequence=["#3E7CB1"])
        fig.update_layout(title="Average billing by insurance provider", xaxis_title="", yaxis_title="Avg. billing ($)")
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        fig = px.histogram(fdf, x="age", nbins=30, color_discrete_sequence=["#8B5FA3"])
        fig.update_layout(title="Patient age distribution", xaxis_title="Age", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        tr_counts = fdf["test_results"].value_counts().reset_index()
        tr_counts.columns = ["result", "count"]
        fig = px.bar(tr_counts, x="result", y="count", color="result",
                     color_discrete_sequence=["#C9564B", "#C98A2E", "#0E4F52"])
        fig.update_layout(title="Test results breakdown", showlegend=False, xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    with c6:
        monthly = fdf.set_index("admission_date").resample("ME").size().reset_index(name="admissions")
        fig = px.line(monthly, x="admission_date", y="admissions", markers=True,
                      color_discrete_sequence=["#0E4F52"])
        fig.update_layout(title="Admissions over time", xaxis_title="", yaxis_title="Admissions")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 View filtered records"):
        st.dataframe(fdf.head(200), use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------
# PAGE: Policy Library
# --------------------------------------------------------------------------
elif page == "📄 Policy Library":
    st.markdown(
        """
        <div class="hero">
          <div class="hero-eyebrow">Phase 1 · Knowledge Base</div>
          <h1>Hospital policy documents</h1>
          <p>The five source documents behind the RAG agent — chunked, embedded with
          all-MiniLM-L6-v2, and indexed in FAISS for retrieval.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    texts = load_policy_texts()
    search = st.text_input("🔍 Search across all policies", placeholder="e.g. minors, refund, pre-authorization")

    cols = st.columns(len(POLICY_FILES))
    for col, (fname, label) in zip(cols, POLICY_FILES.items()):
        col.markdown(
            f'<div class="kpi" style="text-align:center;"><div style="font-size:1.6rem">{POLICY_ICONS[fname]}</div>'
            f'<div class="label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    for fname, label in POLICY_FILES.items():
        text = texts[fname]
        matched = search.lower() in text.lower() if search else True
        if not matched:
            continue
        with st.expander(f"{POLICY_ICONS[fname]}  {label} Policy", expanded=bool(search)):
            if search:
                highlighted = re.sub(f"(?i)({re.escape(search)})", r"**:orange[\1]**", text)
                st.markdown(highlighted)
            else:
                st.text(text)

# --------------------------------------------------------------------------
# PAGE: How It Works
# --------------------------------------------------------------------------
elif page == "🧠 How It Works":
    st.markdown(
        """
        <div class="hero">
          <div class="hero-eyebrow">Architecture</div>
          <h1>How the multi-agent system works</h1>
          <p>Four cooperating components, each with a narrow, testable job.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(pipeline_diagram(), unsafe_allow_html=True)
    st.write("")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="card">
            <h4>🧭 Orchestrator Agent</h4>
            <p class="small-note">Classifies each query as <b>SQL</b>, <b>POLICY</b>, <b>AMBIGUOUS</b>,
            or <b>OUT_OF_DOMAIN</b> using transparent keyword/phrase scoring — auditable and easy to
            extend, unlike a black-box classifier.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(
            """
            <div class="card">
            <h4>🗄️ NLP-to-SQL Agent</h4>
            <p class="small-note">A template-driven engine extracts filters (condition, gender,
            blood type, admission type, insurance, age, dates) and an aggregation type, then
            assembles a parameterized SQL query against the normalized SQLite schema. Deterministic
            and cross-validated against pandas for accuracy.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="card">
            <h4>📄 RAG Agent</h4>
            <p class="small-note">Retrieves the top-k most similar policy chunks from a FAISS index
            (cosine similarity via normalized embeddings), then grounds a flan-t5-base generation
            in that context. If similarity falls below 0.25, it refuses rather than hallucinating.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(
            """
            <div class="card">
            <h4>🧩 Response Formatter</h4>
            <p class="small-note">Normalizes whichever agent ran — or both, for ambiguous queries —
            into one consistent reply shape, so the interface layer never needs to know which
            agent produced the answer.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("#### Tech stack")
    st.markdown(
        """
        <div class="hero-badges" style="margin-top:0;">
          <span class="badge" style="background:var(--teal-tint);color:var(--teal-dark);border-color:var(--teal);">SQLite</span>
          <span class="badge" style="background:var(--blue-tint);color:var(--blue);border-color:var(--blue);">FAISS (IndexFlatIP)</span>
          <span class="badge" style="background:var(--amber-tint);color:var(--amber);border-color:var(--amber);">sentence-transformers</span>
          <span class="badge" style="background:var(--purple-tint);color:var(--purple);border-color:var(--purple);">flan-t5-base</span>
          <span class="badge" style="background:#EEF1F0;color:var(--ink-soft);border-color:var(--border);">Streamlit</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------
# PAGE: Routing Insights
# --------------------------------------------------------------------------
elif page == "📈 Routing Insights":
    st.markdown(
        """
        <div class="hero">
          <div class="hero-eyebrow">Phase 5 · Evaluation</div>
          <h1>Routing &amp; latency log</h1>
          <p>Every query asked in this session, with the orchestrator's routing decision,
          confidence, and latency — the same telemetry the notebook's ConversationManager logs.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.routing_log:
        st.info("No queries yet this session — ask something in the Chat Assistant tab first.")
    else:
        log_df = pd.DataFrame(st.session_state.routing_log)
        k1, k2, k3 = st.columns(3)
        k1.markdown(f'<div class="kpi"><div class="label">Queries</div><div class="value">{len(log_df)}</div></div>', unsafe_allow_html=True)
        k2.markdown(f'<div class="kpi"><div class="label">Avg. Confidence</div><div class="value">{log_df["confidence"].mean()*100:.0f}%</div></div>', unsafe_allow_html=True)
        k3.markdown(f'<div class="kpi"><div class="label">Avg. Latency</div><div class="value">{log_df["latency_ms"].mean():.0f} ms</div></div>', unsafe_allow_html=True)

        st.write("")
        c1, c2 = st.columns(2)
        with c1:
            intent_counts = log_df["intent"].value_counts().reset_index()
            intent_counts.columns = ["intent", "count"]
            fig = px.bar(intent_counts, x="intent", y="count", color="intent",
                         color_discrete_sequence=["#3E7CB1", "#C98A2E", "#8B5FA3", "#5B6C70", "#C9564B"])
            fig.update_layout(title="Queries by intent", showlegend=False, xaxis_title="", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(log_df.reset_index(), x="index", y="latency_ms", color="intent",
                         color_discrete_sequence=["#3E7CB1", "#C98A2E", "#8B5FA3", "#5B6C70", "#C9564B"])
            fig.update_layout(title="Latency per query", xaxis_title="Query #", yaxis_title="ms")
            st.plotly_chart(fig, use_container_width=True)

        st.write("")
        st.markdown("#### Full log")
        st.dataframe(log_df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇ Download log as CSV",
            log_df.to_csv(index=False).encode("utf-8"),
            file_name="routing_log.csv",
            mime="text/csv",
        )
