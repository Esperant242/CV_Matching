# CV Matcher — AI-Powered Resume Matching Engine

> Match any job description against a 13,389-CV corpus in seconds. Dense vector retrieval narrows the field; an LLM judge scores and explains each candidate.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Pipeline Deep-Dive](#pipeline-deep-dive)
  - [1. Data Loading](#1-data-loading)
  - [2. Document Building](#2-document-building)
  - [3. Chunking Strategy](#3-chunking-strategy)
  - [4. Vector Indexing (FAISS)](#4-vector-indexing-faiss)
  - [5. Semantic Retrieval](#5-semantic-retrieval)
  - [6. LLM Scoring](#6-llm-scoring)
- [Scoring Rubric](#scoring-rubric)
- [Streamlit UI](#streamlit-ui)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Environment Variables](#environment-variables)
- [Key Design Decisions](#key-design-decisions)

---

## Overview

**CV Matcher** is a full RAG *(Retrieval-Augmented Generation)* pipeline built for recruiting teams. Given a plain-language job description, it:

1. Embeds the query and retrieves the most semantically relevant CVs from a FAISS vector index
2. Sends each candidate's resume to an LLM with a structured evaluation prompt
3. Returns a ranked shortlist with a score out of 20, a decision badge, and a written justification per candidate

The result is a scored, explainable shortlist — not just a list of similar documents.

---

## Architecture

![CV Matcher — RAG Pipeline Architecture](docs/architecture.png)

---

## How It Works

```
Job Description (text)
        │
        ▼
┌───────────────────┐
│  OpenAI Embeddings │  text-embedding-3-small
│  (query vector)    │
└────────┬──────────┘
         │  cosine similarity
         ▼
┌───────────────────┐
│   FAISS Index      │  67,101 chunks from 13,389 CVs
│   (dense retrieval)│
└────────┬──────────┘
         │  top-K chunks → grouped by resume_id
         ▼
┌───────────────────┐
│   LLM Judge        │  GPT-4.1-mini, temp=0, seed=42
│   (scoring)        │
└────────┬──────────┘
         │
         ▼
  Ranked shortlist
  Score /20 + Decision + Justification
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Orchestration | LangChain 0.3 |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | FAISS (`faiss-cpu` 1.9.0) |
| LLM Judge | OpenAI `gpt-4.1-mini` |
| UI | Streamlit 1.55 |
| Data | pandas, numpy |
| Config | python-dotenv |
| Validation | pydantic |

---

## Project Structure

```
RAG_Project/
├── streamlit_app.py          # Main Streamlit UI (entry point)
├── config.py                 # Central configuration (paths, models, params)
├── main.py                   # CLI entry point
├── requirements.txt
├── .env                      # API keys (not versioned)
├── .streamlit/
│   └── config.toml           # Streamlit theme (dark, indigo)
│
├── src/
│   ├── load_data.py          # CSV loading → pandas DataFrame
│   ├── build_documents.py    # DataFrame → LangChain Documents + chunking
│   ├── build_index.py        # Full indexing pipeline (run once)
│   ├── retrieve.py           # FAISS similarity search
│   ├── rank_with_llm.py      # LLM scoring + ranking
│   ├── app.py                # CLI orchestrator
│   │
│   ├── chunking/
│   │   └── splitter.py
│   ├── ingestion/
│   │   ├── cleaner.py
│   │   └── loader.py
│   ├── ranking/
│   │   └── ranker.py
│   ├── retrieval/
│   │   └── retriever.py
│   └── vectorstore/
│       ├── embedder.py
│       └── store.py
│
├── data/
│   └── processed/
│       └── Preprocessed_Data.txt   # Source dataset (not versioned)
│
└── tests/
    └── test_retrieval.py
```

---

## Dataset

The corpus is built from a publicly available resume dataset containing **13,389 CVs** across **43 professional categories**.

| Stat | Value |
|---|---|
| Total CVs | 13,389 |
| Categories | 43 |
| Avg CV length | ~3,986 characters |
| Total chunks (after split) | 67,101 |
| Avg chunk size | ~1,000 characters (~250 tokens) |
| Missing values | 0 |

**Categories include:** Data Science, HR, Software Development, Finance, Civil Engineering, Blockchain, Aviation, Marketing, Healthcare, Legal, and 33 more.

The raw data file (`Preprocessed_Data.txt`) is a CSV with two columns: `Category` and `Text`. It is excluded from version control due to size.

---

## Pipeline Deep-Dive

### 1. Data Loading

**File:** `src/load_data.py`

Loads the preprocessed CSV using pandas. The path is resolved relative to the script location using `Path(__file__).resolve().parent.parent` to ensure it works regardless of the working directory.

```python
df = pd.read_csv(DATA_PATH, dtype=str).dropna(subset=["Text"])
# → 13,389 rows × 2 columns
```

---

### 2. Document Building

**File:** `src/build_documents.py`

Each row is converted into a LangChain `Document` object:

```python
Document(
    page_content=row["Text"],      # raw CV text
    metadata={
        "resume_id": idx,          # original DataFrame index
        "category": row["Category"]
    }
)
```

Metadata is preserved through all chunking and retrieval steps, making it possible to group chunks back by candidate at scoring time.

---

### 3. Chunking Strategy

**File:** `src/build_documents.py` — `split_documents()`

The chunking parameters were chosen based on dataset analysis:

| Parameter | Value | Rationale |
|---|---|---|
| `chunk_size` | 1,000 chars | ≈ 250 tokens — optimal window for `text-embedding-3-small` |
| `chunk_overlap` | 150 chars | 15% overlap — avoids cutting mid-sentence or mid-skill-list |
| Splitter | `RecursiveCharacterTextSplitter` | Tries `\n\n`, `\n`, `. `, ` ` in order before hard-cutting |

Result: **13,389 documents → 67,101 chunks** (ratio ≈ 5 chunks/CV).

---

### 4. Vector Indexing (FAISS)

**File:** `src/build_index.py`

This step runs **once** to build and persist the vector index. It should be re-run only if the dataset changes.

```
[1/4] Load dataset      → 13,389 CVs
[2/4] Build documents   → 13,389 LangChain Documents
[3/4] Chunk             → 67,101 chunks
[4/4] Embed + Index     → FAISS saved to disk
```

**Embedding model:** `text-embedding-3-small` (OpenAI) — chosen for its balance between quality and cost at scale (67k chunks).

The index is stored outside the project directory (on `D:/RAG_vectorstore/`) because of its size and to avoid versioning large binary files.

```bash
python src/build_index.py
```

---

### 5. Semantic Retrieval

**File:** `src/retrieve.py`

At query time, the job description is embedded and compared against all 67,101 chunk vectors via cosine similarity:

```python
vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings)
results = vectorstore.similarity_search(query, k=K)
```

The top-K chunks are returned and then **grouped by `resume_id`** so that multiple chunks from the same CV are merged into a single candidate entry. Candidates are sorted deterministically by `resume_id` before being passed to the LLM.

---

### 6. LLM Scoring

**File:** `src/rank_with_llm.py`

Each retrieved candidate is scored by `gpt-4.1-mini` with:

- `temperature=0` — deterministic output
- `seed=42` — reproducible results across identical inputs
- A strict structured prompt enforcing a fixed scoring rubric (see below)

The LLM returns a raw JSON object with:

```json
{
  "candidate_id": 1042,
  "category": "Data Science",
  "score_sur_20": 17,
  "decision": "Excellent match",
  "justification": "The candidate demonstrates strong Python and SQL skills..."
}
```

A backtick-stripping fallback handles cases where the model wraps its JSON in a markdown code block despite explicit instructions. Results are sorted descending by `score_sur_20`.

---

## Scoring Rubric

The LLM is constrained to the following fixed intervals — injected into every prompt to prevent hallucination and ensure consistent scoring across queries:

| Score | Decision | Criteria |
|---|---|---|
| **17 – 20** | Excellent match | Meets almost all requirements. Strong skills, relevant experience, right domain. |
| **13 – 16** | Good match | Meets most requirements. Minor gaps on secondary skills or experience level. |
| **8 – 12** | Partial match | Meets some requirements. Noticeable gaps on key skills or domain. |
| **0 – 7** | Not suitable | Does not meet main requirements. Wrong domain or missing core skills. |

Rules enforced in the prompt:
- Never score above 16 unless the candidate clearly meets **all** major requirements
- Never score below 8 unless the candidate is clearly in the wrong domain
- The decision label must exactly match the score interval
- Two similar CVs for the same job must receive similar scores

---

## Streamlit UI

**File:** `streamlit_app.py`

A single-mode, dark-themed recruiting dashboard built with Streamlit.

**Sidebar:**
- Shortlist size slider (1–10 candidates)
- 10 preset role examples (Data Scientist, DevOps, HR Recruiter, Finance Analyst, etc.)
- Recent search history with numbered entries

**Main area:**
- Gradient hero banner (indigo → violet → pink) with product name and description
- Job description textarea with placeholder guidance
- Summary metrics after search: profiles scored, best score, average score, top domain
- Tabbed results: **Candidate cards** (rank pill, decision badge, score /20, progress bar, written assessment) and **Table view** (with sortable progress column)
- Empty state with 3-step explainer

**Theme:** Dark space gradient (`#0F0C29 → #1a1040 → #24243e`) with glassmorphism surfaces (`backdrop-filter: blur`), indigo accent (`#818CF8`), Inter font.

```bash
streamlit run streamlit_app.py
# → http://localhost:8501
```

---

## Setup & Installation

### Prerequisites

- Python 3.11+
- An OpenAI API key
- The FAISS index built and saved to disk (see step 6 below)

### 1. Clone the repository

```bash
git clone https://github.com/Esperant242/CV_Matching.git
cd CV_Matching
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file at the project root:

```env
OPENAI_API_KEY=sk-proj-...
```

### 5. Place the dataset

Put `Preprocessed_Data.txt` in:

```
data/processed/Preprocessed_Data.txt
```

### 6. Build the FAISS index (run once)

```bash
python src/build_index.py
```

This embeds all 67,101 chunks and saves the FAISS index to disk. It takes a few minutes and consumes OpenAI embedding API credits (under $1 for the full corpus at `text-embedding-3-small` pricing).

### 7. Launch the app

```bash
streamlit run streamlit_app.py
```

---

## Usage

### Streamlit UI (recommended)

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501`, paste a job description, adjust the shortlist size in the sidebar, and click **Search candidates**.

### CLI

```bash
python src/app.py --query "Senior Python Developer with FastAPI and AWS experience" --k 5
```

### Retrieval only (no LLM scoring)

```python
from src.retrieve import retrieve_top_matches

results = retrieve_top_matches("Data scientist with NLP expertise", k=5)
for doc in results:
    print(doc.metadata, doc.page_content[:200])
```

### Full pipeline in Python

```python
from src.rank_with_llm import rank_candidates_with_llm

ranking = rank_candidates_with_llm(
    job_description="We are looking for a DevOps engineer with Kubernetes and Terraform...",
    k=5
)

for r in ranking:
    print(f"#{r['candidate_id']} | {r['category']} | {r['score_sur_20']}/20 — {r['decision']}")
    print(f"  {r['justification']}\n")
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | OpenAI API key used for both embeddings and LLM scoring |

---

## Key Design Decisions

**Why FAISS over a managed vector DB?**
For a corpus of ~70k vectors, `faiss-cpu` is fast enough locally with zero infrastructure overhead. The index loads in under a second at query time.

**Why `text-embedding-3-small`?**
Good balance of quality and cost. Embedding 67,101 chunks at ~250 tokens each costs under $1 with `text-embedding-3-small` pricing.

**Why `gpt-4.1-mini` for scoring?**
Fast, cheap, and instruction-following enough to respect the strict JSON output format and scoring rubric consistently at `temperature=0`. More powerful models would be overkill for structured extraction from a focused prompt.

**Why `temperature=0` + `seed=42`?**
Ensures deterministic, reproducible results. The same job description will always return the same ranking. Without this, OpenAI model non-determinism produces different scores for identical inputs.

**Why `chunk_size=1000`?**
~250 tokens — well within the context window of `text-embedding-3-small` while capturing enough content per chunk to be semantically meaningful (a full section of a CV rather than a few lines).

**Why merge chunks by `resume_id` before scoring?**
Retrieval returns chunks, not full CVs. Multiple chunks from the same candidate should be evaluated together. Merging gives the LLM a more complete picture of each profile, reducing false negatives caused by a CV split across a section boundary.

---

## License

MIT — free to use, fork, and adapt.
