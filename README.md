# RAG Course — Data Ingestion, Embeddings & Vector Stores

Hands-on notebooks for the first half of a Retrieval-Augmented Generation (RAG) pipeline: load raw files into LangChain `Document` objects, turn text into vector embeddings, store those vectors in ChromaDB, and build question-answering chains (including LCEL, conversational memory, and Groq LLMs). Each stage builds on the last:

```
raw files (TXT, PDF, DOCX, CSV, Excel, JSON, SQL)
        │  0-DataIngestParsing
        ▼
LangChain Document objects (+ splitting strategies)
        │  1-VectorEmbeddings
        ▼
numeric vectors (HuggingFace / OpenAI embeddings)
        │  2-Vector Stores
        ▼
ChromaDB  →  retriever  →  RAG chains
              ├── create_retrieval_chain (classic helpers)
              ├── LCEL pipe (retriever | prompt | llm)
              ├── conversational / history-aware RAG
              └── swap LLMs (OpenAI gpt-3.5-turbo ↔ Groq)
```

Remote: [Aspect661/RAG_Pipeline_Learning](https://github.com/Aspect661/RAG_Pipeline_Learning)

---

## Repo structure

```
.
├── 0-DataIngestParsing/          # Load & parse files → Document objects
│   ├── 1-dataingestion.ipynb     # Document object, TextLoader, DirectoryLoader, splitters
│   ├── 2-dataparsing.ipynb       # PDF loaders + SmartPDFProcessor
│   ├── 3-dataparsingdoc.ipynb    # Word (.docx) loaders
│   ├── 4-csvexcelparsing.ipynb   # CSV / Excel loaders + custom processors
│   ├── 5-jsonparsing.ipynb       # JSON / JSONL loaders + custom flattener
│   ├── 6-databaseparsing.ipynb   # SQLite → Documents
│   └── data/                     # Sample files (created/used by notebooks)
│       ├── text_files/           # machine_learning.txt, python_intro.txt
│       ├── pdf/                  # attention.pdf
│       ├── word_files/           # proposal.docx / proposal.txt
│       ├── structured_files/     # products.csv, inventory.xlsx
│       ├── json_files/           # company_data.json, events.jsonl
│       └── databases/            # company.db (SQLite)
├── 1-VectorEmbeddings/           # Embeddings from first principles + OpenAI
│   ├── embedding.ipynb           # HuggingFace / sentence-transformers (offline-capable)
│   └── openaiembeddings.ipynb    # OpenAIEmbeddings + semantic search
├── 2-Vector Stores/              # ChromaDB + full RAG systems
│   ├── 1-chromadb.ipynb          # End-to-end RAG notebook (main deliverable)
│   ├── data/                     # Sample .txt articles written by the notebook
│   └── chroma_db/                # Persisted Chroma collection (generated locally)
├── main.py                       # Placeholder (`Hello from udemy-ragcourse!`); unused by notebooks
├── pyproject.toml                # uv project name `udemy-ragcourse`, deps, Python ≥3.11
├── requirements.txt              # Flat dependency list (kept in sync manually)
├── uv.lock                       # Locked resolutions for `uv sync`
├── .python-version               # 3.11
├── .gitignore                    # Ignores `.venv`, `.env`, `__pycache__/`, build artifacts
├── .env                          # Local secrets (git-ignored) — OpenAI + Groq API keys
├── .vscode/settings.json         # Excludes `.venv` from analysis / file watching / search
└── .venv/                        # Created by `uv sync` (not committed)
```

---

## 0. Data Ingestion & Parsing

Loading raw files into LangChain `Document` objects — the first stage of a RAG pipeline. Most notebooks compare a **built-in LangChain loader** against a **custom parsing function**, so you can see convenience vs. control.

| Notebook | Topic | Built-in loader(s) | Custom / extra approach |
|---|---|---|---|
| `1-dataingestion.ipynb` | Intro to `Document`, loading text, splitting | `TextLoader`, `DirectoryLoader` | Manually build `Document(page_content=..., metadata=...)`; compare `CharacterTextSplitter`, `RecursiveCharacterTextSplitter` (recommended), and token-based splitting |
| `2-dataparsing.ipynb` | PDF parsing & cleaning | `PyPDFLoader`, `PyMuPDFLoader` | Page-by-page comparison; `clean_text()` for whitespace/ligatures; `SmartPDFProcessor` (clean → skip empty pages → chunk with richer metadata) |
| `3-dataparsingdoc.ipynb` | Word document parsing | `Docx2txtLoader`, `UnstructuredWordDocumentLoader` | Plain-text extraction vs. element-aware / unstructured parsing |
| `4-csvexcelparsing.ipynb` | CSV & Excel | `CSVLoader`, `UnstructuredCSVLoader`, `UnstructuredExcelLoader` | `process_csv_intelligently()` (row → structured prose + metadata); `process_excel_with_pandas()` (one document per sheet) |
| `5-jsonparsing.ipynb` | JSON / JSONL | `JSONLoader` (with `jq_schema`) | `process_json_intelligently()` — flattens nested employee/project records into natural-language profiles |
| `6-databaseparsing.ipynb` | SQL (SQLite) | `SQLDatabase`, `SQLDatabaseLoader` | `sql_to_documents()` — per-table overviews plus a synthesized employee↔project relationship document via SQL `JOIN` |

**Recurring theme:** built-in loaders parse a format mechanically (they don't know your schema). Custom loaders let you encode domain knowledge — natural-language phrasing, joins, nested-structure flattening — at the cost of more code and schema brittleness. Notebooks 4–6 show this most clearly.

**Data layout** under `0-DataIngestParsing/data/`:

```
data/
├── text_files/        # simple .txt samples for TextLoader / DirectoryLoader
├── pdf/               # attention.pdf
├── word_files/        # proposal.docx / proposal.txt
├── structured_files/  # products.csv, inventory.xlsx
├── json_files/        # company_data.json, events.jsonl
└── databases/         # company.db (SQLite)
```

Run each notebook top-to-bottom; earlier cells create many of the sample files that later cells parse.

---

## 1. Vector Embeddings

Turning text into numeric vectors, measuring similarity, and comparing embedding models/providers.

| Notebook | Topic |
|---|---|
| `embedding.ipynb` | Embeddings from first principles: a toy 2D word-embedding plot with matplotlib; `cosine_similarity()` from scratch; then `HuggingFaceEmbeddings` via `langchain_huggingface` (`sentence-transformers/all-MiniLM-L6-v2`) for single and batch embeds. Ends by comparing five popular `sentence-transformers` models (`all-MiniLM-L6-v2`, `all-mpnet-base-v2`, `all-MiniLM-L12-v2`, `multi-qa-MiniLM-L6-cos-v1`, `paraphrase-multilingual-MiniLM-L12-v2`) on dimensions, quality, and use case. |
| `openaiembeddings.ipynb` | Same ideas with `OpenAIEmbeddings` (`text-embedding-ada-002` / `text-embedding-3-small`): single & batch embedding; model comparison table (`ada-002` vs. `text-embedding-3-small` vs. `text-embedding-3-large` — dimensions, cost per 1M tokens, use case); pairwise cosine similarity; hand-rolled `semantic_search()` that ranks a small document set against a query. |

**API keys:** `openaiembeddings.ipynb` needs `OPENAI_API_KEY` in the root `.env` (loaded with `python-dotenv`). `embedding.ipynb` can run offline with local HuggingFace / sentence-transformers models (first run downloads weights).

---

## 2. Vector Stores

`2-Vector Stores/1-chromadb.ipynb` builds a working RAG system end-to-end with ChromaDB, then extends it with LCEL, incremental document adds, conversational memory, and Groq.

### Pipeline walkthrough

1. **Sample data** — three short articles (ML fundamentals, deep learning / neural nets, NLP) written to `2-Vector Stores/data/doc_0.txt` … `doc_2.txt`.
2. **Document loading** — `DirectoryLoader` + `TextLoader` load all `.txt` files under `data/`.
3. **Document splitting** — `RecursiveCharacterTextSplitter` (`chunk_size=500`, `chunk_overlap=50`, space separators) produces overlapping chunks.
4. **Embedding** — `OpenAIEmbeddings` (needs `OPENAI_API_KEY`).
5. **Vector storage** — `Chroma.from_documents(...)` persists embeddings to `./chroma_db` with collection name `rag_collection`.
6. **Similarity search** — `similarity_search()` and `similarity_search_with_score()`. Chroma’s default distance is **L2** (lower score = more similar; `0` = identical).
7. **LLM setup** — `ChatOpenAI(model_name="gpt-3.5-turbo", ...)` and/or `init_chat_model("openai:gpt-3.5-turbo")`.
8. **Classic RAG chain** — `vectorstore.as_retriever(search_kwargs={"k": 3})` + `ChatPromptTemplate` + `create_stuff_documents_chain` + `create_retrieval_chain`. Invoke with `{"input": "..."}`; response includes `answer` and retrieved `context`.
9. **LCEL RAG chain** — pipe `{context: retriever | format_docs, question: RunnablePassthrough()} | prompt | llm | StrOutputParser()`. Invoke with a plain string. Fetch sources separately via `retriever.invoke(question)` (not the removed `get_relevant_documents`).
10. **Add documents** — append new content (e.g. a reinforcement-learning article) into the existing Chroma collection and query again.
11. **Conversational / history-aware RAG** — `create_history_aware_retriever` reformulates follow-ups using chat history (`MessagesPlaceholder`, `HumanMessage` / `AIMessage`), then `create_retrieval_chain(history_aware_retriever, question_answer_chain)` answers with memory of prior turns.
12. **Groq LLMs** — load `GROQ_API_KEY` from `.env`; construct `ChatGroq(model="qwen/qwen3.6-27b")` or `init_chat_model(model="groq:qwen/qwen3.6-27b")` as an OpenAI substitute.

### Generated locally (safe to delete and recreate)

- `2-Vector Stores/data/` — sample `.txt` files created by the notebook
- `2-Vector Stores/chroma_db/` — Chroma persistence (SQLite + HNSW index files)

Re-run the notebook top-to-bottom to regenerate both.

---

## Setup

### Prerequisites

- Python **3.11** (see `.python-version`; `pyproject.toml` requires `>=3.11`)
- [`uv`](https://github.com/astral-sh/uv) recommended

### Install dependencies

From the repo root:

```bash
uv sync
```

This creates `.venv` with CPython 3.11 and installs everything from `uv.lock` / `pyproject.toml`.

Alternatively, with the flat requirements file:

```bash
uv pip install -r requirements.txt
```

Activate the venv (optional if you use `uv run`):

```bash
source .venv/bin/activate
```

### Environment variables

Create a root-level `.env` (already listed in `.gitignore`):

```
OPENAI_API_KEY="sk-..."
GROQ_API_KEY="gsk_..."
```

| Key | Used by |
|---|---|
| `OPENAI_API_KEY` | `1-VectorEmbeddings/openaiembeddings.ipynb`; embeddings + `ChatOpenAI` / `init_chat_model("openai:...")` in `2-Vector Stores/1-chromadb.ipynb` |
| `GROQ_API_KEY` | Groq section at the end of `1-chromadb.ipynb` (`ChatGroq` / `init_chat_model("groq:...")`) |

Load with `python-dotenv` (`load_dotenv()`) at the top of notebooks that need keys.

### Jupyter / Cursor kernel

Notebooks expect the **`udemy-ragcourse`** kernel (Python 3.11 from this project’s `.venv`). Register once after `uv sync`:

```bash
source .venv/bin/activate
python -m ipykernel install --user --name=udemy-ragcourse --display-name="udemy-ragcourse"
```

In Cursor / VS Code, pick that kernel from the picker in the top-right of each notebook. After installing new packages, **restart the kernel** before re-running cells.

`.vscode/settings.json` excludes `.venv` from Pyright analysis, file watching, and search so the IDE stays responsive.

### Notes on a few packages

- **Excel** (`pd.ExcelWriter`, `pd.read_excel`) needs `openpyxl`.
- **`UnstructuredExcelLoader`** needs the `unstructured[xlsx]` extra (pulls in tools like `msoffcrypto-tool`, `xlrd`); plain `unstructured[docx]` is not enough for `.xlsx`.
- **`JSONLoader`** with `jq_schema` needs the `jq` package.
- **`langchain-classic`** and **`langchain-text-splitters`** are used directly in the notebooks and listed in `requirements.txt` (they may only appear transitively via `langchain-community` otherwise).
- **`langchain-huggingface`** is required for `HuggingFaceEmbeddings` in `embedding.ipynb`.
- **`faiss-cpu`**, **`tiktoken`**, **`chromadb`**, **`sentence-transformers`**, **`pypdf`**, **`pymupdf`**, etc. are declared in `pyproject.toml` / `requirements.txt` for the loaders and vector-store work above.

---

## LangChain 1.x import & API gotchas

This project pins **`langchain>=1.3.13`**, a major rewrite. Snippets from older tutorials often break. Use these paths:

| What you need | Correct import / API |
|---|---|
| Text splitters | `langchain_text_splitters` (e.g. `RecursiveCharacterTextSplitter`) |
| `Document` | `langchain_core.documents` |
| Prompt / messages / LCEL | `langchain_core.prompts`, `langchain_core.messages`, `langchain_core.runnables`, `langchain_core.output_parsers` |
| `create_retrieval_chain`, `create_history_aware_retriever` | `langchain_classic.chains` (**not** `langchain.chains`) |
| `create_stuff_documents_chain` | `langchain_classic.chains.combine_documents` |
| Community loaders / Chroma / SQL utils | `langchain_community.document_loaders`, `langchain_community.vectorstores`, `langchain_community.utilities` |
| OpenAI chat & embeddings | `langchain_openai` (`ChatOpenAI`, `OpenAIEmbeddings`) |
| HuggingFace embeddings | `langchain_huggingface` (`HuggingFaceEmbeddings`) |
| Groq chat | `langchain_groq` (`ChatGroq`) |
| Retriever document fetch | `retriever.invoke(query)` — **`get_relevant_documents` was removed** |
| Classic RAG invoke | `rag_chain.invoke({"input": "..."})` → `result["answer"]`, `result["context"]` |
| LCEL RAG invoke | `rag_chain_lcel.invoke("...")` (string in, string out when using `StrOutputParser`) |
| Conversational RAG invoke | `conversational_rag_chain.invoke({"input": "...", "chat_history": [...]})` |

If you see `ModuleNotFoundError: No module named 'langchain.chains'` or `AttributeError: ... get_relevant_documents`, update the import/call as in the table above (the notebook already uses the fixed forms where we hit these).

---

## Suggested study order

1. `0-DataIngestParsing/1-dataingestion.ipynb` → … → `6-databaseparsing.ipynb`
2. `1-VectorEmbeddings/embedding.ipynb` → `openaiembeddings.ipynb`
3. `2-Vector Stores/1-chromadb.ipynb` (classic chain → LCEL → add docs → conversational memory → Groq)

Each section assumes the concepts (and often the mental model of `Document` → embed → retrieve → generate) from the previous one.
