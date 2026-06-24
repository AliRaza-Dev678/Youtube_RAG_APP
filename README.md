# 🎬 YouTube RAG Chatbot

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/LangChain-RAG%20Pipeline-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white"/>
  <img src="https://img.shields.io/badge/Groq-LLaMA%203.1-F55036?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/PostgreSQL-PGVector-336791?style=for-the-badge&logo=postgresql&logoColor=white"/>
  <img src="https://img.shields.io/badge/HuggingFace-Embeddings-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/>
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/yt--dlp-Transcript-FF0000?style=for-the-badge&logo=youtube&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

<p align="center">
  A production-grade <strong>Retrieval-Augmented Generation (RAG)</strong> application that lets you <strong>chat with any YouTube video</strong>. Paste a URL, the app fetches the transcript, indexes it into a <strong>PostgreSQL vector database</strong>, and lets you ask questions answered by <strong>LLaMA 3.1 via Groq</strong> — all inside a sleek, custom-styled <strong>Streamlit</strong> interface.
</p>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [How It Works](#-how-it-works)
- [RAG Architecture](#-rag-architecture)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Key Features](#-key-features)
- [Getting Started](#-getting-started)
- [Environment Variables](#-environment-variables)
- [How to Run](#-how-to-run)
- [Deploying to Streamlit Cloud](#-deploying-to-streamlit-cloud)
- [Key Concepts Covered](#-key-concepts-covered)
- [Real-World Applications](#-real-world-applications)
- [Author](#-author)

---

## 🧠 Overview

**YouTube RAG Chatbot** is an end-to-end **Generative AI application** built around the **Retrieval-Augmented Generation (RAG)** paradigm. It combines:

- **yt-dlp** to extract transcripts from any YouTube video without API keys or IP bans
- **LangChain** to orchestrate the full RAG pipeline
- **HuggingFace sentence-transformers** to embed transcript chunks into dense vectors
- **PostgreSQL + PGVector** as a persistent vector database for semantic search
- **Groq (LLaMA 3.1-8b-instant)** as the ultra-fast LLM for answer generation
- **Streamlit** for a polished, production-ready chat interface with custom CSS theming

This project represents a leap beyond classical ML — it is a **real Generative AI application** that uses the latest LLM tooling stack used in industry.

---

## 🚀 Live Demo

```
streamlit run app.py
```

Open `http://localhost:8501` in your browser, paste any YouTube URL, and start chatting.

---

## ⚙️ How It Works

```
User pastes YouTube URL
         │
         ▼
① Extract Video ID        →  Regex-parse URL or accept raw 11-char ID
         │
         ▼
② Fetch Transcript         →  yt-dlp downloads auto-captions in JSON3 format
                               (no YouTube API key needed, no IP bans)
         │
         ▼
③ Chunk Transcript         →  RecursiveCharacterTextSplitter
                               chunk_size=1000, chunk_overlap=200
         │
         ▼
④ Embed Chunks             →  HuggingFace all-MiniLM-L6-v2
                               384-dimensional sentence embeddings
         │
         ▼
⑤ Store in PGVector        →  PostgreSQL + pgvector extension
                               Persistent vector store per video
         │
         ▼
⑥ User asks a question
         │
         ▼
⑦ MMR Retrieval            →  Maximal Marginal Relevance (k=4, fetch_k=20)
                               Diverse, relevant context chunks retrieved
         │
         ▼
⑧ Prompt Assembly          →  LangChain PromptTemplate injects context + question
         │
         ▼
⑨ LLM Generation           →  Groq (LLaMA 3.1-8b-instant, temp=0.2, max_tokens=512)
                               Answer grounded ONLY in transcript context
         │
         ▼
⑩ Display in Chat UI       →  Streamlit chat bubbles with custom CSS styling
```

---

## 🏗️ RAG Architecture

### What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that augments an LLM's knowledge with a custom knowledge base by:

1. **Indexing** — splitting documents into chunks and storing them as vector embeddings in a database
2. **Retrieving** — when a question is asked, finding the most semantically similar chunks via vector search
3. **Generating** — feeding the retrieved chunks as context to the LLM, which generates a grounded answer

RAG solves the hallucination problem — the LLM is instructed to answer **only from the provided context**, never from its parametric memory.

### System Architecture

```
┌─────────────────────────────────────────────────────┐
│                  INDEXING PHASE                     │
│                                                     │
│  YouTube Video                                      │
│       │                                             │
│       ▼                                             │
│  yt-dlp Transcript Fetch                            │
│       │                                             │
│       ▼                                             │
│  RecursiveCharacterTextSplitter (1000/200)          │
│       │                                             │
│       ▼                                             │
│  HuggingFace Embeddings (all-MiniLM-L6-v2)         │
│       │                                             │
│       ▼                                             │
│  PGVector (PostgreSQL vector store)                 │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                  RETRIEVAL PHASE                    │
│                                                     │
│  User Question                                      │
│       │                                             │
│       ▼                                             │
│  Embed Question (same MiniLM model)                 │
│       │                                             │
│       ▼                                             │
│  MMR Retrieval from PGVector (k=4, fetch_k=20)     │
│       │                                             │
│       ▼                                             │
│  Top-4 Relevant Transcript Chunks                   │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                 GENERATION PHASE                    │
│                                                     │
│  PromptTemplate (context + question)                │
│       │                                             │
│       ▼                                             │
│  Groq → LLaMA 3.1-8b-instant                       │
│       │                                             │
│       ▼                                             │
│  StrOutputParser → Streamlit Chat UI                │
└─────────────────────────────────────────────────────┘
```

### LangChain Runnable Chain

```python
chain = (
    RunnableParallel({
        "context":  retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    })
    | PromptTemplate(...)
    | ChatGroq(model="llama-3.1-8b-instant")
    | StrOutputParser()
)
```

---

## 📁 Project Structure

```
Youtube_RAG_APP/
│
├── app.py                  ← Full Streamlit application (622 lines)
│   ├── Credential loading  (.env + st.secrets)
│   ├── Custom CSS theming  (dark UI, gradient brand, chat bubbles)
│   ├── yt-dlp transcript   (fetch_transcript function)
│   ├── RAG chain builder   (build_chain — cached by video_id)
│   ├── Sidebar UI          (URL input, quick questions, clear chat)
│   └── Chat UI             (message history, input row, send handler)
│
├── requirements.txt        ← All pinned dependencies (119 packages)
├── .gitignore              ← Excludes .env, __pycache__, etc.
└── README.md
```

---

## 🛠️ Tech Stack

### Core AI / LLM Stack

| Component              | Technology                                      | Role                                            |
|------------------------|-------------------------------------------------|-------------------------------------------------|
| **LLM**                | LLaMA 3.1-8b-instant via Groq API              | Ultra-fast answer generation                    |
| **Embeddings**         | `sentence-transformers/all-MiniLM-L6-v2` (HF)  | 384-dim dense vector embeddings                 |
| **Vector Database**    | PostgreSQL + pgvector extension                 | Persistent semantic vector search               |
| **RAG Framework**      | LangChain (Core, Community, HuggingFace, Groq) | RAG pipeline orchestration                      |
| **Retrieval Strategy** | MMR (Maximal Marginal Relevance)                | Diverse, non-redundant context retrieval        |
| **Text Splitter**      | `RecursiveCharacterTextSplitter`                | Chunk transcript for embedding (1000/200)       |

### Infrastructure Stack

| Component              | Technology              | Role                                            |
|------------------------|-------------------------|-------------------------------------------------|
| **Web App**            | Streamlit 1.58          | Interactive chat UI with custom CSS             |
| **Transcript Fetcher** | yt-dlp 2026.6.9         | YouTube auto-caption extraction (no API key)    |
| **Database ORM**       | SQLAlchemy + psycopg2   | PostgreSQL connection management                |
| **Secret Management**  | python-dotenv + st.secrets | `.env` locally, Streamlit Cloud in production |
| **Caching**            | `@st.cache_resource`    | Cache embeddings, LLM, and chain per video      |

---

## ✨ Key Features

- 🎬 **Any YouTube Video** — Paste any public YouTube URL or 11-character video ID
- ⚡ **Groq-Powered Speed** — LLaMA 3.1 via Groq delivers near-instant LLM responses
- 🔍 **MMR Retrieval** — Maximal Marginal Relevance ensures diverse, non-redundant context
- 🧠 **Grounded Answers** — LLM strictly answers from transcript context; never hallucinates
- 💾 **Persistent Vector Store** — PGVector in PostgreSQL persists embeddings across sessions
- 🔄 **Smart Caching** — `@st.cache_resource` caches the model and chain per video ID
- 🎨 **Custom Dark UI** — 600+ lines of CSS with gradient branding, chat bubbles, sidebar
- ❓ **Quick Questions** — Pre-built sidebar buttons: Summarise, Main Topics, Key Takeaway
- 🔐 **Secure Secrets** — `.env` locally, `st.secrets` on Streamlit Cloud
- ☁️ **Cloud-Ready** — Deployable to Streamlit Cloud with zero code changes

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- PostgreSQL with `pgvector` extension installed
- A [Groq API Key](https://console.groq.com/) (free tier available)
- A [HuggingFace API Token](https://huggingface.co/settings/tokens) (free)

### 1. Clone the Repository

```bash
git clone https://github.com/AliRaza-Dev678/Youtube_RAG_APP.git
cd Youtube_RAG_APP
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up PostgreSQL with pgvector

```sql
-- In your PostgreSQL shell:
CREATE DATABASE youtube_rag;
CREATE EXTENSION IF NOT EXISTS vector;
```

---

## 🔐 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
DATABASE_URL=postgresql://username:password@localhost:5432/youtube_rag
HUGGINGFACEHUB_API_TOKEN=your_hf_token_here
```

| Variable                    | Where to Get It                                          |
|-----------------------------|----------------------------------------------------------|
| `GROQ_API_KEY`              | [console.groq.com](https://console.groq.com) — free tier|
| `DATABASE_URL`              | Your PostgreSQL connection string                        |
| `HUGGINGFACEHUB_API_TOKEN`  | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

---

## ▶️ How to Run

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

**Usage:**
1. Paste a YouTube URL in the sidebar (e.g. `https://youtube.com/watch?v=dQw4w9WgXcQ`)
2. Click **⚡ Load & Index Video** — transcript is fetched and indexed
3. Type any question in the chat input and click **Send →**
4. Use **Quick Questions** in the sidebar for instant prompts
5. Click **🗑 Clear conversation** to reset the chat

---

## ☁️ Deploying to Streamlit Cloud

1. Push your code to GitHub (already done ✅)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New App**
3. Select your repo, branch `main`, and `app.py`
4. Under **Advanced settings → Secrets**, add:

```toml
GROQ_API_KEY = "your_groq_api_key"
DATABASE_URL = "postgresql://user:pass@host:5432/dbname"
HUGGINGFACEHUB_API_TOKEN = "your_hf_token"
```

5. Click **Deploy** — the app reads from `st.secrets` automatically.

> 💡 Use [Neon](https://neon.tech), [Supabase](https://supabase.com), or [Railway](https://railway.app) for a free PostgreSQL + pgvector database hosted in the cloud.

---

## 📚 Key Concepts Covered

- ✅ **RAG (Retrieval-Augmented Generation)** — The core GenAI design pattern for grounded Q&A
- ✅ **Vector Embeddings** — Converting text to 384-dimensional dense vectors with MiniLM
- ✅ **Semantic Search** — Finding contextually relevant chunks via cosine similarity
- ✅ **MMR Retrieval** — Maximal Marginal Relevance for diverse, non-repetitive context
- ✅ **PGVector** — Using PostgreSQL as a production-grade vector database
- ✅ **LangChain Runnables** — `RunnableParallel`, `RunnablePassthrough`, `RunnableLambda`
- ✅ **yt-dlp Transcript Extraction** — Fetching auto-captions in JSON3 format via yt-dlp
- ✅ **Text Chunking** — `RecursiveCharacterTextSplitter` with overlap for context continuity
- ✅ **Groq API** — Calling LLaMA 3.1-8b-instant for ultra-low latency inference
- ✅ **Streamlit Secrets** — Dual credential loading via `.env` + `st.secrets`
- ✅ **Resource Caching** — `@st.cache_resource` for embeddings, LLM, and chains
- ✅ **Custom Streamlit CSS** — Full dark theme with gradient branding and chat bubbles
- ✅ **Production Deployment** — Streamlit Cloud deployment with secret management

---

## 🌍 Real-World Applications

This YouTube RAG pattern directly powers real products:

| Domain                    | Use Case                                                                  |
|---------------------------|---------------------------------------------------------------------------|
| 🎓 **Education**          | Chat with lecture recordings, online courses, and educational videos      |
| 📰 **Media & Journalism** | Query interview recordings and press conference transcripts               |
| 🏢 **Enterprise**         | Internal knowledge base from recorded meetings and webinars               |
| 🎙️ **Podcasts**          | Search and Q&A over long-form podcast episodes                            |
| 🔬 **Research**           | Extract insights from conference talks and academic presentations         |
| 🤖 **AI Assistants**      | Domain-specific chatbots grounded in curated video content                |
| 📱 **EdTech Startups**    | "Ask your textbook" style products built on video content                 |

---

## 👨‍💻 Author

**Ali Raza**

- GitHub: [@AliRaza-Dev678](https://github.com/AliRaza-Dev678)

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute it.

---

<p align="center">
  ⭐ If you found this project helpful, please give it a star! ⭐
</p>
