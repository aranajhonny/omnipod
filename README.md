# 🎙️ **OmniPod** — Chat with Podcast Transcripts

> Turn any podcast corpus into a conversational AI. Ask questions, compare guests, generate essays. Minimizes hallucinations via source grounding — every answer cites transcripts.

[![Python](https://img.shields.io/badge/python-3.13-blue)]()
[![Chainlit](https://img.shields.io/badge/Chainlit-2.x-green)]()
[![Qdrant](https://img.shields.io/badge/Qdrant-vector+DB-red)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-purple)]()

```
╔══════════════════════════════════════════════════════════════════╗
║                        O M N I P O D                            ║
║            Conversational RAG for Podcast Transcripts           ║
╚══════════════════════════════════════════════════════════════════╝

      👤 "What did Karpathy say about neural networks?"
                         │
                         ▼
╔══════════════════════════════════════════════════════════════════╗
║                    C H A I N L I T   U I                        ║
║  ChatGPT-style interface · WebSockets · Source cards            ║
╚══════════════════════════════════════════════════════════════════╝
                         │
                         ▼
╔══════════════════════════════════════════════════════════════════╗
║               R O U T E R   +   H A N D L E R S                 ║
╠══════════════════════════════════════════════════════════════════╣
║  classify_intent() ──┬── answer_factual()   RAG (retrieve→ans)  ║
║                      ├── answer_synthetic() Map-Reduce + dedup  ║
║                      └── answer_generative() Book planner→writer║
╚══════════════════════════════════════════════════════════════════╝
                         │
                         ▼
╔══════════════════════════════════════════════════════════════════╗
║         R E T R I E V A L   (sentence-transformers MPS GPU)     ║
╠══════════════════════════════════════════════════════════════════╣
║  Query → 384d vector → Qdrant cosine search → Top-5 chunks     ║
║  139,168 indexed chunks from transcripts                    ║
╚══════════════════════════════════════════════════════════════════╝
                         │
                         ▼
╔══════════════════════════════════════════════════════════════════╗
║             DeepSeek V4 Flash  (OpenCode API)                    ║
║  Anti-hallucination system prompt · Cites every source          ║
╚══════════════════════════════════════════════════════════════════╝
```

## Quick Start

```bash
# Prerequisites: Python 3.13+, Docker, OpenCode API key

# Setup
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "OPENCODE_API_KEY=sk-your-key" > .env

# Start Qdrant
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant

# Ingest transcripts (place .txt files in data/transcripts/)
python ingest.py --rebuild

# Launch
chainlit run app.py
# → http://localhost:8000
```

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| UI | Chainlit | ChatGPT-like, native WebSockets |
| LLM | DeepSeek V4 Flash | Fast, cheap, 128k context |
| Agent | Pure Python async | classify_intent() + handlers |
| Vector DB | Qdrant (Docker) | Cosine similarity, payload filters |
| Embeddings | sentence-transformers | bge-small-en-v1.5, runs on MPS GPU |
| Chunking | RecursiveCharacterTextSplitter | 512 chars, 128 overlap |

## Features

- **Semantic search** via sentence-transformers on Apple Silicon GPU
- **3 intent modes**: factual Q&A, multi-source synthesis, book/essay generation
- **Source-grounded answers**: every claim verified against transcripts via LLM fact-checking pass
- **Guest-aware retrieval**: detects guest names in queries, matches against indexed guest list
- **Source citations**: every answer links back to guest + title + text snippet
- **Any podcast**: bring your own .txt transcripts, ingest in one command

## Example Queries

```
"What did Andrej Karpathy say about neural networks?"
"Compare views on AI safety across all guests"
"Write a short essay on human consciousness"
"Summarize what Andrew Huberman says about sleep"
```

## Project Structure

```
├── app.py              # Chainlit UI
├── ingest.py           # Chunk → embed → upload pipeline
├── core/
│   ├── agent.py        # Intent router + RAG/synthesis/book handlers
│   ├── config.py       # Env vars & constants
│   ├── llm.py          # DeepSeek client + system prompt
│   ├── parser.py       # YouTube filename parser
│   └── vectorstore.py  # Qdrant + sentence-transformers
├── data/transcripts/   # Your podcast .txt files
├── requirements.txt
└── .env
```

## Performance (Apple M1 Pro)

- **Ingestion**: 139K chunks in ~8 min (sentence-transformers MPS)
- **Query**: ~100ms per search (384d cosine in Qdrant)
- **Response**: ~2-5s per question (DeepSeek V4 Flash API)

---

<p align="center">
  <b>MIT Licensed · 2026</b><br>
  <sub>Built for the love of podcasts and knowledge</sub>
</p>
