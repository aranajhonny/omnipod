# 🎙️ **OmniPod** — Chat with 701 Podcast Transcripts

> Turn any podcast corpus into a conversational AI. Ask questions, compare guests, generate essays. Minimizes hallucinations via source grounding — every answer cites transcripts.

[![Python](https://img.shields.io/badge/python-3.13-blue)]()
[![Chainlit](https://img.shields.io/badge/Chainlit-2.x-green)]()
[![Qdrant](https://img.shields.io/badge/Qdrant-384d+cosine-red)]()
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
║  port 8000 · WebSockets · Source cards                          ║
╚══════════════════════════════════════════════════════════════════╝
                         │
                         ▼
╔══════════════════════════════════════════════════════════════════╗
║               R O U T E R   +   H A N D L E R S                 ║
╠══════════════════════════════════════════════════════════════════╣
║  classify_intent() ──┬── answer_factual()   RAG + verify        ║
║  lru_cache(128)      ├── answer_synthetic() Map-Reduce + dedup  ║
║  Semaphore(5)        └── answer_generative() Book planner→writer║
╚══════════════════════════════════════════════════════════════════╝
                         │
                         ▼
╔══════════════════════════════════════════════════════════════════╗
║         R E T R I E V A L   (sentence-transformers MPS GPU)     ║
╠══════════════════════════════════════════════════════════════════╣
║  Query → bge-small-en-v1.5 384d → Qdrant cosine search          ║
║  19,140 indexed chunks · 701 Lex Fridman transcripts            ║
║  Guest filter via known-guests list from Qdrant                  ║
╚══════════════════════════════════════════════════════════════════╝
                         │
                         ▼
╔══════════════════════════════════════════════════════════════════╗
║             DeepSeek V4 Flash  (OpenCode API)                    ║
║  verify_groundedness() · @retry() · Source citations            ║
╚══════════════════════════════════════════════════════════════════╝
```

## Quick Start

```bash
# Prerequisites: Python 3.13+, Docker, OpenCode API key

git clone https://github.com/aranajhonny/omnipod
cd humans
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

| Component | Choice | Details |
|-----------|--------|---------|
| UI | Chainlit 2.11 | WebSockets, source cards, port 8000 |
| LLM | DeepSeek V4 Flash | OpenCode API, 128k context, `@retry(3)` |
| Agent | Pure Python async | `classify_intent()` → handler, `Semaphore(5)`, `lru_cache(128)` |
| Vector DB | Qdrant 1.18 | 384d cosine, Docker, port 6333 |
| Embeddings | sentence-transformers 3.x | `bge-small-en-v1.5`, MPS GPU, 384d |
| Chunking | RecursiveCharacterTextSplitter | 512 chars, 128 overlap |
| Verification | `verify_groundedness()` | Second LLM pass, claim extraction |
| Guest detection | Known-guests list | Scanned from Qdrant payload, regex fallback |

## Features

- **Semantic search** via sentence-transformers on Apple Silicon MPS GPU
- **3 intent modes**: factual Q&A, multi-source synthesis, book/essay generation
- **Source-grounded answers**: every claim verified against transcripts via LLM fact-checking pass
- **Guest-aware retrieval**: detects guest names from Qdrant's indexed guest list, filters results
- **Rate-limited**: `asyncio.Semaphore(5)` prevents API throttling
- **Cached routing**: `lru_cache` avoids re-classifying repeated queries
- **Resilient**: `@retry()` decorator on API calls (3 attempts, exponential backoff)
- **Any podcast**: bring your own .txt transcripts, run `python ingest.py --rebuild`

## Project Structure

```
├── app.py              # Chainlit UI  (82 lines)
├── ingest.py           # Chunk → embed → upload pipeline  (123 lines)
├── core/
│   ├── agent.py        # Intent router + RAG/synthesis/book  (508 lines)
│   ├── config.py       # Env vars & constants  (33 lines)
│   ├── llm.py          # DeepSeek client + verify_groundedness  (109 lines)
│   ├── parser.py       # YouTube filename parser  (116 lines)
│   └── vectorstore.py  # Qdrant + sentence-transformers  (97 lines)
├── tests/
│   └── test_basic.py   # Unit tests for parser & config  (43 lines)
├── data/transcripts/   # Your podcast .txt files
├── .env.example
├── requirements.txt
└── .gitignore
```

**Total: 1,138 lines of Python across 9 source files.**

## Data

| Metric | Value |
|--------|-------|
| Transcripts | 701 files (Lex Fridman Podcast) |
| Chunks indexed | 19,140 (512 chars, 128 overlap) |
| Vector dimensions | 384 (bge-small-en-v1.5) |
| Embedding backend | MPS GPU (Apple Silicon) |
| Embedding speed | ~1,100 chunks/s |

## Example Queries

```
"What did Andrej Karpathy say about neural networks?"
"Compare views on AI safety across all guests"
"Write a short essay on human consciousness"
"Summarize what Andrew Huberman says about sleep"
```

## Performance (Apple M1 Pro)

| Operation | Time |
|-----------|------|
| Query embedding | ~100ms (384d, MPS) |
| Qdrant search | ~50ms (cosine, 19K points) |
| LLM response | ~2-5s (DeepSeek V4 Flash) |
| Full ingest (139K chunks) | ~8 min (MPS) |

---

<p align="center">
  <b>MIT Licensed · 2026</b><br>
  <sub>Built for the love of podcasts and knowledge</sub>
</p>
