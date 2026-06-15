# 🎙️ **OmniPod** — Chat with 701 Podcast Transcripts

> Turn any podcast corpus into a conversational AI. Minimizes hallucinations via source grounding — every answer cites transcripts.

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

```bash
git clone https://github.com/aranajhonny/omnipod && cd humans
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "OPENCODE_API_KEY=sk-your-key" > .env
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
python ingest.py --rebuild
chainlit run app.py
# → http://localhost:8000
```

## Data

| Metric | Value |
|--------|-------|
| Transcripts | 701 (Lex Fridman Podcast, scraped via `transcript_fetcher.py`) |
| Chunks indexed | 19,140 (512 chars, 128 overlap) |
| Vector dimensions | 384 (bge-small-en-v1.5, MPS GPU) |
| Scraping | lexfridman.com site transcripts + YouTube API (`core/transcript_fetcher.py`) |
| Source files | 1,138 lines Python across 9 files |

## Scraper

`core/transcript_fetcher.py` downloads transcripts from two sources:

1. **lexfridman.com** — scrapes official transcript pages via requests + BeautifulSoup. Covers ~114 episodes.
2. **YouTube API** — for episodes without site transcripts, calls `youtubetranscript.pro` API to fetch auto-generated captions. Covers ~382 more episodes.

Both outputs are `.txt` files in `data/transcripts/`. Then `python ingest.py --rebuild` chunks and indexes them into Qdrant.

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
| Query embedding | ~100ms (MPS) |
| Qdrant search | ~50ms (cosine, 19K points) |
| LLM response | ~2-5s (DeepSeek V4 Flash) |
| Full ingest (139K chunks) | ~8 min |

---

<p align="center">
  <b>MIT Licensed · 2026</b><br>
  <sub>Built for the love of podcasts and knowledge</sub>
</p>
