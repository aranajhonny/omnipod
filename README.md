<img width="790" height="459" alt="Captura de pantalla 2026-06-14 a la(s) 11 32 06 p m" src="https://github.com/user-attachments/assets/57b7b344-2d1e-4b55-8464-95a43b6cbc24" />
# 🎙️ **OmniPod** — Chat with Podcast Transcripts

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
║  19,140 indexed chunks · Lex Fridman transcripts            ║
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
| Transcripts | (Lex Fridman Podcast, scraped via `transcript_fetcher.py`) |
| Chunks indexed | 19,140 (512 chars, 128 overlap) |
| Vector dimensions | 384 (bge-small-en-v1.5, MPS GPU) |
| Scraping | lexfridman.com site transcripts + YouTube API (`core/transcript_fetcher.py`) |
| Source files | 1,138 lines Python across 9 files |

## Scraper

`core/transcript_fetcher.py` downloads transcripts from two sources:

1. **lexfridman.com** — scrapes official transcript pages via `requests` + `BeautifulSoup`. These are human-written transcripts for ~114 episodes.

2. **YouTube API** (free, no key needed) — for the ~382 episodes without site transcripts, it uses `youtubetranscript.pro`, a free proxy that extracts YouTube captions:
   - `POST /api/youtube/metadata` — registers the video ID with the session
   - `GET /api/youtube/transcript` — returns the auto-generated captions as JSON
   
   This avoids needing a YouTube Data API key or paid transcription service.

Both outputs are `.txt` files. Then `python ingest.py --rebuild` chunks and indexes them into Qdrant.

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

<img width="790" height="459" alt="Captura de pantalla 2026-06-14 a la(s) 11 32 06 p m" src="https://github.com/user-attachments/assets/04303c01-0dbe-4916-9374-8f8e58a780b8" />

---

<p align="center">
  <b>MIT Licensed · 2026</b><br>
  <sub>Built for the love of podcasts and knowledge</sub>
</p>
