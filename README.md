<h1 align="center">🎙️ OmniPod</h1>

<p align="center">
<strong>Chat with 936 podcast episodes. Every answer cites its source.</strong>
</p>

<p align="center">
Ask "What did Karpathy say about neural networks?" — get an answer with the exact transcript chunk it came from. No hallucinations. No guessing.
</p>

<p align="center">
<img src="https://img.shields.io/badge/python-3.13-blue" />
<img src="https://img.shields.io/badge/chunks-19,140-green" />
<img src="https://img.shields.io/badge/episodes-936-orange" />
<img src="https://img.shields.io/badge/latency-~2s_M1_Pro-purple" />
<img src="https://img.shields.io/badge/license-MIT-black" />
</p>

---

<!-- SI TENÉS UN GIF O SCREENSHOT, VA ACÁ. ES LO PRIMERO QUE LA GENTE NECESITA VER. -->
<!-- <p align="center"><img src="docs/demo.gif" width="700" /></p> -->

## Why OmniPod?

Most RAG chatbots hallucinate. You ask about a podcast, they invent quotes.

OmniPod doesn't. Every response is **grounded** — verified against the actual transcript before it reaches you. If the source doesn't support the answer, it says so.

**Three query types, one pipeline:**

| Type | Example | Strategy |
|---|---|---|
| Factual | "What did Huberman say about sleep?" | Retrieve → Generate → Verify |
| Synthetic | "Compare AI safety views across guests" | Map-Reduce → Deduplicate → Synthesize |
| Generative | "Write an essay on consciousness from these episodes" | Plan → Draft → Ground |

## How it works

```
You ask a question
        │
        ▼
  ┌─────────────┐
  │   Router     │  classify_intent() — routes to the right handler
  │  LRU cache   │  avoids re-embedding repeated queries
  │  Semaphore   │  caps concurrent LLM calls at 5
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  Retrieval   │  bge-small-en-v1.5 (384d) → Qdrant cosine
  │  19,140      │  chunks from 936 Lex Fridman episodes
  │  chunks      │  Guest filtering via known-guests index
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  Generate +  │  DeepSeek V4 Flash via OpenCode API
  │  Verify      │  verify_groundedness() — rejects ungrounded answers
  └──────┬──────┘
         │
         ▼
  Cited answer in Chainlit UI (localhost:8000)
```

## 60-second setup

```bash
git clone https://github.com/aranajhonny/omnipod && cd omnipod
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "OPENCODE_API_KEY=sk-your-key" > .env
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
python ingest.py --rebuild
chainlit run app.py
# → http://localhost:8000
```

## Numbers that matter

| Metric | Value |
|---|---|
| Episodes indexed | 936 Lex Fridman |
| Chunks | 19,140 (512 chars, 128 overlap) |
| Embedding dim | 384 (bge-small-en-v1.5, MPS GPU) |
| Query embedding | ~100ms |
| Vector search | ~50ms (cosine, 19K points) |
| Full answer | ~2s on M1 Pro |
| Full ingest | ~8 min |
| Codebase | 1,138 lines Python, 9 files |

## Transcript scraper included

No YouTube API key needed. Two sources:

- **lexfridman.com** — scrapes official transcript pages (requests + BeautifulSoup)
- **YouTube** — uses free proxy at `youtubetranscript.pro` for auto-captions

```bash
cd lex_podcast
pip install requests beautifulsoup4
python run.py pipeline  # scrapes all 936 episodes
```

Output lands in `data/transcripts/`.

## Example queries

```
"What did Andrej Karpathy say about neural networks?"
"Compare views on AI safety across all guests"
"Write a short essay on human consciousness based on these episodes"
"Summarize what Andrew Huberman says about sleep"
```

## Architecture decisions

- **Why `bge-small-en-v1.5`?** 384-dim embeddings are fast to search and good enough for conversational podcast text. Runs locally on MPS GPU.
- **Why Qdrant over Chroma?** Cosine search at 19K points in ~50ms. Filterable by guest metadata out of the box.
- **Why intent routing?** Factual, synthetic, and generative queries need fundamentally different retrieval and generation strategies. One prompt fits all fails at scale.
- **Why groundedness verification?** LLMs default to confident BS. `verify_groundedness()` forces the model to check its answer against the retrieved context before showing it to the user.

## License

MIT
