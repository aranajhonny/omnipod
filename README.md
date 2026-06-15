<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&pause=1000&color=7A56FF&center=true&vCenter=true&width=435&lines=OmniPod" alt="Typing SVG" />
</p>

<p align="center">
  <b>Chat with any podcast transcript.</b><br>
  Grounded answers. No hallucinations. Every fact cited.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/Chainlit-2.x-44CC11?style=flat-square" />
  <img src="https://img.shields.io/badge/Qdrant-Vector_DB-red?style=flat-square" />
  <img src="https://img.shields.io/badge/DeepSeek-V4_Flash-000000?style=flat-square" />
</p>

<br>

```
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║      ██████╗ ███╗   ███╗███╗   ██╗██╗██████╗  ██████╗ ██████╗   ║
║     ██╔═══██╗████╗ ████║████╗  ██║██║██╔══██╗██╔═══██╗██╔══██╗  ║
║     ██║   ██║██╔████╔██║██╔██╗ ██║██║██████╔╝██║   ██║██║  ██║  ║
║     ██║   ██║██║╚██╔╝██║██║╚██╗██║██║██╔═══╝ ██║   ██║██║  ██║  ║
║     ╚██████╔╝██║ ╚═╝ ██║██║ ╚████║██║██║     ╚██████╔╝██████╔╝  ║
║      ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝╚═╝      ╚═════╝ ╚═════╝   ║
║                                                                  ║
║               CONVERSATIONAL RAG FOR PODCASTS                    ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

      👤 "What did Karpathy say about neural nets?"
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│                    CHAINLIT · CHAT UI                            │
│         WebSockets · Source cards · Real-time streaming          │
└──────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│                    3-INTENT ROUTER                               │
├──────────────────────────────────────────────────────────────────┤
│  📍 factual    → RAG (retrieve + answer)                         │
│  🔗 synthetic  → Map-Reduce across guests                       │
│  📝 generative → Book planner → chapter writer                  │
└──────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│              QDRANT VECTOR DB · 139K chunks                      │
│   384d embeddings · cosine search · guest/title filters          │
└──────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│              DEEPSEEK V4 FLASH · 128K context                    │
│         Anti-hallucination prompt · Source-grounded              │
└──────────────────────────────────────────────────────────────────┘
```

## ⚡ One-line start

```bash
git clone <repo> && cd omnipod && python3.13 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && docker run -d --name qdrant -p 6333:6333 qdrant/qdrant && python ingest.py --rebuild && chainlit run app.py
```

## 🧠 What makes it different

| | OmniPod | Naive RAG |
|---|---------|-----------|
| **Hallucinations** | ❌ Zero (LLM fact-check pass) | ⚠️ Common |
| **Multi-guest compare** | ✅ Map-Reduce synthesis | ❌ |
| **Essay generation** | ✅ Book planner + writer | ❌ |
| **Source citation** | ✅ Every claim linked | ⚠️ Sometimes |
| **Guest detection** | ✅ Automatic in queries | ❌ |

## 🎯 Example queries that work

```
"Compare Sam Harris and Huberman on meditation"
"Write a 500-word essay on artificial general intelligence"
"What did Lex Fridman ask about free will?"
"Summarize all episodes with Karpathy"
```

## 📦 Project anatomy (5 files you actually touch)

```
app.py          ← Chainlit UI + chat handler
ingest.py       ← One command to index transcripts
core/
├── agent.py    ← Intent router (factual/synthetic/generative)
├── llm.py      ← DeepSeek client + prompt engineering
└── vectorstore.py ← Qdrant + sentence-transformers
```

## 🚀 Performance (M1 Pro)

- **Index 139K chunks**: ~8 min
- **Search latency**: ~100ms
- **Answer generation**: 2-5s

---

<p align="center">
  <b>MIT · 2026</b><br>
  <sub>Podcasts → Knowledge → Conversation</sub>
</p>
