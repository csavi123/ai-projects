# VivaPrep

An interview coach built on RAG (retrieval-augmented generation) that grades you against your own notes. It answers questions using only your notes, with citations. It asks you interview questions written from a single lesson, and refuses when a topic isn't in your notes.

- **Ask mode (v1, RAG):** embeds your question, retrieves the closest note chunks, and has Claude answer only from them. Code refuses before calling Claude if nothing is close enough, and shows an answer only if every citation points to a chunk that was actually sent.
- **Question mode (v2, no retrieval):** the lesson is already known, so the whole lesson is sent instead of searched. Claude writes interview questions (definition, why, scenario), each tagged with the key points it tests, and those tags are checked in code.
- **Coming next:** v3 Feedback (LLM as judge, scored in code), v3.5 judge eval, v4 weak-spot tracking.

How it works: [ARCHITECTURE.md](ARCHITECTURE.md). Full design and roadmap: [PLAN.md](PLAN.md).

## Stack

Python 3.12 · `fastembed` (`BAAI/bge-small-en-v1.5`, local CPU) · numpy cosine search · Claude (`claude-haiku-5-5`) via the `anthropic` SDK · FastAPI · Streamlit

Claude is the only paid part. Embeddings and search run locally for free.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env          # then set ANTHROPIC_API_KEY
```

**Notes:** put chapter YAML files in `notes/`. This folder is gitignored, so the notes aren't in the repo. A chapter file has a header (`id`, `number`, `title`) and a `lessons:` list.

## Build the index

Each stage writes a file in `data/` that the next stage reads. After you edit notes, rerun them in order:

```bash
python -m vivaprep.loader      # notes/*.yaml      → data/<id>_loaded.json
python -m vivaprep.chunker     # loaded JSON       → data/<id>_chunks.json
python -m vivaprep.embedder    # chunks            → data/embeddings.npy + .json
```

The first `embedder` run downloads the embedding model (about 64 MB) into `.cache/`.

## Use it

**Command line**

```bash
python -m vivaprep.ask "why is LLM output non-deterministic?"   # add --debug to see scores and citations
python -m vivaprep.questions                                     # list chapters and lessons
python -m vivaprep.questions ch01                                # one question per lesson
python -m vivaprep.questions ch01.l03                            # three questions for one lesson
python -m vivaprep.retriever "your question"                     # top chunks only, no Claude call
python -m vivaprep.eval_retrieval                                # free retrieval eval
```

**Web app**

```bash
uvicorn vivaprep.api:app --port 8001    # backend; API docs at http://127.0.0.1:8001/docs
streamlit run app.py                    # frontend (Ask and Quiz tabs, voice input in Chrome/Edge)
```

The frontend reads `VIVAPREP_API_URL` (default `http://127.0.0.1:8001`).
