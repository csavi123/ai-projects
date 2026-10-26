# VivaPrep

An interview coach that grades you against your own notes. The full design is in `PLAN.md`; read it before working on anything.

## Notes source
- Folder: `VivaPrep/notes/` (gitignored). Chapter files are copied there **manually** from `/Users/cinnysavitri/AI-Accelerator/AI_Conceptual_Understanding/content`. The two projects are independent: neither is the master copy, and editing one never affects the other.
- Copied so far: ch01 only (7 concept lessons). ch05 and ch06 exist in AI-Accelerator and get copied when needed.
- Only use chapter files that exist. Identify each chapter by **its own file header** (`id`, `number`, `title`), found by scanning for YAML files with a `lessons:` key.
- Don't use `syllabus.yaml` and don't lay out the 17-chapter syllabus: chapters will be added later and their order or numbering may change.
- `ch05_triage_cases.yaml` is lab data, not lesson text. Skip it.
- In AI-Accelerator today: ch01 (How LLMs actually work), ch05 (Prompting & context engineering), ch06 (Structured output & tool calling). That's 18 concept lessons; skip `kind: lab` and `kind: prove_it` entries.

## Decisions made
- Name: VivaPrep. Project folder `VivaPrep/`; Python package `vivaprep/` (lowercase, PEP 8); CLI command `vivaprep`.
- Draft lessons (`draft: true`): include them, but label them `⚠ draft note` everywhere. A `--verified-only` flag excludes them.
- Embeddings: small local model (`sentence-transformers`).
- Vector store: numpy + cosine similarity by hand first → Chroma (embedded, local folder) → optionally Chroma server in a local Docker container → Pinecone only when scaling. All behind one `search(query, k)` function.
- Chunking: first a fixed-size chunking experiment on ch01's plain-text view (`experiments/`, not used by the app, prints chunks to inspect). Then structure-based chunking as the real chunker. Compare both on 5 ch01 quiz questions once embeddings exist.
- Notes format: keep the YAML. The loader also produces a plain-text view of a chapter for the experiment.
- Pipeline: each stage writes a file in `data/` and the next stage reads it (`notes/*.yaml` → `data/<id>_loaded.json` → `data/<id>_chunks.json` → embeddings). Rerun stages in order after editing notes.
- Config: `.env` read with `python-dotenv`.
- Interface: command line first; later FastAPI backend + Streamlit frontend.
- LLM: Claude via the `anthropic` SDK, cheapest model. Exact model ID chosen at v1.
- Cost: zero spend except Claude API calls (vector DB, embeddings, hosting all free).
- Deployment (later): FastAPI on Render + Streamlit Community Cloud, free tiers.
- Setup: plain `python3 -m venv .venv` on the system Python 3.9 (no `uv`). Code must stay 3.9-compatible (`from __future__ import annotations`).
- Packaging: `pyproject.toml`, installed with `pip install -e ".[dev]"`.

## How to work with me
- I will present this project in interviews and need to understand every piece.
- **Explain what each file is and why it's needed before writing it.** Work a few files at a time, never a big batch.
- Build one step at a time. After each step, explain it and let me run it before moving on.
- Prefer the simplest standard approach.

## Progress
- [x] PLAN.md written
- [x] Project named, folder renamed to `VivaPrep`
- [x] v0 step 1: setup (`pyproject.toml`, `.gitignore`, `.env.example`, `vivaprep/__init__.py`, `.venv`, editable install)
- **Current scope: chapter 1 (ch01) only.** Build each step on ch01 first.
- [x] v0 step 2: `vivaprep/config.py` + `vivaprep/loader.py` (header-based chapter discovery → Chapter/Lesson objects; 3 rules: skip entries with `kind`, stop on broken files, skip `status: coming`). Stage output: `python -m vivaprep.loader` → `data/ch01_loaded.json`; `read_loaded()` reads it back
- [x] v0 step 2c: plain-text view, `loader.chapter_text()` (no quiz); `python experiments/plain_text.py` → `data/ch01_text.txt` (10,539 chars)
- [ ] **v0 step 2d (next):** `experiments/fixed_size_chunking.py` (cut the plain text every N chars with overlap, print chunks)
- [ ] v0 step 3: `vivaprep/chunker.py` (chunk ids like `ch01.l03.kp2`)
- [ ] v0 step 4: `vivaprep/cli.py` (`vivaprep lessons`, `vivaprep show ch01.l03`), run on the real notes
- [ ] v0 step 5: tests, then a git commit if I ask for one
- [ ] v1 Ask · v2 Questions · v3 Feedback (+ v3.5 judge eval) · v4 Weak spots

Update this Progress list as steps finish.
