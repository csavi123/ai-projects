from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # VivaPrep/
NOTES_DIR = PROJECT_ROOT / "notes"  # chapter YAML files, copied in by hand
DATA_DIR = PROJECT_ROOT / "data"  # each pipeline stage writes its output here
MODEL_CACHE_DIR = PROJECT_ROOT / ".cache" / "fastembed"  # downloaded embedding model files

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # 384 numbers per text; runs on CPU via fastembed
TOP_K = 4  # how many chunks search returns
# Minimum best score to treat a question as "in my notes". Tuned by hand on the
# retriever's test questions (lowest in-notes 0.638, highest off-notes 0.602).
# Depends on the embedding model: re-tune if EMBEDDING_MODEL changes.
SIMILARITY_THRESHOLD = 0.62

LLM_MODEL = "claude-haiku-5-5"  # cheapest current Claude model; the only paid part of the app

load_dotenv(PROJECT_ROOT / ".env")  # secrets such as ANTHROPIC_API_KEY (needed from v1)


def notes_dir() -> Path:
    """Return the notes folder, or stop with a clear message if it's missing."""
    if not NOTES_DIR.is_dir():
        raise SystemExit(f"Notes folder not found: {NOTES_DIR}. Copy chapter files into VivaPrep/notes/.")
    return NOTES_DIR
