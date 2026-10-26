from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from vivaprep.config import DATA_DIR, PROJECT_ROOT
from vivaprep.loader import Chapter, Lesson, read_loaded

# CHUNK stage: structure-based chunking. Each lesson becomes one chunk per
# meaningful unit (core idea, each key point, deeper notes, own-words Q&A),
# so every chunk is one idea with a stable id like "ch01.l02.kp3".
# The quiz is never chunked: quiz questions are test questions, not notes.


@dataclass
class Chunk:
    id: str        # "ch01.l02.kp3"
    type: str      # core | key_point | deeper | own_words
    text: str      # cleaned note text, exactly what citations quote
    metadata: dict


def clean(text: str) -> str:
    """Remove inline markdown (bold, italic, code) so the text reads as plain sentences."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    return text.strip()


def chunk_lesson(chapter: Chapter, lesson: Lesson) -> list[Chunk]:
    base = {
        "chapter_id": chapter.id,
        "chapter_title": chapter.title,
        "lesson_id": lesson.id,
        "lesson_title": lesson.title,  # kept here, not in text; the embed step adds it
        "draft": lesson.draft,
        "source": lesson.source,
    }
    prefix = lesson.full_id
    chunks = [Chunk(f"{prefix}.core", "core", clean(lesson.core_idea), dict(base))]
    for number, point in enumerate(lesson.key_points, start=1):
        chunks.append(Chunk(f"{prefix}.kp{number}", "key_point", clean(point), {**base, "kp_number": number}))
    if lesson.deeper_notes:
        chunks.append(Chunk(f"{prefix}.deeper", "deeper", clean(lesson.deeper_notes), dict(base)))
    if lesson.own_words_q:
        text = f"Q: {clean(lesson.own_words_q)}\nA: {clean(lesson.own_words_a or '')}"
        chunks.append(Chunk(f"{prefix}.ownwords", "own_words", text, dict(base)))
    return chunks


def chunk_chapter(chapter: Chapter) -> list[Chunk]:
    return [chunk for lesson in chapter.lessons for chunk in chunk_lesson(chapter, lesson)]


def save_chunks(chunks_by_chapter: dict[str, list[Chunk]]) -> list[Path]:
    """Write data/<chapter_id>_chunks.json: the CHUNK stage's output."""
    DATA_DIR.mkdir(exist_ok=True)
    for old in DATA_DIR.glob("*_chunks.json"):  # so data/ never keeps a deleted chapter
        old.unlink()
    paths = []
    for chapter_id, chunks in chunks_by_chapter.items():
        out = DATA_DIR / f"{chapter_id}_chunks.json"
        records = [asdict(chunk) for chunk in chunks]
        out.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        paths.append(out)
    return paths


def read_chunks() -> list[Chunk]:
    """Read every chapter's chunks back (used by EMBED)."""
    paths = sorted(DATA_DIR.glob("*_chunks.json"))
    if not paths:
        raise SystemExit("No chunks in data/. Run: python -m vivaprep.chunker")
    return [Chunk(**record) for path in paths for record in json.loads(path.read_text(encoding="utf-8"))]


if __name__ == "__main__":  # stage 2: python -m vivaprep.chunker
    chunks_by_chapter = {chapter.id: chunk_chapter(chapter) for chapter in read_loaded()}
    for (chapter_id, chunks), out in zip(chunks_by_chapter.items(), save_chunks(chunks_by_chapter)):
        print(f"wrote {out.relative_to(PROJECT_ROOT)}  ({len(chunks)} chunks)")
