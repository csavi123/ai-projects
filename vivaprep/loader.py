from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

#This file is the LOAD stage of your RAG pipeline. 
# It:Finds every .yaml file in your notes folder.
#Reads each one safely into Python data with yaml.safe_load.
#Cleans it up into Chapter and Lesson objects, keeping only useful fields (core idea, key points, deeper notes, own-words Q&A, quiz) and dropping app-only stuff like diagrams and demo IDs.
#Skips entries that aren’t real lessons (prove_it, coming lessons) and stops with an error if a lesson is broken or a chapter ID is duplicated.
#Saves each chapter as data/<chapter>_loaded.json, so the next stage (the chunker) can read it back with read_loaded().
#Bonus: chapter_text() glues a chapter into one plain document, minus the quiz, for your fixed-size chunking experiment.

from vivaprep.config import DATA_DIR, PROJECT_ROOT, notes_dir


@dataclass
class Lesson:
    chapter_id: str          # "ch01", from the file header
    id: str                  # "l01"
    number: str              # "01"
    title: str
    draft: bool
    source: str
    one_liner: str
    core_idea: str
    key_points: list[str]    # raw text, markdown kept (the chunker cleans it)
    deeper_notes: str | None
    own_words_q: str | None
    own_words_a: str | None
    quiz: list[dict]

    @property
    def full_id(self) -> str:
        return f"{self.chapter_id}.{self.id}"  # "ch01.l01"


@dataclass
class Chapter:
    id: str
    number: int
    title: str
    file: Path
    lessons: list[Lesson]


def load_chapters() -> list[Chapter]:
    """Load every chapter file in the notes folder, sorted by header number."""
    chapters = []
    seen = {}  # chapter id -> file that claimed it
    for path in sorted(notes_dir().glob("*.yaml")):
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)  # safe_load builds plain data only, never runs code
        if not isinstance(data, dict) or "lessons" not in data:
            continue  # not a chapter file (e.g. lab data)
        chapter = _parse_chapter(path, data)
        if chapter.id in seen:
            _fail(path, f"chapter id '{chapter.id}' is already used by {seen[chapter.id].name}")
        seen[chapter.id] = path
        chapters.append(chapter)
    return sorted(chapters, key=lambda ch: ch.number)


def _parse_chapter(path: Path, data: dict) -> Chapter:
    for field in ("id", "number", "title"):
        if field not in data:
            _fail(path, f"header is missing '{field}'")
    lessons = []
    for entry in data["lessons"]:
        if "kind" in entry:  # rule 1: prove_it, lab, ... are not concept lessons
            continue
        if entry.get("status") == "coming":  # rule 3: no lesson text yet
            continue
        lessons.append(_parse_lesson(path, data["id"], entry))
    return Chapter(
        id=data["id"],
        number=int(data["number"]),
        title=data["title"],
        file=path,
        lessons=lessons,
    )


def _parse_lesson(path: Path, chapter_id: str, entry: dict) -> Lesson:
    for field in ("id", "number", "title", "core_idea", "key_points"):
        if not entry.get(field):  # rule 2: stop on broken notes instead of guessing
            _fail(path, f"lesson '{entry.get('id', '?')}' is missing '{field}'")
    deeper = entry.get("deeper") or {}
    own_words = deeper.get("own_words") or {}
    return Lesson(
        chapter_id=chapter_id,
        id=entry["id"],
        number=str(entry["number"]),
        title=entry["title"],
        draft=bool(entry.get("draft", False)),
        source=entry.get("source", ""),
        one_liner=entry.get("one_liner", ""),
        core_idea=entry["core_idea"],
        key_points=list(entry["key_points"]),
        deeper_notes=deeper.get("notes"),
        own_words_q=own_words.get("q"),
        own_words_a=own_words.get("answer"),
        quiz=entry.get("quiz") or [],
    )


def _fail(path: Path, message: str):
    raise SystemExit(f"{path.name}: {message}")


def save_loaded(chapters: list[Chapter]) -> list[Path]:
    """Write each chapter to data/<chapter_id>_loaded.json: the LOAD stage's output."""
    DATA_DIR.mkdir(exist_ok=True)
    for old in DATA_DIR.glob("*_loaded.json"):  # so data/ never keeps a deleted chapter
        old.unlink()
    paths = []
    for chapter in chapters:
        record = asdict(chapter)
        record["file"] = str(chapter.file.relative_to(PROJECT_ROOT))  # Path isn't JSON
        for lesson_record, lesson in zip(record["lessons"], chapter.lessons):
            lesson_record["full_id"] = lesson.full_id  # for reading only; recomputed on load
        out = DATA_DIR / f"{chapter.id}_loaded.json"
        out.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        paths.append(out)
    return paths


def read_loaded() -> list[Chapter]:
    """Read the LOAD stage's output back into Chapter/Lesson objects (used by CHUNK)."""
    paths = sorted(DATA_DIR.glob("*_loaded.json"))
    if not paths:
        raise SystemExit("No loaded chapters in data/. Run: python -m vivaprep.loader")
    chapters = []
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        lessons = []
        for lesson_record in record.pop("lessons"):
            lesson_record.pop("full_id", None)
            lessons.append(Lesson(**lesson_record))
        file = PROJECT_ROOT / record.pop("file")
        chapters.append(Chapter(file=file, lessons=lessons, **record))
    return sorted(chapters, key=lambda ch: ch.number)



if __name__ == "__main__":  # stage 1: python -m vivaprep.loader
    loaded = load_chapters()
    for chapter, out in zip(loaded, save_loaded(loaded)):
        print(f"wrote {out.relative_to(PROJECT_ROOT)}  ({len(chapter.lessons)} lessons)")
