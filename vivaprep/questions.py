from __future__ import annotations

import random
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from vivaprep.chunker import Chunk, chunk_lesson
from vivaprep.llm import ask_claude_json
from vivaprep.loader import Chapter, Lesson, read_loaded

# QUESTION MODE (v2): Claude writes interview questions from ONE lesson at a time.
# No search: we know which lesson, so the whole lesson is sent. That scope is the
# guardrail (Claude can't ask about what it wasn't shown). Structured output
# guarantees the reply's shape; code checks its content: every key point tag must
# be real, and each requested question type must be there.
# A chapter quiz = one question per lesson, types rotated, lessons run in parallel.

QUESTION_TYPES = ["definition", "why", "scenario"]
TYPE_GUIDE = {
    "definition": "what something is or means",
    "why": "a reason or a trade-off",
    "scenario": "a short, realistic situation where the person must apply the lesson",
}


def system_prompt(types: list[str]) -> str:
    guide = "\n".join(f"   - {t}: {TYPE_GUIDE[t]}" for t in types)
    return f"""You write interview questions that test someone on one lesson of their study notes, given in <lesson>.
Rules:
1. Write exactly {len(types)} question(s), one of each of these types:
{guide}
2. Every question must be answerable from the lesson alone. Never ask about anything the lesson doesn't say.
3. Tag each question with the key point ids (kp1, kp2, ...) a full answer needs.
4. Don't repeat or rephrase the questions in <already_asked>.
5. Ask one thing per question, in one or two sentences, the way an interviewer would."""


# The shape the API forces the reply into. It can't know which kp ids are real
# for this lesson, so code checks that (check_questions).
SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": QUESTION_TYPES},
                    "question": {"type": "string"},
                    "tests": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["type", "question", "tests"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["questions"],
    "additionalProperties": False,
}


@dataclass
class Question:
    type: str          # definition | why | scenario
    text: str
    tests: list[str]   # key point ids it tests, e.g. ["kp1", "kp3"]: what v3 grades against


@dataclass
class Quiz:
    chapter: Chapter
    lesson: Lesson
    questions: list[Question]
    attempts: int                                      # 1, or 2 if the first reply failed the checks
    problems: list[str] = field(default_factory=list)  # what was wrong with the last reply (dropped questions)


def find_chapter(chapter_id: str) -> Chapter:
    for chapter in read_loaded():
        if chapter.id == chapter_id:
            return chapter
    raise SystemExit(f"No chapter {chapter_id!r}. Run: python -m vivaprep.questions  (lists chapters)")


def find_lesson(full_id: str) -> tuple[Chapter, Lesson]:
    for chapter in read_loaded():
        for lesson in chapter.lessons:
            if lesson.full_id == full_id:
                return chapter, lesson
    raise SystemExit(f"No lesson {full_id!r}. Run: python -m vivaprep.questions  (lists lessons)")


def build_message(lesson: Lesson, chunks: list[Chunk]) -> str:
    """The lesson as labelled lines (kp1: ..., kp2: ...) plus the quiz questions not to repeat."""
    lines = [f'<lesson id="{lesson.full_id}" title="{lesson.title}">']
    for chunk in chunks:
        label = chunk.id.rsplit(".", 1)[-1]  # core, kp1, ..., deeper, ownwords
        lines.append(f"{label}: {chunk.text}")
    lines.append("</lesson>")
    lines.append("<already_asked>")
    lines.extend(f"- {item['q']}" for item in lesson.quiz)
    lines.append("</already_asked>")
    return "\n".join(lines)


def check_questions(items: list[dict], valid_kps: set[str], types: list[str]) -> tuple[list[Question], list[str]]:
    """Keep the questions that pass every check; describe the ones that don't."""
    good, problems, seen_types = [], [], set()
    for item in items:
        tests = [tag.strip().rsplit(".", 1)[-1] for tag in item["tests"]]  # "ch01.l03.kp2" → "kp2"
        unknown = [tag for tag in tests if tag not in valid_kps]
        if item["type"] not in types:
            problems.append(f"{item['type']}: a type that wasn't asked for")
        elif not item["question"].strip():
            problems.append(f"{item['type']}: empty question")
        elif not tests:
            problems.append(f"{item['type']}: no key point tags")
        elif unknown:
            problems.append(f"{item['type']}: tags that aren't key points of this lesson: {', '.join(unknown)}")
        elif item["type"] in seen_types:
            problems.append(f"{item['type']}: a second question of the same type")
        else:
            good.append(Question(item["type"], item["question"].strip(), tests))
            seen_types.add(item["type"])
    missing = [t for t in types if t not in seen_types]
    if missing:
        problems.append(f"missing question type(s): {', '.join(missing)}")
    return good, problems


def generate_questions(chapter: Chapter, lesson: Lesson, types: list[str] = QUESTION_TYPES) -> Quiz:
    """Questions of the given types for one lesson (one Claude call, two if the first fails the checks)."""
    chunks = chunk_lesson(chapter, lesson)
    valid_kps = {chunk.id.rsplit(".", 1)[-1] for chunk in chunks if chunk.type == "key_point"}
    system, message = system_prompt(types), build_message(lesson, chunks)
    for attempt in (1, 2):  # guardrail: if the reply fails the checks, ask once more
        reply = ask_claude_json(system, message, SCHEMA)
        questions, problems = check_questions(reply["questions"], valid_kps, types)
        if not problems:
            break
    return Quiz(chapter, lesson, questions, attempt, problems)  # after 2 tries, keep only what passed


def generate_chapter_quiz(chapter_id: str) -> list[Quiz]:
    """One question per lesson. Types rotate (definition → why → scenario) from a random
    start, so the set is mixed and varies between runs. Lessons are sent in parallel."""
    chapter = find_chapter(chapter_id)
    start = random.randrange(len(QUESTION_TYPES))
    types = [QUESTION_TYPES[(start + i) % len(QUESTION_TYPES)] for i in range(len(chapter.lessons))]
    with ThreadPoolExecutor(max_workers=len(chapter.lessons)) as pool:
        jobs = [pool.submit(generate_questions, chapter, lesson, [t]) for lesson, t in zip(chapter.lessons, types)]
        return [job.result() for job in jobs]  # same order as the lessons


def print_chapters() -> None:
    for chapter in read_loaded():
        print(f"\n{chapter.id}  {chapter.title}")
        for lesson in chapter.lessons:
            draft = "  ⚠ draft note" if lesson.draft else ""
            print(f"  {lesson.full_id:<10} {lesson.title}{draft}")


def print_quizzes(quizzes: list[Quiz], debug: bool = False) -> None:
    number = 0
    for quiz in quizzes:
        draft = "  ⚠ draft note" if quiz.lesson.draft else ""
        for question in quiz.questions:
            number += 1
            print(f"\n{number}. [{question.type}] {question.text}")
            print(f"   from: {quiz.lesson.full_id} {quiz.lesson.title}{draft}")
            if debug:
                print(f"   tests: {', '.join(question.tests)}   attempts: {quiz.attempts}")
        if debug:
            for problem in quiz.problems:
                print(f"⚠ {quiz.lesson.full_id} dropped: {problem}")


if __name__ == "__main__":
    # python -m vivaprep.questions                      → list chapters and lessons
    # python -m vivaprep.questions [--debug] ch01       → one question per lesson of the chapter
    # python -m vivaprep.questions [--debug] ch01.l03   → 3 questions (one per type) for one lesson
    args = sys.argv[1:]
    debug = "--debug" in args
    args = [a for a in args if a != "--debug"]
    if not args:
        print_chapters()
    elif "." in args[0]:
        print_quizzes([generate_questions(*find_lesson(args[0]))], debug)
    else:
        print_quizzes(generate_chapter_quiz(args[0]), debug)
