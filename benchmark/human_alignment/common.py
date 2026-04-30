#!/usr/bin/env python3
"""Shared helpers for human-alignment annotation export and analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

METRIC_FIELDS = [
    "source_faithfulness",
    "personalization",
    "applicability",
    "vividness",
    "logical_depth",
    "pq_fitness",
    "pq_groundedness",
    "pq_diversity",
    "pq_answer_quality",
    "pq_cross_concept",
]

ANNOTATION_COLUMNS = [
    "annotation_id",
    "rater_id",
    *METRIC_FIELDS,
    "comment",
]

RUBRIC_VERSION = "benchmark_step3_human_v1"

RUBRIC_MARKDOWN = """# DeepTutor Human Alignment Rubric

Use the same 1-5 scale as the benchmark Step 3 LLM judge.

## Scale

- 5: Excellent; strong, consistent evidence.
- 4: Good; clear evidence with only minor issues.
- 3: Adequate; acceptable but generic or uneven.
- 2: Weak; notable flaws.
- 1: Poor; mostly missing, incorrect, or unhelpful.

## Metrics

- `source_faithfulness`: Tutor responses are faithful to the provided source excerpts.
- `personalization`: Tutor adapts to the student profile, knowledge state, and current confusion.
- `applicability`: Tutor responses help the student make progress on the task.
- `vividness`: Explanations are concrete, vivid, and supported by examples where useful.
- `logical_depth`: Reasoning and concept development are sufficiently deep and coherent.
- `pq_fitness`: Practice questions fit the student and target gaps.
- `pq_groundedness`: Practice questions are consistent with the source excerpts.
- `pq_diversity`: Practice questions cover varied angles rather than repeating one pattern.
- `pq_answer_quality`: Practice question options/answers are well formed and non-trivial.
- `pq_cross_concept`: Practice questions connect related concepts where appropriate.

`turn_count` is objective metadata and should not be scored by human raters.
"""


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def iter_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clamp_score(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, str) and not raw.strip():
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value < 1 or value > 5:
        return None
    return value


def safe_avg(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None

