#!/usr/bin/env python3
"""
Export anonymized human-alignment annotation packages from Step 2 transcripts.

Inputs follow the benchmark pipeline layout:
  <output_root>/transcripts/<kb_name>/<backend>/<profile_id>.json

Outputs:
  <output_dir>/annotation_package.jsonl   # blind material for raters
  <output_dir>/annotation_template.csv    # score entry template
  <output_dir>/annotation_key.json        # private mapping to backend/eval paths
  <output_dir>/rubric.md                  # human-readable Step 3 rubric
  <output_dir>/manifest.json
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import random
import sys
from typing import Any

_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.human_alignment.common import (
    ANNOTATION_COLUMNS,
    RUBRIC_MARKDOWN,
    RUBRIC_VERSION,
    read_json,
    write_json,
    write_jsonl,
)

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "benchmark" / "data" / "bench_pipeline"
REVIEW_UI_SOURCE = Path(__file__).with_name("review_ui.html")


def _parse_names(raw: str) -> list[str]:
    return sorted(set(n.strip() for n in raw.split(",") if n.strip()))


def _short_text(text: Any, max_chars: int) -> str:
    value = str(text or "").strip()
    return value[:max_chars] + ("..." if len(value) > max_chars else "")


def _format_profile(profile: dict[str, Any]) -> dict[str, Any]:
    ks = profile.get("knowledge_state", {}) or {}
    return {
        "profile_id": profile.get("profile_id", ""),
        "personality": profile.get("personality", ""),
        "education_background": profile.get("education_background", ""),
        "learning_purpose": profile.get("learning_purpose", ""),
        "known_well": ks.get("known_well", []),
        "partially_known": ks.get("partially_known", []),
        "unknown": ks.get("unknown", []),
        "beliefs": profile.get("beliefs", ""),
    }


def _source_excerpt_by_page(source_content: dict[str, Any] | None, max_chars: int) -> dict[str, str]:
    if not isinstance(source_content, dict):
        return {}
    excerpts: dict[str, str] = {}
    for page, text in sorted(source_content.items(), key=lambda item: str(item[0])):
        excerpt = _short_text(text, max_chars)
        if excerpt:
            excerpts[str(page)] = excerpt
    return excerpts


def _format_gaps(gaps: list[dict[str, Any]], source_content: dict[str, Any] | None, max_chars: int) -> list[dict[str, Any]]:
    source_by_page = _source_excerpt_by_page(source_content, max_chars)
    formatted = []
    for gap in gaps:
        pages = [str(p) for p in gap.get("source_pages", [])]
        formatted.append(
            {
                "gap_id": gap.get("gap_id", ""),
                "target_concept": gap.get("target_concept", ""),
                "gap_type": gap.get("gap_type", ""),
                "description": gap.get("description", ""),
                "manifestation": gap.get("manifestation", ""),
                "correct_understanding": gap.get("correct_understanding", ""),
                "source_pages": pages,
                "source_excerpts": {p: source_by_page[p] for p in pages if p in source_by_page},
            }
        )
    return formatted


def _dialog_messages(transcript: list[dict[str, Any]]) -> list[dict[str, str]]:
    messages = []
    for msg in transcript:
        role = msg.get("role")
        if role in {"student", "tutor"}:
            messages.append({"role": role, "content": str(msg.get("content", ""))})
    return messages


def _session_records_from_transcript(
    *,
    kb_name: str,
    backend: str,
    profile_id: str,
    transcript_path: Path,
    output_root: Path,
    source_chars: int,
) -> list[dict[str, Any]]:
    data = read_json(transcript_path)
    eval_path = output_root / "evaluations" / kb_name / backend / f"{profile_id}_eval.json"

    if isinstance(data, dict) and isinstance(data.get("sessions"), list):
        raw_sessions = data["sessions"]
    else:
        raw_sessions = [data]

    records = []
    for idx, session in enumerate(raw_sessions, start=1):
        entry = session.get("entry", {}) if isinstance(session, dict) else {}
        if not entry and isinstance(data, dict):
            entry = data.get("entry", {})
        profile = entry.get("profile", {}) or {}
        task = entry.get("task", {}) or {}
        transcript = session.get("transcript", []) if isinstance(session, dict) else []
        practice_questions = session.get("practice_questions", []) if isinstance(session, dict) else []
        entry_id = session.get("entry_id", entry.get("entry_id", "")) if isinstance(session, dict) else ""
        records.append(
            {
                "key": {
                    "kb_name": kb_name,
                    "backend": backend,
                    "profile_id": profile_id,
                    "entry_id": entry_id,
                    "session_index": idx,
                    "transcript_path": str(transcript_path),
                    "evaluation_path": str(eval_path),
                },
                "item": {
                    "rubric_version": RUBRIC_VERSION,
                    "scale": "1-5",
                    "profile": _format_profile(profile),
                    "task": {
                        "title": task.get("title", ""),
                        "description": task.get("description", ""),
                        "success_criteria": task.get("success_criteria", ""),
                        "target_gaps": task.get("target_gaps", []),
                    },
                    "gaps": _format_gaps(
                        entry.get("gaps", []) or [],
                        entry.get("source_content"),
                        source_chars,
                    ),
                    "dialog": _dialog_messages(transcript),
                    "practice_questions": practice_questions,
                    "turn_count": {
                        "dialog_messages": len(_dialog_messages(transcript)),
                        "practice_questions": len(practice_questions),
                    },
                },
            }
        )
    return records


def _collect_records(
    *,
    output_root: Path,
    kb_names: list[str],
    backends: list[str],
    source_chars: int,
) -> list[dict[str, Any]]:
    transcripts_root = output_root / "transcripts"
    collected = []
    for kb_name in kb_names:
        for backend in backends:
            backend_dir = transcripts_root / kb_name / backend
            if not backend_dir.exists():
                continue
            for transcript_path in sorted(backend_dir.glob("*.json")):
                profile_id = transcript_path.stem
                collected.extend(
                    _session_records_from_transcript(
                        kb_name=kb_name,
                        backend=backend,
                        profile_id=profile_id,
                        transcript_path=transcript_path,
                        output_root=output_root,
                        source_chars=source_chars,
                    )
                )
    return collected


def _write_template(path: Path, annotation_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ANNOTATION_COLUMNS)
        writer.writeheader()
        for annotation_id in annotation_ids:
            writer.writerow({"annotation_id": annotation_id})


def export_annotation_package(
    *,
    output_root: Path,
    kb_names: list[str],
    backends: list[str],
    output_dir: Path,
    source_chars: int = 1500,
    limit_per_backend: int = 0,
    seed: int = 13,
) -> dict[str, Any]:
    records = _collect_records(
        output_root=output_root,
        kb_names=kb_names,
        backends=backends,
        source_chars=source_chars,
    )
    if limit_per_backend > 0:
        rng = random.Random(seed)
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for record in records:
            key = record["key"]
            grouped.setdefault((key["kb_name"], key["backend"]), []).append(record)
        records = []
        for group_records in grouped.values():
            rng.shuffle(group_records)
            records.extend(group_records[:limit_per_backend])

    rng = random.Random(seed)
    rng.shuffle(records)

    package_rows = []
    key_rows = []
    for idx, record in enumerate(records, start=1):
        annotation_id = f"ha_{idx:06d}"
        item = {"annotation_id": annotation_id, **record["item"]}
        key = {"annotation_id": annotation_id, **record["key"]}
        package_rows.append(item)
        key_rows.append(key)

    output_dir.mkdir(parents=True, exist_ok=True)
    package_path = output_dir / "annotation_package.jsonl"
    key_path = output_dir / "annotation_key.json"
    template_path = output_dir / "annotation_template.csv"
    rubric_path = output_dir / "rubric.md"
    review_ui_path = output_dir / "review_ui.html"
    manifest_path = output_dir / "manifest.json"

    write_jsonl(package_path, package_rows)
    write_json(key_path, {"rubric_version": RUBRIC_VERSION, "items": key_rows})
    _write_template(template_path, [row["annotation_id"] for row in package_rows])
    rubric_path.write_text(RUBRIC_MARKDOWN, encoding="utf-8")
    if REVIEW_UI_SOURCE.exists():
        review_ui_path.write_text(REVIEW_UI_SOURCE.read_text(encoding="utf-8"), encoding="utf-8")

    manifest = {
        "step": "human_alignment_export_annotations",
        "timestamp": datetime.now().isoformat(),
        "rubric_version": RUBRIC_VERSION,
        "output_root": str(output_root),
        "kb_names": kb_names,
        "backends": backends,
        "num_annotation_items": len(package_rows),
        "limit_per_backend": limit_per_backend,
        "source_chars_per_page": source_chars,
        "package_path": str(package_path),
        "annotation_key_path": str(key_path),
        "annotation_template_path": str(template_path),
        "rubric_path": str(rubric_path),
        "review_ui_path": str(review_ui_path),
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export blind human annotation packages")
    parser.add_argument("--kb-names", required=True, help="Comma-separated KB names")
    parser.add_argument("--backends", required=True, help="Comma-separated backend names")
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help=f"Pipeline output root (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory (default: <output_root>/human_alignment)",
    )
    parser.add_argument(
        "--source-chars",
        type=int,
        default=1500,
        help="Max source excerpt chars per page included for raters",
    )
    parser.add_argument(
        "--limit-per-backend",
        type=int,
        default=0,
        help="Optional sample cap per (KB, backend); 0 keeps all",
    )
    parser.add_argument("--seed", type=int, default=13, help="Randomization seed")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (PROJECT_ROOT / output_root).resolve()
    output_dir = Path(args.output_dir) if args.output_dir else output_root / "human_alignment"
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()

    manifest = export_annotation_package(
        output_root=output_root,
        kb_names=_parse_names(args.kb_names),
        backends=_parse_names(args.backends),
        output_dir=output_dir,
        source_chars=args.source_chars,
        limit_per_backend=args.limit_per_backend,
        seed=args.seed,
    )
    print(f"Annotation package: {manifest['package_path']}")
    print(f"Annotation template: {manifest['annotation_template_path']}")
    print(f"Review UI: {manifest['review_ui_path']}")
    print(f"Private key: {manifest['annotation_key_path']}")
    print(f"Items: {manifest['num_annotation_items']}")


if __name__ == "__main__":
    main()
