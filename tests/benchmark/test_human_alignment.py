from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmark.human_alignment.common import METRIC_FIELDS
from benchmark.human_alignment.export_annotations import export_annotation_package
from benchmark.human_alignment.summarize_annotations import summarize_annotations


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _fake_entry(entry_id: str) -> dict:
    return {
        "entry_id": entry_id,
        "kb_name": "Calculus",
        "profile": {
            "profile_id": "p1",
            "personality": "curious",
            "education_background": "undergraduate",
            "learning_purpose": "learn derivatives",
            "knowledge_state": {
                "known_well": ["limits"],
                "partially_known": ["derivatives"],
                "unknown": ["chain rule"],
            },
        },
        "task": {
            "title": f"Task {entry_id}",
            "description": "Explain the chain rule.",
            "success_criteria": "Student can apply the chain rule.",
            "target_gaps": ["g1"],
        },
        "gaps": [
            {
                "gap_id": "g1",
                "target_concept": "Chain rule",
                "gap_type": "missing",
                "description": "Does not know nested derivatives.",
                "source_pages": [1],
            }
        ],
        "source_content": {"1": "The chain rule differentiates composite functions."},
    }


def _fake_metrics(offset: float = 0.0) -> dict:
    return {
        "source_faithfulness": {"avg_score": 4.0 + offset},
        "teaching_quality": {
            "personalization": {"avg": 3.0 + offset},
            "applicability": {"avg": 4.0 + offset},
            "vividness": {"avg": 3.0 + offset},
            "logical_depth": {"avg": 4.0 + offset},
        },
        "practice_questions": {
            "summary": {
                "avg_fitness": 4.0 + offset,
                "avg_groundedness": 4.0 + offset,
                "avg_diversity": 3.0 + offset,
                "avg_answer_quality": 4.0 + offset,
                "avg_cross_concept": 3.0 + offset,
            }
        },
    }


def test_export_annotations_splits_sessions_and_hides_backend(tmp_path: Path) -> None:
    output_root = tmp_path / "bench"
    transcript_path = output_root / "transcripts" / "Calculus" / "deep_tutor" / "p1.json"
    _write_json(
        transcript_path,
        {
            "profile_id": "p1",
            "sessions": [
                {
                    "entry_id": "e1",
                    "entry": _fake_entry("e1"),
                    "transcript": [
                        {"role": "student", "content": "I do not get this."},
                        {"role": "tutor", "content": "Let's use a nested function."},
                    ],
                    "practice_questions": ["Q1"],
                },
                {
                    "entry_id": "e2",
                    "entry": _fake_entry("e2"),
                    "transcript": [
                        {"role": "student", "content": "Another example?"},
                        {"role": "tutor", "content": "Try sin(x^2)."},
                    ],
                    "practice_questions": ["Q2"],
                },
            ],
        },
    )

    manifest = export_annotation_package(
        output_root=output_root,
        kb_names=["Calculus"],
        backends=["deep_tutor"],
        output_dir=tmp_path / "human",
        seed=7,
    )

    package_path = Path(manifest["package_path"])
    rows = [json.loads(line) for line in package_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert all("backend" not in row for row in rows)
    assert rows[0]["annotation_id"].startswith("ha_")
    assert rows[0]["profile"]["profile_id"] == "p1"
    assert rows[0]["gaps"][0]["source_excerpts"]["1"]

    key = json.loads(Path(manifest["annotation_key_path"]).read_text(encoding="utf-8"))
    assert {item["backend"] for item in key["items"]} == {"deep_tutor"}
    assert {item["session_index"] for item in key["items"]} == {1, 2}

    template_header = Path(manifest["annotation_template_path"]).read_text(encoding="utf-8").splitlines()[0]
    assert "annotation_id,rater_id,source_faithfulness" in template_header


def test_summarize_annotations_merges_human_and_llm_scores(tmp_path: Path) -> None:
    output_root = tmp_path / "bench"
    eval_path = output_root / "evaluations" / "Calculus" / "deep_tutor" / "p1_eval.json"
    _write_json(
        eval_path,
        {
            "profile_id": "p1",
            "sessions": [
                {"entry_id": "e1", "metrics": _fake_metrics(0.0)},
                {"entry_id": "e2", "metrics": _fake_metrics(1.0)},
            ],
        },
    )
    key_path = tmp_path / "human" / "annotation_key.json"
    _write_json(
        key_path,
        {
            "items": [
                {
                    "annotation_id": "ha_000001",
                    "kb_name": "Calculus",
                    "backend": "deep_tutor",
                    "profile_id": "p1",
                    "entry_id": "e1",
                    "session_index": 1,
                    "evaluation_path": str(eval_path),
                },
                {
                    "annotation_id": "ha_000002",
                    "kb_name": "Calculus",
                    "backend": "deep_tutor",
                    "profile_id": "p1",
                    "entry_id": "e2",
                    "session_index": 2,
                    "evaluation_path": str(eval_path),
                },
            ]
        },
    )

    annotations_path = tmp_path / "human" / "completed.csv"
    annotations_path.parent.mkdir(parents=True, exist_ok=True)
    with open(annotations_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["annotation_id", "rater_id", *METRIC_FIELDS, "comment"])
        writer.writeheader()
        for rater_id in ["r1", "r2"]:
            for idx, annotation_id in enumerate(["ha_000001", "ha_000002"], start=1):
                row = {"annotation_id": annotation_id, "rater_id": rater_id, "comment": ""}
                row.update({metric: str(2 + idx) for metric in METRIC_FIELDS})
                writer.writerow(row)

    summary = summarize_annotations(
        annotations_path=annotations_path,
        key_path=key_path,
        output_path=tmp_path / "human" / "summary.json",
    )

    assert summary["num_annotation_rows"] == 4
    assert summary["num_annotation_items"] == 2
    assert summary["num_raters"] == 2
    assert summary["metrics"]["source_faithfulness"]["n"] == 2
    assert summary["metrics"]["source_faithfulness"]["human_avg"] == 3.5
    assert summary["metrics"]["source_faithfulness"]["llm_avg"] == 4.5
    assert summary["metrics"]["source_faithfulness"]["spearman"] == 1.0
    assert summary["metrics"]["source_faithfulness"]["by_backend"]["deep_tutor"]["human_rank"] == 1
    assert summary["inter_rater"]["source_faithfulness"]["mean_pairwise_pearson"] == 1.0
    assert (tmp_path / "human" / "summary.md").exists()

