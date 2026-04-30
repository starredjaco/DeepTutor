#!/usr/bin/env python3
"""Summarize human annotations against Step 3 LLM-as-judge outputs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime
from itertools import combinations
import json
import math
from pathlib import Path
import sys
from typing import Any

_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.human_alignment.common import (
    METRIC_FIELDS,
    clamp_score,
    read_json,
    safe_avg,
    write_json,
)

LLM_METRIC_LABELS = {
    "source_faithfulness": "source_faithfulness",
    "personalization": "teaching_quality.personalization",
    "applicability": "teaching_quality.applicability",
    "vividness": "teaching_quality.vividness",
    "logical_depth": "teaching_quality.logical_depth",
    "pq_fitness": "practice_questions.fitness",
    "pq_groundedness": "practice_questions.groundedness",
    "pq_diversity": "practice_questions.diversity",
    "pq_answer_quality": "practice_questions.answer_quality",
    "pq_cross_concept": "practice_questions.cross_concept",
}


def _load_key(path: Path) -> dict[str, dict[str, Any]]:
    data = read_json(path)
    items = data.get("items", data if isinstance(data, list) else [])
    return {row["annotation_id"]: row for row in items}


def _load_annotations(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _metric_from_summary(metrics: dict[str, Any], metric: str) -> float | None:
    sf = metrics.get("source_faithfulness", {}) or {}
    tq = metrics.get("teaching_quality", {}) or {}
    pq = metrics.get("practice_questions", {}) or {}
    if "summary" in pq and isinstance(pq["summary"], dict):
        pq = pq["summary"]

    values = {
        "source_faithfulness": sf.get("avg_score_overall", sf.get("avg_score")),
        "personalization": tq.get(
            "avg_personalization_overall",
            (tq.get("personalization", {}) or {}).get("avg"),
        ),
        "applicability": tq.get(
            "avg_applicability_overall",
            (tq.get("applicability", {}) or {}).get("avg"),
        ),
        "vividness": tq.get(
            "avg_vividness_overall",
            (tq.get("vividness", {}) or {}).get("avg"),
        ),
        "logical_depth": tq.get(
            "avg_logical_depth_overall",
            (tq.get("logical_depth", {}) or {}).get("avg"),
        ),
        "pq_fitness": pq.get("avg_fitness"),
        "pq_groundedness": pq.get("avg_groundedness"),
        "pq_diversity": pq.get("avg_diversity"),
        "pq_answer_quality": pq.get("avg_answer_quality"),
        "pq_cross_concept": pq.get("avg_cross_concept"),
    }
    value = values.get(metric)
    return float(value) if isinstance(value, (int, float)) else None


def _load_llm_scores_for_key(key: dict[str, Any]) -> dict[str, float | None]:
    eval_path = Path(key["evaluation_path"])
    if not eval_path.exists():
        return {metric: None for metric in METRIC_FIELDS}
    eval_data = read_json(eval_path)

    metrics: dict[str, Any] = {}
    if isinstance(eval_data.get("sessions"), list):
        session_index = int(key.get("session_index") or 1)
        entry_id = key.get("entry_id")
        session = None
        for candidate in eval_data["sessions"]:
            if entry_id and candidate.get("entry_id") == entry_id:
                session = candidate
                break
        if session is None and 1 <= session_index <= len(eval_data["sessions"]):
            session = eval_data["sessions"][session_index - 1]
        metrics = session.get("metrics", {}) if session else {}
    else:
        metrics = eval_data.get("metrics", {})

    return {metric: _metric_from_summary(metrics, metric) for metric in METRIC_FIELDS}


def _rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    x_den = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_den = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if x_den == 0 or y_den == 0:
        return None
    return round(num / (x_den * y_den), 4)


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    return _pearson(_rank(xs), _rank(ys))


def _kendall_tau_b(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    concordant = discordant = ties_x = ties_y = 0
    for i, j in combinations(range(len(xs)), 2):
        dx = (xs[i] > xs[j]) - (xs[i] < xs[j])
        dy = (ys[i] > ys[j]) - (ys[i] < ys[j])
        if dx == 0 and dy == 0:
            continue
        if dx == 0:
            ties_x += 1
        elif dy == 0:
            ties_y += 1
        elif dx == dy:
            concordant += 1
        else:
            discordant += 1
    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    if denom == 0:
        return None
    return round((concordant - discordant) / denom, 4)


def _pairwise_ranking_agreement(xs: list[float], ys: list[float]) -> float | None:
    same = total = 0
    for i, j in combinations(range(len(xs)), 2):
        dx = (xs[i] > xs[j]) - (xs[i] < xs[j])
        dy = (ys[i] > ys[j]) - (ys[i] < ys[j])
        if dx == 0 or dy == 0:
            continue
        total += 1
        if dx == dy:
            same += 1
    return round(same / total, 4) if total else None


def _backend_ranks(by_backend: dict[str, dict[str, Any]], metric: str) -> dict[str, dict[str, Any]]:
    human_vals = {
        backend: scores["human_avg"]
        for backend, scores in by_backend.items()
        if isinstance(scores.get("human_avg"), (int, float))
    }
    llm_vals = {
        backend: scores["llm_avg"]
        for backend, scores in by_backend.items()
        if isinstance(scores.get("llm_avg"), (int, float))
    }

    def _rank_desc(vals: dict[str, float]) -> dict[str, int]:
        ordered = sorted(vals.items(), key=lambda item: (-item[1], item[0]))
        return {backend: idx + 1 for idx, (backend, _) in enumerate(ordered)}

    human_rank = _rank_desc(human_vals)
    llm_rank = _rank_desc(llm_vals)
    out = {}
    for backend in sorted(set(human_rank) | set(llm_rank)):
        hr = human_rank.get(backend)
        lr = llm_rank.get(backend)
        out[backend] = {
            "human_rank": hr,
            "llm_rank": lr,
            "rank_delta": (lr - hr) if hr is not None and lr is not None else None,
        }
    return out


def _build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Human Alignment Summary",
        "",
        f"- Generated at: {summary['timestamp']}",
        f"- Annotation rows: {summary['num_annotation_rows']}",
        f"- Annotation items: {summary['num_annotation_items']}",
        f"- Raters: {summary['num_raters']}",
        "",
        "## Human vs LLM",
        "",
        "| Metric | N | Human Avg | LLM Avg | MAE | Spearman | Kendall | Pairwise Agree |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for metric in METRIC_FIELDS:
        row = summary["metrics"].get(metric, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    metric,
                    str(row.get("n", 0)),
                    _fmt(row.get("human_avg")),
                    _fmt(row.get("llm_avg")),
                    _fmt(row.get("mae")),
                    _fmt(row.get("spearman")),
                    _fmt(row.get("kendall_tau_b")),
                    _fmt(row.get("pairwise_ranking_agreement")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, int):
        return str(value)
    return "-"


def summarize_annotations(
    *,
    annotations_path: Path,
    key_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    key_by_id = _load_key(key_path)
    rows = _load_annotations(annotations_path)
    human_by_item: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    rows_by_metric_and_rater: dict[str, dict[str, dict[str, float]]] = {
        metric: defaultdict(dict) for metric in METRIC_FIELDS
    }
    raters = set()

    for row in rows:
        annotation_id = str(row.get("annotation_id", "")).strip()
        rater_id = str(row.get("rater_id", "")).strip()
        if not annotation_id or annotation_id not in key_by_id:
            continue
        if rater_id:
            raters.add(rater_id)
        for metric in METRIC_FIELDS:
            score = clamp_score(row.get(metric))
            if score is None:
                continue
            human_by_item[annotation_id][metric].append(score)
            if rater_id:
                rows_by_metric_and_rater[metric][rater_id][annotation_id] = score

    item_records = []
    for annotation_id, metric_scores in sorted(human_by_item.items()):
        key = key_by_id[annotation_id]
        llm_scores = _load_llm_scores_for_key(key)
        human_scores = {
            metric: safe_avg(scores)
            for metric, scores in metric_scores.items()
            if scores
        }
        item_records.append(
            {
                "annotation_id": annotation_id,
                "kb_name": key.get("kb_name"),
                "backend": key.get("backend"),
                "profile_id": key.get("profile_id"),
                "entry_id": key.get("entry_id"),
                "session_index": key.get("session_index"),
                "human_scores": human_scores,
                "llm_scores": llm_scores,
            }
        )

    metric_summary: dict[str, Any] = {}
    backend_metric_values: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: {"human": [], "llm": []})
    )
    for metric in METRIC_FIELDS:
        human_vals = []
        llm_vals = []
        abs_errors = []
        for rec in item_records:
            human = rec["human_scores"].get(metric)
            llm = rec["llm_scores"].get(metric)
            if isinstance(human, (int, float)) and isinstance(llm, (int, float)):
                human_vals.append(float(human))
                llm_vals.append(float(llm))
                abs_errors.append(abs(float(human) - float(llm)))
                backend = rec["backend"] or "unknown"
                backend_metric_values[metric][backend]["human"].append(float(human))
                backend_metric_values[metric][backend]["llm"].append(float(llm))

        by_backend = {}
        for backend, values in backend_metric_values[metric].items():
            by_backend[backend] = {
                "human_avg": safe_avg(values["human"]),
                "llm_avg": safe_avg(values["llm"]),
                "n": len(values["human"]),
            }
        for backend, ranks in _backend_ranks(by_backend, metric).items():
            by_backend.setdefault(backend, {}).update(ranks)

        metric_summary[metric] = {
            "llm_metric": LLM_METRIC_LABELS[metric],
            "n": len(human_vals),
            "human_avg": safe_avg(human_vals),
            "llm_avg": safe_avg(llm_vals),
            "mae": safe_avg(abs_errors),
            "spearman": _spearman(human_vals, llm_vals),
            "kendall_tau_b": _kendall_tau_b(human_vals, llm_vals),
            "pairwise_ranking_agreement": _pairwise_ranking_agreement(human_vals, llm_vals),
            "by_backend": by_backend,
        }

    inter_rater = {}
    for metric, by_rater in rows_by_metric_and_rater.items():
        pair_stats = []
        for rater_a, rater_b in combinations(sorted(by_rater.keys()), 2):
            shared = sorted(set(by_rater[rater_a]) & set(by_rater[rater_b]))
            xs = [by_rater[rater_a][annotation_id] for annotation_id in shared]
            ys = [by_rater[rater_b][annotation_id] for annotation_id in shared]
            corr = _pearson(xs, ys)
            pair_stats.append(
                {
                    "rater_a": rater_a,
                    "rater_b": rater_b,
                    "n": len(shared),
                    "pearson": corr,
                }
            )
        valid = [p["pearson"] for p in pair_stats if isinstance(p["pearson"], (int, float))]
        inter_rater[metric] = {
            "mean_pairwise_pearson": safe_avg([float(v) for v in valid]),
            "pairs": pair_stats,
        }

    summary = {
        "step": "human_alignment_summarize_annotations",
        "timestamp": datetime.now().isoformat(),
        "annotations_path": str(annotations_path),
        "annotation_key_path": str(key_path),
        "num_annotation_rows": len(rows),
        "num_annotation_items": len(item_records),
        "num_raters": len(raters),
        "metrics": metric_summary,
        "inter_rater": inter_rater,
        "items": item_records,
    }
    write_json(output_path, summary)
    md_path = output_path.with_suffix(".md")
    md_path.write_text(_build_markdown(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize human-vs-LLM alignment")
    parser.add_argument("--annotations", required=True, help="Completed CSV or JSONL annotations")
    parser.add_argument("--key", required=True, help="Private annotation_key.json from export")
    parser.add_argument(
        "--output",
        default="",
        help="Output summary JSON (default: sibling human_alignment_summary.json)",
    )
    args = parser.parse_args()

    annotations_path = Path(args.annotations)
    key_path = Path(args.key)
    output_path = Path(args.output) if args.output else key_path.parent / "human_alignment_summary.json"
    summary = summarize_annotations(
        annotations_path=annotations_path,
        key_path=key_path,
        output_path=output_path,
    )
    print(f"Summary: {output_path}")
    print(f"Markdown: {output_path.with_suffix('.md')}")
    print(
        f"Rows: {summary['num_annotation_rows']} | "
        f"Items: {summary['num_annotation_items']} | "
        f"Raters: {summary['num_raters']}"
    )


if __name__ == "__main__":
    main()

