from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from counterfactual_grounding.data.dataset import (
    RAGTruthExample,
    load_ragtruth_examples,
    select_balanced_pilot,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_loader_projects_response_labels_out_of_the_training_contract(tmp_path: Path):
    sources = tmp_path / "source_info.jsonl"
    responses = tmp_path / "response.jsonl"
    _write_jsonl(
        sources,
        [
            {
                "source_id": "s1",
                "task_type": "Summary",
                "prompt": "Summarize:\nEvidence.\noutput:",
                "source_info": "Evidence.",
            }
        ],
    )
    _write_jsonl(
        responses,
        [
            {
                "id": "r1",
                "source_id": "s1",
                "response": "Answer.",
                "split": "train",
                "model": "llama-2-7b-chat",
                "quality": "good",
                "labels": [{"start": 0, "end": 3}],
            }
        ],
    )

    examples = load_ragtruth_examples(responses, sources, split="train")

    assert len(examples) == 1
    payload = asdict(examples[0])
    assert all("label" not in key.casefold() for key in payload)
    assert examples[0].source_record() == {
        "source_id": "s1",
        "task_type": "Summary",
        "prompt": "Summarize:\nEvidence.\noutput:",
        "source_info": "Evidence.",
    }


def test_loader_selects_explicit_response_ids_without_using_labels(tmp_path: Path):
    sources = tmp_path / "source_info.jsonl"
    responses = tmp_path / "response.jsonl"
    _write_jsonl(
        sources,
        [
            {
                "source_id": f"s{i}",
                "task_type": "Summary",
                "prompt": f"P{i} E{i}",
                "source_info": f"E{i}",
            }
            for i in range(3)
        ],
    )
    _write_jsonl(
        responses,
        [
            {
                "id": f"r{i}",
                "source_id": f"s{i}",
                "response": f"A{i}",
                "split": "train",
                "labels": [],
            }
            for i in range(3)
        ],
    )

    examples = load_ragtruth_examples(
        responses, sources, split="train", response_ids=["r2", "r0"]
    )

    assert [item.response_id for item in examples] == ["r2", "r0"]


def test_loader_rejects_source_ids_shared_across_official_splits(tmp_path: Path):
    sources = tmp_path / "source_info.jsonl"
    responses = tmp_path / "response.jsonl"
    _write_jsonl(
        sources,
        [
            {
                "source_id": "shared",
                "task_type": "Summary",
                "prompt": "P E",
                "source_info": "E",
            }
        ],
    )
    _write_jsonl(
        responses,
        [
            {
                "id": "r-train",
                "source_id": "shared",
                "response": "A",
                "split": "train",
            },
            {
                "id": "r-test",
                "source_id": "shared",
                "response": "B",
                "split": "test",
            },
        ],
    )

    with pytest.raises(ValueError, match="source.*split|split.*source"):
        load_ragtruth_examples(responses, sources, split="train")


def test_pilot_selection_balances_tasks_and_never_reuses_a_source():
    examples = [
        RAGTruthExample(
            response_id=f"{task}-{index}",
            source_id=f"{task}-source-{index}",
            split="train",
            task_type=task,
            prompt="P",
            source_info="E",
            response="R",
            generator_model="observer-a",
        )
        for task in ("Summary", "QA", "Data2txt")
        for index in range(4)
    ]
    # A second response from an already represented source must not enter.
    examples.append(
        RAGTruthExample(
            response_id="duplicate-source-response",
            source_id=examples[0].source_id,
            split="train",
            task_type="Summary",
            prompt="P",
            source_info="E",
            response="R",
            generator_model="observer-a",
        )
    )

    selected = select_balanced_pilot(examples, limit=9, seed=42)

    assert len(selected) == 9
    assert len({item.source_id for item in selected}) == 9
    assert {
        task: sum(item.task_type == task for item in selected)
        for task in ("Summary", "QA", "Data2txt")
    } == {
        "Summary": 3,
        "QA": 3,
        "Data2txt": 3,
    }
