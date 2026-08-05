from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch

from counterfactual_grounding import experiment
from counterfactual_grounding.experiment import PilotConfig, run_pilot
from counterfactual_grounding.teacher.mediation import KVStore, MediationRun


class CharacterTokenizer:
    is_fast = True
    vocab_size = 256
    special_tokens_map: ClassVar[dict[str, str]] = {}
    chat_template = None

    def __call__(self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool):
        assert not add_special_tokens
        result: dict[str, object] = {"input_ids": [ord(value) for value in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result


class FakeObserver(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(
            use_cache=False,
            to_dict=lambda: {"model_type": "llama", "fixture": True},
        )


class FakeBackend:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.factual_ids: torch.Tensor | None = None

    @staticmethod
    def _store(tag: int, positions: torch.Tensor) -> KVStore:
        value = torch.full((1, 1, positions.numel(), 1), float(tag))
        return KVStore(positions=positions.clone(), keys={0: value}, values={0: value})

    def run(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor,
            target_positions: torch.Tensor, capture_positions: torch.Tensor | None = None,
            sender: KVStore | None = None, patch_positions: torch.Tensor | None = None) -> MediationRun:
        del attention_mask, patch_positions
        count = target_positions.numel()
        base = torch.arange(count, dtype=torch.float32).unsqueeze(0)
        if capture_positions is not None:
            if self.factual_ids is None:
                self.factual_ids = input_ids.clone()
                return MediationRun(base + 1.0, self._store(1, capture_positions))
            return MediationRun(base, self._store(0, capture_positions))
        assert sender is not None and self.factual_ids is not None
        receiver_factual = torch.equal(input_ids, self.factual_ids)
        sender_factual = int(sender.keys[0][0, 0, 0, 0]) == 1
        if receiver_factual and not sender_factual:
            score = base + 0.5
            score[:, 0] = base[:, 0] + 1.0
        elif not receiver_factual and sender_factual:
            score = base + 0.25
            score[:, 0] = base[:, 0]
        elif receiver_factual and sender_factual:
            score = base + 1.0
        else:
            score = base
        return MediationRun(score, None)


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_full_pilot_writes_recomputable_label_free_artifacts(tmp_path: Path, monkeypatch):
    responses = tmp_path / "response.jsonl"
    sources = tmp_path / "source_info.jsonl"
    model_path = tmp_path / "model"
    output = tmp_path / "output"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type":"llama"}', encoding="utf-8")
    evidence = "The measured value is 8."
    _jsonl(sources, [{
        "source_id": "s1", "task_type": "Summary", "source_info": evidence,
        "prompt": f"Summarize:\n{evidence}\noutput:",
    }])
    _jsonl(responses, [{
        "id": "r1", "source_id": "s1", "response": "The value is eight.",
        "split": "train", "model": "generator-a", "labels": [{"start": 0}],
        "quality": "good", "temperature": 0.7,
    }])
    tokenizer = CharacterTokenizer()
    model = FakeObserver().eval()
    monkeypatch.setattr(experiment, "_tokenizer_only", lambda config: tokenizer)
    monkeypatch.setattr(experiment, "_load_observer", lambda config: (tokenizer, model))
    monkeypatch.setattr(experiment, "LlamaKVMediationBackend", FakeBackend)

    manifest = run_pilot(PilotConfig(
        responses=responses, sources=sources, model=model_path, output_dir=output,
        num_samples=1, device="cpu", dtype="float32", max_counterfactuals=2,
    ))

    assert manifest["run_state"] == "complete"
    assert manifest["gate_status"] == "partial"
    assert manifest["counts"]["mediation_complete"] == 1
    effects = [json.loads(line) for line in (output / "effects.jsonl").read_text().splitlines()]
    assert effects
    assert {"Y11", "Y00", "Y10", "Y01", "predictor_position",
            "non_history_kv_effect", "history_kv_effect"} <= effects[0].keys()
    for artifact in ("manifest.json", "selection.jsonl", "counterfactual_audit.jsonl",
                     "effects.jsonl", "gate_report.json"):
        text = (output / artifact).read_text(encoding="utf-8").casefold()
        assert '"labels"' not in text
        assert '"quality"' not in text
        assert '"temperature"' not in text


def test_interrupted_run_resumes_verified_sample_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses = tmp_path / "response.jsonl"
    sources = tmp_path / "source_info.jsonl"
    model_path = tmp_path / "model"
    output = tmp_path / "output"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        '{"model_type":"llama"}', encoding="utf-8"
    )
    evidence = "The measured value is 8."
    _jsonl(
        sources,
        [
            {
                "source_id": "s1",
                "task_type": "Summary",
                "source_info": evidence,
                "prompt": f"Summarize:\n{evidence}\noutput:",
            }
        ],
    )
    _jsonl(
        responses,
        [
            {
                "id": "r1",
                "source_id": "s1",
                "response": "The value is eight.",
                "split": "train",
                "model": "generator-a",
            }
        ],
    )
    tokenizer = CharacterTokenizer()
    model = FakeObserver().eval()
    monkeypatch.setattr(experiment, "_tokenizer_only", lambda config: tokenizer)
    monkeypatch.setattr(experiment, "_load_observer", lambda config: (tokenizer, model))
    monkeypatch.setattr(experiment, "LlamaKVMediationBackend", FakeBackend)
    config = PilotConfig(
        responses=responses,
        sources=sources,
        model=model_path,
        output_dir=output,
        num_samples=1,
        device="cpu",
        dtype="float32",
    )

    real_atomic_jsonl = experiment.atomic_jsonl

    def interrupt_before_aggregation(
        path: str | Path, records: Sequence[Mapping[str, object]]
    ) -> None:
        if Path(path).name == "effects.jsonl":
            raise RuntimeError("simulated interruption after sample shard")
        real_atomic_jsonl(path, records)

    monkeypatch.setattr(experiment, "atomic_jsonl", interrupt_before_aggregation)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_pilot(config)
    assert json.loads((output / "manifest.json").read_text())["run_state"] == "failed"
    assert len(list((output / "gate1_shards").glob("*.json"))) == 1

    monkeypatch.setattr(experiment, "atomic_jsonl", real_atomic_jsonl)

    def forbidden_recompute(*args: object, **kwargs: object) -> None:
        raise AssertionError("a verified completed shard must not be recomputed")

    monkeypatch.setattr(experiment, "run_gate1_pilot", forbidden_recompute)
    monkeypatch.setattr(experiment, "_load_observer", forbidden_recompute)
    monkeypatch.setattr(experiment, "_cuda_preflight", forbidden_recompute)
    shard_path = next((output / "gate1_shards").glob("*.json"))
    original_shard = shard_path.read_text(encoding="utf-8")
    tampered = json.loads(original_shard)
    tampered["record"]["response_idx"] += 1
    shard_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="shard content hash mismatch"):
        run_pilot(replace(config, resume=True))
    tampered["shard_content_sha256"] = experiment.canonical_hash(
        {
            key: value
            for key, value in tampered.items()
            if key != "shard_content_sha256"
        }
    )
    shard_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="response_idx disagrees"):
        run_pilot(replace(config, resume=True))
    shard_path.write_text(original_shard, encoding="utf-8")

    resumed = run_pilot(replace(config, resume=True))
    assert resumed["run_state"] == "complete"
    assert resumed["attempts"] == 4
    assert resumed["counts"]["mediation_complete"] == 1


def test_resume_requires_existing_matching_manifest_without_polluting_it(
    tmp_path: Path,
) -> None:
    responses = tmp_path / "response.jsonl"
    sources = tmp_path / "source_info.jsonl"
    model = tmp_path / "model"
    output = tmp_path / "output"
    responses.write_text("{}\n", encoding="utf-8")
    sources.write_text("{}\n", encoding="utf-8")
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    config = PilotConfig(
        responses=responses,
        sources=sources,
        model=model,
        output_dir=output,
        device="cpu",
        resume=True,
    )

    with pytest.raises(FileNotFoundError, match="cannot resume"):
        run_pilot(config)
    assert not (output / "manifest.json").exists()

    original = {
        "run_state": "failed",
        "run_identity_sha256": "different-run",
        "active_attempt_token": None,
        "failure": {"type": "OriginalError", "message": "preserve me"},
    }
    (output / "manifest.json").write_text(json.dumps(original), encoding="utf-8")
    with pytest.raises(RuntimeError, match="resume contract mismatch"):
        run_pilot(config)
    assert json.loads((output / "manifest.json").read_text()) == original


def test_pilot_rejects_sequence_ceiling_above_audited_limit(tmp_path: Path) -> None:
    responses = tmp_path / "response.jsonl"
    sources = tmp_path / "source_info.jsonl"
    model = tmp_path / "model"
    responses.write_text("", encoding="utf-8")
    sources.write_text("", encoding="utf-8")
    model.mkdir()

    with pytest.raises(ValueError, match="2048"):
        PilotConfig(
            responses=responses,
            sources=sources,
            model=model,
            output_dir=tmp_path / "out",
            max_sequence_tokens=4096,
        ).validate()
