from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from counterfactual_grounding.cli import _parser

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "run_ragtruth_cept_transport.sh"


def _cache_preflight_source() -> str:
    source = RUNNER.read_text(encoding="utf-8")
    marker = (
        '"${PYTHON_BIN}" - "${ATTENTION_CACHE_ROOT}" "${MODEL_PATH}" <<\'PY\'\n'
    )
    start = source.index(marker) + len(marker)
    end = source.index("\nPY\n", start)
    return source[start:end]


def _eligible_frame_source() -> str:
    source = RUNNER.read_text(encoding="utf-8")
    marker = "# CEPT_ELIGIBLE_FRAME_PYTHON\n"
    start = source.index(marker) + len(marker)
    end = source.index("\nPY\n", start)
    return source[start:end]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_preflight_fixture(
    tmp_path: Path,
    *,
    cache_file_names: list[str],
    expected_files: list[str],
    transformers_version: str = "4.52.3",
) -> tuple[Path, Path]:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_bytes(b"model")
    model_hashes = {"config.json": _sha256(model / "config.json")}
    cache = tmp_path / "cache"
    for split in ("train", "test"):
        directory = cache / split
        directory.mkdir(parents=True)
        for name in ("attention_0.pt", "attention_1.pt"):
            (directory / name).write_bytes(name.encode())
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "state": "complete",
                    "cache_file_names": cache_file_names,
                    "expected_files": expected_files,
                    "matched_samples": 2,
                    "cache_files": 2,
                    "cache_files_sha256": {
                        name: _sha256(directory / name)
                        for name in ("attention_0.pt", "attention_1.pt")
                    },
                    "attention_cache_spec": {
                        "model_files_sha256": model_hashes,
                        "transformers_version": transformers_version,
                        "torch_version": torch.__version__,
                        "dtype": "torch.float16",
                        "attn_implementation": "eager",
                    },
                }
            ),
            encoding="utf-8",
        )
    return cache, model


def test_full_transport_runner_is_one_command_with_research_contract_defaults():
    source = RUNNER.read_text(encoding="utf-8")

    assert 'TEACHER_SAMPLES="${TEACHER_SAMPLES:-300}"' in source
    assert 'HISTORY_BLOCK_SIZE="${HISTORY_BLOCK_SIZE:-4}"' in source
    assert 'MAX_HISTORY_BLOCKS="${MAX_HISTORY_BLOCKS:-8}"' in source
    assert 'EPOCHS="${EPOCHS:-50}"' in source
    assert "layerstate_v5_runtimebound" in source
    assert 'DTYPE="${DTYPE:-${CACHE_DTYPE}}"' in source
    assert "graph_inventory_complete" in source
    assert "direct_seed_rescue" in source
    assert "joint_seed_history_rescue" in source
    assert "artifact hash mismatch" in source
    assert "--variants true rewired mass_only one_hop no_residual" in source
    assert "numeric_digit_surface_preserving_v1_first_candidate" in source
    assert "implementation_sha256" in source
    assert "legacy_model_files_sha256_exact_inventory" in source
    assert "cache_model_identity_{split}=verified" in source
    assert "observer_runtime_signature" in source
    assert "cache_content_identity_status" in source
    assert "eligible_train_response_ids.jsonl" in source
    assert "dataset_files_sha256" in source
    assert "generator_model_selector" in source
    assert "task_type_selector" in source
    assert 'ELIGIBLE_RESPONSE_IDS="${ELIGIBLE_RESPONSE_IDS}"' in source
    assert source.index("eligible_train_response_ids.jsonl") < source.index(
        "bash ./run_ragtruth_cept_pilot.sh"
    )
    assert 'gate2.get("claim_supported") is not True' in source
    assert "RAGTruth test labels were not opened" in source
    assert "evaluate-transport" in source
    assert 'print("detection_claim=", evaluation["detection_claim"])' in source
    assert source.index('gate2.get("claim_supported")') < source.index(
        "evaluate-transport"
    )


def test_transport_runner_rejects_existing_output_before_model_hashing_or_teacher():
    source = RUNNER.read_text(encoding="utf-8")

    output_guard = source.index('if [[ -e "${TRANSPORT_DIR}" ]]')
    model_hashing = source.index('"${PYTHON_BIN}" - "${ATTENTION_CACHE_ROOT}"')
    teacher_launch = source.index("bash ./run_ragtruth_cept_pilot.sh")
    assert output_guard < model_hashing < teacher_launch


@pytest.mark.parametrize(
    ("cache_file_names", "expected_files", "expected_error"),
    (
        (
            ["attention_0.pt", "attention_1.pt", "attention_1.pt"],
            ["attention_0.pt", "attention_1.pt"],
            "duplicate cache_file_names",
        ),
        (
            ["attention_0.pt", "attention_1.pt"],
            ["attention_0.pt", "attention_2.pt"],
            "cache_file_names and expected_files disagree",
        ),
    ),
)
def test_cache_preflight_rejects_ambiguous_declared_inventory(
    tmp_path: Path,
    cache_file_names: list[str],
    expected_files: list[str],
    expected_error: str,
):
    cache, model = _write_preflight_fixture(
        tmp_path,
        cache_file_names=cache_file_names,
        expected_files=expected_files,
    )

    completed = subprocess.run(
        [sys.executable, "-c", _cache_preflight_source(), str(cache), str(model)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert expected_error in completed.stderr


def test_cache_preflight_checks_metadata_before_hashing_model_files():
    source = _cache_preflight_source()

    assert source.index('for split in ("train", "test")') < source.index(
        "model_files = sorted(path for path in model_root.iterdir()"
    )


def test_cache_preflight_accepts_exact_declared_inventory(tmp_path: Path):
    cache, model = _write_preflight_fixture(
        tmp_path,
        cache_file_names=["attention_0.pt", "attention_1.pt"],
        expected_files=["attention_0.pt", "attention_1.pt"],
    )

    completed = subprocess.run(
        [sys.executable, "-c", _cache_preflight_source(), str(cache), str(model)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "cache_manifest_metadata_train=complete files=2" in completed.stdout
    assert "cache_manifest_metadata_test=complete files=2" in completed.stdout
    assert "cache_model_identity_train=verified" in completed.stdout
    assert "cache_model_identity_test=verified" in completed.stdout


def test_cache_preflight_accepts_dataset_order_expected_inventory(tmp_path: Path):
    cache, model = _write_preflight_fixture(
        tmp_path,
        cache_file_names=["attention_0.pt", "attention_1.pt"],
        expected_files=["attention_1.pt", "attention_0.pt"],
    )

    completed = subprocess.run(
        [sys.executable, "-c", _cache_preflight_source(), str(cache), str(model)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_cache_preflight_rejects_transformers_446_before_model_hashing(
    tmp_path: Path,
):
    cache, model = _write_preflight_fixture(
        tmp_path,
        cache_file_names=["attention_0.pt", "attention_1.pt"],
        expected_files=["attention_0.pt", "attention_1.pt"],
        transformers_version="4.46.3",
    )

    completed = subprocess.run(
        [sys.executable, "-c", _cache_preflight_source(), str(cache), str(model)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "4.46.3" in completed.stderr
    assert "4.52.3" in completed.stderr
    assert "re-extract" in completed.stderr.casefold()


def test_eligible_frame_deterministically_excludes_uncached_generator(
    tmp_path: Path,
):
    responses = tmp_path / "response.jsonl"
    sources = tmp_path / "source_info.jsonl"
    responses.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "id": "r-cache",
                    "source_id": "s-cache",
                    "split": "train",
                    "model": "llama-2-7b-chat",
                    "response": "cached",
                },
                {
                    "id": "r-other",
                    "source_id": "s-other",
                    "split": "train",
                    "model": "mistral-7b-instruct",
                    "response": "not cached",
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    sources.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "source_id": "s-cache",
                    "task_type": "Summary",
                    "prompt": "p",
                    "source_info": "e",
                },
                {
                    "source_id": "s-other",
                    "task_type": "Summary",
                    "prompt": "p",
                    "source_info": "e",
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    cache = tmp_path / "cache"
    dataset_hashes = {
        "response.jsonl": _sha256(responses),
        "source_info.jsonl": _sha256(sources),
    }
    for split in ("train", "test"):
        directory = cache / split
        directory.mkdir(parents=True)
        if split == "train":
            (directory / "attention_r-cache.pt").write_bytes(b"cache")
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "attention_cache_spec": {
                        "dataset_files_sha256": dataset_hashes,
                        "generator_model": "llama-2-7b-chat",
                        "task_type": "all",
                    }
                }
            ),
            encoding="utf-8",
        )
    output = tmp_path / "eligible_train_response_ids.jsonl"

    first = subprocess.run(
        [
            sys.executable,
            "-c",
            _eligible_frame_source(),
            str(cache),
            str(responses),
            str(sources),
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    original = output.read_bytes() if output.exists() else b""
    second = subprocess.run(
        [
            sys.executable,
            "-c",
            _eligible_frame_source(),
            str(cache),
            str(responses),
            str(sources),
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert output.read_bytes() == original
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows[0]["generator_model_selector"] == "llama-2-7b-chat"
    assert rows[0]["eligible_response_count"] == 1
    assert [row["response_id"] for row in rows[1:]] == ["r-cache"]


def test_transport_cli_keeps_training_and_label_evaluation_as_separate_commands():
    parser = _parser()
    training = parser.parse_args(
        [
            "train-transport",
            "--graph-index",
            "train.index.jsonl",
            "--teacher",
            "teacher.jsonl",
            "--output-dir",
            "run",
        ]
    )
    evaluation = parser.parse_args(
        [
            "evaluate-transport",
            "--predictions-dir",
            "run",
            "--test-graph-index",
            "test.index.jsonl",
            "--output",
            "evaluation.json",
        ]
    )

    assert training.command == "train-transport"
    assert training.allow_unidentifiable_pilot is False
    assert evaluation.command == "evaluate-transport"

    explicit_pilot = parser.parse_args(
        [
            "train-transport",
            "--graph-index",
            "train.index.jsonl",
            "--teacher",
            "teacher.jsonl",
            "--output-dir",
            "pilot-run",
            "--allow-unidentifiable-pilot",
        ]
    )
    assert explicit_pilot.allow_unidentifiable_pilot is True


def test_gate_pilot_cli_requires_verified_cache_sampling_frame():
    parser = _parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "pilot",
                "--responses",
                "response.jsonl",
                "--sources",
                "source_info.jsonl",
                "--model",
                "model",
                "--output-dir",
                "output",
            ]
        )

    parsed = parser.parse_args(
        [
            "pilot",
            "--responses",
            "response.jsonl",
            "--sources",
            "source_info.jsonl",
            "--model",
            "model",
            "--output-dir",
            "output",
            "--eligible-response-ids",
            "eligible_train_response_ids.jsonl",
        ]
    )
    assert parsed.eligible_response_ids == Path("eligible_train_response_ids.jsonl")
