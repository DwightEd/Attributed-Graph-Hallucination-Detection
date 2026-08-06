"""End-to-end, label-blind Gate 0/1 CEPT pilot runner."""

from __future__ import annotations

import json
import time
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .artifacts import (
    atomic_json,
    atomic_jsonl,
    canonical_hash,
    file_sha256,
    source_inventory_signature,
)
from .data.dataset import load_ragtruth_examples, select_balanced_pilot
from .data.ragtruth import build_ragtruth_layout
from .observer_runtime import (
    observer_runtime_identity,
    parse_observer_runtime_identity,
)
from .run_control import ExclusiveRunLock
from .teacher.counterfactuals import (
    CounterfactualGenerationError,
    generate_equal_token_counterfactuals,
)
from .teacher.mediation import BACKEND_ID, LlamaKVMediationBackend
from .teacher.pilot import (
    Gate1Pair,
    Gate1Record,
    Gate1RuntimeIdentity,
    TokenMediationEffect,
    run_gate1_pilot,
)
from .transport.data import make_transport_teacher_record

MAX_PILOT_SEQUENCE_TOKENS = 2048
ELIGIBLE_FRAME_SCHEMA = "cept-eligible-response-frame-v1"
ELIGIBLE_FRAME_ORIGIN = "verified_train_attention_cache_filename_inventory"


@dataclass(frozen=True)
class PilotConfig:
    responses: Path
    sources: Path
    model: Path
    output_dir: Path
    eligible_response_ids: Path
    split: str = "train"
    num_samples: int = 50
    seed: int = 42
    device: str = "cuda:0"
    dtype: str = "float16"
    max_sequence_tokens: int = 2048
    max_counterfactuals: int = 2
    history_block_size: int = 4
    max_history_blocks: int = 8
    resume: bool = False

    def validate(self) -> None:
        if self.split.casefold() != "train":
            raise ValueError("Gate 0/1 pilot is restricted to label-blind train data")
        if self.num_samples <= 0 or self.max_sequence_tokens <= 1:
            raise ValueError("num_samples and max_sequence_tokens must be positive")
        if self.max_sequence_tokens > MAX_PILOT_SEQUENCE_TOKENS:
            raise ValueError(
                "the audited eager-attention pilot ceiling is "
                f"{MAX_PILOT_SEQUENCE_TOKENS} tokens"
            )
        if self.max_counterfactuals <= 0:
            raise ValueError("max_counterfactuals must be positive")
        if self.history_block_size < 0 or self.max_history_blocks < 0:
            raise ValueError("history block controls must be non-negative")
        if (self.history_block_size == 0) != (self.max_history_blocks == 0):
            raise ValueError(
                "history_block_size and max_history_blocks must both be zero or positive"
            )
        if self.dtype not in {"auto", "float16", "bfloat16", "float32"}:
            raise ValueError("dtype must be auto, float16, bfloat16, or float32")
        for path, description in (
            (self.responses, "RAGTruth response.jsonl"),
            (self.sources, "RAGTruth source_info.jsonl"),
            (self.model, "observer model"),
            (self.eligible_response_ids, "verified cache eligible-response frame"),
        ):
            if not path.expanduser().exists():
                raise FileNotFoundError(f"{description} is absent: {path}")


def _load_eligible_response_frame(
    path: Path,
    *,
    split: str,
    dataset_files_sha256: Mapping[str, str],
) -> tuple[list[str], dict[str, object]]:
    """Read the immutable, label-free sampling frame emitted from cache membership."""

    source = path.expanduser().resolve()
    rows: list[Mapping[str, object]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid eligible-response JSON at {source}:{line_number}"
                ) from error
            if not isinstance(row, Mapping):
                raise TypeError(
                    f"eligible-response row must be an object: {source}:{line_number}"
                )
            if row.get("schema") != ELIGIBLE_FRAME_SCHEMA:
                raise ValueError(
                    f"eligible-response frame schema mismatch: {source}:{line_number}"
                )
            rows.append(row)
    if not rows or rows[0].get("record_type") != "metadata":
        raise ValueError("eligible-response frame must begin with one metadata row")
    if any(row.get("record_type") == "metadata" for row in rows[1:]):
        raise ValueError("eligible-response frame must contain exactly one metadata row")

    metadata = dict(rows[0])
    requested_split = split.casefold()
    if str(metadata.get("official_split", "")).casefold() != requested_split:
        raise ValueError("eligible-response frame official split mismatch")
    if metadata.get("origin") != ELIGIBLE_FRAME_ORIGIN:
        raise ValueError("eligible-response frame was not derived from verified cache")
    declared_dataset = metadata.get("dataset_files_sha256")
    if not isinstance(declared_dataset, Mapping) or {
        str(name): str(digest) for name, digest in declared_dataset.items()
    } != dict(dataset_files_sha256):
        raise RuntimeError(
            "eligible-response frame dataset hashes disagree with current RAGTruth files"
        )
    for selector in ("generator_model_selector", "task_type_selector"):
        value = metadata.get(selector)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"eligible-response frame requires non-empty {selector}")
    manifest_digest = metadata.get("cache_manifest_sha256")
    if (
        not isinstance(manifest_digest, str)
        or len(manifest_digest) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in manifest_digest)
    ):
        raise ValueError("eligible-response frame cache manifest hash is malformed")
    manifest_path = metadata.get("cache_manifest_path")
    if not isinstance(manifest_path, str) or not manifest_path.strip():
        raise ValueError("eligible-response frame requires cache_manifest_path")

    response_ids: list[str] = []
    cache_files: list[str] = []
    for line_number, row in enumerate(rows[1:], start=2):
        if row.get("record_type") != "response":
            raise ValueError(
                f"eligible-response frame has invalid record type at line {line_number}"
            )
        if str(row.get("official_split", "")).casefold() != requested_split:
            raise ValueError(
                f"eligible-response row has wrong split at line {line_number}"
            )
        response_id = str(row.get("response_id", "")).strip()
        cache_file = str(row.get("cache_file", "")).strip()
        if not response_id or not cache_file:
            raise ValueError(
                f"eligible-response row requires response_id/cache_file at line {line_number}"
            )
        if Path(cache_file).name != cache_file or not (
            cache_file.startswith("attention_") and cache_file.endswith(".pt")
        ):
            raise ValueError(
                f"eligible-response row has invalid cache_file at line {line_number}"
            )
        response_ids.append(response_id)
        cache_files.append(cache_file)
    if not response_ids:
        raise ValueError("eligible-response frame contains no response IDs")
    if len(response_ids) != len(set(response_ids)):
        raise ValueError("eligible-response frame contains duplicate response IDs")
    if len(cache_files) != len(set(cache_files)):
        raise ValueError("eligible-response frame contains cache filename collisions")
    declared_count = metadata.get("eligible_response_count")
    if (
        isinstance(declared_count, bool)
        or not isinstance(declared_count, int)
        or declared_count != len(response_ids)
    ):
        raise ValueError("eligible-response frame count does not match its response rows")
    return response_ids, metadata


def _dtype(name: str) -> torch.dtype | str:
    return {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _history_blocks(
    *, response_idx: int, token_count: int, size: int, maximum: int
) -> dict[str, torch.Tensor]:
    if size == 0 or maximum == 0:
        return {}
    history = list(range(response_idx, token_count - 1))
    blocks = [history[start : start + size] for start in range(0, len(history), size)]
    # Cover the full response trajectory instead of selecting high-attention
    # or recent-only blocks.  This keeps onset and continuation testable while
    # fixing the number of expensive intervention calls before labels exist.
    if len(blocks) > maximum:
        if maximum == 1:
            blocks = [blocks[len(blocks) // 2]]
        else:
            indices = [
                round(index * (len(blocks) - 1) / (maximum - 1))
                for index in range(maximum)
            ]
            blocks = [blocks[index] for index in indices]
    return {
        f"history-block-{values[0]}-{values[-1]}": torch.tensor(
            values, dtype=torch.long
        )
        for values in blocks
        if values
    }


def _load_observer(config: PilotConfig) -> tuple[object, torch.nn.Module]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model,
        use_fast=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    if not bool(getattr(tokenizer, "is_fast", False)):
        raise RuntimeError("Gate 0 requires a fast tokenizer with exact offsets")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    model = AutoModelForCausalLM.from_pretrained(
        config.model,
        torch_dtype=_dtype(config.dtype),
        attn_implementation="eager",
        local_files_only=True,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    model.config.use_cache = False
    parameter_devices = {parameter.device for parameter in model.parameters()}
    if parameter_devices != {device}:
        raise RuntimeError(
            "Gate 1 forbids device_map/model sharding; all parameters must share "
            f"{device}, got {sorted(map(str, parameter_devices))}"
        )
    return tokenizer, model


def _checkpoint_weight_bytes(path: Path) -> int:
    source = path.expanduser().resolve()
    if source.is_file():
        return source.stat().st_size
    safetensors = sorted(source.rglob("*.safetensors"))
    candidates = safetensors or sorted(source.rglob("pytorch_model*.bin"))
    return sum(item.stat().st_size for item in candidates)


def _cuda_preflight(config: PilotConfig) -> dict[str, object]:
    device = torch.device(config.device)
    if device.type != "cuda":
        report = {"event": "gpu_preflight", "device": str(device), "skipped": True}
        print(json.dumps(report), flush=True)
        return report
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    torch.cuda.set_device(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    checkpoint_bytes = _checkpoint_weight_bytes(config.model)
    dtype_multiplier = 2.0 if config.dtype == "float32" else 1.0
    estimated_parameter_bytes = int(checkpoint_bytes * dtype_multiplier)
    required_free_bytes = estimated_parameter_bytes + 4 * 1024**3
    report = {
        "event": "gpu_preflight",
        "device": str(device),
        "checkpoint_weight_bytes": checkpoint_bytes,
        "estimated_parameter_bytes": estimated_parameter_bytes,
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "required_free_bytes": required_free_bytes,
        "max_sequence_tokens": config.max_sequence_tokens,
        "attention_implementation": "eager",
    }
    print(json.dumps(report), flush=True)
    if checkpoint_bytes <= 0:
        raise RuntimeError("observer checkpoint contains no .safetensors/.bin weights")
    if free_bytes < required_free_bytes:
        raise RuntimeError(
            "insufficient free GPU memory for the unsharded eager-attention observer: "
            f"free={free_bytes} required>={required_free_bytes}"
        )
    return report


def _model_loaded_heartbeat(model: torch.nn.Module, config: PilotConfig) -> None:
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    report: dict[str, object] = {
        "event": "observer_loaded",
        "device": config.device,
        "parameter_bytes": parameter_bytes,
    }
    if torch.device(config.device).type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info(torch.device(config.device))
        report.update(
            {
                "cuda_free_bytes": free_bytes,
                "cuda_total_bytes": total_bytes,
                "cuda_allocated_bytes": torch.cuda.memory_allocated(
                    torch.device(config.device)
                ),
            }
        )
    print(json.dumps(report), flush=True)


def _tokenizer_only(config: PilotConfig) -> object:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model,
        use_fast=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    if not bool(getattr(tokenizer, "is_fast", False)):
        raise RuntimeError("Gate 0 requires a fast tokenizer with exact offsets")
    return tokenizer


def _runtime_identity(
    *,
    config: PilotConfig,
    tokenizer: object,
    model: torch.nn.Module,
    source_signature: str,
) -> Gate1RuntimeIdentity:
    import transformers

    tokenizer_signature = canonical_hash(
        {
            "source_inventory": source_signature,
            "class": type(tokenizer).__qualname__,
            "vocab_size": getattr(tokenizer, "vocab_size", None),
            "special_tokens": getattr(tokenizer, "special_tokens_map", None),
            "chat_template": getattr(tokenizer, "chat_template", None),
        }
    )
    model_signature = canonical_hash(
        {
            "source_inventory": source_signature,
            "class": type(model).__qualname__,
            "config": model.config.to_dict(),
            "dtype": str(next(model.parameters()).dtype),
        }
    )
    return Gate1RuntimeIdentity(
        model_signature=f"sha256:{model_signature}",
        tokenizer_signature=f"sha256:{tokenizer_signature}",
        transformers_version=transformers.__version__,
        torch_version=torch.__version__,
        backend_id=BACKEND_ID,
        patch_site="post_rope_kv_pre_repeat_kv",
    )


def _effect_rows(records: Sequence[Gate1Record]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        for effect in record.token_effects:
            rows.append(
                {
                    "response_id": record.sample_id,
                    "source_id": record.source_id,
                    "task_type": record.task_type,
                    "generator_model": record.generator_model,
                    "target_position": effect.target_position,
                    "predictor_position": effect.predictor_position,
                    "target_token_id": effect.target_token_id,
                    "Y11": effect.y11,
                    "Y00": effect.y00,
                    "Y10": effect.y10,
                    "Y01": effect.y01,
                    "total_effect": effect.total,
                    "non_history_kv_effect": effect.non_history_kv_effect,
                    "history_kv_effect": effect.mediated,
                    "alternate_history_kv_effect": effect.alternate,
                    "interaction": effect.interaction,
                    "decomposition_residual": effect.contract_residual,
                    "direct_seed_rescue": effect.direct_seed_rescue,
                    "joint_seed_history_rescue": effect.joint_seed_history_rescue,
                    "representation_residual": effect.representation_residual,
                    "seed_history_interaction": effect.seed_history_interaction,
                    "self_patch_error": effect.self_patch_error,
                    "block_rescue": effect.block_rescue,
                }
            )
    return rows


def _record_from_mapping(value: Mapping[str, object]) -> Gate1Record:
    raw_effects = value.get("token_effects")
    if not isinstance(raw_effects, list):
        raise TypeError("invalid Gate-1 shard: token_effects must be a list")
    effects: list[TokenMediationEffect] = []
    for raw in raw_effects:
        if not isinstance(raw, dict):
            raise TypeError("invalid Gate-1 shard: token effect must be an object")
        effects.append(TokenMediationEffect(**raw))
    return Gate1Record(
        sample_id=str(value["sample_id"]),
        source_id=str(value["source_id"]),
        task_type=str(value["task_type"]),
        generator_model=str(value["generator_model"]),
        response_idx=int(value["response_idx"]),
        token_effects=tuple(effects),
    )


def _pair_signature(pair: Gate1Pair) -> str:
    return canonical_hash(
        {
            "sample_id": pair.sample_id,
            "source_id": pair.source_id,
            "task_type": pair.task_type,
            "generator_model": pair.generator_model,
            "factual": pair.factual,
            "counterfactual": pair.counterfactual,
            "rescue_blocks": pair.rescue_blocks,
        }
    )


def _validate_record_against_pair(record: Gate1Record, pair: Gate1Pair) -> None:
    expected_identity = (
        pair.sample_id,
        pair.source_id,
        pair.task_type,
        pair.generator_model,
    )
    observed_identity = (
        record.sample_id,
        record.source_id,
        record.task_type,
        record.generator_model,
    )
    if observed_identity != expected_identity:
        raise RuntimeError("Gate-1 shard record identity disagrees with its pair")
    factual_ids = torch.as_tensor(pair.factual["input_ids"], dtype=torch.long).flatten()
    response_idx = int(pair.factual["response_idx"])
    if record.response_idx != response_idx:
        raise RuntimeError("Gate-1 shard response_idx disagrees with its pair")
    expected_positions = list(range(response_idx, factual_ids.numel()))
    if len(record.token_effects) != len(expected_positions):
        raise RuntimeError("Gate-1 shard token-effect count disagrees with its pair")
    expected_blocks = set(pair.rescue_blocks)
    for effect, target_position in zip(
        record.token_effects, expected_positions, strict=True
    ):
        if (
            effect.target_position != target_position
            or effect.predictor_position != target_position - 1
            or effect.target_token_id != int(factual_ids[target_position])
        ):
            raise RuntimeError(
                "Gate-1 shard target/predictor/token alignment disagrees with its pair"
            )
        if set(effect.block_rescue) != expected_blocks:
            raise RuntimeError("Gate-1 shard rescue blocks disagree with its pair")


def _runtime_from_mapping(value: Mapping[str, object]) -> Gate1RuntimeIdentity:
    try:
        return Gate1RuntimeIdentity(
            model_signature=str(value["model_signature"]),
            tokenizer_signature=str(value["tokenizer_signature"]),
            transformers_version=str(value["transformers_version"]),
            torch_version=str(value["torch_version"]),
            backend_id=str(value["backend_id"]),
            patch_site=str(value["patch_site"]),
        )
    except KeyError as error:
        raise RuntimeError("incomplete runtime identity in resume manifest") from error


def _load_verified_shard(
    path: Path,
    *,
    pair: Gate1Pair,
    runtime: Gate1RuntimeIdentity,
) -> tuple[Gate1Record, float]:
    shard = json.loads(path.read_text(encoding="utf-8"))
    if shard.get("schema") != "cept-gate1-sample-shard-v1":
        raise RuntimeError(f"unsupported resume shard schema: {path}")
    declared_shard_hash = shard.pop("shard_content_sha256", None)
    if declared_shard_hash != canonical_hash(shard):
        raise RuntimeError(f"resume shard content hash mismatch: {path}")
    if (
        shard.get("sample_id") != pair.sample_id
        or shard.get("pair_signature_sha256") != _pair_signature(pair)
        or shard.get("runtime") != asdict(runtime)
    ):
        raise RuntimeError(f"resume shard contract mismatch: {path}")
    raw_record = shard.get("record")
    if not isinstance(raw_record, dict):
        raise TypeError(f"resume shard has no complete record: {path}")
    record = _record_from_mapping(raw_record)
    _validate_record_against_pair(record, pair)
    return record, float(shard["self_patch_max_abs"])


def _length_bucket(token_count: int) -> str:
    if token_count < 512:
        return "0000-0511"
    if token_count < 1024:
        return "0512-1023"
    if token_count < 1536:
        return "1024-1535"
    if token_count <= 2048:
        return "1536-2048"
    return "over-2048"


def _stratified_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    task = Counter(str(row.get("task_type", "unknown")) for row in rows)
    generator = Counter(str(row.get("generator_model", "unknown")) for row in rows)
    joint = Counter(
        f"{row.get('task_type', 'unknown')}|{row.get('generator_model', 'unknown')}"
        for row in rows
    )
    lengths = Counter(
        _length_bucket(int(row["token_count"]))
        for row in rows
        if row.get("token_count") is not None
    )
    return {
        "samples": len(rows),
        "task": dict(sorted(task.items())),
        "generator": dict(sorted(generator.items())),
        "task_generator": dict(sorted(joint.items())),
        "length": dict(sorted(lengths.items())),
    }


def _run_pilot_locked(
    config: PilotConfig, output: Path, attempt_token: str
) -> dict[str, object]:
    import transformers

    print(json.dumps({"event": "provenance_hashing_started"}), flush=True)
    dataset_file_hashes = {
        "response.jsonl": file_sha256(config.responses),
        "source_info.jsonl": file_sha256(config.sources),
    }
    dataset_identity = {
        "responses": str(config.responses.expanduser().resolve()),
        "responses_sha256": dataset_file_hashes["response.jsonl"],
        "sources": str(config.sources.expanduser().resolve()),
        "sources_sha256": dataset_file_hashes["source_info.jsonl"],
    }
    eligible_ids, eligible_metadata = _load_eligible_response_frame(
        config.eligible_response_ids,
        split=config.split,
        dataset_files_sha256=dataset_file_hashes,
    )
    if config.num_samples > len(eligible_ids):
        raise ValueError(
            f"requested num_samples={config.num_samples} exceeds eligible cache "
            f"frame size={len(eligible_ids)}"
        )
    examples = load_ragtruth_examples(
        config.responses,
        config.sources,
        split=config.split,
        response_ids=eligible_ids,
    )
    generator_counts = Counter(
        example.generator_model or "unknown" for example in examples
    )
    estimand_scope = {
        "dataset": "RAGTruth",
        "official_split": config.split.casefold(),
        "eligible_population": "verified_train_attention_cache_inventory",
        "cache_manifest_path": str(eligible_metadata["cache_manifest_path"]),
        "cache_manifest_sha256": str(eligible_metadata["cache_manifest_sha256"]),
        "generator_model_selector": str(
            eligible_metadata["generator_model_selector"]
        ),
        "task_type_selector": str(eligible_metadata["task_type_selector"]),
        "generator_models": dict(sorted(generator_counts.items())),
        "selection_policy": "task_generator_round_robin_source_unique_sha256_v1",
    }
    model_source_signature = source_inventory_signature(config.model)
    contract_config = asdict(config)
    contract_config.pop("output_dir")
    contract_config.pop("resume")
    run_contract = {
        "config": contract_config,
        "dataset": dataset_identity,
        "sampling_frame": {
            "path": str(config.eligible_response_ids.expanduser().resolve()),
            "sha256": file_sha256(config.eligible_response_ids),
            "count": len(eligible_ids),
        },
        "estimand_scope": estimand_scope,
        "model_source_signature": f"sha256:{model_source_signature}",
        "implementation_sha256": {
            relative: file_sha256(Path(__file__).resolve().parent / relative)
            for relative in (
                "experiment.py",
                "teacher/counterfactuals.py",
                "teacher/mediation.py",
                "teacher/pilot.py",
                "observer_runtime.py",
                "transport/data.py",
            )
        },
        "environment": {
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "backend_id": BACKEND_ID,
        },
    }
    run_identity = canonical_hash(run_contract)
    print(
        json.dumps(
            {"event": "provenance_hashing_complete", "run_identity": run_identity}
        ),
        flush=True,
    )
    manifest_path = output / "manifest.json"
    initial_manifest: dict[str, Any] = {
        "schema": "cept-gate01-run-manifest-v2",
        "run_state": "in_progress",
        "gate_status": "not_evaluated",
        "method_scope": "Gate 0/1 intervention pilot; no student training or AUROC",
        "run_identity_sha256": run_identity,
        "run_contract": run_contract,
        "label_policy": (
            "response annotations, quality, and temperature are projected out before "
            "selection/layout/teacher construction"
        ),
        "attempts": 1,
        "active_attempt_token": attempt_token,
    }
    if manifest_path.exists():
        if not config.resume:
            raise FileExistsError(
                f"refusing to overwrite an existing CEPT run: {output}"
            )
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("run_state") == "complete":
            raise FileExistsError(f"CEPT run is already complete: {output}")
        if existing.get("run_identity_sha256") != run_identity:
            raise RuntimeError(
                "resume contract mismatch: config, data, model, or runtime changed"
            )
        initial_manifest = {
            **existing,
            "run_state": "in_progress",
            "failure": None,
            "attempts": int(existing.get("attempts", 1)) + 1,
            "active_attempt_token": attempt_token,
        }
    else:
        if config.resume:
            raise FileNotFoundError(
                f"cannot resume because manifest.json is absent: {output}"
            )
        unexpected = [
            item.name
            for item in output.iterdir()
            if item.name not in {".cept-run.lock", ".cept-shell.lock", "run.log"}
        ]
        if unexpected:
            raise FileExistsError(
                f"output directory contains unowned artifacts: {sorted(unexpected)}"
            )
    atomic_json(manifest_path, initial_manifest)

    tokenizer = _tokenizer_only(config)
    selected = select_balanced_pilot(
        examples, limit=config.num_samples, seed=config.seed
    )
    if len(selected) != config.num_samples:
        raise RuntimeError(
            f"source-unique balanced selection returned {len(selected)} of "
            f"{config.num_samples} requested samples"
        )
    eligible_set = set(eligible_ids)
    if any(example.response_id not in eligible_set for example in selected):
        raise RuntimeError(
            "balanced selection escaped the verified cache eligible-response frame"
        )

    selection_rows: list[dict[str, object]] = []
    counterfactual_rows: list[dict[str, object]] = []
    pairs: list[Gate1Pair] = []
    changed_positions_by_id: dict[str, tuple[int, ...]] = {}
    for rank, example in enumerate(selected):
        row: dict[str, object] = {
            "selection_rank": rank,
            "response_id": example.response_id,
            "source_id": example.source_id,
            "official_split": example.split,
            "task_type": example.task_type,
            "generator_model": example.generator_model or "unknown",
            "state": "selected",
        }
        try:
            layout = build_ragtruth_layout(
                example.source_record(), example.response, tokenizer
            )
            token_count = int(layout.input_ids.numel())
            row["token_count"] = token_count
            row["response_idx"] = layout.response_idx
            if token_count > config.max_sequence_tokens:
                raise CounterfactualGenerationError(
                    f"sequence has {token_count} tokens, exceeding the no-truncation "
                    f"limit {config.max_sequence_tokens}"
                )
            if token_count - layout.response_idx < 2:
                raise CounterfactualGenerationError(
                    "response has fewer than two tokens and no history mediator"
                )
            candidates = generate_equal_token_counterfactuals(
                layout,
                tokenizer,
                max_candidates=config.max_counterfactuals,
            )
            primary = candidates[0]
            pair = Gate1Pair(
                sample_id=example.response_id,
                source_id=example.source_id,
                task_type=example.task_type,
                generator_model=example.generator_model or "unknown",
                factual={
                    "input_ids": primary.factual_input_ids,
                    "response_idx": layout.response_idx,
                    "evidence_token_positions": layout.evidence_token_positions,
                },
                counterfactual={
                    "input_ids": primary.counterfactual_input_ids,
                    "response_idx": layout.response_idx,
                    "evidence_token_positions": layout.evidence_token_positions,
                },
                rescue_blocks=_history_blocks(
                    response_idx=layout.response_idx,
                    token_count=token_count,
                    size=config.history_block_size,
                    maximum=config.max_history_blocks,
                ),
            )
            pairs.append(pair)
            changed_positions_by_id[example.response_id] = tuple(
                int(value) for value in primary.changed_positions.tolist()
            )
            row["state"] = "counterfactual_available"
            row["counterfactual_candidates"] = len(candidates)
            row["pair_signature_sha256"] = _pair_signature(pair)
            counterfactual_rows.append(
                {
                    "response_id": example.response_id,
                    "source_id": example.source_id,
                    "protocol": "numeric_digit_surface_preserving_v1",
                    "response_idx": layout.response_idx,
                    "token_count": token_count,
                    "factual_input_ids_sha256": canonical_hash(
                        primary.factual_input_ids.tolist()
                    ),
                    "candidates": [
                        {
                            "changed_char_span": list(candidate.changed_char_span),
                            "changed_token_positions": candidate.changed_positions.tolist(),
                            "original_text": candidate.original_text,
                            "replacement_text": candidate.replacement_text,
                            "counterfactual_input_ids_sha256": canonical_hash(
                                candidate.counterfactual_input_ids.tolist()
                            ),
                            "used_for_mediation": index == 0,
                        }
                        for index, candidate in enumerate(candidates)
                    ],
                }
            )
        except CounterfactualGenerationError as error:
            row["state"] = "counterfactual_unavailable"
            row["reason"] = str(error)
        selection_rows.append(row)
        print(
            json.dumps(
                {
                    "event": "gate0",
                    "current": rank + 1,
                    "total": len(selected),
                    "state": row["state"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    atomic_jsonl(output / "selection.jsonl", selection_rows)
    atomic_jsonl(output / "counterfactual_audit.jsonl", counterfactual_rows)
    if not pairs:
        raise RuntimeError("Gate 0 produced no valid equal-token counterfactual pairs")

    del tokenizer
    shards = output / "gate1_shards"
    shards.mkdir(parents=True, exist_ok=True)
    shard_paths = [
        shards / f"{index:04d}-{canonical_hash(pair.sample_id)[:20]}.json"
        for index, pair in enumerate(pairs)
    ]
    raw_resume_runtime = initial_manifest.get("runtime")
    all_shards_available = config.resume and all(path.is_file() for path in shard_paths)
    backend: LlamaKVMediationBackend | None
    if all_shards_available and isinstance(raw_resume_runtime, Mapping):
        runtime = _runtime_from_mapping(raw_resume_runtime)
        observer_runtime, observer_runtime_signature = (
            parse_observer_runtime_identity(
                initial_manifest.get("observer_runtime"),
                initial_manifest.get("observer_runtime_signature"),
            )
        )
        gpu_report = {
            "event": "gpu_preflight",
            "skipped": True,
            "reason": "all Gate-1 shards are present and will be verified",
        }
        backend = None
        expected_forwards = 0
        print(json.dumps(gpu_report), flush=True)
    else:
        gpu_report = _cuda_preflight(config)
        print(json.dumps({"event": "observer_load_started"}), flush=True)
        tokenizer, model = _load_observer(config)
        _model_loaded_heartbeat(model, config)
        runtime = _runtime_identity(
            config=config,
            tokenizer=tokenizer,
            model=model,
            source_signature=model_source_signature,
        )
        resolved_dtype = (
            config.dtype
            if config.dtype != "auto"
            else str(next(model.parameters()).dtype)
        )
        observer_runtime, observer_runtime_signature = observer_runtime_identity(
            transformers_version=runtime.transformers_version,
            torch_version=runtime.torch_version,
            dtype=resolved_dtype,
            attn_implementation="eager",
        )
        backend = LlamaKVMediationBackend(model)
        expected_forwards = sum(
            8 + len(pair.rescue_blocks)
            for pair, path in zip(pairs, shard_paths, strict=True)
            if not path.is_file()
        )
        initial_manifest = {
            **initial_manifest,
            "runtime": asdict(runtime),
            "observer_runtime": observer_runtime,
            "observer_runtime_signature": observer_runtime_signature,
        }
        atomic_json(manifest_path, initial_manifest)
    print(
        json.dumps(
            {
                "event": "gate1_plan",
                "samples": len(pairs),
                "expected_forward_calls": expected_forwards,
            }
        ),
        flush=True,
    )

    records: list[Gate1Record] = []
    self_patch_values: list[float] = []
    for index, (pair, shard_path) in enumerate(zip(pairs, shard_paths, strict=True)):
        pair_hash = _pair_signature(pair)
        if shard_path.exists():
            if not config.resume:
                raise FileExistsError(f"unexpected existing Gate-1 shard: {shard_path}")
            record, self_patch = _load_verified_shard(
                shard_path, pair=pair, runtime=runtime
            )
            records.append(record)
            self_patch_values.append(self_patch)
            print(
                json.dumps(
                    {
                        "event": "gate1_resumed_shard",
                        "current": index + 1,
                        "total": len(pairs),
                        "response_id": pair.sample_id,
                    }
                ),
                flush=True,
            )
            continue

        if backend is None:
            raise RuntimeError("observer backend is absent for an incomplete shard set")

        started = time.monotonic()
        print(
            json.dumps(
                {
                    "event": "gate1_sample_started",
                    "current": index + 1,
                    "total": len(pairs),
                    "response_id": pair.sample_id,
                }
            ),
            flush=True,
        )

        def condition_progress(
            sample_id: str,
            condition: str,
            *,
            _index: int = index,
            _started: float = started,
        ) -> None:
            print(
                json.dumps(
                    {
                        "event": "gate1_condition_complete",
                        "current": _index + 1,
                        "total": len(pairs),
                        "response_id": sample_id,
                        "condition": condition,
                        "sample_elapsed_seconds": round(time.monotonic() - _started, 3),
                    }
                ),
                flush=True,
            )

        sample_result = run_gate1_pilot(
            [pair],
            backend=backend,
            runtime=runtime,
            audit_self_patch=True,
            condition_progress=condition_progress,
        )
        if len(sample_result.records) != 1:
            reasons = [asdict(item) for item in sample_result.manifest.rejections]
            raise RuntimeError(
                f"Gate-1 contract rejected a Gate-0 pair: {pair.sample_id}: {reasons}"
            )
        record = sample_result.records[0]
        _validate_record_against_pair(record, pair)
        self_patch = sample_result.manifest.self_patch_max_abs
        if self_patch is None:
            raise RuntimeError("Gate-1 self-patch audit did not produce a value")
        records.append(record)
        self_patch_values.append(self_patch)
        shard_payload = {
            "schema": "cept-gate1-sample-shard-v1",
            "sample_id": pair.sample_id,
            "pair_signature_sha256": pair_hash,
            "runtime": asdict(runtime),
            "self_patch_max_abs": self_patch,
            "record": asdict(record),
        }
        atomic_json(
            shard_path,
            {
                **shard_payload,
                "shard_content_sha256": canonical_hash(shard_payload),
            },
        )
        print(
            json.dumps(
                {
                    "event": "gate1_sample_complete",
                    "current": index + 1,
                    "total": len(pairs),
                    "response_id": pair.sample_id,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "shard": str(shard_path),
                }
            ),
            flush=True,
        )

    effects = _effect_rows(records)
    atomic_jsonl(output / "effects.jsonl", effects)
    pair_by_id = {pair.sample_id: pair for pair in pairs}
    transport_teachers = [
        make_transport_teacher_record(
            record,
            pair_by_id[record.sample_id],
            changed_evidence_positions=changed_positions_by_id[record.sample_id],
            official_split="train",
            model_signature=runtime.model_signature,
            model_source_signature=f"sha256:{model_source_signature}",
            tokenizer_signature=runtime.tokenizer_signature,
            observer_runtime=observer_runtime,
            observer_runtime_signature=observer_runtime_signature,
        )
        for record in records
    ]
    atomic_jsonl(output / "transport_teacher.jsonl", transport_teachers)
    residual_max = max(
        (abs(float(row["decomposition_residual"])) for row in effects), default=0.0
    )
    first_effects = [
        record.token_effects[0] for record in records if record.token_effects
    ]
    first_history_effect_max = max(
        (max(abs(item.mediated), abs(item.alternate)) for item in first_effects),
        default=0.0,
    )
    self_patch_max = max(self_patch_values, default=float("inf"))
    core_pass = (
        self_patch_max <= 1e-5
        and residual_max <= 1e-6
        and first_history_effect_max <= 1e-6
    )
    mediated_rows = [
        {
            "task_type": record.task_type,
            "generator_model": record.generator_model,
            "token_count": len(record.token_effects) + record.response_idx,
        }
        for record in records
    ]
    selected_rows = [dict(row) for row in selection_rows]
    available_rows = [
        {
            "task_type": example.task_type,
            "generator_model": example.generator_model or "unknown",
        }
        for example in examples
    ]
    counterfactual_available_rows = [
        row for row in selected_rows if row["state"] == "counterfactual_available"
    ]
    coverage = {
        "eligible_cache_inventory": _stratified_counts(available_rows),
        "selected": _stratified_counts(selected_rows),
        "counterfactual_available": _stratified_counts(counterfactual_available_rows),
        "mediation_complete": _stratified_counts(mediated_rows),
    }
    gate_report = {
        "schema": "cept-gate01-report-v2",
        "gate_status": "partial" if core_pass else "failed",
        "gate0": {
            "selected_samples": len(selected),
            "counterfactual_available": len(pairs),
            "coverage": len(pairs) / len(selected),
            "equal_token_contract": True,
            "annotations_absent": True,
            "counterfactual_protocol": "numeric_digit_surface_preserving_v1",
            "multiple_candidate_effect_stability_tested": False,
        },
        "gate1": {
            "completed_samples": len(records),
            "self_patch_max_abs": self_patch_max,
            "decomposition_residual_max_abs": residual_max,
            "first_token_history_effect_max_abs": first_history_effect_max,
            "self_patch_pass": self_patch_max <= 1e-5,
            "decomposition_pass": residual_max <= 1e-6,
            "first_token_causal_visibility_pass": first_history_effect_max <= 1e-6,
        },
        "coverage_by_stratum": coverage,
        "status_reason": (
            "Core intervention contracts passed, but multi-counterfactual effect "
            "stability remains untested."
            if core_pass
            else "At least one core intervention contract failed."
        ),
    }
    atomic_json(output / "gate_report.json", gate_report)

    teacher_manifest = {
        "schema": "cept-gate1-mediation-manifest-v2",
        "requested_samples": len(pairs),
        "completed_samples": len(records),
        "coverage": len(records) / len(pairs),
        "rejections": [],
        **asdict(runtime),
        "effect_definitions": {
            "total": "Y11 - Y00: operational total source intervention effect",
            "direct": "Y10 - Y00: operational non-history-K/V effect",
            "mediated": "Y11 - Y10: response-history K/V-mediated effect",
            "alternate": "Y01 - Y00: reverse-direction mediation check",
            "interaction": "mediated - alternate: direction sensitivity check",
            "direct_seed_rescue": (
                "Y(E0, patch changed evidence K/V from E1) - Y00: total recursive "
                "support initiated by the changed-evidence seed"
            ),
            "joint_seed_history_rescue": (
                "Y(E0, patch changed-evidence and response-history K/V from E1) - Y00"
            ),
        },
        "counterfactual_protocol": "numeric_digit_surface_preserving_v1",
        "observer_runtime": observer_runtime,
        "observer_runtime_signature": observer_runtime_signature,
        "self_patch_max_abs": self_patch_max,
    }
    artifact_paths = {
        "selection": output / "selection.jsonl",
        "counterfactual_audit": output / "counterfactual_audit.jsonl",
        "effects": output / "effects.jsonl",
        "transport_teacher": output / "transport_teacher.jsonl",
        "gate_report": output / "gate_report.json",
    }
    artifact_manifest = {
        name: {
            "path": str(path.relative_to(output)),
            "sha256": file_sha256(path),
        }
        for name, path in artifact_paths.items()
    }
    artifact_manifest["gate1_shards"] = {
        "path": "gate1_shards",
        "files": [
            {
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
            }
            for path in shard_paths
        ],
    }
    final_manifest: dict[str, Any] = {
        **initial_manifest,
        "run_state": "complete",
        "gate_status": gate_report["gate_status"],
        "active_attempt_token": None,
        "dataset": dataset_identity,
        "selection": {
            "policy": "task_generator_round_robin_source_unique_sha256_v1",
            "seed": config.seed,
            "selected_ids_sha256": canonical_hash(
                [example.response_id for example in selected]
            ),
            "selected_samples": len(selected),
            "coverage_by_stratum": coverage,
        },
        "counts": {
            "layout_selected": len(selected),
            "counterfactual_available": len(pairs),
            "mediation_complete": len(records),
            "effect_tokens": len(effects),
            "transport_teacher_samples": len(transport_teachers),
        },
        "runtime": asdict(runtime),
        "observer_runtime": observer_runtime,
        "observer_runtime_signature": observer_runtime_signature,
        "gpu_preflight": gpu_report,
        "teacher_manifest": teacher_manifest,
        "artifacts": artifact_manifest,
        "limitations": [
            "No hallucination annotations or AUROC are used in Gate 0/1.",
            "The current counterfactual adapter is limited to one-character numeric edits.",
            "Additional candidates are audited but only candidate 0 is mediated in this pilot.",
            "Multi-counterfactual effect stability has not yet been tested.",
            "The transport student is trained separately and never reads hallucination labels.",
            (
                "The estimand is restricted to the generator/task selectors and "
                "response IDs present in the verified train attention cache."
            ),
        ],
    }
    atomic_json(manifest_path, final_manifest)
    print(
        f"CEPT run complete: gate_status={gate_report['gate_status']} "
        f"selected={len(selected)} counterfactual={len(pairs)} "
        f"mediated={len(records)} output={output}",
        flush=True,
    )
    return final_manifest


def run_pilot(config: PilotConfig) -> dict[str, object]:
    """Run the recoverable 50-sample Gate 0/1 pilot."""

    config.validate()
    output = config.output_dir.expanduser().resolve()
    attempt_token = uuid.uuid4().hex
    with ExclusiveRunLock(output, resume=config.resume):
        try:
            return _run_pilot_locked(config, output, attempt_token)
        except BaseException as error:
            manifest_path = output / "manifest.json"
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if (
                        manifest.get("run_state") != "complete"
                        and manifest.get("active_attempt_token") == attempt_token
                    ):
                        manifest.update(
                            {
                                "run_state": "failed",
                                "gate_status": "not_evaluated",
                                "failure": {
                                    "type": type(error).__name__,
                                    "message": str(error),
                                },
                                "active_attempt_token": None,
                            }
                        )
                        atomic_json(manifest_path, manifest)
                except (OSError, ValueError, TypeError):
                    pass
            raise


__all__ = ["PilotConfig", "run_pilot"]
