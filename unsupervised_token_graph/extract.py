"""Extract label-free traces from one tokenization of the composed input."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .data import TokenGraphExample, read_prepared_examples
from .features import summarize_attention_trace
from .graph import build_token_graph
from .identity import model_source_signature
from .trace import assign_segment_ids


_POSTPROCESS_CHOICES = ("auto", "cpu", "model")
_POSTPROCESS_WORKSPACE_MULTIPLIER = 4
_POSTPROCESS_CUDA_RESERVE_BYTES = 2 * 1024**3


def _estimate_attention_storage_bytes(
    num_layers: int,
    num_heads: int,
    token_count: int,
) -> int:
    """Conservatively estimate a dense float32 attention representation."""

    return int(num_layers) * int(num_heads) * int(token_count) ** 2 * 4


def _attention_tensor_bytes(attentions: Sequence[torch.Tensor]) -> int:
    return sum(
        int(value[0].numel()) * int(value[0].element_size())
        for value in attentions
    )


def _stack_batch_zero_on_device(
    values: Sequence[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Stack batch-zero views without retaining a second CPU list copy."""

    if not values:
        raise ValueError("cannot stack an empty tensor sequence")
    first = values[0][0].detach()
    stacked = torch.empty(
        (len(values), *first.shape),
        dtype=first.dtype,
        device=device,
    )
    for index, value in enumerate(values):
        stacked[index].copy_(value[0].detach().to(device))
    return stacked


def _move_postprocess_tensors(
    trace: dict[str, object],
    device: torch.device | str,
) -> None:
    for name in (
        "attention",
        "hidden_states",
        "token_log_prob",
        "next_token_entropy",
        "token_stat_valid",
    ):
        trace[name] = torch.as_tensor(trace[name]).detach().to(device)
    trace["postprocess_device"] = str(torch.device(device))


def _resolve_postprocess_device(
    policy: str,
    model_device: torch.device | str,
    *,
    attention_bytes: int,
    cuda_free_bytes: int | None = None,
    reserve_bytes: int = _POSTPROCESS_CUDA_RESERVE_BYTES,
) -> torch.device:
    """Choose GPU postprocessing only when a conservative workspace fits."""

    if policy not in _POSTPROCESS_CHOICES:
        raise ValueError(
            f"postprocess_device must be one of {_POSTPROCESS_CHOICES}, got {policy!r}"
        )
    device = torch.device(model_device)
    if policy == "cpu" or device.type != "cuda":
        return torch.device("cpu")
    if policy == "model":
        return device
    if cuda_free_bytes is None:
        try:
            cuda_free_bytes, _ = torch.cuda.mem_get_info(device)
            # PyTorch can reuse free blocks held by its allocator even though
            # the CUDA driver does not report them as globally free.
            cuda_free_bytes += max(
                int(torch.cuda.memory_reserved(device))
                - int(torch.cuda.memory_allocated(device)),
                0,
            )
        except (AssertionError, RuntimeError, TypeError, ValueError):
            return torch.device("cpu")
    required = (
        int(attention_bytes) * _POSTPROCESS_WORKSPACE_MULTIPLIER
        + int(reserve_bytes)
    )
    return device if int(cuda_free_bytes) >= required else torch.device("cpu")


def _unbatch(value):
    if isinstance(value, torch.Tensor) and value.ndim > 1:
        return value[0]
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        return value[0]
    return value


def tokenize_example_once(
    tokenizer,
    example: TokenGraphExample,
    *,
    max_tokens: int | None = None,
) -> dict[str, torch.Tensor]:
    """Tokenize once, or replay the exact token sequence produced for BoolQ."""

    replay_input_ids = example.metadata.get("replay_input_ids")
    if replay_input_ids is not None:
        required = (
            "replay_attention_mask",
            "replay_offset_mapping",
            "replay_special_tokens_mask",
            "replay_segment_ids",
        )
        missing = [name for name in required if name not in example.metadata]
        if missing:
            raise ValueError(
                f"BoolQ example {example.example_id!r} has incomplete exact replay "
                f"metadata: {missing}"
            )
        input_ids = torch.as_tensor(replay_input_ids, dtype=torch.long)
        attention_mask = torch.as_tensor(
            example.metadata["replay_attention_mask"], dtype=torch.long
        )
        offsets = torch.as_tensor(
            example.metadata["replay_offset_mapping"], dtype=torch.long
        )
        special = torch.as_tensor(
            example.metadata["replay_special_tokens_mask"], dtype=torch.bool
        )
        segment_ids = torch.as_tensor(
            example.metadata["replay_segment_ids"], dtype=torch.long
        )
        token_count = len(input_ids)
        if any(
            len(value) != token_count
            for value in (attention_mask, offsets, special, segment_ids)
        ):
            raise ValueError(
                f"BoolQ exact replay arrays differ in length for {example.example_id!r}"
            )
        if offsets.ndim != 2 or offsets.shape[1] != 2:
            raise ValueError(
                f"BoolQ exact replay offsets are malformed for {example.example_id!r}"
            )
        if max_tokens is not None and token_count > max_tokens:
            raise ValueError(
                f"example {example.example_id!r} has {token_count} tokens and "
                f"exceeds max_tokens={max_tokens}; full-context extraction refuses truncation"
            )
        answer_mask = segment_ids == 3
        if not bool(answer_mask.any()):
            raise ValueError(f"answer segment for {example.example_id!r} has no tokens")
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "offset_mapping": offsets,
            "special_tokens_mask": special,
            "segment_ids": segment_ids,
            "answer_mask": answer_mask,
        }

    encoded = tokenizer(
        example.text,
        add_special_tokens=True,
        return_attention_mask=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        return_tensors="pt",
        truncation=False,
    )
    input_ids = torch.as_tensor(_unbatch(encoded["input_ids"]), dtype=torch.long)
    if max_tokens is not None and len(input_ids) > max_tokens:
        raise ValueError(
            f"example {example.example_id!r} has {len(input_ids)} tokens and "
            f"exceeds max_tokens={max_tokens}; full-context extraction refuses truncation"
        )
    offsets = torch.as_tensor(_unbatch(encoded["offset_mapping"]), dtype=torch.long)
    special = torch.as_tensor(
        _unbatch(encoded["special_tokens_mask"]), dtype=torch.bool
    )
    segment_ids = torch.tensor(
        assign_segment_ids(offsets.tolist(), example.segment_char_spans, special.tolist()),
        dtype=torch.long,
    )
    answer_mask = segment_ids == 3
    if not bool(answer_mask.any()):
        raise ValueError(f"answer segment for {example.example_id!r} has no tokens")
    return {
        "input_ids": input_ids,
        "attention_mask": torch.as_tensor(
            _unbatch(encoded["attention_mask"]), dtype=torch.long
        ),
        "offset_mapping": offsets,
        "special_tokens_mask": special,
        "segment_ids": segment_ids,
        "answer_mask": answer_mask,
    }


def compute_teacher_forced_statistics(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align each next-token log probability and entropy to its target token."""

    values = logits[0] if logits.ndim == 3 else logits
    tokens = input_ids[0] if input_ids.ndim == 2 else input_ids
    if values.ndim != 2 or len(values) != len(tokens):
        raise ValueError("logits and input_ids must share their token dimension")
    token_log_prob = values.new_zeros(len(tokens), dtype=torch.float32)
    entropy = values.new_zeros(len(tokens), dtype=torch.float32)
    valid = torch.zeros(len(tokens), dtype=torch.bool, device=values.device)
    if len(tokens) < 2:
        return token_log_prob, entropy, valid
    distribution = torch.log_softmax(values[:-1].float(), dim=-1)
    token_log_prob[1:] = distribution.gather(1, tokens[1:].unsqueeze(1)).squeeze(1)
    entropy[1:] = -(distribution.exp() * distribution).sum(dim=-1)
    valid[1:] = True
    return token_log_prob, entropy, valid


def extract_example_trace(
    model,
    tokenizer,
    example: TokenGraphExample,
    *,
    selected_hidden_layers: Sequence[int],
    max_tokens: int | None = None,
    max_attention_bytes: int | None = None,
    postprocess_device: str = "auto",
) -> dict[str, object]:
    """Run one teacher-forced forward pass and return a label-free trace."""

    encoded = tokenize_example_once(tokenizer, example, max_tokens=max_tokens)
    config = getattr(model, "config", None)
    if max_attention_bytes is not None and config is not None:
        estimated = _estimate_attention_storage_bytes(
            int(config.num_hidden_layers),
            int(config.num_attention_heads),
            len(encoded["input_ids"]),
        )
        if estimated > max_attention_bytes:
            raise MemoryError(
                f"example {example.example_id!r} has a conservative dense-attention "
                f"estimate of {estimated / 1024**3:.2f} GiB; limit is "
                f"{max_attention_bytes / 1024**3:.2f} GiB"
            )
    try:
        device = next(model.parameters()).device
    except (StopIteration, AttributeError):
        device = torch.device("cpu")
    model_inputs = {
        "input_ids": encoded["input_ids"].unsqueeze(0).to(device),
        "attention_mask": encoded["attention_mask"].unsqueeze(0).to(device),
    }
    with torch.inference_mode():
        outputs = model(
            **model_inputs,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
    if not outputs.attentions or any(value is None for value in outputs.attentions):
        raise RuntimeError(
            "The model did not return attention tensors; load it with eager attention"
        )
    if not selected_hidden_layers:
        raise ValueError("at least one hidden-state layer must be selected")
    hidden_count = len(outputs.hidden_states)
    layer_ids = [index if index >= 0 else hidden_count + index for index in selected_hidden_layers]
    if any(index < 0 or index >= hidden_count for index in layer_ids):
        raise ValueError(
            f"selected hidden layers {tuple(selected_hidden_layers)} exceed "
            f"the available range for {hidden_count} hidden-state tensors"
        )
    with torch.inference_mode():
        token_log_prob, entropy, valid = compute_teacher_forced_statistics(
            outputs.logits, model_inputs["input_ids"]
        )
        attention_layers = tuple(outputs.attentions)
        attention_bytes = _attention_tensor_bytes(attention_layers)
        processing_device = _resolve_postprocess_device(
            postprocess_device,
            device,
            attention_bytes=attention_bytes,
        )
        fallback_reason = None
        try:
            attention = _stack_batch_zero_on_device(
                attention_layers, processing_device
            )
            hidden_states = _stack_batch_zero_on_device(
                tuple(outputs.hidden_states[index] for index in layer_ids),
                processing_device,
            )
        except torch.OutOfMemoryError:
            if postprocess_device != "auto" or processing_device.type != "cuda":
                raise
            if "attention" in locals():
                del attention
            if "hidden_states" in locals():
                del hidden_states
            torch.cuda.empty_cache()
            processing_device = torch.device("cpu")
            fallback_reason = "cuda_oom_during_stack"
            attention = _stack_batch_zero_on_device(
                attention_layers, processing_device
            )
            hidden_states = _stack_batch_zero_on_device(
                tuple(outputs.hidden_states[index] for index in layer_ids),
                processing_device,
            )
        token_log_prob = token_log_prob.detach().to(processing_device)
        entropy = entropy.detach().to(processing_device)
        valid = valid.detach().to(processing_device)
    del outputs, attention_layers
    trace = {
        "schema_version": "token_trace_v2",
        "example_id": example.example_id,
        "pair_id": example.pair_id,
        "dataset": example.dataset,
        "text_sha256": hashlib.sha256(example.text.encode("utf-8")).hexdigest(),
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"].bool(),
        "offset_mapping": encoded["offset_mapping"],
        "special_tokens_mask": encoded["special_tokens_mask"],
        "segment_ids": encoded["segment_ids"],
        "answer_mask": encoded["answer_mask"],
        "attention": attention,
        "selected_hidden_layers": torch.tensor(layer_ids, dtype=torch.long),
        "hidden_states": hidden_states,
        "token_log_prob": token_log_prob,
        "next_token_entropy": entropy,
        "token_stat_valid": valid,
        "postprocess_device": str(processing_device),
        "dense_attention_bytes": attention_bytes,
    }
    if fallback_reason is not None:
        trace["postprocess_fallback_reason"] = fallback_reason
    return trace


def _token_spans(segment_ids: torch.Tensor) -> dict[str, tuple[int, int]]:
    spans = {}
    for name, identifier in (("passage", 1), ("question", 2), ("answer", 3)):
        indices = torch.nonzero(segment_ids == identifier, as_tuple=False).flatten()
        if not len(indices):
            raise ValueError(f"trace has no {name} tokens")
        spans[name] = (int(indices.min()), int(indices.max()) + 1)
    return spans


def _scalar(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().to(torch.float32).mean().item())
    return float(np.asarray(value, dtype=float).mean())


def summarize_trace_record(
    trace: dict[str, object],
    *,
    edge_presence=None,
) -> dict[str, object]:
    """Convert a raw trace to scalar observables before labels are available."""

    segment_ids = torch.as_tensor(trace["segment_ids"], dtype=torch.long)
    summary = summarize_attention_trace(
        trace["attention"],
        _token_spans(segment_ids),
        edge_threshold=float(trace.get("edge_threshold", 0.05)),
        edge_presence=edge_presence,
    )
    record: dict[str, object] = {
        "example_id": str(trace["example_id"]),
        "pair_id": str(trace.get("pair_id", trace["example_id"])),
        "dataset": str(trace.get("dataset", "unknown")),
        "token_count": int(len(segment_ids)),
        "passage_token_count": int((segment_ids == 1).sum()),
        "question_token_count": int((segment_ids == 2).sum()),
        "answer_token_count": int((segment_ids == 3).sum()),
    }
    tensor_names = []
    tensor_scalars = []
    for name, value in summary.items():
        if isinstance(value, torch.Tensor):
            tensor_names.append(name)
            tensor_scalars.append(value.detach().to(torch.float32).mean())
        else:
            record[name] = _scalar(value)
    if tensor_scalars:
        # One device synchronization for all attention features, rather than
        # one .item() call per feature.
        scalar_values = torch.stack(tensor_scalars).cpu().tolist()
        record.update(zip(tensor_names, (float(value) for value in scalar_values)))
    answer_mask = segment_ids == 3
    if "token_log_prob" in trace:
        values = torch.as_tensor(trace["token_log_prob"]).detach().to(torch.float32)
        valid = torch.as_tensor(
            trace.get("token_stat_valid", torch.ones_like(answer_mask)),
            dtype=torch.bool,
            device=values.device,
        ).detach()
        selected = answer_mask.to(values.device) & valid
        if bool(selected.any()):
            record["mean_answer_log_prob"] = float(values[selected].mean().item())
    if "next_token_entropy" in trace:
        values = torch.as_tensor(trace["next_token_entropy"]).detach().to(
            torch.float32
        )
        valid = torch.as_tensor(
            trace.get("token_stat_valid", torch.ones_like(answer_mask)),
            dtype=torch.bool,
            device=values.device,
        ).detach()
        selected = answer_mask.to(values.device) & valid
        if bool(selected.any()):
            record["mean_answer_next_token_entropy"] = float(
                values[selected].mean().item()
            )
    return record


def _build_graph_and_feature_record(
    trace: dict[str, object],
    *,
    tau: float,
    include_prefix_edges: bool,
    include_hidden_nodes: bool,
    include_logit_node_features: bool,
    pure_attention: bool = False,
) -> tuple[dict[str, object], dict[str, object]]:
    """Share the expensive all-layer max and release summary temporaries first."""

    with torch.inference_mode():
        edge_presence = (
            torch.as_tensor(trace["attention"])
            .amax(dim=(0, 1))
            .to(torch.float32)
            > float(tau)
        )
        # Finish scalar reductions before allocating edge_attr, whose dense
        # worst case can be several GiB.
        feature_record = summarize_trace_record(
            trace, edge_presence=edge_presence
        )
        graph = build_token_graph(
            trace["input_ids"],
            trace["attention"],
            trace["segment_ids"],
            hidden_states=(trace["hidden_states"] if include_hidden_nodes else None),
            token_log_probs=trace["token_log_prob"],
            next_token_entropy=trace["next_token_entropy"],
            token_stat_valid=trace["token_stat_valid"],
            edge_presence=edge_presence,
            tau=tau,
            include_prefix_edges=include_prefix_edges,
            include_logit_node_features=include_logit_node_features,
            pure_attention=pure_attention,
        )
    return graph, feature_record


def resolve_probe_layers(num_hidden_layers: int, num_probes: int = 6) -> tuple[int, ...]:
    """Choose hidden-state outputs from 25% depth through the final layer."""

    if num_hidden_layers < 1 or num_probes < 1:
        raise ValueError("layer and probe counts must be positive")
    first = max(1, round(num_hidden_layers * 0.25))
    values = np.linspace(first, num_hidden_layers, num_probes)
    return tuple(sorted({int(round(value)) for value in values}))


def _fingerprint(
    example: TokenGraphExample,
    model_id: str,
    selected_layers: Sequence[int],
    tau: float,
    include_prefix_edges: bool,
    include_hidden_nodes: bool,
    *,
    include_logit_node_features: bool = True,
    pure_attention: bool = False,
    max_tokens: int | None = None,
    model_signature: str | None = None,
    extraction_dtype: str = "unknown",
    postprocess_device: str = "auto",
    retain_dense_attention: bool = True,
    trace_storage_dtype: str = "float16",
) -> str:
    effective_hidden_nodes = bool(include_hidden_nodes and not pure_attention)
    effective_logit_node_features = bool(
        include_logit_node_features and not pure_attention
    )
    fingerprint_payload = {
        "schema": "token_trace_graph_v5",
        "text": example.text,
        "model_id": model_id,
        "model_signature": model_signature or model_id,
        "layers": list(selected_layers),
        "tau": tau,
        "include_prefix_edges": include_prefix_edges,
        "include_hidden_nodes": effective_hidden_nodes,
        "include_logit_node_features": effective_logit_node_features,
        "max_tokens": max_tokens,
        "extraction_dtype": extraction_dtype,
        "postprocess_device": postprocess_device,
        "retain_dense_attention": retain_dense_attention,
        "trace_storage_dtype": trace_storage_dtype,
        "exact_replay": {
            name: example.metadata.get(name)
            for name in (
                "replay_input_ids",
                "replay_attention_mask",
                "replay_offset_mapping",
                "replay_special_tokens_mask",
                "replay_segment_ids",
            )
            if name in example.metadata
        },
    }
    # Preserve byte-for-byte compatibility with legacy non-pure caches while
    # giving strict pure-attention artifacts a distinct identity.
    if pure_attention:
        fingerprint_payload["pure_attention"] = True
    payload = json.dumps(
        fingerprint_payload,
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_torch_save(value, path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary_path)
    temporary_path.replace(path)


def _to_cpu_tree(value):
    """Detach every tensor leaf so cached artifacts are device-portable."""

    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu")
    if isinstance(value, dict):
        return {key: _to_cpu_tree(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_to_cpu_tree(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu_tree(child) for child in value)
    return value


def _stored_trace(
    trace: dict[str, object],
    *,
    retain_dense_attention: bool,
) -> dict[str, object]:
    attention = torch.as_tensor(trace["attention"]).detach()
    stored = {
        key: value for key, value in trace.items() if key != "attention"
    }
    stored["attention_shape"] = list(attention.shape)
    stored["hidden_states"] = torch.as_tensor(trace["hidden_states"]).detach().to(
        device="cpu", dtype=torch.float16
    )
    if retain_dense_attention:
        stored["attention"] = attention.to(device="cpu", dtype=torch.float16)
        stored["attention_storage"] = "float16_cpu"
    else:
        stored["attention_storage"] = "discarded_after_postprocessing"
    return _to_cpu_tree(stored)


def _artifact_path(directory: Path, example_id: str) -> Path:
    """Map an external example id to a contained, filesystem-safe path."""

    digest = hashlib.sha256(str(example_id).encode("utf-8")).hexdigest()[:32]
    return directory / f"{digest}.pt"


def extract_prepared_dataset(
    model,
    tokenizer,
    examples: Sequence[TokenGraphExample],
    *,
    output_dir: str | Path,
    model_id: str,
    selected_hidden_layers: Sequence[int],
    max_tokens: int | None,
    tau: float,
    include_prefix_edges: bool,
    include_hidden_nodes: bool,
    include_logit_node_features: bool = True,
    pure_attention: bool = False,
    overwrite: bool = False,
    model_signature: str | None = None,
    max_attention_bytes: int | None = None,
    extraction_dtype: str = "unknown",
    postprocess_device: str = "auto",
    retain_dense_attention: bool = True,
) -> dict[str, object]:
    """Extract traces, token graphs, and scalar pattern features without labels."""

    output_directory = Path(output_dir)
    trace_directory = output_directory / "traces"
    graph_directory = output_directory / "graphs"
    trace_directory.mkdir(parents=True, exist_ok=True)
    graph_directory.mkdir(parents=True, exist_ok=True)
    feature_records = []
    postprocess_device_counts: dict[str, int] = {}
    postprocess_fallback_counts: dict[str, int] = {}
    extracted_examples = 0
    reused_examples = 0
    gpu_peak_allocated_bytes_max = 0
    for position, example in enumerate(examples, start=1):
        stored_trace = None
        trace_path = _artifact_path(trace_directory, example.example_id)
        graph_path = _artifact_path(graph_directory, example.example_id)
        fingerprint = _fingerprint(
            example,
            model_id,
            selected_hidden_layers,
            tau,
            include_prefix_edges,
            include_hidden_nodes,
            include_logit_node_features=include_logit_node_features,
            pure_attention=pure_attention,
            max_tokens=max_tokens,
            model_signature=model_signature,
            extraction_dtype=extraction_dtype,
            postprocess_device=postprocess_device,
            retain_dense_attention=retain_dense_attention,
        )
        if trace_path.exists() and graph_path.exists() and not overwrite:
            trace = torch.load(trace_path, map_location="cpu", weights_only=True)
            if trace.get("extraction_fingerprint") != fingerprint:
                raise RuntimeError(
                    f"Stale extraction cache for {example.example_id}; use --overwrite"
                )
            if max_tokens is not None and len(trace["input_ids"]) > max_tokens:
                raise RuntimeError(
                    f"Cached example {example.example_id!r} exceeds max_tokens="
                    f"{max_tokens}; use --overwrite only after changing the input"
                )
            attention_shape = trace.get("attention_shape")
            if attention_shape is None and "attention" in trace:
                attention_shape = list(torch.as_tensor(trace["attention"]).shape)
            if attention_shape is None or len(attention_shape) != 4:
                raise RuntimeError(
                    f"Cached example {example.example_id!r} has no valid attention shape"
                )
            estimated = _estimate_attention_storage_bytes(
                attention_shape[0], attention_shape[1], attention_shape[-1]
            )
            if max_attention_bytes is not None and estimated > max_attention_bytes:
                raise RuntimeError(
                    f"Cached example {example.example_id!r} exceeds the configured "
                    "attention storage limit"
                )
            graph = torch.load(graph_path, map_location="cpu", weights_only=True)
            if (
                graph.get("extraction_fingerprint") != fingerprint
                or str(graph.get("example_id")) != example.example_id
            ):
                raise RuntimeError(
                    f"Stale or mismatched graph cache for {example.example_id}; "
                    "use --overwrite"
                )
            if str(trace.get("example_id")) != example.example_id:
                raise RuntimeError(
                    f"Stale or mismatched trace cache for {example.example_id}; "
                    "use --overwrite"
                )
            if not isinstance(trace.get("feature_record"), dict):
                raise RuntimeError(
                    f"Cached example {example.example_id!r} has no feature record; "
                    "use --overwrite"
                )
            feature_record = trace["feature_record"]
            reused_examples += 1
        else:
            try:
                model_device = next(model.parameters()).device
            except (StopIteration, AttributeError):
                model_device = torch.device("cpu")
            if model_device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(model_device)
            trace = extract_example_trace(
                model,
                tokenizer,
                example,
                selected_hidden_layers=selected_hidden_layers,
                max_tokens=max_tokens,
                max_attention_bytes=max_attention_bytes,
                postprocess_device=postprocess_device,
            )
            trace["extractor_model_id"] = model_id
            trace["extraction_fingerprint"] = fingerprint
            trace["extraction_dtype"] = extraction_dtype
            trace["edge_threshold"] = float(tau)
            trace["postprocess_device_policy"] = postprocess_device
            trace["retain_dense_attention"] = bool(retain_dense_attention)
            try:
                graph, feature_record = _build_graph_and_feature_record(
                    trace,
                    tau=tau,
                    include_prefix_edges=include_prefix_edges,
                    include_hidden_nodes=include_hidden_nodes,
                    include_logit_node_features=include_logit_node_features,
                    pure_attention=pure_attention,
                )
            except torch.OutOfMemoryError:
                attention_device = torch.as_tensor(trace["attention"]).device
                if postprocess_device != "auto" or attention_device.type != "cuda":
                    raise
                torch.cuda.empty_cache()
                _move_postprocess_tensors(trace, "cpu")
                torch.cuda.empty_cache()
                trace["postprocess_fallback_reason"] = (
                    "cuda_oom_during_graph_or_features"
                )
                graph, feature_record = _build_graph_and_feature_record(
                    trace,
                    tau=tau,
                    include_prefix_edges=include_prefix_edges,
                    include_hidden_nodes=include_hidden_nodes,
                    include_logit_node_features=include_logit_node_features,
                    pure_attention=pure_attention,
                )
            trace["feature_record"] = feature_record
            if model_device.type == "cuda":
                trace["gpu_peak_allocated_bytes"] = int(
                    torch.cuda.max_memory_allocated(model_device)
                )
            graph.update(
                {
                    "example_id": example.example_id,
                    "pair_id": example.pair_id,
                    "dataset": example.dataset,
                    "extractor_model_id": model_id,
                    "extraction_fingerprint": fingerprint,
                }
            )
            stored_trace = _stored_trace(
                trace,
                retain_dense_attention=retain_dense_attention,
            )
            graph = _to_cpu_tree(graph)
            _atomic_torch_save(stored_trace, trace_path)
            _atomic_torch_save(graph, graph_path)
            extracted_examples += 1
        feature_records.append(feature_record)
        resolved_device = str(trace.get("postprocess_device", "unknown"))
        postprocess_device_counts[resolved_device] = (
            postprocess_device_counts.get(resolved_device, 0) + 1
        )
        fallback_reason = trace.get("postprocess_fallback_reason")
        if fallback_reason:
            reason = str(fallback_reason)
            postprocess_fallback_counts[reason] = (
                postprocess_fallback_counts.get(reason, 0) + 1
            )
        gpu_peak_allocated_bytes_max = max(
            gpu_peak_allocated_bytes_max,
            int(trace.get("gpu_peak_allocated_bytes", 0)),
        )
        if position == 1 or position % 25 == 0 or position == len(examples):
            peak_suffix = (
                f"; max_cuda_allocated_gib="
                f"{gpu_peak_allocated_bytes_max / 1024**3:.2f}"
                if gpu_peak_allocated_bytes_max
                else ""
            )
            print(
                f"Extracted {position}/{len(examples)} token graphs; "
                f"postprocess={resolved_device}{peak_suffix}"
            )
        # Release a live GPU trace before the next model forward. Assignment on
        # the next iteration would otherwise keep it alive while evaluating the
        # right-hand side of extract_example_trace().
        del trace, graph, stored_trace

    feature_path = output_directory / "features.jsonl"
    feature_path.write_text(
        "\n".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True)
            for record in feature_records
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "state": "complete",
        "examples": len(examples),
        "model_id": model_id,
        "model_signature": model_signature or model_id,
        "selected_hidden_layers": list(selected_hidden_layers),
        "max_tokens": max_tokens,
        "tau": tau,
        "include_prefix_edges": include_prefix_edges,
        "include_hidden_nodes": bool(include_hidden_nodes and not pure_attention),
        "include_logit_node_features": bool(
            include_logit_node_features and not pure_attention
        ),
        "pure_attention": bool(pure_attention),
        "max_attention_bytes": max_attention_bytes,
        "extraction_dtype": extraction_dtype,
        "postprocess_device_policy": postprocess_device,
        "postprocess_device_counts": postprocess_device_counts,
        "postprocess_fallback_counts": postprocess_fallback_counts,
        "retain_dense_attention": retain_dense_attention,
        "trace_storage_dtype": "float16",
        "postprocessing_schema": "token_trace_graph_v5",
        "feature_reduction_dtype": "float32",
        "extracted_examples": extracted_examples,
        "reused_examples": reused_examples,
        "gpu_peak_allocated_bytes_max": gpu_peak_allocated_bytes_max,
        "trace_dir": str(trace_directory),
        "graph_dir": str(graph_directory),
        "graph_files": [
            _artifact_path(graph_directory, example.example_id).name
            for example in examples
        ],
        "example_ids": [example.example_id for example in examples],
        "features": str(feature_path),
    }
    (output_directory / "extraction_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract label-free token graphs.")
    parser.add_argument("--examples", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument(
        "--max-attention-gib",
        type=float,
        default=12.0,
        help="Refuse a sample whose conservative dense-attention estimate exceeds this",
    )
    parser.add_argument("--probe-layers", help="Comma-separated hidden-state indices")
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--drop-prefix-edges", action="store_true")
    parser.add_argument("--include-hidden-nodes", action="store_true")
    parser.add_argument(
        "--exclude-logit-node-features",
        action="store_true",
        help=(
            "Exclude teacher-forced token log-probability, entropy, and their "
            "validity bits from Graph-MAE node features"
        ),
    )
    parser.add_argument(
        "--pure-attention",
        action="store_true",
        help=(
            "Use only per-layer/per-head attention diagonals as node features "
            "and attention values as edge features; segment IDs remain metadata "
            "for response masking but are not passed to the graph network"
        ),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--postprocess-device",
        choices=_POSTPROCESS_CHOICES,
        default="auto",
        help=(
            "Where graph/features run: auto uses the model GPU only with "
            "conservative free-memory headroom"
        ),
    )
    parser.add_argument(
        "--discard-dense-attention",
        action="store_true",
        help=(
            "Keep graph, hidden states, and scalar features but omit the raw dense "
            "attention tensor from each trace"
        ),
    )
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="float16"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(args.device)
    model.eval()
    source_signature = model_source_signature(args.model, model=model, tokenizer=tokenizer)
    examples = read_prepared_examples(args.examples)
    if args.limit is not None:
        examples = examples[: args.limit]
    if args.probe_layers:
        probe_layers = tuple(int(value) for value in args.probe_layers.split(","))
    else:
        probe_layers = resolve_probe_layers(int(model.config.num_hidden_layers))
    manifest = extract_prepared_dataset(
        model,
        tokenizer,
        examples,
        output_dir=args.output_dir,
        model_id=args.model,
        selected_hidden_layers=probe_layers,
        max_tokens=args.max_tokens,
        tau=args.tau,
        include_prefix_edges=not args.drop_prefix_edges,
        include_hidden_nodes=args.include_hidden_nodes,
        include_logit_node_features=not args.exclude_logit_node_features,
        pure_attention=args.pure_attention,
        overwrite=args.overwrite,
        model_signature=source_signature,
        max_attention_bytes=int(args.max_attention_gib * 1024**3),
        extraction_dtype=args.dtype,
        postprocess_device=args.postprocess_device,
        retain_dense_attention=not args.discard_dense_attention,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
