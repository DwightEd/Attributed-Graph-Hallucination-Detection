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


def _estimate_attention_storage_bytes(
    num_layers: int,
    num_heads: int,
    token_count: int,
) -> int:
    """Estimate the float32 CPU attention tensor retained by this extractor."""

    return int(num_layers) * int(num_heads) * int(token_count) ** 2 * 4


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
    """Tokenize the final passage/question/answer text exactly once."""

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
    replay_input_ids = example.metadata.get("replay_input_ids")
    if replay_input_ids is not None and input_ids.tolist() != list(replay_input_ids):
        raise ValueError(
            f"BoolQ example {example.example_id!r} does not retokenize to the exact "
            "generation ids; refusing an inexact attention replay"
        )
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
                f"example {example.example_id!r} would retain approximately "
                f"{estimated / 1024**3:.2f} GiB of float32 attention; limit is "
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
    with torch.no_grad():
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
    attention = torch.stack(
        [value[0].detach().cpu().float() for value in outputs.attentions], dim=0
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
    hidden_states = torch.stack(
        [outputs.hidden_states[index][0].detach().cpu().float() for index in layer_ids],
        dim=0,
    )
    token_log_prob, entropy, valid = compute_teacher_forced_statistics(
        outputs.logits, model_inputs["input_ids"]
    )
    return {
        "schema_version": "token_trace_v1",
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
        "token_log_prob": token_log_prob.detach().cpu(),
        "next_token_entropy": entropy.detach().cpu(),
        "token_stat_valid": valid.detach().cpu(),
    }


def _token_spans(segment_ids: torch.Tensor) -> dict[str, tuple[int, int]]:
    spans = {}
    for name, identifier in (("passage", 1), ("question", 2), ("answer", 3)):
        indices = torch.nonzero(segment_ids == identifier, as_tuple=False).flatten()
        if not len(indices):
            raise ValueError(f"trace has no {name} tokens")
        spans[name] = (int(indices.min()), int(indices.max()) + 1)
    return spans


def _scalar(value) -> float:
    return float(np.asarray(value, dtype=float).mean())


def summarize_trace_record(trace: dict[str, object]) -> dict[str, object]:
    """Convert a raw trace to scalar observables before labels are available."""

    segment_ids = torch.as_tensor(trace["segment_ids"], dtype=torch.long)
    summary = summarize_attention_trace(
        trace["attention"],
        _token_spans(segment_ids),
        edge_threshold=float(trace.get("edge_threshold", 0.05)),
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
    for name, value in summary.items():
        record[name] = _scalar(value)
    answer_mask = segment_ids == 3
    if "token_log_prob" in trace:
        valid = torch.as_tensor(
            trace.get("token_stat_valid", torch.ones_like(answer_mask)),
            dtype=torch.bool,
        )
        selected = answer_mask & valid
        if bool(selected.any()):
            values = torch.as_tensor(trace["token_log_prob"], dtype=torch.float32)
            record["mean_answer_log_prob"] = float(values[selected].mean())
    if "next_token_entropy" in trace:
        valid = torch.as_tensor(
            trace.get("token_stat_valid", torch.ones_like(answer_mask)),
            dtype=torch.bool,
        )
        selected = answer_mask & valid
        if bool(selected.any()):
            values = torch.as_tensor(trace["next_token_entropy"], dtype=torch.float32)
            record["mean_answer_next_token_entropy"] = float(values[selected].mean())
    return record


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
    max_tokens: int | None = None,
    model_signature: str | None = None,
) -> str:
    payload = json.dumps(
        {
            "schema": "token_trace_graph_v2",
            "text": example.text,
            "model_id": model_id,
            "model_signature": model_signature or model_id,
            "layers": list(selected_layers),
            "tau": tau,
            "include_prefix_edges": include_prefix_edges,
            "include_hidden_nodes": include_hidden_nodes,
            "max_tokens": max_tokens,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_torch_save(value, path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary_path)
    temporary_path.replace(path)


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
    overwrite: bool = False,
    model_signature: str | None = None,
    max_attention_bytes: int | None = None,
) -> dict[str, object]:
    """Extract traces, token graphs, and scalar pattern features without labels."""

    output_directory = Path(output_dir)
    trace_directory = output_directory / "traces"
    graph_directory = output_directory / "graphs"
    trace_directory.mkdir(parents=True, exist_ok=True)
    graph_directory.mkdir(parents=True, exist_ok=True)
    feature_records = []
    for position, example in enumerate(examples, start=1):
        trace_path = _artifact_path(trace_directory, example.example_id)
        graph_path = _artifact_path(graph_directory, example.example_id)
        fingerprint = _fingerprint(
            example,
            model_id,
            selected_hidden_layers,
            tau,
            include_prefix_edges,
            include_hidden_nodes,
            max_tokens=max_tokens,
            model_signature=model_signature,
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
            attention_shape = torch.as_tensor(trace["attention"]).shape
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
        else:
            trace = extract_example_trace(
                model,
                tokenizer,
                example,
                selected_hidden_layers=selected_hidden_layers,
                max_tokens=max_tokens,
                max_attention_bytes=max_attention_bytes,
            )
            trace["extractor_model_id"] = model_id
            trace["extraction_fingerprint"] = fingerprint
            trace["edge_threshold"] = float(tau)
            graph = build_token_graph(
                trace["input_ids"],
                trace["attention"],
                trace["segment_ids"],
                hidden_states=(trace["hidden_states"] if include_hidden_nodes else None),
                token_log_probs=trace["token_log_prob"],
                next_token_entropy=trace["next_token_entropy"],
                token_stat_valid=trace["token_stat_valid"],
                tau=tau,
                include_prefix_edges=include_prefix_edges,
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
            stored_trace = dict(trace)
            stored_trace["attention"] = trace["attention"].half()
            stored_trace["hidden_states"] = trace["hidden_states"].half()
            _atomic_torch_save(stored_trace, trace_path)
            _atomic_torch_save(graph, graph_path)
        feature_records.append(summarize_trace_record(trace))
        if position == 1 or position % 25 == 0 or position == len(examples):
            print(f"Extracted {position}/{len(examples)} token graphs")

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
        "include_hidden_nodes": include_hidden_nodes,
        "max_attention_bytes": max_attention_bytes,
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
        help="Refuse a sample whose retained float32 attention exceeds this estimate",
    )
    parser.add_argument("--probe-layers", help="Comma-separated hidden-state indices")
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--drop-prefix-edges", action="store_true")
    parser.add_argument("--include-hidden-nodes", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda:0")
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
        overwrite=args.overwrite,
        model_signature=source_signature,
        max_attention_bytes=int(args.max_attention_gib * 1024**3),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
